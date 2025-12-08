from flask import Flask, Response
import cv2
from ultralytics import YOLO
import time

import digitalio
import board
from PIL import Image, ImageDraw
from adafruit_rgb_display import gc9a01a

app = Flask(__name__)

# -----------------------------
# YOLO 설정
# -----------------------------
MODEL_PATH = "/home/mj/yolov8n.pt"
model = YOLO(MODEL_PATH)

CAM_INDEX = 0          # USB 카메라 번호
YOLO_INTERVAL = 4      # 4프레임마다 한 번만 추론

# -----------------------------
# GC9A01 눈 디스플레이 설정
# -----------------------------
print("=== GC9A01 눈 + YOLO 시선 추적 모드 ===")

BAUDRATE = 24_000_000
spi = board.SPI()       # SCLK=GPIO11, MOSI=GPIO10

dc_pin = digitalio.DigitalInOut(board.D25)   # GPIO25, 핀 22
rst_pin = digitalio.DigitalInOut(board.D27)  # GPIO27, 핀 13

# 왼쪽 눈 (CE0)
cs_left = digitalio.DigitalInOut(board.CE0)  # GPIO8, 핀 24
disp_left = gc9a01a.GC9A01A(
    spi,
    cs=cs_left,
    dc=dc_pin,
    rst=rst_pin,
    baudrate=BAUDRATE,
    width=240,
    height=240,
    rotation=0,
)

# 오른쪽 눈 (CE1)
cs_right = digitalio.DigitalInOut(board.CE1)  # GPIO7, 핀 26
disp_right = gc9a01a.GC9A01A(
    spi,
    cs=cs_right,
    dc=dc_pin,
    rst=rst_pin,
    baudrate=BAUDRATE,
    width=240,
    height=240,
    rotation=0,
)

width = disp_left.width
height = disp_left.height
print(f"[INFO] 디스플레이 크기: {width}x{height}")

# -----------------------------
# 눈 그리기 함수
#   offset_x, offset_y: -1.0 ~ 1.0 범위로 시선 방향
# -----------------------------
def create_eye_image(offset_x: float, offset_y: float) -> Image.Image:
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    # 배경
    draw.rectangle((0, 0, width, height), fill=(0, 0, 0))

    cx, cy = width // 2, height // 2
    r_eye = min(width, height) // 2 - 4

    # 흰 눈
    draw.ellipse(
        (cx - r_eye, cy - r_eye, cx + r_eye, cy + r_eye),
        fill=(255, 255, 255),
    )

    # 동공 반지름
    r_pupil = int(r_eye / 1.5)

    # 동공 이동 가능한 최대 거리
    max_offset = r_eye - r_pupil - 6
    ox = int(max(-1.0, min(1.0, offset_x)) * max_offset)
    oy = int(max(-1.0, min(1.0, offset_y)) * max_offset)

    px = cx + ox
    py = cy + oy

    # 동공
    draw.ellipse(
        (px - r_pupil, py - r_pupil, px + r_pupil, py + r_pupil),
        fill=(0, 0, 0),
    )

    # 하이라이트는 동공 좌상단 쪽에 고정으로 찍어줌
    r_high = r_pupil // 2
    hx = px - r_pupil + 5
    hy = py - r_pupil + 5
    draw.ellipse(
        (hx, hy, hx + 2 * r_high, hy + 2 * r_high),
        fill=(200, 200, 255),
    )

    return img


# 처음에는 정면 바라보는 중립 눈
neutral_eye = create_eye_image(0.0, 0.0)
disp_left.image(neutral_eye)
disp_right.image(neutral_eye)

print("[OK] 두 눈 중립 상태 초기화 완료")

# 시선 방향 상태값 (부드럽게 따라가도록 EMA 사용)
eye_dir_x = 0.0
eye_dir_y = 0.0
target_dir_x = 0.0
target_dir_y = 0.0


def generate():
    global eye_dir_x, eye_dir_y, target_dir_x, target_dir_y

    cap = cv2.VideoCapture(CAM_INDEX)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    print("YOLOv8 + 눈 디스플레이 시작")

    frame_count = 0
    last_result = None

    # YOLO 입력 크기
    YOLO_W, YOLO_H = 320, 320

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # -------- YOLO 추론 (간헐적으로 실행) --------
        if frame_count % YOLO_INTERVAL == 0:
            small = cv2.resize(frame, (YOLO_W, YOLO_H))

            results = model(small, stream=True)

            main_box_center = None

            for r in results:
                last_result = r

                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    # 가장 큰 박스를 골라서 중심 좌표 계산
                    xyxy = boxes.xyxy.cpu().numpy()
                    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                    idx = areas.argmax()
                    x1, y1, x2, y2 = xyxy[idx]
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    main_box_center = (cx, cy)

            if main_box_center is not None:
                cx, cy = main_box_center
                # 중심 기준 -1.0 ~ 1.0 범위로 정규화
                dx = (cx - YOLO_W / 2.0) / (YOLO_W / 2.0)
                dy = (cy - YOLO_H / 2.0) / (YOLO_H / 2.0)

                # 위쪽을 쳐다볼 때 동공이 위로 가도록 dy 부호 유지
                target_dir_x = dx
                target_dir_y = dy
            else:
                # 객체 없으면 서서히 중앙으로 복귀
                target_dir_x = 0.0
                target_dir_y = 0.0

        # -------- 시선 방향 부드럽게 보간 --------
        alpha = 0.2  # 0에 가까울수록 더 천천히 따라감
        eye_dir_x = (1 - alpha) * eye_dir_x + alpha * target_dir_x
        eye_dir_y = (1 - alpha) * eye_dir_y + alpha * target_dir_y

        # 눈 이미지 생성 후 디스플레이에 출력
        eye_img_left = create_eye_image(eye_dir_x, eye_dir_y)
        eye_img_right = create_eye_image(eye_dir_x, eye_dir_y)

        disp_left.image(eye_img_left)
        disp_right.image(eye_img_right)

        # -------- 브라우저용 영상 스트림 (선택사항) --------
        if last_result is not None:
            annotated = last_result.plot()
            annotated = cv2.resize(annotated, (640, 480))
        else:
            annotated = frame

        ret_jpeg, jpeg = cv2.imencode(
            ".jpg",
            annotated,
            [int(cv2.IMWRITE_JPEG_QUALITY), 70],
        )

        if not ret_jpeg:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpeg.tobytes() +
            b"\r\n"
        )

        frame_count += 1

    cap.release()


@app.route("/video")
def video():
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/")
def index():
    return """
    <h1>YOLOv8 + 눈 시선 추적</h1>
    <img src="/video" width="640">
    """


if __name__ == "__main__":
    # Flask 서버를 켜면, /video 에 접속하는 순간
    # generate() 안에서 YOLO + 눈 제어가 같이 돌아감
    app.run(host="0.0.0.0", port=5000, debug=False)
