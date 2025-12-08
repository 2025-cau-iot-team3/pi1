from flask import Flask, Response
import cv2
from ultralytics import YOLO
import time

import digitalio
import board
from PIL import Image, ImageDraw
from adafruit_rgb_display import gc9a01a

import requests  # p2로 HTTP 요청 보낼 때 사용

# -----------------------------
# p2 서버 설정 (팔 서보모터가 있는 라즈베리파이)
# -----------------------------
# 👉 p2의 IP 주소에 맞게 수정해줘!
P2_EMOTION_URL = "http://192.168.0.50:5001/emotion"  # 예시 IP

def send_emotion_to_p2(emotion: str):
    """
    emotion: "happy" 또는 "pet"
    """
    try:
        resp = requests.post(
            P2_EMOTION_URL,
            json={"emotion": emotion},
            timeout=0.5,
        )
        print(f"[P1->P2] emotion '{emotion}' 전송, status={resp.status_code}", flush=True)
    except Exception as e:
        print(f"[WARN] P2로 emotion 전송 실패: {e}", flush=True)


# -----------------------------
# YOLO + 눈 디스플레이 설정
# -----------------------------
app = Flask(__name__)

MODEL_PATH = "/home/mj/yolov8n.pt"
model = YOLO(MODEL_PATH)

CAM_INDEX = 0
YOLO_INTERVAL = 4

print("=== GC9A01 눈 + YOLO (p1) ===")

BAUDRATE = 24_000_000
spi = board.SPI()

dc_pin = digitalio.DigitalInOut(board.D25)
rst_pin = digitalio.DigitalInOut(board.D27)

# 왼쪽 눈
cs_left = digitalio.DigitalInOut(board.CE0)
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

# 오른쪽 눈
cs_right = digitalio.DigitalInOut(board.CE1)
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

def create_eye_image(offset_x: float, offset_y: float) -> Image.Image:
    """offset_x, offset_y: -1.0 ~ 1.0 시선 방향"""
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

    # 동공
    r_pupil = int(r_eye / 1.5)
    max_offset = r_eye - r_pupil - 6
    ox = int(max(-1.0, min(1.0, offset_x)) * max_offset)
    oy = int(max(-1.0, min(1.0, offset_y)) * max_offset)

    px = cx + ox
    py = cy + oy

    draw.ellipse(
        (px - r_pupil, py - r_pupil, px + r_pupil, py + r_pupil),
        fill=(0, 0, 0),
    )

    # 하이라이트
    r_high = r_pupil // 2
    hx = px - r_pupil + 5
    hy = py - r_pupil + 5
    draw.ellipse(
        (hx, hy, hx + 2 * r_high, hy + 2 * r_high),
        fill=(200, 200, 255),
    )

    return img

# 초기 중립 눈
eye_dir_x = 0.0
eye_dir_y = 0.0
target_dir_x = 0.0
target_dir_y = 0.0

neutral_eye = create_eye_image(0.0, 0.0)
disp_left.image(neutral_eye)
disp_right.image(neutral_eye)
print("[OK] 두 눈 중립 상태 초기화 완료")

# 음식류 (COCO 기준 대략)
FOOD_CLASSES = {
    "apple", "banana", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "bottle", "cup", "bowl",
    "fork", "knife", "spoon"
}
PET_CLASSES = {"dog", "cat"}

# 너무 자주 감정 보내지 않도록 쿨다운
last_sent_time = 0.0
COOLDOWN_SEC = 5.0

def generate():
    global eye_dir_x, eye_dir_y, target_dir_x, target_dir_y, last_sent_time

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ ERROR: Cannot open camera")
        return

    print("⚡ YOLOv8 + 눈 디스플레이 (p1) 시작")

    frame_count = 0
    last_result = None

    YOLO_W, YOLO_H = 320, 320

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        main_box_center = None
        main_label = None

        # ---------- YOLO 추론 ----------
        if frame_count % YOLO_INTERVAL == 0:
            small = cv2.resize(frame, (YOLO_W, YOLO_H))
            results = model(small, stream=True)

            for r in results:
                last_result = r

                boxes = r.boxes
                if boxes is not None and len(boxes) > 0:
                    # 가장 큰 박스를 기준으로
                    xyxy = boxes.xyxy.cpu().numpy()
                    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                    idx = areas.argmax()
                    x1, y1, x2, y2 = xyxy[idx]
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    main_box_center = (cx, cy)

                    # 클래스 라벨
                    cls_tensor = boxes.cls[idx]
                    try:
                        cls_id = int(cls_tensor.item())
                    except AttributeError:
                        cls_id = int(cls_tensor)
                    label = r.names[int(cls_id)]
                    main_label = label

                    print(f"[YOLO] main object: {label} @ ({cx:.1f}, {cy:.1f})", flush=True)

                    # ----- 감정 판단 & p2로 전송 -----
                    now = time.time()
                    if (now - last_sent_time) > COOLDOWN_SEC:
                        # 1) 사람 or 음식류 → 'happy'
                        if label == "person" or label in FOOD_CLASSES:
                            send_emotion_to_p2("happy")
                            last_sent_time = now
                        # 2) 강아지 / 고양이 → 'pet'
                        elif label in PET_CLASSES:
                            send_emotion_to_p2("pet")
                            last_sent_time = now

        # ---------- 눈 시선 방향 업데이트 ----------
        if main_box_center is not None:
            cx, cy = main_box_center
            dx = (cx - YOLO_W / 2.0) / (YOLO_W / 2.0)
            dy = (cy - YOLO_H / 2.0) / (YOLO_H / 2.0)
            target_dir_x = dx
            target_dir_y = dy
        else:
            target_dir_x = 0.0
            target_dir_y = 0.0

        alpha = 0.2
        eye_dir_x = (1 - alpha) * eye_dir_x + alpha * target_dir_x
        eye_dir_y = (1 - alpha) * eye_dir_y + alpha * target_dir_y

        eye_img_left = create_eye_image(eye_dir_x, eye_dir_y)
        eye_img_right = create_eye_image(eye_dir_x, eye_dir_y)
        disp_left.image(eye_img_left)
        disp_right.image(eye_img_right)

        # ---------- 브라우저 스트림 ----------
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
    <h1>⚡ YOLOv8 + 눈 시선 추적 (p1) + p2로 감정 전송</h1>
    <ul>
      <li>person / 음식류 → p2에 'happy' 전송 → 팔 위/아래 3번 흔들기</li>
      <li>dog / cat → p2에 'pet' 전송 → 팔 위로 3초 들기</li>
    </ul>
    <img src="/video" width="640">
    """

if __name__ == "__main__":
    # p1에서 실행 예:
    #   python3 eyes_yolo_p1.py
    app.run(host="0.0.0.0", port=5000, debug=False)
