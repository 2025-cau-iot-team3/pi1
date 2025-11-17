from flask import Flask, Response
import cv2
from ultralytics import YOLO
import time

app = Flask(__name__)

# YOLOv8 모델 로컬 경로 (네 파일 경로로 맞춰!)
MODEL_PATH = "/home/mj/yolov8n.pt"
model = YOLO(MODEL_PATH)

CAM_INDEX = 0  # USB 카메라 번호

# 프레임 간 YOLO 추론 간격 (4프레임마다 1번 YOLO → 속도 대폭 증가)
YOLO_INTERVAL = 4


def generate():
    cap = cv2.VideoCapture(CAM_INDEX)

    # 카메라 캡처 성능 최적화
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ ERROR: Cannot open camera")
        return

    print("⚡ YOLOv8 Fast Stream Started")

    frame_count = 0
    last_result = None  # 최근 YOLO 결과 저장

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # ---- YOLO 추론 (4프레임마다 1번만 실행) ----
        if frame_count % YOLO_INTERVAL == 0:
            # YOLO는 작은 해상도로 넣으면 속도 ↑
            small = cv2.resize(frame, (320, 320))

            results = model(small, stream=True)

            for r in results:
                last_result = r  # 최신 결과 덮어쓰기

        # YOLO 결과를 현재 프레임에 그리기
        if last_result is not None:
            annotated = last_result.plot()
            annotated = cv2.resize(annotated, (640, 480))
        else:
            annotated = frame

        # JPEG 인코딩 (quality 낮추면 스트리밍 더 빨라짐)
        ret, jpeg = cv2.imencode(".jpg", annotated, [
            int(cv2.IMWRITE_JPEG_QUALITY), 70  # 품질 70%
        ])

        if not ret:
            continue

        # ---- 브라우저 스트리밍 ----
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() +
               b"\r\n")

        frame_count += 1

    cap.release()


@app.route("/video")
def video():
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/")
def index():
    return """
    <h1>⚡ YOLOv8 Real-Time Stream (Optimized)</h1>
    <img src="/video" width="640">
    """


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
