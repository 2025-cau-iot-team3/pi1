from flask import Flask, Response
import cv2
from ultralytics import YOLO
import time
import json
import os
from datetime import datetime

app = Flask(__name__)

# YOLO ONNX 모델
MODEL_PATH = "/home/mj/yolov8n.onnx"
model = YOLO(MODEL_PATH, task="detect")

# UDP 카메라 주소
CAM_URL = "udp://127.0.0.1:5000"

# YOLO 추론 간격 (프레임 수)
YOLO_INTERVAL = 4

# 실내에서 탐지할만한 객체만 허용
ALLOWED_NAME = {
    "person", "cat", "dog", "bottle", "cup", "bowl", "chair", "couch", "bed",
    "tv", "laptop", "mouse", "keyboard", "cell phone", "book", "clock",
    "toilet", "potted plant", "dining table", "refrigerator", "microwave",
    "oven", "toaster", "sink", "remote", "vase", "teddy bear", "hair drier",
    "toothbrush", "fork", "knife", "spoon", "wine glass", "cake", "pizza",
    "carrot", "apple", "banana", "broccoli", "orange", "handbag", "suitcase",
    "backpack", "umbrella"
}

# -------------------------------------------------------

def save_json(detection_data):
    """객체가 있을 때만 JSON 저장 + 일 단위 폴더 생성"""

    if len(detection_data["objects"]) == 0:
        return  # 빈 객체 → 저장 안 함

    today = datetime.now().strftime("%Y-%m-%d")
    folder = f"detections/{today}"

    # 폴더 자동 생성
    if not os.path.exists(folder):
        os.makedirs(folder)

    timestamp = datetime.now().strftime("%H-%M-%S")
    filename = f"{folder}/detections_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(detection_data, f, indent=4)

    print(f"📁 JSON saved: {filename}")

# -------------------------------------------------------


def generate():

    # UDP 스트림 안정화 파라미터
    cap = cv2.VideoCapture(
        f"{CAM_URL}?fifo_size=5000000&overrun_nonfatal=1",
        cv2.CAP_FFMPEG
    )

    if not cap.isOpened():
        print("❌ ERROR: Cannot open UDP camera stream")
        return

    print("⚡ YOLOv8 ONNX + Real-Time Stream Started")

    frame_count = 0
    last_boxes, last_scores, last_classes = [], [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue  # 최신 프레임만 처리

        # YOLO 추론
        if frame_count % YOLO_INTERVAL == 0:

            results = model(frame, verbose=False)[0]  # 원본 비율 유지

            if results.boxes:
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
            else:
                boxes, scores, classes = [], [], []

            # 필터링 (실내 객체만)
            filtered_boxes = []
            filtered_scores = []
            filtered_classes = []

            names = results.names

            for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes):
                label = names[int(cls)]
                if label in ALLOWED_NAME:
                    filtered_boxes.append([x1, y1, x2, y2])
                    filtered_scores.append(score)
                    filtered_classes.append(cls)

            last_boxes = filtered_boxes
            last_scores = filtered_scores
            last_classes = filtered_classes

            # 객체가 있을 때만 JSON 저장
            detection_data = {
                "timestamp": time.time(),
                "objects": []
            }

            for (x1, y1, x2, y2), score, cls in zip(last_boxes, last_scores, last_classes):
                detection_data["objects"].append({
                    "label": names[int(cls)],
                    "confidence": round(float(score), 3),
                    "bbox": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    }
                })

            save_json(detection_data)

        # 스트림 화면 그리기
        display = frame.copy()
        names = model.names

        for (x1, y1, x2, y2), score, cls in zip(last_boxes, last_scores, last_classes):
            cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)), 
                          (0, 255, 0), 2)
            cv2.putText(display, f"{names[int(cls)]} {score:.2f}",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        # 화면 해상도 유지
        display = cv2.resize(display, (640, 480))

        ret, jpeg = cv2.imencode(".jpg", display, 
                                 [int(cv2.IMWRITE_JPEG_QUALITY), 70])

        if not ret:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() +
               b"\r\n")

        frame_count += 1


# -------------------------------------------------------

@app.route("/video")
def video():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    return """
    <h1>⚡ YOLOv8 ONNX Real-Time Stream</h1>
    <img src="/video" width="640">
    """

# -------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
