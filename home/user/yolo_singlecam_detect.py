#!/usr/bin/env python3
from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import time
import os
import numpy as np

# ===== YOLO 객체인식 (집 안 객체 전용) =====
IMG_SIZE = (320, 240)
CONF_TH = 0.3
IOU_TH = 0.45
SAVE_DIR = "/home/user/detected"
SAVE_INTERVAL = 1.0  # 초 단위 저장 주기

# 집 안에서 관찰 가능한 객체 클래스
ALLOWED_NAME = {
    "person", "cat", "dog", "bottle", "cup", "bowl", "chair", "couch", "bed",
    "tv", "laptop", "mouse", "keyboard", "cell phone", "book", "clock",
    "toilet", "potted plant", "dining table", "refrigerator", "microwave",
    "oven", "toaster", "sink", "remote", "vase", "teddy bear", "hair drier",
    "toothbrush", "fork", "knife", "spoon", "wine glass", "cake", "pizza",
    "carrot", "apple", "banana", "broccoli", "orange", "handbag", "suitcase",
    "backpack", "umbrella"
}

def main():
    # YOLOv8 모델 로드
    model_path = "/home/user/yolov8n.pt"
    if not os.path.exists(model_path):
        print("YOLOv8n 모델 다운로드 중...")
        os.system(f"wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O {model_path}")
    model = YOLO(model_path)
    names = model.model.names if hasattr(model.model, "names") else None

    # 허용 클래스 이름을 ID로 변환
    allowed_ids = set()
    if names:
        inv = {str(v).lower(): k for k, v in names.items()} if isinstance(names, dict) else {v.lower(): i for i, v in enumerate(names)}
        for n in ALLOWED_NAME:
            if n in inv:
                allowed_ids.add(inv[n])
    print(f"허용된 클래스 수: {len(allowed_ids)}개")

    # 저장 폴더 준비
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 카메라 설정
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(main={"size": IMG_SIZE, "format": "RGB888"}))
    cam.start()
    time.sleep(0.2)

    print("YOLO 객체 인식 (집 안 객체 전용) 시작")

    last_save = 0
    frame_id = 0

    try:
        while True:
            frame = cam.capture_array()  # RGB 포맷
            result = model.predict(frame, imgsz=max(IMG_SIZE), conf=CONF_TH, iou=IOU_TH, verbose=False)
            boxes = result[0].boxes
            vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 감지 및 필터링
            if boxes is not None and boxes.xyxy.numel() > 0:
                xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
                cls = boxes.cls.cpu().numpy().astype(np.int32)
                conf = boxes.conf.cpu().numpy().astype(np.float32)

                for i, box in enumerate(xyxy):
                    if cls[i] not in allowed_ids:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{names[cls[i]] if names else cls[i]} {conf[i]:.2f}"
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # 화면 표시
            cv2.imshow("YOLOv8 Indoor Object Detection", vis)

            # 이미지 저장
            now = time.time()
            if now - last_save > SAVE_INTERVAL:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                save_path = os.path.join(SAVE_DIR, f"{timestamp}_{frame_id:06d}.jpg")
                cv2.imwrite(save_path, vis)
                print(f"감지 이미지 저장됨: {save_path}")
                last_save = now

            frame_id += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("프로그램 종료")

if __name__ == "__main__":
    main()
