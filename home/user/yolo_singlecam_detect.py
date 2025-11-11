#!/usr/bin/env python3
from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import time
import os
import numpy as np

# ===== YOLO 객체인식 + 이미지 저장 버전 =====
IMG_SIZE = (320, 240)
CONF_TH = 0.3
IOU_TH = 0.45
SAVE_DIR = "/home/user/detected"
SAVE_INTERVAL = 1.0  # 초 단위, 저장 주기 설정 (1초마다 저장)

def main():
    # 모델 로드
    model = YOLO("/home/user/yolov8n.pt")
    names = model.model.names if hasattr(model.model, "names") else None

    # 저장 폴더 생성
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 카메라 설정
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(main={"size": IMG_SIZE, "format": "RGB888"}))
    cam.start()
    time.sleep(0.2)

    print("YOLO 객체 인식 및 자동 저장 시작 (단일 카메라)")

    last_save = 0
    frame_id = 0

    try:
        while True:
            frame = cam.capture_array()  # RGB 포맷
            result = model.predict(frame, imgsz=max(IMG_SIZE), conf=CONF_TH, iou=IOU_TH, verbose=False)
            boxes = result[0].boxes
            vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 객체 감지 시 표시
            if boxes is not None and boxes.xyxy.numel() > 0:
                xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
                cls = boxes.cls.cpu().numpy().astype(np.int32)
                conf = boxes.conf.cpu().numpy().astype(np.float32)

                for i, box in enumerate(xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{names[cls[i]] if names else cls[i]} {conf[i]:.2f}"
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # 결과 화면 출력
            cv2.imshow("YOLOv8 Object Detection", vis)

            # 일정 주기마다 감지 이미지 저장
            now = time.time()
            if now - last_save > SAVE_INTERVAL:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                save_path = os.path.join(SAVE_DIR, f"{timestamp}_{frame_id:06d}.jpg")
                cv2.imwrite(save_path, vis)
                print(f"이미지 저장됨: {save_path}")
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
