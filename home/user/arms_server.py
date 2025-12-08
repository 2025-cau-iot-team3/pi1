#!/usr/bin/env python3
import time
import threading
from flask import Flask, request, jsonify
import sys

print("=== 팔 서보 감정 모션 서버 시작 ===", flush=True)

# -----------------------------
# RPi.GPIO import
# -----------------------------
HAS_GPIO = True
try:
    import RPi.GPIO as GPIO
except Exception as e:
    print("[ERROR] RPi.GPIO 모듈을 불러오지 못했습니다.", flush=True)
    print("       sudo apt-get install python3-rpi.gpio 를 실행했는지 확인하세요.", flush=True)
    print("       상세 에러:", e, flush=True)
    HAS_GPIO = False

# -----------------------------
# 핀 설정
# -----------------------------
LEFT_PIN = 17   # 왼팔 서보 신호 (GPIO17, 물리 핀 11)
RIGHT_PIN = 27  # 오른팔 서보 신호 (GPIO27, 물리 핀 13)
PWM_FREQ = 50   # 서보 주파수 50Hz

left_pwm = None
right_pwm = None
servos_initialized = False

motion_running = False
motion_lock = threading.Lock()

# -----------------------------
# 유틸 함수
# -----------------------------
def angle_to_duty(angle):
    """0~180도 → 서보 PWM 듀티비로 변환"""
    angle = max(0, min(180, angle))
    duty = 2.5 + (angle / 180.0) * 10.0
    print(f"[DEBUG] angle={angle} -> duty={duty:.2f}", flush=True)
    return duty

def setup_servos():
    global left_pwm, right_pwm, servos_initialized
    if not HAS_GPIO:
        print("[WARN] GPIO 미지원 환경 - 서보 초기화 생략", flush=True)
        return

    if servos_initialized:
        return

    print("[INFO] GPIO BCM 모드 설정", flush=True)
    GPIO.setmode(GPIO.BCM)

    print(f"[INFO] LEFT_PIN={LEFT_PIN}, RIGHT_PIN={RIGHT_PIN} 출력 설정", flush=True)
    GPIO.setup(LEFT_PIN, GPIO.OUT)
    GPIO.setup(RIGHT_PIN, GPIO.OUT)

    print(f"[INFO] PWM 시작 (freq={PWM_FREQ}Hz)", flush=True)
    left_pwm = GPIO.PWM(LEFT_PIN, PWM_FREQ)
    right_pwm = GPIO.PWM(RIGHT_PIN, PWM_FREQ)

    print("[INFO] 초기 각도 90도로 설정", flush=True)
    left_pwm.start(angle_to_duty(90))
    right_pwm.start(angle_to_duty(90))
    time.sleep(0.5)

    servos_initialized = True

def set_arms(left_angle, right_angle, delay=0.2):
    """두 팔 각도 설정"""
    if not (HAS_GPIO and servos_initialized):
        print(f"[MOCK] set_arms: left={left_angle}, right={right_angle}, delay={delay}", flush=True)
        time.sleep(delay)
        return

    print(f"[INFO] set_arms: left={left_angle}, right={right_angle}, delay={delay}", flush=True)
    left_pwm.ChangeDutyCycle(angle_to_duty(left_angle))
    right_pwm.ChangeDutyCycle(angle_to_duty(right_angle))
    time.sleep(delay)

# -----------------------------
# 감정별 모션
# -----------------------------
def motion_happy():
    """
    사람 / 음식 탐지 → '행복' 모션:
    팔을 위/아래로 3번 흔들기
    """
    print("[INFO] [EMOTION] 행복 모션 시작 (팔 위/아래 3번)", flush=True)

    down_angle = 60
    up_angle   = 140

    # 시작 자세: 살짝 아래
    set_arms(down_angle, down_angle, delay=0.4)

    for i in range(3):
        print(f"[INFO] happy wave #{i+1} - UP", flush=True)
        set_arms(up_angle, up_angle, delay=0.25)
        print(f"[INFO] happy wave #{i+1} - DOWN", flush=True)
        set_arms(down_angle, down_angle, delay=0.25)

    print("[INFO] happy 모션 끝, 중간 자세 복귀", flush=True)
    set_arms(90, 90, delay=0.4)

def motion_pet_hold():
    """
    강아지 / 고양이 탐지 →
    팔을 위로 3초 동안 들고 있기
    """
    print("[INFO] [EMOTION] 반려동물 모션 시작 (팔 위로 3초 유지)", flush=True)

    up_angle = 150

    set_arms(up_angle, up_angle, delay=0.4)
    print("[INFO] 반려동물 자세 유지 3초", flush=True)
    time.sleep(3.0)

    print("[INFO] pet 모션 끝, 중간 자세 복귀", flush=True)
    set_arms(90, 90, delay=0.4)

def run_motion(emotion):
    global motion_running
    with motion_lock:
        if motion_running:
            print("[INFO] 이미 모션 실행 중 - 새 요청 무시", flush=True)
            return
        motion_running = True

    try:
        setup_servos()

        if emotion == "happy":
            motion_happy()
        elif emotion == "pet":
            motion_pet_hold()
        else:
            print(f"[WARN] 알 수 없는 emotion: {emotion}", flush=True)
    finally:
        with motion_lock:
            motion_running = False
        print(f"[INFO] emotion '{emotion}' 모션 종료", flush=True)

# -----------------------------
# Flask 서버
# -----------------------------
app = Flask(__name__)

@app.route("/emotion", methods=["POST"])
def emotion():
    data = request.get_json(silent=True) or {}
    emotion = str(data.get("emotion", "")).lower()

    print(f"[HTTP] /emotion 요청 수신: {emotion}", flush=True)

    if emotion not in ("happy", "pet"):
        return jsonify({"ok": False, "msg": "emotion must be 'happy' or 'pet'"}), 400

    # 비동기로 모션 실행
    t = threading.Thread(target=run_motion, args=(emotion,), daemon=True)
    t.start()

    return jsonify({"ok": True, "msg": f"motion '{emotion}' started"})

if __name__ == "__main__":
    # p2에서 실행 예:
    #   sudo python3 arms_server.py
    #
    # p1에서 접근할 수 있도록 p2 IP와 포트를 확인할 것
    app.run(host="0.0.0.0", port=5001, debug=False)
