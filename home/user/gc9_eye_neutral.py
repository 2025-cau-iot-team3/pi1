import time
import digitalio
import board
from PIL import Image, ImageDraw
from adafruit_rgb_display import gc9a01a

print("=== GC9A01 두 눈 테스트 (중립 눈 2개, DC/RST 공유) ===")

BAUDRATE = 24_000_000
spi = board.SPI()  # SCLK=GPIO11, MOSI=GPIO10

# DC, RST 핀은 두 눈이 같이 사용
dc_pin = digitalio.DigitalInOut(board.D25)     # GPIO25, 핀 22
rst_pin = digitalio.DigitalInOut(board.D27)    # GPIO27, 핀 13

# -----------------------------
# 1. 왼쪽 눈 디스플레이 (CE0)
# -----------------------------
cs_left = digitalio.DigitalInOut(board.CE0)    # GPIO8, 핀 24

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

# -----------------------------
# 2. 오른쪽 눈 디스플레이 (CE1)
# -----------------------------
cs_right = digitalio.DigitalInOut(board.CE1)   # GPIO7, 핀 26

disp_right = gc9a01a.GC9A01A(
    spi,
    cs=cs_right,
    dc=dc_pin,   # 같은 DC, RST 공유
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
# 중립 눈 이미지
# -----------------------------
def create_neutral_eye():
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, width, height), fill=(0, 0, 0))

    cx, cy = width // 2, height // 2
    r_eye = min(width, height) // 2 - 4

    # 흰 눈
    draw.ellipse(
        (cx - r_eye, cy - r_eye, cx + r_eye, cy + r_eye),
        fill=(255, 255, 255),
    )

    # 큰 동공
    r_pupil = r_eye // 1.5
    draw.ellipse(
        (cx - r_pupil, cy - r_pupil, cx + r_pupil, cy + r_pupil),
        fill=(0, 0, 0),
    )

    # 하이라이트
    r_high = r_pupil // 2

    draw.ellipse(
        (cx - r_pupil + 5, cy - r_pupil + 5,
         cx - r_pupil + 5 + 2 * r_high, cy - r_pupil + 5 + 2 * r_high),
        fill=(200, 200, 255),
    )

    return image

eye_img = create_neutral_eye()

disp_left.image(eye_img)
disp_right.image(eye_img)
print("[OK] 두 눈 모두 중립 눈 표시 완료!")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n종료")
