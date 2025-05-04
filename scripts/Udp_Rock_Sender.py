import cv2
import numpy as np
import socket
import json
import time
from ultralytics import YOLO

UDP_IP = "192.168.1.178"  # Cambia a tu IP local si hace falta
UDP_PORT = 5051

ROCK_MODEL_PATH = "../models/rock_model.pt"
RULER_MODEL_PATH = "../models/notebook_model.pt"

DENSITY_DICT = {
    "Igneous": 2.7,
    "Metamorphic": 2.9,
    "Sedimentary": 2.5
}

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def estimate_volume(area_px, px_per_cm):
    area_cm = area_px / (px_per_cm ** 2)
    return area_cm * 2  # Asume grosor fijo

def send_udp_data(data_list):
    try:
        json_data = json.dumps(data_list)
        sock.sendto(json_data.encode(), (UDP_IP, UDP_PORT))
    except Exception as e:
        print(f"[UDP ERROR] {e}")

def draw_label(img, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    size, _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y - size[1] - 6), (x + size[0], y), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y - 2), font, scale, (255, 255, 255), thickness)

# Cargar modelos
ruler_model = YOLO(RULER_MODEL_PATH)
rock_model = YOLO(ROCK_MODEL_PATH)

cap = cv2.VideoCapture(0)
pixels_per_cm = None
ruler_time = None
calibration_duration = 5

print("[INFO] Waiting for calibration...")

# Calibración
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = ruler_model.predict(frame, conf=0.5)
    boxes = results[0].boxes

    if boxes:
        box = boxes[0].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)
        width_px = abs(x2 - x1)
        pixels_per_cm = width_px / 28.0

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        draw_label(frame, "Calibrating...", x1, y1)

        if ruler_time is None:
            ruler_time = time.time()
        elif time.time() - ruler_time >= calibration_duration:
            print(f"[INFO] Calibration done. Pixels/cm: {pixels_per_cm:.2f}")
            break
    else:
        ruler_time = None

    cv2.imshow("Calibration", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

print("[INFO] Starting rock detection...")

# Detección
while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    results = rock_model.predict(frame, conf=0.5)
    boxes = results[0].boxes

    rock_data = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy())
        conf = float(box.conf[0].cpu().numpy())
        label = results[0].names[cls]

        cx = (x1 + x2) / 2 / width
        cy = (y1 + y2) / 2 / height
        area = (x2 - x1) * (y2 - y1)
        volume = estimate_volume(area, pixels_per_cm)
        density = DENSITY_DICT.get(label, 2.6)
        mass = volume * density / 1000
        radius = ((x2 - x1) / 2) / pixels_per_cm

        rock_data.append({
            "x": cx,
            "y": cy,
            "type": label,
            "confidence": round(conf, 2),
            "mass": round(mass, 3),
            "radius": round(radius, 2),
            "rectangle": [x1, y1, x2, y2]
        })

        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), 2)
        draw_label(frame, f"{label} {conf*100:.1f}%", x1, y1)
        draw_label(frame, f"{mass:.2f} kg", x1, y2 + 20)

    if rock_data:
        send_udp_data(rock_data)

    cv2.imshow("Rock Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()