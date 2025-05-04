import cv2
import numpy as np
import time
from ultralytics import YOLO

# Configuration
YOLO_MODEL_PATH = "C:/Users/test/Documents/RockVision/runs/detect/train3/weights/best.pt"  # Adjust path if needed
VIDEO_SOURCE = 0  # Default webcam
RULER_REAL_WIDTH_CM = 9.0
RULER_REAL_HEIGHT_CM = 28.0
MIN_AREA_THRESHOLD = 5000
STABILITY_FRAMES = 15
STABILITY_TOLERANCE = 15

# Color definitions (BGR)
COLOR_RULER = (0, 0, 0)  # Black
COLOR_ROCK_BOX = (0, 0, 0)  # Black
COLOR_TEXT = (0, 0, 0)  # Black

# Densities (in g/cm³)
ROCK_DENSITIES = {
    'Igneous': 2.7,
    'Metamorphic': 2.9,
    'Sedimentary': 2.5
}

# Detect the notebook/ruler
cap = cv2.VideoCapture(VIDEO_SOURCE)
pixels_per_cm = None
stability_counter = 0
previous_contour = None

print("[INFO] Waiting for ruler...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot access camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ruler_contour = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        area = cv2.contourArea(contour)
        if len(approx) == 4 and area > MIN_AREA_THRESHOLD:
            ruler_contour = contour
            break

    if ruler_contour is not None:
        x, y, w, h = cv2.boundingRect(ruler_contour)
        if previous_contour is not None:
            dx = abs(w - previous_contour[0])
            dy = abs(h - previous_contour[1])
            if dx < STABILITY_TOLERANCE and dy < STABILITY_TOLERANCE:
                stability_counter += 1
            else:
                stability_counter = 0
        previous_contour = (w, h)

        color = (0, 255, 0) if stability_counter >= STABILITY_FRAMES else COLOR_RULER
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, "Detecting ruler...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)

        if stability_counter >= STABILITY_FRAMES:
            pixels_per_cm = w / RULER_REAL_WIDTH_CM
            print(f"[INFO] Ruler detected. Pixels per cm: {pixels_per_cm:.2f}")
            time.sleep(1)
            break

    cv2.imshow("Calibration", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyAllWindows()

# Load rock detection model
rock_model = YOLO(YOLO_MODEL_PATH)
print("[INFO] Please place rocks in the frame...")

# Detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Cannot access camera.")
        break

    results = rock_model(frame, verbose=False)[0]

    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = rock_model.names[class_id]
            confidence = float(box.conf[0])

            width_px = x2 - x1
            height_px = y2 - y1
            width_cm = width_px / pixels_per_cm
            height_cm = height_px / pixels_per_cm
            depth_cm = (width_cm + height_cm) / 4
            volume_cm3 = width_cm * height_cm * depth_cm
            density = ROCK_DENSITIES.get(class_name, 2.5)
            mass_g = volume_cm3 * density

            label = f"{class_name}: {mass_g:.1f}g"
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ROCK_BOX, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)

    cv2.imshow("Rock Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
