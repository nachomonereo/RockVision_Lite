# 🪨 RockVision Lite
A lightweight real-time stone detection and classification system using YOLOv5, OpenCV, and UDP communication with Grasshopper.

---

## 📁 Repository Structure

```
RockVision_Lite/
├── datasets/               # Datasets for training (optional)
├── env/                    # Virtual environment (excluded from Git)
├── models/
│   ├── rock_model.pt       # YOLO model for rock classification
│   └── notebook_model.pt   # YOLO model for calibration (notebook detection)
├── scripts/
│   ├── main_rock_detection.py     # Main detection and UDP sender script
│   ├── Udp_Rock_Sender.py         # Standalone version for sending rock data via UDP
│   └── udp-import-rock-vision.gh  # Grasshopper script for receiving and parsing UDP data
├── src/                    # Additional source files
├── .gitignore
└── README.md
```

---

## ⚙️ Requirements

- Python 3.12+
- Git
- Rhino + Grasshopper (for receiving data)
- A working webcam

---

## 🔧 Installation (Step-by-step)

1. **Clone the repository**

```bash
git clone git@github.com:nachomonereo/RockVision_Lite.git
cd RockVision_Lite
```

2. **Create a virtual environment**

```bash
python -m venv env
.\env\Scriptsctivate   # PowerShell
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, manually install:

```bash
pip install ultralytics opencv-python numpy
```

4. **Place the trained models**

Ensure the following files are in the `models/` folder:

- `rock_model.pt`
- `notebook_model.pt`

---

## ▶️ Run the Detection System

```bash
cd scripts
python main_rock_detection.py
```

The script:

- Detects a notebook (used as a scale)
- Calibrates pixels/cm
- Detects rocks and sends: type, mass, radius, normalized position and bounding box via **UDP**

---

## 🧠 Grasshopper Integration

Open `udp-import-rock-vision.gh` in Grasshopper.

- Set the same UDP port (default: `5051`)
- Toggle `Run` to start listening
- Outputs:
  - Normalized (x, y) position
  - Rock type
  - Confidence
  - Estimated mass (kg)
  - Bounding rectangle
  - Inscribed circle radius

---

## 📡 UDP Settings

- IP: `127.0.0.1` (localhost) or local IP (e.g., `192.168.1.xxx`)
- Port: `5051` (adjustable in both scripts)

---

## 🧪 Optional: Train Your Own Models

```bash
yolo task=detect mode=train model=yolov5s.pt data=datasets/data.yaml epochs=100 imgsz=416
```

Move the resulting `best.pt` model to the `models/` folder.

---

## 🪪 License

MIT License © 2025 Nacho Monereo

---

## 📬 Contact

For questions or suggestions, open an issue or contact [nachomonereo](https://github.com/nachomonereo)