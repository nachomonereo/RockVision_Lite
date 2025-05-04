# RockVision Lite

A Python + YOLOv8 + Grasshopper project for real-time rock type classification, mass estimation and spatial communication using UDP.

---

## 🔍 Features

- Calibrate physical size using a notebook (28 cm)
- Detect and classify rocks: **Igneous**, **Metamorphic**, **Sedimentary**
- Estimate their mass using bounding box and density
- Send data to **Grasshopper** (x, y, rock type, confidence, bounding box, radius, mass) via UDP
- Real-time visualization

---

## 📁 Folder Structure

```
RockVision_Lite/
├── datasets/
├── env/                  # virtual environment (excluded via .gitignore)
├── models/
│   ├── rock_model.pt
│   └── notebook_model.pt
├── scripts/
│   ├── Udp_Rock_Sender.py     # main script (run this one)
│   ├── udp-import-rock-vision.gh
├── src/
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/nachomonereo/RockVision_Lite.git
cd RockVision_Lite
```

### 2. Create and Activate Virtual Environment

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

### 3. Install Requirements

```bash
pip install ultralytics opencv-python
```

---

## 🧠 Model Training

You can train your own models or use the one from:

👉 [Rock Classification - Roboflow Dataset](https://universe.roboflow.com/bandrma-onyedi-eyll-niversitesi-byubb/rock-classification-bvis1?utm_source=chatgpt.com)

---

## ▶️ Run the Detection Script

```bash
python scripts/Udp_Rock_Sender.py
```

Then open the Grasshopper file:

```bash
scripts/udp-import-rock-vision.gh
```

---

## 📡 UDP Format

The Python sender transmits a list of rocks like this:

```json
[
  {
    "x": 0.48,
    "y": 0.52,
    "type": "Igneous",
    "confidence": 0.91,
    "mass": 1.24,
    "radius": 3.1,
    "rectangle": [x1, y1, x2, y2]
  },
  ...
]
```

---

## 🛠 Troubleshooting

- Run Grasshopper as administrator if UDP fails to bind
- Make sure IP and port are matching
- UDP port: `5051` by default
- Avoid multiple bindings to same port (error 10048)

---

## 👨‍💻 Author

Developed by [Nacho Monereo](https://nachomonereo.com)  
Model trained using data from [Roboflow Rock Dataset](https://universe.roboflow.com/bandrma-onyedi-eyll-niversitesi-byubb/rock-classification-bvis1?utm_source=chatgpt.com)