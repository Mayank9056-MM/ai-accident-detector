# 🚨 AI-Based Traffic Accident Detection System

An AI-powered system that detects **road accidents from video footage** using
computer vision, object tracking, speed estimation, and collision logic.

Built using **YOLOv8 + ByteTrack**, this project analyzes traffic videos frame by frame
to identify **sudden speed drops combined with vehicle collisions**, and flags them
as accidents in real time.

---

## 🎯 What Problem This Solves

Traditional CCTV systems only record footage — they **do not understand** what is
happening.

This system:

- Automatically detects vehicles
- Tracks them across frames
- Calculates their speed
- Detects collisions
- Confirms accidents using intelligent logic

👉 Useful for **smart cities, traffic monitoring, emergency response**, and **road safety systems**.

---

## 🧠 High-Level System Flow

# 🚨 AI-Based Traffic Accident Detection System

An AI-powered system that detects **road accidents from video footage** using
computer vision, object tracking, speed estimation, and collision logic.

Built using **YOLOv8 + ByteTrack**, this project analyzes traffic videos frame by frame
to identify **sudden speed drops combined with vehicle collisions**, and flags them
as accidents in real time.

---

## 🎯 What Problem This Solves

Traditional CCTV systems only record footage — they **do not understand** what is
happening.

This system:

- Automatically detects vehicles
- Tracks them across frames
- Calculates their speed
- Detects collisions
- Confirms accidents using intelligent logic

👉 Useful for **smart cities, traffic monitoring, emergency response**, and **road safety systems**.

---

## 🧠 High-Level System Flow

# 🚨 AI-Based Traffic Accident Detection System

An AI-powered system that detects **road accidents from video footage** using
computer vision, object tracking, speed estimation, and collision logic.

Built using **YOLOv8 + ByteTrack**, this project analyzes traffic videos frame by frame
to identify **sudden speed drops combined with vehicle collisions**, and flags them
as accidents in real time.

---

## 🎯 What Problem This Solves

Traditional CCTV systems only record footage — they **do not understand** what is
happening.

This system:

- Automatically detects vehicles
- Tracks them across frames
- Calculates their speed
- Detects collisions
- Confirms accidents using intelligent logic

👉 Useful for **smart cities, traffic monitoring, emergency response**, and **road safety systems**.

---

## 🧠 High-Level System Flow

```bash
    Video Input
    ↓
    YOLOv8 Object Detection
    ↓
    ByteTrack Vehicle Tracking (Persistent IDs)
    ↓
    Speed Calculation (Frame-to-frame movement)
    ↓
    Collision Detection (IoU)
    ↓
    Accident Logic (Speed drop + overlap)
    ↓
    Human-readable alerts / Backend-ready events
```

## 🛠️ Technologies Used

- **Python 3**
- **Ultralytics YOLOv8**
- **ByteTrack** (multi-object tracking)
- **OpenCV**
- **NumPy**
- **PyTorch**

---

## 📂 Project Structure

## 🛠️ Technologies Used

- **Python 3**
- **Ultralytics YOLOv8**
- **ByteTrack** (multi-object tracking)
- **OpenCV**
- **NumPy**
- **PyTorch**

---

## 📂 Project Structure

```bash
ai-service/
├── detection_accident.py # Main pipeline (entry point)
├── test_yolo.py # YOLO testing script
├── track_vehicle.py # Tracking experiments
│
├── utils/
│ ├── speed.py # Speed calculation logic
│ ├── collision.py # IoU collision detection
│ └── accident_logic.py # Accident decision rules
│
├── public/
│ ├── traffic.mp4 # Test traffic video
│ └── no_accidents.mp4 # Control (no accident) video
│
├── models/
│ └── yolov8n.pt # YOLOv8 nano model (not committed)
│
├── .gitignore
└── README.md
```

## 🚀 How the System Works (Detailed)
