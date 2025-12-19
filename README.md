# 🚨 AI Accident Detection & Emergency Alert System

An AI-powered system that automatically detects road accidents from video feeds
and sends real-time emergency alerts with location and severity details.

---

## 📌 Problem Statement

Road accidents often go unreported for several minutes, especially on highways
and during nighttime. Delayed emergency response increases fatalities.

This system eliminates human dependency by using Artificial Intelligence to
detect accidents instantly and alert authorities.

---

## 💡 Solution Overview

Our system uses Computer Vision and AI to:
- Detect vehicles in traffic videos
- Track vehicle movement and speed
- Identify collisions and sudden stops
- Estimate accident severity
- Trigger emergency alerts automatically

---

## 🧠 AI Workflow

1. **Video Input** (CCTV / Dashcam / Recorded Video)
2. **Object Detection** using YOLOv8
3. **Vehicle Tracking** using DeepSORT
4. **Collision Detection** using mathematical logic
5. **Severity Estimation**
6. **Emergency Alert Trigger**

---

## 🏗️ System Architecture

Camera / Video
↓
AI Service (Python)
↓
Backend API (Node.js)
↓
Database + Alerts
↓
React Dashboard


---

## 🧰 Tech Stack

### AI & Computer Vision
- Python
- YOLOv8
- DeepSORT
- OpenCV

### Backend
- Node.js
- Express.js
- MongoDB
- Socket.IO

### Frontend
- React (Vite)
- shadcn UI
- Mapbox / Google Maps API

---

## 🚀 Features

- Automatic accident detection
- Vehicle collision analysis
- Severity classification (Low / Medium / High)
- Real-time dashboard alerts
- GPS-based location tracking
- Accident image/video evidence

---

## 📂 Folder Structure

- **ai-service/** → AI detection & analysis
- **backend/** → APIs, database, alerts
- **frontend/** → Dashboard UI
- **docs/** → Architecture, demo, PPT

---

## 🧪 How to Run Locally

### 1️⃣ AI Service
```bash
cd ai-service
pip install -r requirements.txt
python detect_accident.py
``