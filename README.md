# CodeAlpha Artificial Intelligence Internship
## Task 4: Real-Time Object Detection and Tracking with YOLOv8 & Streamlit

**Intern:** Vedika Utturwar  
**Student ID:** CA/DF1/308779  
**Domain:** Artificial Intelligence  
**Batch:** 1st October 2026 to 30th October 2026  

---

## 📌 Project Overview
A real-time object detection and tracking web application built with **Streamlit**, **OpenCV**, and **YOLOv8 (Ultralytics)**. The system identifies objects across 80 COCO categories and tracks them seamlessly across video frames using a Euclidean Centroid Tracking algorithm.

## ✨ Features
- **Real-Time YOLOv8 Detection**: High-accuracy, low-latency detection powered by YOLOv8 nano model (`yolov8n.pt`).
- **Centroid Object Tracking**: Assigns and maintains persistent, unique tracking IDs across video frames using Euclidean distance minimization.
- **Dual Input Sources**: Supports real-time webcam feed as well as video file upload (`.mp4`, `.avi`, `.mov`).
- **Interactive Controls**:
  - Adjustable confidence threshold slider (0.10 to 1.00).
  - Multi-class category filtering (e.g., person, car, bottle, dog).
  - "Detect All Classes" toggle for detecting all 80 classes at once.
  - Video looping toggle for continuous playback.
- **Dynamic Real-Time Analytics**: Live scoreboard showing counts of active detected and tracked objects.
- **One-Click Launcher**: Quick execution via `run.bat` script.

## 🛠️ Technologies Used
- **Python 3**
- **Streamlit**: Interactive web UI and controls
- **Ultralytics YOLOv8**: State-of-the-art deep learning object detection
- **OpenCV (`cv2`)**: Video capture, frame preprocessing, and bounding box rendering
- **SciPy & NumPy**: Vector calculations and Centroid distance tracking

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Vedika-u/CodeAlpha_ObjectDetectionTracking.git
   cd CodeAlpha_ObjectDetectionTracking
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Running the Application

### Option 1: One-Click Launcher (Windows)
Double-click `run.bat` in the project root.

### Option 2: Command Line
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

## 📂 Project Structure
```text
CodeAlpha_ObjectDetectionTracking/
├── app.py              # Main Streamlit web application
├── detector.py         # YOLOv8 wrapper for bounding box inference
├── tracker.py          # CentroidTracker algorithm implementation
├── utils.py            # Drawing routines for bounding boxes & labels
├── requirements.txt    # Project dependencies
├── run.bat             # One-click Windows startup script
├── .gitignore          # Git exclusion rules
└── README.md           # Project documentation
```

## 📄 License
This project is open-source and licensed under the MIT License.

