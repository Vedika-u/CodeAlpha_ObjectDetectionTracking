# Object Detection and Tracking with YOLOv8 & Streamlit

A real-time object detection and tracking web application built with Streamlit, OpenCV, and YOLOv8.

## Features
- Real-time object detection using ultralytics YOLOv8 (nano model for speed).
- Centroid-based object tracking assigning unique IDs to objects across frames.
- Streamlit web interface with a clean, responsive layout.
- Support for webcam input and video file uploads.
- Interactive controls: confidence threshold slider and multi-class filtering.
- Dynamic statistics showing the count of tracked objects per class.

## Installation

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows use:
   venv\Scripts\activate
   # On macOS/Linux use:
   source venv/bin/activate
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

- **YOLOv8 Model**: The application will automatically download the `yolov8n.pt` weights on the first run.
- **Webcam Permissions**: If using the Webcam source, ensure your browser and terminal have permissions to access the camera.

## Technologies Used
- **Streamlit**: Web interface
- **YOLOv8 (Ultralytics)**: Object detection model
- **OpenCV**: Video stream capture and image drawing
- **SciPy**: Distance calculations for tracking

## License
MIT
