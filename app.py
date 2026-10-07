import streamlit as st
import cv2
import tempfile
import time
from detector import YOLODetector
from tracker import CentroidTracker
from utils import draw_boxes_and_tracks
import os

st.set_page_config(page_title="Object Detection & Tracking", layout="wide")

st.title("Object Detection and Tracking")
st.markdown("Real-time object detection and tracking using YOLOv8 and Centroid Tracking.")

@st.cache_resource
def load_detector():
    return YOLODetector("yolov8n.pt")

detector = load_detector()

st.sidebar.title("Settings")

source_type = st.sidebar.radio("Select Source", ["Webcam", "Video File"])

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

all_classes = detector.classes
class_names = list(all_classes.values())

detect_all = st.sidebar.checkbox("Detect All Classes", value=False)
if detect_all:
    class_filters = None
    st.sidebar.info("Detecting all 80 COCO classes.")
else:
    selected_class_names = st.sidebar.multiselect(
        "Select Classes to Detect", 
        class_names, 
        default=["person"],
        help="Select classes to detect. Check 'Detect All Classes' above to detect everything."
    )
    class_filters = [k for k, v in all_classes.items() if v in selected_class_names] if selected_class_names else None

loop_video = st.sidebar.checkbox("Loop Video", value=True) if source_type == "Video File" else False
run_app = st.sidebar.checkbox("Start/Stop")

video_path = None
if source_type == "Video File":
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        video_path = tfile.name

FRAME_WINDOW = st.image([])
stat_container = st.empty()

if run_app:
    if source_type == "Webcam":
        cap = cv2.VideoCapture(0)
    else:
        if video_path is None:
            st.error("Please upload a video file first.")
            st.stop()
        cap = cv2.VideoCapture(video_path)

    tracker = CentroidTracker(max_disappeared=30, max_distance=90)
    
    while cap.isOpened() and run_app:
        ret, frame = cap.read()
        if not ret:
            if source_type == "Video File":
                if loop_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                st.warning("End of video stream.")
            else:
                st.warning("Cannot read frame from webcam.")
            break
        
        # Convert frame from BGR (OpenCV) to RGB (Streamlit display)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect objects using BGR frame (native OpenCV format expected by Ultralytics YOLO)
        detections = detector.detect(frame, conf_threshold=conf_threshold, class_filters=class_filters)
        
        # Track objects
        tracks = tracker.update(detections)
        
        # Draw bounding boxes and tracks
        output_frame = draw_boxes_and_tracks(frame_rgb, detections, tracks, detector.classes)
        
        # Display frame
        FRAME_WINDOW.image(output_frame, channels="RGB")
        
        # Calculate statistics
        counts = {}
        for track in tracks:
            cls_name = detector.classes.get(track['class_id'], "Unknown")
            counts[cls_name] = counts.get(cls_name, 0) + 1
            
        stats_md = "### Current Detections:\n"
        if not counts:
            stats_md += "No objects detected."
        for k, v in counts.items():
            stats_md += f"- **{k}**: {v}\n"
        stat_container.markdown(stats_md)
        
        # Small delay to reduce CPU usage
        time.sleep(0.01)
        
    cap.release()
    
    # Cleanup temporary video file
    if video_path is not None and os.path.exists(video_path):
        try:
            os.remove(video_path)
        except Exception:
            pass
