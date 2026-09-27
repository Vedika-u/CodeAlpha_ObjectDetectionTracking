from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)
        self.classes = self.model.names
        
    def detect(self, frame, conf_threshold=0.5, class_filters=None):
        results = self.model(frame, conf=conf_threshold, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if class_filters and cls_id not in class_filters:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    'box': [x1, y1, x2, y2],
                    'class_id': cls_id,
                    'conf': conf
                })
        return detections
