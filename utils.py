import cv2
import numpy as np

def generate_colors(num_colors):
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_colors, 3), dtype=int)
    return [tuple(map(int, color)) for color in colors]

colors = generate_colors(100)

def draw_boxes_and_tracks(frame, detections, tracks, classes):
    """
    frame: image
    detections: list of dicts {'box': [x1, y1, x2, y2], 'class_id': int, 'conf': float}
    tracks: list of dicts {'id': int, 'box': [x1, y1, x2, y2], 'class_id': int, 'centroid': (x, y)}
    classes: dict mapping class_id to class name
    """
    for track in tracks:
        track_id = track['id']
        x1, y1, x2, y2 = map(int, track['box'])
        class_id = track['class_id']
        class_name = classes.get(class_id, "Unknown")
        color = colors[track_id % len(colors)]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"ID: {track_id} {class_name}"
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        c2 = x1 + t_size[0], y1 - t_size[1] - 3
        cv2.rectangle(frame, (x1, y1), c2, color, -1, cv2.LINE_AA)
        cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1, cv2.LINE_AA)
        
    return frame
