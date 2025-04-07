# helmet_detection.py

import cv2
import torch
from ultralytics import YOLO

model = YOLO(r"C:\Users\raksh\Traffic-Violation-Detection-Systems\helmet_detection\best.pt")  # Update path as needed

def detect_helmet_violations(frame, device="cpu"):
    """
    Detect helmet violations in a frame.
    Returns the frame with bounding boxes and labels.
    """
    results = model(frame)[0]
    annotated_frame = frame.copy()

    motorcyclists = []
    helmets = []
    no_helmets = []

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        if label == "motorcyclist":
            motorcyclists.append((x1, y1, x2, y2))
        if label == "helmet":
            helmets.append((x1, y1, x2, y2))
        elif label == "no_helmet":
            no_helmets.append((x1, y1, x2, y2))

    # Match motorcyclists with helmets or no helmets
    for mx1, my1, mx2, my2 in motorcyclists:
        has_helmet = False
        for hx1, hy1, hx2, hy2 in helmets:
            if abs(mx1 - hx1) < 50 and abs(my1 - hy1) < 50:
                has_helmet = True
                break
        for nx1, ny1, nx2, ny2 in no_helmets:
            if abs(mx1 - nx1) < 50 and abs(my1 - ny1) < 50:
                has_helmet = False
                break

        if has_helmet:
            color = (0, 255, 0)
            label = "No Violation - Safe"
        else:
            color = (0, 0, 255)
            label = "Violation - No Helmet"

        cv2.rectangle(annotated_frame, (mx1, my1), (mx2, my2), color, 2)
        cv2.putText(annotated_frame, label, (mx1, my1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return annotated_frame
