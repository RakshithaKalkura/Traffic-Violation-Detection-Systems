from ultralytics import YOLO
import torch
import cv2

red_model = YOLO(r"C:\Users\raksh\Traffic-Violation-Detection-Systems\runs\runs\detect\train\weights\best.pt")
red_model.to("cpu")

def detect_red_violation(frame, line_y):
    results = red_model(frame)
    violations = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if y2 > line_y:
                violations.append({
                    'box': (x1, y1, x2, y2),
                    'confidence': conf
                })

    return violations
