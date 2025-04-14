import streamlit as st
import tempfile
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
import sys
import time

# Load models (set your paths here)
vehicle_model_path = r"C:\Users\raksh\Traffic-Violation-Detection-Systems\red_light_violation\best.pt"
helmet_model_path = r"C:\Users\raksh\Traffic-Violation-Detection-Systems\helmet_detection\best.pt"
vehicle_model = YOLO(vehicle_model_path)
helmet_model = YOLO(helmet_model_path)

# Function to process single frame
def process_frame(frame, device="cpu"):
    motorbike_class_name = "motorbike"  
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Step 1: Vehicle Detection
    results = vehicle_model(frame_rgb)[0]
    boxes = results.boxes
    annotated_frame = frame.copy()

    for box in boxes:
        cls = int(box.cls[0].item())
        label = vehicle_model.names[cls]
        if label.lower() != "motorbike":
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        motorbike_crop = frame[y1:y2, x1:x2]

        # Step 2: Helmet Detection on cropped motorbike
        if motorbike_crop.size == 0:
            continue
        helmet_results = helmet_model(motorbike_crop)[0]

        has_helmet = False
        for h_box in helmet_results.boxes:
            h_cls = int(h_box.cls[0].item())
            h_label = helmet_model.names[h_cls]
            if h_label.lower() == "with helmet":
                has_helmet = True
                break

        color = (0, 255, 0) if has_helmet else (0, 0, 255)
        status_label = "Safe: Helmet" if has_helmet else "Violation: No Helmet"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, status_label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return annotated_frame

# Streamlit UI
st.title("Traffic Violation Detection System -- Red light Violation and Helmet Detection")  
uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_video:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_video.read())
    temp_file_path = temp_file.name

    stframe = st.empty()
    cap = cv2.VideoCapture(temp_file_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed = process_frame(frame)
        frame_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

    cap.release()
    st.success("Video processing complete.")
