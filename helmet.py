import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from ultralytics import YOLO

# Load YOLOv11 model
model = YOLO(r"C:\Users\raksh\Traffic-Violation-Detection-Systems\helmet_detection\best.pt")  # Update with your YOLOv11 .pt file path

# Define class names based on your trained dataset
# e.g., model.names = {0: 'motorcyclist', 1: 'helmet', 2: 'no_helmet'}
class_names = model.names

st.title("Helmet Violation Detection using YOLOv11")
st.write("Model loaded successfully!")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Could not open video.")
    else:
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)[0]

            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                class_id = int(class_id)
                label = class_names[class_id]

                # Customize display color
                color = (0, 255, 0) if label == "helmet" else (0, 0, 255)

                # Display label and box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            stframe.image(frame, channels="BGR", use_column_width=True)

        cap.release()
        os.remove(video_path)
