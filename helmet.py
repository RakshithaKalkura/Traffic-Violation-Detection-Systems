import streamlit as st
import cv2
import os
import tempfile
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Load YOLOv11 model
model = YOLO(r"C:\Users\raksh\Traffic-Violation-Detection-Systems\helmet_detection\best.pt")  # 🔁 Replace with your trained model path
st.success("YOLOv11 Helmet Detection Model Loaded Successfully ✅")

# Class mapping from your dataset
CLASS_NAMES = model.names  # Assumes: helmet, motorcyclist, no_helmet

st.title("🪖 Helmet Violation Detection using YOLOv11")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)

    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv11
        results = model(frame)[0]

        # Store detected bounding boxes
        motorcyclists, helmets, no_helmets = [], [], []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = CLASS_NAMES[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            if label == "motorcyclist":
                motorcyclists.append((x1, y1, x2, y2))
            elif label == "helmet":
                helmets.append((x1, y1, x2, y2))
            elif label == "no_helmet":
                no_helmets.append((x1, y1, x2, y2))

        # Analyze for violations
        for mx1, my1, mx2, my2 in motorcyclists:
            has_helmet = False
            for hx1, hy1, hx2, hy2 in helmets:
                if abs(mx1 - hx1) < 50 and abs(my1 - hy1) < 50:
                    has_helmet = True
                    break
            for nhx1, nhy1, nhx2, nhy2 in no_helmets:
                if abs(mx1 - nhx1) < 50 and abs(my1 - nhy1) < 50:
                    has_helmet = False
                    break

            if has_helmet:
                label = "No Violation - Safe"
                color = (0, 255, 0)
            else:
                label = "Violation - No Helmet"
                color = (0, 0, 255)

            cv2.rectangle(frame, (mx1, my1), (mx2, my2), color, 2)
            cv2.putText(frame, label, (mx1, my1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Resize for Streamlit
        frame = cv2.resize(frame, (800, 500))
        stframe.image(frame, channels="BGR", use_column_width=True)

    cap.release()
