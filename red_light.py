import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import tempfile
import os

# Constants
INPUT_SIZE = 320
SKIP_FRAMES = 1
device = "cpu"
st.write(f"Using device: {device}")

# Load models
@st.cache_resource
def load_models():
    vehicle_model = YOLO(r"C:\Users\raksh\Traffic-Violation-Detection-Systems\red_light_violation\best.pt")
    helmet_model = YOLO(r"C:\Users\raksh\Traffic-Violation-Detection-Systems\helmet_detection\best.pt")
    vehicle_model.to(device)
    helmet_model.to(device)
    return vehicle_model, helmet_model

vehicle_model, helmet_model = load_models()

# Frame processor
def process_frame(frame, line_y_original):
    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(rgb, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    tensor = torch.from_numpy(tensor).to(device)

    results = vehicle_model(tensor)

    line_y_resized = int(INPUT_SIZE * 0.8)
    cv2.line(resized, (0, line_y_resized), (INPUT_SIZE, line_y_resized), (255, 0, 0), 2)

    for result in results:
        for box in result.boxes:
            coords = box.xyxy[0]
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            conf = box.conf[0].item()

            if y2 > line_y_resized:
                color = (0, 0, 255)
                label = "Violation"
            else:
                color = (0, 255, 0)
                label = "Safe"
            cv2.rectangle(resized, (x1, y1), (x2, y2), color, 2)
            cv2.putText(resized, f"{label} {conf:.2f}", (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return resized

# Streamlit UI
st.title("🚦 Traffic Violation Detection (Streamlit)")
st.markdown("Detecting red-light and helmet violations using YOLOv11.")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

if uploaded_file is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    video_path = temp_file.name

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps else 60

    stframe = st.empty()
    frame_count = 0

    ret, frame = cap.read()
    if ret:
        height, _, _ = frame.shape
        line_y_original = int(height * 0.9)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    st.success("Processing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % SKIP_FRAMES == 0:
            processed_frame = process_frame(frame, line_y_original)
            stframe.image(processed_frame, channels="BGR", use_column_width=True)

    cap.release()
    os.remove(video_path)
    st.success("✅ Video processing complete.")
