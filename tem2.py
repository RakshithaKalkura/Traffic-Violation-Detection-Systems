import streamlit as st
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os

# Constants
INPUT_SIZE = 320
SKIP_FRAMES = 1
device = "cpu"

# Load models
vehicle_model = YOLO(r"C:\Users\raksh\Traffic-Violation-Detection-Systems\red_light_violation\best.pt").to(device)
helmet_model = YOLO(r"C:\Users\raksh\Traffic-Violation-Detection-Systems\helmet_detection\best.pt").to(device)

def process_frame(frame):
    """
    Process a single frame:
    - Resize to INPUT_SIZE
    - Detect motorbikes
    - Detect helmets on cropped motorbike ROIs
    - Check if bottom of bounding box crosses line_y_resized (red light violation)
    """
    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(rgb, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    tensor = torch.from_numpy(tensor).to(device)

    line_y_resized = int(INPUT_SIZE * 0.8)
    cv2.line(resized, (0, line_y_resized), (INPUT_SIZE, line_y_resized), (255, 0, 0), 2)

    results = vehicle_model(tensor)

    for result in results:
        for box in result.boxes:
            coords = box.xyxy[0]  
            x1, y1, x2, y2 = map(int, coords.tolist())
            conf = box.conf[0].item()
            cls = int(box.cls[0])
            label_vehicle = vehicle_model.names[cls]

            if label_vehicle.lower() != "motorbike":
                continue

            cropped = resized[y1:y2, x1:x2]
            if cropped.size == 0:
                continue

            helmet_results = helmet_model(cropped)[0]
            has_helmet = False
            for hbox in helmet_results.boxes:
                h_cls = int(hbox.cls[0])
                h_label = helmet_model.names[h_cls]
                if h_label.lower() == "with helmet":
                    has_helmet = True
                    break

            # Determine label and color
            center_y = y2
            if center_y > line_y_resized:
                if not has_helmet:
                    label = "Violation: No Helmet + Red Light"
                    color = (0, 0, 255)
                else:
                    label = "Violation: Red Light"
                    color = (0, 165, 255)
            else:
                if not has_helmet:
                    label = "Violation: No Helmet"
                    color = (0, 0, 255)
                else:
                    label = "No Violation: Safe"
                    color = (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(resized, (x1, y1), (x2, y2), color, 2)
            cv2.putText(resized, f"{label} {conf:.2f}", (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return resized


# Streamlit Interface
st.title("Traffic Violation Detection App")
st.markdown("Detects Helmet & Red Light Violations for Motorbikes")

video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_path = os.path.join(tempfile.gettempdir(), "output.avi")
    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'XVID'),
                          fps, (INPUT_SIZE, INPUT_SIZE))

    stframe = st.empty()
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % SKIP_FRAMES == 0:
            processed_frame = process_frame(frame)
            out.write(processed_frame)
            stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                          channels="RGB", use_column_width=True)
        frame_num += 1

    cap.release()
    out.release()
    st.success("Video Processing Completed.")
    st.video(output_path)
