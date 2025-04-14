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
    orig_h, orig_w = frame.shape[:2]

    # Resize frame to 640x640 for YOLO
    resized = cv2.resize(frame, (640, 640))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(rgb, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    tensor = torch.from_numpy(tensor).to(device)

    # Line position in original frame
    line_y = int(orig_h * 0.8)
    cv2.line(frame, (0, line_y), (orig_w, line_y), (255, 0, 0), 2)

    results = vehicle_model(tensor)

    for result in results:
        for box in result.boxes:
            coords = box.xyxy[0]  
            x1, y1, x2, y2 = coords.tolist()

            # Scale back to original resolution
            x1 = int(x1 * orig_w / 640)
            x2 = int(x2 * orig_w / 640)
            y1 = int(y1 * orig_h / 640)
            y2 = int(y2 * orig_h / 640)

            conf = box.conf[0].item()
            cls = int(box.cls[0])
            label_vehicle = vehicle_model.names[cls].lower()

            helmet_violation = False
            red_light_violation = y2 > line_y

            # Helmet check only for motorbikes
            if label_vehicle == "motorbike":
                cropped = frame[y1:y2, x1:x2]
                if cropped.size != 0:
                    helmet_results = helmet_model(cropped)[0]
                    has_helmet = False
                    for hbox in helmet_results.boxes:
                        h_cls = int(hbox.cls[0])
                        h_label = helmet_model.names[h_cls].lower()
                        if h_label == "with helmet":
                            has_helmet = True
                            break
                    helmet_violation = not has_helmet

            # Determine label and color
            if helmet_violation and red_light_violation:
                label = "Violation: No Helmet + Red Light"
                color = (0, 0, 255)
            elif helmet_violation:
                label = "Violation: No Helmet"
                color = (0, 0, 255)
            elif red_light_violation:
                label = "Violation: Red Light"
                color = (0, 165, 255)
            else:
                label = "No Violation: Safe"
                color = (0, 255, 0)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


# Streamlit UI
st.title("Traffic Violation Detection App")
st.markdown("🚦 Red Light (All Vehicles) & 🪖 Helmet (Only Motorbikes)")

video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(tempfile.gettempdir(), "output.avi")
    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'XVID'),
                          fps, (frame_w, frame_h))

    stframe = st.empty()
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % SKIP_FRAMES == 0:
            processed_frame = process_frame(frame.copy())
            out.write(processed_frame)
            stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                          channels="RGB", use_column_width=True)
        frame_num += 1

    cap.release()
    out.release()
    st.success("✅ Video Processing Completed.")
    st.video(output_path)

    with open(output_path, "rb") as f:
        st.download_button("Download Processed Video", f, file_name="violations_output.avi")