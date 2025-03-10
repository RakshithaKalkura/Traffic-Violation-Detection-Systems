import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import cv2
import torch
import numpy as np
from PIL import Image, ImageTk

traffic_light_state = "red"

# Load the TorchScript model (ensure the model file is in the specified path)
model = torch.jit.load("model/best_scripted.pt")
model.eval()

def preprocess(image, img_size=640):
    """
    Preprocess an image for the model.
    - Resize to img_size x img_size
    - Convert from BGR to RGB, normalize to [0, 1]
    - Rearrange dimensions to (1, 3, H, W)
    """
    img = cv2.resize(image, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return torch.from_numpy(img)

def postprocess(outputs, conf_threshold=0.5):
    """
    Convert raw model outputs into a list of detections.
    This dummy postprocessing assumes each prediction is:
    [x1, y1, x2, y2, confidence, class]
    Adjust this function to match your model's output format.
    """
    detections = []
    # Assume outputs is a tensor of shape (batch, num_preds, 6)
    outputs = outputs[0]  # Remove batch dimension if needed
    for pred in outputs:
        conf = pred[4].item()
        if conf > conf_threshold:
            x1, y1, x2, y2, _, cls = pred.tolist()
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": conf,
                "class": int(cls)
            })
    return detections

def process_video(video_path, output_path):
    """
    Process the input video:
      - For each frame, run inference,
      - Draw a horizontal detection line,
      - For each detection, check if the bottom of the bounding box (y2)
        crosses the detection line.
      - If the traffic light state is red and the vehicle crosses the line,
        draw a red box; otherwise, draw a green box.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get video properties and prepare VideoWriter
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Define a horizontal detection line (for example, at 80% of frame height)
    line_y = int(height * 0.8)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw the detection line on the frame (blue color)
        cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 0), 2)

        # Preprocess the frame and run inference
        input_tensor = preprocess(frame)
        with torch.no_grad():
            outputs = model(input_tensor)
        detections = postprocess(outputs)

        # For each detection, decide box color based on crossing and traffic light state
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            # Compute bottom-center of the bounding box
            x_center = (x1 + x2) // 2
            y_bottom = y2  # using the bottom y coordinate

            # Check if the vehicle has crossed the line
            has_crossed = y_bottom > line_y

            # Determine color: red for violation (if red light and crossed), else green.
            if traffic_light_state == "red" and has_crossed:
                color = (0, 0, 255)  # Red in BGR
                label = "Violation"
            else:
                color = (0, 255, 0)  # Green in BGR
                label = "Safe"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {det['confidence']:.2f}", (x1, max(y1 - 10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)
        # Optionally, show the frame (press 'q' to exit early)
        cv2.imshow("Traffic Violation Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# GUI Application Class
class App:
    def __init__(self, root):
        self.root = root
        root.title("Traffic Light Violation Detector")
        self.video_path = None

        # Label to show the current traffic light state
        self.state_label = tk.Label(root, text=f"Traffic Light State: {traffic_light_state.upper()}")
        self.state_label.pack(pady=5)

        # Button to toggle traffic light state
        self.toggle_button = tk.Button(root, text="Toggle Traffic Light", command=self.toggle_state)
        self.toggle_button.pack(pady=10)

        self.select_button = tk.Button(root, text="Select Video", command=self.select_video)
        self.select_button.pack(pady=10)

        self.process_button = tk.Button(root, text="Process Video", command=self.process_video_thread, state=tk.DISABLED)
        self.process_button.pack(pady=10)

        self.status_label = tk.Label(root, text="Status: Waiting for video selection")
        self.status_label.pack(pady=10)

    def toggle_state(self):
        # Toggle the global traffic light state between "red" and "green"
        global traffic_light_state
        traffic_light_state = "green" if traffic_light_state == "red" else "red"
        self.state_label.config(text=f"Traffic Light State: {traffic_light_state.upper()}")

    def select_video(self):
        self.video_path = filedialog.askopenfilename(title="Select Video", 
                                                     filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))
        if self.video_path:
            self.status_label.config(text=f"Selected: {self.video_path}")
            self.process_button.config(state=tk.NORMAL)

    def process_video_thread(self):
        if not self.video_path:
            messagebox.showerror("Error", "No video selected!")
            return
        self.process_button.config(state=tk.DISABLED)
        self.status_label.config(text="Processing video...")
        output_path = self.video_path.rsplit('.', 1)[0] + "_output.mp4"
        threading.Thread(target=self.run_processing, args=(self.video_path, output_path), daemon=True).start()

    def run_processing(self, video_path, output_path):
        process_video(video_path, output_path)
        self.status_label.config(text=f"Processing complete! Output saved to: {output_path}")
        self.process_button.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
