import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image, ImageTk


INPUT_SIZE = 320      # Lower resolution for faster processing
SKIP_FRAMES = 1       # Process every nth frame
device = "cpu"
print("Using device:", device)

model = YOLO(r"C:\Users\raksh\Traffic-Violation-Detection-Systems\runs\runs\detect\train\weights\best.pt")
model.to(device)

model2 = YOLO(r"C:\Users\raksh\Traffic-Violation-Detection-Systems\helmet_detection\best.pt")
model2.to(device)

def process_frame(frame, line_y_original):
    """
    Process a frame:
      - Resize to a lower resolution.
      - Preprocess and run YOLO inference.
      - Draw a horizontal detection line.
      - Mark bounding boxes as red (violation) if the bottom of the box crosses the line; else green.
    """
    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(rgb, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    tensor = torch.from_numpy(tensor).to(device)
    
    results = model(tensor)
    
    line_y_resized = int(INPUT_SIZE * 0.8)
    cv2.line(resized, (0, line_y_resized), (INPUT_SIZE, line_y_resized), (255, 0, 0), 2)
    

    for result in results:
        for box in result.boxes:
            coords = box.xyxy[0]  
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            conf = box.conf[0].item()
            
            if y2 > line_y_resized:
                color = (0, 0, 255)  # Red indicates violation
                label = "Violation"
            else:
                color = (0, 255, 0)  # Green indicates safe
                label = "Safe"
            cv2.rectangle(resized, (x1, y1), (x2, y2), color, 2)
            cv2.putText(resized, f"{label} {conf:.2f}", (x1, max(y1-10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return resized

class TrafficViolationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Signal Violation Detector")
        self.video_path = None
        self.cap = None
        self.running = False
        self.frame_count = 0
        self.delay = 60  # Default delay in ms (approx. 30 FPS)
        
        # GUI Elements
        self.video_label = tk.Label(root)
        self.video_label.pack(expand=True, fill="both")
        
        self.select_btn = tk.Button(root, text="Select Video", command=self.select_video)
        self.select_btn.pack(pady=5)
        
        self.start_btn = tk.Button(root, text="Start Processing", command=self.start_video, state=tk.DISABLED)
        self.start_btn.pack(pady=5)
        
        self.stop_btn = tk.Button(root, text="Stop Processing", command=self.stop_video, state=tk.DISABLED)
        self.stop_btn.pack(pady=5)
        
        self.status_label = tk.Label(root, text="Status: Waiting for video selection")
        self.status_label.pack(pady=5)
    
    def select_video(self):
        self.video_path = filedialog.askopenfilename(
            title="Select Video File", 
            filetypes=[("MP4 files", "*.mp4"), ("All Files", "*.*")]
        )
        if self.video_path:
            self.status_label.config(text=f"Selected: {self.video_path}")
            self.start_btn.config(state=tk.NORMAL)
    
    def start_video(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file first.")
            return
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open video.")
            return
        
        # Try to set delay based on video FPS if available
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps and fps > 0:
            self.delay = int(1000 / fps)
        else:
            self.delay = 60  # Default ~30 FPS
        
        self.running = True
        self.frame_count = 0
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
    
        ret, frame = self.cap.read()
        if ret:
            height, _, _ = frame.shape
            # Detection line at 90% of the original height
            self.line_y_original = int(height * 0.9)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to start
        
        self.update_frame()
    
    def update_frame(self):
        if not self.running or self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.stop_video()
            return
        
        self.frame_count += 1
        # Process only every SKIP_FRAMES-th frame
        if self.frame_count % SKIP_FRAMES == 0:
            processed_frame = process_frame(frame, self.line_y_original)
            cv2image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        self.root.after(self.delay, self.update_frame)
    
    def stop_video(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Video processing stopped.")

if __name__ == "__main__":
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    half_width = screen_width // 2
    half_height = screen_height // 2
    root.geometry(f"{half_width}x{half_height}")
    
    app = TrafficViolationApp(root)
    root.mainloop()
