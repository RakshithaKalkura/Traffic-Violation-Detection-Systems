import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageTk

# Load your trained YOLO model using the ultralytics wrapper
model = YOLO("best.pt")

def process_frame(frame, line_y):
    """
    Process a single frame:
      - Run YOLO detection on the frame.
      - Draw a horizontal detection line.
      - For each detection, if the bottom of the bounding box (y2) is below the line,
        mark it as a violation (red box), else mark it as safe (green box).
    """
    # Run inference using the YOLO model
    results = model(frame)
    
    # Draw the detection line (blue) on the frame
    height, width, _ = frame.shape
    cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 0), 2)
    
    # Process each detection result
    for result in results:
        # Loop through all detected boxes in the result
        for box in result.boxes:
            # Get bounding box coordinates (x1, y1, x2, y2)
            coords = box.xyxy[0]  # tensor with coordinates
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            conf = box.conf[0].item()  # confidence score
            
            # Decide color: red if vehicle crosses the line, green otherwise
            if y2 > line_y:
                color = (0, 0, 255)  # Red for violation
                label = "Violation"
            else:
                color = (0, 255, 0)  # Green for safe
                label = "Safe"
            
            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

class TrafficViolationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Signal Violation Detector")
        self.video_path = None
        self.cap = None
        self.running = False
        
        # Label to display video frames
        self.video_label = tk.Label(root)
        self.video_label.pack()
        
        # Button to select a video file
        self.select_btn = tk.Button(root, text="Select Video", command=self.select_video)
        self.select_btn.pack(pady=5)
        
        # Button to start processing the video
        self.start_btn = tk.Button(root, text="Start Processing", command=self.start_video, state=tk.DISABLED)
        self.start_btn.pack(pady=5)
        
        # Button to stop processing
        self.stop_btn = tk.Button(root, text="Stop Processing", command=self.stop_video, state=tk.DISABLED)
        self.stop_btn.pack(pady=5)
        
        # Status label
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
        
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        # Read the first frame to set the detection line at 80% of the frame height
        ret, frame = self.cap.read()
        if ret:
            height, width, _ = frame.shape
            self.line_y = int(height * 0.8)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset to start
        # Start video processing in a separate thread
        threading.Thread(target=self.video_loop, daemon=True).start()
    
    def video_loop(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            processed_frame = process_frame(frame, self.line_y)
            
            # Convert frame to RGB and then to a PIL Image for Tkinter display
            cv2image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
            
            # Wait based on the video's FPS (default to 30 if unavailable)
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            delay = int(1000 / fps)
            self.root.after(delay)
        self.stop_video()
    
    def stop_video(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text
