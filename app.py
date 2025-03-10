import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageTk

model = YOLO(r"C:\Users\raksh\Traffic-Violation-Detection-Systems\runs\runs\detect\train\weights\best.pt")

def process_frame(frame, line_y):
    """
    Process a single frame:
      - Run YOLO detection on the frame.
      - Draw a horizontal detection line.
      - For each detection, if the bottom of the bounding box (y2) is below the line,
        mark it as a violation (red box), else mark it as safe (green box).
    """
    results = model(frame)
    height, width, _ = frame.shape
    cv2.line(frame, (0, line_y), (width, line_y), (255, 0, 0), 2)

    for result in results:
        for box in result.boxes:
            coords = box.xyxy[0]
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            conf = box.conf[0].item()
            if y2 > line_y:
                color = (0, 0, 255)  
                label = "Violation"
            else:
                color = (0, 255, 0) 
                label = "Safe"
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
    
        self.video_label = tk.Label(root)
        self.video_label.pack()
        
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
        
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        
        ret, frame = self.cap.read()
        if ret:
            height, width, _ = frame.shape
            self.line_y = int(height * 0.8)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
        threading.Thread(target=self.video_loop, daemon=True).start()
    
    def video_loop(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            processed_frame = process_frame(frame, self.line_y)
            
            cv2image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
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
        self.status_label.config(text="Status: Video processing stopped.")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficViolationGUI(root)
    root.mainloop()
