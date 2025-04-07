# app.py

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from helmet_detection.helmet_detection import detect_helmet_violations

class HelmetDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Helmet Violation Detection")
        self.root.geometry("900x600")
        
        self.video_label = tk.Label(root)
        self.video_label.pack(pady=10, fill="both", expand=True)
        
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(pady=10)

        self.select_btn = tk.Button(self.btn_frame, text="Select Video", command=self.select_video)
        self.select_btn.pack(side="left", padx=5)

        self.start_btn = tk.Button(self.btn_frame, text="Start Processing", command=self.start_video, state=tk.DISABLED)
        self.start_btn.pack(side="left", padx=5)

        self.stop_btn = tk.Button(self.btn_frame, text="Stop", command=self.stop_video, state=tk.DISABLED)
        self.stop_btn.pack(side="left", padx=5)

        self.status_label = tk.Label(root, text="Status: Waiting for video selection")
        self.status_label.pack(pady=5)

        self.cap = None
        self.running = False
        self.delay = 33  # ~30 FPS

    def select_video(self):
        self.video_path = filedialog.askopenfilename(title="Select Video", filetypes=[("MP4 files", "*.mp4")])
        if self.video_path:
            self.status_label.config(text=f"Selected: {self.video_path}")
            self.start_btn.config(state=tk.NORMAL)

    def start_video(self):
        if not self.video_path:
            messagebox.showerror("Error", "Please select a video file first.")
            return

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open video file.")
            return

        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.update_frame()

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_video()
            return

        annotated_frame = detect_helmet_violations(frame)
        display_frame = cv2.resize(annotated_frame, (800, 500))
        img = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(img))
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(self.delay, self.update_frame)

    def stop_video(self):
        self.running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        if self.cap:
            self.cap.release()
        self.status_label.config(text="Status: Video stopped.")

if __name__ == "__main__":
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    half_width = screen_width // 2
    half_height = screen_height // 2
    root.geometry(f"{half_width}x{half_height}")
    
    app = HelmetDetectionApp(root)
    root.mainloop()

