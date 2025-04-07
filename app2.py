import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
from helmet_detection import detect_helmet
from red_light import detect_red_violation

INPUT_WIDTH = 640
INPUT_HEIGHT = 480

class TrafficViolationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Violation Detector")
        self.root.geometry("960x600")

        self.video_path = None
        self.cap = None
        self.running = False
        self.line_y = int(INPUT_HEIGHT * 0.8)

        self.video_label = tk.Label(root)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self.select_btn = tk.Button(root, text="Select Video", command=self.select_video)
        self.select_btn.pack(pady=5)

        self.start_btn = tk.Button(root, text="Start Processing", command=self.start_video, state=tk.DISABLED)
        self.start_btn.pack(pady=5)

        self.stop_btn = tk.Button(root, text="Stop Processing", command=self.stop_video, state=tk.DISABLED)
        self.stop_btn.pack(pady=5)

        self.status_label = tk.Label(root, text="Status: Waiting")
        self.status_label.pack()

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if self.video_path:
            self.status_label.config(text=f"Selected: {self.video_path}")
            self.start_btn.config(state=tk.NORMAL)

    def start_video(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open video.")
            return

        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.update_frame()

    def stop_video(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.config(image='')
        self.status_label.config(text="Stopped")

    def update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_video()
            return

        frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT))
        red_violations = detect_red_violation(frame, self.line_y)
        helmet_detections = detect_helmet(frame)

        for detection in helmet_detections:
            x1, y1, x2, y2 = detection['box']
            label = detection['label']
            if label == 'no_helmet':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Violation - No Helmet", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif label in ['helmet', 'motorcyclist']:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "No Violation - Safe", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for v in red_violations:
            x1, y1, x2, y2 = v['box']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "Violation - Red Light", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw line
        cv2.line(frame, (0, self.line_y), (INPUT_WIDTH, self.line_y), (255, 255, 0), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.imgtk = img
        self.video_label.configure(image=img)

        self.root.after(30, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficViolationApp(root)
    root.mainloop()
