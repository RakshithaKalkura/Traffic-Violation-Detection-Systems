# Traffic Signal Violation Detection System

This repository contains code for detecting traffic signal violations—specifically, vehicles running red lights—using a YOLO-based model. This project is part of an ongoing research effort aimed at improving traffic safety by developing robust, real-time violation detection systems.

## Overview

The system leverages a trained YOLO model to detect vehicles from video streams. It uses a GUI built with Tkinter that allows users to select a video file, processes the frames in real time, and highlights vehicles that cross a designated detection line during a red signal. Violations are indicated with red bounding boxes, while safe vehicles are marked with green boxes.

## Features

- **Real-Time Video Processing:** Detects vehicles from video streams in real time.
- **Violation Detection:** Flags vehicles as "Violation" if they cross a predetermined line when the traffic signal is red.
- **User-Friendly GUI:** Easy-to-use interface for selecting and processing videos.
- **Research-Oriented:** Developed as part of a research project focused on enhancing traffic safety using deep learning and computer vision techniques.

## Installation

Ensure you have Python 3.7+ installed, then install the required dependencies:

```bash
pip install opencv-python Pillow numpy torch ultralytics
```

## Usage
- Model: Place your trained model file (best.pt) in the project directory.
- Run the Application: Execute the following command to start the GUI application:
```bash
python app.py
```
- Process Video: In the GUI, click on "Select Video" to choose your input video file.
- Click on "Start Processing" to run the detection system.
- The application will display the video in a window. A horizontal detection line is drawn at 80% of the frame height. Vehicles whose bounding box's bottom edge crosses this line during a red signal are flagged as "Violation" with a red box; otherwise, they are marked as "Safe" with a green box.

## Demo Video

https://github.com/user-attachments/assets/e992f2ee-63d5-4fe8-9fef-8a985a750ab0


## License

This project is licensed under the [BSD 2-Clause License](./LICENSE).

title: Traffic Violation Detection System
description: YOLOv11-based detection of red light and helmet violations
author: Rakshitha Kalkura
license: MIT

sections:
  - heading: 🚦 Traffic Violation Detection System
    content: |
      This project is a computer vision-based system for detecting:
      - 🚫 Helmet violations (for motorbike riders)
      - 🚨 Red light violations (for all vehicles)

      It uses YOLOv11 models for real-time object detection and integrates:
      - A model for detecting motorbikes and other vehicles
      - A model for detecting helmets (with/without)

  - heading: 📐 System Architecture
    codeblock: |
      ┌──────────────────────────────┐
      │       Input Video Stream     │
      └────────────┬─────────────────┘
                   │
                   ▼
      ┌──────────────────────────────┐
      │ YOLOv11 Vehicle Detection     │
      └────┬──────────────┬───────────┘
           │              │
           │       For motorbike only
           ▼              ▼
      Red light check     Helmet Detection (YOLOv11)
           │              │
           └────┬─────────┘
                ▼
      ▓ Mark as Violation or Safe with bounding box
                ▼
         Annotated Video Frame Output

  - heading: ⚙️ Installation
    steps:
      - Clone the repository:
        command: |
          git clone https://github.com/your-username/traffic-violation-detector.git
          cd traffic-violation-detector
      - Create a virtual environment (optional):
        command: |
          python -m venv venv
          source venv/bin/activate  # or venv\Scripts\activate on Windows
      - Install dependencies:
        command: |
          pip install -r requirements.txt
      - requirements.txt:
        list:
          - torch
          - opencv-python
          - ultralytics==8.0.20
          - numpy

  - heading: 🧠 Models Used
    items:
      - Vehicle Detection:
          - YOLOv11
          - Classes: car, truck, bus, motorbike
          - Used for red light detection
      - Helmet Detection:
          - YOLOv11
          - Classes: with helmet, without helmet
          - Applied only to motorbike detections
      - Model files to place in project directory:
          - vehicle_model.pt
          - helmet_model.pt

  - heading: ▶️ Running the System
    steps:
      - Add your input video in:
        path: ./input/input_video.mp4
      - Run the application:
        command: python app.py
      - Output video will be saved at:
        path: ./output/output_video.mp4
    output_details: |
      Each processed frame contains:
      - Blue line = red light stop line
      - Red bounding box = violation
      - Green box = safe
      - Orange box = red light violation only (non-bike)

  - heading: 📽️ Demo
    content: |
      You can include a sample processed video:
      - demo/processed_demo.mp4
      - Screenshot: demo/sample_frame.png

  - heading: 📌 Notes
    list:
      - Helmet detection is only for motorbike class
      - Red light detection applies to all vehicles
      - The red light line is at 80% of video height

  - heading: 🚀 Future Improvements
    list:
      - Number plate recognition integration
      - Live RTSP camera stream support
      - Jetson Nano edge deployment

  - heading: 👩‍💻 Contributors
    contributors:
      - Rakshitha Kalkura – Research & Development, Model Training, Integration

  - heading: 📄 License
    license_type: MIT
    file: LICENSE
