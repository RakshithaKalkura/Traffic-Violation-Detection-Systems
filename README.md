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
[![Watch the video]](https://github.com/RakshithaKalkura/Traffic-Violation-Detection-Systems/blob/main/traffic_videos/Screen%20Recording%202025-03-10%20214503.mp4)
