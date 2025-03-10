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

