# 🚦 Traffic Violation Detection System

A computer vision-based system that uses YOLOv11 to automatically detect and flag traffic violations in video feeds.

## 🔍 Key Features

- 🚫 **Helmet violation detection** for motorbike riders
- 🚨 **Red light violation detection** for all vehicles
- ⚡ Real-time processing with YOLOv11 models
- 📊 Visual classification with color-coded violation indicators

## 📐 System Architecture

```
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
```

## 📽️ Demo Video
 
[Demo.webm](https://github.com/user-attachments/assets/c410c7be-39ae-4e7f-8bcf-f5ba67fd214c)


## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/traffic-violation-detector.git
cd traffic-violation-detector
```

### 2. Install Python Dependencies

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```


> ✅ Make sure you're using a system with GPU support for PyTorch for better performance.

## 🧠 Models Used

### ✅ 1. Vehicle Detection (YOLOv11)
- **Classes**: car, truck, bus, motorbike, etc.
- **Purpose**: Detecting vehicles and checking red light crossing.

### ✅ 2. Helmet Detection (YOLOv11)
- **Classes**: with helmet, without helmet
- **Purpose**: Used only when the vehicle class is motorbike.

Both models must be placed in the project directory as:
- `best.pt`
Update File path if required.

You can train your own models or download pre-trained weights compatible with YOLOv11 format.

## ▶️ Running the System

### 1. Run the script
```bash
python app_st.py
```

### 2. Upload the video
The processed video with bounding boxes and violation labels will be saved as:
```bash
./output_video.mp4
```

Each frame will contain:
- A blue line marking the red light boundary
- Bounding boxes:
  - 🟥 Red for red light or helmet violations
  - 🟩 Green for compliant vehicles
  - 🟧 Orange for red light violations only (non-bike)

## 📌 How It Works

1. **Vehicle Detection**: The system first identifies all vehicles in the frame using the YOLOv11 vehicle detection model.
   
2. **Violation Detection**:
   - **Red Light Violations**: All vehicles are checked against a virtual red light boundary (positioned at 80% of the frame height). If a vehicle's bounding box crosses this line during a red light phase, it's marked as a violation.
   
   - **Helmet Violations**: For motorbikes only, a second YOLOv11 model analyzes riders to detect if they're wearing helmets. Riders without helmets are flagged as violations.

3. **Visual Output**: The system generates an annotated video marking violations with color-coded bounding boxes for easy identification.

## 📋 Notes

- Only motorbike vehicles are checked for helmet violations
- All vehicles are evaluated for red light crossing based on their bottom bounding box position
- The red light line is calculated as 80% height of the video frame
- Processing speed depends on your hardware configuration - GPU recommended for real-time analysis

## 👩‍💻 Contributors

- Rakshitha Kalkura – Research & Development, Model Training, Integration

## 🌐 Acknowledgement

This project was conducted under the supervision of Prof. M Prabhu and Anisha ma'am of CVLA group of NIT Calicut. Eternally grateful for all the guidance and mentorship.

## License

This project is licensed under the [BSD 2-Clause License](./LICENSE).

