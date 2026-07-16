# SafeStep Smart Headband

SafeStep is an ML-driven IoT assistive system designed to support visually impaired users by detecting nearby objects in real time and providing voice-based navigation guidance.

The project uses a Raspberry Pi camera feed, an SSD MobileNet model for general object detection, a custom YOLOv8 NCNN model for door/stairs/window detection, and eSpeak NG for spoken feedback.

## Main Features

- Real-time camera-based object detection
- General object detection using SSD MobileNet
- Custom detection for doors, stairs, and windows using YOLOv8 NCNN
- Direction awareness: left, center, and right
- Proximity-based guidance: detected, ahead, and very close
- Voice feedback using eSpeak NG
- Offline execution on Raspberry Pi

## Technology Stack

- Python
- OpenCV DNN
- Ultralytics YOLO
- NCNN model format
- Picamera2
- Raspberry Pi
- eSpeak NG

## Project Structure

```text
SafeStep-Smart-Headband/
├── data/
│   └── coco.names
├── docs/
│   ├── INSTALLATION.md
│   ├── MODEL_FILES.md
│   ├── PROJECT_STRUCTURE.md
│   └── USER_GUIDE.md
├── models/
│   ├── ncnn/
│   │   └── best_ncnn_model/
│   │       ├── metadata.yaml
│   │       ├── model.ncnn.bin
│   │       ├── model.ncnn.param
│   │       └── model_ncnn.py
│   └── ssd/
│       ├── frozen_inference_graph.pb
│       └── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
├── safestep_final_system.py
├── requirements.txt
├── run_safestep.sh
├── run_safestep.bat
├── README.md
└── .gitignore
```

## Requirements

Hardware:

- Raspberry Pi 4 or compatible Raspberry Pi board
- Raspberry Pi Camera Module
- Speaker, wired earphones, Bluetooth earbuds, or other audio output
- Power supply

Software:

- Raspberry Pi OS 64-bit recommended
- Python 3.9 or newer
- OpenCV
- Picamera2
- Ultralytics YOLO
- eSpeak NG

## Quick Start

```bash
sudo apt update
sudo apt install -y python3-pip python3-picamera2 espeak-ng
pip install -r requirements.txt
python3 safestep_final_system.py
```

Press `q` in the preview window to stop the system.

Detailed setup instructions are available in [`docs/INSTALLATION.md`](docs/INSTALLATION.md).

## Running Without Preview

For headless use, run:

```bash
python3 safestep_final_system.py --no-preview
```

## Important Notes

- This project is designed for Raspberry Pi with Picamera2.
- The project uses relative paths, so it can be placed in any folder without editing hardcoded file locations.
- Model files are included in the repository because they are required to run the system.
- Performance depends on Raspberry Pi hardware, lighting, and camera position.

## Academic Context

SafeStep was developed as a final year academic project focused on combining computer vision, embedded AI, and IoT-based assistive technology.

