# Installation Guide

This guide explains how to set up and run SafeStep on Raspberry Pi OS.

## 1. Install Raspberry Pi OS

Flash Raspberry Pi OS using Raspberry Pi Imager and complete the first boot setup. Raspberry Pi OS 64-bit is recommended.

## 2. Update the System

```bash
sudo apt update
sudo apt upgrade -y
```

## 3. Enable the Camera

Open Raspberry Pi configuration:

```bash
sudo raspi-config
```

Then enable the camera interface if required by the OS version, reboot the device, and verify that the camera is detected.

## 4. Install System Packages

```bash
sudo apt install -y python3-pip python3-picamera2 espeak-ng
sudo apt install -y libatlas-base-dev libjpeg-dev libtiff5-dev
```

## 5. Install Python Packages

From the project folder, run:

```bash
pip install -r requirements.txt
```

On some Raspberry Pi OS installations, OpenCV and Picamera2 work better when installed through `apt` instead of `pip`. If there is an OpenCV or camera issue, install these packages:

```bash
sudo apt install -y python3-opencv python3-picamera2
```

## 6. Test Required Modules

```bash
python3 -c "import cv2; print('OpenCV OK')"
python3 -c "from ultralytics import YOLO; print('YOLO OK')"
python3 -c "from picamera2 import Picamera2; print('Picamera2 OK')"
```

## 7. Test Voice Output

```bash
espeak-ng "SafeStep voice test"
```

## 8. Run the Project

```bash
python3 safestep_final_system.py
```

Run without display preview:

```bash
python3 safestep_final_system.py --no-preview
```

## 9. Useful Runtime Options

```bash
python3 safestep_final_system.py --frame-skip 4 --voice-interval 2.5
```

Options:

- `--width` camera frame width
- `--height` camera frame height
- `--frame-skip` controls how often frames are processed
- `--voice-interval` controls voice message frequency
- `--no-preview` disables the OpenCV display window

## 10. Common Issues

### Picamera2 is not installed

Install Picamera2:

```bash
sudo apt install python3-picamera2
```

### No voice output

Check that audio output is connected and test:

```bash
espeak-ng "test"
```

### Model file not found

Make sure these folders exist:

```text
models/ssd/
models/ncnn/best_ncnn_model/
data/
```

### Low FPS

Try increasing `--frame-skip` or lowering camera resolution.
