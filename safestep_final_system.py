"""
SafeStep Smart Headband - Real-Time Assistive Object Detection System

This script runs the SafeStep detection pipeline on Raspberry Pi:
1. Captures frames from Raspberry Pi Camera using Picamera2.
2. Detects general COCO objects using SSD MobileNet.
3. Detects door, stairs, and window using a custom YOLOv8 NCNN model.
4. Generates voice guidance with eSpeak NG.

Press "q" in the preview window to stop the system.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
from ultralytics import YOLO

try:
    from picamera2 import Picamera2
except ImportError as exc:  # Helpful message when opened on a non-Raspberry Pi computer.
    raise SystemExit(
        "Picamera2 is not installed. Run this project on Raspberry Pi OS and install "
        "it using: sudo apt install python3-picamera2"
    ) from exc

# Helps avoid Qt font warnings on Raspberry Pi OS.
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SSD_DIR = BASE_DIR / "models" / "ssd"
NCNN_MODEL_DIR = BASE_DIR / "models" / "ncnn" / "best_ncnn_model"

COCO_NAMES_PATH = DATA_DIR / "coco.names"
SSD_CONFIG_PATH = SSD_DIR / "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
SSD_WEIGHTS_PATH = SSD_DIR / "frozen_inference_graph.pb"

ALLOWED_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "train",
    "traffic light", "bench", "cat", "dog", "cow",
    "backpack", "umbrella", "handbag", "suitcase",
    "bottle", "cup", "fork", "knife", "spoon",
    "banana", "apple", "sandwich", "orange", "pizza", "cake",
    "chair", "couch", "potted plant", "bed", "dining table",
    "laptop", "mouse", "keyboard", "cell phone",
    "book", "clock", "scissors",
}

OBSTACLE_CLASSES = {
    "person", "chair", "couch", "bed", "dining table",
    "bicycle", "car", "motorcycle", "bus", "dog", "cat", "cow",
    "door", "stairs",
}

DetectionInfo = Tuple[str, str, str, bool]


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def load_class_names(path: Path) -> List[str]:
    require_file(path, "COCO class names file")
    return path.read_text(encoding="utf-8").strip().splitlines()


def load_ssd_model() -> cv2.dnn_DetectionModel:
    require_file(SSD_WEIGHTS_PATH, "SSD model weights")
    require_file(SSD_CONFIG_PATH, "SSD model config")

    net = cv2.dnn_DetectionModel(str(SSD_WEIGHTS_PATH), str(SSD_CONFIG_PATH))
    net.setInputSize(256, 256)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    return net


def get_proximity(area: int, box: Sequence[int], frame_shape: Tuple[int, int, int], class_name: str) -> Tuple[str, bool]:
    h, w = frame_shape[:2]
    x, y, bw, bh = box

    area_ratio = area / (w * h)
    bottom_touch = (y + bh) > (h * 0.75)
    large_objects = {"door", "bus", "car", "train"}

    if class_name in large_objects:
        if area_ratio > 0.35 and bottom_touch:
            return "very close", True
        if area_ratio > 0.2:
            return "ahead", False
    else:
        if area_ratio > 0.15 and bottom_touch:
            return "very close", True
        if area_ratio > 0.05:
            return "ahead", False

    return "detected", False


def get_direction(x_center: int, frame_width: int) -> str:
    if x_center < frame_width / 3:
        return "left"
    if x_center > 2 * frame_width / 3:
        return "right"
    return "center"


def get_color(direction: str, is_close: bool) -> Tuple[int, int, int]:
    if is_close:
        return 0, 0, 255
    if direction == "left":
        return 0, 255, 255
    if direction == "center":
        return 255, 0, 0
    return 0, 255, 0


def detect_ssd_objects(
    img,
    net: cv2.dnn_DetectionModel,
    class_names: Sequence[str],
    threshold: float,
    nms_threshold: float,
) -> Tuple[object, List[DetectionInfo]]:
    class_ids, confidences, boxes = net.detect(img, confThreshold=threshold, nmsThreshold=nms_threshold)
    object_info: List[DetectionInfo] = []

    if class_ids is None:
        return img, object_info

    for i in range(len(class_ids)):
        class_id = int(class_ids[i])
        if class_id <= 0 or class_id > len(class_names):
            continue

        box = boxes[i]
        class_name = class_names[class_id - 1]

        if class_name not in ALLOWED_CLASSES:
            continue

        area = int(box[2] * box[3])
        proximity, is_close = get_proximity(area, box, img.shape, class_name)
        x_center = int(box[0] + box[2] // 2)
        direction = get_direction(x_center, img.shape[1])

        if direction != "center":
            is_close = False
            if proximity == "very close":
                proximity = "ahead"

        object_info.append((class_name, proximity, direction, is_close))
        color = get_color(direction, is_close)
        cv2.rectangle(img, box, color, 2)
        cv2.putText(
            img,
            f"{class_name} {direction}",
            (int(box[0]) + 10, int(box[1]) + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return img, object_info


def detect_custom_objects(img, model: YOLO) -> Tuple[object, List[DetectionInfo]]:
    detect_info: List[DetectionInfo] = []
    results = model(img, verbose=False)
    detections = results[0].boxes

    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze()
        x1, y1, x2, y2 = xyxy.astype(int)
        confidence = float(detections[i].conf.item())
        class_id = int(detections[i].cls.item())
        class_name = model.names[class_id]

        if class_name == "door" and confidence < 0.8:
            continue
        if class_name == "stairs" and confidence < 0.5:
            continue
        if class_name == "window" and confidence < 0.6:
            continue

        box = (x1, y1, x2 - x1, y2 - y1)
        area = int(box[2] * box[3])
        proximity, is_close = get_proximity(area, box, img.shape, class_name)
        direction = get_direction((x1 + x2) // 2, img.shape[1])

        if direction != "center":
            is_close = False
            if proximity == "very close":
                proximity = "ahead"

        detect_info.append((class_name, proximity, direction, is_close))
        color = get_color(direction, is_close)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"{class_name}: {int(confidence * 100)}%",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

    return img, detect_info


def build_guidance_message(object_info: Sequence[DetectionInfo]) -> str:
    if not object_info:
        return "Path is clear"

    left_obs, center_obs, right_obs = [], [], []
    left_norm, center_norm, right_norm = [], [], []

    for name, proximity, direction, is_close in object_info:
        if direction == "center":
            is_obstacle = (name in OBSTACLE_CLASSES) or is_close
        else:
            is_obstacle = (name in OBSTACLE_CLASSES and proximity != "detected")

        entry = f"{name} {proximity}"

        if is_obstacle:
            if direction == "left":
                left_obs.append(entry)
            elif direction == "right":
                right_obs.append(entry)
            else:
                center_obs.append(entry)
        else:
            if direction == "left":
                left_norm.append(name)
            elif direction == "right":
                right_norm.append(name)
            else:
                center_norm.append(name)

    message_parts = []

    if center_obs:
        message_parts.append("In front of you, " + " and ".join(sorted(set(center_obs))))
        if not left_obs:
            message_parts.append("Move left")
        elif not right_obs:
            message_parts.append("Move right")
    else:
        message_parts.append("Path is clear")

    if left_obs:
        message_parts.append("On your left, " + " and ".join(sorted(set(left_obs))))
    if right_obs:
        message_parts.append("On your right, " + " and ".join(sorted(set(right_obs))))
    if left_norm:
        message_parts.append("Also on the left, " + " and ".join(sorted(set(left_norm))))
    if right_norm:
        message_parts.append("Also on the right, " + " and ".join(sorted(set(right_norm))))
    if center_norm:
        message_parts.append("Ahead, " + " and ".join(sorted(set(center_norm))))

    return ". ".join(message_parts)


def speak(message: str) -> None:
    # Use subprocess with an argument list so object names cannot break shell quoting.
    try:
        subprocess.run(["espeak-ng", message], check=False)
    except FileNotFoundError:
        print("Voice output skipped: espeak-ng is not installed.")


def run_system(args: argparse.Namespace) -> None:
    require_file(NCNN_MODEL_DIR / "model.ncnn.param", "NCNN model parameter file")
    require_file(NCNN_MODEL_DIR / "model.ncnn.bin", "NCNN model binary file")

    class_names = load_class_names(COCO_NAMES_PATH)
    ssd_net = load_ssd_model()
    custom_model = YOLO(str(NCNN_MODEL_DIR), task="detect")

    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(
            main={"size": (args.width, args.height), "format": "RGB888"}
        )
    )
    picam2.start()

    print("SafeStep system started. Press 'q' to stop.")

    frame_holder = {"frame": None}
    stop_event = threading.Event()

    def camera_thread() -> None:
        while not stop_event.is_set():
            frame_holder["frame"] = picam2.capture_array()

    thread = threading.Thread(target=camera_thread, daemon=True)
    thread.start()

    last_voice_time = 0.0
    frame_count = 0

    try:
        while True:
            frame = frame_holder["frame"]
            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            if frame_count % args.frame_skip != 0:
                continue

            img = frame.copy()
            img, object_info = detect_ssd_objects(img, ssd_net, class_names, args.ssd_confidence, args.nms_threshold)
            img, custom_info = detect_custom_objects(img, custom_model)
            object_info.extend(custom_info)

            current_time = time.time()
            if current_time - last_voice_time >= args.voice_interval:
                message = build_guidance_message(object_info)
                print(message)
                speak(message)
                last_voice_time = current_time

            if not args.no_preview:
                cv2.imshow("SafeStep", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        stop_event.set()
        picam2.stop()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SafeStep Smart Headband object detection system.")
    parser.add_argument("--width", type=int, default=640, help="Camera frame width.")
    parser.add_argument("--height", type=int, default=480, help="Camera frame height.")
    parser.add_argument("--frame-skip", type=int, default=3, help="Process one frame after this many captured frames.")
    parser.add_argument("--voice-interval", type=float, default=2.0, help="Seconds between voice guidance messages.")
    parser.add_argument("--ssd-confidence", type=float, default=0.45, help="SSD confidence threshold.")
    parser.add_argument("--nms-threshold", type=float, default=0.2, help="Non-maximum suppression threshold for SSD.")
    parser.add_argument("--no-preview", action="store_true", help="Run without OpenCV preview window.")
    return parser.parse_args()


if __name__ == "__main__":
    run_system(parse_args())
