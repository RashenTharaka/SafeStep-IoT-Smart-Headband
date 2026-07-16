# User Guide

## Purpose

SafeStep provides real-time assistive feedback using camera-based object detection and voice output.

## How to Start

Run this command from the project folder:

```bash
python3 safestep_final_system.py
```

The system opens a camera preview window and starts processing frames.

## Voice Guidance Examples

Examples of generated messages:

```text
Path is clear
In front of you, person ahead. Move left
On your left, chair ahead
On your right, door detected
```

## Detection Logic

The system analyzes:

- Object type
- Object position in the frame
- Approximate proximity based on bounding box size and position
- Whether the object is treated as an obstacle

## Controls

- Press `q` in the preview window to stop the system.
- Use `--no-preview` for headless mode.

## Best Use Conditions

- Use in good lighting.
- Keep the camera stable.
- Use earphones or a clear audio output device.
- Test indoors first before trying complex environments.

## Safety Notice

This project is an academic prototype. It should not be used as the only navigation method in real-world safety-critical situations.
