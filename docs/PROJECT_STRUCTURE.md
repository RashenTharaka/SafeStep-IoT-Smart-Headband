# Project Structure

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

## Main Files

- `safestep_final_system.py` - main real-time detection and voice guidance script
- `requirements.txt` - Python package list
- `data/coco.names` - COCO class labels used by SSD MobileNet
- `models/ssd/` - SSD MobileNet model files
- `models/ncnn/best_ncnn_model/` - custom YOLOv8 NCNN model files
- `docs/` - user setup and project documentation

## Path Handling

The main Python script uses paths relative to its own location, so the project folder can be moved without editing file paths.
