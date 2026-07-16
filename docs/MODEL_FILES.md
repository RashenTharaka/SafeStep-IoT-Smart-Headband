# Model Files

SafeStep uses two detection models.

## 1. SSD MobileNet

Location:

```text
models/ssd/
```

Files:

```text
frozen_inference_graph.pb
ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
```

Purpose:

- Detects general COCO object classes such as people, vehicles, chairs, bottles, phones, and other common objects.

## 2. Custom YOLOv8 NCNN Model

Location:

```text
models/ncnn/best_ncnn_model/
```

Files:

```text
model.ncnn.bin
model.ncnn.param
metadata.yaml
model_ncnn.py
```

Purpose:

- Detects project-specific classes:
  - door
  - stairs
  - window

## Notes

- Do not delete or rename model files unless the paths are updated in `safestep_final_system.py`.
- `model_ncnn.py` is a small loading test script for the NCNN model.
- Model performance depends on lighting, camera angle, training data, and Raspberry Pi hardware.
