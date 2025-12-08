# SqueakView

Jetson-based capture + DeepStream inference for YOLOv11 detection and pose. Includes a lightweight operator GUI, DeepStream configs/parsers, and a build notebook for converting Ultralytics models to TensorRT engines.

## Layout
- `squeakview_gui.py` – entrypoint for the unified GUI (capture + inference dashboard).
- `apps/capture` – camera capture pipeline (shmsrc producer).
- `apps/inference` – DeepStream runner (nvinfer + nvdsosd), CSV logging, pose drawing.
- `apps/operator` – GUI widgets, launch coordination, experiment/run management.
- `DeepStream-Yolo/nvdsinfer_custom_impl_Yolo` – custom parser for YOLOv11 (bbox + pose) with unletterbox and keypoint support.
- `build-engine/build_engine.ipynb` – helper notebook to export .pt → ONNX → TensorRT engine and auto-generate DeepStream configs.
- `DeepStream-Yolo/configs` – nvinfer configs (bbox/pose).
- `runs/` – per-run outputs (raw video, detections.csv, logs). Ignored in git.

## Quick start
1) Install deps via uv (from repo root):
   ```bash
   uv sync
   ```
2) Launch GUI:
   ```bash
   uv run squeakview_gui.py
   ```
3) In the GUI: configure camera and DeepStream config, then Start Recording.

## Building engines
- Place .pt weights in `DeepStream-Yolo/artifacts/weights/`.
- Use `build-engine/build_engine.ipynb` to:
  - Export to ONNX → `DeepStream-Yolo/artifacts/onnx/<model>_<precision>.onnx`
  - Build TensorRT → `DeepStream-Yolo/engines/<model>_<precision>.engine`
  - Auto-write a matching DeepStream config in `DeepStream-Yolo/configs/`.
- Pose models use the custom parser (`NvDsInferParseYoloV8Pose`) and keypoint labels from `artifacts/labels/`.

## Custom parser
- Source: `DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/yolo_pose_parser.cpp`
- Build (example):
  ```bash
  cd DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
  make CUDA_VER=12.6
  ```
- Output: `libnvdsinfer_custom_impl_Yolo.so` referenced by DeepStream configs.

## Outputs
- `runs/<timestamp>/raw.mp4` – encoded capture.
- `runs/<timestamp>/detections.csv` – per-object (and pose) metadata.
- `runs/<timestamp>/perf_stats.csv` – FPS/latency.
- Toggles during run: `preview_toggle.txt`, `skeleton_toggle.txt`, `video_toggle.txt`.

## Notes
- Ignore large artifacts in git: `runs/`, `engines/`, `artifacts/onnx/`, `artifacts/weights/*.pt`, `*.engine`, `*.mp4`, `*.log`, `.venv/`, `__pycache__/`, `*.so`, `*.whl`.
- For pose keypoint naming, set `pose-kpt-labels-path` (or `labelfile-path`) in the DeepStream config. 
