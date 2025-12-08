# SqueakView

Jetson-first capture + DeepStream inference stack for YOLOv11 detection/pose with a single operator GUI. The repo packages three Python envs (capture, inference, control/GUI), DeepStream configs + custom YOLO parser, and utilities to export Ultralytics weights to TensorRT.

## What it does
- FLIR/Spinnaker capture to GStreamer shared memory, with optional TTL trigger gating.
- DeepStream pipeline (nvinfer + nvdsosd) for YOLOv11 bbox + pose, CSV logging, and dual MP4 recording.
- Qt GUI for run control, experiment metadata, behavior dashboard, and log tails.
- Engine builder notebook to convert `.pt` → ONNX → TensorRT and auto-generate matching DeepStream configs.

## Repo layout
- `squeakview_gui.py` – entrypoint; re-execs into the control env if it exists.
- `apps/capture` – PySpin → GStreamer shared-memory producer.
- `apps/inference` – DeepStream runner; can be launched headless or with preview window.
- `apps/operator` – GUI widgets + backend orchestrator.
- `DeepStream-Yolo/nvdsinfer_custom_impl_Yolo` – YOLOv11 bbox/pose parser (`libnvdsinfer_custom_impl_Yolo.so`).
- `DeepStream-Yolo/configs` – nvinfer configs; `configs/inference_off.txt` is a “no inference” passthrough.
- `build-engine/` – notebook + helper script to export weights and build TensorRT engines.
- `runs/` – per-run outputs (`raw.mp4`, `detections.csv`, `perf_stats.csv`, logs). Ignored in git.

## Requirements
- Jetson (tested with Orin) running JetPack/Ubuntu 20.04+ and DeepStream installed (pyds 1.2.0 wheel included).
- Spinnaker SDK + USB3/GenICam camera (wheel for `spinnaker-python 4.2.0.88` included).
- Python 3.10; `uv` for env management; Git LFS recommended if you plan to version large artifacts.
- (Optional) Arduino/behavior device on `/dev/ttyACM0` for TTL start/stop and event streaming to the dashboard.

## Install (Jetson)
Clone the repo to `/home/jetson/Desktop/squeakview` so the bundled wheel paths resolve, or adjust the wheel paths in the pyprojects under `environments/`.

```bash
cd /home/jetson/Desktop/squeakview
pip install uv  # or use the official installer

# Control / GUI env (PySide, serial, psutil)
uv sync --directory environments/control

# Capture env (PySpin)
uv sync --directory environments/capture

# DeepStream inference env (pyds)
uv sync --directory environments/inference
```

If your workspace is not the repo root, set `SQUEAKVIEW_WORKSPACE=/path/to/squeakview` before launching so helper paths resolve correctly.

## Build the DeepStream parser
Compile the YOLO bbox/pose parser (match `CUDA_VER` to your Jetson CUDA version):

```bash
cd DeepStream-Yolo/nvdsinfer_custom_impl_Yolo
make CUDA_VER=12.6
```

This produces `libnvdsinfer_custom_impl_Yolo.so`, which the configs reference.

## Build TensorRT engines (optional)
1) Drop Ultralytics `.pt` weights into `DeepStream-Yolo/artifacts/weights/`.
2) Run `build-engine/build_engine.ipynb` (or `python -m build_engine.main`) to export ONNX and build TensorRT engines:
   - ONNX → `DeepStream-Yolo/artifacts/onnx/<model>_<precision>.onnx`
   - Engine → `DeepStream-Yolo/engines/<model>_<precision>.engine`
   - Matching DeepStream config → `DeepStream-Yolo/configs/<model>_<precision>.txt`
3) For pose models, set `pose-kpt-labels-path` (or `labelfile-path`) in the generated config to your keypoint labels.

## Run it
- Launch the operator GUI (auto re-execs into the control env if it exists):
  ```bash
  python3 squeakview_gui.py
  # or: environments/control/.venv/bin/python squeakview_gui.py
  ```
- In the GUI:
  - Select DeepStream config (`DeepStream-Yolo/configs/*.txt`) and socket (`/tmp/cam.sock` by default).
  - Configure capture (resolution, fps, exposure, trigger on/off).
  - Optional: enable serial (port + baud) for TTL start/stop and behavior events.
  - Click **Start Recording** to launch capture + inference; outputs land in `runs/<timestamp>` (or `runs/<mouse_id>_<timestamp>`).

## Headless CLI snippets
- Capture only:
  ```bash
  uv run --directory environments/capture -m squeakview.apps.capture.main --width 1440 --height 1080 --fps 30 --socket /tmp/cam.sock
  ```
- Inference only (reuse an existing run dir if desired):
  ```bash
  uv run --directory environments/inference -m squeakview.apps.inference.main \
    --sock /tmp/cam.sock --cfg DeepStream-Yolo/configs/mousehouse_pose_fp16.txt \
    --width 1440 --height 1080 --fps 30 --bitrate 4000 --run-dir runs/your_run_dir
  ```

## Outputs and toggles
- `runs/<timestamp>/raw.mp4` – encoded capture.
- `runs/<timestamp>/detections.csv` – bbox/pose metadata.
- `runs/<timestamp>/perf_stats.csv` – FPS/latency.
- `preview_toggle.txt`, `skeleton_toggle.txt`, `video_toggle.txt` – runtime toggles monitored by the inference app.

## Git/LFS hygiene
- Large artifacts to ignore or keep in LFS: `runs/`, `DeepStream-Yolo/engines/`, `DeepStream-Yolo/artifacts/onnx/`, `DeepStream-Yolo/artifacts/weights/*.pt`, `*.engine`, `*.mp4`, `*.log`, `.venv/`, `*.so`, `*.whl`, `__pycache__/`.
