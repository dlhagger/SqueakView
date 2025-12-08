# SqueakView

Jetson-first capture + DeepStream inference stack for YOLOv11 detection/pose with a single operator GUI. The repo packages three Python envs (capture, inference, control/GUI), DeepStream configs + custom YOLO parser, and utilities to export Ultralytics weights to TensorRT.

Additonal files for 3D printing and Ardiuno Firmware to flash the RP2040 controller board are included (these files can be removed on the Jetson deployment)

## What it does
- FLIR/Spinnaker capture to GStreamer shared memory, with optional TTL trigger gating.
- DeepStream pipeline (nvinfer + nvdsosd) for YOLOv11 bbox + pose, CSV logging, and dual MP4 recording.
- Qt GUI for run control, experiment metadata, behavior dashboard, and log tails.
- Engine builder notebook to convert `.pt` → ONNX → TensorRT and auto-generate matching DeepStream configs.

## Demo
- For a video walkthrough, see: https://www.youtube.com/watch?v=CRolGeF1rnc

## Files
- .STEP files for 3D printing are in the MouseHouse CAD folder
- Arduino code for flashing the RP2040 is in the Arduino Firmware Folder
- 

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
# use the official UV installer https://docs.astral.sh/uv/getting-started/installation/

# Control / GUI env (PySide, serial, psutil)
uv sync --directory environments/control

# Capture env (PySpin)
uv sync --directory environments/capture

# DeepStream inference env (pyds)
uv sync --directory environments/inference
```

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
- Launch the operator GUI:
  ```
  uv run squeakview_gui.py
  ```
- In the GUI:
  - Select DeepStream config (`DeepStream-Yolo/configs/*.txt`) and socket (`/tmp/cam.sock` by default).
  - Configure capture (resolution, fps, exposure, trigger on/off).
  - Optional: enable serial (port + baud) for TTL start/stop and behavior events.
  - Click **Start Recording** to launch capture + inference; outputs land in `runs/<timestamp>` (or `runs/<mouse_id>_<timestamp>`).

## Outputs and toggles
- `runs/<timestamp>/raw.mp4` – encoded capture.
- `runs/<timestamp>/detections.csv` – bbox/pose metadata.
- `runs/<timestamp>/perf_stats.csv` – FPS/latency.
- `preview_toggle.txt`, `skeleton_toggle.txt`, `video_toggle.txt` – runtime toggles monitored by the inference app.

## License & notice
Temporary Research Use Only License
Copyright © 2025 National Institutes of Health

This software is available for academic and non-profit research use under a temporary Research Use Only license. Commercial use may require a separate license from the NIH Office of Technology Transfer. All use, redistribution, and modification must additionally comply with the licensing terms of third-party software dependencies incorporated into the system. The licensing terms will be updated upon completion of the NIH review.
