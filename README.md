# SqueakView

Jetson-first capture + DeepStream inference stack for multi-camera FLIR workflows and experimental ZED support, with a single operator GUI. The repo uses one Python environment for capture, inference, and control/GUI, plus DeepStream configs, a custom YOLO parser, and tooling to export Ultralytics weights to TensorRT.

Additional files for 3D printing and Arduino firmware for the RP2040 controller board are included. These can be removed on a stripped-down Jetson deployment.

## What it does
- FLIR/Spinnaker capture to GStreamer shared memory, with optional TTL trigger gating and multi-camera support.
- DeepStream pipeline (`nvinfer` + `nvdsosd`) for YOLO bbox + pose, CSV logging, and MP4 recording.
- Supports multiclass detection and multiclass pose inference.
- Experimental ZED capture path with DeepStream/GStreamer integration, depth preview, and SVO recording support.
- Qt GUI for run control, experiment metadata, behavior dashboard, task-state panel, and log tails.
- Post-run Google Drive upload prompt via `rclone`, excluding large video files by default.
- Engine builder notebook to convert `.pt` -> ONNX -> TensorRT and auto-generate matching DeepStream configs.

## Demo
- For a video walkthrough, see: https://www.youtube.com/watch?v=CRolGeF1rnc

## Files
- .STEP files for 3D printing are in the MouseHouse CAD folder
- Arduino code for flashing the RP2040 is in the Arduino Firmware Folder
- 

## Repo layout
- `squeakview_gui.py` – entrypoint for the operator GUI.
- `apps/capture` – PySpin → GStreamer shared-memory producer.
- `apps/inference` – DeepStream runner; can be launched headless or with preview window.
- `apps/operator` – GUI widgets + backend orchestrator.
- `DeepStream-Yolo/nvdsinfer_custom_impl_Yolo` – YOLOv11 bbox/pose parser (`libnvdsinfer_custom_impl_Yolo.so`).
- `DeepStream-Yolo/configs` – nvinfer configs; `configs/inference_off.txt` is a “no inference” passthrough.
- `build-engine/` – notebook + helper script to export weights and build TensorRT engines.
- `tutorials/` – setup notes for ZED GStreamer and Google Drive upload configuration.
- `runs/` – per-run outputs (`raw.mp4`, `detections.csv`, `perf_stats.csv`, logs). Ignored in git.

## Requirements
- Jetson (tested with Orin) running JetPack/Ubuntu 20.04+ and DeepStream installed (pyds 1.2.0 wheel included).
- Spinnaker SDK + USB3/GenICam camera (wheel for `spinnaker-python 4.2.0.88` included).
- ZED support requires additional system-level setup beyond `uv sync`:
  - ZED SDK
  - ZED GStreamer plugins (`zedsrc`, `zeddemux`, etc.)
  - manual `pyzed` wheel install matching the local SDK / Python / architecture
- Python 3.10; `uv` for env management; Git LFS recommended if you plan to version large artifacts.
- (Optional) Arduino/behavior device on `/dev/ttyACM0` for TTL start/stop and event streaming to the dashboard.

Upgrade note:
- The current supported Python target is 3.10.
- A JetPack / system upgrade path to Python 3.12 is not ready yet because key binary dependencies still need matching support, especially on the FLIR / Spinnaker, DeepStream `pyds`, and ZED / `pyzed` sides.
- We are waiting on dependency compatibility and working with the FLIR and ZED stacks around that transition.

## Install (Jetson)
Clone the repo to `/home/jetson/Desktop/Squeakview` so bundled wheel paths resolve (or adjust `pyproject.toml`).

```bash
cd /home/jetson/Desktop/Squeakview
# use the official UV installer https://docs.astral.sh/uv/getting-started/installation/

# Single SqueakView environment (GUI + capture + inference)
uv sync
```

Important:
- `uv sync` does not currently recreate a fully working ZED Python environment by itself.
- `pyzed` is manually installed from a local wheel and may be removed again by later `uv sync` operations unless reinstalled.
- FLIR support is much closer to “clone + sync + run” than ZED support.

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
4) If you will run inference on 2 cameras at once, build the model/config for `batch-size=2`.
   - In `build-engine/build_engine.ipynb`, batch size is controlled by the first dimension of `IMG_SIZE`.
   - Example for 2-camera inference:
     - `IMG_SIZE = (2, 3, 640, 640)`
   - The generated DeepStream config must also use:
     - `batch-size=2`
   - For single-camera inference, use batch size 1.

Notes for pose / multiclass models:
- The parser supports multiclass pose models.
- `detections.csv` records multiclass predictions using:
  - `class_id`
  - `class_label`
- `detections.csv` now keeps one row per detection and stores pose payload per row using:
  - `pose_schema`
  - `kpt_count`
  - `kpt_names_json`
  - `kpt_values_json`
- The base per-detection fields remain:
  - `frame`, `ts_us`, `stream_id`, `source`, `obj_id`, `class_id`, `class_label`, `conf`, `x`, `y`, `w`, `h`
- For class-specific keypoint schemas, the pose label file can be organized in bracketed sections such as:
  - `[mouse]`
  - `[landmarks]`
- SqueakView slices the flat pose tensor into the appropriate class-specific subset when writing CSV rows.

## Tutorials
- Google Drive upload setup: `tutorials/google_drive_upload_setup.txt`
- ZED GStreamer setup: `tutorials/zed_gstreamer_setup.txt`

## Run it
- Launch the operator GUI:
  ```
  uv run squeakview_gui.py
  ```
- In the GUI:
  - Select DeepStream config (`DeepStream-Yolo/configs/*.txt`) and socket (`/tmp/cam.sock` by default).
  - Configure capture (resolution, fps, exposure, trigger on/off).
  - Optional: enable serial (port + baud) for TTL start/stop and behavior events.
  - Select a task YAML if you want the behavior dashboard and task-status panel to reflect task-specific serial events.
  - Click **Start Recording** to launch capture + inference; outputs land in `runs/<timestamp>` (or `runs/<mouse_id>_<timestamp>`).

Capture backends:
- FLIR is the primary supported production path.
- ZED basically works, but depends on substantial local system setup and plugin/toolchain state. Treat it as a configured-system feature, not a clean-room install path yet.

## Outputs and toggles
- `runs/<timestamp>/raw.mp4` – encoded video. For single-camera runs with inference enabled, this is recorded on the detection-aligned branch so `detections.csv` frame numbers match the saved video.
- `runs/<timestamp>/detections.csv` – bbox/pose metadata.
- `runs/<timestamp>/perf_stats.csv` – FPS/latency.
- `runs/<timestamp>/serial.csv` – raw behavior / Arduino event stream.
- `runs/<timestamp>/camera_settings.json` – run metadata and config snapshot.
- `runs/<timestamp>/zed_capture.svo2` – ZED SVO recording when enabled and available.
- `preview_toggle.txt`, `skeleton_toggle.txt`, `video_toggle.txt` – runtime toggles monitored by the inference app.

## Google Drive uploads
- After a run stops, the GUI can prompt to upload a copy of the run directory to Google Drive.
- Uploads use `rclone` with remote name `gdrive`.
- Default destination:
  - `gdrive:SqueakViewUploads/<run_dir_name>`
- Excluded by default:
  - `*.mp4`
  - `*.svo`
  - `*.svo2`
- Local data is retained after upload.
- Setup instructions:
  - `tutorials/google_drive_upload_setup.txt`

## Task / behavior dashboard
- Task YAML files define event-matching rules for the behavior dashboard.
- `gonogo_auto.yaml` also enables a task settings panel in the GUI that can display live task-state fields such as:
  - stage
  - hold time
  - go time
  - nogo time
  - go probability
  - side
  - reason
- These values come from serial event lines emitted by the behavior controller, not from inference output.

## Email alerts (optional)
- The serial reader can fire an email when a serial line contains a phrase (default: “Feeder jammed”). Configure via env vars before launching the GUI:
  - `SQUEAKVIEW_ALERT_EMAIL_HOST` / `SQUEAKVIEW_ALERT_EMAIL_PORT` (e.g., smtp.gmail.com / 587)
  - `SQUEAKVIEW_ALERT_EMAIL_USER` / `SQUEAKVIEW_ALERT_EMAIL_PASS` (SMTP/app password)
  - `SQUEAKVIEW_ALERT_EMAIL_FROM` / `SQUEAKVIEW_ALERT_EMAIL_TO`
  - `SQUEAKVIEW_ALERT_EMAIL_TLS` (1/0), `SQUEAKVIEW_ALERT_EMAIL_SUBJECT` (optional)
  - `SQUEAKVIEW_SERIAL_ALERT_PHRASE` to override the trigger text
- Example (use an app password for Gmail):
  ```bash
  export SQUEAKVIEW_ALERT_EMAIL_HOST="smtp.gmail.com"
  export SQUEAKVIEW_ALERT_EMAIL_PORT="587"
  export SQUEAKVIEW_ALERT_EMAIL_USER="squeakview.alerts@gmail.com"
  export SQUEAKVIEW_ALERT_EMAIL_PASS="your_app_password"
  export SQUEAKVIEW_ALERT_EMAIL_FROM="squeakview.alerts@gmail.com"
  export SQUEAKVIEW_ALERT_EMAIL_TO="you@example.com"
  export SQUEAKVIEW_ALERT_EMAIL_TLS="1"
  export SQUEAKVIEW_SERIAL_ALERT_PHRASE="Feeder jammed"
  export SQUEAKVIEW_ALERT_EMAIL_SUBJECT="SqueakView serial alert test"
  uv run squeakview_gui.py
  ```
  alerts dont block capture/serial log. user must config app password with an email domain (this is not your gmail password :))
