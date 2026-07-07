# SqueakView

Trimmed direct-FLIR SqueakView stack for Jetson Orin Nano.

This repo is intended to be cloned onto a Jetson that already has NVIDIA
DeepStream and FLIR/Teledyne Spinnaker installed locally. It does not vendor
DeepStream, Spinnaker, TensorRT engines, trained model weights, generated ONNX
files, or run outputs.

This repo owns:

- `native/flir_gst_source`: GStreamer `flirspinsrc` built against FLIR Spinnaker.
- `native/nvdsinfer_custom_impl_yolo`: DeepStream custom parser for YOLO26 pose.
- `models/`: local generated model packages. This directory is ignored by Git.
- `build_engine`: notebook workflow for importing YOLO26 pose `.pt` models into the `models/` package layout.
- `squeakview/apps/inference`: Python DeepStream pipeline runner.
- `squeakview/apps/operator`: operator GUI/backend used to launch the direct FLIR path.
- `squeakview/common`: run metadata, profiles, dashboard, and serial helpers.
- `squeakview/config.py`: central path configuration for the local repo and `/opt/nvidia/deepstream`.

This repo does not vendor NVIDIA DeepStream. It expects the device install at:

```bash
/opt/nvidia/deepstream/deepstream
```

Override with:

```bash
export SQUEAKVIEW_DEEPSTREAM_SDK=/opt/nvidia/deepstream/deepstream
```

## Fresh Clone Setup

For a new Jetson clone, the normal setup order is:

1. Install system prerequisites.
2. Sync the Python environment.
3. Build the two native `.so` files.
4. Import/build a YOLO26 pose model package under `models/`.
5. Run preflight.
6. Launch the GUI.

`models/`, `build_me/`, `runs/`, and `profiles/` are local machine state and are
ignored by Git.

## Install Prerequisites

Install these on the Jetson before setting up the Python environment:

- NVIDIA DeepStream SDK at `/opt/nvidia/deepstream/deepstream`
- FLIR/Teledyne Spinnaker SDK at `/opt/spinnaker`
- CUDA/TensorRT matching the JetPack/DeepStream install
- GStreamer runtime and development headers
- CMake and a C++ compiler
- `uv`

Typical Ubuntu packages for the native builds:

```bash
sudo apt install build-essential cmake pkg-config \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

The native build expects these local paths:

```text
/opt/spinnaker/include
/opt/spinnaker/lib
/opt/nvidia/deepstream/deepstream/sources/includes
/usr/local/cuda-<CUDA_VER>
```

## Python Environment

The project targets Python 3.10 on Jetson:

```bash
uv sync
source .venv/bin/activate
```

`pyds` is expected from:

```text
wheel_files/pyds-1.2.0-cp310-cp310-linux_aarch64.whl
```

If `uv sync` reinstalls PyTorch, make sure the selected wheels match the Jetson
CUDA driver. A mismatched PyTorch wheel can still let DeepStream run, but it can
break notebook-based model export.

## Build Native Components

Build the FLIR Spinnaker GStreamer source:

```bash
cmake -S native/flir_gst_source -B native/flir_gst_source/build
cmake --build native/flir_gst_source/build -j
```

This creates:

```text
native/flir_gst_source/build/gstflirspinsrc.so
```

Quick check:

```bash
GST_PLUGIN_PATH=$PWD/native/flir_gst_source/build gst-inspect-1.0 flirspinsrc
```

Build the DeepStream YOLO parser used by `nvinfer`:

```bash
cd native/nvdsinfer_custom_impl_yolo
CUDA_VER=12.6 make -j
cd ../..
```

Use the CUDA version installed on the Jetson. For example, CUDA 12.6 uses
`CUDA_VER=12.6`. This creates:

```text
native/nvdsinfer_custom_impl_yolo/libnvdsinfer_custom_impl_Yolo.so
```

Quick check:

```bash
test -f native/nvdsinfer_custom_impl_yolo/libnvdsinfer_custom_impl_Yolo.so
```

Both native outputs are build artifacts and are ignored by Git. The source and
build recipes are shipped; each Jetson rebuilds the `.so` files locally.

## Native Runtime Roles

`flirspinsrc` is the live camera source. It talks to the FLIR camera through
Spinnaker and publishes `GRAY8` GStreamer frames:

```text
flirspinsrc ! video/x-raw,format=GRAY8 ! nvvideoconvert ! nvstreammux
```

`libnvdsinfer_custom_impl_Yolo.so` is the DeepStream parser library. Generated
YOLO26 pose model configs point `nvinfer` at this library:

```text
parse-bbox-func-name=NvDsInferParseYolo26Pose
custom-lib-path=<repo>/native/nvdsinfer_custom_impl_yolo/libnvdsinfer_custom_impl_Yolo.so
```

Without the FLIR plugin, the camera pipe cannot start. Without the YOLO parser,
DeepStream can load the TensorRT engine but cannot decode YOLO26 pose outputs
into detections/keypoints.

## Build A YOLO26 Model Package

Use the notebook workflow when bringing over a new YOLO26 pose `.pt` model and
training/data `.yaml`:

```bash
uv run jupyter lab build_engine/build_engine.ipynb
```

The notebook writes a complete package under `models/<model_name>/`, including
labels, ONNX, TensorRT engine, DeepStream config, pose sidecar JSON, and a
`model.yaml` manifest.

`build_me/` and `models/` are local artifact directories and are ignored by Git.
Experiment profiles should use repo-relative paths such as
`models/<model_name>/configs/<model_name>.txt` and `tasks/default.yaml`.
Generated DeepStream `nvinfer` configs use paths relative to the config file.
The GUI still localizes those configs into each run directory before launching
inference so DeepStream receives concrete paths for the active clone.

Before running acquisition, make sure the selected model package contains:

```text
models/<model_name>/
  configs/<model_name>.txt
  configs/<model_name>.pose.json
  engines/<model>_<precision>_b<batch>.engine
  labels/classes.txt
  labels/labels.txt
  model.yaml
```

## Preflight

Run this after the native plugins and the selected model package exist:

```bash
bash scripts/preflight.sh
```

Preflight checks DeepStream, GStreamer elements, the FLIR source plugin, the
YOLO parser library, the selected model package, Jetson memory state, and Python
imports. If you are using a non-default model config, pass it with `DS_CFG`:

```bash
DS_CFG=models/<model_name>/configs/<model_name>.txt bash scripts/preflight.sh
```

## Launch GUI

```bash
uv run squeakview_gui.py
```

The default capture backend is `FLIR Direct (Spinnaker/GStreamer)`.

If you need to override the workspace, DeepStream SDK, model root, or run
directory:

```bash
export SQUEAKVIEW_WORKSPACE=/path/to/SqueakView
export SQUEAKVIEW_DEEPSTREAM_SDK=/opt/nvidia/deepstream/deepstream
export SQUEAKVIEW_MODEL_ROOT=/path/to/models
export SQUEAKVIEW_RUNS_DIR=/path/to/runs
uv run squeakview_gui.py
```

## Runtime Shape

The operator GUI launches only one camera path:

```text
flirspinsrc -> source frame tap -> tee
  record branch: non-leaky queue -> compressed MP4
  inference branch: leaky queue -> nvvideoconvert -> nvstreammux -> nvinfer -> nvdsosd -> preview
```

Runs are local-only and are created under:

```text
runs/<experiment>/<subject>/<subject>_<YYYY-MM-DD_HH-MM-SS>_<shortid>/
```

Each run writes `run_status.json`, `run_manifest.json`, `camera_settings.json`,
`frames.csv`, `drop_events.csv`, `raw.mp4`, `serial.csv` when the Arduino is
enabled, and `detections.csv` when inference is enabled. Single-file recording
is the default because it gives unambiguous frame provenance.
The GUI verifies that `SQUEAKVIEW_RUNS_DIR` is writable and has at least 1 GB
free before starting; override that threshold with
`SQUEAKVIEW_MIN_RUN_FREE_BYTES` for long recordings.

Use `frames.csv` as the source of truth for global frame identity. In default
single-file mode, `record_segment_file=raw.mp4`,
`segment_local_frame_index=raw_frame_index`, and
`segment_mapping_source=single_file`.

On stop, the backend builds `analysis/` transactionally from `analysis.tmp`.
Chunked MP4 recording is disabled unless `SQUEAKVIEW_ENABLE_CHUNKED_RECORDING=1`
is set. Chunked runs must include writer-owned `video_segments.csv`, emitted
from `splitmuxsink` boundary signals. Analysis fails hard if chunked `raw_*.mp4`
files exist without that ledger; runtime PTS estimates are not accepted as
authoritative chunk provenance. The aligned CSVs include Arduino TTL timing and
detection rows.

The old external capture worker, shared-memory camera sockets, and ZED path are intentionally not part of this repo.

Debug logging is opt-in:

```bash
SQUEAKVIEW_DEBUG_INFER=1 SQUEAKVIEW_SURF_DEBUG=1 SQUEAKVIEW_POSE_PARSER_DEBUG=1 uv run squeakview_gui.py
```

Known optional DeepStream plugin scanner warnings for Triton/Rivermax are hidden in the GUI log by default. To show them:

```bash
SQUEAKVIEW_SHOW_PLUGIN_WARNINGS=1 uv run squeakview_gui.py
```

Fan control is also opt-in because `jetson_clocks --fan` usually requires privileges:

```bash
SQUEAKVIEW_SET_FAN=1 uv run squeakview_gui.py
```

## Git Policy

Commit source, configuration templates, notebooks, scripts, and documentation.
Do not commit generated local artifacts:

```text
.venv/
runs/
profiles/
build_me/
models/
native/**/build/
native/**/*.o
native/**/*.so
```

The repo should ship enough source for a Jetson user to rebuild the FLIR source
plugin and YOLO parser locally, then import their own YOLO26 pose model package.
