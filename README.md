# SqueakView

SqueakView is a scientific FLIR capture, behavior logging, and YOLO26 pose
inference application for the NVIDIA Jetson Orin Nano. The current platform is
JetPack 7.2, CUDA 13.2, TensorRT 10.16, and DeepStream 9.1 using the
PyServiceMaker Pipeline API.

The primary validated configuration is one FLIR camera at 1440×1080 and 30 FPS,
with optional RP2040 serial triggering/logging and a batch-1 TensorRT pose
model. Generated engines are device-specific and are built locally on the
Jetson.

## Data-flow policy

SqueakView treats the compressed camera recording as scientific ground truth:

- The recording branch is non-leaky. It never intentionally drops a frame.
- Inference is downstream-leaky so slow inference cannot backpressure recording.
- GUI preview is leaky because it is only an operator spot check.
- Every recorded frame is reconciled against a durable source audit and the
  recording admission ledger.
- Missed live inference can be recovered later on downstream compute by
  replaying `raw.mp4`.

`raw.mp4` is H.264-compressed with CPU `x264enc`. It is authoritative, but not
an uncompressed sensor dump. CPU encoding is intentional because the Orin Nano
does not provide the hardware encoder used by larger Jetson modules.

## Acquisition and analysis boundary

The Jetson is the acquisition appliance. It records the run and performs only
the bounded post-run reconciliation and timing audit required to declare the
capture valid. Exploratory notebooks, production-scale analysis, video
transcoding, and offline re-inference belong on the DGX Spark.

The current transfer procedure is a manual copy of the complete finalized run
directory. Keep the Jetson original until the DGX copy has been inspected and
validated. See [data_viz/README.md](data_viz/README.md) for the run contract,
manual transfer checklist, DGX directory layout, alignment commands, and
downstream data-handling rules. Resumable `rsync` transfer is flagged for a
future release and is not part of the initial workflow.

## Repository layout

```text
build_me/                         Tracked source checkpoints and dataset YAMLs
build_engine/build_engine.ipynb  YOLO26 → ONNX/TensorRT model-package builder
configs/                         Runtime tracker configuration
native/flir_gst_source/          Scientific Spinnaker GStreamer source
native/nvdsinfer_custom_impl_yolo/ DeepStream YOLO26 detector parser
squeakview/apps/inference/        Live and offline PyServiceMaker pipelines
squeakview/apps/operator/         Qt operator GUI and run lifecycle
squeakview/common/                Run, profile, serial, and dashboard utilities
scripts/preflight.sh              Device/model readiness checks
scripts/align_run_outputs.py      Camera, video, inference, and TTL alignment
data_viz/README.md                Jetson-to-DGX transfer and analysis workflow
data_viz/analysis_demo_viz.ipynb Scientific run analysis and visualization
tests/                            Pure-Python and on-device pipeline tests
```

Device-local state is written under `models/`, `profiles/`, and `runs/`. Those
directories are ignored by Git. `build_me/` is intentionally tracked and its
source YAML files are treated as read-only ground truth by the builder.

To build and deploy a custom model, place its `.pt` checkpoint and corresponding
dataset YAML under `build_me/`, then run `build_engine/build_engine.ipynb` to
generate a device-specific package.

## Platform prerequisites

Install these before setting up the repository:

- JetPack 7.2 with CUDA/TensorRT
- NVIDIA DeepStream 9.1 at `/opt/nvidia/deepstream/deepstream`
- FLIR/Teledyne Spinnaker SDK at `/opt/spinnaker`
- [Spinnaker for JetPack 7.2](https://teledyne.app.box.com/s/ccj73r4xu8rusbnu12pytcisbexdchfa)
  is currently distributed as a beta build by Teledyne.
- GStreamer runtime and development headers
- Jetson Orin Nano specific install of ffmpeg (sudo apt install ffmpeg)
- CMake, a C++ compiler, and `uv`

Typical native-build packages:

```bash
sudo apt install build-essential cmake pkg-config \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

Expected vendor paths:

```text
/opt/spinnaker/include
/opt/spinnaker/lib
/opt/nvidia/deepstream/deepstream/sources/includes
/usr/local/cuda-13.2
```

Override the DeepStream root when necessary:

```bash
export SQUEAKVIEW_DEEPSTREAM_SDK=/opt/nvidia/deepstream/deepstream
```

## Python environment

SqueakView targets Python 3.12 on Linux AArch64. Create the project environment
with access to the JetPack/DeepStream system packages:

```bash
uv venv --python 3.12 --system-site-packages
uv sync
source .venv/bin/activate
```

The project uses CUDA 13.2 PyTorch wheels and a CUDA 13 ONNX Runtime index as
configured in `pyproject.toml`. DeepStream 9.1 supplies PyServiceMaker through
the system Python installation.

The live tensor decoder currently uses PyTorch DLPack to read DeepStream tensor
metadata. DeepStream performs TensorRT inference; ONNX Runtime is used by model
export tooling rather than by the live pipeline.

## Build native components

### FLIR GStreamer source

```bash
cmake -S native/flir_gst_source -B native/flir_gst_source/build
cmake --build native/flir_gst_source/build -j
GST_PLUGIN_PATH=$PWD/native/flir_gst_source/build gst-inspect-1.0 flirspinsrc
```

This produces `native/flir_gst_source/build/gstflirspinsrc.so`.

`flirspinsrc` uses Spinnaker directly and emits `GRAY8` frames. Its scientific
metadata profile records the FLIR chunk frame ID, acquisition-local image ID,
camera timestamp, exposure, gain, black level, ROI, pixel format, sequencer
state, CRC, image status, payload size, host clocks, camera/host clock
correlation, temperature, and transport counters. Incomplete images fail the
acquisition instead of being silently accepted.

See [native/flir_gst_source/README.md](native/flir_gst_source/README.md) for the
standalone camera smoke tests and complete metadata contract.

### DeepStream YOLO parser

Build against the CUDA version installed on the device:

```bash
make -C native/nvdsinfer_custom_impl_yolo CUDA_VER=13.2 -j
```

This produces:

```text
native/nvdsinfer_custom_impl_yolo/libnvdsinfer_custom_impl_Yolo.so
```

The library satisfies DeepStream's detector parser ABI for the YOLO26
end-to-end output. The PyServiceMaker `Yolo26PoseTensorOperator` is the
authoritative pose decoder: it reads `output0` tensor metadata, applies the
model-sidecar thresholds, restores source-image coordinates, and hands objects
and keypoints to NvDCF tracking and CSV persistence.

Rebuild both native components after JetPack, CUDA, TensorRT, DeepStream, or
Spinnaker upgrades.

## Build a YOLO26 pose model package

Fresh clones contain model inputs but no generated TensorRT packages. Start
Jupyter from the repository root:

```bash
uv run jupyter lab build_engine/build_engine.ipynb
```

Edit only the first notebook cell for the source checkpoint, dataset YAML,
package name, precision, batch size, image size, and confidence thresholds.
The remaining cells:

1. Read standard class names, keypoint names, and keypoint shape from the YAML
   and checkpoint without modifying the YAML.
2. Pass the YAML to `Ultralytics.export(data=...)`.
3. Export an end-to-end FP16 or FP32 TensorRT engine on the target Jetson.
4. Strip the Ultralytics metadata prefix to produce the raw TensorRT plan
   expected by DeepStream.
5. Validate ONNX and TensorRT input/output shapes.
6. Write and validate a schema-v2 model package.

Generated layout:

```text
models/<model_name>/
  weights/<source>.pt
  onnx/<model>_<precision>_b<batch>.onnx
  engines/<model>_<precision>_b<batch>.engine
  labels/classes.txt
  labels/labels.txt
  configs/<model_name>.txt
  configs/<model_name>.pose.json
  validation/import_report.json
  model.yaml
```

The pose sidecar contains tensor geometry, class thresholds, tracking policy, and keypoint labels. SqueakView stores and displays labeled keypoint dots.

TensorRT plan files are device-specific. Build each engine on the Jetson that
will run it. The model package batch size must match the configured camera
count. The currently validated packages and normal GUI workflow are batch 1.
Multi-camera support is planned, you may implement it now at your own risk by
matching the batch size to the number of configured cameras.

See [build_engine/README.md](build_engine/README.md) for builder details.

## Preflight

Select a generated model explicitly and run preflight with the project
interpreter:

```bash
PYTHON_BIN=.venv/bin/python \
SQUEAKVIEW_MODEL_NAME=<model_name> \
bash scripts/preflight.sh
```

You can provide the config directly instead:

```bash
PYTHON_BIN=.venv/bin/python \
DS_CFG=models/<model_name>/configs/<model_name>.txt \
bash scripts/preflight.sh
```

Run the same check remotely:

```bash
ssh -t jetson@<jetson-host> 'cd ~/Documents/SqueakView && PYTHON_BIN=.venv/bin/python SQUEAKVIEW_MODEL_NAME=<model_name> bash scripts/preflight.sh'
```

For camera-only testing without inference:

```bash
PYTHON_BIN=.venv/bin/python INFERENCE_ENABLED=0 bash scripts/preflight.sh
```

Preflight checks the DeepStream install, required GStreamer elements, the FLIR
plugin and capture-ledger API, NvDCF dependencies, the selected model package,
Jetson memory state, `ffprobe`, and Python imports.

DeepStream may print plugin-scanner warnings for unused optional plugins when an
OpenTelemetry library is absent. They do not affect SqueakView if every required
element passes preflight. Low contiguous NvMap memory is worth addressing before
a long run by closing Jupyter/GPU processes or rebooting the Jetson.

Note: clearing memory cache and setting the jetson to "cool" thermal profiles are
easily achieved by installing jtop.

## Launch the operator GUI

```bash
uv run squeakview_gui.py
```

Create or select an experiment, select the model config explicitly, configure
the FLIR camera and optional serial controller, and start the run. Serial
capture defaults to `/dev/ttyACM0` at 115200 baud; choose the actual device shown
by `ls /dev/ttyACM*`.

Useful path overrides:

```bash
export SQUEAKVIEW_WORKSPACE=/path/to/SqueakView
export SQUEAKVIEW_DEEPSTREAM_SDK=/opt/nvidia/deepstream/deepstream
export SQUEAKVIEW_MODEL_ROOT=/path/to/models
export SQUEAKVIEW_RUNS_DIR=/path/to/runs
uv run squeakview_gui.py
```

The GUI launches inference first and reports the run as recording only after the
PyServiceMaker pipeline reaches `PLAYING`. In triggered mode, it sends `START`
to the controller only after that readiness signal. The default readiness
timeout is 30 seconds and can be changed with
`SQUEAKVIEW_INFERENCE_READY_TIMEOUT`.

The timeout is a maximum wait for pipeline readiness, not a fixed warm-up delay.
Reducing it may reject a healthy pipeline that needs longer to load the engine
and enter `PLAYING`.

The run header and bottom status bar remain concise during normal operation.
Open **Events** for the full runtime and serial log. The capture-health panel
shows recording backlog, camera transport health, frame/drop counters, and the
current stop/finalization stage. Run identity and elapsed state remain visible
while the independent post-run finalizer is working.

## Live DeepStream pipeline

The validated single-camera path is:

```text
flirspinsrc → GRAY8 caps → tee
  ├─ record queue (120 frames / 4 s minimum, non-leaky)
  │    → x264enc (native GRAY8) → h264parse → mp4mux → raw.mp4
  └─ inference queue (32 frames, downstream-leaky)
       → VIC nvvideoconvert/NVMM-NV12 → nvstreammux
       → nvinfer/TensorRT → Python YOLO26 pose decode
       → CUDA NvDCF → [nvosdbin → nvstreamdemux when preview is enabled]
       → preview queue (1 frame, downstream-leaky) → nvunixfdsink
       → Qt nvunixfdsrc preview
```

The FLIR transport is configured with at least 64 host buffers (two seconds at
30 FPS), and the record queue intentionally backpressures rather than discarding
data. Recording backlog is sampled in `diagnostics/recording.csv`; a sustained
three-second backlog fails the run before the four-second queue can fill. If CPU
encoding cannot sustain acquisition, the run therefore stops with explicit
evidence instead of silently producing a plausible but incomplete video.

SqueakView fails a run explicitly when it cannot preserve timing and data
integrity. It does not silently continue with an incomplete but plausible video.
The current production candidate has completed a validated 16-hour,
1,707,205-frame run at 1440×1080 and 30 FPS with zero recording drops, frame
gaps, inference skips, or TTL mismatches.

The GUI/runtime can construct multiple camera branches, producing `raw.mp4` for
camera zero and `raw_camN.mp4` for additional cameras. Multi-camera inference
also requires a batch-N model package and has not received the same scientific
validation as the single-camera path. 

Multi-camera operation remains an advanced configuration and should not be used
for production acquisition until it receives equivalent long-run validation.

SqueakView uses the CUDA NvDCF tracker in `configs/tracker_mouse_nvdcf.yml`.
Jetson Orin Nano has no PVA hardware, so a PVA/VPI tracker profile is not an
acceleration option. Tracker thresholds remain model- and experiment-specific.

## Run outputs and frame identity

Runs are stored under:

```text
runs/<experiment>/<subject>/<subject>_<timestamp>_<shortid>/
```

Important outputs include:

```text
raw.mp4                         Authoritative compressed camera recording
frames.csv                      Authoritative recorded-frame ledger
serial.csv                      RP2040 events, TTLs, markers, and host clocks
objects.csv                     Detector/tracker object rows
keypoints.csv                   Normalized pose keypoints
run_manifest.json               Configuration, model identity, and artifacts
run_status.json                 Lifecycle, reconciliation, and validation
alignment_summary.json          Compact frame/video/controller audit
diagnostics/camera.csv          Temperature and transport health samples
diagnostics/recording.csv       Recording queue and encoder telemetry
diagnostics/errors.csv          Camera gaps, CRC, and metadata failures
diagnostics/camera_runtime.json Camera identity and clock calibration
diagnostics/post_run.log        Bounded finalizer log
config/                         Run-local capture/inference configuration
```

`frames.csv` is the source of truth for recorded frame identity. It contains
one row per recorded buffer and an `inference_admitted` field, so inference
admission does not require a second frame ledger. `objects.csv` is the single
object-observation table; track summaries are derived from it when analyzed.
Once a run reaches a terminal state, its acquisition, model, Git, and storage
provenance in `run_manifest.json` is immutable. Later bottle entry updates only
the bottle summary, artifact inventory, and manifest update timestamp.

During capture and finalization, recovery ledgers (`capture_cam*.jsonl`,
`record_admission*.csv`, and `inference/`) exist temporarily. They are removed
only after video validation, reconciliation, and timing alignment pass. A failed
finalization retains them for diagnosis.

The FLIR chunk `FrameID` is stored as `camera_frame_id`. It increments for
every acquired image. The aligner derives a per-run offset between that hardware
frame sequence and RP2040 `CAMERA_HIGH` counts. It validates frame continuity,
MP4 length, inference mapping, PTS, and camera/controller elapsed-clock agreement.

On stop, SqueakView asks the controller to stop before draining DeepStream,
keeps the serial reader open for final acknowledgements, validates `raw.mp4`,
reconciles the ledgers with bounded memory, and writes one compact alignment
summary. It does not create expanded copies of the canonical CSVs.

Finalization time scales with ledger length. The validated 16-hour run required
about 7.5 minutes to reconcile 1.7 million frames and 3.4 million serial rows.
During this period the run is safely stopped and the GUI reports the active
finalization stage; do not close the application or move the run directory.
Bottle intake is calculated as initial minus final weight. A final weight above
the initial weight is saved but shown as a plausibility warning.

To rerun the compact validator manually:

```bash
uv run python scripts/align_run_outputs.py /path/to/run
```

The command writes `alignment_summary.json` and exits nonzero when frame,
video, controller, or object mapping validation fails.

## Analyze a run

Copy the complete finalized run directory to the DGX Spark before beginning
downstream analysis. The source copy should remain unmodified; figures, tables,
new alignment results, and future re-inference outputs belong in a separate
analysis-results directory.

The visualization notebook is a short-run example of the current tables and
their timing relationships, not the production engine for a full-length run.
Launch it on the DGX from an analysis environment:

```bash
jupyter lab data_viz/analysis_demo_viz.ipynb
```

Set `RUN_DIR` in the first code cell to an explicit copied run. Leave
`INFERENCE_RESULT = "live"` to inspect the acquisition-time objects and
keypoints.

The notebook reports acquisition and inference health, TTL/PTS timing,
behavioral events, mapping provenance, NvDCF tracks, keypoint confidence, event-
locked object data, and an optional exact-frame `raw.mp4` preview with boxes and
keypoint dots.

The current notebook loads canonical CSVs into memory and is intended for short
or sampled demo runs. Do not run it on the acquisition Jetson or point it at a
complete 16-hour run. See [data_viz/README.md](data_viz/README.md) for the full
workflow and source-data rules.

## Offline re-inference (downstream only)

Do not run offline re-inference on the acquisition Jetson. The existing module
is a development path for short compatible runs, not the production DGX
workflow. Production downstream replay will require a DGX-specific TensorRT
engine, long-run ledger handling, and a derived-results destination that never
overwrites `raw.mp4`, `objects.csv`, or `keypoints.csv`. Those requirements are
recorded in [data_viz/README.md](data_viz/README.md).

## Diagnostics

Show optional DeepStream plugin warnings that the GUI normally filters:

```bash
SQUEAKVIEW_SHOW_PLUGIN_WARNINGS=1 uv run squeakview_gui.py
```

Request fan control when the current user has the required privileges:

```bash
SQUEAKVIEW_SET_FAN=1 uv run squeakview_gui.py
```

Temporarily bypass GUI preflight only for deliberate diagnosis:

```bash
SQUEAKVIEW_SKIP_PREFLIGHT=1 uv run squeakview_gui.py
```

Run the test suite without changing the environment:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Git policy

Commit source code, tests, documentation, notebooks without saved outputs,
configuration, native build recipes, and the tracked `build_me/` model inputs.
Do not commit device-local or generated state:

```text
.venv/
models/
profiles/
runs/
native/flir_gst_source/build/
native/**/*.o
native/**/*.so
```

A fresh clone should contain everything needed to rebuild the native plugins
and local model packages on a compatible Jetson, but no device-specific engine
or experimental run output.
