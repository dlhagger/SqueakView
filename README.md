# SqueakView

SqueakView is a scientific FLIR capture, behavior logging, and YOLO26 pose
inference application for the NVIDIA Jetson Orin Nano. The current platform is
JetPack 7.2.1, CUDA 13.2, TensorRT 10.16, and DeepStream 9.1 using the
PyServiceMaker Pipeline API.

The current development baseline is one FLIR camera at 1440×1080 and 30 FPS,
with optional RP2040 serial triggering/logging and a batch-1 TensorRT pose
model. Generated engines are device-specific and are built locally on the
Jetson. The GUI's wider numeric ranges permit development experiments; they do
not imply scientific qualification. The checked-in production matrix supports
only its explicitly listed 1440×1080/30 FPS profile, and that profile remains
unqualified until the acceptance gates below are completed.

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

## Jetson device setup

Run the device setup once as the desktop user who will launch SqueakView:

```bash
bash scripts/setup_jetson.sh
```

The script installs NVIDIA's Jetson FFmpeg package (including `ffprobe`) and
the complete GStreamer runtime/plugin and native build prerequisites, builds
both the FLIR GStreamer source and the
DeepStream YOLO parser against the installed CUDA toolkit, and adds that user
to the `dialout` group for RP2040/USB serial access. It can be launched from any
working directory. Reboot after it completes because a group change cannot
affect an already-running login session. Do not run the application itself
with `sudo`.

## Platform prerequisites

Install these before setting up the repository:

- JetPack 7.2.1 with CUDA/TensorRT
- NVIDIA DeepStream 9.1 at `/opt/nvidia/deepstream/deepstream`
- FLIR/Teledyne Spinnaker SDK at `/opt/spinnaker`
- [Spinnaker for the JetPack 7.2 series](https://teledyne.app.box.com/s/ccj73r4xu8rusbnu12pytcisbexdchfa)
  is currently distributed as a beta build by Teledyne and is deployed here on
  JetPack 7.2.1.
- GStreamer runtime and development headers
- Jetson Orin Nano specific install of FFmpeg (`scripts/setup_jetson.sh` installs it)
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
3. Export an NMS-free, end-to-end FP16 or FP32 TensorRT engine on the target
   Jetson using Ultralytics `nms=False`.
4. Strip the Ultralytics metadata prefix to produce the raw TensorRT plan
   expected by DeepStream.
5. Validate ONNX and TensorRT input/output shapes.
6. Write and validate a schema-3 model package with a schema-2 pose sidecar.

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

Each run manifest records SHA-256 identities for the model manifest, pose
sidecar, ONNX graph, DeepStream config, and TensorRT engine. Production
qualification requires the complete identity set and the exact installed
`nvidia-jetpack` package version is retained with the other platform packages.

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

For a manual check of the configured controller as well, provide its device:

```bash
SQUEAKVIEW_MODEL_NAME=<model_name> SERIAL_ENABLED=1 \
SERIAL_PORT=/dev/ttyACM0 PYTHON_BIN=.venv/bin/python bash scripts/preflight.sh
```

Preflight checks the DeepStream install, required GStreamer elements, the FLIR
plugin and capture-ledger API, NvDCF dependencies, the selected model package,
Jetson memory state, `ffprobe`, and Python imports. It exercises DeepStream
9.1 Stream multiplexer 2 through the same VIC-to-surface-array NVMM path used
by acquisition. Before every recording, it also
creates a temporary 30-frame H.264 clip and fully decodes it through the same
isolated `nvv4l2decoder` used for anomaly escalation. Acquisition is
blocked if the Jetson hardware decoder cannot complete that real decoder
self-test. No readiness result is reused across recordings.

Inference preflight requires a schema-3 engine build identity matching the
current TensorRT, CUDA, Jetson Linux, GPU capability, and device. A schema-2
package remains readable for migration inspection but is blocked from
scientific acquisition; rebuild it with `build_engine/build_engine.ipynb`.

DeepStream may print plugin-scanner warnings for unused optional plugins when an
OpenTelemetry library is absent. They do not affect SqueakView if every required
element passes preflight. Preflight reports the current `tegrastats` sample for
diagnosis. NVIDIA documents its LFB field as an allocator statistic whose largest
normal block is at most 4 MB, so LFB is not treated as a standalone readiness
threshold.

See [`docs/REFACTOR_PLAN.md`](docs/REFACTOR_PLAN.md) for the staged JetPack 7.2.1
and DeepStream 9.1 modernization plan and the scientific recording invariants.

Note: clearing memory cache and setting the jetson to "cool" thermal profiles are
easily achieved by installing jtop.

## Launch the operator GUI

```bash
bash scripts/launch_operator.sh
```

This detached launcher is the production path. It starts a Qt-free supervisor,
which owns the controller, capture, run lock, and finalization while launching
and monitoring the required GUI. If the GUI exits or loses its exclusive IPC
lease during a run, the supervisor stops acquisition and completes fail-closed
finalization; acquisition never continues headlessly. Terminal or VS Code loss
does not end the supervised session. Separate bounded GUI and supervisor logs
are written under `runs/logs/`; each mirror is capped at 32 MiB and partial
lines are capped at 64 KiB.
The launcher reports success only after the supervisor has authenticated the
GUI's IPC connection; an early GUI failure or connection timeout returns
nonzero with the supervisor log path. The bounded wait defaults to 35 seconds,
slightly longer than the supervisor's 30-second GUI connection deadline,
and can be adjusted with `SQUEAKVIEW_LAUNCH_TIMEOUT_S`.

The lease is renewed only by a Qt-main-loop timer, so a frozen GUI is treated
as lost even if its background IPC thread still exists. The DeepStream capture
child also uses Linux parent-death signaling, preventing a hard supervisor
failure from leaving the sessionized capture process alive. A hard owner failure
cannot perform ordered finalization, and that run must not pass qualification.

The RP2040 is a separate safety boundary. This repository currently contains no
versioned controller firmware or qualified negotiated watchdog, so an
independently powered controller may continue generating TTL pulses after total
Jetson/supervisor failure. Triggered scientific deployment requires a
firmware-level, hardware-qualified lease/watchdog before that failure mode can
be claimed safe. The proposed wire contract, host integration requirements, and
electrical acceptance evidence are defined in
[`docs/CONTROLLER_WATCHDOG_PROTOCOL.md`](docs/CONTROLLER_WATCHDOG_PROTOCOL.md).
The host includes an explicitly opt-in, production-disqualified experimental v1
implementation for firmware development and bench testing. Legacy behavior is
unchanged by default, and the application does not claim watchdog protection.

Direct GUI startup is blocked by default because it would bypass durable
supervision. For foreground development only, opt in visibly:

```bash
SQUEAKVIEW_ALLOW_INPROCESS_BACKEND=1 uv run squeakview_gui.py
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
bash scripts/launch_operator.sh
```

The supervised backend launches inference first. For free-running capture, the
GUI reports recording after the PyServiceMaker pipeline reaches `PLAYING`. In
triggered mode, it sends `START` only after that readiness signal and reports
recording only after the first controller TTL is observed. The default readiness
timeout is 30 seconds and can be changed with
`SQUEAKVIEW_INFERENCE_READY_TIMEOUT`.

Serial logging may remain enabled during free-running capture, but its data are
auxiliary and no trigger/frame alignment is claimed. Exact TTL alignment is a
mandatory finalization and qualification gate only when both serial logging and
camera triggering are enabled.

The durable backend, not the GUI, owns the required preflight gate. It records a
hashed, structured result in both run status and manifest metadata; a missing
FFmpeg check, skipped/failed preflight, or indeterminate automatic-suspend
policy cannot qualify as scientific production. This also prevents direct IPC
clients from bypassing the check.

Startup and the capture child enforce the same free-space reserve (1 GB by
default). During long runs the child rechecks it every five seconds and requests
a fatal but orderly EOS/container close before the filesystem is exhausted.
Set a larger experiment-appropriate reserve with
`SQUEAKVIEW_MIN_RUN_FREE_BYTES`; `SQUEAKVIEW_STORAGE_CHECK_INTERVAL_S` controls
the check interval. Both resolved values are recorded in `run_manifest.json`.

Only one SqueakView process may own scientific acquisition at a time. The
backend holds an OS-managed lock in the run store from pre-start through final
status persistence. A second GUI receives an explicit refusal; a crashed
process releases the lock automatically, so the presence of the small lock file
alone does not indicate an active run.

The timeout is a maximum wait for pipeline readiness, not a fixed warm-up delay.
Reducing it may reject a healthy pipeline that needs longer to load the engine
and enter `PLAYING`.

The run header remains fixed while the operator workspace is composed from
native Qt cards. Open **Layout** while idle to unlock card dragging/floating,
save the current arrangement immediately, or restore the versioned default;
card dividers remain resizable, and a clean GUI close also saves the current
arrangement. Card movement and reset are locked during startup, recording, and
finalization, while explicit Save remains available. Right-click any card title
to open Qt's unified checklist for showing or hiding every card, including Live
Task State and Operator Events. The capture-health panel
shows recording backlog, camera transport health, frame/drop counters, and the
current stop/finalization stage. Run identity and elapsed state remain visible
while the independent post-run finalizer is working.

The capture child's DeepStream output is also retained in
`diagnostics/deepstream.log` inside each run. This log is capped at 64 MiB so a
verbose debug profile cannot consume storage without bound; the operator log
continues after the cap is reached. With
`SQUEAKVIEW_DEEPSTREAM_DEBUG_PROFILE=1`, the run manifest records the profile
and any NVIDIA frame/component latency output is retained in that diagnostic
log. The environment variables alone do not prove that a custom Service Maker
graph emits latency records. The qualification exception therefore fails
unless a bounded scan finds both NVIDIA's structured `Source id`/`Frame_num`/
`Frame latency` record and its `Comp name`/`Component latency` record, with
finite values, and the diagnostic log was not truncated.
The profile remains qualification-only until paired debug-off/debug-on runs
establish its acquisition overhead. Debug-profile runs are explicitly marked
non-production. Qualify only the debug-on member with the narrow exception:

```bash
.venv/bin/python scripts/qualify_run.py runs/<debug-off-run>
.venv/bin/python scripts/qualify_run.py runs/<debug-on-run> --allow-debug-profile
```

After both matched runs pass `scripts/qualify_run.py`, compare their bounded
recording and system metrics with:

```bash
.venv/bin/python scripts/compare_debug_overhead.py \
  runs/<debug-off-run> runs/<debug-on-run> \
  --thresholds qualification/<approved-overhead-limits>.yaml \
  --output qualification/debug-overhead.json
```

Follow the complete non-automating acquisition and evidence-retention procedure
in [qualification/DEBUG_OVERHEAD_PROTOCOL.md](qualification/DEBUG_OVERHEAD_PROTOCOL.md).
The checker accepts `--case-id` to require both GUI-launched runs to carry the
same backend-enforced qualification binding, and rejects nonterminal or
non-supervised runs.

The comparison fails closed unless the capture, inference, preview, model,
power-mode, application-commit, and complete platform/package/native-plugin
identities match and the manifests explicitly identify opposite debug-profile
states. It compares all five model content
hashes and the engine build identity, verifies that neither cached qualification
summary is stale against its hashed source artifacts, requires the same exact
validated limits file, and requires duration to differ by no more than the
larger of five seconds or 1% of the longer run (both tolerances are CLI options).

Without `--thresholds`, a comparable report remains `incomplete`; the tool does
not invent empirical limits. A version-1 threshold file must set `approved:
true` and provide every reported metric under `metrics`, with at least one
nonnegative `max_increase` or `max_percent_increase` per metric. These bounds
apply only to increases from debug-off to debug-on. An approved profile makes
the report and command exit explicitly passed or failed.

## Live DeepStream pipeline

The validated single-camera path is:

```text
flirspinsrc → GRAY8 caps → tee
  ├─ record queue (120 frames / 4 s minimum, non-leaky)
  │    → x264enc (native GRAY8) → h264parse → mp4mux → raw.mp4
  └─ inference queue (32 frames, downstream-leaky)
       → VIC nvvideoconvert/surface-array NVMM-NV12 → nvstreammux v2
       → nvinfer/TensorRT → Python YOLO26 pose decode
       → CUDA NvDCF → [nvosdbin → nvstreamdemux when preview is enabled]
       → preview admission identity → preview queue (1 frame, downstream-leaky)
       → preview delivery identity → nvunixfdsink
       → Qt nvunixfdsrc preview
```

The FLIR transport is configured with at least 64 host buffers (two seconds at
30 FPS), and the record queue intentionally backpressures rather than discarding
data. Recording backlog is sampled in `diagnostics/recording.csv`; a sustained
three-second backlog fails the run before the four-second queue can fill. If CPU
encoding cannot sustain acquisition, the run therefore stops with explicit
evidence instead of silently producing a plausible but incomplete video.

This contract proves temporal frame completeness, not pixel-lossless storage.
The current `x264enc` profile is bitrate-controlled H.264 with one reference
frame, adaptive quantization disabled, no B-frames, and no lookahead. It is explicitly
recorded as `pixel_fidelity: lossy` in `run_manifest.json`. Do not use it for an
experiment requiring exact source pixels until a separately named lossless/raw
profile has been implemented and throughput-qualified.

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
diagnostics/system.csv          Capture-owned bounded Jetson resource telemetry
diagnostics/errors.csv          Camera gaps, CRC, and metadata failures
diagnostics/camera_runtime.json Camera identity and clock calibration
diagnostics/deepstream.log      Size-bounded capture/DeepStream child output
diagnostics/post_run.log        Finalizer subprocess log
diagnostics/preview_admission*.csv Source IDs entering each leaky preview queue
diagnostics/preview_delivery*.csv  Source IDs delivered after preview shedding
config/task.yaml               Immutable run-local task definition snapshot
```

`frames.csv` is the source of truth for recorded frame identity. It contains
one row per recorded buffer and an `inference_admitted` field, so inference
admission does not require a second frame ledger. `objects.csv` is the single
object-observation table; track summaries are derived from it when analyzed.
Once a run reaches a terminal state, its acquisition, model, Git, and storage
provenance in `run_manifest.json` is immutable. Later bottle entry updates only
the bottle summary, artifact inventory, and manifest update timestamp.

Source, admission, and inference ledgers (`capture_cam*.jsonl`,
`record_admission*.csv`, and `inference/`) are retained after successful
validation as primary scientific provenance. Only transient finalizer progress
state is cleaned up. This costs additional storage but preserves the evidence
needed to reproduce frame reconciliation and diagnose a later integrity issue.
Routine finalization validates every MP4 sample table and parses the complete
H.264 stream to clean EOS without reconstructing pixels. It requires the MP4
sample-size, timing, sample-to-chunk, and parser access-unit counts to agree.
Any structural error or ledger-count mismatch escalates automatically to a
complete hardware decode through Jetson's `nvv4l2decoder`; FFmpeg's explicitly
selected `h264_nvv4l2dec` path remains a fail-closed fallback. Qualification
can explicitly require a complete decode. The resulting authoritative count is reused by
controller alignment so a long recording is not decoded twice. It
requires the decoded count to equal both source and recording-admission counts,
and stores SHA-256 identities for the exact MP4, capture ledger, and admission
ledger set. The exact installed FFmpeg package version is part of device and
campaign provenance. Qualification rehashes those primary artifacts and fails if they
were missing, added, symlinked, or changed after finalization.

Generate a bounded-memory qualification summary after finalization with:

```bash
.venv/bin/python scripts/qualify_run.py runs/<experiment>/<subject>/<run>
```

The checked-in JetPack 7.2.1 limits profile is intentionally marked
measurement-only. Results remain `incomplete` until sustained baseline runs are
used to approve explicit thermal, resource, telemetry-coverage, and recording
limits. Qualification writes `qualification_summary.json`; it never changes the
capture-validity decision in `run_status.json`. The summary hashes the manifest,
status, limits, primary recording artifacts, system telemetry, and every
recording telemetry file it consumed so paired comparisons reject stale or
substituted evidence.
Preview admission and delivery ledgers attribute intentional GUI-preview
shedding by source sequence, camera FrameID, and PTS. Missing preview evidence
fails preview-enabled qualification, but preview loss and preview-observability
failure never invalidate the independent scientific recording.

The full 24-cell short/one-hour/full-duration matrix is defined in
`qualification/matrix.v1.yaml` across inference, preview, and 25 W/MAXN_SUPER
modes. The complete operator procedure, including hardware prerequisites,
measurement-only limits, durable campaign outputs, and recovery from incomplete
cells, is in [qualification/QUALIFICATION_WORKFLOW.md](qualification/QUALIFICATION_WORKFLOW.md).
Enumerate the canonical IDs and create a non-overwriting assignment checklist:

```bash
.venv/bin/python scripts/qualify_matrix.py --list-cases
.venv/bin/python scripts/qualify_matrix.py \
  --init-assignments qualification/assignments.local.yaml
```

Print the next unassigned case, its exact environment, and a copyable terminal
launcher command without changing the checklist or starting hardware:

```bash
.venv/bin/python scripts/qualify_matrix.py \
  qualification/assignments.local.yaml \
  --next-case
```

After each completed run, record its case with `--assign`, then evaluate matrix
coverage with:

```bash
.venv/bin/python scripts/qualify_matrix.py \
  qualification/assignments.local.yaml \
  --assign '<case-id>' '/absolute/path/to/completed/run'
.venv/bin/python scripts/qualify_matrix.py qualification/assignments.local.yaml
```

For the acquisition itself, export
`SQUEAKVIEW_QUALIFICATION_CASE_ID='<case-id>'` before launching the operator.
The backend rejects factor, preview, or exact nvpmodel mismatches before run
creation and persists the binding. `SQUEAKVIEW_QUALIFICATION_MATRIX` may point
to a deliberately selected alternate versioned matrix. Ordinary runs remain
unbound when the case variable is unset; they cannot later be substituted into
the production matrix.

The assignment YAML must be a stable regular file; its read is size-bounded and
duplicate-key rejecting, and it rejects case IDs that are not present in the
selected matrix. The same stable, bounded regular-file policy applies to limits,
matrix, failure-plan, and debug-threshold inputs.

Unassigned cells and runs evaluated against measurement-only limits remain
`incomplete`; factor mismatches or failed run qualification fail the cell. The
matrix also fails closed on dirty/non-production runs and requires every cell
to use the same application commit and complete device/package/native-plugin
identity, including inference-off cells. Power modes must match the exact
labeled `nvpmodel` mode. Every inference-enabled cell must use the same hashed
model/engine identity.

After all cells pass against reviewed limits, create and verify the bounded,
content-addressed campaign inventory described in the workflow with
`scripts/archive_qualification_campaign.py`. The tool does not copy or modify
run data; it rejects incomplete campaigns and verifies the separately copied
archive tree byte-for-byte.

The destructive bench procedure for validating behavior after sudden input
power loss is documented in
[`qualification/HARD_POWER_LOSS_PROTOCOL.md`](qualification/HARD_POWER_LOSS_PROTOCOL.md).
The gated boundary-fault procedure and supported plan schema are documented in
[`qualification/FAILURE_INJECTION_PROTOCOL.md`](qualification/FAILURE_INJECTION_PROTOCOL.md).

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
Video validation normally reads the MP4 sample tables and sends the complete
compressed stream through `h264parse` to clean EOS. It does not reconstruct
pixels. Structural errors or count mismatches automatically escalate to the
isolated hardware decoder and then the FFmpeg fallback. During validation, the
GUI reports processed and expected samples, percentage, rate, elapsed time,
and ETA; do not close the application or move the run directory. Set
`SQUEAKVIEW_VIDEO_VALIDATION_FULL_DECODE=1` for qualification runs that require
every pixel frame to be decoded. Add `SQUEAKVIEW_VIDEO_VALIDATION_AB_VERIFY=1`
to run both full decoders and reject any count disagreement.
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

Do not run offline re-inference on the acquisition Jetson. On a downstream
system with a compatible TensorRT engine, launch it with:

```bash
python -m squeakview.apps.inference.offline RUN_DIR --cfg MODEL_CONFIG \
  --out-dir NEW_DERIVED_DIRECTORY
```

The command keeps the long-run frame ledger in a temporary disk-backed index
and performs one authoritative decode. It does not do a second full-file
FFprobe count pass: the in-pipeline audit requires contiguous decoded ordinals,
rejects frames beyond the ledger immediately, and only publishes a complete
manifest after decoder EOS and an exact final count. A small supervisor process
owns the native Service Maker worker. SIGINT/SIGTERM requests orderly teardown;
if native `stop()`/`wait()` remains blocked for 45 seconds, the supervisor
terminates the isolated worker process group and returns failure. Linux
parent-death containment also kills the worker if the supervisor itself dies.
Derived output never overwrites the acquisition `raw.mp4`, `objects.csv`, or
`keypoints.csv`.
See [data_viz/README.md](data_viz/README.md) for downstream source-data rules.

## Diagnostics

Show optional DeepStream plugin warnings that the GUI normally filters:

```bash
SQUEAKVIEW_SHOW_PLUGIN_WARNINGS=1 bash scripts/launch_operator.sh
```

Request fan control when the current user has the required privileges:

```bash
SQUEAKVIEW_SET_FAN=1 bash scripts/launch_operator.sh
```

Temporarily bypass GUI preflight only for deliberate diagnosis:

```bash
SQUEAKVIEW_SKIP_PREFLIGHT=1 bash scripts/launch_operator.sh
```

Any run started with this bypass is explicitly marked non-production and cannot
pass qualification.

Qualification-only failure injection requires both an explicit safety gate and
a versioned JSON plan. Every such run is labeled `production_eligible: false`
in status and manifest metadata. Never enable this for data collection:

```bash
SQUEAKVIEW_ENABLE_FAILURE_INJECTION=1 \
SQUEAKVIEW_FAILURE_PLAN=/absolute/path/to/failure-plan.json \
bash scripts/launch_operator.sh
```

Supported boundaries are the FLIR source, non-leaky recording queue, encoder,
muxer, filesink/disk, serial controller, and shutdown supervisor. The plan is
rejected if the gate is absent, its schema is unknown, or its target/kind is
unsupported.

The `/dev/full`-based `filesink/disk_full` case is intentionally immediate: it
fails on the selected sink's first write and requires `after_frames: 1` as a
sentinel. It does not simulate one successful recorded frame followed by disk
exhaustion. Use `filesink/error` when a delayed filesink-boundary fault is
required.

To exercise the required preview-off qualification cell while still using the
GUI as the operator interface, launch with `SQUEAKVIEW_DISABLE_PREVIEW=1`. This
omits the preview branch from the capture graph; it does not run the app
headlessly:

```bash
SQUEAKVIEW_DISABLE_PREVIEW=1 bash scripts/launch_operator.sh
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
