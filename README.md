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

The Jetson is the acquisition appliance. Normal shutdown validates the final
MP4 sample count against the durable capture, recording-admission, and
controller totals without rereading the multi-gigabyte ledgers. Frame-level
reconciliation, controller timing alignment, exploratory notebooks,
production-scale analysis, video transcoding, and offline re-inference are
explicit analysis work and do not block safe recording closure.

The current transfer procedure is a manual copy of the complete finalized run
directory. Keep the Jetson original until the DGX copy has been inspected and
validated. See [data_viz/README.md](data_viz/README.md) for the run contract,
manual transfer checklist, DGX directory layout, alignment commands, and
downstream data-handling rules. Resumable `rsync` transfer is flagged for a
future release and is not part of the initial workflow.

## Repository layout

```text
build_engine/                    Project Setup model-builder documentation
configs/                         Runtime tracker configuration
native/flir_gst_source/          Scientific Spinnaker GStreamer source
native/nvdsinfer_custom_impl_yolo/ DeepStream YOLO26 detector parser
resources/project_template/      New-project tasks, qualification, and model seeds
squeakview/apps/inference/        Live and offline PyServiceMaker pipelines
squeakview/apps/operator/         Qt operator GUI and run lifecycle
squeakview/common/                Run, profile, serial, and dashboard utilities
scripts/preflight.sh              Device/model readiness checks
scripts/legacy/align_run_outputs.py      Camera, video, inference, and TTL alignment
data_viz/README.md                Jetson-to-DGX transfer and analysis workflow
data_viz/analysis_demo_viz.ipynb Scientific run analysis and visualization
tests/                            Pure-Python and on-device pipeline tests
```

Application code and scientific data have separate ownership. Each project owns
its `runs/`, `models/`, `model_sources/`, `tasks/`, `profiles/`, and
`qualification/` directories outside this repository. The application install
is replaceable and normal operation does not write to it. New projects receive
independent copies of MouseHouse v2, stock YOLO pose, the default task, and the
qualification templates from `resources/project_template/`.

To build a model, put any additional `.pt` checkpoint and dataset YAML together
in a direct child of the project's `model_sources/` directory. Launch
SqueakView, select the project, and use the pre-acquisition **Project Setup**
screen. Generated device-specific packages are published only into that
project's `models/` directory.

## Install and update layout

Keep the Git checkout in Documents and let SqueakView create projects beside it:

```text
~/Documents/
├── SqueakView/           application code, `.venv`, and native builds
└── SqueakView Projects/  project settings, models, profiles, and runs
```

For a fresh checkout:

```bash
cd "$HOME/Documents"
git clone https://github.com/dlhagger/SqueakView.git
cd SqueakView
uv venv --python 3.12 --system-site-packages
uv sync
bash scripts/setup_jetson.sh
```

Reboot after the one-time setup. To update later, finish any active run, close
SqueakView, and update only the application checkout:

```bash
cd "$HOME/Documents/SqueakView"
git pull --ff-only
uv sync
bash scripts/build_native.sh
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests
bash squeakview.sh
```

Routine application updates do not require a reboot. Git, dependency, and
native-build operations never target `~/Documents/SqueakView Projects`, so they
cannot replace scientific data. See [the project workflow](docs/PROJECT_WORKFLOW.md)
for project creation, model building, rollback, and acceptance checks.

## Jetson device setup

Run the device setup once as the desktop user who will launch SqueakView:

```bash
bash scripts/setup_jetson.sh
```

The script installs NVIDIA's Jetson FFmpeg package (including `ffprobe`) and
the complete GStreamer runtime/plugin and native build prerequisites, builds
both the FLIR GStreamer source and the
DeepStream YOLO parser against the installed CUDA toolkit, and adds that user
to the `dialout` group for RP2040/USB serial access. It also creates
`~/Documents/SqueakView Projects` when absent and leaves an existing directory
untouched. It can be launched from any working directory. Reboot after it
completes because a group change cannot affect an already-running login session.
If the whole setup script is invoked through `sudo`, it deliberately drops back
to the desktop user for native compilation so the checkout does not acquire
root-owned build files. Do not run the application itself with `sudo`.

After the one-time privileged setup, checkout updates rebuild their native
components without sudo through `bash scripts/build_native.sh`.

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

Fresh projects contain model inputs but no generated TensorRT packages. Launch
SqueakView normally:

```bash
bash squeakview.sh
```

After project selection, Project Setup defaults to the project-owned MouseHouse
v2 source package. Its isolated builder worker holds the project's exclusive
writer lock and:

1. Reads standard class names, keypoint names, and keypoint shape from the YAML
   and checkpoint without modifying the YAML.
2. Passes the YAML to `Ultralytics.export(data=...)`.
3. Exports an NMS-free, end-to-end FP16 or FP32 TensorRT engine on the target
   Jetson using Ultralytics `nms=False`.
4. Strips the Ultralytics metadata prefix to produce the raw TensorRT plan
   expected by DeepStream.
5. Validates ONNX and TensorRT input/output shapes.
6. Writes and validates a schema-3 model package with a schema-2 pose sidecar.
7. Exits and releases its CUDA context before the acquisition supervisor starts.

Generated layout:

```text
$SQUEAKVIEW_PROJECT/models/<model_name>/
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
export SQUEAKVIEW_PROJECT="$HOME/Documents/SqueakView Projects/My Project"
PYTHON_BIN=.venv/bin/python \
SQUEAKVIEW_MODEL_NAME=<model_name> \
bash scripts/preflight.sh
```

You can provide the config directly instead:

```bash
PYTHON_BIN=.venv/bin/python \
DS_CFG="$SQUEAKVIEW_PROJECT/models/<model_name>/configs/<model_name>.txt" \
bash scripts/preflight.sh
```

Run the same check remotely:

```bash
ssh -t jetson@<jetson-host> 'cd ~/Documents/SqueakView && export SQUEAKVIEW_PROJECT="$HOME/Documents/SqueakView Projects/My Project" && PYTHON_BIN=.venv/bin/python SQUEAKVIEW_MODEL_NAME=<model_name> bash scripts/preflight.sh'
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
scientific acquisition; rebuild it from the Project Setup screen.

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
bash squeakview.sh
```

The small project chooser appears on every launch and preselects the most
recently opened valid project. You can create a project, browse to another one,
or confirm the preselected project. It creates the default projects parent when
missing and never rewrites an existing one. Project Setup then lets you build or
select the project default model before the supervisor, experiments, or subjects
start. For automation or development, pass an existing project explicitly:
`bash squeakview.sh /absolute/project/path`; Project Setup still appears.

This detached launcher is the production path. It starts a Qt-free supervisor,
which owns the controller, capture, run lock, and finalization while launching
and monitoring the GUI. If the GUI exits or its main-loop heartbeat expires
during a run, acquisition continues under the same supervisor with the same
controller, files, and counters. The supervisor replaces the failed GUI with
bounded restart backoff and restores its configuration and current run state.
Terminal or VS Code loss
does not end the supervised session. Separate bounded GUI and supervisor logs
are written under `${XDG_STATE_HOME:-$HOME/.local/state}/SqueakView/logs/`;
each mirror is capped at 32 MiB and partial
lines are capped at 64 KiB.
Stop requests are acknowledged immediately by the supervisor. Capture closure
and post-run analysis continue in a supervisor-owned worker while the responsive
GUI receives lifecycle events and heartbeats; run duration therefore cannot
turn a slow alignment pass into an IPC timeout or a false GUI-loss failure.
The launcher reports success only after the supervisor has authenticated the
GUI's IPC connection; an early GUI failure or connection timeout returns
nonzero with the supervisor log path. The bounded wait defaults to 35 seconds,
slightly longer than the supervisor's 30-second GUI connection deadline,
and can be adjusted with `SQUEAKVIEW_LAUNCH_TIMEOUT_S`.

The presentation lease is renewed only by a Qt-main-loop timer, so a frozen GUI
is replaced even if its background IPC thread still exists. This lease does not
own acquisition or close scientific writers. The DeepStream capture
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

The complete project layout, model-build handoff, path ownership rules, and
short on-device acceptance pass are in
[`docs/PROJECT_WORKFLOW.md`](docs/PROJECT_WORKFLOW.md).

Direct GUI startup is blocked by default because it bypasses durable
supervision. For foreground development only, opt in visibly and provide an
explicit project:

```bash
SQUEAKVIEW_PROJECT="/absolute/path/to/My Project" \
SQUEAKVIEW_ALLOW_INPROCESS_BACKEND=1 \
.venv/bin/python squeakview_gui.py
```

Scientific operation must use `bash squeakview.sh`. Serial capture defaults to
`/dev/ttyACM0` at 115200 baud; choose the actual device shown by
`ls /dev/ttyACM*`.

Useful path overrides:

```bash
export SQUEAKVIEW_DEEPSTREAM_SDK=/opt/nvidia/deepstream/deepstream
export SQUEAKVIEW_PROJECTS_DIR="$HOME/Documents/SqueakView Projects"
bash squeakview.sh /absolute/path/to/a/project
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
camera triggering are enabled under the legacy controller protocol. Protocol v2
is the production default for controller-backed GUI runs. A deliberate
`SQUEAKVIEW_CONTROLLER_PROTOCOL=legacy` override remains available only for
non-production bench development with older firmware.

Protocol v2 is negotiated before the clock preflight and capture launch. The
idempotent negotiation supports a freshly booted v1 controller and a controller
that remained in v2 after an earlier run; idle `TIME_SYNC` and `SET_RTC`
responses then use reliable v2 framing. A new run sends a unique `TIME_SYNC`
challenge after `PROTO,2`. Its reliable idle-controller reply must echo both the
request ID and host timestamp, pass CRC/schema checks, and be durably saved
before negotiation completes. That reply establishes the new run's sequence
baseline. Partial packets and queued messages preceding this boundary are
counted separately as `startup_discarded_fragments`/`startup_discarded_frames`;
they cannot establish or acknowledge the new journal. The verified boundary is
recorded as `startup_handshake` in the controller transport summary. This uses
existing firmware commands and requires no firmware update.

An existing run's reconnect continues from its durable acknowledgement watermark
and requests missing records; it never resets its sequence baseline. CRC errors,
sequence gaps, and reboot detection after the startup boundary retain their
normal validation and recovery behavior.

Controller frames are CRC checked and stored, with an `fsync`, in
`_metadata/diagnostics/controller_v2.jsonl` before SqueakView sends a cumulative
`ACK_EVENTS`. The controller can then release the acknowledged RAM records.
Camera timing uses `CAMERA_EPOCH`, periodic `CAMERA_CHECKPOINT`, and
`CAMERA_STOP` rather than two serial lines per frame. A transport integrity
fault, conflicting replay, unrecoverable sequence gap, or controller reboot
during a run fails the run closed. `sensors.csv` contains decoded events with per-record timing; the JSONL journal is the v2 source of truth.

The host indexes v2 identities and sequence gaps in a disposable SQLite cache
beside the journal (`diagnostics/.controller_v2.jsonl.index-*.sqlite3`), with a
bounded 512 KiB page cache. This keeps long experiments from retaining every
controller payload and sequence number in RAM. The index is rebuilt from the
JSONL on every open and removed on normal close; stale index files after a crash
are ignored and may be removed once the run is inactive. They are not scientific
evidence. Only the canonical JSONL flush and `fsync` authorize acknowledgements.

Run the physical-controller qualification before production use:

```bash
uv run mousehouse-protocol-v2-test /dev/ttyACM0 \
  --fps 30 --duration 60 --withhold-acks 5 \
  --request-resend --status --correct-clock --verbose \
  --output /tmp/mousehouse-protocol-v2-test
```

`--correct-clock` explicitly authorizes one idle `SET_RTC` if the initial clock
burst is outside tolerance. The tool always collects and verifies a fresh
seven-sample burst after correction before it may start acquisition.

The controller remains in v2 until it reboots. SqueakView safely rejoins that
mode on subsequent GUI runs, so a reboot between ordinary runs is unnecessary.
Reboot after a standalone qualification before using a host build that predates
this idempotent negotiation support.
The separate `--exercise-overflow-failsafe` mode intentionally fills the queue,
invalidates and stops the run, and also requires a reboot. Do not use it during
an experiment.

The durable backend, not the GUI, owns the required preflight gate. It records a
hashed, structured result in both run status and manifest metadata; a missing
FFmpeg check, skipped/failed preflight, or indeterminate automatic-suspend
policy cannot qualify as scientific production. This also prevents direct IPC
clients from bypassing the check.

For every controller-backed experiment, startup also validates the MouseHouse
controller RTC through the already-open serial transport before capture is
launched or controller `START` is sent. The controller uses the PCF8523 as its
sole RTC. Never attach a DS3231 at the same time: both chips have the fixed I²C
address `0x68`. SqueakView first requires the Jetson to report NTP
synchronization, then collects seven `TIME_SYNC` exchanges approximately 100 ms
apart. All seven must report `RTC_VALID`; the median offset of the three
lowest-round-trip samples must be within ±1.5 seconds.

If the clock is invalid or outside tolerance, acquisition remains blocked. RTC
correction is available only when **Allow one pre-run controller RTC correction**
was explicitly enabled in run configuration. SqueakView then sends one
whole-second `SET_RTC`, requires its acknowledgement, and validates a completely
new seven-sample burst. An already-passing clock is never corrected. Validation
and correction are prohibited while a controller session or feed is active;
`DEVICE_BUSY`, timeouts, malformed responses, and NACKs all fail closed. The
complete record is saved as `diagnostics/clock_validation.json`, referenced by
run status and `run_manifest.json`, and summarized in the GUI's Clock Preflight
section.

The RTC establishes UTC only before the experiment. During acquisition, event
timestamps remain anchored to the RP2040 monotonic clock; SqueakView performs no
continuous RTC synchronization and does not alter session timestamp formats.

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
export SQUEAKVIEW_PROJECT="/absolute/path/to/My Project"
.venv/bin/python scripts/qualify_run.py "$SQUEAKVIEW_PROJECT/runs/<debug-off-run>"
.venv/bin/python scripts/qualify_run.py "$SQUEAKVIEW_PROJECT/runs/<debug-on-run>" --allow-debug-profile
```

After both matched runs pass `scripts/qualify_run.py`, compare their bounded
recording and system metrics with:

```bash
.venv/bin/python scripts/qualification/compare_debug_overhead.py \
  "$SQUEAKVIEW_PROJECT/runs/<debug-off-run>" "$SQUEAKVIEW_PROJECT/runs/<debug-on-run>" \
  --thresholds "$SQUEAKVIEW_PROJECT/qualification/<approved-overhead-limits>.yaml" \
  --output "$SQUEAKVIEW_PROJECT/qualification/debug-overhead.json"
```

Follow the complete non-automating acquisition and evidence-retention procedure
in [docs/qualification/DEBUG_OVERHEAD_PROTOCOL.md](docs/qualification/DEBUG_OVERHEAD_PROTOCOL.md).
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

New runs expose five primary files (additional cameras retain their own
`raw_camN.mp4` and `frames_camN.csv`):

```text
raw.mp4                         Camera recording; video timing maps to frames.csv
frames.csv                      One timestamped row per recorded frame
sensors.csv                     Box events, sensor streams, and synchronization records
detections.csv                  One timestamped object/pose observation per row
weigh_in.csv                    Timestamped bottle and mouse weight measurements
_metadata/                      Configuration, provenance, validation, and recovery evidence
```

Every primary CSV row carries its own timing. `frames.csv` preserves native
camera timestamps, host timestamps, frame identity, and the movie frame index.
`detections.csv` includes that frame's timestamps directly, together with tracking
and bounding-box fields and named `<keypoint>_x_px`, `<keypoint>_y_px`,
`<keypoint>_confidence`, and `<keypoint>_visible` columns. Tracking-only observations
have empty keypoint columns when no pose is available. A frame without an object
observation still has its row in `frames.csv`.

`sensors.csv` retains the decoded serial event fields and host receipt clocks,
and adds controller time, boot/session identity, and reliable-record sequence.
Sensor events, task activity, synchronization markers, and controller setup/
protocol messages all stay in this table. Non-event messages use `context` and
`reason` for their payload instead of putting settings or build text into time
and measurement columns. Unstructured firmware text uses `CONTROLLER_LOG` as
its event type. `rawLine` preserves the complete original message. Clock-sync,
clock-status, camera-anchor, and bottle-state records use their own timestamp
schemas so host timestamps cannot be mistaken for controller microseconds.
CSV and terminal controller output share one event-record builder. Existing
sensor-event fields pass through unchanged; other messages use the same logger
with their content in `context`/`reason`. Controller terminal output uses the
same ten event columns as the first ten columns of `sensors.csv`, including startup text, protocol replies, outgoing
commands (`HOST_COMMAND`), and serial-service messages (`HOST_LOG`). Existing
RX/TX tags remain in front of each record. CSV quoting protects commas and
quotes; embedded newlines are escaped for one terminal line per message. The
presentation formatter does not change transmitted commands or raw protocol
records, so this formatting update requires no firmware flash.

`weigh_in.csv` records bottle identity, side, fluid, initial/final weights, mouse
weight, notes, and host entry time. It also holds the complete bottle-swap
history: installations/removals, controller responses, pause/resume boundaries,
cumulative lick counts, shared observation gaps, pending replacements, failed
operations, interrupted swaps, and weight corrections. `record_type` distinguishes
`measurement`, `bottle_state`, `swap_gap`, `swap_state`, and `swap_event` rows.
Frequently used fields have their own columns; `details_json` preserves the
complete structured record for recovery and less common fields. All rows have
host timestamps, and controller-confirmed event rows preserve their direct
controller time as `direct_boundary`. Changed measurements append new records;
unchanged values keep their original entry timestamps. Manual entry time does
not claim to be the physical time the weighing happened.

Every row is recorded with its native clocks. Camera and detection rows retain
camera timestamps and host UTC/monotonic timestamps; sensor records retain their
direct controller timestamps. Stop does not rewrite these CSVs.

Controller-time mappings are resolved from retained synchronization evidence by
`squeakview.common.row_timing.iter_timestamped_rows(run_dir, "frames.csv")`
(or `detections.csv`, `sensors.csv`, `weigh_in.csv`). Camera-trigger anchors are
`exact_anchor`; intervening frame times are `piecewise_interpolated`. Host-only
entries use the preflight midpoint as a labelled `host_clock_estimate`, which
does not correct subsequent host wall-clock steps or clock drift. Without a
usable anchor the mapping is `unavailable`; native timestamps remain intact.
The v2 analysis loader and pose-table reader handle both native tables and
historical CSVs with materialized controller columns. The run layout declares
`timing_contract: native_with_sync_anchors_v1`.

Every MP4 frame retains video presentation timing and maps by frame index to
`frames.csv`; controller timestamps are not embedded in the MP4.

`_metadata/run_manifest.json` contains acquisition provenance, lifecycle status,
and validation results. Bottle setup, measurements, and swap recovery state live
in `weigh_in.csv`; separate bottle setup/summary JSON caches are no longer written.
`_metadata/` also retains `config/`, `diagnostics/`, and recovery ledgers. These support reliable controller acknowledgements,
recording validation, and crash recovery. New runs no longer create separate
`objects.csv`, `keypoints.csv`, `serial.csv`, or `bottle_measurements.csv` streams.
Older runs retain their existing paths and remain readable without migration.

`frames.csv` is written live by the FLIR source. Native timestamps are checked
while recording; invalid/regressed camera times fail acquisition. Shutdown
compares final source, frame-manifest, admission and MP4 sample counts, checks
cumulative camera faults, and persists the result. It does not create a timing
SQLite index, scan primary CSVs, or archive recovery ledgers. The preflight check
rejects an older FLIR plugin lacking live timestamp validation: rebuild with
`bash scripts/build_native.sh` before using this version. No firmware update is
required for this change.

Acquisition provenance remains immutable. Later weigh-ins update measurement
records and bottle metadata after the bounded Stop operation releases ownership.
Recovery ledgers remain plain under `_metadata/` by default. With SqueakView
closed and exclusive ownership of a completed run, optional maintenance is:

```bash
.venv/bin/python -m squeakview.apps.inference.post_run /path/to/run --compact-metadata
```

This verifies gzip round trips before removing originals. Small retained final
records allow subsequent fast checks without decompressing entire ledgers.
A failed archive is reported separately from capture validity. Historical
archives lacking those summaries require explicit `--full-analysis` for a
recovery audit. Full analysis and `--materialize-timestamps` intentionally scan
or rewrite data and require exclusive ownership; they are never automatic Stop
work. `--align` requests detailed analysis explicitly, including legacy runs.

Routine finalization reads the bounded MP4 sample tables without parsing or
decoding the complete H.264 stream. It requires the sample-size, timing, and
sample-to-chunk totals to agree with the live frame manifest, source ledger,
recording-admission ledger, and controller count. Full decode remains an
explicit qualification option, not an automatic shutdown fallback.

Generate a bounded-memory qualification summary after finalization with:

```bash
export SQUEAKVIEW_PROJECT="/absolute/path/to/My Project"
.venv/bin/python scripts/qualify_run.py "$SQUEAKVIEW_PROJECT/runs/<experiment>/<subject>/<run>"
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
`$SQUEAKVIEW_PROJECT/qualification/matrix.v1.yaml` across inference, preview, and 25 W/MAXN_SUPER
modes. The complete operator procedure, including hardware prerequisites,
measurement-only limits, durable campaign outputs, and recovery from incomplete
cells, is in [docs/qualification/QUALIFICATION_WORKFLOW.md](docs/qualification/QUALIFICATION_WORKFLOW.md).
Enumerate the canonical IDs and create a non-overwriting assignment checklist:

```bash
.venv/bin/python scripts/qualification/qualify_matrix.py --list-cases
.venv/bin/python scripts/qualification/qualify_matrix.py \
  --init-assignments "$SQUEAKVIEW_PROJECT/qualification/assignments.local.yaml"
```

Print the next unassigned case, its exact environment, and a copyable terminal
launcher command without changing the checklist or starting hardware:

```bash
.venv/bin/python scripts/qualification/qualify_matrix.py \
  "$SQUEAKVIEW_PROJECT/qualification/assignments.local.yaml" \
  --next-case
```

After each completed run, record its case with `--assign`, then evaluate matrix
coverage with:

```bash
.venv/bin/python scripts/qualification/qualify_matrix.py \
  "$SQUEAKVIEW_PROJECT/qualification/assignments.local.yaml" \
  --assign '<case-id>' '/absolute/path/to/completed/run'
.venv/bin/python scripts/qualification/qualify_matrix.py "$SQUEAKVIEW_PROJECT/qualification/assignments.local.yaml"
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
`scripts/qualification/archive_qualification_campaign.py`. The tool does not copy or modify
run data; it rejects incomplete campaigns and verifies the separately copied
archive tree byte-for-byte.

The destructive bench procedure for validating behavior after sudden input
power loss is documented in
[`docs/qualification/HARD_POWER_LOSS_PROTOCOL.md`](docs/qualification/HARD_POWER_LOSS_PROTOCOL.md).
The gated boundary-fault procedure and supported plan schema are documented in
[`docs/qualification/FAILURE_INJECTION_PROTOCOL.md`](docs/qualification/FAILURE_INJECTION_PROTOCOL.md).

The FLIR chunk `FrameID` is stored as `camera_frame_id` and increments for every
acquired image. The legacy aligner derives a per-run offset between that hardware
frame sequence and RP2040 `CAMERA_HIGH` counts. Protocol-v2 runs preserve camera
epoch/checkpoint/stop anchors for later pandas-based alignment on the analysis
device without emitting per-frame controller edges.

On stop, SqueakView asks the controller to stop before draining DeepStream and
keeps the serial reader open for final acknowledgements. The FLIR source finishes
delivering an in-flight frame before emitting EOS from its streaming thread;
this also applies when the camera is running freely with triggering off. An idle
triggered source finishes EOS at its bounded image-read timeout. After the capture
process confirms EOS and closes `raw.mp4`, the shutdown validator reads the
last monotonic indices from the capture and non-leaky recording-admission
ledgers, reads the MP4 sample-size/timing/sample-to-chunk/chunk-offset tables,
and requires all totals (plus the controller total when triggered) to agree.
These are bounded tail/table reads; normal shutdown does not hash or decode the
complete video and never rebuilds `frames.csv`. Legacy triggered runs then
generate the controller/camera alignment summary automatically from the
already-written manifest and serial ledger. Protocol-v2 runs validate recording
and acquisition counts on-device and leave detailed controller/camera alignment
to the analysis workflow. Structural errors and count mismatches fail directly.
Set
`SQUEAKVIEW_VIDEO_VALIDATION_FULL_DECODE=1` for qualification runs that require
every pixel frame to be decoded. Add `SQUEAKVIEW_VIDEO_VALIDATION_AB_VERIFY=1`
to run both full decoders and reject any count disagreement.
Bottle intake is calculated as initial minus final weight. A final weight above
the initial weight is saved but shown as a plausibility warning.

The **Weigh-In** panel groups the left/right bottle measurements and an optional
mouse body weight in grams. Before **Start Recording**, enter the fluid types,
initial bottle weights, and mouse weight. Start validates the form and saves
these entries automatically with the new run; a separate Save is unnecessary.
The form is locked during startup, recording, and finalization. After **Stop
Recording** finishes, enter final bottle weights and click **Save weights**.
The panel shows the current stage and next action without warning borders.
Live lick counts, swap controls, and history are under **Bottle tools and history**.

With the updated controller firmware, lick sensing automatically recovers from
chip faults, configuration mismatches, or a contact held active for 5 seconds.
Autoconfiguration stays enabled; touch/release thresholds remain 2/0 and
debounce remains 0/0. Recovery preserves cumulative counts, pauses both lick
channels, and resumes only after configuration verification, settling, and
confirmed release. Camera recording and the rest of the task continue.
Retries have a cooldown and a bounded budget; repeated failures leave lick
sensing unavailable until a successful manual bottle recalibration or a new
recording. These limits still require physical-rig checks before production use.

The Weigh-In card shows recovering/unavailable status even when bottle tools
are collapsed. Recovery is separate from bottle-swap state and never changes
bottle identities. Timestamped `LICK_RECOVERY_*` events in `sensors.csv` record
per-side absolute counts, observation-gap duration, attempt count, and recovery
ID; `MPR121_DIAG` rows preserve signal and register diagnostics. Updating the
application alone does not enable recovery: flash the matching task firmware
onto each RP2040 controller after stopping recording. Older firmware remains
readable but does not provide the new recovery status.
Status polls preserve
unsaved entries, and a new recording cannot clear unsaved final weights. Mouse
weight must be a positive finite number; empty fields remain optional. Check
the retained initial measurements before each new run.
`weigh_in.csv` contains bottle and mouse weight measurements with entry timestamps.
The manifest includes a derived `bottles` snapshot. Historical runs with
`bottle_setup.json` and `bottle_summary.json` remain supported.

During a running experiment, use **Swap Bottles…** in Weigh-In to replace the
left, right, or both bottles. Select the sides, then click **Pause Both Lick
Channels** and wait for confirmation before touching the bottles. Enter the
selected bottles' final weights, install replacements, select or type each new
fluid, and enter each replacement's initial weight. Clear hands from the spouts
and click **Recalibrate & Resume**. Both lick channels resume only after the
shared MPR121 finishes calibration and passes readiness checks. Camera recording,
feeding, nose pokes, and the experiment clock continue throughout the swap.

Session lick counters never reset. Each replaced bottle starts a fresh bottle
counter; an unchanged bottle keeps its fluid, weight, history entry, and count.
The pause/calibration interval is an observation gap for both channels. Expand
**Bottle history** to view completed and current bottles, intake, lick counts,
and installation/removal boundaries. **Older**, **Newer**, and **Latest** browse
all completed entries in pages of 50. Select a completed entry and use
**Edit Selected Final Weight…** to add or correct a weight later. A missing final
weight does not prevent resuming. After stopping, save final weights for the
current bottles using the normal Weigh-In fields.

Transient stop/configuration register verification failures are retried
automatically, up to three attempts within the same ten-second calibration
deadline. Both channels stay paused throughout; readiness and released-spout
checks must still pass before counting resumes. Persistent calibration failures
leave both channels paused; retain the entered replacement values and retry.
If acknowledgement is uncertain, use **Check Controller
Status**; a pause request can be retried with **Check / Retry Pause**. An
unconfirmed state is not permission to handle bottles. **Check Status** can also
retry the capability check after a transient connection failure. Swapping requires
compatible reliable-v2 controller firmware. Mouse-weight saves remain available
without changing swapped bottle metadata.

Bottle swaps and behavioral recording also work with **camera triggering off**.
With serial enabled, SqueakView sends `START_BEHAVIOR` and waits for its reliable-v2
acknowledgement before publishing the run as recording. The controller runs the
normal behavioral tasks, lick detection, and bottle controls while the camera
free-runs; it emits no camera TTL pulses. Triggered runs retain the existing
`START,<fps>` activation. `STOP` ends either kind of controller session. A
trigger-off run validates video against its capture/frame ledgers without
requiring camera-TTL alignment or comparing the zero TTL count to video frames.

New runs store the complete bottle-swap record and recovery state in
`weigh_in.csv`; they do not create separate `bottle_history.json` or
`bottle_history_events.jsonl` files. Analysis and recovery reconstruct bottle
identities and shared gaps from the CSV. Historical runs using those JSON/JSONL
files remain readable. Pre-consolidation new-layout history is imported when
reopened without deleting the original files.

To rerun the legacy bounded-memory recovery reconstruction explicitly:

```bash
.venv/bin/python -m squeakview.apps.inference.post_run /path/to/run \
  --camera-count 1 --full-analysis --enable-infer --align
```

The command rebuilds `frames.csv` from recovery ledgers and performs the deeper
inference/object reconciliation. Normal runs do not need this command because
capture writes `frames.csv` live and Stop generates `alignment_summary.json`.

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

During supervised acquisition the GUI automatically records stall evidence in
`diagnostics/gui_stalls.jsonl` inside the run. A background monitor captures
Python thread stacks after 3, 6, and 9 seconds without GUI progress or heartbeat
transmission, plus recovery and connection-loss evidence. Records include UTC
time, hostname, process ID, run ID, and heartbeat timing. Compare `gui_stall`
with `heartbeat_send_stall` to distinguish event-loop delays from sender delays.
Stacks capture call sites, not local variable values; native calls may require
additional diagnosis. A native extension holding Python's GIL indefinitely can
also prevent this Python monitor from running.

Diagnostic output is best-effort, limited to 2 MiB per run, and written only by
a daemon thread; acquisition and scientific logging never wait for it. There
are no per-frame diagnostic writes or video/ledger scans. Storage failure or
exhausting the cap can leave evidence unavailable. This instrumentation does
not stop acquisition. The GUI heartbeat timeout is 30 seconds;
stall diagnostics begin after 3 seconds, before that timeout.

GPU preview rendering runs in a separate disposable process. A renderer crash
or blocked driver teardown disables preview without crashing Qt or stopping
recording. Display events are buffered with fixed bounds before entering Qt;
display backlog can be shed while canonical serial/video records remain intact.
Console output also runs off the Qt thread. A missing log consumer, diagnostic
file failure, or malformed child output cannot block draining capture stdout.
In the GUI process, cyclic garbage collection runs on the Qt thread every five
seconds. This prevents automatic collection on an IPC worker from destroying
Qt objects and timers on the wrong thread; acquisition-process GC is unchanged.

Repeated readiness/shutdown messages confirm the prepared run directory without
closing and reopening its serial CSV/journal. Serial persistence owns its files
until its producers and consumers have finished. Native pipeline shutdown and
resource cleanup have bounded waits, with explicit errors when a real acquisition
failure prevents a complete drain. GUI recovery does not invoke that shutdown.

Controller duplicate/conflict detection uses a disposable SQLite index beside
the canonical JSONL journal, with a 512 KiB page cache rather than retaining all
events in Python memory. It still checks arbitrarily old duplicates and sequence
gaps. Each record reaches canonical JSONL `fsync` before its acknowledgement
watermark advances. The `.controller_v2.jsonl.index-*.sqlite3` cache is rebuilt
from JSONL on every open and removed on normal close; a cache left by a killed
process is disposable and is never used as scientific evidence.

GUI, preview, capture and supervisor processes install independent crash logs
under the user log directory (`*.crash.log`). Native fatal signals record Python
stacks directly through a dedicated descriptor; unhandled Python/thread exceptions
are recorded too. SIGKILL, including an OOM kill, cannot run a handler: compare
supervisor exit/signal evidence with kernel messages. These diagnostics cannot
guarantee recovery from power loss, a killed supervisor, GPU/driver failure in
capture itself, or a disk that cannot accept data. Genuine scientific persistence
and acquisition faults still fail visibly rather than reporting invented success.

Collect bounded evidence on an affected rig without stopping any process:

```bash
.venv/bin/python -m squeakview.tools.collect_crash_report \
  --project "/absolute/project/path" --output /tmp/squeakview-crash-report.json
```

The report contains recent log tails, up to three run-status snapshots, recovery
and stall evidence, current memory information, and accessible kernel OOM/crash
messages. Missing files or journal permissions are recorded in the report. Use
a new output filename for each report.

Show optional DeepStream plugin warnings that the GUI normally filters:

```bash
SQUEAKVIEW_SHOW_PLUGIN_WARNINGS=1 bash squeakview.sh
```

Request fan control when the current user has the required privileges:

```bash
SQUEAKVIEW_SET_FAN=1 bash squeakview.sh
```

Temporarily bypass GUI preflight only for deliberate diagnosis:

```bash
SQUEAKVIEW_SKIP_PREFLIGHT=1 bash squeakview.sh
```

Any run started with this bypass is explicitly marked non-production and cannot
pass qualification.

Qualification-only failure injection requires both an explicit safety gate and
a versioned JSON plan. Every such run is labeled `production_eligible: false`
in status and manifest metadata. Never enable this for data collection:

```bash
SQUEAKVIEW_ENABLE_FAILURE_INJECTION=1 \
SQUEAKVIEW_FAILURE_PLAN=/absolute/path/to/failure-plan.json \
bash squeakview.sh
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
SQUEAKVIEW_DISABLE_PREVIEW=1 bash squeakview.sh
```

Run the software checks without changing the environment (Python tests plus
camera-free native EOS and pose-parser tests):

```bash
.venv/bin/python scripts/test.py
```

The command uses standard-library `unittest`; pytest is not required. An SDK
build environment with CMake and the native headers is required. Use
`--python-only` for an explicitly partial check, or `--pattern test_name.py` to
focus Python discovery. Empty discovery and uncaught worker-thread exceptions
fail the command. No recording, firmware flashing, CUDA inference, or hardware
qualification is performed.

The upgraded NVIDIA cuSPARSELt 0.8.1 ARM64 wheel has a platform-tag mismatch
inside its `WHEEL` metadata. The test command reports this as a warning only
when it is the sole `uv pip check` error, Torch requires that exact version,
and the installed library, header and metadata match the reviewed upstream
hashes in `squeakview/common/dependency_health.py`. It does not rewrite the
wheel, environment or lockfile. A changed payload or any other dependency error
fails. Plain `uv pip check` and `uv sync --check` can still report the upstream
metadata issue; this exception supplies no evidence of GPU compatibility.

## Git policy

Commit source code, tests, documentation, notebooks without saved outputs,
configuration, native build recipes, and shipped project-template model inputs.
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
