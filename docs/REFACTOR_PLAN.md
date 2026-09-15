# SqueakView refactor plan

This plan targets the validated Jetson Orin Nano Super deployment on JetPack
7.2.1 and preserves the scientific acquisition contract while the application
is decomposed. NVIDIA currently lists JetPack 7.2.1 as Jetson Linux 39.2.1,
Ubuntu 24.04, CUDA 13.2.1, cuDNN 9.20.0, TensorRT 10.16.2, and DeepStream 9.1.
DeepStream 9.1 itself was built on the JetPack 7.2 GA/r39.2 baseline, so run
provenance must record exact installed package versions rather than assuming
that the marketing version identifies the complete runtime.

## Non-negotiable scientific contract

The recording path remains upstream of inference and preview:

```text
flirspinsrc -> GRAY8 -> tee -> non-leaky record queue -> x264enc -> MP4
                         `-> leaky inference/preview work
```

- The recording queue must remain non-leaky and must backpressure. A slow
  encoder causes a visible, fatal acquisition fault before the bounded queue
  fills; it must never silently discard a frame.
- Normal shutdown must reconcile the encoded MP4 sample total with the durable
  capture, recording-admission, and controller totals in bounded time. Full
  row-level camera-ID, trigger, inference, and object reconciliation remains a
  required explicit analysis/qualification step rather than blocking safe MP4
  closure.
- Inference and preview may shed work without changing the ground-truth
  recording. Their skips remain measured and attributable by source frame ID.
- Refactors must preserve shutdown order: stop new triggers/acquisition, send
  EOS, drain and finalize the recording container, then validate artifacts.
- Pipeline changes are accepted only after unit tests, failure-injection tests,
  an on-device short run, and a sustained on-device validation run.

Frame completeness and pixel fidelity are separate requirements. This project's
scientific requirement is complete temporal sampling; lossy pixel encoding is
an accepted, explicit constraint. The current
bitrate-controlled H.264 recording can contain every camera frame while still
being lossy at the pixel level. Do not silently replace it with a lossless/raw
profile or segmented recording. Any future change to that decision requires a
new explicit scientific requirement and a separately qualified format.

## Stage 0: freeze behavior and evidence

Status: repository implementation complete; hardware qualification remains in
progress. The shared capture policy, explicit non-leaky queue/QoS
contract, strict ledger parsers, nonzero recording validation, retained source
provenance, controller handshake, process-exit gate, device/native-binary
identity, concurrent atomic metadata writes, fail-closed serial/metadata I/O,
and restart-safe artifact promotion are implemented. Versioned failure plans
are rejected unless the explicit qualification gate is enabled, label the run
non-production, and have been exercised during local development on-device at
the source, recording
queue, encoder, muxer, filesink, and physical disk-write boundaries. The source
fault after 12 frames returned nonzero with exactly 12 source, admission, and
MP4 frames. Encoder, muxer, and filesink faults returned nonzero and left the
container explicitly invalid. The immediate-only `/dev/full` sink
(`filesink/disk_full` with the required `after_frames: 1` sentinel) produced a
propagated `No space left` error and nonzero close; it does not represent one
successfully written frame before exhaustion. The 250 ms queue stall reached the
90-frame fatal backlog threshold with zero telemetry evictions and exited with
code 4 under application control; native drain is deliberately bypassed after
this already-invalid integrity failure because the blocked Service Maker stop
call can retain the Python runtime indefinitely. Explicitly gated serial read,
write, and ledger-write injection plus controller-ACK, capture-exit, and
finalizer-timeout shutdown injection are implemented and unit-tested; their
on-device qualification, hard-power-loss testing, and sustained hardware
qualification remain open.

Those numerical fault-injection observations are operator-reported development
notes. Their raw run directories are not committed and do not count as retained
qualification evidence. Repeat them under the checked-in protocol and retain
immutable run IDs, report hashes, and archive location for release qualification.

Triggered, serial-enabled acquisition retains all evidence required for
controller/camera alignment, but that multi-million-row analysis no longer
blocks the Stop operation. It remains mandatory before scientific
qualification. Model provenance hashes the
portable manifest, pose sidecar, ONNX source, DeepStream configuration, and
generated engine independently. Runtime preparation additionally binds the
exact localized config/pose schema/class labels/keypoint labels plus the
package ONNX, engine, and selected parser; all seven are rehashed before spawn
and again at structured model-loaded readiness before a triggered `START`.
Run-local effective files are included in qualification source evidence, and a
missing, divergent, or modified effective-runtime identity fails qualification.
Device provenance records the installed
JetPack/L4T/DeepStream/CUDA/cuDNN/TensorRT package versions when available.

Qualification requires a clean commit, complete required component package
versions, and size/SHA-256 identities for the native FLIR and inference plugins.
The `nvidia-jetpack` convenience meta-package is optional when its component
packages provide exact evidence. The durable backend owns preflight, so direct
IPC cannot bypass it; matched hashed status/manifest evidence must prove an
actual GOP-sized H.264 decode through Jetson's explicit `nvv4l2decoder`
path, FFmpeg/FFprobe fallback availability, and disabled automatic AC suspend.
The short hardware-decoder exercise runs before every recording; readiness
evidence is never reused across scientific runs. The exact FFmpeg package
version is part of required platform provenance. The selected task YAML is
copied atomically and boundedly to `config/task.yaml`, with its source path,
run-relative path, size, and hash.

The checked-in `qualification/HARD_POWER_LOSS_PROTOCOL.md` defines the physical
test and retained evidence. The evaluator now treats every nonterminal persisted
run state as a failed qualification gate, so an interrupted acquisition cannot
be mistaken for an incomplete-but-otherwise-valid matrix cell.

Qualification now fails closed for missing/unsupported run-manifest schemas,
dirty or failure-injected/non-production provenance, camera FrameID gaps or
regressions, duplicate/out-of-order inference identities, and orphan inference
rows. Skipped preflight and the qualification-only DeepStream debug profile are
explicit production disqualifiers; the latter has a narrow single-run exception
only for paired overhead measurement and remains forbidden in the production
matrix. The matrix additionally requires one application commit, one complete
device/package/native-plugin identity across every cell, and one hashed
model/engine identity across inference-enabled cells. It binds trigger mode and
edge, controller rate, exposure, recording bitrate, and serial state/port/baud
plus controller protocol at startup, then requires the task snapshot and actual camera
serial/model/firmware/runtime identity to remain identical across the campaign.
A process-owned acquisition lock
prevents two GUI instances from competing for the camera/controller and is
released automatically after a process crash.

1. Turn the pipeline topology and the invariants above into contract tests.
   Assert that the recording queue is non-leaky, all display/inference queues
   are isolated, and every fatal recording condition propagates to run status.
2. Add deterministic failure injection at the source, record queue, encoder,
   muxer, disk, serial controller, and shutdown boundaries.
3. Store a versioned run manifest with Jetson model, Jetson Linux/JetPack,
   kernel, CUDA, TensorRT, DeepStream, GStreamer, Spinnaker, native plugin build
   identity, model/engine hashes, power mode, and application commit.
4. Keep generated TensorRT plans out of the portable source-of-truth model
   package. Build them on an otherwise idle target and reject a plan whose
   recorded TensorRT/device identity does not match the runtime.

   The engine builder now emits schema-3 build identity for TensorRT, CUDA,
   compute capability, Jetson model, and Jetson Linux. Schema-3 model selection
   rejects any runtime mismatch before acquisition. Existing schema-2 packages
   remain readable only as a migration path and must be rebuilt with the
   updated notebook to gain the strict identity gate.

   On September 8, 2026 this JetPack 7.2.1 Orin Nano Super built the custom
   `mousehouse_best.pt` source into a separately named
   `models/mousehouse_jp721` schema-3 candidate. Independent strict validation
   matched CUDA 13.2, TensorRT 10.16.2.10, compute capability 8.7, device/L4T
   identity, and every path-bound artifact digest. A bounded `trtexec` check
   executed the plan successfully with input `1x3x640x640` and output
   `1x300x63`; its non-truncated execution report is itself hashed by the model
   manifest. The older `models/mousehouse` schema-2 package was deliberately
   preserved and remains production-ineligible.

   The same device also completed an operator-observed bounded direct-source
   FLIR smoke test with
   10/10 complete Mono8 frames, contiguous camera chunk FrameIDs, contiguous
   acquisition stream IDs 0 through 9, valid CRCs, and 1,555,360 valid payload
   bytes per frame. Its raw artifact is not retained in this repository; this
   is a development note, not a GUI recording run or a sustained qualification
   cell.

Exit criterion: the existing validated graph can be reconstructed from tests
and a run can prove its software, model, device, timing, and frame identity.

### Interrupted 88-hour development observation

The operator-reported August 27–31, 2026 bench run demonstrated healthy
camera/encoder endurance
for 88.09 hours: 9,513,447 recording admissions matched 9,513,447 MP4 frames,
the full 134.78 GB MP4 demuxed without error, frame IDs were contiguous, and
recording backlog stayed far below its warning threshold. It is not a validated
scientific run. VS Code died after 45.58 hours and took the GUI-owned serial
ledger with it, while the independently sessionized capture child continued for
another 42.51 hours. The eventual reboot left one final source frame outside
the recording branch and prevented finalization. This is endurance evidence for
the dirty working-tree build only; its raw directory is not retained in this
repository and it is not release qualification evidence. The GUI remains the required operator
interface. The production launcher now starts a detached, Qt-free supervisor
which owns serial control, capture, the acquisition lock, and ordered
finalization, then launches and monitors the GUI. Loss of the GUI or its
exclusive local IPC lease cancels startup or fails an active run through the
same ordered finalizer; acquisition cannot continue headlessly. This boundary
still requires on-device GUI-crash qualification before scientific collection.

## Stage 1: separate responsibilities without changing the graph

Status: repository implementation complete; on-device equivalence remains in
progress. Immutable inference and operator contracts, video probing,
frame audit, recording operators, complete graph construction, capture,
inference, video, and acquisition reconciliation, capture drain, versioned
child/backend events, and pure GUI health/presentation mapping now have
independent modules. Manifest/provenance, preflight execution, process-group
supervision, fail-closed startup orchestration, ordered shutdown coordination,
and bounded post-run finalizer supervision are independent Qt-free services.
`ServiceMakerApp` owns lifecycle
rather than graph construction, and the GUI consumes typed backend phases.
Preview, system meters, finalization presentation, bottle measurements,
dashboard modeling/presentation, session configuration, configuration form and
validation policy, main-view construction/theme, and individual profile dialogs
have been extracted with compatibility façades. `MainWindow` is now an
integration shell whose run orchestration delegates to `RunPresenter` and pure
presentation policy. Capture-output interpretation and the drain coordinator
are independently tested backend services; a human-readable legacy readiness
line can no longer arm the controller without its validated structured event.
Recording liveness and pose CSV persistence are independently tested inference
services.

The operator surface now uses versioned native Qt dock cards instead of a
pixel-tuned central grid. The run-control header remains pinned, the canonical
three-column plus full-width behavior arrangement is always recoverable, and
operator card positions persist independently from run data. Card movement and
floating require an explicit idle-only unlock and are forced locked throughout
startup, recording, and finalization. Every card uses Qt's native hide/show
action: right-click any card title to toggle cards from one unified checklist,
and Reset Layout remains the recovery path if multiple cards are hidden.

Pipeline construction is physically separated into source/recording,
inference/tracking, and preview/output graph modules, with a small mux/resource
orchestrator retaining the golden element order and properties. The GUI consumes
a narrow backend protocol and immutable run snapshots instead of mutating or
reading backend internals. Ordinary child stdout can no longer select an
ambient latest run; only the prepared directory carried by validated structured
lifecycle events is accepted.

The streaming callback audit now hard-caps the pose handoff and dashboard
display history, bounds native metadata inspection, and uses fixed-size
userspace buffering only for sidecars that are read after orderly close. The
record-admission ledger remains line-visible because controller shutdown reads
it before capture closes, and recording telemetry remains line-visible for live
operator health. Abrupt loss of a buffered sidecar tail cannot silently pass:
the nonterminal/unreconciled run fails validation.

Camera runtime identity is accumulated in bounded memory and persisted only
during close, not by atomic JSON replacement on the streaming callback. An
identity change or an unexpected camera count fails closed. The serial reader
also bounds individual lines and its pre-ledger queue, actively closes the port
to unblock shutdown, and promotes reader, port, and ledger-close failures into
run-integrity failures. GUI close requests now pass through the same backend
lifecycle gate as the Stop button: startup is allowed to resolve, capture is
stopped and finalized in order, and the window refuses to disappear when child
exit or finalization cannot be confirmed.

The production process boundary is now separated as well: a Qt-free supervisor
owns the backend and launches the mandatory GUI as its sole authenticated lease
holder. Versioned, bounded Unix-socket messages carry commands and typed events;
backend callbacks only enqueue bounded work and never write the socket. A full
recording-critical queue fails closed, while presentation events may be dropped
with an explicit count. GUI loss is tested before run creation and during
recording/finalization. Startup cancellation and controller `START` are
lease-atomic, and the supervisor remains alive until ordered finalization and
terminal persistence finish.
Run stop is asynchronous at the IPC boundary: the command worker acknowledges
ownership immediately, a non-daemon supervisor worker performs ordered shutdown,
and the GUI waits for terminal lifecycle events while continuing its main-loop
lease. Accelerated blocked-finalizer tests verify stop acknowledgement and
progress snapshots without requiring an hours-long acquisition.

A bounded lease emitted by a Qt-main-loop timer also makes a frozen GUI fail
closed; a proxy background thread cannot renew that lease by itself. The
supervisor validates renewals inline so long startup/finalization commands do
not create false expiry. The acquisition child additionally arms Linux
`PR_SET_PDEATHSIG` before executing DeepStream and verifies its expected parent
PID, preventing a hard supervisor death from leaving the sessionized capture
process running indefinitely.

1. Split the inference runner into pipeline specification, source/recording,
   inference/tracking, preview, telemetry, lifecycle, and artifact-validation
   modules. Keep one lifecycle owner and explicit state transitions.
2. Split the operator backend into immutable run configuration, preflight,
   process supervision, state machine, and finalization services. GUI code
   observes typed state and does not own acquisition lifecycle decisions.
3. Replace unstructured cross-process strings with a versioned event schema;
   retain human-readable logs as a presentation layer.
4. Use bounded queues and constant-memory summaries for long runs. Audit every
   callback/probe so it does bounded work and never performs avoidable blocking
   I/O on a streaming thread.
5. Keep DeepStream Service Maker. DeepStream 9.1 deprecates `pyds` and recommends
   `pyservicemaker`; the Pipeline API is appropriate where this application
   needs exact graph control.

Exit criterion: module boundaries are testable independently, while graph,
artifacts, and on-device frame counts remain identical to the frozen baseline.

## Stage 2: observability and sustained-operation validation

Status: repository tooling complete; empirical qualification remains in
progress. Preflight reports `tegrastats` without the invalid LFB
threshold, rejects automatic AC suspend, and an opt-in NVIDIA latency profile
is available. The capture child now owns a constant-memory `tegrastats`
recorder; during acquisition the GUI follows that CSV instead of launching a
duplicate sampler. `scripts/qualify_run.py` performs bounded-memory evidence
analysis with a versioned measurement-only limits profile. Empirical limits,
instrumentation-overhead measurements, and execution of the sustained
qualification matrix remain open. The versioned 24-cell matrix definition and
bounded evaluator are checked in; they reject run reuse and verify duration,
capture, inference, preview, power-mode, clean Git, production eligibility,
shared commit, and hashed model/engine identity for every inference-enabled
assigned cell.
The evaluator also rejects any cross-cell change in the complete acquisition
protocol or actual camera identity; inference and preview remain deliberate
matrix dimensions rather than acquisition-identity fields.
Each leaky preview queue now has source-identity probes immediately before and
after the queue. Bounded, durable per-camera ledgers reconcile delivered and
intentionally shed preview frames after capture. Their failure blocks
preview-enabled qualification but is explicitly excluded from recording
validity; both probes remain downstream of the recording tee.
Validated limits are schema-checked as a complete finite threshold set, and
qualification fails on invalid/unknown recording telemetry or missing
per-camera recording telemetry coverage.

Limits, matrix, assignment, failure-plan, and debug-threshold inputs must be
stable regular files; reads are size-bounded, strictly typed, duplicate-key
rejecting, and fail closed on unknown fields or case IDs.
Every long-run CSV/JSONL evidence consumer also bounds each physical record,
requires strict UTF-8, and either fails the scientific gate or reports invalid
diagnostics on malformed input. Offline replay indexes frame identity in a
temporary disk-backed database rather than retaining a run-length-sized map in
memory, and rechecks its input identities after decoding before publishing
derived results. Replay uses the decoder EOS audit itself for the exact video
count, avoiding a redundant, fixed-time FFprobe scan of multi-day recordings.
The CLI runs native Service Maker in an isolated worker process: signals request
normal stop/drain first, while an external 45-second supervisor boundary kills
a stuck native teardown and returns failure. Linux parent-death containment
prevents a hard supervisor death from orphaning the native worker. A replay can
never publish success without decoder EOS and the exact ledger count.
System qualification measures per-field and explicit thermal-state coverage so
unknown samples are not treated as safe. Recording metrics reject negative or
non-finite rows, and acquisition health requires camera transport-counter
coverage for every configured stream. Telemetry-recorder loss is persisted as
soon as observed without interrupting the authoritative recording branch, and
it independently fails qualification.

Qualification summaries hash every manifest, status, telemetry, and limits
artifact they consumed. The paired debug comparison rejects stale summaries,
requires the exact debug-only exception, the same validated limits file, and
nonnegative finite NVIDIA latency values. Its checked protocol requires
successfully terminal durable-supervisor runs with the same backend-enforced
qualification case binding and retains both source-evidence reference sets.
Matrix cells require an exact labeled
power mode and one canonical platform/package/native-plugin identity across all
cells, including inference-off cases.

Campaign execution now has a read-only `--next-case` worksheet that exposes the
first unassigned case, exact bound factors, duration, progress, environment,
and terminal launch command without starting acquisition. A strict unapproved
debug-overhead threshold template enumerates all required metrics without
turning measurement defaults into acceptance limits. Completed campaigns can
be inventoried outside their run directories with bounded streaming SHA-256;
the archive verifier rejects missing, extra, symlinked, special, traversing, or
modified copied evidence and never copies or bundles large videos itself.
The recording validator does not accept a duration/header estimate as proof.
The FLIR source writes canonical `frames.csv` rows during acquisition. Routine
shutdown validates the final MP4 sample-size, timing, sample-to-chunk, and
chunk-offset tables against the live manifest and durable final
capture/admission indices. Triggered runs then generate controller/camera
alignment automatically. Shutdown does not reconstruct `frames.csv`, hash or
decode the complete recording, or perform detailed object reconciliation.
The source also writes low-rate camera telemetry, camera integrity events, and
the resolved camera runtime snapshot live, so those diagnostic artifacts do
not require post-run reconstruction. Deferred object statistics are recorded
as unavailable (`null`), never as a misleading measured zero.
Qualification may explicitly request the deeper object or decode checks.

Future development should add richer progress/checkpointing to optional deep
analysis and add a
bounded complete compressed-bitstream scan that does not depend on
`qtdemux`'s per-sample index. None of those items changes the non-leaky
recording path or blocks the metadata/sample-table validation policy above.

Controller alignment now requires the ordered `START_SENT`,
`CAPTURE_STOP_REQUESTED`, `STOP_SENT`, and `CAPTURE_STOP_DONE` markers. Its
trigger epoch starts immediately after `START_SENT` and continues through the
end of the bounded serial ledger: every `CAMERA_HIGH` in that epoch, including
the shutdown tail after `CAPTURE_STOP_REQUESTED`, must map bijectively to one
recorded camera frame. The summary records unmatched epoch and tail counts;
automatic alignment and later qualification both fail closed unless those counts
are zero and the video/frame-ledger count also matches. Qualification binds the
exact alignment-summary bytes it parsed into its source evidence.

1. Add an opt-in debug profile that captures DeepStream frame/component latency
   (`NVDS_ENABLE_LATENCY_MEASUREMENT` and
   `NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT`) and Service Maker performance
   data beside existing recording telemetry. Measure its overhead before use
   during scientific runs.

   Initial support is available with
   `SQUEAKVIEW_DEEPSTREAM_DEBUG_PROFILE=1`; it enables both NVIDIA latency
   variables in the capture subprocess and attaches NVIDIA's shipped
   `measure_latency_probe` and `measure_fps_probe` only downstream of the leaky
   inference input. Preflight hashes both bounded regular-file modules into the
   run manifest, and child output is retained in the run-local, 64 MiB-bounded
   `diagnostics/deepstream.log`. When that profile makes
   `NVDS_ENABLE_LATENCY_MEASUREMENT` truthy, the raw FLIR source now calls
   NVIDIA's public `nvds_add_reference_timestamp_meta` immediately after setting
   buffer timestamps, using its real GStreamer element name and acquisition-local
   source sequence. The API returns no status, and NVIDIA's header documents
   decoder names as its standard anchors, so this correct raw-source anchor still
   does not prove that frame-latency records will be nonempty on this graph. It
   marks the run non-production and remains qualification-only until both structured frame
   and component latency plus finite Service Maker FPS evidence is observed
   without log truncation and its overhead is measured. Only
   `scripts/qualify_run.py --allow-debug-profile` may waive that single reason
   for the paired overhead comparison; the sustained production matrix still
   rejects it. The comparison remains incomplete until explicitly approved,
   evidence-derived overhead thresholds are supplied.

Acquisition additionally enforces the same free-space reserve used at startup
through a constant-memory live monitor. A reserve breach or disk-query failure
emits a structured fatal event and enters normal EOS/container draining; it is
not converted into an abrupt process kill. The resolved threshold, check
interval, and fail-closed policy are retained in the run manifest.
2. Capture `tegrastats` CPU, GPU, EMC, RAM, temperatures, throttling, and power
   rails at a bounded interval. Treat LFB as informational: NVIDIA defines its
   largest block as at most 4 MB, so a 16 MB warning threshold is invalid.
3. Benchmark sustained acquisition in the bounded 25 W profile and
   MAXN_SUPER. NVIDIA describes MAXN/MAXN_SUPER as experimental and warns that
   prolonged heavy workloads may throttle. Select the mode that produces stable
   no-drop runs, not merely the highest short benchmark.
4. Disable automatic suspend on acquisition systems and report the policy in
   preflight. Jetson Linux 39.2.1 documents an Orin Nano watchdog-reset risk
   during SC7 suspend/resume.
5. Establish short, one-hour, and full-duration qualification matrices across
   inference on/off, preview on/off, production-supported capture profiles, and
   power mode. The current production scope is explicitly the matrix's single
   1440×1080/30 FPS/one-camera profile. Wider GUI validation ranges are for
   development runs and are not claims of scientific support; any additional
   resolution, rate, pixel format, or camera count requires a new versioned
   capture profile and the complete matrix before production use.

Exit criterion: resource saturation, thermal throttling, and latency regressions
are visible and have defined pass/fail limits derived from validated runs.

## Stage 3: implemented pipeline choices and optional accuracy work

Status: the recording-codec and stream-multiplexer decisions are complete.
Their on-device sustained qualification is covered by the Stage 2 matrix; they
are not competing implementation options that remain to be selected.

1. Preserve CPU x264 recording on Orin Nano unless hardware changes. NVIDIA
   confirms that Orin Nano has no NVENC and documents software encoding as the
   supported path. The system-memory GRAY8 feed avoids an unnecessary
   NVMM-to-CPU round trip. The encoder now explicitly uses one reference frame,
   disables adaptive quantization, B-frames, and lookahead, and retains the
   selected lossy bitrate-controlled scientific format.
2. DeepStream 9.1 Stream multiplexer 2 is now pinned. Because it performs no
   scaling or color conversion, every live and offline input is explicitly
   normalized to NV12 in Jetson surface-array NVMM before the mux. Preflight
   exercises that exact VIC/NVMM path; the physical smoke and sustained matrix
   must still validate FLIR metadata and frame identity through mux and demux.
3. Consider TensorRT precision or model changes only with accuracy validation
   against a fixed scientific dataset. Never reuse JetPack 6/TensorRT 8 plans or
   calibration artifacts without rebuilding and revalidation.
4. Tune tracker and preview accuracy/performance last, independently of the
   recording branch and only against a fixed annotated dataset.

Exit criterion: each optimization demonstrates equal recording integrity and
acceptable scientific accuracy under the full sustained-run matrix.

## Remaining acceptance gates

The repository implementation is not the same as scientific qualification.
The following work intentionally remains open and must not be inferred from a
green unit-test suite:

1. Select and qualify the device-built `mousehouse_jp721` schema-3 candidate in
   the real GUI pipeline. Promote it to the canonical deployment name only if
   that name is operationally required; do not overwrite the preserved
   schema-2 `models/mousehouse` package merely to rename it. Fresh devices must
   rebuild their own schema-3 plan because generated model packages are not
   portable repository content.
2. Run the short, one-hour, and full-duration on-device matrix with real
   camera/controller hardware, approve evidence-derived limits, and retain the
   resulting qualification summaries. Include the physical serial-fault and
   hard-power-loss protocols.
3. Run a matched DeepStream debug-off/debug-on pair and confirm that NVIDIA
   latency records are actually present before accepting the instrumentation;
   then measure its acquisition overhead with the checked-in comparison tool.
4. Exercise the durable supervisor on-device by freezing and forcibly
   terminating the GUI during startup, recording, and each finalization phase,
   and by terminating the supervisor during capture. Retain evidence that
   startup never arms after lease loss, active acquisition stops, MP4 EOS/drain
   completes when possible, serial and frame ledgers close, the acquisition
   lock releases only after terminal status persistence, and no capture process
   survives headlessly. A hard supervisor death cannot perform ordered
   finalization, so its run must remain visibly nonterminal/failed qualification.
   Use the guarded, dry-run-first procedure in
   [SUPERVISOR_FAILURE_PROTOCOL.md](SUPERVISOR_FAILURE_PROTOCOL.md) for the
   active-capture and individual persisted finalization-stage cases; it records
   and immediately rechecks exact PID/start-time/parent/workload identities and
   never performs broad process-name signaling. Its supervisor action is a
   catchable `SIGTERM` test. An uncatchable supervisor-death test remains
   coupled to the controller watchdog safety gate below.

   Qualification-only bounded barriers now make `pre_capture`,
   `after_spawn_before_ready`, and the capture-reconciliation,
   inference-admission, recording-validation, and streaming-alignment stages
   deterministic. They require the explicit destructive-test gate, mark the run
   non-production, persist their waiting state, respond to cancellation, and
   self-release within 60 seconds. The injection helper can wait for the exact
   barrier while retaining its dry-run and UID/PID/start-time/parent/argv/run
   identity checks.
5. Obtain and version the deployed RP2040 firmware, implement its lease/watchdog,
   and produce durable electrical failsafe evidence. The host side now has an
   explicitly opt-in, production-disqualified experimental v1 state machine:
   nonce-bound negotiation and ACKs, bounded startup/disarm waits, monotonic
   heartbeat renewal from the supervisor, and fatal shutdown after two missed
   acknowledgements. Legacy behavior remains the default. The current repository
   has no firmware package or qualified watchdog, so host parent-death containment
   alone cannot guarantee
   that an independently powered controller stops pulses after Jetson failure.
   [CONTROLLER_WATCHDOG_PROTOCOL.md](CONTROLLER_WATCHDOG_PROTOCOL.md) now defines
   the exact wire/host/electrical contract and opt-in instructions; none of the
   host-only tests count as controller or electrical safety evidence.

Lossless/raw recording and the legacy stream multiplexer are explicitly not
remaining acceptance gates. The production architecture is the current lossy,
bitrate-controlled H.264 recording path and DeepStream 9.1 Streammux 2. Those
choices must pass the sustained matrix above, but no alternative recording
format or mux implementation needs to be developed or compared for this
release. Reversing either decision would require a new scientific requirement,
an isolated implementation, and a separate qualification campaign. Inference
precision, tracker, and preview changes remain deferred until fixed-dataset
accuracy evidence justifies them.

## NVIDIA references

- [JetPack 7.2.1 downloads and supported component versions](https://developer.nvidia.com/embedded/jetpack/downloads)
- [Jetson Linux 39.2.1 release notes and known issues](https://docs.nvidia.com/jetson/archives/r39.2.1/ReleaseNotes/Jetson_Linux_Release_Notes_r39.2.1.pdf)
- [Orin Nano software encoding and libx264 guidance](https://docs.nvidia.com/jetson/archives/r39.2.1/DeveloperGuide/SD/Multimedia/SoftwareEncodeInOrinNano.html)
- [Jetson `tegrastats` field definitions](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html)
- [Jetson Orin power and performance guidance](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)
- [DeepStream 9.1 release notes](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Release_notes.html)
- [DeepStream 9.1 migration guidance](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Migration_guide.html)
- [DeepStream performance troubleshooting](https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_troubleshooting.html)
- [DeepStream Service Maker overview](https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_service_maker_intro.html)
- [Service Maker Python advanced features](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_service_maker_python_advanced_features.html)
- [TensorRT engine compatibility](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)
