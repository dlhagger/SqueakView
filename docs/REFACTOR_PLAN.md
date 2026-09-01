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
- Source capture, recording admission, encoded output, camera frame IDs, and
  trigger ledgers must reconcile at finalization. A run that cannot prove this
  equality fails.
- Inference and preview may shed work without changing the ground-truth
  recording. Their skips remain measured and attributable by source frame ID.
- Refactors must preserve shutdown order: stop new triggers/acquisition, send
  EOS, drain and finalize the recording container, then validate artifacts.
- Pipeline changes are accepted only after unit tests, failure-injection tests,
  an on-device short run, and a sustained on-device validation run.

Frame completeness and pixel fidelity are separate requirements. The current
bitrate-controlled H.264 recording can contain every camera frame while still
being lossy at the pixel level. NVIDIA documents `qp=0` as the libx264 lossless
setting. Do not silently redefine the scientific format: first document whether
an experiment requires complete temporal sampling, lossless pixels, or both.
If lossless pixels are required, introduce an explicit lossless/raw profile and
validate CPU throughput, storage bandwidth, file size, decoding, and ledger
reconciliation before production use.

## Stage 0: freeze behavior and evidence

Status: in progress. The shared capture policy, explicit non-leaky queue/QoS
contract, strict ledger parsers, nonzero recording validation, retained source
provenance, controller handshake, process-exit gate, device/native-binary
identity, and concurrent atomic metadata writes are implemented and covered by
the automated suite. Failure-injection and sustained hardware qualification
remain open.

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

Exit criterion: the existing validated graph can be reconstructed from tests
and a run can prove its software, model, device, timing, and frame identity.

### Interrupted 88-hour qualification evidence

The August 27–31, 2026 bench run demonstrated healthy camera/encoder endurance
for 88.09 hours: 9,513,447 recording admissions matched 9,513,447 MP4 frames,
the full 134.78 GB MP4 demuxed without error, frame IDs were contiguous, and
recording backlog stayed far below its warning threshold. It is not a validated
scientific run. VS Code died after 45.58 hours and took the GUI-owned serial
ledger with it, while the independently sessionized capture child continued for
another 42.51 hours. The eventual reboot left one final source frame outside
the recording branch and prevented finalization. This is endurance evidence for
the dirty working-tree build only; it is also direct evidence that terminal
detachment is insufficient as the final architecture. Serial control, capture
supervision, and finalization must ultimately share a durable session service.

## Stage 1: separate responsibilities without changing the graph

Status: started. Recording admission and telemetry operators have moved into a
dedicated module with compatibility re-exports; the remaining runner, backend,
post-run, and GUI boundaries are still to be decomposed incrementally.

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

Status: started. Preflight now reports `tegrastats` without the invalid LFB
threshold, rejects automatic AC suspend, and an opt-in NVIDIA latency profile
is available. Bounded long-run system telemetry and the sustained qualification
matrix remain open.

1. Add an opt-in debug profile that captures DeepStream frame/component latency
   (`NVDS_ENABLE_LATENCY_MEASUREMENT` and
   `NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT`) and Service Maker performance
   data beside existing recording telemetry. Measure its overhead before use
   during scientific runs.

   Initial support is available with
   `SQUEAKVIEW_DEEPSTREAM_DEBUG_PROFILE=1`; it enables both NVIDIA latency
   variables in the capture subprocess and labels the operator log. It remains
   qualification-only until its overhead is measured.
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
   inference on/off, preview on/off, supported resolutions/FPS, and power mode.

Exit criterion: resource saturation, thermal throttling, and latency regressions
are visible and have defined pass/fail limits derived from validated runs.

## Stage 3: optional performance changes

1. Preserve CPU x264 recording on Orin Nano unless hardware changes. NVIDIA
   confirms that Orin Nano has no NVENC and documents software encoding as the
   supported path. The current system-memory GRAY8 feed also avoids an
   unnecessary NVMM-to-CPU round trip.
2. Tune tracker and inference independently of recording. Orin Nano has no PVA,
   so retain CUDA NvDCF unless measurement supports a different tracker.
3. Evaluate the new `nvstreammux` only in an isolated branch. Its scaling and
   `live-source` semantics differ from the legacy mux, and synchronization can
   drop late inference buffers. Pin the selected mux behavior and require
   frame-identity tests before migration.
4. Consider TensorRT precision or model changes only with accuracy validation
   against a fixed scientific dataset. Never reuse JetPack 6/TensorRT 8 plans or
   calibration artifacts without rebuilding and revalidation.

Exit criterion: each optimization demonstrates equal recording integrity and
acceptable scientific accuracy under the full sustained-run matrix.

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
