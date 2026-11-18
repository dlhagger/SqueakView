# Recording hardening and Jetson checks

Raw video and controller/sensor data determine capture success. Inference and
preview are recoverable services. Camera drops are acceptable when their locations
and timestamps can be proven; an unexplained discrepancy is still a failure.

## Changes

- Capture owns the FLIR sources, frame manifests, admission ledgers, and MP4s.
  Optional inference/preview owns a separate process and its own GIL. A bounded,
  preallocated `/dev/shm` mailbox holds one latest frame per camera. Neither side
  waits for the other side's file lock. A failed live setup disables live services.
- The live watchdog kills an unresponsive worker; capture keeps running. Worker
  output goes to `_metadata/diagnostics/live_processing.log`. State and errors are
  recorded in the run status's `live_services`; its heartbeat is in
  `_metadata/diagnostics/live_status.json`.
- Native gap events now include received-frame indices on each side, missing
  frame counts, and native/PTS timestamps. Finalization checks those timestamps
  against MP4 `stts`/`ctts` tables, and checks received + missing = controller
  triggers. `frame_gaps`, `recording_quality: complete_with_gaps`, and
  `recording_warnings` persist the result. The UI shows complete with warnings.
- Controller timestamp mapping skips the missing triggers instead of shifting
  later frames earlier. MP4 playback holds the prior image over the gap;
  downstream analysis must use timestamps rather than `frame_index / fps`.
- Recording faults now emit the supervisor's structured fatal event and attempt
  bounded native EOS/drain so that recoverable MP4 data can be closed.
  If native drain cannot be confirmed, callback-owned writers are left open
  until forced process exit instead of racing late writes during cleanup.

No lost pixels are reconstructed. Initial/trailing losses without enough frame
identity evidence, nonmonotonic clocks, CRC/incomplete images, raw write failures,
and unexplained source/admission/video/controller discrepancies remain failures.
Historical runs without the new gap evidence are not automatically relabelled.
Strict production qualification may still require zero losses even when a run is
usable with warnings.

## DID_DLH audit

The read-only review of `/home/dlh/Desktop/DID_DLH` found 67 manifests: 50 marked
finalized and 17 marked failed. These recordings predate the inspected HEAD,
`cc1025d` (which already changed Stop to nonblocking camera drain requests).

| Observed failure group | Runs | Interpretation |
| --- | ---: | --- |
| Camera gaps with controller/count mismatch | 8 | The previous zero-drop policy rejected missing camera frames; one also received SIGTERM. New runs can pass with warnings only when gap timing is proven. |
| SIGTERM during shutdown, without the gap failure above | 4 | Process lifecycle failure; an exit of -15 is not itself evidence that the saved video is corrupt. |
| Timestamp materialization | 2 | Finalization reported `primary entry is missing its native host timestamp`. These are not camera capture crashes. Current HEAD already avoids automatic primary-table rewriting. |
| Source stalled, followed by process abort | 1 | Capture/inference process exited -6 after readiness. The earlier log review showed the stall and a writer failure during teardown; the original stall's hardware cause is not established. |
| GUI loss followed by drain/finalizer failure | 1 | Capture was coupled to the operator lifecycle; current HEAD already contains GUI recovery work. |
| Startup readiness timeout | 1 | The controller was never started because the combined pipeline did not become ready within 30 seconds. Raw readiness now comes from the recording sink. |

The eight gap-labelled runs are mouse 1 on Oct 28, Nov 6 and Nov 7; mouse 2 on
Oct 28; mouse 3 on Oct 28; and mouse 4 on Oct 28 at 16:22, Nov 2 at 14:19, and
Nov 3. The four SIGTERM-only runs are mouse 3 on Oct 30 and mouse 4 on Oct 31,
Nov 4 and Nov 5. The two timestamp-materialization failures are mice 2 and 3
on Oct 27. The remaining failures are mouse 2 on Oct 22 (abort), mouse 6 on
Oct 23 at 10:36 (GUI loss), and mouse 6 on Oct 27 at 17:15 (readiness timeout).
Dates are the recorded run identifiers, in 2026. These categories explain the
failure labels; they do not certify every old MP4 as recoverable.

## Shutdown outcomes

Capture now writes version 2 of `capture_closure.json` (under `_metadata` for new
runs) only after an explicit Stop, an observed pipeline EOS, verified native
pipeline NULL state, stopped capture monitors, and successful closure of the
recording writers. The native sources expose read-only `pipeline-null` using
GStreamer's public state API; Service Maker returning from `wait()` is not enough.
Startup/preflight reject a stale source plugin without this property. Version 1
receipts are not accepted for late-exit recovery. This atomic, fsynced receipt
is written before optional system diagnostics and process teardown. It is not a
substitute for validation, and a new capture attempt removes any old receipt.

If the capture process subsequently exits nonzero, the supervisor can finalize
the run with a shutdown warning only when the receipt is valid, process exit is
confirmed, and all required recording, acquisition-integrity and controller
alignment checks pass. Controller stop failures, missing frames at drain, sensor
errors, validation timeouts, and failures during recording still fail the run.
Runs without closure evidence retain the conservative exit-code policy.

System diagnostic cleanup errors are warnings; controller/sensor acquisition
remains required. The final status records `capture_returncode`,
`shutdown_degraded`, and `shutdown_warnings`, and the UI shows **Complete with
warnings** for validated recordings affected only by cleanup problems.

Recording admission, ingress, and encoder-egress callbacks contain Python
exceptions before returning to GStreamer. A required ledger fault is reported
once and requests a controlled stop; later video buffers are allowed through
while shutdown proceeds. The fault remains fatal to validation. Sampled recording
diagnostic CSV failures disable those writes, preserve in-memory activity counters,
and become shutdown warnings. The active recording watchdog uses nonblocking
activity snapshots; a telemetry lock that stays unavailable triggers the same
bounded stall policy rather than hanging the watchdog itself.

The optional worker reports per-camera output counts as well as its heartbeat.
The parent compares these with nonblocking raw-capture counters (falling back to
published mailbox sequences when a snapshot is temporarily unavailable). Available camera
frames with no output progress for 10 seconds degrade and terminate only the
optional worker, even when its heartbeat is fresh. No frames before triggering,
or input paused after output catches up, do not cause a false stall. A healthy
camera cannot conceal another camera's stalled live processing.

Shutdown uses idle deadlines with separate absolute caps:

| Phase | Evidence that extends the idle deadline | Default idle / absolute cap |
| --- | --- | --- |
| Trigger-off ledger drain | Increasing source or admission frame counts | 5 / 60 seconds |
| Encoder/MP4 drain | Increasing source, admission, encoded frame counts, or raw video bytes | 20 / 120 seconds |
| Required writer cleanup | A completed monitor join or writer close | 5 / 30 seconds |

Counters establish a baseline, then must exceed their previous maximum. Repeated
logs, heartbeats, counter resets, and diagnostic-file growth do not extend the
deadline. Successful drain still requires the same count reconciliation, quiet
window, native completion, and independent validation. Progress alone never
establishes recording validity. `shutdown_stages` records observed counters,
elapsed and idle time, completion, and the limit reached for capture stages.

These defaults are engineering estimates pending Jetson load measurements. Set
`SQUEAKVIEW_CAPTURE_DRAIN_TIMEOUT_S` / `SQUEAKVIEW_CAPTURE_DRAIN_MAX_S` for ledger
idle / maximum time, and `SQUEAKVIEW_PIPELINE_SHUTDOWN_TIMEOUT_S` /
`SQUEAKVIEW_PIPELINE_SHUTDOWN_MAX_S` for pipeline idle / maximum time. Pipeline
idle time is bounded to 0.1–120 seconds, and absolute caps cannot exceed 600
seconds or be less than the corresponding idle time; invalid/nonfinite values
fall back to defaults.

The supervisor's external process deadline covers the pipeline absolute cap,
writer cleanup cap, other cleanup and a scheduling margin (166 seconds by
default). It normally observes exit much sooner. This also bounds a native call
that holds Python's GIL and prevents the child's progress monitor from running;
that case can require the external deadline rather than the child's idle timeout.
A blocked filesystem progress sampler cannot block the child's deadline owner.
An actual native drain timeout still needs Jetson diagnosis and does not produce
a clean closure receipt.

## On the Jetson

Rebuild the native source; older binaries lack the required `pipeline-null`
closure check (and may also lack mailbox/gap support).
With the Jetson environment activated, run:

```bash
bash scripts/build_native.sh
.venv/bin/python scripts/test.py
```

The software suite includes camera-free GStreamer source/EOS tests, graph ownership
checks, and a subprocess test that kills a reader holding the mailbox lock. These
do not replace camera/GPU testing. The non-Jetson development machine cannot build
the vendor-linked plugin without Spinnaker, GStreamer, and DeepStream headers.

Use short disposable runs first, keeping the full run directories for inspection:

1. Baseline with raw capture, controller/sensors, inference, and preview enabled.
   Stop normally and check raw sample counts, final sensor records, timestamps,
   and a completed live status.
2. Identify the **optional live worker** PID from its `live_worker --config`
   command line. Send that PID `SIGKILL` during capture. Repeat in another run
   with `SIGSTOP` and allow the watchdog to detect it. Raw frame counts and sensor
   records must keep advancing; Stop must close and validate the recording, with
   a live-service warning. Do not signal the capture or supervisor PID.
3. Disconnect/close the preview consumer and induce a live inference failure.
   Raw/controller acquisition must continue and later offline processing must
   still be able to read the recording.
4. Exercise a real, located transport frame-ID gap. Check that its native and
   MP4 time intervals agree, received + missing equals trigger count, and the
   first frame after the hole maps to the correct controller trigger. Expect
   `complete_with_gaps`, not failure. A missing tail without evidence must fail.
5. Repeat Stop during inference failure and under sustained load, then perform
   the target-duration run at the intended resolution/FPS and camera count.
   Check CPU, memory, shared memory, disk, and finalization duration. The extra
   frame copy and two-process DeepStream setup need on-device measurement.
   In disposable slow-drain tests, verify that advancing encoder/video counters
   allow drain beyond the idle interval, that a real stall ends the wait, and
   that continuous activity cannot exceed the absolute cap. Inspect
   `shutdown_stages` and the ledger-drain status before tuning the defaults.
6. On a disposable run, exercise a late capture-process abort **after** the new
   closure receipt appears, while teardown is still running. Require successful
   independent validation and a shutdown warning that preserves the exit code.
   Repeat without closure evidence or with an invalid recording: it must fail.
   Save the `[CLOSE]` log stages and timing for any native drain timeout.

Send back the full affected run folder (including hidden `_metadata`), the commit
and rebuilt-plugin version, camera count/resolution/FPS, and which fault was
introduced. That is enough to diagnose the next round without guessing from the
GUI's terminal label.
