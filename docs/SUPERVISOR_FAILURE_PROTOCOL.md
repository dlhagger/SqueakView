# Durable Supervisor Failure Qualification Protocol

This protocol validates that GUI crashes, disconnection, and missed GUI
heartbeats preserve scientific acquisition, while actual durable-supervisor
shutdown still stops acquisition safely. It is fault qualification, never a
production-data procedure. Use a test
subject/input, remain at the Jetson, and verify the controller has a physical
safe state before beginning.

The guarded helper is `scripts/qualification/inject_supervisor_failure.py`. It never uses
`pgrep`, process-name matching across the machine, shell globs, or process
groups. Its default is inspection-only. Before it can send one signal, it
requires and rechecks:

- the explicit supervisor PID and Linux `/proc` start-time ticks;
- the exact target PID and start-time ticks printed by a prior dry run;
- same-user ownership and direct supervisor parentage;
- the supervisor, GUI, and capture command-line identities;
- exactly one GUI and exactly one capture child for the explicit run directory;
- the exact expected lifecycle phase and, during post-capture work, its exact
  persisted finalization stage;
- durable-supervisor ownership in `run_status.json`;
- qualification-only environment and non-production provenance.

PID start time is required because Linux can reuse a PID. Any changed identity,
phase, relationship, metadata, duplicate process, missing process, or malformed
input aborts without signaling. The remaining scheduler-scale race between a
final `/proc` check and `kill(2)` cannot be eliminated through a PID-number API;
therefore run this only on the dedicated qualification Jetson with no unrelated
SqueakView processes. Do not use `kill`, `pkill`, or `killall` manually.

## Preparation

Run the normal preflight, connect the test controller and camera, and launch the
application with the explicit qualification gate:

```bash
SQUEAKVIEW_ENABLE_SUPERVISOR_FAILURE_INJECTION=1 \
  bash squeakview.sh
```

The launcher prints the supervisor PID. Every run created by this supervisor is
marked `production_eligible: false` with the
`supervisor_failure_injection` disqualifier. Do not collect production data in
this application session.

For scheduler-independent boundary placement, select exactly one bounded
qualification barrier before launching. Barriers are inert unless the failure
gate above is also exactly `1`; an invalid selection or timeout fails startup.
The default hold is 15 seconds and the hard maximum is 60 seconds, after which
the barrier releases itself so a missed injection cannot hang collection:

```bash
SQUEAKVIEW_ENABLE_SUPERVISOR_FAILURE_INJECTION=1 \
SQUEAKVIEW_SUPERVISOR_FAILURE_BARRIER=after_spawn_before_ready \
SQUEAKVIEW_SUPERVISOR_FAILURE_BARRIER_TIMEOUT_S=30 \
  bash squeakview.sh
```

Supported barriers are `pre_capture`, `after_spawn_before_ready`, and
`finalizer:capture_reconciliation`, `finalizer:inference_admission`,
`finalizer:recording_validation`, or `finalizer:streaming_alignment`. The first
holds after required run metadata exists but before serial/capture setup. The
second holds after the capture child is supervisor-owned but before readiness
can arm the controller. Finalizer barriers hold immediately after the named
phase/stage is persisted. GUI loss does not cancel an accepted start; every
barrier self-releases at its deadline. The gate continues to mark every such run
non-production even when no signal reaches the barrier.

Configure and start a short run through the mandatory GUI. Wait until the GUI
shows Recording. Record the absolute run directory and the supervisor PID.

## Dry run and explicit identity capture

The following command only inspects and prints a JSON plan:

```bash
.venv/bin/python scripts/qualification/inject_supervisor_failure.py \
  --action gui-terminate \
  --supervisor-pid SUPERVISOR_PID \
  --run-dir /absolute/path/to/run \
  --expected-phase recording
```

For a selected barrier, let the helper boundedly wait for the exact persisted
state instead of racing a brief stage:

```bash
.venv/bin/python scripts/qualification/inject_supervisor_failure.py \
  --action gui-terminate \
  --supervisor-pid SUPERVISOR_PID \
  --run-dir /absolute/path/to/run \
  --expected-phase starting \
  --expected-barrier after_spawn_before_ready \
  --wait-timeout 45
```

The wait is capped at 300 seconds and does not signal. After it observes the
exact waiting barrier, the helper performs all existing UID, parent/child,
argv, run-directory, non-production, and PID start-time checks. Execution still
requires a separate invocation with the printed exact identities,
`--execute`, and the confirmation phrase. For `pre_capture`, the helper proves
there is no capture or finalizer child; for `after_spawn_before_ready`, it
requires the exact run-bound capture child. Finalizer barriers retain the exact
post-run-child and phase/stage checks.

Confirm that `supervisor`, `gui`, `capture`, `run_dir`, and `phase` describe the
intended test. Preserve the printed supervisor `start_ticks`, target `pid`, and
target `start_ticks` in the qualification notes. The dry run always ends with
`DRY RUN: no signal sent`.

## Execute one fault

Repeat the same command with all three values copied exactly from the dry run:

```bash
.venv/bin/python scripts/qualification/inject_supervisor_failure.py \
  --action gui-terminate \
  --supervisor-pid SUPERVISOR_PID \
  --supervisor-start-ticks SUPERVISOR_START_TICKS \
  --target-pid TARGET_PID \
  --target-start-ticks TARGET_START_TICKS \
  --run-dir /absolute/path/to/run \
  --expected-phase recording \
  --execute \
  --confirm SUPERVISOR-FAILURE-QUALIFICATION
```

Run only one fault per application session and use a new run for each case:

- `gui-terminate` sends `SIGTERM` to the verified GUI. If its guarded close
  lifecycle disconnects, acquisition must continue while the supervisor replaces
  the GUI. A close confirmation dialog is not itself a crash or disconnection.
- `gui-freeze` sends `SIGSTOP` to the verified GUI. Heartbeat expiry replaces the
  stalled GUI; it must never stop controller events, capture, or scientific
  writers. The supervisor may kill a frozen GUI that cannot terminate normally.
- `supervisor-terminate` sends `SIGTERM` to the verified supervisor. Its minimal
  signal handler requests fail-closed shutdown; this is not a `SIGKILL` test.

For active acquisition, supported phases are `starting`, `recording`, and
`stopping`; the helper requires one exact capture child and no post-run worker.
For post-capture work, the helper instead requires that capture is absent and
that exactly one direct supervisor-owned `squeakview.apps.inference.post_run`
worker names the run directory. Use both the persisted phase and stage:

```bash
.venv/bin/python scripts/qualification/inject_supervisor_failure.py \
  --action gui-terminate \
  --supervisor-pid SUPERVISOR_PID \
  --run-dir /absolute/path/to/run \
  --expected-phase finalizing \
  --expected-stage capture_reconciliation
```

The supported exact post-capture cells are:

| Phase | Required stage |
|---|---|
| `finalizing` | `capture_reconciliation` |
| `finalizing` | `inference_admission` (inference-enabled runs only) |
| `finalizing` | `recording_validation` |
| `analyzing` | `streaming_alignment` (triggered, serial-enabled runs only) |

Copy the dry-run identities into the execute command exactly as for an active
capture and include the same `--expected-stage` value. Stages can be brief;
failure to catch one is not permission to weaken the checks. Use a sufficiently
long qualification run and repeat from a new run.

GUI presentation is optional throughout acquisition and post-capture work.
The supervisor keeps its backend, controller connection, capture process, run
identity, writers, and accepted operations across GUI loss. It replaces the GUI
using the same authenticated private socket, with restart backoff capped at
30 seconds. Repeated GUI-launch failures leave acquisition running headlessly;
they do not impose a recording deadline. A replacement retrieves runtime
configuration and the current run snapshot rather than starting another run.
Only explicit Stop or genuine acquisition/supervisor failure ends acquisition.

The bounded `diagnostics/supervisor_recovery.jsonl` log records GUI losses,
launches, reconnections, return codes, and signals independently of the GUI.
SIGKILL is compatible with OOM but does not establish OOM by itself; correlate
kernel evidence. Recovery logging uses a bounded asynchronous queue so a disk
or console diagnostic failure cannot block acquisition callbacks.

Do not test `SIGKILL` on the supervisor while a controller can generate pulses.
An uncatchable supervisor death cannot execute ordered controller shutdown; the
capture child has a parent-death signal, but independently powered controller
failsafe behavior is a separate hardware/firmware acceptance gate.

## Required evidence and pass criteria

For each case retain the GUI and supervisor bounded logs, `run_status.json`,
`run_manifest.json`, post-run progress, controller serial CSV, capture ledger,
recording telemetry, and video validation output. Record the dry-run JSON and
the helper's signal confirmation in the qualification worksheet.

For GUI-loss cases, the case passes only when:

1. The same run, supervisor, capture process, controller session, and scientific
   writers remain active across the presentation gap. Frame/event counters
   continue; no automatic controller STOP or backend abort is caused by GUI loss.
2. A replacement GUI reconnects, restores the active run configuration and
   status, and offers explicit Stop without creating a new run.
3. Stop from the replacement GUI follows normal controller acknowledgement,
   capture drain, file closure, and validation order.
4. Accepted startup/finalization operations continue despite GUI loss. Repeated
   GUI crashes or launch failures do not stop an otherwise healthy recording.
5. Recovery logs identify presentation loss and reconnect boundaries without
   converting a scientifically sound run into a failed run solely for GUI loss.
   Qualification runs remain non-production because of their qualification gate.

For actual supervisor termination, verify ordered controller STOP, capture
drain/closure, terminal failure evidence, and no orphaned acquisition processes.
The existing controller-watchdog limitations still apply to uncatchable
supervisor death or complete host loss.

If the helper refuses a check, preserve its error as evidence, stop manually
through the GUI when possible, and investigate. Never weaken or bypass a guard
to complete a qualification cell.
