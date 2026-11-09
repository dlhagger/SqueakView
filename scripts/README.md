# SqueakView scripts

Run these from the application checkout. The normal operator entry point is
`bash squeakview.sh`; these scripts support installation, checks, analysis,
and qualification.

```text
scripts/
  setup_jetson.sh          # Initial OS/device setup
  build_native.sh          # Rebuild native components
  preflight.sh             # Readiness checks
  test.py                  # Python and camera-free native software checks
  qualify_run.py           # Check a completed run
  qualification/           # Campaign, overhead, and failure-injection tools
  legacy/                  # Older controller alignment command
```

Start the operator with `bash squeakview.sh`. The unused
`scripts/launch_operator.sh` forwarding alias has been removed.

Setup installs the bundled official `99-platformio-udev.rules` and reloads udev
for PlatformIO flashing and RP2040 bootloader access. Unplug and reconnect the
controller afterward to apply the rules; setup prints this reminder. The rules
are bundled under `scripts/udev/` so setup does not need a live rules download.

`setup_jetson.sh` also installs the controller USB recovery helper as a fixed,
root-owned executable, a no-argument sudo rule for the selected desktop user,
and a boot-created `dialout` lock file. When Start is pressed for a fresh run,
SqueakView holds the acquisition lock and recovers a missing Feather controller
on the dedicated Jetson USB-C port. Recovery is bounded to three attempts and
refuses an occupied port or a controller connection owned by SqueakView.
An already-present serial device is left in place. A stable by-id selection
never falls back to another board; an unambiguous ttyACM port can follow a
changed enumeration number.

Before a fresh production-v2 run, the app sends STOP to the verified Feather
controller and requires a fresh, durably recorded idle handshake before capture.
Shutdown waits for STOP acknowledgement and confirms idle with another fresh
challenge; failure is reported in the run result. Idle checks use the existing
firmware protocol and do not set the RTC. Opening or reattaching the GUI does
not reset USB or stop an experiment owned by the supervisor. Recovery does not
run mid-experiment. Restart the application after installing these changes so
its serial connection participates in the shared USB lock.

| Script | Purpose | When to use it |
| --- | --- | --- |
| `setup_jetson.sh` | Install OS packages, configure serial access, disable sleep, enable the R39.2.x firmware workaround, sync the UV environment, and build native components as the desktop user. | Initial machine setup; it requests a reboot afterward. |
| `build_pose_parser.sh` | Build the parser-only YOLO26 candidate and compare it against the legacy/model-package library. | Explicit equivalence testing; does not modify existing model packages. |
| `build_native.sh` | Build and check the FLIR source and SDK-selected parser (9.2 parser-only; 9.1 legacy). | After native source changes or SDK updates. It does not install OS packages. |
| `test.py` | Check dependencies, run nonempty unittest discovery, and build/run native EOS and pose-parser CTests. | Software changes or dependency upgrades; run with `.venv/bin/python scripts/test.py`. `--python-only` explicitly omits native checks. |
| `preflight.sh` | Check system, model, controller access, and recording prerequisites. | The operator runs it before acquisition; it can also be run manually with the selected project/model environment. |
| `qualify_run.py` | Evaluate a completed run against qualification limits. | Verify a run after acquisition. |
| `qualification/qualify_matrix.py` | Create a qualification checklist, assign completed runs, and evaluate coverage. | Hardware/software qualification campaigns. |
| `qualification/compare_debug_overhead.py` | Compare matched debug-off and debug-on runs. | Measure instrumentation overhead. |
| `qualification/archive_qualification_campaign.py` | Create or verify an inventory of campaign evidence without copying run data. | Preserve and verify qualification evidence. |
| `qualification/inject_supervisor_failure.py` | Inspect and deliberately inject supervisor/process failures. | Explicit failure-recovery qualification, using its documented arguments. |
| `legacy/align_run_outputs.py` | Validate legacy per-trigger camera/controller timing and detections. | Older `CAMERA_HIGH` controller runs. This is not the production-v2 sparse-anchor alignment tool. |

Shared alignment implementation now lives in
`squeakview/apps/inference/legacy_alignment.py`, alongside the application's
other validators. The CLI and application import the same implementation.

The qualification commands have moved from `scripts/<name>.py` to
`scripts/qualification/<name>.py`; the alignment command has moved to
`scripts/legacy/align_run_outputs.py`. Their command-line arguments are retained.
Update external shortcuts or automation that used the previous paths.

## Run files

New runs have `frames.csv`, `raw.mp4`, `sensors.csv`, `detections.csv`, and
`weigh_in.csv`. Configuration, validation, logs, and capture recovery ledgers
are under `_metadata/`. Bottle swaps and their recovery state are in
`weigh_in.csv`. Status and validation share `_metadata/run_manifest.json`.
Stop leaves recovery ledgers plain. Explicit `post_run --compact-metadata`
maintenance compresses and verifies them under `_metadata/recovery/`; readers
support both formats. Run maintenance with the app closed. Existing recordings
retain their historical layout.

Qualification tools resolve both layouts. Their default per-run summaries go
under `_metadata/` for new runs. Explicit analysis output directories remain
available when supported by a command.

Qualification of fast-finalized runs performs a full decode, hashes recording
evidence and reconciles preview. These results live in the derived summary's
`recording_audit`; scientific tables and capture status are unchanged. Run it
after acquisition and allow time for the full video scan. Each re-evaluation
repeats this audit. Normal Stop still uses the bounded count validator.

The legacy alignment CLI reads `sensors.csv` and `detections.csv` in the new
layout, or `serial.csv` and `objects.csv` in historical recordings. Its default
`alignment_summary.json` goes under `_metadata/` in new runs. `--detections`
selects a different detection table; `--objects` remains an alias. Canonical
data files are not rewritten by this validator.

Production protocol v2 uses `CAMERA_EPOCH`, `CAMERA_CHECKPOINT`, and
`CAMERA_STOP` anchors rather than one `CAMERA_HIGH` record per frame. The normal
Stop validates final counts and live integrity evidence without rewriting primary
tables. Controller mappings are derived on read from native clocks and anchors. `data_viz.v2_alignment.load_v2_run` reconstructs the
v2 analysis timeline for finalized runs; the teaching notebook uses it.

Python command-line scripts provide `--help`. The shell setup/build scripts
perform their work directly; they do not implement `--help`.
