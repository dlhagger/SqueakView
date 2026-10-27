# SqueakView scripts

Run these from the application checkout. The normal operator entry point is
`bash squeakview.sh`; these scripts support installation, checks, analysis,
and qualification.

```text
scripts/
  setup_jetson.sh          # Initial OS/device setup
  build_native.sh          # Rebuild native components
  preflight.sh             # Readiness checks
  qualify_run.py           # Check a completed run
  qualification/           # Campaign, overhead, and failure-injection tools
  legacy/                  # Older controller alignment command
```

Start the operator with `bash squeakview.sh`. The unused
`scripts/launch_operator.sh` forwarding alias has been removed.

| Script | Purpose | When to use it |
| --- | --- | --- |
| `setup_jetson.sh` | Install OS packages, configure serial permissions and the project parent, then build native components as the desktop user. | Initial machine setup; it requests a reboot afterward. |
| `build_native.sh` | Build and check the FLIR source and pose parser. | After native source changes or SDK updates. It does not install OS packages. |
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
After successful validation, ordinary runs compress recovery ledgers into
`_metadata/recovery/`; readers support both plain and gzip files. Existing
recordings retain their historical layout.

Qualification tools resolve both layouts. Their default per-run summaries go
under `_metadata/` for new runs. Explicit analysis output directories remain
available when supported by a command.

The legacy alignment CLI reads `sensors.csv` and `detections.csv` in the new
layout, or `serial.csv` and `objects.csv` in historical recordings. Its default
`alignment_summary.json` goes under `_metadata/` in new runs. `--detections`
selects a different detection table; `--objects` remains an alias. Canonical
data files are not rewritten by this validator.

Production protocol v2 uses `CAMERA_EPOCH`, `CAMERA_CHECKPOINT`, and
`CAMERA_STOP` anchors rather than one `CAMERA_HIGH` record per frame. The normal
Stop/finalization workflow validates recording counts and adds controller times
to the primary tables. `data_viz.v2_alignment.load_v2_run` reconstructs the
v2 analysis timeline for finalized runs; the teaching notebook uses it.

Python command-line scripts provide `--help`. The shell setup/build scripts
perform their work directly; they do not implement `--help`.
