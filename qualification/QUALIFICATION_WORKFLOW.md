# Sustained qualification workflow

This workflow prepares and evaluates the complete qualification matrix without
silently treating missing hardware, missing evidence, or provisional limits as
a pass. The checked-in matrix currently contains 24 cases. Its nominal capture
time is 136 hours 40 minutes (eight five-minute, eight one-hour, and eight
16-hour runs), before startup, finalization, review, reruns, and storage-copy
time. Plan capacity and operator coverage accordingly.

## Prepare the checklist

List the canonical case IDs and all required factors without accessing a camera
or controller:

```bash
.venv/bin/python scripts/qualify_matrix.py --list-cases
```

Create a checklist containing every case ID. Existing files are never replaced:

```bash
.venv/bin/python scripts/qualify_matrix.py \
  --init-assignments qualification/assignments.local.yaml
```

Each value initially is `null`, which means unassigned. Keep this file for the
entire campaign; it is the durable record of completed and outstanding cells.
Do not commit it when it contains local run paths.

Print the first unassigned case as a read-only worksheet:

```bash
.venv/bin/python scripts/qualify_matrix.py \
  qualification/assignments.local.yaml \
  --next-case
```

The JSON output includes progress counts, every bound capture/inference/preview
factor, the power mode, required minimum duration, an explicit environment map,
and a copyable launcher command. It never starts, stops, assigns, or modifies a
run. Set the named power mode separately and verify it with `nvpmodel -q` before
using the command. The generated environment requires
`SQUEAKVIEW_VIDEO_VALIDATION_FULL_DECODE=1`; routine structural validation is
valid for ordinary scientific runs but cannot satisfy the qualification gate.

## Acquire and assign a cell

The production matrix requires the specified FLIR camera for every cell. A
triggered protocol also requires the qualified controller and firmware safety
boundary. Inference-off is a capture test, not a hardware-free simulation. If
the camera or required controller is unavailable, prepare or inspect the
checklist only; leave those cells unassigned. They correctly remain
`incomplete`.

Only capture profiles explicitly listed in the selected versioned matrix are
in production qualification scope. The GUI accepts wider dimensions and frame
rates for development and characterization, but those settings are not
scientifically supported until added to a new matrix version and exercised
through every required cell.

Before each cell:

1. Run `bash scripts/preflight.sh` with the same model, camera, controller, and
   environment that the acquisition will use. Do not use a skipped preflight in
   the production matrix.
2. Select the matrix cell's capture, inference, and preview settings in the GUI.
   For preview-off cells, export `SQUEAKVIEW_DISABLE_PREVIEW=1` before launching;
   remove it for preview-on cells. Select inference on/off explicitly in the GUI.
   Set the named `25W` or `MAXN_SUPER` nvpmodel mode through the device's normal
   provisioning procedure and verify the exact label with `nvpmodel -q`.
3. Launch with `bash scripts/launch_operator.sh` from a standalone terminal and
   let the requested minimum duration elapse. Wait for terminal finalization;
   never assign an active or partially finalized run.

Bind the launch to the exact canonical case before starting the supervisor:

```bash
export SQUEAKVIEW_QUALIFICATION_CASE_ID='<case-id>'
# Only for a deliberately selected alternate versioned matrix:
# export SQUEAKVIEW_QUALIFICATION_MATRIX='/absolute/path/to/matrix.yaml'
bash scripts/launch_operator.sh
```

The supervisor-owned backend loads the strict matrix and compares the effective
width, height, FPS, camera count, pixel format, trigger mode/edge, controller
rate, exposure, recording bitrate, serial port/baud/state, inference state,
preview state, and exact current nvpmodel label before creating a run directory
or arming the controller. A mismatch aborts startup. The immutable manifest and
initial run status retain the matrix ID and SHA-256, case ID, and expected factors. Unset
`SQUEAKVIEW_QUALIFICATION_CASE_ID` after the campaign for ordinary unbound runs.

Record the run with the canonical case ID printed by `--list-cases`:

```bash
.venv/bin/python scripts/qualify_matrix.py \
  qualification/assignments.local.yaml \
  --assign '<case-id>' '/absolute/path/to/completed/run'
```

The command rejects unknown IDs, missing manifest/status artifacts, and reuse of
one run for two cells. It atomically updates the checklist. Re-running it for
the same case intentionally replaces that case's assignment; the prior run
directory remains untouched.

## Limits and evaluation

`qualification/limits.v1.yaml` is deliberately measurement-only. Evaluating
against it produces `incomplete`, even when capture integrity passes. Use the
short and sustained baseline evidence to create a separately reviewed,
versioned limits file, fill every required finite threshold, and set
`validated: true`. Do not turn the checked-in measurement template into an
approval merely to obtain a passing exit code.

Evaluate at any time to see missing, failed, and completed cells. Give each
campaign report a durable filename instead of relying on the replaceable
default report:

```bash
.venv/bin/python scripts/qualify_matrix.py \
  qualification/assignments.local.yaml \
  --limits qualification/<reviewed-validated-limits>.yaml \
  --output qualification/results/<campaign>-matrix-report.json
```

Exit status is `0` only for a fully passed matrix, `1` for any failed cell, and
`2` for incomplete input or cells. Retain together:

- the exact matrix and validated-limits files;
- the completed assignment checklist;
- the generated matrix report; and
- every assigned immutable run directory and its qualification summary.

The evaluator checks exact factors and power-mode labels, minimum durations,
production/preflight status, clean shared Git provenance, one complete shared
device/package/native-plugin identity, and shared model content identity for
inference-enabled cases. It also requires one canonical acquisition identity
across every assigned cell: the task snapshot hash, all trigger/controller and
recording settings, and the actual camera serial/model/firmware/runtime
configuration. This prevents runs from different experimental protocols or
different camera hardware from being combined into one passing campaign. An
assigned run must also contain an exact backend
startup binding to that matrix document and case; a manually assigned legacy or
unbound run fails the cell. The evaluator does not synthesize evidence for
unavailable hardware or extrapolate short runs into sustained cells.

## Inventory and verify the retained campaign

Create a durable content inventory outside every assigned run directory. This
command never copies, archives, or changes run data:

```bash
.venv/bin/python scripts/archive_qualification_campaign.py \
  --matrix qualification/matrix.v1.yaml \
  --assignments qualification/assignments.local.yaml \
  --report qualification/results/<campaign>-matrix-report.json \
  --limits qualification/<reviewed-validated-limits>.yaml \
  --inventory /absolute/archive-staging/<campaign>-inventory.json
```

It rejects incomplete assignments, reused/missing runs, active or nonterminal
status, mismatched report assignments, symlinks, special files, and unbounded
artifact trees. It records streaming SHA-256 and byte-size identities for all
four control documents and every assigned-run file. Copy files to the recorded
`archive_path` values using the site's validated storage procedure, then verify
the exact copied tree without modifying it:

```bash
.venv/bin/python scripts/archive_qualification_campaign.py \
  --verify /absolute/copied-campaign-root \
  --inventory /absolute/archive-staging/<campaign>-inventory.json
```

Verification rejects missing, extra, traversing, symlinked, or content-mismatched
entries. Keep the inventory beside, not inside, the tree it verifies.
