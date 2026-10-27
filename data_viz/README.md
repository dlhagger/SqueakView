# Run Transfer and Analysis Workflow

SqueakView separates acquisition from downstream scientific analysis. The
Jetson is the acquisition appliance; the DGX Spark is the analysis system.

## System Responsibilities

The Jetson:

- Captures `raw.mp4` and the canonical CSV tables.
- Runs the bounded post-run integrity audit required to finalize a run.
- Records acquisition, model, software, and validation provenance.
- Does not run exploratory notebooks, full-run downstream analysis, video
  transcoding, or offline re-inference.

The DGX Spark:

- Stores a verified copy of each finalized run.
- Builds the protocol-v2 frame/controller timeline from durable camera anchors.
- Runs notebooks, statistics, visualization, and project-specific analysis.
- Stores every derived result outside the copied source run.

The Jetson performs bounded acquisition-integrity and frame-count validation.
Protocol v2 does not build a legacy per-frame TTL alignment on the Jetson;
derived analysis timelines are built on the analysis device. Wait for capture
validation to finish before copying a run.

For the exact frame identities, controller-clock mapping, inference joins, and
large-run processing rules, see [ALIGNMENT_GUIDE.md](ALIGNMENT_GUIDE.md).

## Current Transfer Procedure

The initial workflow uses a manual copy. Transfer automation is intentionally
deferred.

1. Wait until the GUI reports that capture validation and finalization are
   complete.
2. Enter and save final bottle weights when the experiment uses them.
3. Confirm that `_metadata/run_manifest.json` contains `"state": "finalized"`
   (`run_status.json` for historical recordings).
4. Copy the entire run directory to an external drive, network share, or other
   manually selected destination.
5. Copy that complete directory to the DGX Spark.
6. Keep the original Jetson directory until the DGX copy has been inspected and
   validated.

Copy the directory; do not move it off the Jetson during this first transfer.
Do not copy a hand-selected subset of files, rename files, flatten the directory,
or transcode `raw.mp4`.

Before using removable media, confirm that it has enough free space and uses a
filesystem that supports files larger than 4 GB, such as exFAT, NTFS, or ext4.
FAT32 cannot hold a normal long-run `raw.mp4`.

Example source directory:

```text
runs/<experiment>/<subject>/<subject>_<timestamp>_<shortid>/
```

## Canonical Run Files

A new finalized single-camera run contains:

```text
raw.mp4                         Authoritative H.264 camera recording
frames.csv                      Frame identity and timing
sensors.csv                     Sensor events and controller/host messages
detections.csv                  Pose observations and timing
weigh_in.csv                    Bottle/mouse measurements and bottle swap journal
_metadata/
  run_manifest.json             Provenance, lifecycle status, validation results
  layout.json                   Run layout marker
  config/                       Configuration snapshots
  diagnostics/                  Controller journal and runtime evidence
  recovery/                     Verified gzip capture/admission/inference ledgers
```

Historical recordings retain their original files, including `run_status.json`,
`serial.csv`, `objects.csv`, and `keypoints.csv`. Readers resolve both layouts.
For new runs, references to run status below mean the status fields in
`_metadata/run_manifest.json`; bottle history and audit records are in
`weigh_in.csv`. Debug and qualification runs retain plain recovery ledgers.

Some optional CSVs may contain only their header when no observations or errors
occurred. That is not by itself a failure. `run_status.json` and
`diagnostics/controller_v2_summary.json` determine whether a v2 acquisition
passed validation. `alignment_validated=false` is expected for v2 because the
legacy `CAMERA_HIGH` aligner is intentionally skipped.

For experiments with bottle replacements, use `bottle_history.json` (schema
version 1) rather than interpreting the compatibility bottle CSV as the complete
session. `current` holds the open bottle on each side and `history` holds closed
bottles. Bottle IDs, fluids, initial/final weights, derived intake, controller
installation/removal timestamps, and cumulative-counter boundaries associate
licks with the actual bottle. Session counters continue globally; only replaced
bottles receive a new baseline. The unchanged side retains its bottle and count.

A serial-enabled run with `capture.trigger_on=false` is a behavior-only
controller session: `START_BEHAVIOR` starts normal behavioral tasks and bottle
controls, while cameras free-run without controller TTL pulses. There is no
camera-TTL alignment in this mode. Use camera capture/admission/frame ledgers and
video counts for recording integrity; a controller TTL count of zero is expected
and must not be compared to the video's frame count. Behavioral timestamps and
shared bottle observation gaps remain valid controller evidence.

Each `gaps` entry marks a shared pause/calibration interval for both lick channels.
Exclude those intervals from valid observation time and lick-rate denominators;
absence of lick events during a gap does not mean zero drinking. Inspect
`bottle_history_events.jsonl` for calibration retries and interruption history.
Missing final weights produce incomplete intake measurements, and final weights
above initial weights remain visible with a plausibility warning. Mouse weight is
recorded in `weigh_in.csv` and reflected in the manifest bottle snapshot.

## DGX Directory Layout

Keep source data and derived work separate:

```text
/data/squeakview/
  source_runs/
    <run_id>/                   Unmodified copy from the Jetson
  analysis_results/
    <run_id>/
      alignment/
      figures/
      tables/
      reinference/
      logs/
```

Treat `source_runs/<run_id>/` as read-only. Analysis code should write to the
matching `analysis_results/<run_id>/` directory.

## Verify the Copied Run

First inspect `run_status.json` and confirm that the copied video and canonical
tables have the same byte sizes as the Jetson originals. For protocol-v2 runs,
load and validate the copied evidence without changing the source directory:

```python
from pathlib import Path
from data_viz.v2_alignment import load_v2_run

run = load_v2_run(Path("/data/squeakview/source_runs/<run_id>"))
print(len(run.frames), len(run.events), len(run.anchors))
```

The loader rejects incomplete acquisition validation, corrupt or discontinuous
v2 journals, controller boot boundaries, malformed camera anchors, and final
controller/frame count mismatches. Write resulting Parquet tables under the
matching analysis-results directory, never into the copied source run.

Runs recorded without serial input do not contain the RP2040 time base required
by this workflow. For those runs, use the recording and capture reconciliation
results in `run_status.json`.

## Demonstration Notebook

`analysis_demo_viz.ipynb` is an example of the table relationships and plotting
workflow. It is appropriate for short runs and sampled development data. It is
not the production-scale analysis engine for a 16-hour recording.

On the DGX, open the notebook from a Python environment containing Jupyter,
NumPy, pandas, Matplotlib, Seaborn, and IPython:

```bash
jupyter lab data_viz/analysis_demo_viz.ipynb
```

Leave `RUN_DIR = None` to select the newest run referenced by the project
`.latest_run` markers, or set it to a copied source-run path explicitly. Leave
`INFERENCE_RESULT = "live"` to inspect the acquisition-time inference outputs.
The notebook currently reads canonical CSVs into memory, so do not point it at a
full long-duration run unless a suitably sampled copy has been prepared.

Production analyses should use chunked readers, a columnar store, or a database
appropriate to the scientific question, then write figures and tables under
`analysis_results/<run_id>/`.

## Video and Re-inference Rules

`raw.mp4` is the authoritative compressed recording. Downstream tools should
decode it directly and must not replace or overwrite it. A proxy or transcoded
video may be created only as a clearly labeled derived artifact.

Offline re-inference is reserved for the DGX workflow and is not yet the
production long-run path. A DGX-specific TensorRT engine and runtime validation
will be required because TensorRT plans are device-specific. Future re-inference
results must preserve frame ordinal mapping through `frames.csv` and remain
under `analysis_results/<run_id>/reinference/`; they must never overwrite live
`objects.csv` or `keypoints.csv`.

## Future Transfer Release

A future release should add an optional `rsync`-based Jetson-to-DGX transfer
workflow with resumable copying and automated verification. Until that release,
the supported procedure is the complete manual directory copy described above.

## Bottle History and Swap Gaps

Runs with bottle swapping store every installed bottle in `bottle_history.json`.
`bottle_history_events.jsonl` retains swap requests and historical weight
corrections; the reliable controller journal contains the hardware transitions.
Global lick counters continue across swaps. A replacement starts its own bottle
counter, while the unchanged bottle keeps its identity and accumulated licks.

To annotate events loaded through `v2_alignment`:

```python
from data_viz.bottle_history import load_bottle_history, annotate_bottle_events

history = load_bottle_history(run_dir)
if history is not None:
    events = annotate_bottle_events(events, history)
```

The returned DataFrame includes `bottle_id`, `bottle_fluid`, and nullable
`lick_observation_valid`. Use controller `rp2040Time` microseconds from the v2
loader. Both sides are unavailable throughout each pause/calibration gap;
exclude those intervals from observation duration instead of interpreting them
as zero licking. The helper leaves unrelated sensor events unassigned and
returns `None` when loading an older run without bottle history.

New recordings use `frames.csv`, `raw.mp4`, `sensors.csv`, `detections.csv`, and
`weigh_in.csv`, with recovery and provenance artifacts under `_metadata/`.
The Python loaders resolve both this layout and historical runs. Primary CSVs
contain native timestamps and labelled controller-time mappings on each row
after finalization; manual weigh-in times describe entry time.

Bottle history and observation gaps for new runs are loaded directly from
`weigh_in.csv`, including controller-confirmed swap boundaries and cumulative
lick counters. Separate bottle-history JSON/JSONL files are only used by older
recordings.
