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
- Rechecks alignment when required.
- Runs notebooks, statistics, visualization, and project-specific analysis.
- Stores every derived result outside the copied source run.

The Jetson post-run aligner is part of acquisition validation, not downstream
scientific analysis. Wait for it to finish before copying a run.

## Current Transfer Procedure

The initial workflow uses a manual copy. Transfer automation is intentionally
deferred.

1. Wait until the GUI reports that capture validation and finalization are
   complete.
2. Enter and save final bottle weights when the experiment uses them.
3. Confirm that `run_status.json` contains `"state": "finalized"`.
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

A normal finalized single-camera run contains:

```text
raw.mp4                         Authoritative H.264 camera recording
frames.csv                      One row per recorded frame
serial.csv                      Controller events and camera TTLs, when enabled
objects.csv                     Live detector/tracker observations
keypoints.csv                   Live normalized pose observations
run_manifest.json               Immutable acquisition and model provenance
run_status.json                 Lifecycle and validation result
alignment_summary.json          Jetson-generated timing audit
diagnostics/                    Camera, recording, error, and finalizer records
config/                         Run-local configuration snapshots
bottle_setup.json               Bottle setup, when used
bottle_measurements.csv         Entered bottle weights, when used
bottle_summary.json             Derived bottle intake, when used
```

Some optional CSVs may contain only their header when no observations or errors
occurred. That is not by itself a failure. `run_status.json` and
`alignment_summary.json` determine whether the acquisition passed validation.

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
CSVs have the same byte sizes as the Jetson originals. For serial-enabled runs,
the bounded aligner can independently validate the DGX copy without changing
the source directory:

```bash
python3 scripts/align_run_outputs.py \
  /data/squeakview/source_runs/<run_id> \
  --out-dir /data/squeakview/analysis_results/<run_id>/alignment
```

The command requires `ffprobe`, writes a new `alignment_summary.json` under the
selected analysis-results directory, and exits nonzero if frame, video,
controller, or object mapping validation fails. Compare this result with the
Jetson-generated summary retained in the source run.

Runs recorded without serial input do not contain the RP2040 time base required
by this aligner. For those runs, use the recording and capture reconciliation
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

Set `RUN_DIR` in the first code cell to the copied source-run path and leave
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
