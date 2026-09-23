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
serial.csv                      Decoded compatibility export and host markers
objects.csv                     Live detector/tracker observations
keypoints.csv                   Live normalized pose observations
run_manifest.json               Immutable acquisition and model provenance
run_status.json                 Lifecycle and validation result
diagnostics/                    Camera, recording, error, and finalizer records
  controller_v2.jsonl          Durable CRC-checked controller journal
  controller_v2_summary.json   Controller transport integrity summary
config/                         Run-local configuration snapshots
bottle_setup.json               Bottle setup, when used
bottle_measurements.csv         Entered bottle weights, when used
bottle_summary.json             Derived bottle intake, when used
```

Some optional CSVs may contain only their header when no observations or errors
occurred. That is not by itself a failure. `run_status.json` and
`diagnostics/controller_v2_summary.json` determine whether a v2 acquisition
passed validation. `alignment_validated=false` is expected for v2 because the
legacy `CAMERA_HIGH` aligner is intentionally skipped.

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
