# Aligning Protocol-v2 SqueakView Outputs

This guide describes how to construct an analysis timeline from a finalized
SqueakView protocol-v2 run. Perform this work on the analysis device, treat the
copied run as read-only, and write derived tables to a separate directory.

No video decode is required. The Jetson already verifies that the MP4 sample
count agrees with the camera and recording ledgers.

## 1. Qualification gates

Require the following before analyzing a run:

```text
run_status.json.state == "finalized"
run_status.json.overall_validation_passed == true
run_status.json.recording_validation_passed == true
run_manifest.json.serial.controller_protocol == "v2"
run_manifest.json.serial.alignment_required == false
diagnostics/controller_v2_summary.json.integrity_latched == false
diagnostics/controller_v2_summary.json.counts.crc_or_framing_errors == 0
diagnostics/controller_v2_summary.json.counts.conflicting_duplicates == 0
```

`alignment_validated=false` is expected for v2. It means the obsolete legacy
`CAMERA_HIGH` aligner was not run; it is not a failed validation. The v2
acquisition gates are the controller transport, `CAMERA_STOP` reconciliation,
camera integrity, recording admission, and MP4 sample-count checks recorded in
`run_status.json`.

## 2. Canonical inputs

| File | Meaning | Natural key |
| --- | --- | --- |
| `frames.csv` | One authoritative row per recorded camera frame | `(stream_id, source_sequence_index)` |
| `record_admission.csv` | Non-leaky recording-branch ledger | `(stream_id, record_frame_index)` |
| `inference/frames.csv` | Frames admitted to inference | `(stream_id, source_sequence_index)` |
| `objects.csv` | Detector/tracker observations | `observation_id` |
| `keypoints.csv` | Pose points belonging to observations | `(observation_id, keypoint_index)` |
| `diagnostics/controller_v2.jsonl` | Durable CRC-checked controller records | `(boot_id, sequence)` |
| `diagnostics/controller_v2_summary.json` | Controller transport integrity summary | one per run |
| `serial.csv` | Compatibility export plus host lifecycle markers | physical row order |

The JSONL journal—not `serial.csv`—is the source of truth for v2 controller
records. `serial.csv` remains useful for human inspection and host-originated
markers.

`frames.csv` is the spine of frame-level analysis. Never use inference rows as
the spine: inference may be absent while the recording branch remains complete.

## 3. The v2 timing model

Legacy firmware emitted `CAMERA_HIGH` and `CAMERA_LOW` for every trigger.
Protocol v2 intentionally avoids that serial load. It emits:

```text
CAMERA_EPOCH       exact timestamp and count for the first trigger
CAMERA_CHECKPOINT  exact timestamp and count at a periodic checkpoint
CAMERA_STOP        exact timestamp and final trigger count
```

For the current single-camera system:

```text
controller_count = CAMERA_EPOCH.count + source_sequence_index
```

The final value must equal `CAMERA_STOP.count`. The acquisition finalizer also
requires that this count equal the source, recording-admission, frame-ledger,
and MP4 sample counts.

Anchor timestamps are exact controller measurements. Per-frame controller
timestamps between anchors are piecewise-linear reconstructions. Keep the
`controller_time_method` column in derived data so an exact anchor is never
confused with an interpolated timestamp. The camera's own
`camera_timestamp_ns` remains the exact frame-to-frame camera clock.

This distinction matters for sub-frame timing claims: v2 gives exact trigger
count identity for every frame, but only sparse exact controller timestamps.

## 4. Recommended pandas workflow

The repository includes a reusable, fail-closed loader:

```python
from pathlib import Path
from data_viz.v2_alignment import associate_events_to_frames, load_v2_run

RUN = Path("/data/squeakview/source_runs/<run_id>")
OUT = Path("/data/squeakview/analysis_results/<run_id>/alignment")
OUT.mkdir(parents=True, exist_ok=True)

run = load_v2_run(RUN)
frames = run.frames
events = run.events
anchors = run.anchors
aligned_events = associate_events_to_frames(events, frames)

frames.to_parquet(OUT / "aligned_frames.parquet", index=False)
aligned_events.to_parquet(OUT / "aligned_events.parquet", index=False)
anchors.to_parquet(OUT / "controller_camera_anchors.parquet", index=False)
```

`load_v2_run` verifies the finalized status, recording validation, v2 manifest,
transport integrity summary, contiguous durable sequences, one controller boot,
camera epoch structure, frame ordinals, and final controller/frame count. It
then adds these columns to `frames.csv`:

```text
controller_count
frame_controller_us
controller_time_method        exact_anchor | piecewise_interpolated
controller_anchor_left_count
controller_anchor_right_count
controller_anchor_span_frames
frame_time_s
video_frame_index
```

`associate_events_to_frames` maps each non-camera controller message to the
immediately preceding reconstructed frame and retains
`offset_from_frame_ms`. The original journal sequence, controller timestamp,
payload, host-receipt timestamps, boot ID, and session ID remain available.

## 5. Exact inference and object joins

Join inference to frames by exact identity:

```python
import pandas as pd

inference = pd.read_csv(RUN / "inference" / "frames.csv", low_memory=False)
inference["inference_present"] = True

aligned_frames = frames.merge(
    inference,
    on=["stream_id", "source_sequence_index"],
    how="left",
    suffixes=("", "_inference"),
    validate="one_to_one",
)
aligned_frames["inference_present"] = (
    aligned_frames["inference_present"].fillna(False).astype(bool)
)

present = aligned_frames["inference_present"]
for column in (
    "camera_frame_id", "camera_timestamp_ns", "gst_pts_ns", "raw_frame_index"
):
    assert aligned_frames.loc[present, column].eq(
        aligned_frames.loc[present, f"{column}_inference"]
    ).all(), f"inference identity mismatch: {column}"
```

Join objects with `(stream_id, source_sequence_index)` and validate
`camera_frame_id`. Join keypoints to objects using `observation_id`, then
cross-check their duplicated stream, frame, camera, track, class, and object
fields. Never fuzzy-match inference with timestamps.

## 6. Keep normalized tables

The relationships are one-to-many:

```text
one camera frame
  -> zero or one inference-frame row
  -> zero or more controller events
  -> zero or more objects
       -> zero or more keypoints
```

A direct wide join creates a Cartesian multiplication and corrupts event and
object counts. Keep normalized Parquet tables and join only what an analysis
needs. If a long-format export is required, use explicit `FRAME`, `BEHAVIOR`,
and `OBJECT` record types and aggregate keypoints per object first.

## 7. Large runs

Install a Parquet engine with pandas on the analysis device:

```bash
python -m pip install pandas pyarrow
```

Frames, anchors, and controller events are normally small enough for memory.
Read `objects.csv` and especially `keypoints.csv` in chunks:

```python
frame_lookup = frames[[
    "stream_id", "source_sequence_index", "raw_frame_index",
    "camera_frame_id", "controller_count", "frame_controller_us",
    "controller_time_method", "frame_time_s",
]]

objects_out = OUT / "aligned_objects.parquet"
objects_out.mkdir(exist_ok=True)
for part, objects in enumerate(
    pd.read_csv(RUN / "objects.csv", chunksize=1_000_000)
):
    aligned = objects.merge(
        frame_lookup,
        on=["stream_id", "source_sequence_index"],
        how="left",
        suffixes=("", "_frame"),
        validate="many_to_one",
        indicator=True,
    )
    assert aligned["_merge"].eq("both").all()
    assert aligned["camera_frame_id"].eq(
        aligned["camera_frame_id_frame"]
    ).all()
    aligned.drop(columns="_merge").to_parquet(
        objects_out / f"part-{part:05d}.parquet", index=False
    )
```

A directory of numbered Parquet parts is one logical dataset and keeps peak
memory bounded. The copied CSV and JSONL files remain the canonical evidence.

## 8. Required downstream checks

Record these checks with every derived dataset:

- the source run was finalized and production eligible;
- MP4, source, recording-admission, frame-ledger, and controller-stop counts
  agree;
- the durable controller journal has one boot and contiguous stored sequences;
- camera anchors have increasing unique counts and timestamps;
- every frame has one controller count and a declared timing method;
- frame order is strictly increasing by `raw_frame_index`;
- inference joins have no duplicate frame keys;
- every object references an existing frame;
- every keypoint references an existing `observation_id`;
- duplicated identity fields agree with their parent tables;
- source-run files remain unchanged.

Missing inference is not automatically a recording failure. Missing canonical
frames, a controller/frame count mismatch, a controller boot boundary, or a
durable v2 transport integrity failure is an acquisition-integrity failure.

## 9. Legacy runs

The old `scripts/align_run_outputs.py` utility and `alignment_summary.json`
workflow are retained only for historical runs containing per-frame
`CAMERA_HIGH` records. Do not run that legacy aligner on protocol-v2 data.
