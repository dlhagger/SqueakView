# Aligning SqueakView Outputs on the Analysis Device

This guide describes how to combine a finalized SqueakView run into an
analysis-ready timeline. Perform this work on the analysis device, not on the
Jetson acquisition system. Treat the copied run directory as read-only and
write derived tables to a separate analysis directory.

No video decode is required for this alignment. `raw.mp4` sample ordinal zero
corresponds to `frames.csv` `raw_frame_index=0`; the Jetson has already checked
the MP4 sample count against the authoritative frame ledgers.

## 1. Check the run before analysis

Open `run_status.json` and require all of the following:

```text
state == "finalized"
overall_validation_passed == true
recording_validation_passed == true
alignment_validated == true          # for controller-enabled runs
```

Also inspect `alignment_summary.json`. A valid triggered run has:

```text
frame_alignment.validated == true
counts.recorded_frames == counts.camera_high_events
counts.frames_missing_ttl == 0
counts.camera_frames_missing == 0
validation.video_frame_count_matches_frames_csv == true
```

Do not silently analyze a failed or incomplete run. Preserve its files, but
record the failed qualification in the downstream analysis.

## 2. Canonical inputs and their cardinality

| File | Meaning | Natural key |
| --- | --- | --- |
| `frames.csv` | One authoritative row per recorded camera frame | `(stream_id, source_sequence_index)` |
| `inference/frames.csv` | Frames admitted to the inference branch | `(stream_id, source_sequence_index)` |
| `objects.csv` | Zero or more detector/tracker observations per inference frame | `observation_id` |
| `keypoints.csv` | Zero or more pose keypoints per object observation | `(observation_id, keypoint_index)` |
| `serial.csv` | Ordered controller messages, camera TTLs, and lifecycle markers | physical row order |
| `record_admission.csv` | Durable recording-branch admission ledger | `(stream_id, record_frame_index)` |

`frames.csv` is the spine of a frame-level analysis. Never use
`inference/frames.csv` as the spine because inference or preview may omit
frames without affecting the non-leaky recording branch.

## 3. Preserve the three levels of data

The files are normalized because the relationships are one-to-many:

```text
one camera frame
  -> zero or one inference-frame row
  -> zero or more controller behavior events
  -> zero or more object observations
       -> zero or more keypoints
```

A direct wide join of all five CSVs creates a Cartesian multiplication. For
example, a frame with two behavior events, three objects, and nineteen
keypoints per object would incorrectly become 114 rows. Event and object
counts calculated from that table would be wrong.

Use one of these analysis representations:

1. Keep normalized aligned tables and join only the relationship needed by an
   analysis. This is the recommended representation.
2. Build one long-format table with a `record_type` column (`FRAME`,
   `BEHAVIOR`, or `OBJECT`) and common frame/time columns. Aggregate the
   keypoints for each object into a list or JSON value before adding an
   `OBJECT` row.

Do not put each keypoint into the same flat join as behavior events.

## 4. Align inference by frame identity

Join inference frames to `frames.csv` with:

```text
(stream_id, source_sequence_index)
```

For every joined row, cross-check that these values also agree when present:

```text
camera_frame_id
camera_timestamp_ns
gst_pts_ns
raw_frame_index
```

The identity join is exact; do not use nearest timestamps for inference.
A frame absent from `inference/frames.csv` should remain in the frame table
with `inference_present=false`, rather than being removed by an inner join.

Join `objects.csv` to the frame table using the same
`(stream_id, source_sequence_index)` key and cross-check `camera_frame_id`.
Join `keypoints.csv` to `objects.csv` using `observation_id`, then cross-check
its duplicated stream, source-sequence, camera-frame, track, class, and object
fields. Those duplicated fields are integrity evidence, not alternative fuzzy
join keys.

## 5. Establish the controller-to-camera mapping

`serial.csv` order is scientifically meaningful. Add a zero-based
`serial_index` while ingesting it and do not reorder the file first.

Locate these marker rows in order:

```text
START_SENT
CAPTURE_STOP_REQUESTED
STOP_SENT
CAPTURE_STOP_DONE
```

Marker names are stored on rows whose `eventType` is `MARKER`; depending on
the controller message, the name is present in `context`, `reason`, or
`rawLine`. Reject missing, duplicate, or out-of-order epoch markers.

Within the epoch, select the first `CAMERA_HIGH` after `START_SENT`. For the
current single-camera system, calculate:

```text
camera_frame_id_offset =
    first frames.csv camera_frame_id - first CAMERA_HIGH count

frame_ttl_count = camera_frame_id - camera_frame_id_offset
```

Join every frame to exactly one `CAMERA_HIGH` using `frame_ttl_count=count`.
This mapping must include TTLs received during the shutdown tail after
`CAPTURE_STOP_REQUESTED`.

For every matched frame, retain at least:

```text
raw_frame_index
camera_frame_id
frame_ttl_count
camera_timestamp_ns
gst_pts_ns
frame host timestamps
CAMERA_HIGH rp2040Time
CAMERA_HIGH host timestamps
```

Define elapsed experiment time on the controller clock:

```text
elapsed_s =
    (CAMERA_HIGH rp2040Time - first CAMERA_HIGH rp2040Time) / 1_000_000
```

Do not align controller and camera data through wall-clock time. Host Unix
timestamps are useful diagnostics but are not the primary controller mapping.

## 6. Attach behavior events

Behavior rows have their own `rp2040Time`. Associate each behavior event with
the immediately preceding `CAMERA_HIGH` in controller time (an as-of join):

```text
matched frame = CAMERA_HIGH with the greatest rp2040Time <= event rp2040Time
offset_from_frame_ms =
    (event rp2040Time - matched CAMERA_HIGH rp2040Time) / 1000
```

Retain the original behavior timestamp and the calculated offset. This makes
the association auditable and prevents an event from being presented as if it
occurred exactly on the frame boundary.

It can also be useful to calculate the nearest frame as a diagnostic, but the
preceding frame is the canonical causal association. Keep lifecycle messages,
acknowledgements, and raw controller lines available; filter to behavioral
event types only in an analysis-specific view.

## 7. Pandas workflow

Use pandas for the alignment. On a large analysis machine, the frame and
controller tables can normally remain in memory. Read `objects.csv` and
especially `keypoints.csv` in chunks when they do not fit comfortably.
Install a Parquet engine alongside pandas:

```bash
python -m pip install pandas pyarrow
```

A practical directory layout is:

```text
/data/squeakview/
  source_runs/<run_id>/                 # read-only Jetson copy
  analysis_results/<run_id>/alignment/  # derived outputs
```

The following is the core frame/controller/inference alignment. It intentionally
selects only the columns needed for the join; add scientific columns to the
`usecols` lists rather than initially loading every CSV column as Python
objects.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

RUN = Path("/data/squeakview/source_runs/<run_id>")
OUT = Path("/data/squeakview/analysis_results/<run_id>/alignment")
OUT.mkdir(parents=True, exist_ok=True)

status = json.loads((RUN / "run_status.json").read_text())
assert status["state"] == "finalized"
assert status["overall_validation_passed"] is True
assert status["recording_validation_passed"] is True
assert status["alignment_validated"] is True

frame_columns = [
    "stream_id", "source_sequence_index", "raw_frame_index",
    "camera_frame_id", "camera_timestamp_ns", "gst_pts_ns",
    "host_monotonic_ns", "host_unix_ns", "status",
]
frames = pd.read_csv(RUN / "frames.csv", usecols=frame_columns)
frames = frames.sort_values(["stream_id", "source_sequence_index"])
assert not frames.duplicated(["stream_id", "source_sequence_index"]).any()
assert frames["raw_frame_index"].is_monotonic_increasing

# Physical serial row order defines the controller epoch.
serial = pd.read_csv(RUN / "serial.csv", low_memory=False)
serial.insert(0, "serial_index", np.arange(len(serial), dtype=np.int64))

required_markers = [
    "START_SENT", "CAPTURE_STOP_REQUESTED", "STOP_SENT",
    "CAPTURE_STOP_DONE",
]
marker_rows = serial.loc[serial["eventType"].eq("MARKER")]
marker_index = {}
for name in required_markers:
    matches = marker_rows.loc[marker_rows["reason"].eq(name), "serial_index"]
    assert len(matches) == 1, f"expected exactly one {name} marker"
    marker_index[name] = int(matches.iloc[0])
assert list(marker_index.values()) == sorted(marker_index.values())

highs = serial.loc[
    serial["eventType"].eq("CAMERA_HIGH")
    & serial["serial_index"].gt(marker_index["START_SENT"]),
    ["serial_index", "count", "rp2040Time", "hostUnixNs", "hostMonotonicNs"],
].copy()
highs[["count", "rp2040Time"]] = highs[["count", "rp2040Time"]].apply(
    pd.to_numeric, errors="raise"
)
highs = highs.rename(columns={
    "count": "frame_ttl_count",
    "rp2040Time": "frame_rp2040_us",
    "hostUnixNs": "ttl_host_unix_ns",
    "hostMonotonicNs": "ttl_host_monotonic_ns",
})
assert highs["frame_ttl_count"].is_unique

offset = int(frames.iloc[0]["camera_frame_id"] - highs.iloc[0]["frame_ttl_count"])
frames["frame_ttl_count"] = frames["camera_frame_id"] - offset
aligned_frames = frames.merge(
    highs,
    on="frame_ttl_count",
    how="left",
    validate="one_to_one",
    indicator="ttl_join",
)
assert aligned_frames["ttl_join"].eq("both").all()
assert len(aligned_frames) == len(frames) == len(highs)

first_rp2040_us = int(aligned_frames.iloc[0]["frame_rp2040_us"])
aligned_frames["elapsed_s"] = (
    aligned_frames["frame_rp2040_us"] - first_rp2040_us
) / 1_000_000

inference = pd.read_csv(
    RUN / "inference" / "frames.csv",
    usecols=[
        "stream_id", "source_sequence_index", "camera_frame_id",
        "camera_timestamp_ns", "gst_pts_ns", "raw_frame_index",
    ],
)
assert not inference.duplicated(["stream_id", "source_sequence_index"]).any()
inference["inference_present"] = True

aligned_frames = aligned_frames.merge(
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

aligned_frames.to_parquet(OUT / "aligned_frames.parquet", index=False)
```

Associate behavior events with the preceding camera trigger using
`pandas.merge_asof`:

```python
events = serial.loc[
    serial["serial_index"].gt(marker_index["START_SENT"])
    & serial["serial_index"].le(marker_index["CAPTURE_STOP_DONE"])
    & ~serial["eventType"].isin(["CAMERA_HIGH", "CAMERA_LOW"])
].copy()
events["event_rp2040_us"] = pd.to_numeric(
    events["rp2040Time"], errors="coerce"
)
# Startup/status lines without a numeric controller timestamp remain in the
# canonical serial.csv, but cannot participate in controller-time alignment.
events = events.loc[events["event_rp2040_us"].notna()].copy()
events["event_rp2040_us"] = events["event_rp2040_us"].astype(np.int64)
events["pre_capture"] = events["event_rp2040_us"].lt(first_rp2040_us)

frame_clock = aligned_frames[[
    "stream_id", "source_sequence_index", "raw_frame_index",
    "camera_frame_id", "frame_ttl_count", "frame_rp2040_us", "elapsed_s",
]].sort_values("frame_rp2040_us")
events = events.sort_values("event_rp2040_us")

aligned_events = pd.merge_asof(
    events,
    frame_clock,
    left_on="event_rp2040_us",
    right_on="frame_rp2040_us",
    direction="backward",
    allow_exact_matches=True,
)
aligned_events["offset_from_frame_ms"] = (
    aligned_events["event_rp2040_us"]
    - aligned_events["frame_rp2040_us"]
) / 1000
aligned_events.to_parquet(OUT / "aligned_events.parquet", index=False)
```

Run-start acknowledgements can occur after `START_SENT` but before the first
camera trigger. They remain in `aligned_events` with `pre_capture=true` and no
matched frame. Do not force them onto frame zero. Actual behavior events during
capture should have a frame mapping.

The example assumes the current single-camera/controller configuration. For a
future multi-camera controller protocol, perform the as-of join separately per
stream only after the controller events carry an explicit stream identity.

Join objects to the compact frame lookup in pandas. Use `chunksize` if the
object table is large:

```python
frame_lookup = aligned_frames[[
    "stream_id", "source_sequence_index", "raw_frame_index",
    "camera_frame_id", "frame_ttl_count", "frame_rp2040_us", "elapsed_s",
]]

objects_dir = OUT / "aligned_objects.parquet"
objects_dir.mkdir(exist_ok=True)
object_parts = []
for part, objects in enumerate(pd.read_csv(RUN / "objects.csv", chunksize=1_000_000)):
    aligned = objects.merge(
        frame_lookup,
        on=["stream_id", "source_sequence_index"],
        how="left",
        suffixes=("", "_frame"),
        validate="many_to_one",
        indicator=True,
    )
    assert aligned["_merge"].eq("both").all()
    assert aligned["camera_frame_id"].eq(aligned["camera_frame_id_frame"]).all()
    aligned.drop(columns="_merge").to_parquet(
        objects_dir / f"part-{part:05d}.parquet", index=False
    )
    object_parts.append(aligned[[
        "observation_id", "stream_id", "source_sequence_index",
        "camera_frame_id", "raw_frame_index", "frame_ttl_count",
        "frame_rp2040_us", "elapsed_s",
    ]])
```

`aligned_objects.parquet/` is a partitioned Parquet dataset. Pandas can read it
as one logical table with `pd.read_parquet(objects_dir)`. On an analysis device
with enough memory, concatenate the chunks and call `to_parquet` once if a
single physical file is preferable.

Keypoints are normally the largest table. Build an object lookup from the
minimal columns retained in `object_parts`, then process keypoints in chunks:

```python
object_lookup = pd.concat(object_parts, ignore_index=True)
assert object_lookup["observation_id"].is_unique

keypoints_dir = OUT / "aligned_keypoints.parquet"
keypoints_dir.mkdir(exist_ok=True)
for part, keypoints in enumerate(
    pd.read_csv(RUN / "keypoints.csv", chunksize=2_000_000)
):
    aligned = keypoints.merge(
        object_lookup,
        on="observation_id",
        how="left",
        suffixes=("", "_object"),
        validate="many_to_one",
        indicator=True,
    )
    assert aligned["_merge"].eq("both").all()
    for column in ("stream_id", "source_sequence_index", "camera_frame_id"):
        assert aligned[column].eq(aligned[f"{column}_object"]).all()
    aligned.drop(columns="_merge").to_parquet(
        keypoints_dir / f"part-{part:05d}.parquet", index=False
    )
```

Recommended processing order:

1. Stream `serial.csv` into a table while assigning `serial_index`.
2. Validate the epoch markers and create a `camera_high` table.
3. Stream `frames.csv`, calculate `frame_ttl_count`, and create
   `aligned_frames`.
4. Left-join inference admission onto `aligned_frames` by exact identity.
5. As-of join behavior events to `camera_high` by `rp2040Time`.
6. Join objects to frames by exact identity.
7. Join keypoints to objects by `observation_id`, retaining their
   `keypoint_index` ordering.
8. Write derived tables as Parquet with `DataFrame.to_parquet`. Export a CSV
   only when a downstream tool specifically requires it.

Parquet is recommended because it preserves pandas dtypes, supports column
selection, and avoids repeatedly parsing tens of gigabytes of CSV. A directory
of numbered Parquet parts is one logical dataset and keeps peak pandas memory
bounded. The copied Jetson CSVs remain the canonical source evidence.

## 8. Required validation after combining

Record these checks alongside the derived outputs:

- frame rows equal the authoritative `frames.csv` count;
- frame rows equal the validated MP4 sample count in `run_status.json`;
- every frame has exactly one TTL mapping for controller-enabled runs;
- the derived frame order is strictly increasing by `raw_frame_index`;
- camera frame IDs and controller TTL counts are strictly increasing;
- inference joins have no duplicate frame keys;
- every object references an existing inference/camera frame;
- every keypoint references an existing `observation_id`;
- each object/keypoint cross-check field agrees with its parent;
- all input file sizes or hashes used by the analysis are recorded;
- source-run files remain unchanged.

Missing inference for a frame is not automatically a recording failure. It
must be represented explicitly and evaluated according to the scientific
question. Missing canonical frames, missing controller TTLs, or mismatched MP4
samples are acquisition-integrity failures.

## 9. Existing SqueakView audit utility

The repository command below recreates the compact alignment audit on a copied
run:

```bash
python3 scripts/align_run_outputs.py \
  /data/squeakview/source_runs/<run_id> \
  --out-dir /data/squeakview/analysis_results/<run_id>/alignment
```

At present this command writes `alignment_summary.json`; it does not create a
combined analysis table. Use it to verify the copy and controller/frame
mapping before building analysis-specific Parquet or CSV outputs.
