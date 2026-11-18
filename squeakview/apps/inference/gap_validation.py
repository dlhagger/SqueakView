"""Validate located camera gaps against native evidence and MP4 time tables."""
from __future__ import annotations

from pathlib import Path
import csv

from squeakview.common.capture_gaps import gap_events
from squeakview.common.capture_health import read_capture_health
from squeakview.common.run_layout import artifact_path
from .mp4_sample_table import read_video_timestamps


def validate_gaps(run_dir: Path, stream_id: int, source_count: int) -> dict:
    report = {"stream_id": stream_id, "missing_frames": 0, "events": [],
              "passed": False, "error": None}
    try:
        health = read_capture_health(run_dir, stream_id)
        events = gap_events(run_dir, stream_id)
        report["events"] = events
        if len(events) != health.get("total_frame_gap_events"):
            raise ValueError("gap events disagree with the source health counter")
        missing = sum(event["missing_frames"] for event in events)
        report["missing_frames"] = missing
        if not events:
            if health.get("total_missing_frames", 0) != 0:
                raise ValueError("source reports missing frames without gap locations")
            report["passed"] = True
            return report
        if (health.get("gap_mapping_validated") is not True
                or health.get("native_timing_validated") is not True
                or type(health.get("total_missing_frames")) is not int
                or health["total_missing_frames"] != missing):
            raise ValueError("source did not validate native gap identities and timestamps")
        expected = {0: 0}
        tolerance_ns = 1_000_000
        previous_after = -1
        for event in events:
            for key in ("frame_index_before", "frame_index_after", "pts_before_ns", "pts_after_ns",
                        "camera_timestamp_before_ns", "camera_timestamp_after_ns"):
                if type(event.get(key)) is not int or event[key] < 0:
                    raise ValueError(f"gap is missing valid {key}")
            before, after = event["frame_index_before"], event["frame_index_after"]
            if not (after == before + 1 and after > previous_after and after < source_count):
                raise ValueError("gap frame indices are inconsistent")
            delta = event["pts_after_ns"] - event["pts_before_ns"]
            if delta <= 0 or delta != event["camera_timestamp_after_ns"] - event["camera_timestamp_before_ns"]:
                raise ValueError("gap timestamps do not preserve the native camera interval")
            tolerance_ns = min(tolerance_ns, delta // (4 * (event["missing_frames"] + 1)))
            for index, timestamp in ((before, event["pts_before_ns"]), (after, event["pts_after_ns"])):
                if index in expected and expected[index] != timestamp:
                    raise ValueError("gap records disagree about a frame timestamp")
                expected[index] = timestamp
            previous_after = after
        path = artifact_path(run_dir, "raw.mp4" if stream_id == 0 else f"raw_cam{stream_id}.mp4")
        scale, timestamps = read_video_timestamps(path, set(expected))
        for index, ns in expected.items():
            # Allow two media-clock ticks, capped at 1 ms and a quarter of
            # a frame interval. A coarse timebase must not hide a dropped frame.
            if abs((timestamps[index] - timestamps[0]) * 1_000_000_000 - ns * scale) > min(2_000_000_000, tolerance_ns * scale):
                raise ValueError(f"MP4 timestamp does not preserve camera gap at frame {index}")
        report.update(passed=True, video_timestamps_validated=True, video_timescale=scale)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, csv.Error) as exc:
        report["error"] = str(exc)
    return report
