"""Explicit camera losses; never infer an unexplained loss from count differences."""
from __future__ import annotations

import csv
import json
from bisect import bisect_right
from pathlib import Path

from .bounded_csv import bounded_csv_lines
from .run_layout import artifact_path


def gap_events(run_dir: Path, stream_id: int) -> list[dict]:
    suffix = "" if stream_id == 0 else f"_cam{stream_id}"
    path = artifact_path(run_dir, f"diagnostics/errors{suffix}.csv")
    if not path.exists():
        return []
    events = []
    for row in csv.DictReader(bounded_csv_lines(path)):
        if row.get("event_type") != "camera_frame_gap":
            continue
        if int(row["stream_id"]) != stream_id:
            raise ValueError("gap event belongs to a different camera")
        expected, actual = int(row["expected_frame_id"]), int(row["actual_frame_id"])
        detail = json.loads(row["details"])
        if not isinstance(detail, dict) or expected < 0 or actual < 0:
            raise ValueError("invalid camera gap identities or details")
        missing = detail.get("missing_frames")
        if type(missing) is not int or missing <= 0 or actual - expected != missing:
            raise ValueError("camera gap is not an explained forward frame-ID jump")
        events.append(dict(detail, expected_frame_id=expected, actual_frame_id=actual))
        if len(events) > 100_000:
            raise ValueError("camera gap evidence exceeds the bounded audit limit")
    return events


def known_missing_frames(run_dir: Path, stream_id: int) -> int:
    return sum(event["missing_frames"] for event in gap_events(run_dir, stream_id))


class GapIndex:
    """Map a stored frame ordinal to its trigger offset, including known holes."""
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self._streams = {}

    def missing_before(self, stream_id, ordinal):
        if stream_id not in self._streams:
            positions, totals = [], [0]
            for event in gap_events(self.run_dir, stream_id):
                position = event.get("frame_index_after")
                if type(position) is not int or position < 1 or (positions and position <= positions[-1]):
                    raise ValueError("gap has no ordered recorded-frame location")
                positions.append(position)
                totals.append(totals[-1] + event["missing_frames"])
            self._streams[stream_id] = positions, totals
        positions, totals = self._streams[stream_id]
        return totals[bisect_right(positions, ordinal)]

    def trigger_offset(self, stream_id, ordinal):
        return ordinal + self.missing_before(stream_id, ordinal)
