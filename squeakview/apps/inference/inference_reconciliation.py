"""Bounded-memory reconciliation of inference and recorded frame identities."""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from squeakview.common.bounded_csv import bounded_csv_lines
from squeakview.common.recording_evidence import MAX_RECORDING_CAMERAS


@dataclass(frozen=True, slots=True)
class InferenceAdmissionResult:
    """Immutable inference-admission counts ready for manifest serialization."""

    captured_frames: tuple[tuple[int, int], ...]
    admitted_frames: tuple[tuple[int, int], ...]
    orphan_frames: int

    @property
    def passed(self) -> bool:
        return self.orphan_frames == 0

    def to_dict(self) -> dict:
        captured = dict(self.captured_frames)
        admitted = dict(self.admitted_frames)
        return {
            "schema_version": "1.0",
            "policy": "capture_non_leaky_inference_leaky_downstream",
            "captured_frames": captured,
            "inference_admitted_frames": admitted,
            "inference_skipped_frames": {
                stream_id: count - admitted.get(stream_id, 0)
                for stream_id, count in captured.items()
            },
            "orphan_inference_frames": self.orphan_frames,
            "passed": self.passed,
        }


def open_index(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        "CREATE TABLE recorded("
        "ordinal INTEGER PRIMARY KEY, stream_id INTEGER NOT NULL, "
        "source_sequence_index INTEGER NOT NULL, camera_frame_id TEXT)"
    )
    connection.execute(
        "CREATE TABLE inferred("
        "stream_id INTEGER NOT NULL, source_sequence_index INTEGER NOT NULL, "
        "PRIMARY KEY(stream_id, source_sequence_index)) WITHOUT ROWID"
    )
    return connection


def index_inference_frames(connection: sqlite3.Connection, path: Path) -> None:
    """Index a validated ledger; duplicates and regressions are fatal."""

    if not path.exists():
        return
    with connection:
        rows: list[tuple[int, int]] = []
        reader = csv.DictReader(bounded_csv_lines(path))
        required = {"stream_id", "source_sequence_index"}
        missing = required.difference(reader.fieldnames or ())
        duplicate_columns = len(reader.fieldnames or ()) != len(
            set(reader.fieldnames or ())
        )
        if missing or duplicate_columns:
            raise RuntimeError(
                f"inference frame ledger {path.name} has invalid columns: "
                + (
                    f"missing {', '.join(sorted(missing))}"
                    if missing
                    else "duplicate column names"
                )
            )
        previous_by_stream: dict[int, int] = {}
        for line_number, row in enumerate(reader, 2):
            try:
                stream_text = row["stream_id"]
                sequence_text = row["source_sequence_index"]
                if any(
                    isinstance(value, str)
                    and value.startswith("-")
                    and value[1:].isascii()
                    and value[1:].isdigit()
                    for value in (stream_text, sequence_text)
                ):
                    raise RuntimeError(
                        f"negative inference frame identity {path.name}:{line_number}"
                    )
                if (
                    not isinstance(stream_text, str)
                    or not stream_text.isascii()
                    or not stream_text.isdigit()
                    or not isinstance(sequence_text, str)
                    or not sequence_text.isascii()
                    or not sequence_text.isdigit()
                ):
                    raise ValueError("frame identity is not an unsigned integer")
                stream_id = int(stream_text)
                sequence = int(sequence_text)
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"invalid inference frame ledger {path.name}:{line_number}"
                ) from exc
            if stream_id >= MAX_RECORDING_CAMERAS:
                raise RuntimeError(
                    f"inference stream ID exceeds supported camera count at "
                    f"{path.name}:{line_number}"
                )
            previous = previous_by_stream.get(stream_id)
            if previous is not None and sequence <= previous:
                raise RuntimeError(
                    "inference frame ledger is duplicate or out of order at "
                    f"{path.name}:{line_number}"
                )
            previous_by_stream[stream_id] = sequence
            rows.append((stream_id, sequence))
            if len(rows) >= 10_000:
                connection.executemany("INSERT INTO inferred VALUES (?, ?)", rows)
                rows.clear()
        if rows:
            connection.executemany("INSERT INTO inferred VALUES (?, ?)", rows)


def summarize_inference_admission(
    connection: sqlite3.Connection,
    recorded_counts: dict[int, int],
) -> InferenceAdmissionResult:
    admitted_counts = {stream_id: 0 for stream_id in recorded_counts}
    query = (
        "SELECT r.stream_id, COUNT(i.stream_id) "
        "FROM recorded r LEFT JOIN inferred i "
        "ON i.stream_id=r.stream_id "
        "AND i.source_sequence_index=r.source_sequence_index "
        "GROUP BY r.stream_id"
    )
    for stream_id, admitted in connection.execute(query):
        admitted_counts[int(stream_id)] = int(admitted)
    orphan_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM inferred i LEFT JOIN recorded r "
            "ON r.stream_id=i.stream_id "
            "AND r.source_sequence_index=i.source_sequence_index "
            "WHERE r.ordinal IS NULL"
        ).fetchone()[0]
    )
    return InferenceAdmissionResult(
        captured_frames=tuple(sorted(recorded_counts.items())),
        admitted_frames=tuple(sorted(admitted_counts.items())),
        orphan_frames=orphan_count,
    )
