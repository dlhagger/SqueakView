from __future__ import annotations

"""Pandas helpers for protocol-v2 controller/camera alignment.

Protocol v2 intentionally records sparse camera timing anchors instead of one
serial message per trigger.  Exact anchor timestamps are retained; timestamps
for intervening frames are explicitly labelled as piecewise interpolations.
"""

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CAMERA_MESSAGE_TYPES = {3, 4, 5}
CAMERA_RECORDS = {"CAMERA_EPOCH", "CAMERA_CHECKPOINT", "CAMERA_STOP"}
EVENT_COLUMNS = (
    "eventType",
    "unixTime",
    "rp2040Time",
    "side",
    "count",
    "duration",
    "latency",
    "value",
    "context",
    "reason",
)


@dataclass(frozen=True, slots=True)
class V2RunData:
    status: dict[str, Any]
    manifest: dict[str, Any]
    transport_summary: dict[str, Any]
    journal: pd.DataFrame
    anchors: pd.DataFrame
    frames: pd.DataFrame
    events: pd.DataFrame


def find_latest_run(
    *,
    project_root: Path | None = None,
    recent_projects_path: Path | None = None,
    projects_parent: Path | None = None,
) -> Path:
    """Find the most recently created run referenced by a project marker."""

    explicit_project = project_root or os.environ.get("SQUEAKVIEW_PROJECT")
    if explicit_project:
        project_roots = [Path(explicit_project).expanduser()]
    else:
        config_root = Path(
            os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
        )
        catalog_path = Path(
            recent_projects_path
            or config_root / "SqueakView" / "recent-projects.json"
        )
        project_roots = []
        try:
            if catalog_path.stat().st_size > 65_536:
                raise ValueError("recent-project catalog is unexpectedly large")
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            if (
                isinstance(catalog, dict)
                and catalog.get("schema_version") == 1
                and isinstance(catalog.get("paths"), list)
            ):
                project_roots.extend(
                    Path(value).expanduser()
                    for value in catalog["paths"]
                    if isinstance(value, str)
                )
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        parent = Path(
            projects_parent
            or os.environ.get(
                "SQUEAKVIEW_PROJECTS_DIR",
                Path.home() / "Documents" / "SqueakView Projects",
            )
        ).expanduser()
        if parent.is_dir():
            project_roots.extend(marker.parent.parent for marker in parent.glob("*/runs/.latest_run"))

    candidates: list[tuple[int, Path]] = []
    seen: set[Path] = set()
    for root in project_roots:
        try:
            root = root.resolve(strict=True)
            if root in seen:
                continue
            seen.add(root)
            runs_dir = (root / "runs").resolve(strict=True)
            marker = runs_dir / ".latest_run"
            if marker.stat().st_size > 4096:
                continue
            run_dir = Path(marker.read_text(encoding="utf-8").strip()).expanduser()
            run_dir = run_dir.resolve(strict=True)
            if not run_dir.is_dir() or not run_dir.is_relative_to(runs_dir):
                continue
            candidates.append((marker.stat().st_mtime_ns, run_dir))
        except (OSError, UnicodeDecodeError, ValueError):
            continue
    if not candidates:
        raise FileNotFoundError(
            "No project run marker was found. Set RUN_DIR explicitly or run "
            "an experiment first."
        )
    return max(candidates, key=lambda item: item[0])[1]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"required v2 run artifact is missing: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_v2_journal(path: Path) -> pd.DataFrame:
    """Load and verify the durable, de-duplicated protocol-v2 journal."""

    path = Path(path)
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in {path} at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"journal row {line_number} is not a JSON object"
                    )
                rows.append(row)
    except FileNotFoundError as exc:
        raise ValueError(f"protocol-v2 journal is missing: {path}") from exc
    if not rows:
        raise ValueError(f"protocol-v2 journal is empty: {path}")

    journal = pd.DataFrame.from_records(rows)
    required = {
        "boot_id",
        "session_id",
        "sequence",
        "message_type",
        "monotonic_us",
        "host_unix_ns",
        "host_monotonic_ns",
        "payload_utf8",
    }
    missing = required.difference(journal.columns)
    if missing:
        raise ValueError(f"protocol-v2 journal is missing columns: {sorted(missing)}")
    for column in required.difference({"payload_utf8"}):
        journal[column] = pd.to_numeric(journal[column], errors="raise").astype(
            np.int64 if column != "boot_id" else object
        )
    if journal["boot_id"].nunique(dropna=False) != 1:
        raise ValueError("protocol-v2 journal crosses a controller boot boundary")
    if journal["sequence"].duplicated().any():
        raise ValueError("protocol-v2 journal contains duplicate stored sequences")
    sequence_steps = journal["sequence"].diff().dropna()
    if not sequence_steps.eq(1).all():
        raise ValueError("protocol-v2 journal contains a durable sequence gap")
    journal.insert(0, "journal_index", np.arange(len(journal), dtype=np.int64))
    return journal


def camera_anchors(journal: pd.DataFrame) -> pd.DataFrame:
    """Extract exact camera epoch/checkpoint/stop anchors from a v2 journal."""

    rows: list[dict[str, Any]] = []
    camera_rows = journal.loc[journal["message_type"].isin(CAMERA_MESSAGE_TYPES)]
    for row in camera_rows.itertuples(index=False):
        fields = str(row.payload_utf8).split(",")
        record = fields[0]
        if record not in CAMERA_RECORDS:
            raise ValueError(f"unexpected v2 camera record: {record!r}")
        parsed: dict[str, str] = {}
        for field in fields[1:]:
            if "=" not in field:
                raise ValueError(f"malformed {record} field: {field!r}")
            key, value = field.split("=", 1)
            if not key or key in parsed:
                raise ValueError(f"duplicate or empty {record} key: {key!r}")
            parsed[key] = value
        required = {
            "count",
            "timestamp_us",
            "period_us",
            "pulse_us",
            "health",
            "queue",
            "suppressed",
            "reason",
        }
        missing = required.difference(parsed)
        if missing:
            raise ValueError(f"{record} is missing fields: {sorted(missing)}")
        rows.append(
            {
                "record": record,
                "count": int(parsed["count"]),
                "timestamp_us": int(parsed["timestamp_us"]),
                "period_us": int(parsed["period_us"]),
                "pulse_us": int(parsed["pulse_us"]),
                "health": int(parsed["health"], 0),
                "queue": parsed["queue"],
                "suppressed": int(parsed["suppressed"]),
                "reason": parsed["reason"],
                "boot_id": row.boot_id,
                "session_id": int(row.session_id),
                "sequence": int(row.sequence),
                "host_unix_ns": int(row.host_unix_ns),
                "host_monotonic_ns": int(row.host_monotonic_ns),
            }
        )
    anchors = pd.DataFrame.from_records(rows)
    if anchors.empty:
        raise ValueError("protocol-v2 journal contains no camera timing anchors")
    if anchors["record"].eq("CAMERA_EPOCH").sum() != 1:
        raise ValueError("expected exactly one CAMERA_EPOCH")
    if anchors["record"].eq("CAMERA_STOP").sum() != 1:
        raise ValueError("expected exactly one CAMERA_STOP")
    epoch_session = int(
        anchors.loc[anchors["record"].eq("CAMERA_EPOCH"), "session_id"].iloc[0]
    )
    anchors = anchors.loc[anchors["session_id"].eq(epoch_session)].copy()
    anchors = anchors.sort_values("count", kind="stable").reset_index(drop=True)
    if anchors["count"].duplicated().any():
        raise ValueError("camera timing anchors contain duplicate counts")
    if not anchors["count"].is_monotonic_increasing:
        raise ValueError("camera timing anchor counts are not monotonic")
    if not anchors["timestamp_us"].is_monotonic_increasing:
        raise ValueError("camera timing anchor timestamps are not monotonic")
    if not anchors["period_us"].eq(anchors["period_us"].iloc[0]).all():
        raise ValueError("camera period changed within the acquisition epoch")
    if anchors["suppressed"].ne(0).any():
        raise ValueError("controller reported suppressed camera telemetry")
    return anchors


def controller_events(journal: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    """Return controller messages inside the camera epoch in plotting form."""

    start_sequence = int(anchors["sequence"].min())
    stop_sequence = int(anchors["sequence"].max())
    active = journal.loc[
        journal["sequence"].between(start_sequence, stop_sequence)
    ].copy()
    camera_by_sequence = anchors.set_index("sequence")
    records: list[dict[str, Any]] = []
    for row in active.itertuples(index=False):
        payload = str(row.payload_utf8)
        parsed = next(csv.reader([payload]))
        event_type = parsed[0] if parsed else ""
        record: dict[str, Any] = {
            "eventType": event_type,
            "unixTime": np.nan,
            "rp2040Time": int(row.monotonic_us),
            "side": np.nan,
            "count": np.nan,
            "duration": np.nan,
            "latency": np.nan,
            "value": np.nan,
            "context": np.nan,
            "reason": np.nan,
            "journal_index": int(row.journal_index),
            "sequence": int(row.sequence),
            "message_type": int(row.message_type),
            "boot_id": row.boot_id,
            "session_id": int(row.session_id),
            "hostUnixNs": int(row.host_unix_ns),
            "hostMonotonicNs": int(row.host_monotonic_ns),
            "rawLine": payload,
        }
        if int(row.message_type) == 1 and len(parsed) >= 2:
            values = parsed[:9]
            values.append(",".join(parsed[9:]) if len(parsed) > 9 else "")
            for name, value in zip(EVENT_COLUMNS, values, strict=True):
                record[name] = value
        if int(row.sequence) in camera_by_sequence.index:
            anchor = camera_by_sequence.loc[int(row.sequence)]
            record["rp2040Time"] = int(anchor["timestamp_us"])
            record["count"] = int(anchor["count"])
            record["reason"] = anchor["reason"]
        records.append(record)
    events = pd.DataFrame.from_records(records)
    for column in (
        "unixTime",
        "rp2040Time",
        "count",
        "duration",
        "latency",
        "value",
    ):
        events[column] = pd.to_numeric(events[column], errors="coerce")
    epoch_us = int(anchors.iloc[0]["timestamp_us"])
    events["event_time_s"] = (events["rp2040Time"] - epoch_us) / 1_000_000.0
    return events


def align_frames_v2(frames: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    """Map single-camera frame ordinals onto sparse v2 camera anchors."""

    required = {"stream_id", "source_sequence_index", "raw_frame_index"}
    missing = required.difference(frames.columns)
    if missing:
        raise ValueError(f"frames.csv is missing columns: {sorted(missing)}")
    if frames.empty:
        raise ValueError("frames.csv contains no recorded frames")
    if frames["stream_id"].nunique() != 1:
        raise ValueError("protocol-v2 visualization currently requires one camera")
    aligned = frames.sort_values(
        ["stream_id", "source_sequence_index"], kind="stable"
    ).reset_index(drop=True)
    source_index = pd.to_numeric(
        aligned["source_sequence_index"], errors="raise"
    ).astype(np.int64)
    expected_index = np.arange(len(aligned), dtype=np.int64)
    if not np.array_equal(source_index.to_numpy(), expected_index):
        raise ValueError("source_sequence_index is not contiguous from zero")
    raw_index = pd.to_numeric(aligned["raw_frame_index"], errors="raise").astype(
        np.int64
    )
    if not np.array_equal(raw_index.to_numpy(), expected_index):
        raise ValueError("raw_frame_index is not contiguous from zero")

    epoch_count = int(anchors.iloc[0]["count"])
    stop_count = int(anchors.iloc[-1]["count"])
    if str(anchors.iloc[0]["record"]) != "CAMERA_EPOCH":
        raise ValueError("first v2 camera anchor is not CAMERA_EPOCH")
    if str(anchors.iloc[-1]["record"]) != "CAMERA_STOP":
        raise ValueError("last v2 camera anchor is not CAMERA_STOP")
    expected_stop = epoch_count + len(aligned) - 1
    if stop_count != expected_stop:
        raise ValueError(
            "controller/frame count mismatch: "
            f"CAMERA_STOP={stop_count}, expected={expected_stop}"
        )

    controller_count = epoch_count + expected_index
    anchor_count = anchors["count"].to_numpy(dtype=np.int64)
    anchor_time = anchors["timestamp_us"].to_numpy(dtype=np.int64)
    estimated_time = np.rint(
        np.interp(controller_count, anchor_count, anchor_time)
    ).astype(np.int64)
    insertion = np.searchsorted(anchor_count, controller_count, side="left")
    exact = (insertion < len(anchor_count)) & (
        anchor_count[np.minimum(insertion, len(anchor_count) - 1)] == controller_count
    )
    right_index = np.minimum(insertion, len(anchor_count) - 1)
    left_index = np.maximum(right_index - (~exact).astype(np.int64), 0)

    aligned["controller_count"] = controller_count
    aligned["frame_controller_us"] = estimated_time
    aligned["controller_time_method"] = np.where(
        exact, "exact_anchor", "piecewise_interpolated"
    )
    aligned["controller_anchor_left_count"] = anchor_count[left_index]
    aligned["controller_anchor_right_count"] = anchor_count[right_index]
    aligned["controller_anchor_span_frames"] = (
        anchor_count[right_index] - anchor_count[left_index]
    )
    aligned["frame_time_s"] = (
        estimated_time - int(anchors.iloc[0]["timestamp_us"])
    ) / 1_000_000.0
    aligned["video_frame_index"] = expected_index
    return aligned


def associate_events_to_frames(
    events: pd.DataFrame, frames: pd.DataFrame
) -> pd.DataFrame:
    """Associate each controller event with its preceding reconstructed frame."""

    event_rows = events.loc[~events["eventType"].isin(CAMERA_RECORDS)].copy()
    event_rows = event_rows.dropna(subset=["rp2040Time"]).sort_values("rp2040Time")
    frame_lookup = frames[
        [
            "source_sequence_index",
            "raw_frame_index",
            "camera_frame_id",
            "controller_count",
            "frame_controller_us",
            "controller_time_method",
        ]
    ].sort_values("frame_controller_us")
    associated = pd.merge_asof(
        event_rows,
        frame_lookup,
        left_on="rp2040Time",
        right_on="frame_controller_us",
        direction="backward",
    )
    associated["offset_from_frame_ms"] = (
        associated["rp2040Time"] - associated["frame_controller_us"]
    ) / 1_000.0
    return associated


def load_v2_run(run_dir: Path) -> V2RunData:
    """Load a finalized production-v2 run and construct its analysis timeline."""

    run_dir = Path(run_dir).expanduser().resolve()
    status = _read_json(run_dir / "run_status.json")
    if status.get("state") != "finalized":
        raise ValueError(f"run is not finalized: state={status.get('state')!r}")
    if status.get("overall_validation_passed") is not True:
        raise ValueError("run did not pass overall acquisition validation")
    if status.get("recording_validation_passed") is not True:
        raise ValueError("run did not pass recording validation")

    manifest = _read_json(run_dir / "run_manifest.json")
    serial = manifest.get("serial", {})
    if serial.get("controller_protocol") != "v2":
        raise ValueError("run manifest does not declare controller protocol v2")
    if serial.get("alignment_required") is not False:
        raise ValueError("v2 run unexpectedly requests legacy TTL alignment")

    summary = _read_json(run_dir / "diagnostics" / "controller_v2_summary.json")
    if summary.get("protocol") != "mousehouse_v2":
        raise ValueError("controller transport summary is not protocol v2")
    counts = summary.get("counts", {})
    if summary.get("integrity_latched"):
        raise ValueError("controller transport integrity was latched")
    if int(counts.get("boot_boundaries", 0)) != 0:
        raise ValueError("controller transport crossed a boot boundary")
    if int(counts.get("crc_or_framing_errors", 0)) != 0:
        raise ValueError("controller transport reported CRC/framing errors")
    if int(counts.get("conflicting_duplicates", 0)) != 0:
        raise ValueError("controller transport reported conflicting duplicates")

    journal = load_v2_journal(run_dir / "diagnostics" / "controller_v2.jsonl")
    if int(counts.get("frames_stored", -1)) != len(journal):
        raise ValueError("controller summary/journal stored-frame counts disagree")
    journal_start = summary.get("journal_start_sequence")
    if journal_start is not None and int(journal_start) != int(
        journal.iloc[0]["sequence"]
    ):
        raise ValueError("controller summary/journal start sequences disagree")
    anchors = camera_anchors(journal)
    summary_session = summary.get("session_id")
    if summary_session is not None and int(summary_session) != int(
        anchors.iloc[0]["session_id"]
    ):
        raise ValueError("controller summary/camera session IDs disagree")
    frames = align_frames_v2(
        pd.read_csv(run_dir / "frames.csv", low_memory=False), anchors
    )
    events = controller_events(journal, anchors)
    return V2RunData(
        status=status,
        manifest=manifest,
        transport_summary=summary,
        journal=journal,
        anchors=anchors,
        frames=frames,
        events=events,
    )
