"""Shared run-directory utilities for the Jetson capture suite."""
from __future__ import annotations

import json
import csv
import fcntl
import math
import os
import re
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from squeakview import config as squeakview_config

RUNS_DIR = squeakview_config.ensure_runs_dir()
RUN_MARKER = RUNS_DIR / ".latest_run"

RUN_STATUS_FILENAME = "run_status.json"
RUN_MANIFEST_FILENAME = "run_manifest.json"
BOTTLE_SETUP_FILENAME = "bottle_setup.json"
BOTTLE_MEASUREMENTS_FILENAME = "bottle_measurements.csv"
BOTTLE_SUMMARY_FILENAME = "bottle_summary.json"

_STATUS_TIMESTAMP_FIELDS = {
    "created": "created_at",
    "starting": "starting_at",
    "recording": "started_at",
    "stopping": "stopped_at",
    "capture_closed": "capture_closed_at",
    "finalizing": "finalizing_at",
    "finalization_failed": "finalization_failed_at",
    "post_run_complete": "post_run_completed_at",
    "analyzing": "analyzing_at",
    "finalized": "finalized_at",
    "failed": "failed_at",
}
_METADATA_LOCK = threading.RLock()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def slugify(value: str | None, *, fallback: str) -> str:
    text = (value or "").strip()
    if not text:
        return fallback
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    slug = slug.strip("._-")
    return slug or fallback


def atomic_write_text(path: Path, text: str) -> Path:
    """Atomically replace a text file and make the replacement crash-resistant."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_tmp = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return path


def atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    return atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_write_latest(run_dir: Path) -> None:
    atomic_write_text(RUN_MARKER, str(run_dir.resolve()) + "\n")


def latest_run_dir() -> Path | None:
    """Return the most recent run directory recorded by timestamped_run_dir."""
    try:
        text = RUN_MARKER.read_text().strip()
    except FileNotFoundError:
        return None
    if not text:
        return None
    path = Path(text)
    return path if path.exists() else None


def timestamped_run_dir(
    prefix: str | None = None,
    *,
    random_suffix: bool = True,
    parent: Path | None = None,
) -> Path:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{prefix}_{ts}" if prefix else ts
    if random_suffix:
        name = f"{name}_{os.urandom(4).hex()}"
    base = Path(parent) if parent is not None else RUNS_DIR
    base.mkdir(parents=True, exist_ok=True)
    root = base / name
    root.mkdir(parents=True, exist_ok=True)
    _atomic_write_latest(root)
    return root


def create_run_dir(
    *,
    experiment_name: str | None = None,
    mouse_id: str | None = None,
    prefix: str | None = None,
) -> tuple[Path, str]:
    """Create a collision-proof run directory for new local recordings.

    New runs are grouped by experiment and subject:

        runs/<experiment>/<mouse_id>/<timestamp>_<shortid>/

    The final directory is created with exist_ok=False so a run cannot silently
    reuse or truncate a previous run's files.
    """
    experiment_slug = slugify(experiment_name, fallback="default_experiment")
    subject_slug = slugify(mouse_id, fallback="unassigned")
    prefix_slug = slugify(prefix or mouse_id or "run", fallback="run")
    base = RUNS_DIR / experiment_slug / subject_slug
    base.mkdir(parents=True, exist_ok=True)
    for _ in range(100):
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        short_id = os.urandom(3).hex()
        run_id = f"{prefix_slug}_{ts}_{short_id}"
        root = base / run_id
        try:
            root.mkdir(parents=False, exist_ok=False)
        except FileExistsError:
            continue
        _atomic_write_latest(root)
        return root, run_id
    raise FileExistsError(f"could not create unique run directory under {base}")


def assert_runs_dir_ready(min_free_bytes: int | None = None) -> dict[str, Any]:
    """Verify local run storage is writable and has basic free-space headroom."""
    if min_free_bytes is None:
        min_free_bytes = _env_int("SQUEAKVIEW_MIN_RUN_FREE_BYTES", 1_000_000_000)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    probe = RUNS_DIR / f".write_test_{os.getpid()}"
    try:
        probe.write_text("ok")
        probe.unlink(missing_ok=True)
    except Exception as exc:
        raise OSError(f"runs directory is not writable: {RUNS_DIR}: {exc}") from exc
    usage = shutil.disk_usage(RUNS_DIR)
    if usage.free < min_free_bytes:
        raise OSError(f"runs directory low on free space: {usage.free} bytes available at {RUNS_DIR}")
    return {
        "runs_dir": str(RUNS_DIR),
        "free_bytes": int(usage.free),
        "total_bytes": int(usage.total),
    }


@dataclass(slots=True)
class RunArtifacts:
    raw_video: Path
    frames_csv: Path
    drop_events_csv: Path
    objects_csv: Path
    keypoints_csv: Path
    manifest_json: Path
    status_json: Path
    bottle_setup_json: Path
    bottle_measurements_csv: Path
    bottle_summary_json: Path
    serial_csv: Path | None = None


def run_artifacts(run_dir: Path, include_serial: bool = True) -> RunArtifacts:
    return RunArtifacts(
        raw_video=run_dir / "raw.mp4",
        frames_csv=run_dir / "frames.csv",
        drop_events_csv=run_dir / "diagnostics" / "errors.csv",
        objects_csv=run_dir / "objects.csv",
        keypoints_csv=run_dir / "keypoints.csv",
        manifest_json=run_dir / RUN_MANIFEST_FILENAME,
        status_json=run_dir / RUN_STATUS_FILENAME,
        bottle_setup_json=run_dir / BOTTLE_SETUP_FILENAME,
        bottle_measurements_csv=run_dir / BOTTLE_MEASUREMENTS_FILENAME,
        bottle_summary_json=run_dir / BOTTLE_SUMMARY_FILENAME,
        serial_csv=(run_dir / "serial.csv") if include_serial else None,
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return round(number, 6)


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    text = f"{value:.6f}"
    return text.rstrip("0").rstrip(".")


def normalize_bottle_measurements(bottles: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    source = bottles if isinstance(bottles, dict) else {}
    normalized: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        raw = source.get(side)
        raw_side = raw if isinstance(raw, dict) else {}
        normalized[side] = {
            "fluid": str(raw_side.get("fluid") or "").strip(),
            "initial_weight_g": _optional_float(raw_side.get("initial_weight_g")),
            "final_weight_g": _optional_float(raw_side.get("final_weight_g")),
            "note": str(raw_side.get("note") or "").strip(),
        }
    return normalized


def build_bottle_summary(bottles: dict[str, Any] | None, *, updated_at: str | None = None) -> dict[str, Any]:
    normalized = normalize_bottle_measurements(bottles)
    sides: dict[str, dict[str, Any]] = {}
    complete = True
    missing_fields: list[str] = []
    warnings: list[str] = []
    for side, info in normalized.items():
        fluid = str(info["fluid"]).strip()
        initial = info["initial_weight_g"]
        final = info["final_weight_g"]
        intake = round(initial - final, 6) if initial is not None and final is not None else None
        side_missing: list[str] = []
        if not fluid:
            side_missing.append("fluid")
        if initial is None:
            side_missing.append("initial_weight_g")
        if final is None:
            side_missing.append("final_weight_g")
        missing_fields.extend(f"{side}.{field}" for field in side_missing)
        side_complete = not side_missing
        complete = complete and side_complete
        if intake is not None and intake < 0:
            warnings.append(
                f"{side} final weight exceeds initial weight; calculated intake is negative"
            )
        sides[side] = {
            "fluid": fluid,
            "initial_weight_g": initial,
            "final_weight_g": final,
            "intake_g": intake,
            "note": info["note"],
            "complete": side_complete,
        }
    return {
        "schema_version": "1.0",
        "updated_at": updated_at or _now_iso(),
        "complete": complete,
        "missing_fields": missing_fields,
        "warnings": warnings,
        "measurements": BOTTLE_MEASUREMENTS_FILENAME,
        "sides": sides,
    }


def _read_bottle_measurement_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    try:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                side = str(row.get("side") or "").strip().lower()
                phase = str(row.get("phase") or "").strip().lower()
                if side in {"left", "right"} and phase in {"initial", "final"}:
                    rows[(side, phase)] = {str(key): str(value or "") for key, value in row.items()}
    except Exception:
        pass
    return rows


def _same_bottle_measurement(existing: dict[str, str] | None, row: dict[str, str]) -> bool:
    if not existing:
        return False
    existing_weight = _optional_float(existing.get("weight_g"))
    row_weight = _optional_float(row.get("weight_g"))
    return (
        str(existing.get("fluid") or "").strip() == str(row.get("fluid") or "").strip()
        and existing_weight == row_weight
        and str(existing.get("note") or "").strip() == str(row.get("note") or "").strip()
    )


def write_bottle_artifacts(run_dir: Path, bottles: dict[str, Any] | None) -> dict[str, Any]:
    artifacts = run_artifacts(run_dir)
    updated_at = _now_iso()
    updated_at_ns = time.time_ns()
    summary = build_bottle_summary(bottles, updated_at=updated_at)
    setup = {
        "schema_version": "1.0",
        "updated_at": updated_at,
        "measurements": BOTTLE_MEASUREMENTS_FILENAME,
        "summary": BOTTLE_SUMMARY_FILENAME,
        "sides": {
            side: {
                "fluid": info["fluid"],
                "initial_weight_g": info["initial_weight_g"],
                "final_weight_g": info["final_weight_g"],
                "note": info["note"],
            }
            for side, info in summary["sides"].items()
        },
    }

    header = [
        "side",
        "fluid",
        "phase",
        "weight_g",
        "entered_at_iso",
        "entered_at_unix_ns",
        "note",
    ]
    lines: list[str] = []
    rows: list[dict[str, str]] = []
    existing_rows = _read_bottle_measurement_rows(artifacts.bottle_measurements_csv)
    for side, info in summary["sides"].items():
        for phase in ("initial", "final"):
            weight = info[f"{phase}_weight_g"]
            if weight is None:
                continue
            row = {
                "side": side,
                "fluid": str(info["fluid"]),
                "phase": phase,
                "weight_g": _format_optional_float(weight),
                "entered_at_iso": updated_at,
                "entered_at_unix_ns": str(updated_at_ns),
                "note": str(info["note"]),
            }
            existing = existing_rows.get((side, phase))
            if _same_bottle_measurement(existing, row):
                row["entered_at_iso"] = str(existing.get("entered_at_iso") or updated_at)
                row["entered_at_unix_ns"] = str(existing.get("entered_at_unix_ns") or updated_at_ns)
            rows.append(row)

    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=header, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    lines.append(buffer.getvalue())

    atomic_write_json(artifacts.bottle_setup_json, setup)
    atomic_write_text(artifacts.bottle_measurements_csv, "".join(lines))
    atomic_write_json(artifacts.bottle_summary_json, summary)
    return summary


def status_path(run_dir: Path) -> Path:
    return run_dir / RUN_STATUS_FILENAME


def manifest_path(run_dir: Path) -> Path:
    return run_dir / RUN_MANIFEST_FILENAME


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


@contextmanager
def _metadata_file_lock(path: Path):
    """Serialize read-modify-replace metadata updates across threads/processes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with _METADATA_LOCK, lock_path.open("a") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _apply_status_lifecycle_timestamps(payload: dict[str, Any], history: list[Any]) -> None:
    for item in history:
        if not isinstance(item, dict):
            continue
        field = _STATUS_TIMESTAMP_FIELDS.get(str(item.get("state") or ""))
        timestamp = item.get("timestamp")
        if field and timestamp and payload.get(field) is None:
            payload[field] = timestamp


def write_status(run_dir: Path, state: str, **updates: Any) -> Path:
    path = status_path(run_dir)
    with _metadata_file_lock(path):
        payload = read_json(path)
        history = payload.get("history")
        if not isinstance(history, list):
            history = []
        entry = {"state": state, "timestamp": _now_iso()}
        entry.update({key: value for key, value in updates.items() if value is not None})
        history.append(entry)
        payload.update(updates)
        payload["state"] = state
        payload["updated_at"] = entry["timestamp"]
        payload["history"] = history
        _apply_status_lifecycle_timestamps(payload, history)
        return atomic_write_json(path, payload)


def update_status(run_dir: Path, **updates: Any) -> Path:
    path = status_path(run_dir)
    with _metadata_file_lock(path):
        payload = read_json(path)
        payload.update({key: value for key, value in updates.items() if value is not None})
        payload["updated_at"] = _now_iso()
        return atomic_write_json(path, payload)


def write_manifest(run_dir: Path, payload: dict[str, Any]) -> Path:
    path = manifest_path(run_dir)
    with _metadata_file_lock(path):
        return atomic_write_json(path, payload)


def update_manifest(run_dir: Path, **updates: Any) -> Path:
    path = manifest_path(run_dir)
    with _metadata_file_lock(path):
        payload = read_json(path)
        payload.update(updates)
        payload["updated_at"] = _now_iso()
        return atomic_write_json(path, payload)
