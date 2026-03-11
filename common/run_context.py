"""Shared run-directory utilities for the Jetson capture suite."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from squeakview import config as squeakview_config

RUNS_DIR = squeakview_config.ensure_runs_dir()
RUN_MARKER = RUNS_DIR / ".latest_run"


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


def timestamped_run_dir(prefix: str | None = None, *, random_suffix: bool = True) -> Path:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{prefix}_{ts}" if prefix else ts
    if random_suffix:
        name = f"{name}_{os.urandom(4).hex()}"
    root = RUNS_DIR / name
    root.mkdir(parents=True, exist_ok=True)
    RUN_MARKER.write_text(str(root))
    return root


@dataclass(slots=True)
class RunArtifacts:
    raw_video: Path
    annotated_video: Path
    detections_csv: Path
    metadata_json: Path
    serial_csv: Path | None = None


def run_artifacts(run_dir: Path, include_serial: bool = True) -> RunArtifacts:
    return RunArtifacts(
        raw_video=run_dir / "raw.mp4",
        annotated_video=run_dir / "annotated.mp4",
        detections_csv=run_dir / "detections.csv",
        metadata_json=run_dir / "camera_settings.json",
        serial_csv=(run_dir / "serial.csv") if include_serial else None,
    )


def metadata_path(run_dir: Path) -> Path:
    return run_dir / "camera_settings.json"


def write_metadata(run_dir: Path, payload: dict[str, Any]) -> Path:
    path = metadata_path(run_dir)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path
