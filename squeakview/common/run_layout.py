"""Versioned run paths; old recordings remain readable without migration."""
from __future__ import annotations

import json
import os
from pathlib import Path

PRIMARY_FILES = ("frames.csv", "raw.mp4", "sensors.csv", "detections.csv", "weigh_in.csv")
ALIASES = {"serial.csv": "sensors.csv", "objects.csv": "detections.csv",
           "bottle_measurements.csv": "weigh_in.csv"}
NATIVE_TIMING_CONTRACT = "native_with_sync_anchors_v1"


def uses_native_timing(run_dir: Path) -> bool:
    marker = Path(run_dir) / "_metadata" / "layout.json"
    if not marker.exists():
        return False
    from .run_context import read_json_required
    return read_json_required(marker, max_bytes=4096).get("timing_contract") == NATIVE_TIMING_CONTRACT


def publish_native_timing(run_dir: Path) -> None:
    """Declare native CSV clocks plus read-time mapping; never alter CSV rows."""
    if not is_primary_layout(run_dir):
        return
    from .run_context import read_json_required, atomic_write_json
    marker = Path(run_dir) / "_metadata" / "layout.json"
    payload = read_json_required(marker, max_bytes=4096)
    payload.update(timing_contract=NATIVE_TIMING_CONTRACT,
                   timestamps="native timestamps on every row; controller mappings resolved from synchronization evidence")
    atomic_write_json(marker, payload)


def is_primary_layout(run_dir: Path) -> bool:
    return (Path(run_dir) / "_metadata" / "layout.json").is_file()


def uses_compact_metadata(run_dir: Path) -> bool:
    marker = Path(run_dir) / "_metadata" / "layout.json"
    try:
        if marker.stat().st_size > 4096:
            return False
        return json.loads(marker.read_text()).get("compact_metadata") is True
    except (OSError, ValueError):
        return False


def initialize_layout(run_dir: Path, *, compact_metadata: bool = False) -> None:
    directory = Path(run_dir) / "_metadata"
    directory.mkdir()
    with (directory / "layout.json").open("x") as handle:
        handle.write(json.dumps({"schema_version": 1, "compact_metadata": compact_metadata, "primary_files": PRIMARY_FILES,
            "timing_contract": NATIVE_TIMING_CONTRACT,
            "timestamps": "native timestamps on every row; controller mappings resolved from synchronization evidence"}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    for path in (directory, Path(run_dir)):
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def artifact_path(run_dir: Path, relative: str | Path) -> Path:
    """Resolve an acquisition artifact without relocating historical runs."""
    root, relative = Path(run_dir), Path(relative)
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"artifact must be run-relative: {relative}")
    if not is_primary_layout(root) or relative.parts[0] == "_metadata":
        return root / relative
    if str(relative) == "run_status.json" and uses_compact_metadata(root):
        relative = Path("run_manifest.json")
    relative = Path(ALIASES.get(str(relative), str(relative)))
    if (str(relative) in PRIMARY_FILES or
        (relative.name.startswith(("raw_cam", "frames_cam")) and len(relative.parts) == 1)):
        return root / relative
    target = root / "_metadata" / relative
    if not target.exists():
        compressed = root / "_metadata" / "recovery" / relative.parent / (relative.name + ".gz")
        if compressed.is_file():
            return compressed
        if target.with_name(target.name + ".gz").is_file():
            return target.with_name(target.name + ".gz")
    return target


def open_artifact(path: Path, mode: str = "rb"):
    """Stream a retained artifact, whether plain or losslessly compressed."""
    import gzip
    path = Path(path)
    return gzip.open(path, mode) if path.suffix == ".gz" else path.open(mode)
