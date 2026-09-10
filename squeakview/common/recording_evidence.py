"""Content identities for the primary scientific recording artifacts."""

from __future__ import annotations

import os
import hashlib
import stat
from pathlib import Path
from typing import Mapping

MAX_RECORDING_CAMERAS = 64


def _artifact_path(run_dir: Path, stream_id: int, kind: str) -> Path:
    if kind == "video":
        return run_dir / ("raw.mp4" if stream_id == 0 else f"raw_cam{stream_id}.mp4")
    if kind == "capture_ledger":
        return run_dir / f"capture_cam{stream_id}.jsonl"
    if kind == "admission_ledger":
        return run_dir / (
            "record_admission.csv"
            if stream_id == 0
            else f"record_admission_cam{stream_id}.csv"
        )
    raise ValueError(f"unsupported recording artifact kind: {kind}")


def _direct_regular_identity(run_dir: Path, path: Path) -> dict[str, object]:
    """Hash a direct run-local regular file without accepting symlinks."""

    root = run_dir.resolve()
    unresolved = path.absolute()
    try:
        metadata = unresolved.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("not a direct regular file")
        resolved = unresolved.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError("artifact escapes run directory")
    except (OSError, ValueError) as exc:
        return {"path": str(unresolved), "available": False, "error": str(exc)}
    try:
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("not a regular file")
            remaining = int(before.st_size)
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise OSError("file became shorter while hashing")
                digest.update(chunk)
                remaining -= len(chunk)
            if handle.read(1):
                raise OSError("file grew while hashing")
            after = os.fstat(handle.fileno())
        final = resolved.stat()
        identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, name) != getattr(after, name) for name in identity_fields) or any(
            getattr(before, name) != getattr(final, name) for name in identity_fields
        ):
            raise OSError("file changed while hashing")
    except (OSError, ValueError) as exc:
        return {"path": str(resolved), "available": False, "error": str(exc)}
    return {
        "path": str(resolved),
        "available": True,
        "size_bytes": int(before.st_size),
        "mtime_ns": int(before.st_mtime_ns),
        "ctime_ns": int(before.st_ctime_ns),
        "device": int(before.st_dev),
        "inode": int(before.st_ino),
        "sha256": digest.hexdigest(),
    }


def _bounded_matching_names(run_dir: Path, pattern: str, limit: int) -> tuple[set[str], bool]:
    names: set[str] = set()
    overflow = False
    try:
        with os.scandir(run_dir) as entries:
            for entry in entries:
                if not Path(entry.name).match(pattern):
                    continue
                if len(names) >= limit:
                    overflow = True
                    break
                names.add(entry.name)
    except OSError:
        overflow = True
    return names, overflow


def capture_recording_evidence(run_dir: Path, camera_count: int) -> dict[str, object]:
    """Hash the exact video/source/admission set validated for one run."""

    if type(camera_count) is not int or not 1 <= camera_count <= MAX_RECORDING_CAMERAS:
        raise ValueError(
            f"camera_count must be an integer from 1 through {MAX_RECORDING_CAMERAS}"
        )
    root = Path(run_dir).resolve()
    cameras = []
    expected: dict[str, set[str]] = {
        kind: {
            _artifact_path(root, stream_id, kind).name
            for stream_id in range(camera_count)
        }
        for kind in ("video", "capture_ledger", "admission_ledger")
    }
    patterns = {
        "video": "raw*.mp4",
        "capture_ledger": "capture_cam*.jsonl",
        "admission_ledger": "record_admission*.csv",
    }
    sets_valid = True
    for kind, pattern in patterns.items():
        observed, overflow = _bounded_matching_names(root, pattern, camera_count + 1)
        if overflow or observed != expected[kind]:
            sets_valid = False
    for stream_id in range(camera_count):
        cameras.append(
            {
                "stream_id": stream_id,
                **{
                    kind: _direct_regular_identity(
                        root, _artifact_path(root, stream_id, kind)
                    )
                    for kind in ("video", "capture_ledger", "admission_ledger")
                },
            }
        )
    return {
        "schema_version": "1.0",
        "camera_count": camera_count,
        "artifact_sets_exact": sets_valid,
        "cameras": cameras,
    }


def recording_evidence_complete(evidence: object, camera_count: int) -> bool:
    if not isinstance(evidence, Mapping):
        return False
    cameras = evidence.get("cameras")
    return bool(
        evidence.get("schema_version") == "1.0"
        and evidence.get("camera_count") == camera_count
        and evidence.get("artifact_sets_exact") is True
        and isinstance(cameras, list)
        and len(cameras) == camera_count
        and all(
            isinstance(camera, Mapping)
            and camera.get("stream_id") == stream_id
            and all(
                isinstance(camera.get(kind), Mapping)
                and camera[kind].get("available") is True
                and type(camera[kind].get("size_bytes")) is int
                and camera[kind]["size_bytes"] > 0
                and isinstance(camera[kind].get("sha256"), str)
                and len(camera[kind]["sha256"]) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in camera[kind]["sha256"]
                )
                for kind in ("video", "capture_ledger", "admission_ledger")
            )
            for stream_id, camera in enumerate(cameras)
        )
    )


def recording_evidence_matches(
    run_dir: Path, evidence: object, camera_count: int
) -> bool:
    """Rehash primary artifacts and compare them with finalization evidence."""

    if not recording_evidence_complete(evidence, camera_count):
        return False
    current = capture_recording_evidence(run_dir, camera_count)
    return recording_evidence_same_content(evidence, current, camera_count)


def recording_evidence_same_content(
    expected: object, current: object, camera_count: int
) -> bool:
    """Compare two complete snapshots by paths, sizes, and content hashes."""

    if not recording_evidence_complete(expected, camera_count):
        return False
    if not recording_evidence_complete(current, camera_count):
        return False
    expected_cameras = expected["cameras"]
    current_cameras = current["cameras"]
    for expected, observed in zip(expected_cameras, current_cameras, strict=True):
        for kind in ("video", "capture_ledger", "admission_ledger"):
            if any(
                expected[kind].get(field) != observed[kind].get(field)
                for field in ("path", "available", "size_bytes", "sha256")
            ):
                return False
    return True


def recording_evidence_metadata_matches(
    run_dir: Path, evidence: object, camera_count: int
) -> bool:
    """Cheaply prove that files have not changed since a content snapshot."""

    if not recording_evidence_complete(evidence, camera_count):
        return False
    root = Path(run_dir).resolve()
    expected_sets = {
        kind: {
            _artifact_path(root, stream_id, kind).name
            for stream_id in range(camera_count)
        }
        for kind in ("video", "capture_ledger", "admission_ledger")
    }
    patterns = {
        "video": "raw*.mp4",
        "capture_ledger": "capture_cam*.jsonl",
        "admission_ledger": "record_admission*.csv",
    }
    for kind, pattern in patterns.items():
        observed, overflow = _bounded_matching_names(root, pattern, camera_count + 1)
        if overflow or observed != expected_sets[kind]:
            return False
    for stream_id, camera in enumerate(evidence["cameras"]):
        for kind in ("video", "capture_ledger", "admission_ledger"):
            path = _artifact_path(root, stream_id, kind)
            expected = camera[kind]
            try:
                metadata = path.lstat()
                resolved = path.resolve(strict=True)
            except OSError:
                return False
            if not stat.S_ISREG(metadata.st_mode) or not resolved.is_relative_to(root):
                return False
            if any(
                expected.get(field) != value
                for field, value in (
                    ("path", str(resolved)),
                    ("available", True),
                    ("size_bytes", int(metadata.st_size)),
                    ("mtime_ns", int(metadata.st_mtime_ns)),
                    ("ctime_ns", int(metadata.st_ctime_ns)),
                    ("device", int(metadata.st_dev)),
                    ("inode", int(metadata.st_ino)),
                )
            ):
                return False
    return True


__all__ = [
    "MAX_RECORDING_CAMERAS",
    "capture_recording_evidence",
    "recording_evidence_complete",
    "recording_evidence_matches",
    "recording_evidence_metadata_matches",
    "recording_evidence_same_content",
]
