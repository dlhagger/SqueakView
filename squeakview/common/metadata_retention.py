"""Compact successful runs without discarding scientific recovery evidence."""
from __future__ import annotations

import gzip
import hashlib
import os
from pathlib import Path
import stat
import tempfile

from .run_layout import artifact_path, uses_compact_metadata


def _digest(path, *, compressed=False):
    digest = hashlib.sha256()
    size = 0
    opener = gzip.open if compressed else open
    with opener(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _compress(path, destination):
    """Verify decompressed bytes before retiring the plain recovery ledger."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"recovery artifact must be a regular file: {path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".gz.tmp", dir=destination.parent)
    temporary = Path(name)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as target:
            with path.open("rb") as source, gzip.GzipFile(filename="", mode="wb", fileobj=target, compresslevel=1, mtime=0) as output:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        restored, size = _digest(temporary, compressed=True)
        after = path.lstat()
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if restored != digest.hexdigest() or size != before.st_size or any(getattr(before,k)!=getattr(after,k) for k in fields):
            raise RuntimeError(f"recovery artifact changed or compression verification failed: {path.name}")
        encoded, encoded_size = _digest(temporary)
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
            path.unlink()
            source_directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(source_directory)
            finally:
                os.close(source_directory)
        finally:
            os.close(directory)
        return {"compression":"gzip", "uncompressed_size_bytes":size,
                "stored_size_bytes":encoded_size, "uncompressed_sha256":restored,
                "stored_sha256":encoded}
    finally:
        temporary.unlink(missing_ok=True)


def compact_successful_run(run_dir: Path, camera_count: int):
    """Compress redundant ledgers only after all primary timing work passed.

    Debug, qualification, historical, and validation-failed runs retain their
    original evidence. Originals are not discarded until gzip round-trip and
    source-identity verification both succeed.
    """
    from . import run_context
    run_dir = Path(run_dir)
    if not uses_compact_metadata(run_dir):
        return
    status = run_context.read_json_required(run_context.status_path(run_dir))
    if (status.get("overall_validation_passed") is not True or
        status.get("state") not in {"post_run_complete", "analysis_complete", "finalized"}):
        return
    manifest = run_context.read_json(run_context.manifest_path(run_dir))
    if (manifest.get("qualification") or manifest.get("failure_injection") or
        manifest.get("observability", {}).get("deepstream_debug_profile") or
        (status.get("recording_validation") or {}).get("evidence")):
        return
    retained = dict((status.get("metadata_retention") or {}).get("artifacts", {}))
    names = [f"capture_cam{index}.jsonl" for index in range(camera_count)]
    names += ["record_admission.csv" if index == 0 else f"record_admission_cam{index}.csv" for index in range(camera_count)]
    names += ["inference/frames.csv"]
    for name in names:
        path = artifact_path(run_dir, name)
        if path.is_file() and path.suffix != ".gz":
            destination = run_dir / "_metadata" / "recovery" / (name + ".gz")
            retained[name] = _compress(path, destination)
            run_context.update_status(run_dir, metadata_retention={"policy":"compact_verified_gzip", "complete":False, "artifacts":retained})
        elif path.is_file() and path.suffix == ".gz" and name not in retained:
            decoded, size = _digest(path, compressed=True)
            encoded, stored = _digest(path)
            retained[name] = {"compression":"gzip", "uncompressed_size_bytes":size,
                              "stored_size_bytes":stored, "uncompressed_sha256":decoded, "stored_sha256":encoded}
    run_context.update_status(run_dir, metadata_retention={"policy":"compact_verified_gzip", "complete":True, "artifacts":retained})
    inference = run_dir / "_metadata" / "inference"
    if inference.is_dir() and not any(inference.iterdir()):
        inference.rmdir()
