from __future__ import annotations

"""Bounded, Qt-free readers for live scientific-run health artifacts."""

import csv
import os
from pathlib import Path


def read_latest_system_telemetry(path: Path) -> dict[str, str] | None:
    """Read the latest complete telemetry row without scanning a long run."""

    path = Path(path)
    try:
        with path.open("rb") as handle:
            header = handle.readline().decode("utf-8", errors="replace").rstrip("\r\n")
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            if not header or end <= len(header) + 1:
                return None
            read_size = min(end, 131_072)
            handle.seek(end - read_size)
            tail = handle.read(read_size)
    except OSError:
        return None

    lines = tail.splitlines()
    if tail and not tail.endswith((b"\n", b"\r")):
        lines = lines[:-1]
    for raw_line in reversed(lines):
        if not raw_line.strip():
            continue
        line = raw_line.decode("utf-8", errors="replace")
        if line == header:
            continue
        try:
            row = next(csv.DictReader([header, line]))
        except (csv.Error, StopIteration):
            return None
        if row.get("schema_version") and row.get("sample_index") is not None:
            return {str(key): "" if value is None else str(value) for key, value in row.items()}
        return None
    return None
