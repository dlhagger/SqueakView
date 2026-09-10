"""Strict, per-record bounded CSV input for long-running evidence files."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path


MAX_CSV_RECORD_BYTES = 256 * 1024


class BoundedCsvError(RuntimeError):
    """A CSV physical record exceeded policy or was not strict UTF-8."""


def bounded_csv_lines(
    path: Path, *, max_record_bytes: int = MAX_CSV_RECORD_BYTES
) -> Iterator[str]:
    """Yield strict UTF-8 physical lines with a fixed upper size bound.

    ``csv`` may combine multiple yielded physical lines into one quoted logical
    record. Bounding each physical line still prevents an individual read from
    allocating memory based on an untrusted, unterminated line.
    """

    if isinstance(max_record_bytes, bool) or not isinstance(max_record_bytes, int):
        raise ValueError("max_record_bytes must be an integer")
    if max_record_bytes < 1:
        raise ValueError("max_record_bytes must be positive")
    path = Path(path)
    with path.open("rb") as handle:
        line_number = 0
        while True:
            raw = handle.readline(max_record_bytes + 1)
            if not raw:
                return
            line_number += 1
            if len(raw) > max_record_bytes:
                raise BoundedCsvError(
                    f"CSV record exceeds {max_record_bytes} byte limit at "
                    f"{path.name}:{line_number}"
                )
            try:
                yield raw.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise BoundedCsvError(
                    f"CSV input is not strict UTF-8 at {path.name}:{line_number}"
                ) from exc
