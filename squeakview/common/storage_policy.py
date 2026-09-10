"""Shared startup and capture-time storage reserve policy."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass


DEFAULT_MIN_FREE_BYTES = 1_000_000_000
DEFAULT_CHECK_INTERVAL_S = 5.0


@dataclass(frozen=True, slots=True)
class StorageReservePolicy:
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES
    check_interval_s: float = DEFAULT_CHECK_INTERVAL_S


def resolve_storage_reserve_policy(
    environ: Mapping[str, str] | None = None,
) -> StorageReservePolicy:
    values = os.environ if environ is None else environ
    try:
        minimum = max(
            1,
            int(values.get("SQUEAKVIEW_MIN_RUN_FREE_BYTES", DEFAULT_MIN_FREE_BYTES)),
        )
    except (TypeError, ValueError):
        minimum = DEFAULT_MIN_FREE_BYTES
    try:
        raw_interval = float(
            values.get(
                "SQUEAKVIEW_STORAGE_CHECK_INTERVAL_S",
                DEFAULT_CHECK_INTERVAL_S,
            )
        )
        if not math.isfinite(raw_interval):
            raise ValueError("interval must be finite")
        interval = max(0.1, raw_interval)
    except (TypeError, ValueError):
        interval = DEFAULT_CHECK_INTERVAL_S
    return StorageReservePolicy(min_free_bytes=minimum, check_interval_s=interval)


__all__ = [
    "DEFAULT_CHECK_INTERVAL_S",
    "DEFAULT_MIN_FREE_BYTES",
    "StorageReservePolicy",
    "resolve_storage_reserve_policy",
]
