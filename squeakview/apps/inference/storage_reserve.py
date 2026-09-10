"""Capture-owned free-space reserve supervision for long scientific runs."""

from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Callable

from squeakview.common.storage_policy import (
    DEFAULT_CHECK_INTERVAL_S,
    DEFAULT_MIN_FREE_BYTES,
    StorageReservePolicy,
    resolve_storage_reserve_policy,
)


class StorageReserveMonitor:
    """Request graceful fail-closed capture shutdown before storage is exhausted."""

    def __init__(
        self,
        run_dir: Path,
        stop_event: threading.Event,
        on_fatal: Callable[[str], None],
        policy: StorageReservePolicy,
        *,
        disk_usage: Callable[[Path], object] = shutil.disk_usage,
    ) -> None:
        self.run_dir = Path(run_dir)
        self._stop_event = stop_event
        self._on_fatal = on_fatal
        self.policy = policy
        self._disk_usage = disk_usage

    def run(self) -> None:
        while not self._stop_event.wait(self.policy.check_interval_s):
            try:
                free_bytes = int(getattr(self._disk_usage(self.run_dir), "free"))
            except Exception as exc:
                self._on_fatal(
                    "could not verify the recording storage reserve: "
                    f"{type(exc).__name__}: {exc}"
                )
                return
            if free_bytes < self.policy.min_free_bytes:
                self._on_fatal(
                    "recording storage reserve exhausted: "
                    f"{free_bytes} bytes free is below the required "
                    f"{self.policy.min_free_bytes} bytes"
                )
                return
