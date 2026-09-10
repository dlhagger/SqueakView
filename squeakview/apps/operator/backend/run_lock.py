"""Process-owned exclusion for scientific acquisition sessions."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path


class AcquisitionLock:
    """Hold an advisory lock for at most one acquisition on this run store."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._handle = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self) -> bool:
        if self._handle is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        self._handle = handle
        return True

    def release(self) -> None:
        handle = self._handle
        self._handle = None
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> "AcquisitionLock":
        if not self.acquire():
            raise RuntimeError("another SqueakView acquisition is active")
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


__all__ = ["AcquisitionLock"]
