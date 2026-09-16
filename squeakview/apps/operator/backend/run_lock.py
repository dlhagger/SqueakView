"""Process-owned exclusion for scientific acquisition sessions."""

from __future__ import annotations

from pathlib import Path

from squeakview.project.locking import OwnershipLock


class AcquisitionLock:
    """Compatibility facade over the hardened device-runtime ownership lock."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = OwnershipLock(self.path, purpose="scientific acquisition")

    @property
    def held(self) -> bool:
        return self._lock.held

    def acquire(self) -> bool:
        return self._lock.acquire()

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> "AcquisitionLock":
        if not self.acquire():
            raise RuntimeError("another SqueakView acquisition is active")
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


__all__ = ["AcquisitionLock"]
