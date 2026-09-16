from __future__ import annotations

"""Advisory ownership-token locks for projects and acquisition hardware."""

import fcntl
import json
import os
import socket
import stat
import time
import uuid
from pathlib import Path
from typing import IO


class OwnershipLock:
    """Hold one process-owned non-blocking lock with diagnostic identity."""

    def __init__(self, path: Path, *, purpose: str) -> None:
        self.path = Path(path)
        self.purpose = str(purpose).strip()
        if not self.purpose:
            raise ValueError("lock purpose must be non-empty")
        self.token = str(uuid.uuid4())
        self._handle: IO[str] | None = None

    @property
    def held(self) -> bool:
        return self._handle is not None

    def acquire(self) -> bool:
        if self._handle is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        handle = os.fdopen(descriptor, "r+", encoding="utf-8")
        try:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("ownership lock must be a regular file")
            if info.st_uid != os.getuid():
                raise PermissionError("ownership lock must be owned by the current user")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return False
        except Exception:
            handle.close()
            raise
        payload = {
            "schema_version": 1,
            "purpose": self.purpose,
            "token": self.token,
            "pid": os.getpid(),
            "uid": os.getuid(),
            "hostname": socket.gethostname(),
            "acquired_at_unix_ns": time.time_ns(),
        }
        handle.seek(0)
        handle.truncate()
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        self._handle = handle
        return True

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> "OwnershipLock":
        if not self.acquire():
            raise RuntimeError(f"another process owns the {self.purpose} lock")
        return self

    def __exit__(self, *_args: object) -> None:
        self.release()


__all__ = ["OwnershipLock"]
