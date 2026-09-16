from __future__ import annotations

"""Lifetime ownership for one validated SqueakView project."""

from pathlib import Path

from .locking import OwnershipLock
from .metadata import Project, open_project
from .paths import ProjectPaths


class ProjectSession:
    """Keep a project's exclusive writer lock for the complete app session."""

    def __init__(self, project: Project, lock: OwnershipLock) -> None:
        if not lock.held:
            raise ValueError("project session requires an acquired ownership lock")
        self.project = project
        self._lock = lock

    @classmethod
    def open(cls, root: Path) -> "ProjectSession":
        paths = ProjectPaths.from_existing_root(root)
        lock = OwnershipLock(paths.ownership_lock, purpose="project writer")
        if not lock.acquire():
            raise RuntimeError(
                f"another SqueakView process is using project: {paths.root}"
            )
        try:
            project = open_project(paths.root)
            return cls(project, lock)
        except Exception:
            lock.release()
            raise

    @property
    def active(self) -> bool:
        return self._lock.held

    def close(self) -> None:
        self._lock.release()

    def __enter__(self) -> "ProjectSession":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = ["ProjectSession"]
