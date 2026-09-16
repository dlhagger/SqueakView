from __future__ import annotations

"""Per-user recent-project catalog and serialized project creation."""

import re
from pathlib import Path

from squeakview.common.bounded_input import read_json_object
from squeakview.common.run_context import atomic_write_json

from .locking import OwnershipLock
from .metadata import Project, create_project, open_project
from .paths import AppPaths, UserPaths


RECENT_PROJECTS_SCHEMA_VERSION = 1
MAX_RECENT_PROJECTS = 20
MAX_RECENT_PROJECTS_BYTES = 64 * 1024


def project_directory_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    value = re.sub(r"_+", "_", value).strip("._-")
    return value or "SqueakView_Project"


class ProjectCatalog:
    def __init__(self, *, app: AppPaths, user: UserPaths) -> None:
        user.validate_for_app(app)
        self.app = app
        self.user = user

    def ensure_projects_parent(self) -> Path:
        """Create the configured project container once without altering it later."""

        destination = self.user.validate_project_parent(
            self.user.projects_parent,
            self.app,
        )
        if destination.exists():
            if not destination.is_dir():
                raise ValueError(
                    f"project parent exists but is not a directory: {destination}"
                )
            return destination.resolve(strict=True)
        if destination.is_symlink():
            raise ValueError(
                f"project parent is a dangling symbolic link: {destination}"
            )
        try:
            destination.mkdir(parents=True, mode=0o700)
        except FileExistsError:
            # Another first-launch process may have created it concurrently.
            if not destination.is_dir():
                raise ValueError(
                    f"project parent exists but is not a directory: {destination}"
                )
        return destination.resolve(strict=True)

    def recent(self) -> tuple[Project, ...]:
        try:
            payload = read_json_object(
                self.user.recent_projects,
                max_bytes=MAX_RECENT_PROJECTS_BYTES,
                label="recent-project catalog",
            )
            if set(payload) != {"schema_version", "paths"}:
                return ()
            if payload["schema_version"] != RECENT_PROJECTS_SCHEMA_VERSION:
                return ()
            values = payload["paths"]
            if not isinstance(values, list):
                return ()
        except (OSError, ValueError):
            return ()
        projects: list[Project] = []
        seen: set[Path] = set()
        for raw in values[:MAX_RECENT_PROJECTS]:
            if not isinstance(raw, str):
                continue
            try:
                project = open_project(Path(raw))
                self.app.validate_for_project(project.paths)
                self.user.validate_for_project(project.paths)
            except (OSError, ValueError):
                continue
            if project.paths.root in seen:
                continue
            seen.add(project.paths.root)
            projects.append(project)
        return tuple(projects)

    def remember(self, project: Project) -> None:
        self.app.validate_for_project(project.paths)
        self.user.validate_for_project(project.paths)
        self.user.ensure()
        roots = [project.paths.root]
        roots.extend(
            item.paths.root
            for item in self.recent()
            if item.paths.root != project.paths.root
        )
        atomic_write_json(
            self.user.recent_projects,
            {
                "schema_version": RECENT_PROJECTS_SCHEMA_VERSION,
                "paths": [str(path) for path in roots[:MAX_RECENT_PROJECTS]],
            },
        )

    def open(self, root: Path) -> Project:
        project = open_project(root)
        self.app.validate_for_project(project.paths)
        self.user.validate_for_project(project.paths)
        self.remember(project)
        return project

    def create(self, *, name: str, parent: Path | None = None) -> Project:
        self.user.ensure()
        destination_parent = (
            self.ensure_projects_parent()
            if parent is None
            else self.user.validate_project_parent(Path(parent), self.app)
        )
        if parent is not None:
            destination_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        destination_parent = destination_parent.resolve(strict=True)
        lock = OwnershipLock(
            self.user.runtime / "project-creation.lock",
            purpose="project creation",
        )
        if not lock.acquire():
            raise RuntimeError("another SqueakView project creation is in progress")
        try:
            project = create_project(
                destination_parent / project_directory_name(name),
                name=name,
                app=self.app,
            )
            self.remember(project)
            return project
        finally:
            lock.release()


__all__ = [
    "MAX_RECENT_PROJECTS",
    "ProjectCatalog",
    "project_directory_name",
]
