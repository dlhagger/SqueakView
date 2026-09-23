from __future__ import annotations

"""Validated composition of application, project, and user ownership roots."""

from dataclasses import dataclass

from .metadata import Project
from .paths import AppPaths, UserPaths


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    """All path owners for one SqueakView supervisor lifetime."""

    app: AppPaths
    project: Project
    user: UserPaths

    def __post_init__(self) -> None:
        self.app.validate_for_project(self.project.paths)
        self.user.validate_for_project(self.project.paths)
        self.user.validate_for_app(self.app)


__all__ = ["RuntimeContext"]
