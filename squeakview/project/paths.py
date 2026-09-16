from __future__ import annotations

"""Explicit ownership boundaries for application, project, and user paths."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


DEFAULT_PROJECTS_DIRECTORY_NAME = "SqueakView Projects"
PROJECT_METADATA_FILENAME = "squeakview_project.json"


def _canonical_existing_directory(path: Path, *, label: str) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} does not exist or cannot be resolved: {exc}") from exc
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a directory: {resolved}")
    return resolved


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_external_output_path(
    path: Path,
    *,
    app: "AppPaths",
    project: "ProjectPaths | None" = None,
    label: str = "runtime output",
) -> Path:
    """Resolve one output and reject writes into app or scientific-project trees."""

    candidate = Path(path).expanduser().resolve()
    app_root = app.root.resolve(strict=True)
    if _contains(app_root, candidate):
        raise ValueError(f"{label} must be outside the application checkout: {candidate}")
    if project is not None:
        project_root = project.root.resolve(strict=True)
        if _contains(project_root, candidate):
            raise ValueError(
                f"{label} must be outside the scientific project: {candidate}"
            )
    return candidate


@dataclass(frozen=True, slots=True)
class AppPaths:
    """Read-mostly files shipped as one replaceable SqueakView release."""

    root: Path

    @classmethod
    def discover(cls) -> "AppPaths":
        return cls.from_root(Path(__file__).resolve().parents[2])

    @classmethod
    def from_root(cls, root: Path) -> "AppPaths":
        return cls(_canonical_existing_directory(root, label="application root"))

    @property
    def resources(self) -> Path:
        return self.root / "resources"

    @property
    def project_template(self) -> Path:
        return self.resources / "project_template"

    @property
    def native(self) -> Path:
        return self.root / "native"

    @property
    def scripts(self) -> Path:
        return self.root / "scripts"

    @property
    def logo(self) -> Path:
        return self.root / "SqueakView_logo.png"

    def validate_for_project(self, project: "ProjectPaths") -> None:
        """Require application and project ownership trees to be disjoint."""

        app_root = self.root.resolve(strict=True)
        project_root = project.root.resolve(strict=True)
        if _contains(app_root, project_root) or _contains(project_root, app_root):
            raise ValueError(
                "application and project roots must be separate, "
                "non-overlapping directories"
            )


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """All durable, project-owned scientific configuration and output paths."""

    root: Path

    @classmethod
    def from_existing_root(cls, root: Path) -> "ProjectPaths":
        return cls(_canonical_existing_directory(root, label="project root"))

    @property
    def metadata(self) -> Path:
        return self.root / PROJECT_METADATA_FILENAME

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def model_sources(self) -> Path:
        return self.root / "model_sources"

    @property
    def tasks(self) -> Path:
        return self.root / "tasks"

    @property
    def profiles(self) -> Path:
        return self.root / "profiles"

    @property
    def qualification(self) -> Path:
        return self.root / "qualification"

    @property
    def ownership_lock(self) -> Path:
        return self.root / ".squeakview.lock"

    def managed_path(
        self,
        relative: str | os.PathLike[str],
        *,
        must_exist: bool = False,
    ) -> Path:
        """Resolve a project-relative path and reject every project escape."""

        raw = Path(relative)
        if raw.is_absolute():
            raise ValueError("managed project paths must be relative")
        if not raw.parts or any(part in {"", ".", ".."} for part in raw.parts):
            raise ValueError("managed project path is empty or contains traversal")
        try:
            candidate = (self.root / raw).resolve(strict=must_exist)
        except OSError as exc:
            raise ValueError(f"managed project path cannot be resolved: {exc}") from exc
        if not _contains(self.root, candidate):
            raise ValueError("managed project path escapes the project root")
        return candidate

    def resolve_path(
        self,
        value: str | os.PathLike[str],
        *,
        within: Path | None = None,
        must_exist: bool = False,
    ) -> Path:
        """Resolve a stored project path and enforce its ownership boundary."""

        raw = Path(value).expanduser()
        if raw.is_absolute():
            try:
                candidate = raw.resolve(strict=must_exist)
            except OSError as exc:
                raise ValueError(f"project path cannot be resolved: {exc}") from exc
            if not _contains(self.root, candidate):
                raise ValueError("project path escapes the project root")
        else:
            candidate = self.managed_path(raw, must_exist=must_exist)
        boundary = (within or self.root).resolve(strict=True)
        if not _contains(self.root, boundary) or not _contains(boundary, candidate):
            raise ValueError(f"project path is outside its required directory: {boundary}")
        return candidate

    def portable_path(self, value: str | os.PathLike[str]) -> str:
        """Return one validated project-root-relative path for durable settings."""

        return self.resolve_path(value).relative_to(self.root).as_posix()

    def validate_layout(self) -> None:
        for name in (
            "runs",
            "models",
            "model_sources",
            "tasks",
            "profiles",
            "qualification",
        ):
            path = self.managed_path(name, must_exist=True)
            if not path.is_dir():
                raise ValueError(f"project entry must be a directory: {name}")
        for name in ("profiles/experiments", "profiles/subjects"):
            path = self.managed_path(name, must_exist=True)
            if not path.is_dir():
                raise ValueError(f"project entry must be a directory: {name}")


@dataclass(frozen=True, slots=True)
class UserPaths:
    """Per-user presentation, persistent operational state, and runtime paths."""

    config: Path
    state: Path
    runtime: Path
    projects_parent: Path

    @classmethod
    def discover(cls, environ: Mapping[str, str] | None = None) -> "UserPaths":
        values = os.environ if environ is None else environ
        raw_home = values.get("HOME")
        home = Path(raw_home).expanduser() if raw_home else Path.home()
        config_base = Path(values.get("XDG_CONFIG_HOME", home / ".config")).expanduser()
        state_base = Path(
            values.get("XDG_STATE_HOME", home / ".local" / "state")
        ).expanduser()
        runtime_base = Path(
            values.get("XDG_RUNTIME_DIR", f"/tmp/squeakview-{os.getuid()}")
        ).expanduser()
        projects_parent = Path(
            values.get(
                "SQUEAKVIEW_PROJECTS_DIR",
                home / "Documents" / DEFAULT_PROJECTS_DIRECTORY_NAME,
            )
        ).expanduser()
        return cls(
            config=(config_base / "SqueakView").resolve(),
            state=(state_base / "SqueakView").resolve(),
            runtime=(runtime_base / "squeakview").resolve(),
            # Preserve the final path component until creation so a dangling
            # storage symlink can be rejected rather than followed implicitly.
            projects_parent=projects_parent.absolute(),
        )

    @property
    def launch_logs(self) -> Path:
        return self.state / "logs"

    @property
    def recent_projects(self) -> Path:
        return self.config / "recent-projects.json"

    @property
    def acquisition_lock(self) -> Path:
        return self.runtime / "acquisition.lock"

    def ensure(self) -> None:
        for path in (self.config, self.state, self.runtime, self.launch_logs):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                path.chmod(0o700)
            except OSError:
                if path == self.runtime:
                    raise

    def validate_for_app(self, app: AppPaths) -> None:
        """Reject user-controlled roots that could write into an app release."""

        app_root = app.root.resolve(strict=True)
        roots = {
            "user configuration": self.config.resolve(),
            "user state": self.state.resolve(),
            "user runtime": self.runtime.resolve(),
            "project parent": self.projects_parent.resolve(),
        }
        for label, root in roots.items():
            if _contains(app_root, root) or _contains(root, app_root):
                raise ValueError(
                    f"{label} and application root must be separate, "
                    "non-overlapping directories"
                )
        user_roots = tuple(roots.items())[:3]
        for index, (left_label, left) in enumerate(user_roots):
            for right_label, right in user_roots[index + 1 :]:
                if _contains(left, right) or _contains(right, left):
                    raise ValueError(
                        f"{left_label} and {right_label} must be separate, "
                        "non-overlapping directories"
                    )
        project_parent = roots["project parent"]
        for label, root in user_roots:
            if _contains(project_parent, root) or _contains(root, project_parent):
                raise ValueError(
                    f"{label} and project parent must be separate, "
                    "non-overlapping directories"
                )

    def validate_project_parent(self, parent: Path, app: AppPaths) -> Path:
        """Validate a proposed existing-or-new project parent without writing it."""

        raw = Path(parent).expanduser().absolute()
        if raw.is_symlink() and not raw.exists():
            raise ValueError(f"project parent is a dangling symbolic link: {raw}")
        candidate = raw.resolve()
        app_root = app.root.resolve(strict=True)
        if _contains(app_root, candidate) or _contains(candidate, app_root):
            raise ValueError(
                "project parent and application root must be separate, "
                "non-overlapping directories"
            )
        for label, root in (
            ("user configuration", self.config.resolve()),
            ("user state", self.state.resolve()),
            ("user runtime", self.runtime.resolve()),
        ):
            if _contains(candidate, root) or _contains(root, candidate):
                raise ValueError(
                    f"project parent and {label} must be separate, "
                    "non-overlapping directories"
                )
        return candidate

    def validate_for_project(self, project: ProjectPaths) -> None:
        """Reject user-controlled writes inside an active scientific project."""

        project_root = project.root.resolve(strict=True)
        for label, root in (
            ("user configuration", self.config.resolve()),
            ("user state", self.state.resolve()),
            ("user runtime", self.runtime.resolve()),
        ):
            if _contains(project_root, root) or _contains(root, project_root):
                raise ValueError(
                    f"{label} and project root must be separate, "
                    "non-overlapping directories"
                )


__all__ = [
    "DEFAULT_PROJECTS_DIRECTORY_NAME",
    "PROJECT_METADATA_FILENAME",
    "AppPaths",
    "ProjectPaths",
    "UserPaths",
    "validate_external_output_path",
]
