from __future__ import annotations

"""Versioned project metadata and non-destructive project creation."""

import json
import os
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from squeakview.common.bounded_input import read_json_object

from .paths import AppPaths, PROJECT_METADATA_FILENAME, ProjectPaths


PROJECT_SCHEMA_VERSION = 1
MAX_PROJECT_METADATA_BYTES = 64 * 1024
MAX_PROJECT_NAME_CHARS = 128
_PROJECT_DIRECTORIES = (
    "runs",
    "models",
    "model_sources",
    "tasks",
    "profiles",
    "profiles/experiments",
    "profiles/subjects",
    "qualification",
)
_METADATA_FIELDS = {
    "schema_version",
    "project_id",
    "name",
    "created_at",
    "default_model",
    "default_task",
}


def _validate_name(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("project name must be a string")
    name = value.strip()
    if not name or len(name) > MAX_PROJECT_NAME_CHARS:
        raise ValueError(
            f"project name must contain 1 to {MAX_PROJECT_NAME_CHARS} characters"
        )
    if any(ord(character) < 0x20 for character in name):
        raise ValueError("project name may not contain control characters")
    return name


def _validate_project_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("project_id must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError("project_id must be a valid UUID") from exc
    if str(parsed) != value:
        raise ValueError("project_id must use canonical UUID syntax")
    return str(parsed)


def _validate_timestamp(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("created_at must be an RFC-3339 timestamp string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("created_at must be a valid RFC-3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("created_at must include a timezone")
    return value


def _validate_optional_slug(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string or null")
    candidate = value.strip()
    path = PurePosixPath(candidate)
    if path.is_absolute() or len(path.parts) != 1 or path.parts[0] in {".", ".."}:
        raise ValueError(f"{field} must be one project-local name")
    return candidate


@dataclass(frozen=True, slots=True)
class ProjectMetadata:
    schema_version: int
    project_id: str
    name: str
    created_at: str
    default_model: str | None = None
    default_task: str | None = "default.yaml"

    @classmethod
    def create(cls, name: str) -> "ProjectMetadata":
        return cls(
            schema_version=PROJECT_SCHEMA_VERSION,
            project_id=str(uuid.uuid4()),
            name=_validate_name(name),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    @classmethod
    def from_mapping(cls, payload: dict[str, object]) -> "ProjectMetadata":
        unknown = sorted(set(payload) - _METADATA_FIELDS)
        missing = sorted(_METADATA_FIELDS - set(payload))
        if unknown or missing:
            problems = []
            if unknown:
                problems.append(f"unknown fields: {', '.join(unknown)}")
            if missing:
                problems.append(f"missing fields: {', '.join(missing)}")
            raise ValueError("project metadata has invalid fields; " + "; ".join(problems))
        schema = payload["schema_version"]
        if isinstance(schema, bool) or not isinstance(schema, int):
            raise ValueError("project schema_version must be an integer")
        if schema != PROJECT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported project schema {schema}; expected {PROJECT_SCHEMA_VERSION}"
            )
        return cls(
            schema_version=schema,
            project_id=_validate_project_id(payload["project_id"]),
            name=_validate_name(payload["name"]),
            created_at=_validate_timestamp(payload["created_at"]),
            default_model=_validate_optional_slug(
                payload["default_model"], field="default_model"
            ),
            default_task=_validate_optional_slug(
                payload["default_task"], field="default_task"
            ),
        )


@dataclass(frozen=True, slots=True)
class Project:
    paths: ProjectPaths
    metadata: ProjectMetadata


def _write_new_metadata(path: Path, metadata: ProjectMetadata) -> None:
    encoded = (
        json.dumps(asdict(metadata), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_PROJECT_METADATA_BYTES:
        raise ValueError("project metadata exceeds its size limit")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _replace_metadata(path: Path, metadata: ProjectMetadata) -> None:
    """Crash-safely replace existing project metadata with private permissions."""

    encoded = (
        json.dumps(asdict(metadata), indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_PROJECT_METADATA_BYTES:
        raise ValueError("project metadata exceeds its size limit")
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _create_empty_ownership_lock(path: Path) -> None:
    """Publish the private lock inode as part of the complete project layout."""

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def open_project(root: Path) -> Project:
    paths = ProjectPaths.from_existing_root(root)
    payload = read_json_object(
        paths.metadata,
        max_bytes=MAX_PROJECT_METADATA_BYTES,
        label="SqueakView project metadata",
    )
    metadata = ProjectMetadata.from_mapping(payload)
    paths.validate_layout()
    if metadata.default_task is not None:
        task = paths.managed_path(
            Path("tasks") / metadata.default_task,
            must_exist=True,
        )
        if not task.is_file():
            raise ValueError("project default task must be a regular file")
    if metadata.default_model is not None:
        model = paths.managed_path(
            Path("models") / metadata.default_model,
            must_exist=True,
        )
        if not model.is_dir():
            raise ValueError("project default model must be a directory")
        config = paths.managed_path(
            Path("models")
            / metadata.default_model
            / "configs"
            / f"{metadata.default_model}.txt",
            must_exist=True,
        )
        manifest = paths.managed_path(
            Path("models") / metadata.default_model / "model.yaml",
            must_exist=True,
        )
        if not config.is_file() or not manifest.is_file():
            raise ValueError(
                "project default model must contain its config and model manifest"
            )
    return Project(paths=paths, metadata=metadata)


def set_default_model(project: Project, model_name: str | None) -> Project:
    """Persist a validated project-local default model while its lock is held.

    The caller owns serialization through :class:`ProjectSession`.  Keeping the
    write here makes the package publication and the metadata contract share the
    same strict path validation used when reopening the project.
    """

    validated_name = _validate_optional_slug(model_name, field="default_model")
    if validated_name is not None:
        package = project.paths.managed_path(
            Path("models") / validated_name,
            must_exist=True,
        )
        if not package.is_dir():
            raise ValueError("project default model must be a directory")
        config = project.paths.managed_path(
            Path("models")
            / validated_name
            / "configs"
            / f"{validated_name}.txt",
            must_exist=True,
        )
        manifest = project.paths.managed_path(
            Path("models") / validated_name / "model.yaml",
            must_exist=True,
        )
        if not config.is_file() or not manifest.is_file():
            raise ValueError(
                "project default model must contain its config and model manifest"
            )
    metadata = ProjectMetadata(
        schema_version=project.metadata.schema_version,
        project_id=project.metadata.project_id,
        name=project.metadata.name,
        created_at=project.metadata.created_at,
        default_model=validated_name,
        default_task=project.metadata.default_task,
    )
    _replace_metadata(project.paths.metadata, metadata)
    return open_project(project.paths.root)


def create_project(root: Path, *, name: str, app: AppPaths) -> Project:
    """Create one project in a staged sibling and publish it once complete."""

    destination = Path(root).expanduser()
    parent = destination.parent.resolve(strict=True)
    destination = parent / destination.name
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"project destination already exists: {destination}")
    metadata = ProjectMetadata.create(name)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.creating-",
            dir=parent,
        )
    )
    published = False
    try:
        staging.chmod(0o700)
        for relative in _PROJECT_DIRECTORIES:
            (staging / relative).mkdir(parents=True, exist_ok=True, mode=0o700)
        template = app.project_template
        if not template.is_dir():
            raise ValueError(f"application project template is missing: {template}")
        for source in template.rglob("*"):
            if source.is_symlink():
                raise ValueError("application project template may not contain symlinks")
            relative = source.relative_to(template)
            target = staging / relative
            if source.is_dir():
                target.mkdir(parents=True, exist_ok=True, mode=0o700)
            elif source.is_file():
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                shutil.copyfile(source, target)
                target.chmod(0o600)
            else:
                raise ValueError(f"unsupported project template entry: {source}")
        _write_new_metadata(staging / PROJECT_METADATA_FILENAME, metadata)
        _create_empty_ownership_lock(ProjectPaths(staging).ownership_lock)
        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        # Publishing is serialized by a private parent-directory creation lock
        # in the GUI service. This final same-filesystem rename makes a fully
        # populated project appear at once.
        staging.rename(destination)
        published = True
        return open_project(destination)
    finally:
        if not published:
            shutil.rmtree(staging, ignore_errors=True)


__all__ = [
    "MAX_PROJECT_METADATA_BYTES",
    "PROJECT_METADATA_FILENAME",
    "PROJECT_SCHEMA_VERSION",
    "Project",
    "ProjectMetadata",
    "create_project",
    "open_project",
    "set_default_model",
]
