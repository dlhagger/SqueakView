"""Qt-free scientific-project ownership and path primitives."""

from .metadata import (
    PROJECT_METADATA_FILENAME,
    Project,
    ProjectMetadata,
    create_project,
    open_project,
    set_default_model,
)
from .context import RuntimeContext
from .catalog import ProjectCatalog
from .environment import PROJECT_ENV, project_from_environment
from .paths import AppPaths, ProjectPaths, UserPaths, validate_external_output_path
from .locking import OwnershipLock
from .session import ProjectSession

__all__ = [
    "PROJECT_METADATA_FILENAME",
    "AppPaths",
    "OwnershipLock",
    "Project",
    "ProjectCatalog",
    "ProjectMetadata",
    "ProjectPaths",
    "ProjectSession",
    "PROJECT_ENV",
    "RuntimeContext",
    "UserPaths",
    "create_project",
    "open_project",
    "set_default_model",
    "project_from_environment",
    "validate_external_output_path",
]
