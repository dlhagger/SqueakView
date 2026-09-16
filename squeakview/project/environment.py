from __future__ import annotations

"""Strict handoff of the supervisor-owned project to the GUI process."""

import os
from collections.abc import Mapping
from pathlib import Path

from .metadata import Project, open_project


PROJECT_ENV = "SQUEAKVIEW_PROJECT"


def project_from_environment(
    environ: Mapping[str, str] | None = None,
) -> Project:
    values = os.environ if environ is None else environ
    raw = str(values.get(PROJECT_ENV, "")).strip()
    if not raw:
        raise RuntimeError(
            f"{PROJECT_ENV} is required; launch SqueakView through its project launcher"
        )
    return open_project(Path(raw))


__all__ = ["PROJECT_ENV", "project_from_environment"]
