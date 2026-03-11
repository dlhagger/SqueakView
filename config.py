from __future__ import annotations

"""Centralized path helpers for the Jetson capture stack."""

import os
from pathlib import Path


def _resolve_workspace() -> Path:
    env_override = os.environ.get("SQUEAKVIEW_WORKSPACE") or os.environ.get("PRODUCT_WORKSPACE")
    if env_override:
        return Path(env_override).expanduser().resolve()
    # config.py lives directly under the workspace root.
    return Path(__file__).resolve().parent


def _resolve_path(env_var: str, legacy_env: str, default: Path) -> Path:
    candidate = os.environ.get(env_var) or os.environ.get(legacy_env)
    if candidate:
        return Path(candidate).expanduser().resolve()
    return default


def _resolve_deepstream(workspace: Path) -> Path:
    env_override = os.environ.get("SQUEAKVIEW_DEEPSTREAM") or os.environ.get("PRODUCT_DEEPSTREAM")
    if env_override:
        return Path(env_override).expanduser().resolve()
    candidate = workspace / "DeepStream-Yolo"
    if candidate.exists():
        return candidate
    sibling = workspace.parent / "DeepStream-Yolo"
    if sibling.exists():
        return sibling
    return candidate


WORKSPACE = _resolve_workspace()
DEEPSTREAM_ROOT = _resolve_deepstream(WORKSPACE)
RUNS_DIR = _resolve_path("SQUEAKVIEW_RUNS_DIR", "PRODUCT_RUNS_DIR", WORKSPACE / "runs")
TASKS_DIR = WORKSPACE / "tasks"


def ensure_runs_dir() -> Path:
    """Create the runs directory if it doesn't exist."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR


def workspace_path(*parts: str) -> Path:
    """Join relative parts onto the workspace root."""
    return WORKSPACE.joinpath(*parts)
