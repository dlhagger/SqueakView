from __future__ import annotations

"""Centralized paths for the trimmed SqueakView direct-FLIR stack."""

import os
from pathlib import Path

_WORKSPACE_PATH_ANCHORS = (
    "models",
    "tasks",
    "profiles",
    "native",
    "scripts",
    "build_engine",
    "data_viz",
    "build_me",
    "runs",
)


def _resolve_workspace() -> Path:
    env_override = os.environ.get("SQUEAKVIEW_WORKSPACE") or os.environ.get("PRODUCT_WORKSPACE")
    if env_override:
        return Path(env_override).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _resolve_path(env_var: str, default: Path) -> Path:
    candidate = os.environ.get(env_var)
    if candidate:
        return Path(candidate).expanduser().resolve()
    return default


def _resolve_deepstream_sdk() -> Path:
    candidate = os.environ.get("SQUEAKVIEW_DEEPSTREAM_SDK")
    if candidate:
        return Path(candidate).expanduser().resolve()
    return Path("/opt/nvidia/deepstream/deepstream").resolve()


WORKSPACE = _resolve_workspace()
DEEPSTREAM_SDK_ROOT = _resolve_deepstream_sdk()

MODEL_ROOT = _resolve_path("SQUEAKVIEW_MODEL_ROOT", WORKSPACE / "models")
DEFAULT_MODEL_NAME = os.environ.get("SQUEAKVIEW_MODEL_NAME", "").strip()
DEFAULT_MODEL_ROOT = MODEL_ROOT / DEFAULT_MODEL_NAME if DEFAULT_MODEL_NAME else MODEL_ROOT


def _resolve_default_infer_config() -> Path | None:
    explicit = os.environ.get("SQUEAKVIEW_DS_CFG") or os.environ.get("DS_CFG")
    if explicit:
        path = Path(explicit).expanduser()
        return (WORKSPACE / path).resolve() if not path.is_absolute() else path.resolve()
    if DEFAULT_MODEL_NAME:
        return (DEFAULT_MODEL_ROOT / "configs" / f"{DEFAULT_MODEL_NAME}.txt").resolve()
    return None


DEFAULT_INFER_CONFIG = _resolve_default_infer_config()

# Compatibility for older GUI/backend code that used DEEPSTREAM_ROOT for model configs.
DEEPSTREAM_ROOT = DEFAULT_MODEL_ROOT

NATIVE_ROOT = WORKSPACE / "native"
FLIR_GST_SOURCE_ROOT = NATIVE_ROOT / "flir_gst_source"
FLIR_GST_PLUGIN_DIR = FLIR_GST_SOURCE_ROOT / "build"
CUSTOM_YOLO_LIB = NATIVE_ROOT / "nvdsinfer_custom_impl_yolo" / "libnvdsinfer_custom_impl_Yolo.so"

RUNS_DIR = _resolve_path("SQUEAKVIEW_RUNS_DIR", WORKSPACE / "runs")
TASKS_DIR = WORKSPACE / "tasks"
PROFILES_DIR = WORKSPACE / "profiles"


def ensure_runs_dir() -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR


def ensure_profiles_dir() -> Path:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    return PROFILES_DIR


def workspace_path(*parts: str) -> Path:
    return WORKSPACE.joinpath(*parts)


def resolve_workspace_path(value: str | os.PathLike[str] | None, *, base: Path | None = None) -> Path | None:
    """Resolve app-owned paths independent of launch cwd or clone location.

    Relative paths are interpreted from the repo root. Absolute paths that
    contain a known repo-owned anchor such as ``models`` or ``tasks`` are
    remapped to the current clone when the same anchored path exists here; that
    lets old profile JSON survive a repo move even if the old clone still exists.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        return ((base or WORKSPACE) / path).resolve()
    parts = path.parts
    for anchor in _WORKSPACE_PATH_ANCHORS:
        if anchor not in parts:
            continue
        idx = parts.index(anchor)
        candidate = WORKSPACE.joinpath(*parts[idx:])
        if candidate.exists():
            return candidate.resolve()
        if path.exists():
            return path.resolve()
        return candidate.resolve()
    if path.exists():
        return path.resolve()
    return path


def portable_workspace_path(value: str | os.PathLike[str] | None) -> str | None:
    """Return a repo-relative string for app-owned paths when possible."""
    path = resolve_workspace_path(value)
    if path is None:
        return None
    try:
        return path.relative_to(WORKSPACE).as_posix()
    except ValueError:
        return str(path)
