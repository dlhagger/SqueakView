"""Shared shutdown budgets and evidence for post-capture exit failures."""
from __future__ import annotations

import math
import os
from pathlib import Path

from . import run_context
from .run_layout import artifact_path


CLOSURE_FILENAME = "capture_closure.json"


def pipeline_shutdown_timeout_s(*, failure_injection: bool = False) -> float:
    default = 5.0 if failure_injection else 20.0
    try:
        value = float(os.environ.get("SQUEAKVIEW_PIPELINE_SHUTDOWN_TIMEOUT_S", "") or default)
    except ValueError:
        return default
    return max(0.1, value) if math.isfinite(value) and value <= 120.0 else default


def capture_exit_grace_s(*, failure_injection: bool = False) -> float:
    pipeline = pipeline_shutdown_timeout_s(failure_injection=failure_injection)
    # Pipeline drain, critical writers, closure receipt, optional telemetry,
    # status persistence and optional worker escalation, plus scheduling margin.
    return pipeline_shutdown_max_s(failure_injection=failure_injection) + 6 * min(5.0, pipeline) + 16.0


def pipeline_shutdown_max_s(*, failure_injection: bool = False) -> float:
    stall = pipeline_shutdown_timeout_s(failure_injection=failure_injection)
    default = max(stall, 15.0 if failure_injection else 120.0)
    try:
        value = float(os.environ.get("SQUEAKVIEW_PIPELINE_SHUTDOWN_MAX_S", "") or default)
    except ValueError:
        return default
    return value if math.isfinite(value) and stall <= value <= 600.0 else default


def write_capture_closure(run_dir: Path) -> None:
    """Publish only after an explicitly requested, error-free capture close.

    This is evidence of completed capture teardown, not a validation result.
    The supervisor must still confirm process exit and validate all artifacts.
    """
    run_context.atomic_write_json(artifact_path(run_dir, CLOSURE_FILENAME), {
        "schema_version": 2,
        "run_dir": str(Path(run_dir).resolve()),
        "pipeline_drained": True,
        "eos_observed": True,
        "pipeline_null_confirmed": True,
        "writers_closed": True,
        "capture_exit_code": 0,
        "explicit_stop": True,
    })


def has_clean_capture_closure(run_dir: Path) -> bool:
    try:
        value = run_context.read_json_required(
            artifact_path(run_dir, CLOSURE_FILENAME), max_bytes=4096
        )
    except (OSError, ValueError):
        return False
    return (
        type(value.get("schema_version")) is int and value["schema_version"] == 2
        and value.get("run_dir") == str(Path(run_dir).resolve())
        and value.get("pipeline_drained") is True
        and value.get("eos_observed") is True
        and value.get("pipeline_null_confirmed") is True
        and value.get("writers_closed") is True
        and type(value.get("capture_exit_code")) is int and value["capture_exit_code"] == 0
        and value.get("explicit_stop") is True
    )
