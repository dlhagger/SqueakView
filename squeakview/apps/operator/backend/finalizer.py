"""Bounded supervision for the independent post-run validator."""

from __future__ import annotations

import math
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Callable

from squeakview.apps.operator.backend import process
from squeakview.common import run_context


def _bounded_timeout(name: str, default: float, maximum: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value) or not 0.1 <= value <= maximum:
        return default
    return value


def _progress_count(progress: object) -> int:
    if not isinstance(progress, dict):
        return 0
    value = progress.get("frames_processed")
    return value if type(value) is int and value >= 0 else 0


def run_capture_finalizer(
    run_dir: Path,
    *,
    camera_count: int,
    enable_infer: bool,
    enable_align: bool,
    emit: Callable[[str], None],
    force_timeout: bool = False,
) -> int:
    """Run the validator with bounded TERM/KILL escalation.

    The validator owns its own process group.  A timeout therefore terminates
    ffprobe and any other descendants as well as the Python coordinator.
    """

    emit("[POST-RUN] starting independent bounded-memory finalizer")
    worker = process.spawn_post_run(
        run_dir,
        camera_count=int(camera_count),
        enable_infer=bool(enable_infer),
        enable_align=bool(enable_align),
    )
    timeout_s = _bounded_timeout(
        "SQUEAKVIEW_POST_RUN_TIMEOUT_S", 21_600.0, 86_400.0
    )
    terminate_grace_s = _bounded_timeout(
        "SQUEAKVIEW_POST_RUN_TERMINATE_GRACE_S", 30.0, 300.0
    )
    kill_grace_s = _bounded_timeout(
        "SQUEAKVIEW_POST_RUN_KILL_GRACE_S", 10.0, 60.0
    )

    deadline = time.monotonic() + timeout_s
    last_reported = -1
    while force_timeout or worker.poll() is None:
        if force_timeout or time.monotonic() >= deadline:
            prefix = (
                "qualification-injected post-run finalizer timeout; "
                if force_timeout
                else f"post-run finalizer timed out after {timeout_s:.1f}s; "
            )
            error = prefix + "recording validation did not complete"
            emit(f"[POST-RUN] ERROR: {error}")
            try:
                run_context.write_status(
                    run_dir,
                    "finalization_failed",
                    error=error,
                    post_run_timeout_s=timeout_s,
                )
            except Exception as exc:
                emit(f"[POST-RUN] could not persist timeout state: {exc}")

            try:
                os.killpg(worker.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception as exc:
                emit(f"[POST-RUN] SIGTERM process-group error: {exc}")
            try:
                worker.wait(timeout=terminate_grace_s)
            except subprocess.TimeoutExpired:
                emit("[POST-RUN] finalizer still running; sending SIGKILL")
                try:
                    os.killpg(worker.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except Exception as exc:
                    emit(f"[POST-RUN] SIGKILL process-group error: {exc}")
                try:
                    worker.wait(timeout=kill_grace_s)
                except subprocess.TimeoutExpired:
                    emit(
                        "[POST-RUN] ERROR: finalizer process group did not exit "
                        "after SIGKILL"
                    )
            # A worker that exits zero only because SIGTERM requested a clean
            # shutdown still did not complete validation in time.
            return 124

        progress = run_context.read_json(run_dir / "post_run_progress.json")
        processed = _progress_count(progress)
        if processed >= last_reported + 100_000:
            emit(
                f"[POST-RUN] {progress.get('stage', 'starting')}: "
                f"{processed} frames"
            )
            last_reported = processed
        time.sleep(0.2)
    return int(worker.returncode or 0)
