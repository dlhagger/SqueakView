#!/usr/bin/env python3
"""Helper to launch the SqueakView operator GUI from the repo root with one command."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from squeakview.common.log_mirror import LineBufferedLogMirror
from squeakview.project import (
    PROJECT_ENV,
    AppPaths,
    ProjectPaths,
    UserPaths,
    validate_external_output_path,
)


ROOT = Path(__file__).resolve().parent
LOG_ENV = "SQUEAKVIEW_LOGFILE"


def _setup_logging() -> None:
    log_path = os.environ.get(LOG_ENV)
    if not log_path:
        return
    path = Path(log_path)
    sys.stdout = LineBufferedLogMirror(path, sys.stdout)
    sys.stderr = LineBufferedLogMirror(path, sys.stderr)
    print(f"[squeakview] Logging to {path}", flush=True)


def _configure_logging() -> None:
    app_paths = AppPaths.discover()
    user_paths = UserPaths.discover()
    user_paths.validate_for_app(app_paths)
    raw_project = os.environ.get(PROJECT_ENV, "").strip()
    project_paths = (
        ProjectPaths.from_existing_root(Path(raw_project)) if raw_project else None
    )
    if project_paths is not None:
        app_paths.validate_for_project(project_paths)
        user_paths.validate_for_project(project_paths)
    if not os.environ.get(LOG_ENV, "").strip():
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        user_paths.ensure()
        os.environ[LOG_ENV] = str(
            user_paths.launch_logs / f"squeakview_gui_{ts}.log"
        )
    log_path = validate_external_output_path(
        Path(os.environ[LOG_ENV]),
        app=app_paths,
        project=project_paths,
        label="GUI launch log",
    )
    os.environ[LOG_ENV] = str(log_path)
    _setup_logging()


def main() -> int:
    try:
        _configure_logging()
    except Exception as exc:
        print(
            f"[FATAL] SqueakView GUI logging configuration is invalid: {exc}",
            file=sys.stderr,
        )
        return 1
    from squeakview.apps.operator import main as operator_main

    return operator_main.main()


if __name__ == "__main__":
    raise SystemExit(main())
