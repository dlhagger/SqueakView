#!/usr/bin/env python3
"""Helper to launch the SqueakView operator GUI from the repo root with one command."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from squeakview.common.log_mirror import LineBufferedLogMirror


ROOT = Path(__file__).resolve().parent
LOG_ENV = "SQUEAKVIEW_LOGFILE"
RUN_ENV = "SQUEAKVIEW_RUN_DIR"


def _setup_logging() -> None:
    log_path = os.environ.get(LOG_ENV)
    if not log_path:
        return
    path = Path(log_path)
    sys.stdout = LineBufferedLogMirror(path, sys.stdout)
    sys.stderr = LineBufferedLogMirror(path, sys.stderr)
    print(f"[squeakview] Logging to {path}", flush=True)


# Ensure the repo root is importable and set up logging in the current process.
sys.path.insert(0, str(ROOT))
if LOG_ENV not in os.environ:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir_env = os.environ.get(RUN_ENV)
    if run_dir_env:
        os.environ[LOG_ENV] = str(Path(run_dir_env) / "squeakview_gui.log")
    else:
        os.environ[LOG_ENV] = str(ROOT / "runs" / "logs" / f"squeakview_gui_{ts}.log")
_setup_logging()

from squeakview.apps.operator import main as operator_main  # noqa: E402


if __name__ == "__main__":
    operator_main.main()
