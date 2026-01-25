#!/usr/bin/env python3
"""Helper to launch the SqueakView operator GUI from the repo root with one command."""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONTROL_PY = ROOT / "environments" / "control" / ".venv" / "bin" / "python"
LOG_ENV = "SQUEAKVIEW_LOGFILE"
RUN_ENV = "SQUEAKVIEW_RUN_DIR"


class _Tee:
    """Mirror stdout/stderr to a log file, filtering serial chatter."""

    _SER_PAT = re.compile(r"\bCAMERA_(LOW|HIGH)\b")

    def __init__(self, path: Path, stream):
        self._stream = stream
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", buffering=1)

    def write(self, data: str) -> None:
        if self._SER_PAT.search(data):
            self._stream.write(data)
            return
        self._stream.write(data)
        try:
            self._fh.write(data)
        except Exception:
            pass

    def flush(self) -> None:
        try:
            self._fh.flush()
        except Exception:
            pass
        try:
            self._stream.flush()
        except Exception:
            pass


def _setup_logging() -> None:
    log_path = os.environ.get(LOG_ENV)
    if not log_path:
        return
    path = Path(log_path)
    sys.stdout = _Tee(path, sys.stdout)
    sys.stderr = _Tee(path, sys.stderr)
    print(f"[squeakview] Logging to {path}", flush=True)


# If the control venv exists, always use it (unless we're already in it).
if CONTROL_PY.exists() and os.environ.get("SQUEAKVIEW_CONTROL_ENV") != "1":
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["SQUEAKVIEW_CONTROL_ENV"] = "1"
    # If a run dir is provided (by the backend), use it for the log file; else a temp fallback.
    if LOG_ENV not in env:
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        run_dir_env = env.get(RUN_ENV)
        if run_dir_env:
            env[LOG_ENV] = str(Path(run_dir_env) / "squeakview_gui.log")
        else:
            env[LOG_ENV] = str(ROOT / "runs" / "logs" / f"squeakview_gui_{ts}.log")
    os.execve(str(CONTROL_PY), [str(CONTROL_PY), __file__, *sys.argv[1:]], env)

# Ensure the repo root is importable and set up logging in the current process.
sys.path.insert(0, str(ROOT))
_setup_logging()

from squeakview.apps.operator import main as operator_main  # noqa: E402


if __name__ == "__main__":
    operator_main.main()
