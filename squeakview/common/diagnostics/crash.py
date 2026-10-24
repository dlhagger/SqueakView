"""Best-effort process crash evidence independent of Qt and presentation IPC.

Fatal native signals still terminate the process. A private, separately opened
descriptor lets faulthandler preserve stacks even when stdout has disappeared
or its Python mirror is locked. SIGKILL/OOM cannot execute a crash handler;
the supervising process must record that exit separately.
"""
from __future__ import annotations

import atexit
from datetime import datetime, timezone
import faulthandler
import os
from pathlib import Path
import platform
import sys
import threading
import traceback


MAX_EXCEPTION_BYTES = 2 * 1024 * 1024
_installed: "CrashDiagnostics | None" = None


class CrashDiagnostics:
    def __init__(self, role: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        self.file = os.fdopen(fd, "a", encoding="utf-8", buffering=1)
        self.path = path
        self._lock = threading.RLock()
        self._previous_sys = sys.excepthook
        self._previous_thread = threading.excepthook
        self._previous_unraisable = sys.unraisablehook
        self.file.write(
            f"SqueakView crash diagnostics: role={role} pid={os.getpid()} "
            f"host={platform.node()} python={platform.python_version()} "
            f"utc={datetime.now(timezone.utc).isoformat()}\n"
        )
        try:
            faulthandler.enable(file=self.file, all_threads=True)
        except BaseException:
            self.file.close()
            raise
        sys.excepthook = self._sys_exception
        threading.excepthook = self._thread_exception
        sys.unraisablehook = self._unraisable

    def _record(self, label, exc_type, exc_value, exc_traceback) -> None:
        try:
            # No locals or environment variables. Bound exception output; fatal
            # signal stacks use the descriptor directly and bypass this lock.
            with self._lock:
                if self.file.closed or os.fstat(self.file.fileno()).st_size >= MAX_EXCEPTION_BYTES:
                    return
                rendered = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback, limit=64))
                self.file.write(
                    f"\n{datetime.now(timezone.utc).isoformat()} {label}\n"
                    + rendered[:65536] + "\n"
                )
                self.file.flush()
        except Exception:
            pass

    def _sys_exception(self, kind, value, tb) -> None:
        self._record("UNHANDLED MAIN/Qt CALLBACK EXCEPTION", kind, value, tb)
        self._previous_sys(kind, value, tb)

    def _thread_exception(self, args) -> None:
        name = args.thread.name if args.thread is not None else "unknown"
        self._record(f"UNHANDLED THREAD EXCEPTION ({name})", args.exc_type, args.exc_value, args.exc_traceback)
        self._previous_thread(args)

    def _unraisable(self, args) -> None:
        self._record("UNRAISABLE EXCEPTION", args.exc_type, args.exc_value, args.exc_traceback)
        self._previous_unraisable(args)

    def close(self) -> None:
        if self.file.closed:
            return
        # Restore only hooks still owned by this installation.
        if sys.excepthook == self._sys_exception:
            sys.excepthook = self._previous_sys
        if threading.excepthook == self._thread_exception:
            threading.excepthook = self._previous_thread
        if sys.unraisablehook == self._unraisable:
            sys.unraisablehook = self._previous_unraisable
        faulthandler.disable()
        self.file.close()


def install_crash_diagnostics(role: str, log_path: Path | None = None) -> Path | None:
    """Install once per process; diagnostic setup failure never aborts capture."""
    global _installed
    if _installed is not None:
        return _installed.path
    try:
        if log_path is None:
            from squeakview.project import AppPaths, ProjectPaths, PROJECT_ENV, UserPaths
            from squeakview.project import validate_external_output_path

            app = AppPaths.discover()
            user = UserPaths.discover()
            user.validate_for_app(app)
            project_root = os.environ.get(PROJECT_ENV, "").strip()
            project = ProjectPaths.from_existing_root(Path(project_root)) if project_root else None
            if project is not None:
                user.validate_for_project(project)
            safe_role = "".join(c for c in role if c.isalnum() or c in "-_")[:32] or "process"
            log_path = validate_external_output_path(
                user.launch_logs / f"squeakview_{safe_role}_{os.getpid()}.crash.log",
                app=app, project=project, label="process crash log",
            )
        path = Path(log_path)
        if not path.is_absolute():
            return None  # Never create diagnostic output relative to a checkout.
        _installed = CrashDiagnostics(role, path)
        atexit.register(_installed.close)
        return _installed.path
    except Exception:
        return None
