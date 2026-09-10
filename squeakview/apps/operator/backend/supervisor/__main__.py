from __future__ import annotations

"""Launch the durable backend supervisor and its mandatory GUI process."""

import argparse
import os
import signal
import stat
import sys
import tempfile
import threading
from pathlib import Path
from typing import Sequence

from .server import SupervisorServer
from squeakview.common.log_mirror import LineBufferedLogMirror


SUPERVISOR_LOG_ENV = "SQUEAKVIEW_SUPERVISOR_LOGFILE"
LAUNCH_STATUS_ENV = "SQUEAKVIEW_LAUNCH_STATUS_FILE"


def _install_log_mirror() -> LineBufferedLogMirror | None:
    raw_path = os.environ.get(SUPERVISOR_LOG_ENV, "").strip()
    if not raw_path:
        return None
    mirror = LineBufferedLogMirror(Path(raw_path), sys.stdout)
    sys.stdout = mirror
    sys.stderr = mirror
    return mirror


def _install_signal_handlers(requested: threading.Event) -> None:
    """Install async-safe handlers that only publish a shutdown request."""

    def stop(_signum: int, _frame: object) -> None:
        requested.set()

    for signum in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if signum is not None:
            signal.signal(signum, stop)


def _mark_gui_ready() -> None:
    """Acknowledge a Qt-main-loop heartbeat to the detached launcher."""

    raw_path = os.environ.get(LAUNCH_STATUS_ENV, "").strip()
    if not raw_path:
        return
    path = Path(raw_path)
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.getuid()
        or before.st_mode & 0o077
    ):
        raise RuntimeError("launch status file must be a private regular file owned by this user")
    flags = os.O_WRONLY | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        current = os.fstat(fd)
        if (current.st_dev, current.st_ino) != (before.st_dev, before.st_ino):
            raise RuntimeError("launch status file identity changed")
        os.write(fd, b"GUI_READY\n")
        os.fsync(fd)
    finally:
        os.close(fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--socket",
        type=Path,
        help="private Unix socket path (default: a new mode-0700 runtime directory)",
    )
    parser.add_argument(
        "--gui-command",
        nargs=argparse.REMAINDER,
        required=True,
        help="explicit GUI argv; executed directly without a shell",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        mirror = _install_log_mirror()
    except Exception as exc:
        print(f"SqueakView supervisor log setup failed: {exc}", file=sys.stderr)
        return 1
    runtime_dir: Path | None = None
    if args.socket is None:
        base = os.environ.get("XDG_RUNTIME_DIR")
        runtime_dir = Path(
            tempfile.mkdtemp(
                prefix="squeakview-supervisor-",
                dir=base if base and Path(base).is_dir() else None,
            )
        )
        os.chmod(runtime_dir, 0o700)
        socket_path = runtime_dir / "operator.sock"
    else:
        socket_path = args.socket

    try:
        server = SupervisorServer(socket_path)
    except Exception as exc:
        print(f"SqueakView supervisor initialization failed: {exc}", file=sys.stderr)
        if mirror is not None:
            mirror.close()
        if runtime_dir is not None:
            try:
                runtime_dir.rmdir()
            except OSError:
                pass
        return 1

    shutdown_requested = threading.Event()
    watcher_finished = threading.Event()

    def watch_shutdown() -> None:
        shutdown_requested.wait()
        if not watcher_finished.is_set():
            server.request_shutdown()

    watcher = threading.Thread(
        target=watch_shutdown,
        name="squeakview-supervisor-signals",
        daemon=False,
    )
    watcher.start()
    _install_signal_handlers(shutdown_requested)
    try:
        try:
            result = server.serve(
                args.gui_command,
                on_gui_ready=_mark_gui_ready,
            )
        except Exception as exc:
            print(f"SqueakView supervisor failed: {exc}", file=sys.stderr)
            result = 1
        if server.last_error:
            print(f"SqueakView supervisor failed: {server.last_error}", file=sys.stderr)
        return result
    finally:
        watcher_finished.set()
        shutdown_requested.set()
        watcher.join()
        if mirror is not None:
            mirror.close()
        if runtime_dir is not None:
            try:
                runtime_dir.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
