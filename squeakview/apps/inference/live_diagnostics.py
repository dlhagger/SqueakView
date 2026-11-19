"""Hang evidence collected independently of the optional worker's Python loop."""
from __future__ import annotations

import atexit
from datetime import datetime, timezone
import faulthandler
import os
from pathlib import Path
import signal

from squeakview.common import run_context
from squeakview.common.diagnostics.crash import install_crash_diagnostics


class LiveWorkerDiagnostics:
    def __init__(self, directory: Path):
        self.file = None
        self.error = None
        install_crash_diagnostics('live', directory / 'live_crash.log')
        try:
            directory.mkdir(parents=True, exist_ok=True)
            fd = os.open(directory / 'live_stacks.log',
                         os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
            self.file = os.fdopen(fd, 'w')
            # faulthandler's native signal handler runs without acquiring the
            # GIL or waiting for a Python signal callback / stdout lock.
            faulthandler.register(signal.SIGUSR1, file=self.file, all_threads=True)
            atexit.register(self.close)
        except Exception as exc:
            self.error = f'{type(exc).__name__}: {exc}'
            if self.file is not None:
                self.file.close()
                self.file = None

    @property
    def ready(self):
        return self.file is not None

    def close(self):
        if self.file is not None:
            faulthandler.unregister(signal.SIGUSR1)
            self.file.close()
            self.file = None


def save_hang_evidence(directory: Path, pid: int | None, reason: str, status: dict):
    """Called on a disposable parent thread; procfs/filesystem reads may block."""
    evidence = {
        'schema_version': 1,
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'reason': reason,
        'pid': pid,
        'last_worker_status': status,
        'python_stacks': 'live_stacks.log',
        'threads': {},
    }
    try:
        if type(pid) is int and pid > 0:
            tasks = sorted((Path('/proc') / str(pid) / 'task').iterdir())[:128]
            for task in tasks:
                thread = {}
                for name in ('comm', 'wchan', 'syscall', 'stack'):
                    try:
                        with (task / name).open() as handle:
                            thread[name] = handle.read(8192)
                    except OSError as exc:
                        thread[name] = f'unavailable: {exc}'
                evidence['threads'][task.name] = thread
    except OSError as exc:
        evidence['proc_error'] = str(exc)
    run_context.atomic_write_json(directory / 'live_hang.json', evidence)
