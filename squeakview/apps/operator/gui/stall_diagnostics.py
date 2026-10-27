"""Best-effort GUI stall evidence; never owns or waits for acquisition resources."""

from __future__ import annotations

from squeakview.common.run_layout import artifact_path

import json
import os
import platform
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


class GuiStallDiagnostics:
    """Sample Python stacks at 3, 6 and 9 seconds of missing GUI progress.

    Only the daemon writes files. GUI/IPC callbacks update small in-memory
    values without locks or I/O. Output is limited to 2 MiB per run, and stack
    collection is limited to three snapshots per stall. No locals are captured.
    A Python thread cannot diagnose native code that holds the GIL indefinitely.
    """

    MAX_BYTES = 2 * 1024 * 1024

    def __init__(self, *, monotonic=time.monotonic, start=True) -> None:
        self._clock = monotonic
        self._run_dir: Path | None = None
        self._active = False
        self._generated = self._clock()
        self._started = self._generated
        self._sent: float | None = None
        self._tick_gap = 0.0
        self._failure: str | None = None
        self._stop = threading.Event()
        self._samples = 0
        self._episode = False
        self._written: dict[Path, int] = {}
        self._thread = threading.Thread(
            target=self._watch, name="squeakview-gui-stall-diagnostics", daemon=True
        )
        if start:
            self._thread.start()

    def set_run(self, run_dir: Path | None, active: bool) -> None:
        self._run_dir = run_dir
        self._active = active

    def generated(self) -> None:
        now = self._clock()
        self._tick_gap = now - self._generated
        self._generated = now

    def sent(self) -> None:
        self._sent = self._clock()

    def close(self, failure: str | None = None) -> None:
        self._failure = failure
        self._stop.set()  # Never join: diagnostic storage may itself be stalled.

    def _watch(self) -> None:
        while not self._stop.wait(0.25):
            self.poll()
        if self._failure:
            self._capture("connection_lost", stacks=True)

    def poll(self) -> None:
        if not self._active or self._run_dir is None:
            self._samples = 0
            self._episode = False
            return
        now = self._clock()
        gui_age = now - self._generated
        send_age = now - (self._sent if self._sent is not None else self._started)
        age = max(gui_age, send_age)
        if age >= 3.0:
            self._episode = True
            if self._samples < 3 and age >= 3.0 * (self._samples + 1):
                self._samples += 1
                self._capture("gui_stall" if gui_age >= 3.0 else "heartbeat_send_stall", stacks=True)
        else:
            # A delayed tick can recover between watchdog polls.
            if self._episode or self._tick_gap >= 3.0:
                self._capture("recovered", stacks=False)
            self._samples = 0
            self._episode = False
        self._tick_gap = 0.0

    def _capture(self, event: str, *, stacks: bool) -> None:
        run_dir = self._run_dir
        if run_dir is None:
            return
        path = artifact_path(run_dir, "diagnostics") / "gui_stalls.jsonl"
        try:
            used = self._written.get(path)
            if used is None:
                used = path.stat().st_size if path.exists() else 0
            if used >= self.MAX_BYTES:
                return
            now = self._clock()
            record = {
                "schema_version": 1,
                "event": event,
                "utc": datetime.now(timezone.utc).isoformat(),
                "monotonic_s": now,
                "hostname": platform.node(),
                "pid": os.getpid(),
                "run_id": run_dir.name,
                "python": platform.python_version(),
                "gui_tick_age_s": now - self._generated,
                "last_tick_gap_s": self._tick_gap,
                "last_send_age_s": None if self._sent is None else now - self._sent,
                "failure": self._failure,
            }
            if stacks:
                names = {thread.ident: thread.name for thread in threading.enumerate()}
                record["threads"] = [
                    {"ident": ident, "name": names.get(ident, "unknown"),
                     "stack": traceback.format_stack(frame, limit=32)}
                    for ident, frame in list(sys._current_frames().items())[:64]
                ]
            data = (json.dumps(record) + "\n").encode("utf-8")
            if used + len(data) > self.MAX_BYTES:
                self._written[path] = self.MAX_BYTES
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "ab") as handle:
                handle.write(data)
            self._written[path] = used + len(data)
        except Exception:
            # Diagnostic failure must never propagate into scientific capture.
            pass
