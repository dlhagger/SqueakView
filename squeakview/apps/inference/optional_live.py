"""Disposable inference/preview process; acquisition never waits for its frames."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time

from squeakview.common import run_context
from squeakview.common.run_layout import artifact_path
from .live_progress import LiveProgressWatchdog, mailbox_sequence


class OptionalLiveSession:
    def __init__(self, config, emit=print):
        self.config = config
        self.emit = emit
        self.mailboxes: tuple[Path, ...] = ()
        self.recording_sources = ()
        self.process = None
        self._temporary = None
        self._thread = None
        self._stop = threading.Event()
        self._stop_requested = False
        self.state = "disabled"
        self._started = 0.0
        self.status_path = artifact_path(Path(config.run_dir), "diagnostics") / "live_status.json"

    def _emit(self, message):
        try:
            self.emit(message)
        except Exception:
            pass

    def _report(self, state, error=None):
        changed = self.state != state
        self.state = state
        payload = {"state": state, "error": error, "isolated_process": True,
                   "inference_requested": self.config.enable_infer,
                   "preview_requested": bool(self.config.preview_sockets),
                   "recoverable_offline": True}
        if changed:
            self._emit(
                f"[LIVE] inference/preview worker {state}: "
                f"inference_requested={'yes' if self.config.enable_infer else 'no'} "
                f"preview_requested={'yes' if self.config.preview_sockets else 'no'}"
            )
        try:
            run_context.update_status(Path(self.config.run_dir), live_services=payload)
        except Exception as exc:
            self._emit(f"[LIVE] WARN: cannot persist live service status: {exc}")
        if error:
            self._emit(f"[LIVE] WARN: {error}; raw video and controller acquisition continue")

    def prepare(self):
        if not (self.config.enable_infer or self.config.preview_sockets):
            return self.mailboxes
        try:
            pixels = self.config.width * self.config.height
            if not 0 < pixels <= 64 * 1024 * 1024:
                raise ValueError("live frame exceeds mailbox capacity")
            self._temporary = tempfile.TemporaryDirectory(prefix="squeakview-live-", dir="/dev/shm")
            root = Path(self._temporary.name)
            paths = []
            for index in range(self.config.num_cameras):
                path = root / f"camera{index}"
                with path.open("xb") as handle:
                    os.chmod(path, 0o600)
                    # Allocate before triggers start. Never mmap sparse tmpfs
                    # pages that could SIGBUS in the acquisition process.
                    os.posix_fallocate(handle.fileno(), 0, 4096 + 65536 + pixels)
                paths.append(path)
            self.mailboxes = tuple(paths)
            values = asdict(self.config)
            values.update(raw_only=False, live_mailboxes=[str(path) for path in paths], failure_plan=None)
            (root / "config.json").write_text(json.dumps(values, default=str))
            self._report("starting")
        except Exception as exc:
            self.mailboxes = ()
            self._report("degraded", f"live processing could not start: {exc}")
        return self.mailboxes

    def start(self):
        if not self.mailboxes:
            return
        try:
            self.status_path.parent.mkdir(parents=True, exist_ok=True)
            self.status_path.unlink(missing_ok=True)
            # ProcessHandle bounds diagnostic output and drains it independently
            # of the operator GUI. Its output is NOT capture protocol input.
            from squeakview.apps.operator.backend.supervision import spawn
            self.process = spawn(
                "squeakview.apps.inference.live_worker",
                ["--config", str(Path(self._temporary.name) / "config.json")],
                lambda _line: None, "optional live", workspace=Path(__file__).resolve().parents[3],
                output_log_path=self.status_path.with_name("live_processing.log"),
                parent_death_signal=signal.SIGKILL, keep_stdin_open=True,
            )
            self._started = time.monotonic()
            self._thread = threading.Thread(target=self._watch, daemon=True, name="optional-live-watch")
            self._thread.start()
        except Exception as exc:
            self._report("degraded", f"live processing launch failed: {exc}")
            if self.process and self.process.is_running():
                self.process.send_signal_group(signal.SIGKILL)

    def _watch(self):
        progress = LiveProgressWatchdog(self.config.num_cameras)
        last_status = {}
        while not self._stop.wait(0.5):
            try:
                status = run_context.read_live_snapshot(self.status_path)
                # One unavailable read is not a dead worker.
                # Keep its last heartbeat (and dump-handler handshake) until
                # that heartbeat actually expires; never refresh it locally.
                if status:
                    last_status = status
                else:
                    status = last_status
                if not self.process.is_running():
                    code = self.process.wait(timeout=0)
                    # A source EOS just before normal capture shutdown is okay.
                    # Native teardown can still abort after writing that status.
                    if code == 0 and status.get("state") == "completed":
                        self._report("completed")
                    else:
                        self._report("degraded", f"live process exited ({code}): {status.get('error') or 'see live_processing.log'}")
                    return
                now = time.monotonic()
                heartbeat = status.get("heartbeat_monotonic")
                if status.get("state") == "running" and self.state == "starting":
                    self._report("running")
                if status.get("state") == "running":
                    inputs = [mailbox_sequence(path) for path in self.mailboxes]
                    # Prefer capture counters: a stuck mailbox reader may hold
                    # its shared lock and prevent publication of newer frames.
                    for index, source in enumerate(self.recording_sources):
                        activity = source.try_activity()
                        if activity is not None:
                            inputs[index] = activity.source_count
                    error = progress.check(inputs, status.get("frames_processed"), now)
                    if error:
                        self._kill_stalled(error, status)
                        return
                if ((heartbeat is None and now - self._started > 60)
                        or (heartbeat is not None and now - float(heartbeat) > 10)):
                    self._kill_stalled("live processing stopped responding", status)
                    return
            except Exception as exc:
                self._report("degraded", f"live processing monitor failed: {exc}")
                if self.process and self.process.is_running():
                    self.process.send_signal_group(signal.SIGKILL)
                return

    def _kill_stalled(self, reason, status):
        self._report('degraded', reason)
        # Only signal workers that confirmed registration. SIGUSR1's default
        # action would otherwise terminate a worker still importing Python.
        try:
            if status.get('stack_dump_ready') is True:
                self.process.send_signal_group(signal.SIGUSR1)
            from .live_diagnostics import save_hang_evidence
            completed = threading.Event()

            def collect():
                try:
                    pid = getattr(getattr(self.process, 'p', None), 'pid', None)
                    save_hang_evidence(self.status_path.parent, pid, reason, status)
                except Exception as exc:
                    self._emit(f'[LIVE] WARN: hang evidence could not be saved: {exc}')
                finally:
                    completed.set()

            threading.Thread(target=collect, daemon=True, name='live-hang-evidence').start()
            completed.wait(0.25)
            if status.get('stack_dump_ready') is True:
                try:
                    self.process.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    pass
            self._emit('[LIVE] Hang evidence: diagnostics/live_hang.json and live_stacks.log')
        except Exception as exc:
            self._emit(f'[LIVE] WARN: hang diagnostics failed: {exc}')
        finally:
            if self.process.is_running():
                self.process.send_signal_group(signal.SIGKILL)

    def request_stop(self):
        # Capture stop and final cleanup both call this. A second SIGINT can
        # interrupt native teardown after the worker has reported completion.
        if self._stop_requested:
            return
        self._stop_requested = True
        self._stop.set()
        try:
            if self.process and self.process.is_running():
                self.process.send_signal_group(signal.SIGINT)
        except Exception as exc:
            self._report("degraded", f"live stop request failed: {exc}")

    def close(self):
        # Optional teardown must not mask the capture result, even when its
        # status/log filesystem or child process handle has failed.
        try:
            self._close()
        except Exception as exc:
            self._report("degraded", f"live cleanup failed: {exc}")

    def _close(self):
        self.request_stop()
        if self._thread:
            self._thread.join(timeout=1)
        if self.process:
            for sig, timeout in ((None, 1), (signal.SIGTERM, 1), (signal.SIGKILL, 0.5)):
                if sig and self.process.is_running():
                    self.process.send_signal_group(sig)
                try:
                    code = self.process.wait(timeout=timeout)
                    if code and self.state != "degraded":
                        self._report("degraded", f"live processing needed forced shutdown ({code})")
                    elif self.state not in {"degraded", "disabled"}:
                        self._report("completed")
                    break
                except subprocess.TimeoutExpired:
                    continue
            else:
                self._report("degraded", "live process exit remains unconfirmed")
                return  # Do not remove mapped storage while the consumer lives.
        if self._temporary:
            self._temporary.cleanup()
