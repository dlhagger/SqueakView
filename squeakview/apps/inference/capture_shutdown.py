"""Bounded capture closure, independently testable without the Jetson SDK."""
from __future__ import annotations

import threading
import time
from typing import Callable, TYPE_CHECKING

from squeakview.common import run_context
from squeakview.common.child_events import encode_child_event
from squeakview.common.shutdown_policy import (
    pipeline_shutdown_timeout_s, pipeline_shutdown_max_s, write_capture_closure,
)
from squeakview.common.shutdown_progress import ProgressDeadline, wait_for_completion

if TYPE_CHECKING:
    from .service_maker_runner import ServiceMakerApp


def ts() -> str:
    return time.strftime("%H:%M:%S")


class RecordingShutdownProgress:
    """Sample counters and video sizes off the deadline-owning thread.

    A blocked stat or unavailable telemetry lock yields no progress. Shutdown
    never waits for this diagnostic sampler to finish.
    """
    def __init__(self, app):
        self.telemetry = tuple(app.record_telemetry)
        # Raw videos are top-level artifacts in both supported run layouts.
        # Do not touch the filesystem on the deadline-owning thread here.
        self.paths = tuple(app.run_dir / ('raw.mp4' if i == 0 else f'raw_cam{i}.mp4')
                           for i in range(getattr(app.config, 'num_cameras', 1)))
        self._latest: dict[str, int] = {}
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True, name='shutdown-progress').start()

    def _run(self):
        while not self._stop.is_set():
            snapshot = {}
            for i, telemetry in enumerate(self.telemetry):
                try:
                    counts = telemetry.shutdown_progress()
                except Exception:
                    continue  # Unavailable diagnostics are not proof of progress.
                if isinstance(counts, tuple) and len(counts) == 3:
                    snapshot.update({f'camera_{i}_{name}': count for name, count in
                                     zip(('source', 'admitted', 'encoded'), counts)})
            # Publish frame counters even if filesystem sampling subsequently
            # stalls. Never count diagnostic log growth as recording progress.
            self._latest = snapshot
            for i, path in enumerate(self.paths):
                try:
                    snapshot = {**snapshot, f'camera_{i}_video_bytes': path.stat().st_size}
                    self._latest = snapshot
                except OSError:
                    pass
            self._stop.wait(0.05)

    def snapshot(self):
        return self._latest

    def close(self):
        self._stop.set()


def stop_capture(self: "ServiceMakerApp", *, emit: Callable[[str], None]) -> None:
    if self._stopped:
        return
    progress = RecordingShutdownProgress(self)
    try:
        _stop_capture(self, emit=emit, progress=progress)
    finally:
        progress.close()


def _stop_capture(self, *, emit, progress) -> None:
    if self._stopped:
        return
    self._stopped = True
    # Wake every capture-owned monitor even when pipeline.start() or a
    # lifecycle callback raised before the normal stop request path.
    self._stop_event.set()
    close_errors: list[str] = []
    close_warnings: list[str] = []
    if self.live_session is not None:
        self.live_session.request_stop()

    shutdown_timeout_s = pipeline_shutdown_timeout_s(
        failure_injection=self.config.failure_plan is not None
    )
    budget = ProgressDeadline(shutdown_timeout_s, pipeline_shutdown_max_s(
        failure_injection=self.config.failure_plan is not None))
    progress_snapshot = progress.snapshot
    stage_reports: dict[str, object] = {}

    def timeout_detail(name):
        report = budget.report()
        return (f"{name}: timed out ({report['limit']} limit); "
                f"idle {report['idle_s']:.1f}s, elapsed {report['elapsed_s']:.1f}s; "
                f"progress={report['counters']}")

    def close_pipeline_with_timeout(
        name: str, close_fn: Callable[[], object], *, required: bool = True
    ) -> bool:
        emit(f"[{ts()}] [CLOSE] beginning {name}")
        completed = threading.Event()
        failure: list[BaseException] = []

        def invoke() -> None:
            try:
                close_fn()
            except BaseException as exc:  # preserve native binding failures
                failure.append(exc)
            finally:
                completed.set()

        threading.Thread(
            target=invoke,
            daemon=True,
            name=f"squeakview-{name.replace(' ', '-')}",
        ).start()
        finished = wait_for_completion(completed.wait, budget, progress=progress_snapshot)
        stage_reports[name] = {**budget.report(), "completed": finished and not failure}
        if not finished:
            if required and self.exit_code == 0:
                self.exit_code = 1
            self._force_process_exit = True
            detail = (
                timeout_detail(name) + "; "
                + ("capture closure unconfirmed" if required else "optional cleanup incomplete")
            )
            (close_errors if required else close_warnings).append(detail)
            emit(f"[{ts()}] [RECORD] {'ERROR' if required else 'WARN'} closing {detail}")
            return False
        if failure:
            if required and self.exit_code == 0:
                self.exit_code = 1
            exc = failure[0]
            detail = f"{name}: {type(exc).__name__}: {exc}"
            (close_errors if required else close_warnings).append(detail)
            emit(f"[{ts()}] [RECORD] {'ERROR' if required else 'WARN'} closing {detail}")
            return False
        emit(f"[{ts()}] [CLOSE] completed {name}")
        budget.completed_step()
        return True

    # CapturePipeline.stop now sets source EOS atomics, including after a
    # recording fault. Try to save the MP4 tail before bounded escalation.
    pipeline_drained = self.pipeline is None
    if self.pipeline is not None:
        close_pipeline_with_timeout("capture pipeline stop", self.pipeline.stop)
        if self._pipeline_wait_thread is not None:
            emit(f"[{ts()}] [CLOSE] waiting for pipeline EOS and camera release")
            def wait_pipeline(timeout):
                self._pipeline_wait_thread.join(timeout=timeout)
                return not self._pipeline_wait_thread.is_alive()
            finished = wait_for_completion(wait_pipeline, budget, progress=progress_snapshot)
            stage_reports["capture pipeline wait"] = {**budget.report(), "completed": finished
                                                       and self._pipeline_wait_error is None}
            if not finished:
                if self.exit_code == 0:
                    self.exit_code = 1
                self._force_process_exit = True
                detail = timeout_detail("capture pipeline wait") + "; capture closure unconfirmed"
                close_errors.append(detail)
                emit(f"[{ts()}] [RECORD] ERROR closing {detail}")
            if self._pipeline_wait_error is not None:
                self.exit_code = 1
                close_errors.append(
                    f"capture pipeline wait: {self._pipeline_wait_error}"
                )
            if (finished and self._pipeline_wait_error is None
                    and getattr(self.pipeline, 'closure_confirmed', False) is True):
                pipeline_drained = True
                emit(f"[{ts()}] [CLOSE] pipeline EOS and camera release completed")
        else:
            pipeline_drained = (close_pipeline_with_timeout("capture pipeline wait", self.pipeline.wait)
                                and getattr(self.pipeline, 'closure_confirmed', False) is True)
    progress.close()
    progress_snapshot = dict
    if not pipeline_drained:
        if self.exit_code == 0:
            self.exit_code = 1
        self._force_process_exit = True
        close_errors.append("native drain unconfirmed; callback-owned writers left open until process exit")
    # A blocked monitor or callback can hold a writer lock even after a
    # failed native drain. Bound cleanup as a whole, rather than hanging
    # forever in join()/close() and preventing the supervisor from seeing
    # the capture failure. Healthy pipelines still finish EOS/wait above
    # before any scientific writer is closed.
    cleanup_stall_s = min(5.0, shutdown_timeout_s)
    budget = ProgressDeadline(cleanup_stall_s, 6 * cleanup_stall_s)
    close_resource = close_pipeline_with_timeout
    if self._recording_liveness_thread is not None:
        close_resource(
            "recording liveness monitor",
            self._recording_liveness_thread.join,
        )
    if self._storage_reserve_thread is not None:
        close_resource(
            "storage reserve monitor",
            self._storage_reserve_thread.join,
        )
    # Native callbacks may still hold these resources after a failed drain.
    # Closing them can race a late write and abort before failure is reported.
    if pipeline_drained:
        if self.observations is not None:
            close_resource("observations", self.observations.close)
        for index, admission in enumerate(self.record_admissions):
            close_resource(f"record admission {index}", admission.close)
        for index, telemetry in enumerate(self.record_telemetry):
            close_resource(f"record telemetry {index}", telemetry.close)
            diagnostic_error = getattr(telemetry, 'diagnostic_error', None)
            if isinstance(diagnostic_error, str) and diagnostic_error:
                close_warnings.append(f"Recording diagnostics stream {index}: {diagnostic_error}")
        for index, boundary in enumerate(self.preview_boundaries):
            close_resource(f"preview boundary {index}", boundary.close)
        if self.frames is not None:
            close_resource("frame ledger", self.frames.close)
    # Persist evidence before optional cleanup or interpreter/native destructor
    # teardown can hang or abort. A failed capture never gets this receipt.
    budget = ProgressDeadline(min(1.0, shutdown_timeout_s), min(1.0, shutdown_timeout_s))
    if (pipeline_drained and self.pipeline is not None
            and self._explicit_stop_requested and self.exit_code == 0
            and not close_errors):
        close_resource("capture closure receipt", lambda: write_capture_closure(self.run_dir))

    # System diagnostics are optional, unlike the controller/sensor stream
    # closed separately by the supervisor. Give them their own bounded stage.
    budget = ProgressDeadline(cleanup_stall_s, cleanup_stall_s)
    close_resource("system telemetry", self.system_telemetry.stop, required=False)
    if self.system_telemetry.last_error:
        close_warnings.append(f"System telemetry unavailable: {self.system_telemetry.last_error}")

    def write_capture_status() -> None:
        run_context.write_status(
            self.run_dir,
            "capture_closed",
            capture_exit_code=self.exit_code,
            capture_close_error=close_errors[0] if close_errors else None,
            capture_close_errors=close_errors,
            shutdown_warnings=close_warnings,
            shutdown_degraded=bool(close_warnings),
            shutdown_stages=dict(stage_reports),
            system_telemetry={
                "path": str(self.system_telemetry.path),
                "sample_count": self.system_telemetry.sample_count,
                "error": self.system_telemetry.last_error,
            },
        )
    # Keep failure reporting bounded even if the filesystem is stalled.
    # The supervisor receives the terminal event and owns final run status
    # when this best-effort local status write cannot complete.
    budget = ProgressDeadline(min(1.0, shutdown_timeout_s), min(1.0, shutdown_timeout_s))
    close_resource("capture status", write_capture_status)

    emit(
        encode_child_event(
            "capture_closed",
            run_dir=str(self.run_dir),
            exit_code=self.exit_code,
            close_errors=close_errors,
        )
    )
    emit(
        f"[{ts()}] [CAPTURE] closed; post-run audit is handled independently"
    )
    emit(f"[{ts()}] [INFO] done. Files in: {self.run_dir}")
