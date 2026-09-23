"""DeepStream 9.1 inference runner built on the PyServiceMaker Pipeline API."""
from __future__ import annotations

import signal
import os
import threading
import time
import sys
from pathlib import Path
from typing import Callable

from pyservicemaker import (
    EOSMessage,
    Pipeline,
    PipelineState,
    Probe,
    StateTransitionMessage,
)

from squeakview.common import run_context
from squeakview.common.child_events import encode_child_event
from squeakview.common.diagnostics.system_telemetry import SystemTelemetryRecorder
from .contracts import (
    InferenceConfig,
    _load_class_names,
    _read_config_value,
    _validate_config,
)
from .frame_audit import (
    FLIR_FRAME_META_DESCRIPTOR,
    FrameCsvOperator,
    _flir_frame_meta_type,
    _user_meta_type,
)
from .pose_pipeline import ObservationOperator
from .pipeline_builder import (
    build_pipeline,
    camera_source_properties,
    flir_pixel_format,
)
from .recording import (
    RecordingAdmissionOperator,
    RecordingPathTelemetry,
)
from .recording_liveness import (
    RecordingLivenessMonitor,
    resolve_recording_liveness_policy,
)
from .storage_reserve import (
    StorageReserveMonitor,
    resolve_storage_reserve_policy,
)
from .video_probe import probe_video_frames


def ts() -> str:
    return time.strftime("%H:%M:%S")


def _safe_print(message: str) -> None:
    """Best-effort child logging that cannot break capture shutdown."""

    try:
        print(message, flush=True)
    except (BrokenPipeError, OSError):
        pass


def _flir_pixel_format(value: str | None) -> str:
    """Compatibility wrapper for callers importing the legacy helper."""

    return flir_pixel_format(value)


class ServiceMakerApp:
    """Own and run the PyServiceMaker SqueakView pipeline."""

    def __init__(
        self,
        config: InferenceConfig,
        *,
        pipeline_factory: Callable[[str], Pipeline] = Pipeline,
        probe_factory: Callable[[str, object], Probe] = Probe,
        system_telemetry_factory: Callable[..., SystemTelemetryRecorder] = SystemTelemetryRecorder,
    ):
        _validate_config(config)
        self.config = config
        self.pipeline_factory = pipeline_factory
        self.probe_factory = probe_factory
        if config.run_dir is None:
            raise ValueError(
                "run_dir is required; acquisition output must belong to an open project"
            )
        else:
            self.run_dir = Path(config.run_dir).expanduser()
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts = run_context.run_artifacts(self.run_dir)
        if config.failure_plan is not None:
            run_context.update_status(
                self.run_dir,
                failure_injection=config.failure_plan.as_manifest(),
                production_eligible=False,
            )
        self.pipeline: Pipeline | None = None
        self.observations: ObservationOperator | None = None
        self.frames: FrameCsvOperator | None = None
        self.record_admissions: list[RecordingAdmissionOperator] = []
        self.record_telemetry: list[RecordingPathTelemetry] = []
        self.preview_boundaries = []
        self._stop_event = threading.Event()
        self.system_telemetry = system_telemetry_factory(
            self.run_dir / "diagnostics" / "system.csv",
            on_error=self._system_telemetry_fault,
            shutdown_requested=self._stop_event.is_set,
        )
        self._ready = False
        self._stopped = False
        self.exit_code = 0
        self._explicit_stop_requested = False
        self._pipeline_wait_thread: threading.Thread | None = None
        self._pipeline_wait_error: str | None = None
        self._recording_liveness_thread: threading.Thread | None = None
        self._recording_liveness_monitor: RecordingLivenessMonitor | None = None
        self._storage_reserve_thread: threading.Thread | None = None
        self._storage_reserve_monitor: StorageReserveMonitor | None = None
        self._force_process_exit = False
        self._recording_integrity_failed = False

    def _system_telemetry_fault(self, message: str) -> None:
        """Persist observability loss immediately without aborting recording."""

        detail = str(message).strip() or "unknown system telemetry failure"
        _safe_print(f"[{ts()}] [SYSTEM] WARN: {detail}")
        try:
            run_context.update_status(
                self.run_dir,
                system_telemetry_degraded=True,
                system_telemetry_error=detail,
            )
        except Exception as exc:
            _safe_print(
                f"[{ts()}] [SYSTEM] WARN: could not persist telemetry failure: {exc}"
            )

    def _recording_fault(self, message: str) -> None:
        self._recording_integrity_failed = True
        if self.exit_code == 0:
            self.exit_code = 4
        self._stop_event.set()
        _safe_print(f"[{ts()}] [RECORD] FATAL: {message}")

    def _storage_reserve_fault(self, message: str) -> None:
        """Stop through normal EOS drain while enough reserve remains to finalize."""

        if self.exit_code == 0:
            self.exit_code = 5
        self._stop_event.set()
        _safe_print(
            encode_child_event(
                "fatal",
                run_dir=str(self.run_dir),
                error=message,
            )
        )
        _safe_print(f"[{ts()}] [STORAGE] FATAL: {message}")

    def _prewarm_cuda(self) -> None:
        if not self.config.enable_infer:
            return
        started = time.monotonic()
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch CUDA is unavailable")
            probe = torch.arange(16, device="cuda", dtype=torch.float32)
            _ = probe.to(device="cpu")
            torch.cuda.synchronize()
        except Exception as exc:
            raise RuntimeError(f"CUDA prewarm failed before camera acquisition: {exc}") from exc
        elapsed = time.monotonic() - started
        print(
            f"[{ts()}] [CUDA] prewarm complete in {elapsed:.2f}s before acquisition",
            flush=True,
        )

    @staticmethod
    def _video_frame_probe(path: Path) -> dict[str, object]:
        return probe_video_frames(path)

    @classmethod
    def _video_frame_count(cls, path: Path) -> int | None:
        count = cls._video_frame_probe(path).get("count")
        return int(count) if count is not None else None

    def _camera_properties(self, index: int) -> dict[str, object]:
        """Compatibility wrapper around the extracted source builder."""

        return camera_source_properties(self.config, self.run_dir, index)

    def build(self) -> Pipeline:
        resources = build_pipeline(
            self.config,
            self.run_dir,
            self.artifacts.raw_video,
            pipeline_factory=self.pipeline_factory,
            probe_factory=self.probe_factory,
            on_recording_fault=self._recording_fault,
        )
        self.pipeline = resources.pipeline
        self._ready_origin = resources.ready_origin
        self.frames = resources.frames
        self.observations = resources.observations
        self.record_admissions = list(resources.admissions)
        self.record_telemetry = list(resources.recording_telemetry)
        self.preview_boundaries = list(resources.preview_boundaries)
        print(
            f"[{ts()}] [INFO] PyServiceMaker pipeline built: cameras={self.config.num_cameras} "
            f"inference={'on' if self.config.enable_infer else 'off'} "
            f"run_dir={self.run_dir}",
            flush=True,
        )
        return resources.pipeline

    def _on_message(self, message) -> None:
        if (
            isinstance(message, StateTransitionMessage)
            and message.new_state == PipelineState.PLAYING
            and message.origin == self._ready_origin
            and not self._ready
        ):
            self._ready = True
            _safe_print(
                encode_child_event(
                    "pipeline_ready",
                    run_dir=str(self.run_dir),
                    ready_origin=self._ready_origin,
                )
            )
            _safe_print(f"[{ts()}] [READY] inference playing")
        elif isinstance(message, EOSMessage):
            if not self._explicit_stop_requested and self.exit_code == 0:
                self.exit_code = 1
                detail = "live capture reached EOS without an explicit stop request"
                _safe_print(
                    encode_child_event(
                        "fatal",
                        run_dir=str(self.run_dir),
                        error=detail,
                    )
                )
                _safe_print(f"[{ts()}] [FATAL] {detail}")
            self._stop_event.set()

    def _watch_pipeline_end(self) -> None:
        """Wake the lifecycle owner if Service Maker ends without an EOS event."""

        assert self.pipeline is not None
        try:
            self.pipeline.wait()
        except Exception as exc:
            self._pipeline_wait_error = f"{type(exc).__name__}: {exc}"
        if not self._stop_event.is_set() and not self._stopped:
            self.exit_code = 1
            detail = self._pipeline_wait_error or "pipeline ended without EOS"
            _safe_print(
                encode_child_event(
                    "fatal",
                    run_dir=str(self.run_dir),
                    error=detail,
                )
            )
            _safe_print(f"[{ts()}] [FATAL] capture pipeline ended unexpectedly: {detail}")
            self._stop_event.set()

    def run(self) -> int:
        if self.pipeline is None:
            self.build()
        assert self.pipeline is not None
        try:
            if not self.system_telemetry.start():
                _safe_print(
                    f"[{ts()}] [SYSTEM] WARN: system telemetry is unavailable; "
                    "capture may continue but this run cannot satisfy telemetry qualification"
                )
            self._prewarm_cuda()
            self.pipeline.start(self._on_message)
            self._pipeline_wait_thread = threading.Thread(
                target=self._watch_pipeline_end,
                daemon=True,
                name="squeakview-pipeline-wait",
            )
            self._pipeline_wait_thread.start()
            self._recording_liveness_monitor = RecordingLivenessMonitor(
                self.record_telemetry,
                self._stop_event,
                self._recording_fault,
                resolve_recording_liveness_policy(
                    self.config.fps,
                    failure_injection=self.config.failure_plan is not None,
                ),
            )
            self._recording_liveness_thread = threading.Thread(
                target=self._recording_liveness_monitor.run,
                daemon=True,
                name="squeakview-recording-liveness",
            )
            self._recording_liveness_thread.start()
            self._storage_reserve_monitor = StorageReserveMonitor(
                self.run_dir,
                self._stop_event,
                self._storage_reserve_fault,
                resolve_storage_reserve_policy(),
            )
            self._storage_reserve_thread = threading.Thread(
                target=self._storage_reserve_monitor.run,
                daemon=True,
                name="squeakview-storage-reserve",
            )
            self._storage_reserve_thread.start()
            while not self._stop_event.wait(0.2):
                pass
        except KeyboardInterrupt:
            print(f"[{ts()}] [INFO] Ctrl-C; stopping PyServiceMaker pipeline", flush=True)
        except Exception as exc:
            self.exit_code = 1
            _safe_print(
                encode_child_event(
                    "fatal",
                    run_dir=str(self.run_dir),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            _safe_print(f"[{ts()}] [FATAL] PyServiceMaker pipeline failed: {exc}")
        finally:
            self.stop()
        return self.exit_code

    def request_stop(self) -> None:
        self._explicit_stop_requested = True
        self._stop_event.set()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        # Wake every capture-owned monitor even when pipeline.start() or a
        # lifecycle callback raised before the normal stop request path.
        self._stop_event.set()
        close_errors: list[str] = []

        try:
            shutdown_timeout_s = float(
                os.environ.get("SQUEAKVIEW_PIPELINE_SHUTDOWN_TIMEOUT_S", "")
                or (5.0 if self.config.failure_plan is not None else 20.0)
            )
        except ValueError:
            shutdown_timeout_s = 20.0
        shutdown_timeout_s = max(0.1, shutdown_timeout_s)

        def close_resource(name: str, close_fn: Callable[[], object]) -> None:
            try:
                close_fn()
            except Exception as exc:
                self.exit_code = 1
                detail = f"{name}: {type(exc).__name__}: {exc}"
                close_errors.append(detail)
                _safe_print(f"[{ts()}] [RECORD] ERROR closing {detail}")

        def close_pipeline_with_timeout(
            name: str, close_fn: Callable[[], object]
        ) -> bool:
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
            if not completed.wait(shutdown_timeout_s):
                if self.exit_code == 0:
                    self.exit_code = 1
                self._force_process_exit = True
                detail = (
                    f"{name}: timed out after {shutdown_timeout_s:.1f}s; "
                    "capture is not valid"
                )
                close_errors.append(detail)
                _safe_print(f"[{ts()}] [RECORD] ERROR closing {detail}")
                return False
            if failure:
                self.exit_code = 1
                exc = failure[0]
                detail = f"{name}: {type(exc).__name__}: {exc}"
                close_errors.append(detail)
                _safe_print(f"[{ts()}] [RECORD] ERROR closing {detail}")
                return False
            return True

        if self.pipeline is not None and self._recording_integrity_failed:
            # A blocked or failed loss-intolerant recording branch may make the
            # native Service Maker stop call retain the GIL indefinitely.  The
            # run is already invalid at this point, so flush the Python-owned
            # scientific ledgers/status below and terminate without entering
            # an unsafe native drain.  ``run`` uses os._exit only after those
            # resources have been closed.
            self._force_process_exit = True
            detail = (
                "capture pipeline native shutdown bypassed after a fatal "
                "recording-integrity failure; capture is not valid"
            )
            close_errors.append(detail)
            _safe_print(f"[{ts()}] [RECORD] ERROR closing {detail}")
        elif self.pipeline is not None:
            close_pipeline_with_timeout("capture pipeline stop", self.pipeline.stop)
            if self._pipeline_wait_thread is not None:
                self._pipeline_wait_thread.join(timeout=shutdown_timeout_s)
                if self._pipeline_wait_thread.is_alive():
                    if self.exit_code == 0:
                        self.exit_code = 1
                    self._force_process_exit = True
                    detail = (
                        "capture pipeline wait: timed out after "
                        f"{shutdown_timeout_s:.1f}s; capture is not valid"
                    )
                    close_errors.append(detail)
                    _safe_print(f"[{ts()}] [RECORD] ERROR closing {detail}")
                if self._pipeline_wait_error is not None:
                    self.exit_code = 1
                    close_errors.append(
                        f"capture pipeline wait: {self._pipeline_wait_error}"
                    )
            else:
                close_pipeline_with_timeout("capture pipeline wait", self.pipeline.wait)
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
        if self.observations is not None:
            close_resource("observations", self.observations.close)
        for index, admission in enumerate(self.record_admissions):
            close_resource(f"record admission {index}", admission.close)
        for index, telemetry in enumerate(self.record_telemetry):
            close_resource(f"record telemetry {index}", telemetry.close)
        for index, boundary in enumerate(self.preview_boundaries):
            close_resource(f"preview boundary {index}", boundary.close)
        if self.frames is not None:
            close_resource("frame ledger", self.frames.close)
        close_resource("system telemetry", self.system_telemetry.stop)
        try:
            run_context.write_status(
                self.run_dir,
                "capture_closed",
                capture_exit_code=self.exit_code,
                capture_close_error=close_errors[0] if close_errors else None,
                capture_close_errors=close_errors,
                system_telemetry={
                    "path": str(self.system_telemetry.path),
                    "sample_count": self.system_telemetry.sample_count,
                    "error": self.system_telemetry.last_error,
                },
            )
        except Exception as exc:
            self.exit_code = 1
            _safe_print(f"[{ts()}] [RECORD] ERROR marking capture closed: {exc}")

        _safe_print(
            encode_child_event(
                "capture_closed",
                run_dir=str(self.run_dir),
                exit_code=self.exit_code,
                close_errors=close_errors,
            )
        )
        _safe_print(
            f"[{ts()}] [CAPTURE] closed; post-run audit is handled independently"
        )
        _safe_print(f"[{ts()}] [INFO] done. Files in: {self.run_dir}")

def run(config: InferenceConfig) -> int:
    app = ServiceMakerApp(config)

    def _handle_signal(sig_num, _frame) -> None:
        app.request_stop()
        _safe_print(f"[{ts()}] [SIG] {signal.Signals(sig_num).name}; stopping")

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    app.build()
    result = app.run()
    if app._force_process_exit:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(result if result != 0 else 1)
    return result
