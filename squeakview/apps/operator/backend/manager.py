from __future__ import annotations

"""Backend orchestrator for the operator GUI."""

import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

from squeakview.apps.operator.backend import (
    capture_events,
    capture_drain,
    finalizer,
    lifecycle,
    manifest,
    preflight,
    process,
    startup,
)
from squeakview.apps.operator.backend.run_lock import AcquisitionLock
from squeakview.apps.operator.backend.events import (
    BackendEvent,
    EventSubscriber,
    RunPhase,
    RunSnapshot,
    RunStateMachine,
)
from squeakview import model_package
from squeakview.common import dashboard as dashboard_util, run_context
from squeakview.common import qualification_barrier
from squeakview.common.device_context import device_context_snapshot
from squeakview.common.diagnostics.qualification_matrix import (
    resolve_qualification_case_binding,
)
from squeakview.common.failure_injection import FailurePlan, load_failure_plan
from squeakview.common import serial as serial_util
from squeakview.project import PROJECT_ENV, RuntimeContext


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _bounded_ready_timeout() -> float:
    try:
        value = float(os.environ.get("SQUEAKVIEW_INFERENCE_READY_TIMEOUT", "30"))
    except (TypeError, ValueError):
        return 30.0
    if not math.isfinite(value) or not 1.0 <= value <= 600.0:
        return 30.0
    return value



@dataclass(slots=True)
class RunState:
    inference: Optional[process.ProcessHandle] = None
    serial: Optional[serial_util.SerialHandle] = None
    run_dir: Optional[Path] = None

    def any_running(self) -> bool:
        return bool(self.inference and self.inference.is_running())


class OperatorBackend:
    def __init__(
        self,
        emit_log: Callable[[str], None],
        ingest_dashboard: Optional[Callable[[dashboard_util.DashboardEvent], None]] = None,
        on_run_started: Callable[[], None] | None = None,
        on_run_failed: Callable[[str], None] | None = None,
        *,
        runtime_context: RuntimeContext,
        acquisition_owner: str = manifest.IN_PROCESS_DEV_OWNER,
    ):
        if acquisition_owner not in {
            manifest.DURABLE_SUPERVISOR_OWNER,
            manifest.IN_PROCESS_DEV_OWNER,
        }:
            raise ValueError(f"unsupported acquisition owner: {acquisition_owner!r}")
        self._acquisition_owner = acquisition_owner
        self.runtime_context = runtime_context
        self.emit = emit_log
        self.ingest = ingest_dashboard
        self.on_run_started = on_run_started
        self.on_run_failed = on_run_failed
        self.state = RunState()
        self.launch_cfg = process.LaunchConfig()
        self._metadata_written = False
        self._run_storage_info: dict[str, Any] = {}
        self._run_started_at: str | None = None
        self._inference_ready = threading.Event()
        self._stop_requested = threading.Event()
        # Lifecycle subscribers may synchronously report IPC backpressure and
        # cancel the lease while a terminal transition is being published.
        self._operator_lease_lock = threading.RLock()
        self._operator_lease_available = True
        self._finalize_lock = threading.Lock()
        self._run_finalized = True
        self._recording_started = False
        self._controller_started = False
        self._serial_runtime_armed = False
        self._state_machine = RunStateMachine()
        self._subscribers: list[EventSubscriber] = []
        self._last_error: str | None = None
        self._model_snapshot: dict[str, Any] | None = None
        self._effective_deepstream_config: (
            process.EffectiveDeepStreamConfig | None
        ) = None
        self._device_context: dict[str, object] | None = None
        self._task_config_snapshot: dict[str, object] | None = None
        self._preflight_evidence: dict[str, object] | None = None
        self._qualification_case: dict[str, Any] | None = None
        self._manifest_service = manifest.RunManifestService(self._log)
        self._capture_drain_coordinator = capture_drain.CaptureDrainCoordinator()
        self._failure_plan: FailurePlan | None = None
        self._acquisition_lock = AcquisitionLock(
            runtime_context.user.acquisition_lock
        )

    def _log(self, message: str) -> None:
        self.emit(f"[{_now()}] {message}")

    def subscribe(self, callback: EventSubscriber) -> None:
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def snapshot(self) -> RunSnapshot:
        capture = self.state.inference
        return RunSnapshot(
            phase=self._state_machine.phase,
            run_dir=self.state.run_dir,
            error=self._last_error,
            capture_running=bool(capture and capture.is_running()),
            finalization_in_progress=self.finalization_in_progress,
        )

    @property
    def current_snapshot(self) -> RunSnapshot:
        return self.snapshot()

    @property
    def runtime_config(self) -> process.LaunchConfig:
        return self.launch_cfg

    @property
    def acquisition_owner(self) -> str:
        """Immutable provenance for the process that owns acquisition."""

        return self._acquisition_owner

    def _transition(
        self,
        phase: RunPhase,
        *,
        message: str | None = None,
        **payload: object,
    ) -> None:
        previous = self._state_machine.phase
        self._state_machine.transition(phase)
        if previous == phase:
            return
        event = BackendEvent(
            type="phase_changed",
            phase=phase,
            run_dir=self.state.run_dir,
            message=message,
            payload={"previous_phase": previous.value, **payload},
        )
        for callback in tuple(self._subscribers):
            try:
                callback(event)
            except Exception as exc:
                self._log(f"[BACKEND] event subscriber failed: {exc}")

    @property
    def finalization_in_progress(self) -> bool:
        """True while a stop/failure worker owns the run-finalization path."""

        return self._finalize_lock.locked() and not self._run_finalized

    def _serial_emit(self, message: str) -> None:
        self.emit(message)
        if self.ingest and "【SER】" in message:
            raw = message.split("【SER】", 1)[1].strip()
            event = dashboard_util.DashboardEvent.parse(raw)
            if event is not None:
                self.ingest(event)

    def _capture_emit(self, message: str) -> None:
        self.emit(message)

    def _on_serial_fatal(
        self, source: serial_util.SerialHandle, message: str
    ) -> None:
        """Schedule fail-closed shutdown after runtime serial integrity loss.

        This callback originates on the serial reader thread.  Finalization
        runs on a separate thread because it closes and joins the serial
        reader as part of the ordered capture shutdown.
        """

        if (
            not self._serial_runtime_armed
            or self.state.serial is not source
            or self._run_finalized
        ):
            return
        error = f"serial acquisition integrity failed: {message}"
        self._log(f"[SER] {error}")
        threading.Thread(
            target=self._finalize_run,
            kwargs={"final_state": "failed", "error": error},
            daemon=True,
            name="squeakview-serial-fatal",
        ).start()

    def _inference_emit(self, message: str) -> None:
        self.emit(message)
        action = capture_events.interpret_capture_output(
            message,
            expected_run_dir=self.state.run_dir,
            recording_started=self._recording_started,
        )
        if action.warning is not None:
            self._log(f"[BACKEND] WARN: {action.warning}")
        if action.run_dir is not None:
            self._attach_run_dir(action.run_dir)
        if action.kind == "ready":
            self._mark_inference_ready()
            return
        if action.kind == "fatal":
            self._on_capture_protocol_fatal(action.error or "capture protocol failure")
            return

    def _attach_run_dir(self, run_dir: Path) -> None:
        self.state.run_dir = run_dir
        if self.state.serial:
            self.state.serial.set_csv_path(run_dir)
        self._ensure_metadata(run_dir)

    def _on_capture_protocol_fatal(self, message: str) -> None:
        if self._run_finalized or self._stop_requested.is_set():
            return
        error = f"capture supervision failed: {message}"
        self._log(f"[DS] {error}")
        self._stop_requested.set()
        # Wake startup so it observes the stop request and cannot issue START.
        self._inference_ready.set()
        threading.Thread(
            target=self._finalize_run,
            kwargs={"final_state": "failed", "error": error},
            daemon=True,
            name="squeakview-capture-protocol-fatal",
        ).start()

    def _mark_inference_ready(self) -> None:
        if self._recording_started or self._run_finalized or self._stop_requested.is_set():
            return
        if self._effective_deepstream_config is not None:
            try:
                # nvinfer opens package artifacts after process spawn.  Bind
                # the model-loaded readiness boundary before recording or a
                # triggered controller START can be published.
                process.verify_effective_deepstream_config(
                    self._effective_deepstream_config
                )
            except Exception as exc:
                error = f"effective DeepStream artifact verification failed: {exc}"
                self._log(f"[MODEL] {error}")
                self._stop_requested.set()
                self._inference_ready.set()
                threading.Thread(
                    target=self._finalize_run,
                    kwargs={"final_state": "failed", "error": error},
                    daemon=True,
                    name="squeakview-model-provenance-fatal",
                ).start()
                return
        run_dir = self.state.run_dir
        if run_dir is not None:
            try:
                # Persist the complete pre-start manifest before waking the
                # startup worker.  A triggered run is not yet recording here:
                # the controller still has to START and produce its first TTL.
                self._write_run_manifest(run_dir, required=True)
            except Exception as exc:
                error = (
                    "required run metadata could not be persisted before "
                    f"controller START: {exc}"
                )
                self._log(f"[SAVE] {error}")
                self._stop_requested.set()
                # Wake a start_run call waiting for readiness; it checks the
                # stop request before it can send START to the controller.
                self._inference_ready.set()
                threading.Thread(
                    target=self._finalize_run,
                    kwargs={"final_state": "failed", "error": error},
                    daemon=True,
                    name="squeakview-metadata-fatal",
                ).start()
                return
        if not bool(getattr(self.launch_cfg, "trigger_on", False)):
            self._confirm_recording()
        else:
            self._log("[BACKEND] inference ready; waiting for controller START and first TTL")
        self._inference_ready.set()

    def _confirm_recording(self) -> bool:
        """Publish recording only once acquisition is actually confirmed."""

        if self._recording_started:
            return True
        if self._run_finalized or self._stop_requested.is_set():
            return False
        run_dir = self.state.run_dir
        if run_dir is not None:
            try:
                run_context.write_status(run_dir, "recording")
                self._write_run_manifest(run_dir, required=True)
            except Exception as exc:
                error = f"required recording-state metadata could not be persisted: {exc}"
                self._log(f"[SAVE] {error}")
                self._stop_requested.set()
                self._inference_ready.set()
                threading.Thread(
                    target=self._finalize_run,
                    kwargs={"final_state": "failed", "error": error},
                    daemon=True,
                    name="squeakview-recording-metadata-fatal",
                ).start()
                return False
        self._recording_started = True
        self._transition(RunPhase.RECORDING)
        self._log("[BACKEND] run is recording")
        if self.on_run_started is not None:
            try:
                self.on_run_started()
            except Exception as exc:
                self._log(f"[BACKEND] run-start callback failed: {exc}")
        return True

    def _on_inference_exit(self, returncode: int) -> None:
        if self._stop_requested.is_set() or self._run_finalized:
            return
        phase = "after readiness" if self._inference_ready.is_set() else "before readiness"
        error = f"inference process exited unexpectedly {phase} (exit code {returncode})"
        self._log(f"[DS] {error}")
        self._finalize_run(
            final_state="failed",
            error=error,
            terminate_inference=False,
            known_inference_returncode=returncode,
        )

    @staticmethod
    def _serial_open_failure_message(handle: serial_util.SerialHandle, port: str, baud: int) -> str:
        detail = str(getattr(handle, "last_error", "") or "unknown serial error").strip()
        message = f"Could not open serial port {port} at {baud} baud.\n\nSystem error: {detail}"
        permission_terms = ("permission denied", "access denied", "operation not permitted")
        if any(term in detail.lower() for term in permission_terms):
            message += (
                "\n\nSerial access was denied. Run `bash scripts/setup_jetson.sh`, then "
                "reboot before retrying. The setup script safely adds the desktop user "
                "to dialout and installs the other required device dependencies."
            )
        elif "no such file" in detail.lower() or "cannot find" in detail.lower():
            message += "\n\nCheck that the controller is connected and that the selected serial port is correct."
        return message

    def _start_run_dir_watch(self) -> None:
        run_dir = self.state.run_dir
        if run_dir is None or not run_dir.is_dir():
            raise RuntimeError("prepared run directory is unavailable")

    def _set_fan_max(self) -> None:
        """Best-effort fan control, opt-in because jetson_clocks usually needs privileges."""
        fan_flag = os.environ.get("SQUEAKVIEW_SET_FAN", "0").lower()
        if fan_flag not in {"1", "true", "yes", "on"}:
            return
        try:
            result = subprocess.run(
                ["jetson_clocks", "--fan"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=4,
            )
            if result.returncode == 0:
                self._log("[SYS] jetson_clocks --fan applied")
            else:
                self._log(f"[SYS] jetson_clocks --fan failed (rc={result.returncode}): {result.stderr.strip()}")
        except FileNotFoundError:
            self._log("[SYS] jetson_clocks not found; fan not adjusted")
        except Exception as exc:
            self._log(f"[SYS] jetson_clocks error: {exc}")

    def _manifest_context(self) -> manifest.RunManifestContext:
        return manifest.RunManifestContext(
            config=self.launch_cfg,
            application_root=self.runtime_context.app.root,
            project_root=self.runtime_context.project.paths.root,
            project_id=self.runtime_context.project.metadata.project_id,
            project_name=self.runtime_context.project.metadata.name,
            runs_root=self.runtime_context.project.paths.runs,
            created_at=self._run_started_at,
            storage=self._run_storage_info,
            model_snapshot=self._model_snapshot,
            effective_deepstream=(
                self._effective_deepstream_config.manifest_snapshot()
                if self._effective_deepstream_config is not None
                else None
            ),
            device_context=self._device_context,
            failure_plan=(
                self._failure_plan.as_manifest()
                if self._failure_plan is not None
                else None
            ),
            acquisition_owner=self._acquisition_owner,
            task_config_snapshot=self._task_config_snapshot,
            preflight_evidence=self._preflight_evidence,
            qualification_case=self._qualification_case,
            controller_watchdog=(
                self.state.serial.watchdog_snapshot
                if self.state.serial is not None
                else None
            ),
        )

    # Compatibility delegates are intentionally retained: callers and tests
    # can patch these seams while manifest policy lives in RunManifestService.
    @staticmethod
    def _file_info(path: Path) -> dict[str, Any]:
        return manifest.RunManifestService.file_info(path)

    @staticmethod
    def _git_snapshot() -> dict[str, Any]:
        return manifest.RunManifestService.git_snapshot(process.APPLICATION_ROOT)

    def _run_output_snapshot(self, run_dir: Path) -> dict[str, Any]:
        return self._manifest_service.output_snapshot(run_dir)

    def _bottle_manifest_snapshot(self, run_dir: Path) -> dict[str, Any]:
        return self._manifest_service.bottle_snapshot(run_dir)

    def _build_run_manifest(self, run_dir: Path) -> dict[str, Any]:
        return self._manifest_service.build(
            run_dir,
            self._manifest_context(),
            output_snapshot=self._run_output_snapshot,
            bottle_snapshot=self._bottle_manifest_snapshot,
        )

    def _write_run_manifest(self, run_dir: Path, *, required: bool = False) -> bool:
        return self._manifest_service.write(
            run_dir,
            self._manifest_context(),
            required=required,
            build_manifest=self._build_run_manifest,
            output_snapshot=self._run_output_snapshot,
            bottle_snapshot=self._bottle_manifest_snapshot,
        )

    def _write_bottle_measurements(self, run_dir: Path, bottles: dict[str, Any] | None) -> dict[str, Any]:
        summary = run_context.write_bottle_artifacts(run_dir, bottles)
        state = "complete" if summary.get("complete") else "incomplete"
        self._log(f"[BOTTLES] saved {state} bottle metadata → {run_dir / run_context.BOTTLE_MEASUREMENTS_FILENAME}")
        for warning in summary.get("warnings", []):
            self._log(f"[BOTTLES] warning: {warning}")
        return summary

    def save_bottle_measurements(
        self,
        bottles: dict[str, Any] | None,
        run_dir: Path | None = None,
    ) -> dict[str, Any]:
        target = Path(run_dir) if run_dir is not None else self.state.run_dir
        if target is None:
            raise RuntimeError("no active run directory for bottle metadata")
        target = self.runtime_context.project.paths.resolve_path(
            target,
            within=self.runtime_context.project.paths.runs,
            must_exist=True,
        )
        summary = self._write_bottle_measurements(target, bottles)
        self._write_run_manifest(target)
        try:
            run_context.update_status(
                target,
                bottle_measurements_complete=bool(summary.get("complete")),
                outputs=self._run_output_snapshot(target),
            )
        except Exception as exc:
            self._log(f"[BOTTLES] status update failed: {exc}")
        return summary

    def _ensure_metadata(self, run_dir: Path) -> None:
        if self._metadata_written:
            return
        if self._write_run_manifest(run_dir):
            self._metadata_written = True

    @staticmethod
    def _capture_drain_snapshot(run_dir: Path, camera_count: int) -> tuple[int, ...]:
        return capture_drain.ledger_size_snapshot(run_dir, camera_count)

    @staticmethod
    def _last_ledger_line(path: Path) -> str | None:
        return capture_drain.last_complete_ledger_line(path)

    @classmethod
    def _capture_drain_counts(
        cls, run_dir: Path, camera_count: int
    ) -> tuple[int | None, ...]:
        """Return source/admission counts inferred from monotonic ledger indices."""

        del cls
        return capture_drain.ledger_frame_counts(run_dir, camera_count)

    def _wait_for_capture_drain(
        self, run_dir: Path, *, expected_ttl_count: int | None = None
    ) -> bool:
        """Wait for source and recording ledgers to become quiescent after ACK_STOP."""

        return self._capture_drain_coordinator.wait(
            capture_drain.CaptureDrainRequest(
                run_dir=run_dir,
                camera_count=int(getattr(self.launch_cfg, "num_cameras", 1)),
                phase=self._state_machine.phase,
                expected_ttl_count=expected_ttl_count,
            ),
            capture_drain.CaptureDrainHooks(
                write_status=run_context.write_status,
                transition=self._transition,
            ),
        )

    def _run_capture_finalizer(self, run_dir: Path) -> int:
        alignment_required = bool(
            getattr(self.launch_cfg, "serial_enabled", False)
            and getattr(self.launch_cfg, "trigger_on", False)
        )
        return finalizer.run_capture_finalizer(
            run_dir,
            camera_count=int(getattr(self.launch_cfg, "num_cameras", 1)),
            enable_infer=bool(getattr(self.launch_cfg, "inference_enabled", True)),
            enable_align=alignment_required,
            emit=self._log,
            force_timeout=bool(
                self._failure_plan is not None
                and self._failure_plan.target == "shutdown"
                and self._failure_plan.kind == "finalizer_timeout"
            ),
        )

    def _finalize_run(
        self,
        *,
        final_state: str,
        error: str | None = None,
        terminate_inference: bool = True,
        known_inference_returncode: int | None = None,
    ) -> bool:
        notify_failure = False
        with self._finalize_lock:
            if self._run_finalized:
                return False
            self._stop_requested.set()
            inference = self.state.inference
            try:
                result = lifecycle.finalize_run(
                    lifecycle.FinalizationRequest(
                        final_state=final_state,
                        error=error,
                        terminate_capture=terminate_inference,
                        known_capture_returncode=known_inference_returncode,
                        run_dir=self.state.run_dir,
                        capture=inference,
                        serial=self.state.serial,
                        capture_running=bool(inference and inference.is_running()),
                        controller_started=self._controller_started,
                        trigger_on=bool(getattr(self.launch_cfg, "trigger_on", False)),
                        phase=self._state_machine.phase,
                        controller_protocol=getattr(
                            self.launch_cfg, "controller_protocol", "legacy"
                        ),
                        alignment_required=bool(
                            getattr(self.launch_cfg, "serial_enabled", False)
                            and getattr(self.launch_cfg, "trigger_on", False)
                        ),
                        failure_plan=self._failure_plan,
                    ),
                    lifecycle.FinalizationHooks(
                        log=self._log,
                        transition=self._transition,
                        wait_for_capture_drain=self._wait_for_capture_drain,
                        run_capture_finalizer=self._run_capture_finalizer,
                        write_run_manifest=self._write_run_manifest,
                        run_output_snapshot=self._run_output_snapshot,
                        sleep=time.sleep,
                    ),
                )
            except Exception as exc:
                # No shutdown helper failure may leave the application looking
                # active or retain the process-wide acquisition lock.
                detail = f"shutdown coordinator failed: {type(exc).__name__}: {exc}"
                self._log(f"[BACKEND] {detail}")
                try:
                    if self.state.serial is not None:
                        self.state.serial.close()
                except Exception as close_exc:
                    self._log(f"[SER] close after coordinator failure failed: {close_exc}")
                if self.state.run_dir is not None:
                    try:
                        run_context.write_status(
                            self.state.run_dir, "failed", error=detail
                        )
                    except Exception as status_exc:
                        self._log(f"[SAVE] failed to persist coordinator error: {status_exc}")
                result = lifecycle.FinalizationResult(
                    final_state="failed",
                    error=detail,
                    capture_returncode=None,
                    serial_closed=self.state.serial is not None,
                )
            # Make terminal persistence/phase atomic against GUI-lease loss.
            # If the lease disappeared while an existing finalizer held its
            # lock, successful artifact closure remains useful evidence but
            # the run itself must be recorded as failed.
            with self._operator_lease_lock:
                if not self._operator_lease_available and not result.failed:
                    lease_error = "operator GUI lost"
                    if self.state.run_dir is not None:
                        try:
                            run_context.write_status(
                                self.state.run_dir,
                                "failed",
                                error=lease_error,
                            )
                            self._write_run_manifest(self.state.run_dir)
                        except Exception as exc:
                            self._log(
                                "[SAVE] failed to persist operator-loss terminal state: "
                                f"{exc}"
                            )
                    result = lifecycle.FinalizationResult(
                        final_state="failed",
                        error=lease_error,
                        capture_returncode=result.capture_returncode,
                        serial_closed=result.serial_closed,
                    )
                self.state.inference = None
                if result.serial_closed:
                    self.state.serial = None
                self._run_finalized = True
                self._acquisition_lock.release()
                notify_failure = result.failed
                error = result.error
                self._last_error = error if notify_failure else None
                self._transition(
                    RunPhase.FAILED if notify_failure else RunPhase.FINALIZED,
                    message=error,
                )

        if notify_failure and self.on_run_failed is not None:
            try:
                self.on_run_failed(error or "run failed")
            except Exception as exc:
                self._log(f"[BACKEND] run-failure callback failed: {exc}")
        return True


    def _startup_hooks(self) -> startup.StartupHooks:
        def acquire_lock() -> bool:
            with self._operator_lease_lock:
                if not self._operator_lease_available:
                    return False
                return self._acquisition_lock.acquire()

        def validate_model(path: Path) -> startup.ModelSelection:
            selected = model_package.validate_model_package(path)
            from squeakview.apps.inference.contracts import read_config_value

            raw_batch_size = read_config_value(selected.config, "batch-size")
            return startup.ModelSelection(
                name=selected.name,
                snapshot=selected.manifest_snapshot(),
                engine_build_identity=selected.engine_build_identity,
                batch_size=(
                    int(raw_batch_size) if raw_batch_size is not None else None
                ),
            )

        def initialize_runtime(
            cfg: process.LaunchConfig,
            started_at: str,
            storage: dict[str, Any],
            model: startup.ModelSelection | None,
            failure_plan: FailurePlan | None,
            device: dict[str, object],
            qualification_case: dict[str, Any] | None,
        ) -> None:
            with self._operator_lease_lock:
                if not self._operator_lease_available:
                    raise RuntimeError("operator GUI lost")
                self.launch_cfg = cfg
                self.state.run_dir = None
                self._metadata_written = False
                self._run_started_at = started_at
                self._run_storage_info = dict(storage)
                self._model_snapshot = dict(model.snapshot) if model else None
                self._effective_deepstream_config = None
                self._failure_plan = failure_plan
                self._device_context = dict(device)
                self._qualification_case = (
                    dict(qualification_case)
                    if qualification_case is not None
                    else None
                )
                self._task_config_snapshot = None
                self._inference_ready.clear()
                self._stop_requested.clear()
            self._recording_started = False
            self._controller_started = False
            self._serial_runtime_armed = False

        def establish_run(prepared: startup.PreparedRun) -> None:
            self.launch_cfg = prepared.config
            self.state.run_dir = prepared.run_dir
            self._run_finalized = False
            self._last_error = None
            self._state_machine.reset()
            self._transition(RunPhase.CREATED)

        def persist_prestart(prepared: startup.PreparedRun) -> None:
            cfg = prepared.config
            if cfg.inference_enabled:
                if cfg.ds_cfg is None or prepared.model is None:
                    raise RuntimeError("validated inference configuration is unavailable")
                artifact_fields = {
                    "deepstream_config": "config",
                    "pose_sidecar": "pose_sidecar",
                    "class_labels": "classes",
                    "keypoint_labels": "keypoint_labels",
                    "onnx": "onnx",
                    "engine": "engine",
                    "custom_parser": "parser_library",
                }
                validated_artifacts: dict[str, Path] = {}
                for runtime_name, snapshot_name in artifact_fields.items():
                    raw_path = prepared.model.snapshot.get(snapshot_name)
                    if not isinstance(raw_path, str) or not raw_path:
                        raise RuntimeError(
                            f"validated model {snapshot_name} path is unavailable"
                        )
                    validated_artifacts[runtime_name] = Path(raw_path)
                effective = process.prepare_effective_deepstream_config(
                    cfg.ds_cfg,
                    prepared.run_dir,
                    self._inference_emit,
                    validated_artifacts=validated_artifacts,
                )
                self._effective_deepstream_config = effective
                self.launch_cfg = replace(cfg, ds_cfg=effective.path)
            self._task_config_snapshot = (
                manifest.snapshot_task_config(prepared.run_dir, cfg.task_cfg)
                if cfg.task_cfg is not None
                else None
            )
            disqualifiers = manifest.production_disqualifiers(
                prepared.failure_plan.as_manifest()
                if prepared.failure_plan is not None
                else None,
                acquisition_owner=self._acquisition_owner,
                preflight_evidence=self._preflight_evidence,
            )
            if cfg.controller_protocol == "watchdog_v1_experimental":
                disqualifiers = (*disqualifiers, "controller_watchdog_unqualified")
            run_context.write_status(
                prepared.run_dir,
                "created",
                run_id=prepared.run_dir.name,
                run_directory=str(prepared.run_dir),
                experiment_name=(cfg.experiment_name or "").strip() or None,
                mouse_id=(cfg.mouse_id or "").strip() or None,
                failure_injection=(
                    prepared.failure_plan.as_manifest()
                    if prepared.failure_plan is not None
                    else None
                ),
                production_eligible=not disqualifiers,
                production_disqualifiers=list(disqualifiers),
                process_topology={"acquisition_owner": self._acquisition_owner},
                task_config=self._task_config_snapshot,
                preflight=self._preflight_evidence,
                qualification=self._qualification_case,
            )
            self._write_bottle_measurements(
                prepared.run_dir, getattr(cfg, "bottles", None)
            )
            self._write_run_manifest(prepared.run_dir, required=True)
            run_context.write_status(prepared.run_dir, "starting")
            self._write_run_manifest(prepared.run_dir, required=True)
            self._metadata_written = True
            self._transition(RunPhase.STARTING)
            qualification_barrier.wait_at_barrier(
                prepared.run_dir,
                "pre_capture",
                stop_requested=self._stop_requested.is_set,
            )

        def create_serial(
            cfg: process.LaunchConfig, failure_plan: FailurePlan | None
        ) -> serial_util.SerialHandle:
            holder: dict[str, serial_util.SerialHandle] = {}

            def on_serial_fatal(message: str) -> None:
                source = holder.get("handle")
                if source is not None:
                    self._on_serial_fatal(source, message)

            handle = serial_util.SerialHandle(
                cfg.serial_port,
                cfg.serial_baud,
                self._serial_emit,
                on_fatal=on_serial_fatal,
                failure_plan=failure_plan,
            )
            holder["handle"] = handle
            return handle

        def ready_timeout() -> float:
            return _bounded_ready_timeout()

        def spawn_capture(_cfg: process.LaunchConfig) -> process.ProcessHandle:
            kwargs: dict[str, object] = {"on_exit": self._on_inference_exit}
            if self._effective_deepstream_config is not None:
                kwargs["effective_config"] = self._effective_deepstream_config
            return process.spawn_inference(
                self.launch_cfg,
                self._inference_emit,
                **kwargs,
            )

        def start_controller(
            handle: serial_util.SerialHandle, fps: int
        ) -> bool:
            # Cancellation and controller arming are one lease-atomic action;
            # the potentially blocking TTL wait happens after this lock.
            with self._operator_lease_lock:
                if (
                    not self._operator_lease_available
                    or self._stop_requested.is_set()
                ):
                    return False
                if self.launch_cfg.controller_protocol == "watchdog_v1_experimental":
                    handle.arm_watchdog_v1(int(fps), timeout_s=2.0)
                    handle.log_marker("WATCHDOG_V1_ARM_ACKED")
                else:
                    handle.send_start(int(fps))
                self._controller_started = True
                return True

        return startup.StartupHooks(
            log=self._log,
            resolve_task_path=lambda path: self.runtime_context.project.paths.resolve_path(
                path,
                within=self.runtime_context.project.paths.tasks,
            ),
            resolve_model_path=lambda path: self.runtime_context.project.paths.resolve_path(
                path,
                within=self.runtime_context.project.paths.models,
            ),
            resolve_failure_plan_path=lambda path: self.runtime_context.project.paths.resolve_path(
                path,
                within=self.runtime_context.project.paths.qualification,
            ),
            load_failure_plan=load_failure_plan,
            validate_model=validate_model,
            assert_storage_ready=lambda: run_context.assert_runs_dir_ready(
                self.runtime_context.project.paths.runs
            ),
            acquire_lock=acquire_lock,
            release_lock=self._acquisition_lock.release,
            now_iso=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"),
            device_context=device_context_snapshot,
            set_fan_max=self._set_fan_max,
            create_run_dir=lambda **kwargs: run_context.create_run_dir(
                runs_dir=self.runtime_context.project.paths.runs,
                **kwargs,
            ),
            preview_socket_paths=process.preview_socket_paths,
            initialize_runtime=initialize_runtime,
            establish_run=establish_run,
            persist_prestart=persist_prestart,
            serial_available=serial_util.have_pyserial,
            create_serial=create_serial,
            set_serial=lambda handle: setattr(self.state, "serial", handle),
            arm_serial_runtime=lambda: setattr(self, "_serial_runtime_armed", True),
            spawn_capture=spawn_capture,
            set_capture=lambda handle: setattr(self.state, "inference", handle),
            after_capture_spawn=lambda run_dir: qualification_barrier.wait_at_barrier(
                run_dir,
                "after_spawn_before_ready",
                stop_requested=self._stop_requested.is_set,
            ),
            run_finalized=lambda: self._run_finalized,
            finalize_failure=lambda error, terminate: self._finalize_run(
                final_state="failed",
                error=error,
                terminate_inference=terminate,
            ),
            wait_ready=lambda timeout: self._inference_ready.wait(timeout=timeout),
            stop_requested=self._stop_requested.is_set,
            start_controller=start_controller,
            start_run_dir_watch=self._start_run_dir_watch,
            ready_timeout=ready_timeout,
            serial_open_failure_message=self._serial_open_failure_message,
            resolve_qualification_case=lambda cfg, device: (
                resolve_qualification_case_binding(
                    cfg,
                    device,
                    default_matrix_path=(
                        self.runtime_context.project.paths.qualification
                        / "matrix.v1.yaml"
                    ),
                    environ=os.environ,
                )
            ),
        )

    def start_run(self, cfg: process.LaunchConfig) -> bool:
        try:
            startup.validate_startup_policy(cfg)
        except ValueError as exc:
            self._last_error = f"invalid run configuration: {exc}"
            self._log(f"[BACKEND] {self._last_error}")
            return False
        if not self.state.any_running():
            result = preflight.run_preflight(
                preflight.PreflightRequest(
                    capture_backend=cfg.capture_backend,
                    inference_enabled=cfg.inference_enabled,
                    ds_cfg=cfg.ds_cfg,
                    serial_enabled=cfg.serial_enabled,
                    serial_port=cfg.serial_port,
                ),
                workspace=self.runtime_context.app.root,
                python_bin=os.fsdecode(os.environ.get("PYTHON_BIN") or sys.executable),
                emit=self._log,
                environ={
                    **os.environ,
                    PROJECT_ENV: str(self.runtime_context.project.paths.root),
                },
            )
            self._preflight_evidence = preflight.evidence_snapshot(result)
            if not result.passed:
                self._last_error = result.message
                self._log(f"[BACKEND] preflight blocked acquisition: {result.message}")
                return False
        result = startup.start_run(
            startup.StartupRequest(
                config=cfg,
                already_running=self.state.any_running(),
            ),
            self._startup_hooks(),
        )
        if not result.started and result.error:
            self._last_error = result.error
        if result.started and bool(getattr(cfg, "trigger_on", False)):
            return self._confirm_recording()
        return result.started

    def stop_run(self) -> None:
        self._finalize_run(final_state="finalized")

    def clear_feeder_jam(self) -> str:
        """Clear the firmware latch through the currently owned serial link."""

        handle = self.state.serial
        if handle is None:
            raise ConnectionError(
                "the controller is not connected; the feeder jam remains latched"
            )
        return handle.clear_feeder_jam(timeout_s=2.0)

    def cancel_operator_lease(self) -> None:
        """Persistently cancel startup and wake any readiness wait."""

        with self._operator_lease_lock:
            self._operator_lease_available = False
            self._stop_requested.set()
            self._inference_ready.set()

    def abort_run(self, error: str) -> bool:
        """Interrupt an active run and execute ordered fail-closed finalization."""

        self.cancel_operator_lease()
        if self.snapshot().phase in {
            RunPhase.IDLE,
            RunPhase.FINALIZED,
            RunPhase.FAILED,
        }:
            return False
        # Wake a startup wait immediately so it cannot arm the controller after
        # the supervisor has observed client EOF.
        return self._finalize_run(
            final_state="failed",
            error=str(error).strip() or "run aborted by supervisor",
        )

    def fail_operator_loss(self) -> bool:
        """Compatibility helper for the durable supervisor's GUI lease."""

        return self.abort_run("operator GUI lost")

    def shutdown(self) -> None:
        self.stop_run()
