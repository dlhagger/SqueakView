"""Ordered, fail-closed run shutdown and finalization coordination.

This module deliberately has no Qt dependency.  The operator manager supplies
the small set of side-effect callbacks needed by the coordinator, while the
request and result remain immutable records that can be exercised directly in
tests.  The order here is a scientific-integrity invariant: stop triggers,
drain capture ledgers, close capture, close serial, then validate artifacts.
"""

from __future__ import annotations

import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from squeakview.apps.operator.backend.events import RunPhase
from squeakview.common import run_context
from squeakview.common.failure_injection import FailurePlan


class CaptureHandle(Protocol):
    def is_running(self) -> bool: ...

    def terminate_group_graceful(
        self, sig: signal.Signals, timeout_s: float, kill: bool
    ) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


class SerialHandle(Protocol):
    stop_ack_count: int | None
    fatal_error: str | None

    def log_marker(self, marker: str) -> None: ...

    def send_line(self, text: str) -> None: ...

    def wait_for_stop_ack(self, timeout_s: float = 2.0) -> bool: ...

    def wait_for_camera_stop(self, timeout_s: float = 3.0) -> bool: ...

    def disarm_watchdog_v1(self, *, timeout_s: float = 2.0) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class FinalizationRequest:
    """Immutable snapshot of the state needed for one finalization attempt."""

    final_state: str
    error: str | None
    terminate_capture: bool
    known_capture_returncode: int | None
    run_dir: Path | None
    capture: CaptureHandle | None
    serial: SerialHandle | None
    capture_running: bool
    controller_started: bool
    trigger_on: bool
    phase: RunPhase
    controller_protocol: str = "v2"
    alignment_required: bool = False
    failure_plan: FailurePlan | None = None


@dataclass(frozen=True, slots=True)
class FinalizationHooks:
    """Side-effect boundary supplied by the application orchestrator."""

    log: Callable[[str], None]
    transition: Callable[..., None]
    wait_for_capture_drain: Callable[..., bool]
    run_capture_finalizer: Callable[[Path], int]
    write_run_manifest: Callable[[Path], bool]
    run_output_snapshot: Callable[[Path], Mapping[str, Any]]
    sleep: Callable[[float], None] = time.sleep


@dataclass(frozen=True, slots=True)
class FinalizationResult:
    """Immutable outcome returned to the owning manager."""

    final_state: str
    error: str | None
    capture_returncode: int | None
    serial_closed: bool

    @property
    def failed(self) -> bool:
        return self.final_state == "failed"


def _injected(request: FinalizationRequest, kind: str) -> bool:
    plan = request.failure_plan
    return bool(plan is not None and plan.target == "shutdown" and plan.kind == kind)


def finalize_run(
    request: FinalizationRequest,
    hooks: FinalizationHooks,
) -> FinalizationResult:
    """Perform ordered run shutdown and fail closed on incomplete evidence."""

    final_state = request.final_state
    error = request.error
    run_dir = request.run_dir
    capture = request.capture
    serial_handle = request.serial
    capture_returncode = request.known_capture_returncode
    phase = request.phase
    capture_exit_error: str | None = None
    controller_stop_error: str | None = None
    capture_drain_error: str | None = None
    serial_integrity_error: str | None = None
    serial_closed = False

    def transition(next_phase: RunPhase, *, message: str | None = None) -> None:
        nonlocal phase
        hooks.transition(next_phase, message=message)
        phase = next_phase

    if phase in {RunPhase.STARTING, RunPhase.RECORDING}:
        transition(RunPhase.STOPPING)

    if final_state == "finalized":
        hooks.log("[BACKEND] stopping run")
        if run_dir:
            try:
                run_context.write_status(run_dir, "stopping")
            except Exception:
                pass

    stopping_capture = bool(
        request.terminate_capture and request.capture_running and capture is not None
    )
    if (stopping_capture or request.controller_started) and serial_handle:
        serial_handle.log_marker("CAPTURE_STOP_REQUESTED")

    # Scientific shutdown invariant: disable the trigger source before asking
    # DeepStream to drain/close, preventing controller pulses with no frame.
    if serial_handle:
        try:
            if (
                request.controller_protocol == "watchdog_v1_experimental"
                and request.controller_started
            ):
                serial_handle.log_marker("DISARM_SENT")
                if _injected(request, "stop_ack_timeout"):
                    raise RuntimeError(
                        "qualification-injected DISARM acknowledgement timeout"
                    )
                serial_handle.disarm_watchdog_v1(timeout_s=2.0)
                serial_handle.log_marker("WATCHDOG_V1_DISARM_ACKED")
            else:
                serial_handle.log_marker("STOP_SENT")
                serial_handle.send_line("STOP")
                if stopping_capture or request.controller_started:
                    stop_failures: list[str] = []
                    if (
                        request.controller_protocol == "v2"
                        and request.controller_started
                        and not serial_handle.wait_for_camera_stop(timeout_s=3.0)
                    ):
                        stop_failures.append("CAMERA_STOP was not received")
                    stop_acked = not _injected(
                        request, "stop_ack_timeout"
                    ) and serial_handle.wait_for_stop_ack(timeout_s=2.0)
                    serial_handle.log_marker(
                        "CAPTURE_STOP_ACKED" if stop_acked else "CAPTURE_STOP_ACK_TIMEOUT"
                    )
                    if not stop_acked:
                        stop_failures.append("controller STOP was not acknowledged")
                    if stop_failures:
                        controller_stop_error = "; ".join(stop_failures)
        except Exception as exc:
            action = (
                "DISARM"
                if request.controller_protocol == "watchdog_v1_experimental"
                else "STOP"
            )
            controller_stop_error = f"controller {action} failed: {exc}"
            hooks.log(f"[BACKEND] {controller_stop_error}")

    if (
        run_dir is not None
        and serial_handle is not None
        and request.trigger_on
        and request.controller_started
    ):
        drained = hooks.wait_for_capture_drain(
            Path(run_dir),
            expected_ttl_count=getattr(serial_handle, "stop_ack_count", None),
        )
        serial_handle.log_marker(
            "CAPTURE_SOURCE_DRAINED" if drained else "CAPTURE_SOURCE_DRAIN_TIMEOUT"
        )
        if not drained:
            capture_drain_error = (
                "capture ledgers did not reach the controller TTL count and become quiet"
            )

    if stopping_capture and capture is not None:
        # Capture owns EOS/MP4 closure.  Validation can only begin after its
        # concrete process exit has been observed.
        capture.terminate_group_graceful(signal.SIGINT, 30.0, True)
        try:
            if _injected(request, "capture_exit_unconfirmed"):
                raise subprocess.TimeoutExpired("capture", 2.0)
            capture_returncode = capture.wait(timeout=2)
        except subprocess.TimeoutExpired:
            capture_exit_error = (
                "capture process did not exit after shutdown escalation; "
                "recording files were not validated"
            )
            hooks.log(f"[BACKEND] {capture_exit_error}")
        if serial_handle and capture_returncode is not None:
            serial_handle.log_marker("CAPTURE_STOP_DONE")
    elif capture is not None and not request.capture_running:
        try:
            capture_returncode = capture.wait(timeout=0)
        except subprocess.TimeoutExpired:
            capture_exit_error = (
                "capture process exit could not be confirmed; "
                "recording files were not validated"
            )

    if serial_handle:
        try:
            hooks.sleep(0.5)
        except Exception:
            pass
        serial_handle.close()
        serial_closed = True
        fatal_error = getattr(serial_handle, "fatal_error", None)
        if fatal_error:
            serial_integrity_error = (
                "serial acquisition integrity failed during shutdown: "
                f"{fatal_error}"
            )

    if phase in {RunPhase.STOPPING, RunPhase.DRAINING}:
        transition(RunPhase.CAPTURE_CLOSED)

    failure_reasons: list[str] = []
    if controller_stop_error:
        failure_reasons.append(controller_stop_error)
    if capture_drain_error:
        failure_reasons.append(capture_drain_error)
    if capture_exit_error:
        failure_reasons.append(capture_exit_error)
    if serial_integrity_error:
        failure_reasons.append(serial_integrity_error)
    if final_state == "finalized":
        if capture_returncode is None:
            if capture is not None and not capture_exit_error:
                failure_reasons.append(
                    "capture process exit was not confirmed; recording files were not validated"
                )
        elif capture_returncode != 0:
            failure_reasons.append(f"inference exit code {capture_returncode}")

    if run_dir and capture_returncode is not None:
        if phase == RunPhase.CAPTURE_CLOSED:
            transition(RunPhase.VALIDATING)
        finalizer_returncode = hooks.run_capture_finalizer(Path(run_dir))
        if finalizer_returncode == 124:
            failure_reasons.append("post-run finalizer timed out")
        elif finalizer_returncode not in (0,):
            failure_reasons.append(f"post-run finalizer exit code {finalizer_returncode}")
        if final_state == "finalized":
            status = run_context.read_json_required(Path(run_dir) / "run_status.json")
            validation = status.get("recording_validation")
            if not isinstance(validation, dict):
                failure_reasons.append("recording validation was not produced")
            elif validation.get("passed") is not True:
                failure_reasons.append("recording frame-count validation failed")
            integrity = status.get("acquisition_integrity")
            if not isinstance(integrity, dict):
                failure_reasons.append("acquisition integrity audit was not produced")
            elif integrity.get("passed") is not True:
                failure_reasons.append("camera acquisition integrity validation failed")
            if (
                request.alignment_required
                and status.get("alignment_validated") is not True
            ):
                failure_reasons.append(
                    "controller/camera alignment validation was not produced or failed"
                )

    if failure_reasons:
        final_state = "failed"
        error = "; ".join(([error] if error else []) + failure_reasons)
        hooks.log(f"[BACKEND] run failed validation: {error}")

    finalizer_status = (
        run_context.read_json_required(Path(run_dir) / "run_status.json")
        if run_dir
        else {}
    )
    analysis_summary = (
        run_context.read_json_required(Path(run_dir) / "alignment_summary.json")
        if run_dir and (Path(run_dir) / "alignment_summary.json").exists()
        else None
    )
    if run_dir:
        try:
            hooks.write_run_manifest(run_dir)
            updates: dict[str, Any] = {
                "analysis_complete": bool(
                    analysis_summary is not None
                    and finalizer_status.get("alignment_validated") is True
                ),
                "outputs": dict(hooks.run_output_snapshot(run_dir)),
            }
            if error:
                updates["error"] = error
            run_context.write_status(run_dir, final_state, **updates)
            hooks.write_run_manifest(run_dir)
        except Exception as exc:
            persistence_error = f"final run metadata could not be persisted: {exc}"
            final_state = "failed"
            error = "; ".join(
                item for item in (error, persistence_error) if item
            )
            hooks.log(f"[SAVE] finalize status failed: {exc}")
            # A failure writing the desired terminal state must never leave
            # the in-memory lifecycle looking successfully finalized.  Make
            # one best-effort attempt to persist the fail-closed outcome.
            try:
                run_context.write_status(run_dir, "failed", error=error)
                hooks.write_run_manifest(run_dir)
            except Exception as recovery_exc:
                hooks.log(
                    "[SAVE] failed to persist fail-closed finalization state: "
                    f"{recovery_exc}"
                )

    return FinalizationResult(
        final_state=final_state,
        error=error,
        capture_returncode=capture_returncode,
        serial_closed=serial_closed,
    )


__all__ = [
    "FinalizationHooks",
    "FinalizationRequest",
    "FinalizationResult",
    "finalize_run",
]
