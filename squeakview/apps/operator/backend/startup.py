from __future__ import annotations

"""Qt-free, fail-closed orchestration for starting a scientific run.

The service owns startup policy and ordering.  Runtime ownership remains with
``OperatorBackend`` through the injected hooks, so child callbacks and GUI
notifications do not cross this boundary.
"""

import copy
import math
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol

from squeakview.apps.operator.backend.contracts import RunRequest
from squeakview.common.failure_injection import FailurePlan
from squeakview.common.bounded_input import read_json_object, read_yaml_mapping


MAX_TASK_CONFIG_BYTES = 1024 * 1024


class SerialHandle(Protocol):
    last_error: object
    fatal_error: object

    def open(self, run_dir: Path) -> bool: ...
    def send_line(self, line: str) -> None: ...
    def log_marker(self, marker: str) -> None: ...
    def wait_for_ttl(self, *, timeout_s: float) -> bool: ...
    def negotiate_watchdog_v1(
        self, *, requested_lease_ms: int, timeout_s: float
    ) -> object: ...

    def negotiate_protocol_v2(self, *, timeout_s: float = 3.0) -> None: ...

    def exchange_time_sync(
        self, sequence: int, jetson_send_ns: int, *, timeout_s: float
    ) -> tuple[str, int]: ...

    def exchange_set_rtc(self, unix_seconds: int, *, timeout_s: float) -> str: ...


class CaptureHandle(Protocol):
    def is_running(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class ModelSelection:
    name: str
    snapshot: Mapping[str, Any]
    engine_build_identity: Mapping[str, object] | None
    batch_size: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "snapshot",
            MappingProxyType(copy.deepcopy(dict(self.snapshot))),
        )
        if self.engine_build_identity is not None:
            object.__setattr__(
                self,
                "engine_build_identity",
                MappingProxyType(copy.deepcopy(dict(self.engine_build_identity))),
            )


@dataclass(frozen=True, slots=True)
class PreparedRun:
    config: RunRequest
    run_dir: Path
    started_at: str
    storage: Mapping[str, Any]
    model: ModelSelection | None
    failure_plan: FailurePlan | None
    device_context: Mapping[str, object]
    qualification_case: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "storage",
            MappingProxyType(copy.deepcopy(dict(self.storage))),
        )
        object.__setattr__(
            self,
            "device_context",
            MappingProxyType(copy.deepcopy(dict(self.device_context))),
        )
        if self.qualification_case is not None:
            object.__setattr__(
                self,
                "qualification_case",
                MappingProxyType(copy.deepcopy(dict(self.qualification_case))),
            )


@dataclass(frozen=True, slots=True)
class StartupRequest:
    config: RunRequest
    already_running: bool


@dataclass(frozen=True, slots=True)
class StartupResult:
    started: bool
    prepared: PreparedRun | None = None
    serial: SerialHandle | None = None
    capture: CaptureHandle | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class StartupHooks:
    log: Callable[[str], None]
    resolve_task_path: Callable[[Path], Path]
    resolve_model_path: Callable[[Path], Path]
    resolve_failure_plan_path: Callable[[Path], Path]
    load_failure_plan: Callable[[Path], FailurePlan]
    validate_model: Callable[[Path], ModelSelection]
    assert_storage_ready: Callable[[], Mapping[str, Any]]
    acquire_lock: Callable[[], bool]
    release_lock: Callable[[], None]
    now_iso: Callable[[], str]
    device_context: Callable[[], Mapping[str, object]]
    set_fan_max: Callable[[], None]
    create_run_dir: Callable[..., tuple[Path, str]]
    preview_socket_paths: Callable[[Path, int], tuple[Path, ...]]
    initialize_runtime: Callable[
        [
            RunRequest,
            str,
            Mapping[str, Any],
            ModelSelection | None,
            FailurePlan | None,
            Mapping[str, object],
            Mapping[str, Any] | None,
        ],
        None,
    ]
    establish_run: Callable[[PreparedRun], None]
    persist_prestart: Callable[[PreparedRun], None]
    serial_available: Callable[[], bool]
    create_serial: Callable[[RunRequest, FailurePlan | None], SerialHandle]
    set_serial: Callable[[SerialHandle | None], None]
    arm_serial_runtime: Callable[[], None]
    validate_clock: Callable[[SerialHandle, PreparedRun], Mapping[str, Any]]
    spawn_capture: Callable[[RunRequest], CaptureHandle]
    set_capture: Callable[[CaptureHandle | None], None]
    after_capture_spawn: Callable[[Path], None]
    run_finalized: Callable[[], bool]
    finalize_failure: Callable[[str, bool], None]
    wait_ready: Callable[[float], bool]
    stop_requested: Callable[[], bool]
    start_controller: Callable[[SerialHandle, int], bool]
    start_run_dir_watch: Callable[[], None]
    ready_timeout: Callable[[], float]
    serial_open_failure_message: Callable[[SerialHandle, str, int], str]
    resolve_qualification_case: Callable[
        [RunRequest, Mapping[str, object]], Mapping[str, Any] | None
    ] = lambda _config, _device: None


def _reject(hooks: StartupHooks, message: str, *, prefix: str) -> StartupResult:
    hooks.log(f"[{prefix}] {message}")
    return StartupResult(started=False, error=message)


def validate_startup_policy(config: RunRequest) -> None:
    """Reject scientifically inconsistent requests before creating a run.

    This is the authoritative backend policy boundary.  GUI validators remain
    useful presentation, while the capture subprocess repeats graph-specific
    checks as defence in depth.
    """

    bool_fields = (
        "trigger_on",
        "inference_enabled",
        "serial_enabled",
        "allow_rtc_correction",
        "preview_enabled",
    )
    for name in bool_fields:
        if not isinstance(getattr(config, name), bool):
            raise ValueError(f"{name} must be boolean")

    for name in (
        "width",
        "height",
        "fps",
        "bitrate",
        "num_cameras",
        "serial_baud",
        "controller_watchdog_lease_ms",
        "arduino_fps",
    ):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    if not isinstance(config.capture_backend, str):
        raise ValueError("capture_backend must be a string")
    backend = config.capture_backend.strip().lower()
    if backend != "flir_direct":
        raise ValueError(
            "capture_backend must be 'flir_direct' "
            f"(got {backend!r})"
        )
    if not isinstance(config.pixel_format, str) or not config.pixel_format.strip():
        raise ValueError("pixel_format must be a non-empty string")
    if not isinstance(config.trigger_activation, str) or config.trigger_activation not in {
        "rising",
        "falling",
    }:
        raise ValueError("trigger_activation must be 'rising' or 'falling'")
    if config.exposure_us is not None and (
        isinstance(config.exposure_us, bool)
        or not isinstance(config.exposure_us, (int, float))
        or not math.isfinite(float(config.exposure_us))
        or float(config.exposure_us) <= 0
    ):
        raise ValueError("exposure_us must be finite and positive or null")

    serials = config.camera_serials
    if not isinstance(serials, tuple) or any(
        not isinstance(serial, str) or not serial.strip() for serial in serials
    ):
        raise ValueError("camera_serials must contain non-empty strings")
    if serials and len(serials) != config.num_cameras:
        raise ValueError("camera serial count must match camera count")
    if len(set(serials)) != len(serials):
        raise ValueError("camera serials must be unique")

    if config.serial_enabled and (
        not isinstance(config.serial_port, str) or not config.serial_port.strip()
    ):
        raise ValueError(
            "serial_port must be non-empty when serial acquisition is enabled"
        )
    if config.controller_protocol not in {"legacy", "watchdog_v1_experimental", "v2"}:
        raise ValueError(
            "controller_protocol must be 'legacy', 'watchdog_v1_experimental', or 'v2'"
        )
    if config.controller_protocol == "watchdog_v1_experimental" and not (
        config.serial_enabled and config.trigger_on
    ):
        raise ValueError(
            "experimental watchdog protocol requires triggered serial acquisition"
        )
    if config.trigger_on and not config.serial_enabled:
        raise ValueError("triggered capture requires serial controller acquisition")
    if config.trigger_on and config.arduino_fps != config.fps:
        raise ValueError(
            "triggered capture requires Arduino FPS to match camera FPS "
            f"({config.arduino_fps} != {config.fps})"
        )


def start_run(request: StartupRequest, hooks: StartupHooks) -> StartupResult:
    """Start one run while preserving the scientific startup safety order."""

    cfg = request.config
    if request.already_running:
        return _reject(hooks, "run already active", prefix="BACKEND")
    try:
        validate_startup_policy(cfg)
    except ValueError as exc:
        return _reject(hooks, f"invalid run configuration: {exc}", prefix="BACKEND")
    if not cfg.task_cfg:
        return _reject(hooks, "task config required; aborting run", prefix="BACKEND")

    try:
        cfg = replace(
            cfg,
            task_cfg=hooks.resolve_task_path(cfg.task_cfg),
            ds_cfg=(hooks.resolve_model_path(cfg.ds_cfg) if cfg.ds_cfg else None),
            failure_plan=(
                hooks.resolve_failure_plan_path(cfg.failure_plan)
                if cfg.failure_plan
                else None
            ),
        )
    except (OSError, ValueError) as exc:
        return _reject(
            hooks,
            f"scientific input path is outside the active project: {exc}",
            prefix="BACKEND",
        )
    if cfg.task_cfg is None or not Path(cfg.task_cfg).exists():
        return _reject(hooks, f"task config missing: {cfg.task_cfg}", prefix="BACKEND")
    try:
        if Path(cfg.task_cfg).suffix.lower() == ".json":
            read_json_object(
                Path(cfg.task_cfg),
                max_bytes=MAX_TASK_CONFIG_BYTES,
                label="task config",
            )
        else:
            read_yaml_mapping(
                Path(cfg.task_cfg),
                max_bytes=MAX_TASK_CONFIG_BYTES,
                label="task config",
            )
    except (OSError, ValueError) as exc:
        return _reject(hooks, f"invalid task config: {exc}", prefix="BACKEND")

    failure_plan: FailurePlan | None = None
    if cfg.failure_plan is not None:
        try:
            failure_plan = hooks.load_failure_plan(cfg.failure_plan)
        except ValueError as exc:
            return _reject(hooks, f"invalid failure plan: {exc}", prefix="QUALIFICATION")
        if failure_plan.target == "serial_controller" and not cfg.serial_enabled:
            return _reject(
                hooks,
                "serial-controller failure plan requires serial acquisition to be enabled",
                prefix="QUALIFICATION",
            )

    model: ModelSelection | None = None
    if cfg.inference_enabled:
        if cfg.ds_cfg is None:
            return _reject(
                hooks,
                "inference is enabled but no model package was selected",
                prefix="MODEL",
            )
        try:
            model = hooks.validate_model(cfg.ds_cfg)
        except Exception as exc:
            # The injected validator is required to convert only model-package
            # validation failures to ValueError; unexpected faults still fail
            # closed at this boundary without creating a run.
            return _reject(
                hooks,
                f"selected model package is invalid: {exc}",
                prefix="MODEL",
            )
        if model.engine_build_identity is None:
            return _reject(
                hooks,
                "selected model package has no verified TensorRT engine build identity; "
                "rebuild it on this device from the Project Setup screen to create "
                "a schema-3 model manifest before starting inference",
                prefix="MODEL",
            )
        if model.batch_size is None:
            return _reject(
                hooks,
                "selected model package did not provide a validated inference batch size",
                prefix="MODEL",
            )
        if model.batch_size != cfg.num_cameras:
            return _reject(
                hooks,
                f"selected model batch size ({model.batch_size}) does not match camera "
                f"count ({cfg.num_cameras})",
                prefix="MODEL",
            )
        hooks.log(f"[MODEL] selected package: {model.name}")

    try:
        storage = hooks.assert_storage_ready()
    except Exception as exc:
        return _reject(hooks, f"local run storage unavailable: {exc}", prefix="SAVE")

    if not hooks.acquire_lock():
        return _reject(
            hooks,
            "another SqueakView process already owns the scientific acquisition lock",
            prefix="BACKEND",
        )

    try:
        started_at = hooks.now_iso()
        device = hooks.device_context()
        qualification_case = hooks.resolve_qualification_case(cfg, device)
        hooks.initialize_runtime(
            cfg,
            started_at,
            storage,
            model,
            failure_plan,
            device,
            qualification_case,
        )
        hooks.set_fan_max()
    except Exception as exc:
        error = f"startup initialization failed: {type(exc).__name__}: {exc}"
        hooks.log(f"[BACKEND] {error}")
        hooks.release_lock()
        return StartupResult(started=False, error=error)
    if hooks.stop_requested():
        error = "startup cancelled before run creation"
        hooks.log(f"[BACKEND] {error}")
        hooks.release_lock()
        return StartupResult(started=False, error=error)

    mouse_id = (cfg.mouse_id or "").strip()
    experiment_name = (cfg.experiment_name or "").strip()
    try:
        run_dir, _run_id = hooks.create_run_dir(
            experiment_name=experiment_name,
            mouse_id=mouse_id,
            prefix=mouse_id or "ds",
        )
    except Exception as exc:
        error = f"failed to create run directory: {exc}"
        hooks.log(f"[SAVE] {error}")
        hooks.release_lock()
        return StartupResult(started=False, error=error)

    cfg = replace(
        cfg,
        run_dir=run_dir,
        preview_socket_paths=(
            hooks.preview_socket_paths(run_dir, cfg.num_cameras)
            if cfg.preview_enabled
            else ()
        ),
    )
    prepared = PreparedRun(
        config=cfg,
        run_dir=run_dir,
        started_at=started_at,
        storage=storage,
        model=model,
        failure_plan=failure_plan,
        device_context=device,
        qualification_case=qualification_case,
    )
    try:
        hooks.establish_run(prepared)
    except Exception as exc:
        error = f"failed to establish run lifecycle: {type(exc).__name__}: {exc}"
        hooks.log(f"[BACKEND] {error}")
        hooks.release_lock()
        return StartupResult(started=False, prepared=prepared, error=error)
    if hooks.stop_requested():
        error = "startup cancelled before pre-start persistence"
        hooks.log(f"[BACKEND] {error}")
        hooks.finalize_failure(error, False)
        return StartupResult(started=False, prepared=prepared, error=error)
    try:
        hooks.persist_prestart(prepared)
    except Exception as exc:
        error = f"required pre-start run metadata persistence failed: {exc}"
        hooks.log(f"[SAVE] {error}")
        hooks.finalize_failure(error, False)
        return StartupResult(started=False, prepared=prepared, error=error)
    if hooks.stop_requested():
        error = "startup cancelled before controller initialization"
        hooks.log(f"[BACKEND] {error}")
        hooks.finalize_failure(error, False)
        return StartupResult(started=False, prepared=prepared, error=error)

    serial_handle: SerialHandle | None = None
    if cfg.serial_enabled:
        if cfg.controller_protocol != "v2":
            hooks.log(
                "[SER] WARNING: legacy controller transport is a non-production "
                "bench override; protocol v2 is required for scientific runs"
            )
        if not hooks.serial_available():
            error = (
                "Serial controller support was requested, but pyserial is not installed. "
                "Install the project dependencies before starting a scientific run."
            )
            hooks.log(f"[SER] {error}")
            hooks.finalize_failure(error, False)
            return StartupResult(started=False, prepared=prepared, error=error)
        try:
            serial_handle = hooks.create_serial(cfg, failure_plan)
        except Exception as exc:
            error = f"failed to initialize serial controller: {type(exc).__name__}: {exc}"
            hooks.log(f"[SER] {error}")
            hooks.finalize_failure(error, False)
            return StartupResult(started=False, prepared=prepared, error=error)
        hooks.set_serial(serial_handle)
        try:
            serial_opened = serial_handle.open(run_dir)
        except Exception as exc:
            error = f"serial controller open failed: {type(exc).__name__}: {exc}"
            hooks.log(f"[SER] {error}")
            hooks.finalize_failure(error, False)
            return StartupResult(
                started=False, prepared=prepared, serial=serial_handle, error=error
            )
        if not serial_opened:
            hooks.set_serial(None)
            error = hooks.serial_open_failure_message(
                serial_handle, cfg.serial_port, cfg.serial_baud
            )
            hooks.log(f"[SER] {error.replace(chr(10), ' ')}")
            hooks.finalize_failure(error, False)
            return StartupResult(started=False, prepared=prepared, error=error)
        hooks.arm_serial_runtime()
        if serial_handle.fatal_error is not None:
            error = f"serial acquisition integrity failed: {serial_handle.fatal_error}"
            hooks.log(f"[SER] {error}")
            hooks.finalize_failure(error, True)
            return StartupResult(started=False, prepared=prepared, serial=serial_handle, error=error)
        if cfg.controller_protocol == "v2":
            try:
                # PROTO,2 is idempotent. Negotiating first supports both a
                # freshly booted v1 controller and a controller that remained
                # in v2 after an earlier run. TIME_SYNC/SET_RTC are accepted as
                # framed command results while v2 is idle.
                serial_handle.negotiate_protocol_v2(timeout_s=3.0)
                serial_handle.log_marker("PROTOCOL_V2_NEGOTIATED")
                hooks.log("[SER] controller protocol v2 active")
            except Exception as exc:
                error = f"controller protocol-v2 negotiation failed: {exc}"
                hooks.log(f"[SER] {error}")
                hooks.finalize_failure(error, False)
                return StartupResult(
                    started=False, prepared=prepared, serial=serial_handle, error=error
                )
        try:
            clock_record = hooks.validate_clock(serial_handle, prepared)
        except Exception as exc:
            error = f"controller clock preflight failed: {type(exc).__name__}: {exc}"
            hooks.log(f"[CLOCK] {error}")
            hooks.finalize_failure(error, False)
            return StartupResult(
                started=False, prepared=prepared, serial=serial_handle, error=error
            )
        if clock_record.get("result") != "PASS":
            reason = str(clock_record.get("reason") or "VALIDATION_ERROR")
            detail = str(clock_record.get("detail") or "").strip()
            error = f"controller clock preflight failed: {reason}"
            guidance = {
                "JETSON_NTP_NOT_SYNCHRONIZED": (
                    "Synchronize the Jetson clock with NTP before retrying; RTC correction "
                    "is unsafe while the host clock is unsynchronized."
                ),
                "CONTROLLER_RTC_INVALID": (
                    "The controller RTC is unset or invalid. Inspect the PCF8523 and enable "
                    "explicit RTC correction authorization in Configure before retrying."
                ),
                "CLOCK_OFFSET_OUT_OF_TOLERANCE": (
                    "The controller RTC exceeds the ±1.5-second limit. Enable explicit RTC "
                    "correction authorization in Configure before retrying."
                ),
                "DEVICE_BUSY": (
                    "Wait until the current controller session or feed has stopped, then retry."
                ),
                "TIME_SYNC_TIMEOUT": (
                    "The controller did not answer TIME_SYNC; check the USB connection and retry."
                ),
                "SET_RTC_TIMEOUT": (
                    "The controller did not acknowledge SET_RTC; its clock was not accepted."
                ),
            }.get(reason)
            if detail:
                error += f" — {detail}"
            if guidance:
                error += f" {guidance}"
            hooks.log(f"[CLOCK] {error}")
            hooks.finalize_failure(error, False)
            return StartupResult(
                started=False, prepared=prepared, serial=serial_handle, error=error
            )
        if cfg.controller_protocol == "watchdog_v1_experimental":
            try:
                serial_handle.negotiate_watchdog_v1(
                    requested_lease_ms=cfg.controller_watchdog_lease_ms,
                    timeout_s=2.0,
                )
                serial_handle.log_marker("WATCHDOG_V1_NEGOTIATED")
            except Exception as exc:
                error = f"controller watchdog negotiation failed before capture launch: {exc}"
                hooks.log(f"[SER] {error}")
                hooks.finalize_failure(error, False)
                return StartupResult(
                    started=False, prepared=prepared, serial=serial_handle, error=error
                )

    if hooks.stop_requested():
        error = "startup cancelled before capture launch"
        hooks.log(f"[BACKEND] {error}")
        hooks.finalize_failure(error, False)
        return StartupResult(
            started=False, prepared=prepared, serial=serial_handle, error=error
        )

    capture_backend = (cfg.capture_backend or "flir_direct").lower().strip()
    if capture_backend != "flir_direct":
        error = f"unsupported capture backend: {capture_backend}"
        hooks.log(f"[CAP] {error}")
        hooks.finalize_failure(error, False)
        return StartupResult(started=False, prepared=prepared, serial=serial_handle, error=error)

    hooks.log("[CAP] FLIR direct capture will be sourced inside DeepStream (flirspinsrc)")
    try:
        capture = hooks.spawn_capture(cfg)
        hooks.set_capture(capture)
    except Exception as exc:
        error = f"failed to launch inference process: {exc}"
        hooks.log(f"[DS] {error}")
        hooks.finalize_failure(error, False)
        return StartupResult(started=False, prepared=prepared, serial=serial_handle, error=error)
    try:
        hooks.after_capture_spawn(run_dir)
    except Exception as exc:
        error = f"post-spawn qualification barrier failed: {exc}"
        hooks.log(f"[DS] {error}")
        hooks.finalize_failure(error, True)
        return StartupResult(
            started=False,
            prepared=prepared,
            serial=serial_handle,
            capture=capture,
            error=error,
        )
    if hooks.run_finalized():
        hooks.set_capture(None)
        return StartupResult(started=False, prepared=prepared, serial=serial_handle, capture=capture)
    if hooks.stop_requested():
        error = "startup cancelled after capture launch"
        hooks.log(f"[BACKEND] {error}")
        hooks.finalize_failure(error, True)
        return StartupResult(
            False, prepared, serial_handle, capture, error
        )
    hooks.log("[DS] inference launched")

    if serial_handle is not None and cfg.trigger_on:
        hooks.log("[BACKEND] waiting for inference ready before START")
        try:
            timeout = hooks.ready_timeout()
            ready = hooks.wait_ready(timeout)
        except Exception as exc:
            error = f"inference readiness check failed: {type(exc).__name__}: {exc}"
            hooks.log(f"[BACKEND] {error}")
            hooks.finalize_failure(error, True)
            return StartupResult(False, prepared, serial_handle, capture, error)
        if not ready:
            error = f"inference was not ready within {timeout:.1f}s; controller was not started"
            hooks.log(f"[BACKEND] {error}")
            hooks.finalize_failure(error, True)
            return StartupResult(False, prepared, serial_handle, capture, error)
        if hooks.stop_requested():
            return StartupResult(False, prepared, serial_handle, capture)
        if hooks.run_finalized() or not capture.is_running():
            error = "inference exited before the controller could be started"
            hooks.log(f"[BACKEND] {error}")
            hooks.finalize_failure(error, False)
            return StartupResult(False, prepared, serial_handle, capture, error)
        try:
            hooks.log("[BACKEND] inference ready; sending START")
            if not hooks.start_controller(serial_handle, int(cfg.arduino_fps)):
                error = "startup cancelled before controller START"
                hooks.log(f"[BACKEND] {error}")
                hooks.finalize_failure(error, True)
                return StartupResult(False, prepared, serial_handle, capture, error)
            if not serial_handle.wait_for_ttl(timeout_s=3.0):
                serial_handle.log_marker("START_TTL_TIMEOUT")
                error = (
                    "controller START was sent, but no camera TTL was detected within "
                    "3.0s; the run was aborted"
                )
                hooks.log(f"[BACKEND] {error}")
                hooks.finalize_failure(error, True)
                return StartupResult(False, prepared, serial_handle, capture, error)
        except Exception as exc:
            error = f"failed to start controller: {exc}"
            hooks.log(f"[BACKEND] {error}")
            hooks.finalize_failure(error, True)
            return StartupResult(False, prepared, serial_handle, capture, error)

    if hooks.stop_requested():
        error = "startup cancelled before run-directory supervision"
        hooks.log(f"[BACKEND] {error}")
        hooks.finalize_failure(error, True)
        return StartupResult(False, prepared, serial_handle, capture, error)
    try:
        hooks.start_run_dir_watch()
    except Exception as exc:
        error = f"run-directory watcher failed: {type(exc).__name__}: {exc}"
        hooks.log(f"[BACKEND] {error}")
        hooks.finalize_failure(error, True)
        return StartupResult(False, prepared, serial_handle, capture, error)
    return StartupResult(True, prepared, serial_handle, capture)


__all__ = [
    "ModelSelection",
    "PreparedRun",
    "StartupHooks",
    "StartupRequest",
    "StartupResult",
    "validate_startup_policy",
    "start_run",
]
