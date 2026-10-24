from __future__ import annotations

"""Qt-free owner for one GUI lease and one :class:`OperatorBackend`."""

import dataclasses
import enum
import json
import math
import os
import queue
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from dataclasses import dataclass

from squeakview.apps.operator.backend.contracts import RunRequest
from squeakview.apps.operator.backend.events import BackendEvent, RunPhase
from squeakview.apps.operator.backend import manifest
from squeakview.apps.operator.backend.manager import OperatorBackend
from squeakview.common.dashboard import DashboardEvent
from squeakview.project import PROJECT_ENV, RuntimeContext

from .protocol import (
    CommandEnvelope,
    EventEnvelope,
    HEARTBEAT_COMMAND_NAME,
    HEARTBEAT_REQUEST_ID,
    ProtocolError,
    SocketEnvelopeReader,
    SocketEnvelopeWriter,
)


SOCKET_ENV = "SQUEAKVIEW_SUPERVISOR_SOCKET"
MAX_PENDING_COMMANDS = 32
MAX_PENDING_CRITICAL_EVENTS = 256
MAX_PENDING_PRESENTATION_EVENTS = 512
MAX_MUTATION_REQUESTS = 1024
MAX_PRESENTATION_LOG_CHARS = 8192
DEFAULT_GUI_CONNECT_TIMEOUT_S = 30.0
DEFAULT_GUI_LEASE_TIMEOUT_S = 30.0
_PATH_FIELDS = {"ds_cfg", "task_cfg", "failure_plan", "run_dir"}
_PATH_TUPLE_FIELDS = {"preview_socket_paths"}
_STRING_TUPLE_FIELDS = {"camera_serials"}
_MUTATING_COMMANDS = {
    "start_run",
    "stop_run",
    "clear_feeder_jam",
    "save_bottle_measurements",
    "begin_bottle_swap",
    "resume_bottle_swap",
    "update_bottle_history_weight",
    "shutdown_server",
}


@dataclass(slots=True)
class _Outbound:
    name: str
    payload: Mapping[str, object]
    request_id: str | None = None
    sent: threading.Event | None = None


def _json_value(value: object) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _request_from_payload(payload: Mapping[str, object]) -> RunRequest:
    allowed = {field.name for field in dataclasses.fields(RunRequest)}
    unknown = sorted(set(payload) - allowed)
    missing = sorted(allowed - set(payload))
    if unknown or missing:
        problems = []
        if unknown:
            problems.append(f"unknown fields: {', '.join(unknown)}")
        if missing:
            problems.append(f"missing fields: {', '.join(missing)}")
        raise ValueError("run configuration must contain exact fields; " + "; ".join(problems))
    bool_fields = {
        "trigger_on",
        "inference_enabled",
        "serial_enabled",
        "allow_rtc_correction",
        "preview_enabled",
    }
    for name in bool_fields:
        if not isinstance(payload[name], bool):
            raise ValueError(f"run configuration {name} must be boolean")
    positive_int_fields = {
        "width",
        "height",
        "fps",
        "bitrate",
        "serial_baud",
        "controller_watchdog_lease_ms",
        "arduino_fps",
        "num_cameras",
    }
    nullable_positive = {"width", "height", "fps"}
    for name in positive_int_fields:
        value = payload[name]
        if value is None and name in nullable_positive:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"run configuration {name} must be a positive integer")
    window_id = payload["preview_window_id"]
    if window_id is not None and (
        isinstance(window_id, bool) or not isinstance(window_id, int) or window_id < 0
    ):
        raise ValueError("run configuration preview_window_id must be a nonnegative integer or null")
    exposure = payload["exposure_us"]
    if exposure is not None and (
        isinstance(exposure, bool)
        or not isinstance(exposure, (int, float))
        or not math.isfinite(float(exposure))
        or float(exposure) <= 0
    ):
        raise ValueError("run configuration exposure_us must be finite and positive or null")
    required_strings = {
        "capture_backend",
        "trigger_activation",
        "serial_port",
        "controller_protocol",
    }
    optional_strings = {"pixel_format", "mouse_id", "experiment_name"}
    for name in required_strings:
        if not isinstance(payload[name], str) or not payload[name].strip():
            raise ValueError(f"run configuration {name} must be a non-empty string")
    for name in optional_strings:
        if payload[name] is not None and not isinstance(payload[name], str):
            raise ValueError(f"run configuration {name} must be a string or null")
    if not isinstance(payload["bottles"], Mapping):
        raise ValueError("run configuration bottles must be an object")
    values = dict(payload)
    for name in _PATH_FIELDS:
        value = values.get(name)
        if value is not None:
            if not isinstance(value, str):
                raise ValueError(f"run configuration {name} must be a path string or null")
            values[name] = Path(value)
    for name in _PATH_TUPLE_FIELDS:
        value = values.get(name)
        if value is not None:
            if not isinstance(value, (tuple, list)) or not all(
                isinstance(item, str) for item in value
            ):
                raise ValueError(f"run configuration {name} must be a list of paths")
            values[name] = tuple(Path(item) for item in value)
    for name in _STRING_TUPLE_FIELDS:
        value = values.get(name)
        if value is not None:
            if not isinstance(value, (tuple, list)) or not all(
                isinstance(item, str) for item in value
            ):
                raise ValueError(f"run configuration {name} must be a list of strings")
            values[name] = tuple(value)
    return RunRequest(**values)


class SupervisorServer:
    """Serve one exclusive GUI client while retaining backend ownership."""

    def __init__(
        self,
        socket_path: Path,
        *,
        runtime_context: RuntimeContext,
        backend_factory: Callable[..., OperatorBackend] = OperatorBackend,
        gui_connect_timeout_s: float = DEFAULT_GUI_CONNECT_TIMEOUT_S,
        gui_lease_timeout_s: float = DEFAULT_GUI_LEASE_TIMEOUT_S,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.socket_path = Path(socket_path).resolve()
        self.runtime_context = runtime_context
        self._client: socket.socket | None = None
        self._client_lock = threading.Lock()
        self._sequence = 0
        self._stopping = threading.Event()
        self._client_lost = threading.Event()
        self._graceful_shutdown = threading.Event()
        self._lease_lock = threading.Lock()
        self._last_heartbeat: float | None = None
        self._loss_lock = threading.Lock()
        self._commands: queue.Queue[CommandEnvelope | None] = queue.Queue(
            MAX_PENDING_COMMANDS
        )
        self._critical_events: queue.Queue[_Outbound] = queue.Queue(
            MAX_PENDING_CRITICAL_EVENTS
        )
        self._presentation_events: queue.Queue[_Outbound] = queue.Queue(
            MAX_PENDING_PRESENTATION_EVENTS
        )
        self._dropped_presentation = 0
        self._dropped_lock = threading.Lock()
        self._loss_thread: threading.Thread | None = None
        self._operation_lock = threading.Lock()
        self._async_stop_lock = threading.Lock()
        self._async_stop_thread: threading.Thread | None = None
        # Once Stop has been accepted, acquisition shutdown and finalization
        # are owned by the supervisor rather than by the GUI connection.  A
        # presentation-channel failure after this point must not invalidate an
        # otherwise sound scientific run.
        self._stop_accepted = threading.Event()
        self._mutation_ledger: OrderedDict[
            str, tuple[str, Mapping[str, object], bool]
        ] = OrderedDict()
        self._socket_identity: tuple[int, int, int] | None = None
        self._presentation_loss_count = 0
        self._feeder_dashboard_event: DashboardEvent | None = None
        self._diagnostic_events: queue.Queue[dict[str, object]] = queue.Queue(64)
        self._diagnostic_lock = threading.Lock()
        self._diagnostic_thread: threading.Thread | None = None
        self.gui_connect_timeout_s = max(1.0, float(gui_connect_timeout_s))
        self.gui_lease_timeout_s = max(1.0, float(gui_lease_timeout_s))
        self._monotonic = monotonic
        self.last_error: str | None = None
        self.last_warning: str | None = None
        self.backend = backend_factory(
            self._emit_log,
            self._emit_dashboard,
            runtime_context=runtime_context,
            acquisition_owner=manifest.DURABLE_SUPERVISOR_OWNER,
        )
        self.backend.subscribe(self._emit_backend_event)

    def _next_sequence(self) -> int:
        value = self._sequence
        self._sequence += 1
        return value

    def _enqueue(
        self,
        name: str,
        payload: Mapping[str, object],
        *,
        request_id: str | None = None,
        critical: bool,
        sent: threading.Event | None = None,
    ) -> bool:
        if self._client_lost.is_set() or (
            self._presentation_loss_count and self._client is None
        ):
            # Scientific writers and state live in the backend. A disconnected
            # presentation channel must not accumulate or invalidate events.
            return False
        target = self._critical_events if critical else self._presentation_events
        try:
            target.put_nowait(_Outbound(name, payload, request_id, sent))
            return True
        except queue.Full:
            if critical:
                self._signal_client_loss("critical supervisor event queue overflow")
            else:
                with self._dropped_lock:
                    self._dropped_presentation += 1
            return False

    def _emit_log(self, message: str) -> None:
        if not isinstance(message, str):
            message = str(message)
        if len(message) > MAX_PRESENTATION_LOG_CHARS:
            message = message[:MAX_PRESENTATION_LOG_CHARS] + "… [presentation truncated]"
        self._enqueue("log", {"message": message}, critical=False)

    def _emit_dashboard(self, event: DashboardEvent) -> None:
        if not isinstance(event, DashboardEvent):
            raise TypeError("dashboard callback requires DashboardEvent")
        if event.event_uc == "FEED_JAM" or event.event_uc.startswith("ACK_CLEAR_JAM"):
            self._feeder_dashboard_event = event
        elif event.event_uc == "SYSTEM_START" or event.event_uc.startswith("SYSTEM_START,"):
            self._feeder_dashboard_event = None
        self._enqueue("dashboard", event.to_payload(), critical=False)

    def _emit_backend_event(self, event: BackendEvent) -> None:
        self._enqueue(
            "backend_event",
            {
                "schema_version": event.schema_version,
                "type": event.type,
                "phase": event.phase.value,
                "run_dir": str(event.run_dir) if event.run_dir is not None else None,
                "message": event.message,
                "payload": event.payload,
                "host_unix_ns": event.host_unix_ns,
            },
            critical=True,
        )

    def _sender(self, client: socket.socket) -> None:
        # The socket's 250 ms timeout belongs to the receive loop so heartbeat
        # expiry is noticed promptly. Outbound traffic receives the complete
        # GUI lease interval before the connection is declared lost.
        writer = SocketEnvelopeWriter(
            client,
            send_timeout_s=self.gui_lease_timeout_s,
        )
        critical_events = self._critical_events
        presentation_events = self._presentation_events
        while not self._stopping.is_set():
            with self._client_lock:
                if self._client is not client:
                    return
            item: _Outbound | None = None
            try:
                item = critical_events.get_nowait()
            except queue.Empty:
                with self._dropped_lock:
                    dropped = self._dropped_presentation
                    self._dropped_presentation = 0
                if dropped:
                    item = _Outbound(
                        "presentation_dropped", {"count": dropped}
                    )
                else:
                    try:
                        item = presentation_events.get(timeout=0.05)
                    except queue.Empty:
                        continue
            try:
                with self._client_lock:
                    if self._client is not client:
                        return
                    sequence = self._next_sequence()
                writer.send(
                    EventEnvelope(
                        name=item.name,
                        sequence=sequence,
                        request_id=item.request_id,
                        payload=_json_value(item.payload),
                    )
                )
                if item.sent is not None:
                    item.sent.set()
            except Exception as exc:
                detail = str(exc).strip() or repr(exc)
                self._signal_client_loss(
                    "supervisor IPC send failed: "
                    f"{type(exc).__name__}: {detail}", client=client,
                )
                return

    def _snapshot(self) -> dict[str, object]:
        snapshot = self.backend.snapshot()
        return {
            "schema_version": "1.0",
            "phase": snapshot.phase.value,
            "run_dir": str(snapshot.run_dir) if snapshot.run_dir is not None else None,
            "error": snapshot.error,
            "capture_running": snapshot.capture_running,
            "finalization_in_progress": snapshot.finalization_in_progress,
        }

    def _renew_gui_lease(self) -> None:
        with self._lease_lock:
            self._last_heartbeat = self._monotonic()

    def _gui_lease_expired(self) -> bool:
        with self._lease_lock:
            renewed = self._last_heartbeat
        return bool(
            renewed is not None
            and self._monotonic() - renewed > self.gui_lease_timeout_s
        )

    def _handle_heartbeat(self, envelope: CommandEnvelope) -> bool:
        """Validate and renew the receive-side GUI lease inline.

        Heartbeats intentionally never enter the command queue.  Consequently
        a long start or finalization operation cannot delay lease renewal.
        """

        if envelope.name != HEARTBEAT_COMMAND_NAME:
            return False
        if envelope.request_id != HEARTBEAT_REQUEST_ID or envelope.payload:
            raise ProtocolError("invalid operator GUI heartbeat")
        self._renew_gui_lease()
        return True

    def _run_async_stop(self) -> None:
        """Own one complete stop without occupying the IPC command worker."""

        try:
            self.backend.stop_run()
        except Exception as exc:
            error = f"asynchronous stop failed: {type(exc).__name__}: {exc}"
            self._emit_log(f"[SUPERVISOR] {error}")
            try:
                self.backend.abort_run(error)
            except Exception as abort_exc:
                self._emit_log(
                    "[SUPERVISOR] asynchronous stop recovery failed: "
                    f"{type(abort_exc).__name__}: {abort_exc}"
                )

    def _request_async_stop(self) -> bool:
        """Start finalization once and return without waiting for completion."""

        with self._async_stop_lock:
            existing = self._async_stop_thread
            if existing is not None and existing.is_alive():
                return False
            if self.backend.snapshot().phase in {RunPhase.IDLE, RunPhase.FINALIZED, RunPhase.FAILED}:
                return False
            worker = threading.Thread(
                target=self._run_async_stop,
                name="squeakview-supervisor-stop",
                daemon=False,
            )
            self._async_stop_thread = worker
            self._stop_accepted.set()
            try:
                worker.start()
            except Exception:
                self._stop_accepted.clear()
                self._async_stop_thread = None
                raise
            return True

    def _dispatch(self, command: CommandEnvelope) -> tuple[object, bool]:
        payload = command.payload
        if command.name == "ping":
            return {"protocol": 1}, False
        if command.name == "snapshot":
            return self._snapshot(), False
        if command.name == "runtime_state":
            feeder_event = self._feeder_dashboard_event
            return {"snapshot": self._snapshot(), "config": _json_value(self.backend.launch_cfg),
                    "started_at": getattr(self.backend, "_run_started_at", None),
                    "dashboard_events": [feeder_event.to_payload()] if feeder_event is not None else []}, False
        if command.name == "start_run":
            config_payload = payload.get("config", payload)
            if not isinstance(config_payload, Mapping):
                raise ValueError("start_run config must be an object")
            self._stop_accepted.clear()
            started = self.backend.start_run(_request_from_payload(config_payload))
            return {
                "started": started,
                "config": _json_value(self.backend.launch_cfg),
                "snapshot": self._snapshot(),
            }, False
        if command.name == "stop_run":
            accepted = self._request_async_stop()
            return {"accepted": accepted, "snapshot": self._snapshot()}, False
        if command.name == "clear_feeder_jam":
            return {"response": self.backend.clear_feeder_jam()}, False
        if command.name == "bottle_swap_snapshot":
            force_probe = payload.get("force_probe", False)
            if type(force_probe) is not bool:
                raise ValueError("force_probe must be boolean")
            return self.backend.bottle_swap_snapshot(force_probe=force_probe), False
        if command.name == "bottle_history_page":
            return self.backend.bottle_history_page(payload.get("offset", 0), payload.get("limit", 50)), False
        if command.name == "begin_bottle_swap":
            sides = payload.get("sides")
            if not isinstance(sides, (list, tuple)) or not sides or any(
                side not in {"left", "right"} for side in sides
            ):
                raise ValueError("select left, right, or both bottles")
            return self.backend.begin_bottle_swap(list(sides)), False
        if command.name == "resume_bottle_swap":
            replacements = payload.get("replacements")
            if not isinstance(replacements, Mapping):
                raise ValueError("replacement bottles must be an object")
            return self.backend.resume_bottle_swap(dict(replacements)), False
        if command.name == "update_bottle_history_weight":
            bottle_id, weight = payload.get("bottle_id"), payload.get("weight")
            if not isinstance(bottle_id, str) or isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise ValueError("bottle identity and final weight are required")
            return self.backend.update_bottle_history_weight(bottle_id, weight), False
        if command.name == "save_bottle_measurements":
            bottles = payload.get("bottles")
            if bottles is not None and not isinstance(bottles, Mapping):
                raise ValueError("bottles must be an object or null")
            raw_run_dir = payload.get("run_dir")
            if raw_run_dir is not None and not isinstance(raw_run_dir, str):
                raise ValueError("run_dir must be a path string or null")
            result = self.backend.save_bottle_measurements(
                dict(bottles) if bottles is not None else None,
                run_dir=Path(raw_run_dir) if raw_run_dir is not None else None,
            )
            return result, False
        if command.name == "shutdown_server":
            self.backend.shutdown()
            self._graceful_shutdown.set()
            return {"snapshot": self._snapshot()}, True
        raise ValueError(f"unsupported supervisor command: {command.name}")

    def _command_worker(self) -> None:
        while not self._stopping.is_set() or not self._commands.empty():
            try:
                command = self._commands.get(timeout=0.1)
            except queue.Empty:
                continue
            if command is None:
                return
            signature = json.dumps(
                [command.name, _json_value(command.payload)],
                sort_keys=True,
                separators=(",", ":"),
            )
            if command.name in _MUTATING_COMMANDS:
                previous = self._mutation_ledger.get(command.request_id)
                if previous is not None:
                    self._mutation_ledger.move_to_end(command.request_id)
                    old_signature, response, stop = previous
                    if old_signature != signature:
                        self._enqueue(
                            "command_result",
                            {
                                "ok": False,
                                "result": None,
                                "error": "request_id reused for a different mutation",
                            },
                            request_id=command.request_id,
                            critical=True,
                        )
                        self._signal_client_loss("conflicting supervisor request_id")
                        return
                    self._enqueue(
                        "command_result",
                        response,
                        request_id=command.request_id,
                        critical=True,
                    )
                    if stop:
                        self._stopping.set()
                        return
                    continue
                if len(self._mutation_ledger) >= MAX_MUTATION_REQUESTS:
                    # Retain a fixed recent replay window instead of entering a
                    # permanent state where even shutdown is rejected. Request
                    # IDs are random and retries are immediate, so keeping the
                    # newest completed mutations is the useful idempotency bound.
                    self._mutation_ledger.popitem(last=False)
            try:
                with self._operation_lock:
                    result, stop = self._dispatch(command)
                response = {"ok": True, "result": result, "error": None}
            except Exception as exc:
                stop = False
                response = {
                    "ok": False,
                    "result": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            if command.name in _MUTATING_COMMANDS:
                self._mutation_ledger[command.request_id] = (
                    signature,
                    response,
                    stop,
                )
            delivered = threading.Event() if stop else None
            self._enqueue(
                "command_result",
                response,
                request_id=command.request_id,
                critical=True,
                sent=delivered,
            )
            if stop:
                if delivered is not None:
                    delivered.wait(timeout=1.0)
                self._stopping.set()
                self._close_client()
                return

    @staticmethod
    def _close_detached_client(client: socket.socket | None) -> None:
        if client is not None:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                client.close()
            except OSError:
                pass

    def _close_client(self) -> None:
        with self._client_lock:
            client, self._client = self._client, None
        self._close_detached_client(client)

    def _finalize_client_loss(self) -> None:
        # This path is reserved for actual supervisor/admin shutdown, never
        # presentation disconnection or missed GUI heartbeats.
        with self._operation_lock:
            self.backend.abort_run("supervisor shutdown requested")

    def _record_gui_event(self, event: str, **details: object) -> None:
        """Never perform diagnostic file/console I/O on acquisition callbacks."""
        record = {"event": event, "host_unix_ns": time.time_ns(),
                  "supervisor_pid": os.getpid(), **details}
        try:
            self._diagnostic_events.put_nowait(record)
        except queue.Full:
            return
        with self._diagnostic_lock:
            if self._diagnostic_thread is None:
                self._diagnostic_thread = threading.Thread(
                    target=self._write_gui_diagnostics,
                    name="squeakview-supervisor-recovery-log", daemon=True)
                try:
                    self._diagnostic_thread.start()
                except RuntimeError:
                    self._diagnostic_thread = None
                    # Resource-starved diagnostics must not stop acquisition.

    def _write_gui_diagnostics(self) -> None:
        while not self._stopping.is_set() or not self._diagnostic_events.empty():
            try:
                record = self._diagnostic_events.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                snapshot = self.backend.snapshot()
                run_dir = snapshot.run_dir
                if run_dir is not None and not isinstance(run_dir, (str, Path)):
                    run_dir = None
                directory = (Path(run_dir) / "diagnostics" if run_dir is not None
                             else self.runtime_context.user.state / "logs")
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / "supervisor_recovery.jsonl"
                record.update(run_dir=str(run_dir) if run_dir is not None else None,
                              phase=snapshot.phase.value)
                if not path.exists() or path.stat().st_size < 2 * 1024 * 1024:
                    with path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, ensure_ascii=True) + "\n")
                if record["event"] in {"gui_connection_lost", "gui_relaunch_failed"}:
                    print(f"[SUPERVISOR] {record.get('reason', record.get('error'))}; acquisition preserved",
                          file=sys.stderr, flush=True)
            except Exception:
                pass  # Recovery evidence cannot interrupt acquisition.

    def _signal_client_loss(
        self, error: str | None = None, *, client: socket.socket | None = None,
    ) -> None:
        with self._loss_lock:
            with self._client_lock:
                if client is not None and self._client is not client:
                    return  # An old sender cannot detach a replacement GUI.
                if self._client_lost.is_set():
                    return
                detached, self._client = self._client, None
                self._client_lost.set()
            self._presentation_loss_count += 1
            detail = error or "operator GUI disconnected"
            suffix = (
                "operator connection was lost after Stop was accepted; durable finalization continued"
                if self._stop_accepted.is_set() else
                "acquisition remains owned by the supervisor; GUI recovery will reconnect"
            )
            self.last_warning = f"{detail}; {suffix}"
        self._close_detached_client(detached)
        self._record_gui_event("gui_connection_lost", reason=detail,
                               acquisition_preserved=True)

    def _acquisition_active(self) -> bool:
        snapshot = self.backend.snapshot()
        return bool(
            snapshot.phase not in {RunPhase.IDLE, RunPhase.FINALIZED, RunPhase.FAILED}
            or snapshot.capture_running or snapshot.finalization_in_progress
            or self._operation_lock.locked() or not self._commands.empty()
        )

    def request_shutdown(self) -> None:
        """An actual supervisor shutdown still stops acquisition safely."""
        if self._stopping.is_set():
            return
        self._stopping.set()
        self._close_client()
        if self._graceful_shutdown.is_set() or self._stop_accepted.is_set():
            return
        self.backend.cancel_operator_lease()
        self._loss_thread = threading.Thread(
            target=self._finalize_client_loss,
            name="squeakview-supervisor-shutdown-finalizer", daemon=False,
        )
        self._loss_thread.start()

    def _prepare_listener(self) -> socket.socket:
        parent = self.socket_path.parent
        if not parent.exists():
            parent.mkdir(parents=True, mode=0o700)
        parent_stat = parent.lstat()
        if (
            not stat.S_ISDIR(parent_stat.st_mode)
            or parent_stat.st_uid != os.getuid()
            or parent_stat.st_mode & 0o077
        ):
            raise RuntimeError(
                "supervisor socket directory must be a private directory owned by this user"
            )
        if self.socket_path.exists():
            mode = self.socket_path.lstat().st_mode
            if not stat.S_ISSOCK(mode):
                raise RuntimeError(
                    f"refusing to replace non-socket path: {self.socket_path}"
                )
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            probe.settimeout(0.1)
            try:
                probe.connect(str(self.socket_path))
            except (ConnectionRefusedError, FileNotFoundError):
                self.socket_path.unlink(missing_ok=True)
            else:
                raise RuntimeError(
                    f"a supervisor is already listening at {self.socket_path}"
                )
            finally:
                probe.close()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            bound = self.socket_path.lstat()
            self._socket_identity = (bound.st_dev, bound.st_ino, bound.st_ctime_ns)
            listener.listen(1)
            listener.settimeout(0.25)
            return listener
        except Exception:
            listener.close()
            self._unlink_owned_socket()
            raise

    @staticmethod
    def _peer_credentials(client: socket.socket) -> tuple[int, int]:
        if not hasattr(socket, "SO_PEERCRED"):
            raise RuntimeError("SO_PEERCRED is required for the supervisor GUI lease")
        credentials = client.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
        )
        pid, uid, _gid = struct.unpack("3i", credentials)
        return int(pid), int(uid)

    def _unlink_owned_socket(self) -> None:
        try:
            current = self.socket_path.lstat()
            if stat.S_ISSOCK(current.st_mode) and self._socket_identity == (
                current.st_dev,
                current.st_ino,
                current.st_ctime_ns,
            ):
                self.socket_path.unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def _retire_gui(gui: subprocess.Popen[bytes] | None) -> None:
        if gui is None or gui.poll() is not None:
            return
        try:
            gui.terminate()
            gui.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                gui.kill()
                gui.wait(timeout=2.0)
            except (subprocess.TimeoutExpired, OSError, ChildProcessError):
                pass
        except (OSError, ChildProcessError):
            pass

    def serve(
        self, gui_command: Sequence[str], *,
        on_gui_ready: Callable[[], None] | None = None,
    ) -> int:
        """Keep acquisition owned while replacing failed presentation clients."""
        if not gui_command or any(not isinstance(item, str) or not item for item in gui_command):
            raise ValueError("gui_command must contain explicit non-empty argv")
        listener = self._prepare_listener()
        gui_env = os.environ.copy()
        gui_env[SOCKET_ENV] = str(self.socket_path)
        gui_env[PROJECT_ENV] = str(self.runtime_context.project.paths.root)
        gui: subprocess.Popen[bytes] | None = None
        worker: threading.Thread | None = None
        client: socket.socket | None = None
        reader: SocketEnvelopeReader | None = None
        connected_once = False
        gui_ready = False
        restart_count = 0
        next_launch = 0.0
        next_retire = 0.0
        connect_deadline = 0.0
        try:
            worker = threading.Thread(target=self._command_worker,
                name="squeakview-supervisor-commands", daemon=False)
            worker.start()
            while not self._stopping.is_set():
                if self._client_lost.is_set() and gui is not None:
                    if time.monotonic() < next_retire:
                        self._stopping.wait(0.1)
                        continue
                    previous_code = gui.poll()
                    self._retire_gui(gui)
                    exit_code = gui.poll()
                    if exit_code is None and self._acquisition_active():
                        # A kernel-blocked GUI may survive even SIGKILL until
                        # its driver returns. Keep owning that child instead
                        # of accumulating another renderer/GUI each restart.
                        self._record_gui_event("gui_retirement_pending", gui_pid=gui.pid,
                                               acquisition_preserved=True)
                        next_retire = time.monotonic() + 5.0
                        self._stopping.wait(0.1)
                        continue
                    next_retire = 0.0
                    self._record_gui_event("gui_exit", gui_pid=gui.pid,
                        returncode=exit_code,
                        signal=(-exit_code if isinstance(exit_code, int) and exit_code < 0 else None),
                        possible_oom=(previous_code == -9),
                        supervisor_retired=(previous_code is None))
                    gui = None
                    client = None
                    reader = None
                    if not self._acquisition_active():
                        return 0
                    restart_count += 1
                    # Bound a crash loop without imposing a recording deadline.
                    next_launch = time.monotonic() + min(30.0, 2.0 ** min(restart_count - 1, 5))
                if gui is None:
                    if time.monotonic() < next_launch:
                        self._stopping.wait(0.1)
                        continue
                    try:
                        env = dict(gui_env)
                        if connected_once or restart_count:
                            env["SQUEAKVIEW_GUI_RECOVERY"] = "1"
                        gui = subprocess.Popen(list(gui_command), env=env)
                        self._record_gui_event("gui_launched", gui_pid=gui.pid,
                                               restart_count=restart_count)
                    except OSError as exc:
                        if not self._acquisition_active():
                            raise
                        self._signal_client_loss(f"GUI recovery launch failed: {exc}")
                        self._record_gui_event("gui_relaunch_failed", error=str(exc),
                                               restart_count=restart_count)
                        restart_count += 1
                        next_launch = time.monotonic() + min(30.0, 2.0 ** min(restart_count, 5))
                        self._stopping.wait(0.1)
                        continue
                    connect_deadline = time.monotonic() + self.gui_connect_timeout_s
                    self._client_lost.clear()
                    gui_ready = False
                returncode = gui.poll()
                if returncode is not None:
                    detail = f"operator GUI exited with returncode={returncode}"
                    if returncode < 0:
                        import signal
                        try:
                            detail += f" ({signal.Signals(-returncode).name})"
                        except ValueError:
                            pass
                    self._signal_client_loss(detail)
                    if not self._acquisition_active():
                        self._record_gui_event("gui_exit", gui_pid=gui.pid,
                            returncode=returncode,
                            signal=(-returncode if returncode < 0 else None),
                            possible_oom=(returncode == -9), supervisor_retired=False)
                        return int(returncode)
                    continue
                if client is None:
                    if time.monotonic() >= connect_deadline:
                        self._signal_client_loss(
                            f"GUI did not connect within {self.gui_connect_timeout_s:.1f}s")
                        if not self._acquisition_active():
                            return 1
                        continue
                    try:
                        candidate, _ = listener.accept()
                    except socket.timeout:
                        continue
                    except OSError as exc:
                        self._signal_client_loss(f"GUI IPC accept failed: {exc}")
                        if not self._acquisition_active():
                            return 1
                        continue
                    try:
                        peer_pid, peer_uid = self._peer_credentials(candidate)
                    except (OSError, RuntimeError, struct.error) as exc:
                        candidate.close()
                        self._record_gui_event("gui_peer_rejected", error=str(exc))
                        continue
                    if peer_pid != gui.pid or peer_uid != os.getuid():
                        candidate.close()
                        continue
                    client = candidate
                    with self._client_lock:
                        self._client = client
                    # Old senders retain their old queues and cannot steal a
                    # replacement GUI's responses. Backend state is rehydrated
                    # through runtime_state, never stale presentation replay.
                    self._critical_events = queue.Queue(MAX_PENDING_CRITICAL_EVENTS)
                    self._presentation_events = queue.Queue(MAX_PENDING_PRESENTATION_EVENTS)
                    self._renew_gui_lease()
                    try:
                        client.settimeout(0.25)
                        threading.Thread(target=self._sender, args=(client,),
                            name="squeakview-supervisor-events", daemon=True).start()
                    except (OSError, RuntimeError) as exc:
                        self._signal_client_loss(f"GUI IPC sender setup failed: {exc}", client=client)
                        if not self._acquisition_active():
                            return 1
                        continue
                    reader = SocketEnvelopeReader()
                    self._record_gui_event("gui_connected", gui_pid=gui.pid,
                                           recovered=bool(connected_once or restart_count))
                    connected_once = True
                if self._gui_lease_expired():
                    self._signal_client_loss("operator GUI main-loop heartbeat lease expired", client=client)
                    continue
                assert reader is not None
                try:
                    envelope = reader.receive(client)
                except socket.timeout:
                    continue
                except (EOFError, OSError, ProtocolError) as exc:
                    self._signal_client_loss(f"operator GUI IPC disconnected: {type(exc).__name__}: {exc}", client=client)
                    continue
                if not isinstance(envelope, CommandEnvelope):
                    self._signal_client_loss("operator GUI sent an invalid IPC envelope", client=client)
                    continue
                try:
                    if self._handle_heartbeat(envelope):
                        if not gui_ready:
                            if on_gui_ready is not None and not restart_count:
                                try:
                                    on_gui_ready()
                                except Exception as exc:
                                    self._record_gui_event("gui_ready_ack_failed", error=str(exc))
                            gui_ready = True
                        continue
                except ProtocolError:
                    self._signal_client_loss("operator GUI sent an invalid heartbeat", client=client)
                    continue
                if envelope.name == "stop_run":
                    interrupt = getattr(self.backend, "interrupt_bottle_swap_for_stop", None)
                    if callable(interrupt):
                        interrupt()
                try:
                    self._commands.put_nowait(envelope)
                except queue.Full:
                    self._enqueue("command_result", {
                        "ok": False, "result": None,
                        "error": "supervisor command queue is full",
                    }, request_id=envelope.request_id, critical=True)
            return 0
        finally:
            # This is actual supervisor termination, not GUI detachment.
            if self._acquisition_active() and not self._graceful_shutdown.is_set():
                self.request_shutdown()
            self._stopping.set()
            self._close_client()
            try:
                self._commands.put_nowait(None)
            except queue.Full:
                pass
            listener.close()
            self._retire_gui(gui)
            if self._loss_thread is not None:
                self._loss_thread.join()
            if self._async_stop_thread is not None:
                self._async_stop_thread.join()
            if worker is not None and worker.ident is not None:
                worker.join()
            if self._diagnostic_thread is not None:
                self._diagnostic_thread.join(timeout=1.0)
            self._unlink_owned_socket()


__all__ = [
    "MAX_PENDING_COMMANDS",
    "MAX_PENDING_CRITICAL_EVENTS",
    "MAX_PENDING_PRESENTATION_EVENTS",
    "DEFAULT_GUI_CONNECT_TIMEOUT_S",
    "DEFAULT_GUI_LEASE_TIMEOUT_S",
    "SOCKET_ENV",
    "SupervisorServer",
]
