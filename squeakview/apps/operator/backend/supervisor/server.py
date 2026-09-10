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
DEFAULT_GUI_LEASE_TIMEOUT_S = 10.0
_PATH_FIELDS = {"ds_cfg", "task_cfg", "failure_plan", "run_dir"}
_PATH_TUPLE_FIELDS = {"preview_socket_paths"}
_STRING_TUPLE_FIELDS = {"camera_serials"}
_MUTATING_COMMANDS = {
    "start_run",
    "stop_run",
    "save_bottle_measurements",
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
    bool_fields = {"trigger_on", "inference_enabled", "serial_enabled", "preview_enabled"}
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
        backend_factory: Callable[..., OperatorBackend] = OperatorBackend,
        gui_connect_timeout_s: float = DEFAULT_GUI_CONNECT_TIMEOUT_S,
        gui_lease_timeout_s: float = DEFAULT_GUI_LEASE_TIMEOUT_S,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.socket_path = Path(socket_path).resolve()
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
        self._mutation_ledger: OrderedDict[
            str, tuple[str, Mapping[str, object], bool]
        ] = OrderedDict()
        self._socket_identity: tuple[int, int, int] | None = None
        self.gui_connect_timeout_s = max(1.0, float(gui_connect_timeout_s))
        self.gui_lease_timeout_s = max(1.0, float(gui_lease_timeout_s))
        self._monotonic = monotonic
        self.last_error: str | None = None
        self.backend = backend_factory(
            self._emit_log,
            self._emit_dashboard,
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
        writer = SocketEnvelopeWriter(client)
        while not self._stopping.is_set():
            item: _Outbound | None = None
            try:
                item = self._critical_events.get_nowait()
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
                        item = self._presentation_events.get(timeout=0.05)
                    except queue.Empty:
                        continue
            try:
                writer.send(
                    EventEnvelope(
                        name=item.name,
                        sequence=self._next_sequence(),
                        request_id=item.request_id,
                        payload=_json_value(item.payload),
                    )
                )
                if item.sent is not None:
                    item.sent.set()
            except (OSError, ProtocolError):
                self._signal_client_loss("supervisor IPC send failed")
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

    def _dispatch(self, command: CommandEnvelope) -> tuple[object, bool]:
        payload = command.payload
        if command.name == "ping":
            return {"protocol": 1}, False
        if command.name == "snapshot":
            return self._snapshot(), False
        if command.name == "start_run":
            config_payload = payload.get("config", payload)
            if not isinstance(config_payload, Mapping):
                raise ValueError("start_run config must be an object")
            started = self.backend.start_run(_request_from_payload(config_payload))
            return {
                "started": started,
                "config": _json_value(self.backend.launch_cfg),
                "snapshot": self._snapshot(),
            }, False
        if command.name == "stop_run":
            self.backend.stop_run()
            return {"snapshot": self._snapshot()}, False
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
            if command is None or self._client_lost.is_set():
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

    def _close_client(self) -> None:
        with self._client_lock:
            client, self._client = self._client, None
        if client is not None:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            client.close()

    def _finalize_client_loss(self) -> None:
        with self._operation_lock:
            self.backend.abort_run("operator GUI lost")

    def _signal_client_loss(self, error: str | None = None) -> None:
        with self._loss_lock:
            if (
                error
                and not self._graceful_shutdown.is_set()
                and self.last_error is None
            ):
                self.last_error = error
            if self._client_lost.is_set():
                return
            self._client_lost.set()
        self._stopping.set()
        self._close_client()
        self.backend.cancel_operator_lease()
        self._loss_thread = threading.Thread(
            target=self._finalize_client_loss,
            name="squeakview-operator-loss-finalizer",
            daemon=False,
        )
        self._loss_thread.start()

    def request_shutdown(self) -> None:
        """Stop accepting IPC and fail closed if acquisition is active."""

        self._signal_client_loss("supervisor shutdown requested")

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
            bound = self.socket_path.lstat()
            self._socket_identity = (bound.st_dev, bound.st_ino, bound.st_ctime_ns)
            os.chmod(self.socket_path, 0o600)
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

    def serve(
        self,
        gui_command: Sequence[str],
        *,
        on_gui_ready: Callable[[], None] | None = None,
    ) -> int:
        """Launch the mandatory GUI and serve its sole IPC lease."""

        if not gui_command or any(not isinstance(item, str) or not item for item in gui_command):
            raise ValueError("gui_command must contain explicit non-empty argv")
        listener = self._prepare_listener()
        gui_env = os.environ.copy()
        gui_env[SOCKET_ENV] = str(self.socket_path)
        gui: subprocess.Popen[bytes] | None = None
        worker: threading.Thread | None = None
        client: socket.socket | None = None
        try:
            gui = subprocess.Popen(list(gui_command), env=gui_env)
            worker = threading.Thread(
                target=self._command_worker,
                name="squeakview-supervisor-commands",
                daemon=False,
            )
            worker.start()
            connect_deadline = time.monotonic() + self.gui_connect_timeout_s
            while client is None and not self._stopping.is_set():
                if gui.poll() is not None:
                    self._signal_client_loss("mandatory operator GUI exited")
                    break
                if time.monotonic() >= connect_deadline:
                    self.last_error = (
                        "mandatory GUI did not connect to the supervisor within "
                        f"{self.gui_connect_timeout_s:.1f}s"
                    )
                    self._signal_client_loss(self.last_error)
                    break
                try:
                    client, _ = listener.accept()
                except socket.timeout:
                    continue
                peer_pid, peer_uid = self._peer_credentials(client)
                if peer_pid != gui.pid or peer_uid != os.getuid():
                    client.close()
                    client = None
                    continue
            if client is None:
                return int(gui.returncode or 1)
            with self._client_lock:
                self._client = client
            self._renew_gui_lease()
            client.settimeout(0.25)
            sender = threading.Thread(
                target=self._sender,
                args=(client,),
                name="squeakview-supervisor-events",
                daemon=True,
            )
            sender.start()
            reader = SocketEnvelopeReader()
            gui_ready = False
            while not self._stopping.is_set():
                if self._gui_lease_expired():
                    self._signal_client_loss(
                        "operator GUI main-loop heartbeat lease expired"
                    )
                    break
                if gui.poll() is not None:
                    self._signal_client_loss("mandatory operator GUI exited")
                    break
                try:
                    envelope = reader.receive(client)
                except socket.timeout:
                    continue
                except (EOFError, OSError, ProtocolError):
                    self._signal_client_loss("operator GUI IPC lease was lost")
                    break
                if not isinstance(envelope, CommandEnvelope):
                    self._signal_client_loss(
                        "operator GUI sent an invalid IPC envelope"
                    )
                    break
                try:
                    if self._handle_heartbeat(envelope):
                        if not gui_ready:
                            if on_gui_ready is not None:
                                on_gui_ready()
                            gui_ready = True
                        continue
                except ProtocolError:
                    self._signal_client_loss(
                        "operator GUI sent an invalid heartbeat"
                    )
                    break
                try:
                    self._commands.put_nowait(envelope)
                except queue.Full:
                    self._enqueue(
                        "command_result",
                        {
                            "ok": False,
                            "result": None,
                            "error": "supervisor command queue is full",
                        },
                        request_id=envelope.request_id,
                        critical=True,
                    )
            if self.last_error is not None and not self._graceful_shutdown.is_set():
                return 1
            return int(gui.poll() or 0)
        finally:
            self._signal_client_loss()
            try:
                self._commands.put_nowait(None)
            except queue.Full:
                pass
            self._close_client()
            listener.close()
            if gui is not None and gui.poll() is None:
                gui.terminate()
                try:
                    gui.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    gui.kill()
                    gui.wait(timeout=5.0)
            if self._loss_thread is not None:
                self._loss_thread.join()
            if worker is not None and worker.ident is not None:
                worker.join()
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
