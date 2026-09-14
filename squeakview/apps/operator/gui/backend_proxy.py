from __future__ import annotations

"""GUI-side proxy for the durable operator supervisor.

The Qt process never owns acquisition resources.  This object deliberately
mirrors only the small ``OperatorBackend`` surface consumed by MainWindow and
RunLifecycleController.
"""

import dataclasses
import os
import queue
import socket
import stat
import struct
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from squeakview.apps.operator.backend.contracts import LaunchConfig
from squeakview.apps.operator.backend.events import BackendEvent, RunPhase, RunSnapshot
from squeakview.apps.operator.backend.supervisor.protocol import (
    CommandEnvelope,
    EventEnvelope,
    HEARTBEAT_COMMAND_NAME,
    HEARTBEAT_REQUEST_ID,
    SocketEnvelopeReader,
    SocketEnvelopeWriter,
)
from squeakview.common.dashboard import DashboardEvent


SUPERVISOR_SOCKET_ENV = "SQUEAKVIEW_SUPERVISOR_SOCKET"
ALLOW_INPROCESS_BACKEND_ENV = "SQUEAKVIEW_ALLOW_INPROCESS_BACKEND"
_CONNECT_TIMEOUT_SECONDS = 10.0
_SOCKET_IO_TIMEOUT_SECONDS = 1.0
_COMMAND_TIMEOUT_SECONDS = 120.0
# Normal stop performs only bounded ledger-tail and MP4 sample-table checks.
# Allow ample time for EOS/container closure without masking a stuck shutdown
# behind the offline analysis worker's former multi-hour ceiling.
_FINALIZE_TIMEOUT_SECONDS = 180.0
_MAX_PENDING_COMMANDS = 32
_MAX_PENDING_HEARTBEATS = 1
_MUTATING_COMMANDS = {
    "start_run",
    "stop_run",
    "save_bottle_measurements",
    "shutdown_server",
}
_SNAPSHOT_SCHEMA_VERSION = "1.0"
_SNAPSHOT_KEYS = {
    "schema_version",
    "phase",
    "run_dir",
    "error",
    "capture_running",
    "finalization_in_progress",
}
_BACKEND_EVENT_KEYS = {
    "schema_version",
    "type",
    "phase",
    "run_dir",
    "message",
    "payload",
    "host_unix_ns",
}


def _plain(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def launch_config_payload(config: LaunchConfig) -> dict[str, object]:
    """Return a strict JSON-compatible snapshot of a launch request."""

    return {
        field.name: _plain(getattr(config, field.name))
        for field in dataclasses.fields(LaunchConfig)
    }


def launch_config_from_payload(payload: Mapping[str, object]) -> LaunchConfig:
    """Validate and reconstruct a launch request received from the supervisor."""

    fields = {field.name for field in dataclasses.fields(LaunchConfig)}
    unknown = set(payload) - fields
    missing = fields - set(payload)
    if unknown or missing:
        raise ValueError(
            "launch config fields do not match the local schema "
            f"(missing={sorted(missing)!r}, unknown={sorted(unknown)!r})"
        )
    values = dict(payload)
    for name in ("ds_cfg", "run_dir", "task_cfg", "failure_plan"):
        raw = values[name]
        values[name] = None if raw is None else Path(str(raw))
    values["preview_socket_paths"] = tuple(
        Path(str(path)) for path in values["preview_socket_paths"]
    )
    values["camera_serials"] = tuple(str(item) for item in values["camera_serials"])
    bottles = values["bottles"]
    if not isinstance(bottles, Mapping):
        raise ValueError("launch config bottles must be an object")
    values["bottles"] = dict(bottles)
    return LaunchConfig(**values)


def _optional_wire_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError(f"{name} must be a string or null")
    return value


def _snapshot_from_payload(payload: Mapping[str, object]) -> RunSnapshot:
    if not isinstance(payload, Mapping) or set(payload) != _SNAPSHOT_KEYS:
        raise ValueError("snapshot must contain exact version-1 fields")
    if payload["schema_version"] != _SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("snapshot schema_version must be the string '1.0'")
    if type(payload["phase"]) is not str:
        raise ValueError("snapshot phase must be a string")
    if type(payload["capture_running"]) is not bool:
        raise ValueError("snapshot capture_running must be boolean")
    if type(payload["finalization_in_progress"]) is not bool:
        raise ValueError("snapshot finalization_in_progress must be boolean")
    raw_run_dir = _optional_wire_string(payload["run_dir"], "snapshot run_dir")
    error = _optional_wire_string(payload["error"], "snapshot error")
    return RunSnapshot(
        phase=RunPhase(payload["phase"]),
        run_dir=None if raw_run_dir is None else Path(raw_run_dir),
        error=error,
        capture_running=payload["capture_running"],
        finalization_in_progress=payload["finalization_in_progress"],
    )


def _backend_event_from_payload(payload: Mapping[str, object]) -> BackendEvent:
    if not isinstance(payload, Mapping) or set(payload) != _BACKEND_EVENT_KEYS:
        raise ValueError("backend event must contain exact version-1 fields")
    if payload["schema_version"] != "1.0":
        raise ValueError("backend event schema_version must be the string '1.0'")
    event_type = payload["type"]
    if type(event_type) is not str or not event_type:
        raise ValueError("backend event type must be a non-empty string")
    if type(payload["phase"]) is not str:
        raise ValueError("backend event phase must be a string")
    raw_run_dir = _optional_wire_string(payload["run_dir"], "backend event run_dir")
    message = _optional_wire_string(payload["message"], "backend event message")
    event_payload = payload["payload"]
    if not isinstance(event_payload, Mapping):
        raise ValueError("backend event payload must be an object")
    host_unix_ns = payload["host_unix_ns"]
    if type(host_unix_ns) is not int or host_unix_ns < 0:
        raise ValueError("backend event host_unix_ns must be a nonnegative integer")
    return BackendEvent(
        type=event_type,
        phase=RunPhase(payload["phase"]),
        run_dir=None if raw_run_dir is None else Path(raw_run_dir),
        message=message,
        payload=event_payload,
        schema_version="1.0",
        host_unix_ns=host_unix_ns,
    )


class _RemoteCaptureHandle:
    def __init__(self) -> None:
        self.running = False

    def is_running(self) -> bool:
        return self.running


@dataclass(slots=True)
class _RemoteState:
    inference: _RemoteCaptureHandle | None = None
    run_dir: Path | None = None
    phase: RunPhase = RunPhase.IDLE


@dataclass(slots=True)
class _PendingResponse:
    name: str
    ready: threading.Event
    payload: Mapping[str, object] | None = None
    error: BaseException | None = None


class SupervisorBackendProxy:
    """Synchronous backend facade backed by one supervised Unix connection."""

    def __init__(
        self,
        socket_path: Path,
        emit_log: Callable[[str], None],
        ingest_dashboard: Callable[[DashboardEvent], None] | None = None,
        on_run_started: Callable[[], None] | None = None,
        on_run_failed: Callable[[str], None] | None = None,
        *,
        connect_timeout: float = _CONNECT_TIMEOUT_SECONDS,
    ) -> None:
        self.emit = emit_log
        self.ingest = ingest_dashboard
        self.on_run_started = on_run_started
        self.on_run_failed = on_run_failed
        self.launch_cfg = LaunchConfig()
        self.state = _RemoteState()
        self._finalization_in_progress = False
        self._last_error: str | None = None
        self._subscribers: list[Callable[[BackendEvent], None]] = []
        self._pending: dict[str, _PendingResponse] = {}
        self._pending_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._failure_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._closed = threading.Event()
        self._intentional_close = False
        self._shutdown_requested = False
        self._last_sequence: int | None = None
        self._socket = self._connect(Path(socket_path), connect_timeout)
        self._reader = SocketEnvelopeReader()
        self._writer = SocketEnvelopeWriter(self._socket)
        self._heartbeats: queue.Queue[CommandEnvelope] = queue.Queue(
            maxsize=_MAX_PENDING_HEARTBEATS
        )
        self._reader_thread = threading.Thread(
            target=self._read_events,
            name="squeakview-supervisor-events",
            daemon=True,
        )
        self._reader_thread.start()
        self._heartbeat_writer_thread = threading.Thread(
            target=self._write_heartbeats,
            name="squeakview-supervisor-heartbeats",
            daemon=True,
        )
        self._heartbeat_writer_thread.start()

    @staticmethod
    def _connect(path: Path, timeout: float) -> socket.socket:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                metadata = path.lstat()
                if not stat.S_ISSOCK(metadata.st_mode):
                    raise RuntimeError(f"supervisor path is not a Unix socket: {path}")
                if metadata.st_uid != os.getuid():
                    raise RuntimeError(f"supervisor socket is not owned by this user: {path}")
                if stat.S_IMODE(metadata.st_mode) & 0o077:
                    raise RuntimeError(f"supervisor socket permissions are not private: {path}")
                remaining = max(0.001, deadline - time.monotonic())
                connection.settimeout(min(_SOCKET_IO_TIMEOUT_SECONDS, remaining))
                connection.connect(str(path))
                if hasattr(socket, "SO_PEERCRED"):
                    credentials = connection.getsockopt(
                        socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
                    )
                    _pid, peer_uid, _gid = struct.unpack("3i", credentials)
                    if peer_uid != os.getuid():
                        raise RuntimeError("supervisor peer is not owned by this user")
                connection.settimeout(_SOCKET_IO_TIMEOUT_SECONDS)
                return connection
            except RuntimeError:
                connection.close()
                raise
            except OSError as exc:
                connection.close()
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"could not connect to SqueakView supervisor at {path}: {exc}"
                    ) from exc
                time.sleep(0.05)

    def subscribe(self, callback: Callable[[BackendEvent], None]) -> None:
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def heartbeat(self) -> bool:
        """Queue one lease renewal without blocking the Qt main thread.

        The queue deliberately holds at most one renewal.  If the socket writer
        is stalled, later timer ticks coalesce rather than growing memory.
        """

        if self._closed.is_set() or self._shutdown_requested:
            return False
        try:
            self._heartbeats.put_nowait(
                CommandEnvelope(
                    HEARTBEAT_COMMAND_NAME,
                    HEARTBEAT_REQUEST_ID,
                    {},
                )
            )
            return True
        except queue.Full:
            return False

    def _write_heartbeats(self) -> None:
        while not self._closed.is_set():
            try:
                heartbeat = self._heartbeats.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                with self._send_lock:
                    self._writer.send(heartbeat)
            except Exception as exc:
                if not self._intentional_close:
                    self._connection_failed(exc)
                return

    @property
    def finalization_in_progress(self) -> bool:
        with self._state_lock:
            return self._finalization_in_progress

    @property
    def runtime_config(self) -> LaunchConfig:
        return self.launch_cfg

    @property
    def current_snapshot(self) -> RunSnapshot:
        with self._state_lock:
            capture = self.state.inference
            return RunSnapshot(
                phase=self.state.phase,
                run_dir=self.state.run_dir,
                error=self._last_error,
                capture_running=bool(capture and capture.is_running()),
                finalization_in_progress=self._finalization_in_progress,
            )

    def snapshot(self) -> RunSnapshot:
        result = self._command("snapshot", {}, timeout=10.0)
        if not isinstance(result, Mapping):
            raise RuntimeError("supervisor returned an invalid snapshot")
        return self._apply_snapshot(result)

    def start_run(self, cfg: LaunchConfig) -> bool:
        self.launch_cfg = cfg
        result = self._command(
            "start_run", {"config": launch_config_payload(cfg)}
        )
        try:
            # Until every response field is validated, an accepted start is
            # conservatively treated as remotely active.
            with self._state_lock:
                if self.state.inference is None:
                    self.state.inference = _RemoteCaptureHandle()
                self.state.inference.running = True
                self._finalization_in_progress = True
            if not isinstance(result, Mapping) or set(result) != {
                "started",
                "config",
                "snapshot",
            }:
                raise ValueError("start_run result must contain exact fields")
            if type(result["started"]) is not bool:
                raise ValueError("start_run started must be boolean")
            raw_config = result["config"]
            snapshot = result["snapshot"]
            if not isinstance(raw_config, Mapping) or not isinstance(snapshot, Mapping):
                raise ValueError("start_run config and snapshot must be objects")
            self.launch_cfg = launch_config_from_payload(raw_config)
            self._apply_snapshot(snapshot)
            return result["started"]
        except Exception as exc:
            # A semantically invalid reply cannot prove whether the accepted
            # mutation started capture. Mark ownership unknown before closing
            # the lease so the GUI cannot present a safe idle state.
            self._connection_failed(exc)
            raise

    def stop_run(self) -> None:
        result = self._command("stop_run", {}, timeout=_FINALIZE_TIMEOUT_SECONDS)
        if isinstance(result, Mapping):
            snapshot = result.get("snapshot", result)
            if isinstance(snapshot, Mapping) and "phase" in snapshot:
                self._apply_snapshot(snapshot)

    def save_bottle_measurements(
        self,
        bottles: dict[str, Any] | None,
        run_dir: Path | None = None,
    ) -> dict[str, Any]:
        result = self._command(
            "save_bottle_measurements",
            {
                "bottles": _plain(bottles or {}),
                "run_dir": None if run_dir is None else str(run_dir),
            },
        )
        if not isinstance(result, Mapping):
            raise RuntimeError("supervisor returned invalid bottle metadata")
        return dict(result)

    def shutdown(self) -> None:
        if self._closed.is_set():
            return
        self._shutdown_requested = True
        try:
            result = self._command(
                "shutdown_server", {}, timeout=_FINALIZE_TIMEOUT_SECONDS
            )
        except Exception:
            self._shutdown_requested = False
            raise
        if isinstance(result, Mapping):
            snapshot = result.get("snapshot", result)
            if isinstance(snapshot, Mapping) and "phase" in snapshot:
                self._apply_snapshot(snapshot)
        with self._state_lock:
            active = bool(
                self.state.inference is not None
                and self.state.inference.is_running()
            )
            finalizing = self._finalization_in_progress
        if active or finalizing:
            raise RuntimeError("supervisor acknowledged shutdown while acquisition remained active")
        self._intentional_close = True
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._socket.close()
        self._closed.set()

    def _command(
        self,
        name: str,
        payload: Mapping[str, object],
        *,
        timeout: float = _COMMAND_TIMEOUT_SECONDS,
    ) -> object:
        if self._closed.is_set():
            raise RuntimeError("SqueakView supervisor connection is closed")
        request_id = uuid.uuid4().hex
        pending = _PendingResponse(name, threading.Event())
        with self._pending_lock:
            if len(self._pending) >= _MAX_PENDING_COMMANDS:
                raise RuntimeError("too many concurrent supervisor commands")
            self._pending[request_id] = pending
        try:
            try:
                with self._send_lock:
                    self._writer.send(CommandEnvelope(name, request_id, payload))
            except Exception as exc:
                # A timed-out send may have written only part of its frame. The
                # stream cannot be reused safely, and a start request may already
                # have reached the supervisor, so fail the transport closed.
                self._connection_failed(exc)
                raise
            if not pending.ready.wait(timeout=max(0.0, timeout)):
                error = TimeoutError(
                    f"supervisor command {name!r} did not complete within {timeout:g}s"
                )
                # The supervisor may have accepted a mutation even though its
                # reply was delayed.  Retaining the GUI lease would allow a
                # late start/stop to proceed after the GUI reported failure.
                # Closing the transport makes ownership uncertainty fail closed
                # through the supervisor's operator-loss finalizer.
                if name in _MUTATING_COMMANDS:
                    self._connection_failed(error)
                raise error
            if pending.error is not None:
                raise pending.error
            response = pending.payload
            if response is None or not isinstance(response.get("ok"), bool):
                raise RuntimeError(f"supervisor returned an invalid response to {name!r}")
            if not response["ok"]:
                detail = response.get("error") or f"supervisor command {name!r} failed"
                raise RuntimeError(str(detail))
            return response.get("result")
        finally:
            with self._pending_lock:
                self._pending.pop(request_id, None)

    def _read_events(self) -> None:
        try:
            while not self._closed.is_set():
                try:
                    envelope = self._reader.receive(self._socket)
                except socket.timeout:
                    # A quiet supervisor is normal during long acquisition and
                    # finalization commands. The finite timeout exists so both
                    # directions remain interruptible, not as a liveness test.
                    continue
                if not isinstance(envelope, EventEnvelope):
                    raise RuntimeError("supervisor sent a command on the event stream")
                if self._last_sequence is not None and envelope.sequence != self._last_sequence + 1:
                    raise RuntimeError(
                        "supervisor event sequence was not contiguous "
                        f"({self._last_sequence} -> {envelope.sequence})"
                    )
                self._last_sequence = envelope.sequence
                if envelope.name == "command_result":
                    self._complete_command(envelope)
                else:
                    self._dispatch_event(envelope)
        except Exception as exc:
            if not self._intentional_close:
                self._connection_failed(exc)

    def _complete_command(self, envelope: EventEnvelope) -> None:
        if envelope.request_id is None:
            raise RuntimeError("command_result omitted request_id")
        with self._pending_lock:
            pending = self._pending.get(envelope.request_id)
        if pending is None:
            # A timed-out command may finish later; it must not grow retained state.
            return
        pending.payload = envelope.payload
        if pending.name == "shutdown_server":
            # The server closes immediately after flushing this response.  Mark
            # that EOF as intentional before waking the caller to remove the
            # response/EOF scheduling race.
            self._intentional_close = True
        pending.ready.set()

    def _dispatch_event(self, envelope: EventEnvelope) -> None:
        if envelope.name == "log":
            self._safe_emit(str(envelope.payload.get("message", "")))
            return
        if envelope.name == "dashboard":
            event = DashboardEvent.from_payload(envelope.payload)
            if self.ingest is not None:
                try:
                    self.ingest(event)
                except Exception as exc:
                    self._safe_emit(f"[SUPERVISOR] dashboard callback failed: {exc}")
            return
        if envelope.name == "presentation_dropped":
            self._safe_emit(
                "[SUPERVISOR] presentation backlog dropped "
                f"{int(envelope.payload.get('count', 0))} noncritical event(s)"
            )
            return
        if envelope.name != "backend_event":
            raise RuntimeError(f"unknown supervisor event {envelope.name!r}")
        event = _backend_event_from_payload(envelope.payload)
        phase = event.phase
        self._apply_phase(event.phase, event.run_dir)
        for callback in tuple(self._subscribers):
            try:
                callback(event)
            except Exception as exc:
                self._safe_emit(f"[SUPERVISOR] backend event subscriber failed: {exc}")
        try:
            if phase == RunPhase.RECORDING and self.on_run_started is not None:
                self.on_run_started()
            elif phase == RunPhase.FAILED and self.on_run_failed is not None:
                self.on_run_failed(event.message or "supervised run failed")
        except Exception as exc:
            self._safe_emit(f"[SUPERVISOR] lifecycle callback failed: {exc}")

    def _apply_phase(self, phase: RunPhase, run_dir: Path | None) -> None:
        active = phase in {
            RunPhase.STARTING,
            RunPhase.RECORDING,
            RunPhase.STOPPING,
            RunPhase.DRAINING,
        }
        finalizing = phase in {
            RunPhase.STOPPING,
            RunPhase.DRAINING,
            RunPhase.CAPTURE_CLOSED,
            RunPhase.VALIDATING,
        }
        with self._state_lock:
            self.state.phase = phase
            self.state.run_dir = run_dir
            if self.state.inference is None:
                self.state.inference = _RemoteCaptureHandle()
            self.state.inference.running = active
            self._finalization_in_progress = finalizing

    def _apply_snapshot(self, payload: Mapping[str, object]) -> RunSnapshot:
        try:
            snapshot = _snapshot_from_payload(payload)
        except Exception as exc:
            self._connection_failed(exc)
            raise
        self._apply_phase(snapshot.phase, snapshot.run_dir)
        with self._state_lock:
            self._last_error = snapshot.error
            if self.state.inference is None:
                self.state.inference = _RemoteCaptureHandle()
            self.state.inference.running = snapshot.capture_running
            self._finalization_in_progress = snapshot.finalization_in_progress
        return snapshot

    def _connection_failed(self, error: BaseException) -> None:
        with self._failure_lock:
            if self._closed.is_set():
                return
            with self._pending_lock:
                pending = tuple(self._pending.values())
            if self._shutdown_requested and not pending:
                self._intentional_close = True
                self._closed.set()
                self._socket.close()
                return
            self._closed.set()
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._socket.close()
        failure = RuntimeError(
            f"SqueakView supervisor connection lost: {type(error).__name__}: {error}"
        )
        for request in pending:
            request.error = failure
            request.ready.set()
        self._safe_emit(f"[SUPERVISOR] {failure}")
        with self._state_lock:
            was_active = any(
                request.name == "start_run" for request in pending
            ) or self.state.phase not in {
                RunPhase.IDLE,
                RunPhase.FINALIZED,
                RunPhase.FAILED,
            } or bool(
                self.state.inference is not None
                and self.state.inference.is_running()
            ) or self._finalization_in_progress
            # Unknown remote ownership is represented as active.  The guarded
            # GUI close path therefore cannot claim a safe shutdown after IPC loss.
            if was_active:
                if self.state.inference is None:
                    self.state.inference = _RemoteCaptureHandle()
                self.state.inference.running = True
                self._finalization_in_progress = True
        if was_active and self.on_run_failed is not None:
            try:
                self.on_run_failed(str(failure))
            except Exception as exc:
                self._safe_emit(f"[SUPERVISOR] failure callback failed: {exc}")

    def _safe_emit(self, message: str) -> None:
        try:
            self.emit(message)
        except Exception:
            pass


def supervisor_socket_from_environment() -> Path:
    raw = os.environ.get(SUPERVISOR_SOCKET_ENV, "").strip()
    if not raw:
        raise RuntimeError(
            f"{SUPERVISOR_SOCKET_ENV} is required. Launch SqueakView with "
            "squeakview.sh so the durable supervisor owns acquisition."
        )
    return Path(raw)


__all__ = [
    "ALLOW_INPROCESS_BACKEND_ENV",
    "SUPERVISOR_SOCKET_ENV",
    "SupervisorBackendProxy",
    "launch_config_from_payload",
    "launch_config_payload",
    "supervisor_socket_from_environment",
]
