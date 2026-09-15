from __future__ import annotations

"""Versioned, bounded JSON-lines protocol for local supervisor IPC.

The transport is intentionally small and independent of both Qt and backend
state.  A Unix-domain socket supplies local transport and filesystem access
control; this module owns only immutable messages and their framing.
"""

import json
import math
import re
import select
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping, TypeAlias


PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 64 * 1024
SOCKET_READ_BYTES = 16 * 1024
MAX_JSON_DEPTH = 32
HEARTBEAT_COMMAND_NAME = "heartbeat"
HEARTBEAT_REQUEST_ID = "gui-heartbeat"

_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_MAX_REQUEST_ID_CHARS = 128
_MAX_SEQUENCE = (1 << 63) - 1

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]


class ProtocolError(ValueError):
    """The peer supplied an invalid or incomplete protocol frame."""


class FrameTooLargeError(ProtocolError):
    """A frame exceeded the fixed protocol memory bound."""


def _validate_version(version: object) -> int:
    if isinstance(version, bool) or not isinstance(version, int):
        raise ProtocolError("version must be an integer")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version: {version}")
    return version


def _validate_name(name: object) -> str:
    if not isinstance(name, str) or _NAME_RE.fullmatch(name) is None:
        raise ProtocolError(
            "name must start with a lowercase letter and contain at most "
            "64 lowercase ASCII letters, digits, '.', '_', or '-'"
        )
    return name


def _validate_request_id(request_id: object, *, optional: bool) -> str | None:
    if request_id is None and optional:
        return None
    if not isinstance(request_id, str):
        raise ProtocolError("request_id must be a string")
    if not request_id or len(request_id) > _MAX_REQUEST_ID_CHARS:
        raise ProtocolError("request_id must contain 1 to 128 characters")
    if any(ord(character) < 0x21 or ord(character) > 0x7E for character in request_id):
        raise ProtocolError("request_id must contain printable non-space ASCII only")
    return request_id


def _freeze_json(
    value: object, *, path: str = "payload", depth: int = 0
) -> JsonValue:
    if depth > MAX_JSON_DEPTH:
        raise ProtocolError(f"{path} exceeds maximum JSON depth {MAX_JSON_DEPTH}")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ProtocolError(f"{path} contains an invalid Unicode surrogate") from exc
        return value
    if isinstance(value, int):
        if not (-_MAX_SEQUENCE - 1 <= value <= _MAX_SEQUENCE):
            raise ProtocolError(f"{path} integer is outside signed 64-bit range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]", depth=depth + 1)
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProtocolError(f"{path} keys must be strings")
            try:
                key.encode("utf-8", errors="strict")
            except UnicodeEncodeError as exc:
                raise ProtocolError(
                    f"{path} contains an invalid Unicode surrogate key"
                ) from exc
            frozen[key] = _freeze_json(
                item, path=f"{path}.{key}", depth=depth + 1
            )
        return MappingProxyType(frozen)
    raise ProtocolError(f"{path} contains unsupported value type {type(value).__name__}")


def _freeze_payload(payload: object) -> Mapping[str, JsonValue]:
    if not isinstance(payload, Mapping):
        raise ProtocolError("payload must be a JSON object")
    frozen = _freeze_json(payload)
    if not isinstance(frozen, Mapping):  # pragma: no cover - narrowed above
        raise ProtocolError("payload must be a JSON object")
    return frozen


@dataclass(frozen=True, slots=True)
class CommandEnvelope:
    """One idempotently identifiable GUI-to-supervisor request."""

    name: str
    request_id: str
    payload: Mapping[str, JsonValue] = field(default_factory=dict)
    version: int = PROTOCOL_VERSION
    type: Literal["command"] = field(default="command", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _validate_version(self.version))
        object.__setattr__(self, "name", _validate_name(self.name))
        object.__setattr__(
            self,
            "request_id",
            _validate_request_id(self.request_id, optional=False),
        )
        object.__setattr__(self, "payload", _freeze_payload(self.payload))


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    """One ordered supervisor-to-GUI notification or command response."""

    name: str
    sequence: int
    payload: Mapping[str, JsonValue] = field(default_factory=dict)
    request_id: str | None = None
    version: int = PROTOCOL_VERSION
    type: Literal["event"] = field(default="event", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _validate_version(self.version))
        object.__setattr__(self, "name", _validate_name(self.name))
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or not 0 <= self.sequence <= _MAX_SEQUENCE
        ):
            raise ProtocolError("sequence must be an integer from 0 through 2^63-1")
        object.__setattr__(
            self,
            "request_id",
            _validate_request_id(self.request_id, optional=True),
        )
        object.__setattr__(self, "payload", _freeze_payload(self.payload))


Envelope: TypeAlias = CommandEnvelope | EventEnvelope


def _plain_json(value: JsonValue) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


def _envelope_object(envelope: Envelope) -> dict[str, object]:
    if isinstance(envelope, CommandEnvelope):
        return {
            "version": envelope.version,
            "type": envelope.type,
            "name": envelope.name,
            "request_id": envelope.request_id,
            "payload": _plain_json(envelope.payload),
        }
    if isinstance(envelope, EventEnvelope):
        return {
            "version": envelope.version,
            "type": envelope.type,
            "name": envelope.name,
            "sequence": envelope.sequence,
            "request_id": envelope.request_id,
            "payload": _plain_json(envelope.payload),
        }
    raise TypeError(f"unsupported envelope type: {type(envelope).__name__}")


def encode_envelope(envelope: Envelope) -> bytes:
    """Encode one envelope, including its terminating newline."""

    try:
        body = json.dumps(
            _envelope_object(envelope),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:  # defensive if an object is forged
        raise ProtocolError(f"envelope is not JSON encodable: {exc}") from exc
    frame = body + b"\n"
    if len(frame) > MAX_FRAME_BYTES:
        raise FrameTooLargeError(
            f"encoded frame is {len(frame)} bytes; maximum is {MAX_FRAME_BYTES}"
        )
    return frame


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ProtocolError(f"non-finite JSON number is forbidden: {value}")


def _decode_body(body: bytes) -> Envelope:
    if not body:
        raise ProtocolError("empty protocol frame")
    if len(body) + 1 > MAX_FRAME_BYTES:
        raise FrameTooLargeError(
            f"frame exceeds maximum of {MAX_FRAME_BYTES} bytes"
        )
    try:
        text = body.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ProtocolError("frame is not valid UTF-8") from exc
    try:
        raw = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except ProtocolError:
        raise
    except RecursionError as exc:
        raise ProtocolError("JSON nesting exceeds the parser depth limit") from exc
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise ProtocolError("envelope must be a JSON object")

    envelope_type = raw.get("type")
    if envelope_type == "command":
        expected = {"version", "type", "name", "request_id", "payload"}
        if set(raw) != expected:
            raise ProtocolError(
                f"command fields must be exactly {sorted(expected)!r}"
            )
        return CommandEnvelope(
            version=raw["version"],
            name=raw["name"],
            request_id=raw["request_id"],
            payload=raw["payload"],
        )
    if envelope_type == "event":
        expected = {
            "version",
            "type",
            "name",
            "sequence",
            "request_id",
            "payload",
        }
        if set(raw) != expected:
            raise ProtocolError(f"event fields must be exactly {sorted(expected)!r}")
        return EventEnvelope(
            version=raw["version"],
            name=raw["name"],
            sequence=raw["sequence"],
            request_id=raw["request_id"],
            payload=raw["payload"],
        )
    raise ProtocolError("type must be 'command' or 'event'")


def decode_envelope(frame: bytes) -> Envelope:
    """Decode exactly one newline-terminated frame."""

    if not isinstance(frame, bytes):
        raise TypeError("frame must be bytes")
    if len(frame) > MAX_FRAME_BYTES:
        raise FrameTooLargeError(
            f"frame is {len(frame)} bytes; maximum is {MAX_FRAME_BYTES}"
        )
    if not frame.endswith(b"\n"):
        raise ProtocolError("frame is missing its newline terminator")
    if b"\n" in frame[:-1]:
        raise ProtocolError("decode_envelope accepts exactly one frame")
    return _decode_body(frame[:-1])


class NewlineFrameDecoder:
    """Incrementally decode frames while bounding the unfinished-frame buffer."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._failed = False

    @property
    def pending_bytes(self) -> int:
        return len(self._buffer)

    def feed(self, data: bytes) -> tuple[Envelope, ...]:
        if self._failed:
            raise ProtocolError("decoder cannot continue after a protocol error")
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        decoded: list[Envelope] = []
        offset = 0
        try:
            while offset < len(data):
                newline = data.find(b"\n", offset)
                if newline < 0:
                    self._buffer.extend(data[offset:])
                    if len(self._buffer) >= MAX_FRAME_BYTES:
                        raise FrameTooLargeError(
                            f"unterminated frame reached {len(self._buffer)} bytes"
                        )
                    break
                self._buffer.extend(data[offset:newline])
                if len(self._buffer) + 1 > MAX_FRAME_BYTES:
                    raise FrameTooLargeError(
                        f"frame exceeds maximum of {MAX_FRAME_BYTES} bytes"
                    )
                decoded.append(_decode_body(bytes(self._buffer)))
                self._buffer.clear()
                offset = newline + 1
        except Exception:
            self._failed = True
            self._buffer.clear()
            raise
        return tuple(decoded)

    def end_of_stream(self) -> None:
        """Validate that peer shutdown did not truncate a frame."""

        if self._buffer:
            self._failed = True
            self._buffer.clear()
            raise ProtocolError("connection closed with an incomplete frame")


class SocketEnvelopeReader:
    """Blocking bounded reader for one connected Unix-domain stream socket."""

    def __init__(self) -> None:
        self._decoder = NewlineFrameDecoder()
        self._ready: deque[Envelope] = deque()

    def receive(self, connection: socket.socket) -> Envelope:
        while not self._ready:
            chunk = connection.recv(SOCKET_READ_BYTES)
            if not chunk:
                self._decoder.end_of_stream()
                raise EOFError("peer closed the IPC connection")
            self._ready.extend(self._decoder.feed(chunk))
        return self._ready.popleft()


class SocketEnvelopeWriter:
    """Thread-safe writer for async events and responses on one connection."""

    def __init__(
        self,
        connection: socket.socket,
        *,
        send_timeout_s: float | None = None,
    ) -> None:
        self._connection = connection
        self._lock = threading.Lock()
        self._send_timeout_s = (
            None if send_timeout_s is None else max(0.001, float(send_timeout_s))
        )

    def _send_with_deadline(self, frame: bytes) -> None:
        """Send one complete frame without inheriting the short read timeout.

        The supervisor reads frequently to enforce its heartbeat lease, while
        outbound delivery may legitimately need the whole lease interval when
        the GUI is momentarily busy. ``MSG_DONTWAIT`` plus ``select`` gives the
        writer its own bounded deadline without changing the socket timeout
        observed concurrently by the reader thread.
        """

        assert self._send_timeout_s is not None
        deadline = time.monotonic() + self._send_timeout_s
        pending = memoryview(frame)
        flags = getattr(socket, "MSG_DONTWAIT", 0)
        while pending:
            try:
                sent = self._connection.send(pending, flags)
            except (BlockingIOError, socket.timeout):
                sent = None
            except InterruptedError:
                continue
            if sent is not None and sent > 0:
                pending = pending[sent:]
                continue
            if sent == 0:
                raise BrokenPipeError("IPC socket closed during frame delivery")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"IPC frame delivery exceeded {self._send_timeout_s:.3f}s"
                )
            _, writable, _ = select.select([], [self._connection], [], remaining)
            if not writable:
                raise TimeoutError(
                    f"IPC frame delivery exceeded {self._send_timeout_s:.3f}s"
                )

    def send(self, envelope: Envelope) -> None:
        # Encoding is pure and may proceed concurrently. Exactly one complete
        # frame is delivered while holding the connection's write lock.
        frame = encode_envelope(envelope)
        with self._lock:
            if self._send_timeout_s is None:
                self._connection.sendall(frame)
            else:
                self._send_with_deadline(frame)


def send_envelope(connection: socket.socket, envelope: Envelope) -> None:
    """Send one frame; concurrent users must share ``SocketEnvelopeWriter``."""

    connection.sendall(encode_envelope(envelope))


__all__ = [
    "MAX_FRAME_BYTES",
    "PROTOCOL_VERSION",
    "CommandEnvelope",
    "Envelope",
    "EventEnvelope",
    "FrameTooLargeError",
    "HEARTBEAT_COMMAND_NAME",
    "HEARTBEAT_REQUEST_ID",
    "JsonValue",
    "MAX_JSON_DEPTH",
    "NewlineFrameDecoder",
    "ProtocolError",
    "SOCKET_READ_BYTES",
    "SocketEnvelopeReader",
    "SocketEnvelopeWriter",
    "decode_envelope",
    "encode_envelope",
    "send_envelope",
]
