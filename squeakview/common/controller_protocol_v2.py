from __future__ import annotations

"""MouseHouse controller protocol-v2 framing and durable journal support."""

import json
import os
import struct
import threading
import zlib
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any


VERSION = 2
HEADER = struct.Struct("<BBHQIQQH")
HEADER_SIZE = HEADER.size
CRC = struct.Struct("<I")
MAX_PAYLOAD_SIZE = 384
MAX_DECODED_SIZE = 422
MAX_ENCODED_SIZE = 425

FLAG_RELIABLE = 0x0001
FLAG_RETRANSMISSION = 0x0002
FLAG_INTEGRITY_LATCHED = 0x0004


class MessageType(IntEnum):
    EVENT = 1
    COMMAND_RESULT = 2
    CAMERA_EPOCH = 3
    CAMERA_CHECKPOINT = 4
    CAMERA_STOP = 5
    TRANSPORT_STATUS = 6
    INTEGRITY_FAULT = 7
    DIAGNOSTIC = 8


class ProtocolV2Error(ValueError):
    """A controller frame is malformed or violates the v2 contract."""


@dataclass(frozen=True, slots=True)
class Frame:
    message_type: int
    flags: int
    boot_id: int
    session_id: int
    sequence: int
    monotonic_us: int
    payload: bytes
    decoded: bytes

    @property
    def reliable(self) -> bool:
        return bool(self.flags & FLAG_RELIABLE)

    @property
    def retransmission(self) -> bool:
        return bool(self.flags & FLAG_RETRANSMISSION)

    @property
    def integrity_latched(self) -> bool:
        return bool(self.flags & FLAG_INTEGRITY_LATCHED)

    @property
    def text(self) -> str:
        try:
            return self.payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ProtocolV2Error(f"payload is not valid UTF-8: {exc}") from exc

    def identity(self) -> tuple[object, ...]:
        """Identity that deliberately ignores the retransmission flag and CRC."""

        return (
            self.message_type,
            self.flags & ~FLAG_RETRANSMISSION,
            self.session_id,
            self.sequence,
            self.monotonic_us,
            self.payload,
        )


def crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def cobs_decode(encoded: bytes) -> bytes:
    if not encoded:
        raise ProtocolV2Error("empty COBS frame")
    output = bytearray()
    index = 0
    while index < len(encoded):
        code = encoded[index]
        if code == 0:
            raise ProtocolV2Error("zero byte inside COBS body")
        index += 1
        end = index + code - 1
        if end > len(encoded):
            raise ProtocolV2Error("COBS code exceeds encoded frame")
        output.extend(encoded[index:end])
        index = end
        if code != 0xFF and index < len(encoded):
            output.append(0)
    return bytes(output)


def cobs_encode(decoded: bytes) -> bytes:
    """Reference encoder used by tests and diagnostic tooling."""

    output = bytearray(b"\x00")
    code_index = 0
    code = 1
    for byte in decoded:
        if byte == 0:
            output[code_index] = code
            code_index = len(output)
            output.append(0)
            code = 1
        else:
            output.append(byte)
            code += 1
            if code == 0xFF:
                output[code_index] = code
                code_index = len(output)
                output.append(0)
                code = 1
    output[code_index] = code
    output.append(0)
    return bytes(output)


def decode_frame(encoded: bytes) -> Frame:
    if len(encoded) + 1 > MAX_ENCODED_SIZE:
        raise ProtocolV2Error(
            f"encoded frame exceeds {MAX_ENCODED_SIZE} bytes including delimiter"
        )
    decoded = cobs_decode(encoded)
    if len(decoded) < HEADER_SIZE + CRC.size:
        raise ProtocolV2Error("decoded frame is shorter than header plus CRC")
    if len(decoded) > MAX_DECODED_SIZE:
        raise ProtocolV2Error(f"decoded frame exceeds {MAX_DECODED_SIZE} bytes")
    (
        version,
        message_type,
        flags,
        boot_id,
        session_id,
        sequence,
        monotonic_us,
        payload_length,
    ) = HEADER.unpack_from(decoded)
    if version != VERSION:
        raise ProtocolV2Error(f"unsupported protocol version {version}")
    if payload_length > MAX_PAYLOAD_SIZE:
        raise ProtocolV2Error(
            f"payload length {payload_length} exceeds {MAX_PAYLOAD_SIZE}"
        )
    expected = HEADER_SIZE + payload_length + CRC.size
    if len(decoded) != expected:
        raise ProtocolV2Error(
            f"decoded length {len(decoded)} does not match declared length {expected}"
        )
    expected_crc = CRC.unpack_from(decoded, HEADER_SIZE + payload_length)[0]
    actual_crc = crc32(decoded[: HEADER_SIZE + payload_length])
    if actual_crc != expected_crc:
        raise ProtocolV2Error(
            f"CRC mismatch: expected 0x{expected_crc:08x}, calculated 0x{actual_crc:08x}"
        )
    payload = decoded[HEADER_SIZE : HEADER_SIZE + payload_length]
    return Frame(
        message_type=message_type,
        flags=flags,
        boot_id=boot_id,
        session_id=session_id,
        sequence=sequence,
        monotonic_us=monotonic_us,
        payload=payload,
        decoded=decoded,
    )


def encode_frame(
    *,
    message_type: int,
    flags: int,
    boot_id: int,
    session_id: int,
    sequence: int,
    monotonic_us: int,
    payload: bytes,
) -> bytes:
    if len(payload) > MAX_PAYLOAD_SIZE:
        raise ValueError(f"payload exceeds {MAX_PAYLOAD_SIZE} bytes")
    header = HEADER.pack(
        VERSION,
        message_type,
        flags,
        boot_id,
        session_id,
        sequence,
        monotonic_us,
        len(payload),
    )
    body = header + payload
    return cobs_encode(body + CRC.pack(crc32(body)))


class StreamDecoder:
    """Incremental zero-delimited COBS decoder with bounded fragment storage."""

    def __init__(self) -> None:
        self._fragment = bytearray()
        self._discarding = False

    @property
    def has_partial_frame(self) -> bool:
        return bool(self._fragment) or self._discarding

    def discard_through_delimiter(self) -> None:
        self._fragment.clear()
        self._discarding = True

    def reset(self) -> None:
        self._fragment.clear()
        self._discarding = False

    def feed(self, data: bytes) -> tuple[list[Frame], list[str]]:
        frames: list[Frame] = []
        errors: list[str] = []
        for byte in data:
            if self._discarding:
                if byte == 0:
                    self._discarding = False
                continue
            if byte == 0:
                if not self._fragment:
                    continue
                encoded = bytes(self._fragment)
                self._fragment.clear()
                try:
                    frames.append(decode_frame(encoded))
                except ProtocolV2Error as exc:
                    errors.append(str(exc))
                continue
            if len(self._fragment) >= MAX_ENCODED_SIZE - 1:
                errors.append(
                    f"unterminated encoded frame exceeded {MAX_ENCODED_SIZE - 1} bytes"
                )
                self._fragment.clear()
                self._discarding = True
                continue
            self._fragment.append(byte)
        return frames, errors


def parse_camera_payload(text: str) -> dict[str, str]:
    fields = text.split(",")
    if not fields or fields[0] not in {
        "CAMERA_EPOCH",
        "CAMERA_CHECKPOINT",
        "CAMERA_STOP",
    }:
        raise ProtocolV2Error(f"unrecognized camera payload: {text!r}")
    parsed = {"record": fields[0]}
    for field in fields[1:]:
        if "=" not in field:
            raise ProtocolV2Error(f"malformed camera field {field!r}")
        key, value = field.split("=", 1)
        if not key or key in parsed:
            raise ProtocolV2Error(f"duplicate or empty camera key {key!r}")
        parsed[key] = value
    required = {
        "count",
        "timestamp_us",
        "period_us",
        "pulse_us",
        "health",
        "queue",
        "suppressed",
        "reason",
    }
    missing = required.difference(parsed)
    if missing:
        raise ProtocolV2Error(f"camera payload missing {sorted(missing)}")
    return parsed


TRANSPORT_STATUS_KEYS = (
    "b",
    "m",
    "n",
    "q",
    "h",
    "o",
    "a",
    "t",
    "k",
    "x",
    "l",
    "u",
    "c",
    "p",
    "z",
    "r",
    "v",
    "f",
    "d",
)


def parse_transport_status(text: str) -> dict[str, str]:
    fields = text.split(",")
    if not fields or fields[0] != "TRANSPORT_STATUS":
        raise ProtocolV2Error(f"unrecognized transport status: {text!r}")
    parsed: dict[str, str] = {}
    for field in fields[1:]:
        if "=" not in field:
            raise ProtocolV2Error(f"malformed transport status field {field!r}")
        key, value = field.split("=", 1)
        if key in parsed:
            raise ProtocolV2Error(f"duplicate transport status key {key!r}")
        parsed[key] = value
    if tuple(parsed) != TRANSPORT_STATUS_KEYS:
        raise ProtocolV2Error(
            "transport status keys/order differ from firmware specification"
        )
    return parsed


class StoreResult(IntEnum):
    NEW = 1
    DUPLICATE = 2
    CONFLICT = 3


class DurableJournal:
    """Append-only fsync journal indexed by the reliable record identity."""

    def __init__(self, path: Path, *, allow_midstream_start: bool = False):
        self.path = Path(path)
        self.allow_midstream_start = bool(allow_midstream_start)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._identities: dict[tuple[int, int], tuple[object, ...]] = {}
        self._sequences: dict[int, set[int]] = {}
        self._contiguous: dict[int, int] = {}
        self._load()
        self._file = open(self.path, "a", encoding="utf-8", buffering=1)

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                try:
                    record = json.loads(line)
                    decoded = bytes.fromhex(record["decoded_hex"])
                    payload_length = int.from_bytes(decoded[32:34], "little")
                    frame = Frame(
                        message_type=int(record["message_type"]),
                        flags=int(record["flags"]),
                        boot_id=int(record["boot_id"]),
                        session_id=int(record["session_id"]),
                        sequence=int(record["sequence"]),
                        monotonic_us=int(record["monotonic_us"]),
                        payload=decoded[34 : 34 + payload_length],
                        decoded=decoded,
                    )
                except Exception as exc:
                    raise ProtocolV2Error(
                        f"invalid durable journal line {line_number}: {exc}"
                    ) from exc
                self._remember(frame)
        for boot_id in self._sequences:
            if self.allow_midstream_start and self._sequences[boot_id]:
                self._contiguous[boot_id] = min(self._sequences[boot_id]) - 1
            self._advance_contiguous(boot_id)

    def _remember(self, frame: Frame) -> None:
        key = (frame.boot_id, frame.sequence)
        self._identities[key] = frame.identity()
        self._sequences.setdefault(frame.boot_id, set()).add(frame.sequence)

    def _advance_contiguous(self, boot_id: int) -> int:
        sequences = self._sequences.setdefault(boot_id, set())
        value = self._contiguous.get(boot_id, 0)
        while value + 1 in sequences:
            value += 1
        self._contiguous[boot_id] = value
        return value

    def store(
        self, frame: Frame, *, host_unix_ns: int, host_monotonic_ns: int
    ) -> tuple[StoreResult, int]:
        if frame.reliable and frame.sequence == 0:
            raise ProtocolV2Error("reliable record has sequence zero")
        key = (frame.boot_id, frame.sequence)
        with self._lock:
            previous = self._identities.get(key)
            if previous is not None:
                result = (
                    StoreResult.DUPLICATE
                    if previous == frame.identity()
                    else StoreResult.CONFLICT
                )
                return result, self._contiguous.get(frame.boot_id, 0)
            if (
                self.allow_midstream_start
                and frame.boot_id not in self._sequences
                and frame.reliable
            ):
                # Sequence numbers are boot-global while this journal is
                # run-local. The first complete record received by a new run is
                # therefore an explicit continuity baseline, not evidence that
                # earlier, already-acknowledged runs are missing locally.
                self._contiguous[frame.boot_id] = frame.sequence - 1
            record: dict[str, Any] = {
                "schema_version": 1,
                "host_unix_ns": host_unix_ns,
                "host_monotonic_ns": host_monotonic_ns,
                "message_type": frame.message_type,
                "flags": frame.flags,
                "boot_id": frame.boot_id,
                "session_id": frame.session_id,
                "sequence": frame.sequence,
                "monotonic_us": frame.monotonic_us,
                "payload_utf8": frame.text,
                "decoded_hex": frame.decoded.hex(),
            }
            self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._file.flush()
            os.fsync(self._file.fileno())
            self._remember(frame)
            return StoreResult.NEW, self._advance_contiguous(frame.boot_id)

    def contiguous(self, boot_id: int) -> int:
        with self._lock:
            return self._contiguous.get(boot_id, 0)

    def start_sequence(self, boot_id: int) -> int | None:
        """Return the first sequence represented by this run-local journal."""

        with self._lock:
            sequences = self._sequences.get(boot_id)
            return min(sequences) if sequences else None

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.flush()
                os.fsync(self._file.fileno())
                self._file.close()


__all__ = [
    "DurableJournal",
    "FLAG_INTEGRITY_LATCHED",
    "FLAG_RELIABLE",
    "FLAG_RETRANSMISSION",
    "Frame",
    "HEADER_SIZE",
    "MAX_DECODED_SIZE",
    "MAX_ENCODED_SIZE",
    "MAX_PAYLOAD_SIZE",
    "MessageType",
    "ProtocolV2Error",
    "StoreResult",
    "StreamDecoder",
    "cobs_decode",
    "cobs_encode",
    "crc32",
    "decode_frame",
    "encode_frame",
    "parse_camera_payload",
    "parse_transport_status",
]
