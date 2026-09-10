from __future__ import annotations

"""Experimental, firmware-independent host state machine for watchdog v1.

The state machine is transport agnostic so its framing, nonce binding, bounded
waits, heartbeat policy, and fail-close behavior can be tested without opening
hardware.  It is not evidence that matching controller firmware exists.
"""

import re
import secrets
import threading
import time
import math
from dataclasses import dataclass
from typing import Callable


PROTOCOL_VERSION = 1
MIN_LEASE_MS = 100
MAX_LEASE_MS = 60_000
REQUIRED_FEATURES = frozenset(
    {
        "lease_watchdog",
        "trigger_low_failsafe",
        "session_nonce",
        "telemetry_v1",
    }
)
_NONCE_RE = re.compile(r"^[0-9a-f]{32}$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_FEATURE_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_STATES = frozenset({"DISARMED", "ARMED", "WATCHDOG_TRIPPED"})
_REASONS = frozenset(
    {
        "BOOT",
        "ARM_COMMAND",
        "DISARM_COMMAND",
        "LEASE_EXPIRED",
        "USB_DISCONNECT",
        "INTERNAL_WATCHDOG",
        "INVALID_COMMAND",
    }
)
MAX_FEATURES = 32
MAX_SEQUENCE = (1 << 63) - 1
MAX_HOST_WAIT_S = 60.0


def _uint(text: str, label: str, *, maximum: int = MAX_SEQUENCE) -> int:
    if not text.isascii() or not text.isdecimal():
        raise ValueError(f"{label} must be an unsigned decimal integer")
    value = int(text)
    if value > maximum:
        raise ValueError(f"{label} exceeds {maximum}")
    return value


def _nonce(text: str) -> str:
    if _NONCE_RE.fullmatch(text) is None:
        raise ValueError("controller session nonce must be 32 lowercase hexadecimal characters")
    return text


def _host_wait_seconds(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("controller host wait must be a finite number")
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("controller host wait must be a finite number") from exc
    if not math.isfinite(timeout) or not 0.0 <= timeout <= MAX_HOST_WAIT_S:
        raise ValueError(
            f"controller host wait must be between 0 and {MAX_HOST_WAIT_S:g} seconds"
        )
    return timeout


@dataclass(frozen=True, slots=True)
class ControllerCapabilities:
    protocol_version: int
    session_nonce: str
    firmware_version: str
    watchdog_timeout_ms: int
    features: frozenset[str]


@dataclass(frozen=True, slots=True)
class ControllerAck:
    kind: str
    protocol_version: int
    session_nonce: str
    sequence: int
    controller_monotonic_us: int
    ttl_count: int | None = None


@dataclass(frozen=True, slots=True)
class ControllerState:
    protocol_version: int
    session_nonce: str
    state: str
    controller_monotonic_us: int
    ttl_count: int
    reason: str


def parse_capabilities(line: str) -> ControllerCapabilities | None:
    if not line.startswith("CONTROLLER_CAPS,"):
        return None
    fields = line.split(",")
    if len(fields) != 6:
        raise ValueError("CONTROLLER_CAPS must contain exactly six CSV fields")
    _, version, nonce, firmware, timeout, raw_features = fields
    protocol_version = _uint(version, "controller protocol version", maximum=255)
    session_nonce = _nonce(nonce)
    if _TOKEN_RE.fullmatch(firmware) is None:
        raise ValueError("controller firmware version is invalid")
    watchdog_timeout_ms = _uint(timeout, "controller watchdog timeout", maximum=MAX_LEASE_MS)
    if watchdog_timeout_ms < MIN_LEASE_MS:
        raise ValueError(
            f"controller watchdog timeout must be between {MIN_LEASE_MS} and {MAX_LEASE_MS} ms"
        )
    items = raw_features.split(";") if raw_features else []
    if len(items) > MAX_FEATURES:
        raise ValueError("controller capability list is too large")
    if any(_FEATURE_RE.fullmatch(item) is None for item in items):
        raise ValueError("controller capability list contains an invalid feature")
    if len(set(items)) != len(items):
        raise ValueError("controller capability list contains a duplicate feature")
    return ControllerCapabilities(
        protocol_version,
        session_nonce,
        firmware,
        watchdog_timeout_ms,
        frozenset(items),
    )


def parse_ack(line: str) -> ControllerAck | None:
    prefix = line.split(",", 1)[0]
    if prefix not in {"ACK_ARM", "ACK_HEARTBEAT", "ACK_DISARM"}:
        return None
    fields = line.split(",")
    expected = 6 if prefix in {"ACK_HEARTBEAT", "ACK_DISARM"} else 5
    if len(fields) != expected:
        raise ValueError(f"{prefix} must contain exactly {expected} CSV fields")
    protocol_version = _uint(fields[1], "ack protocol version", maximum=255)
    session_nonce = _nonce(fields[2])
    sequence = _uint(fields[3], "ack sequence")
    controller_us = _uint(fields[4], "controller monotonic clock")
    ttl_count = None
    if prefix == "ACK_HEARTBEAT":
        ttl_count = _uint(fields[5], "heartbeat TTL count")
    elif prefix == "ACK_DISARM":
        ttl_count = _uint(fields[5], "disarm TTL count")
    return ControllerAck(
        kind=prefix.removeprefix("ACK_"),
        protocol_version=protocol_version,
        session_nonce=session_nonce,
        sequence=sequence,
        controller_monotonic_us=controller_us,
        ttl_count=ttl_count,
    )


def parse_state(line: str) -> ControllerState | None:
    if not line.startswith("CONTROLLER_STATE,"):
        return None
    fields = line.split(",")
    if len(fields) != 7:
        raise ValueError("CONTROLLER_STATE must contain exactly seven CSV fields")
    _, version, nonce, state, controller_us, ttl_count, reason = fields
    if state not in _STATES:
        raise ValueError("controller state is not a v1 enumerated state")
    if reason not in _REASONS:
        raise ValueError("controller state reason is not a v1 enumerated reason")
    return ControllerState(
        _uint(version, "controller state protocol version", maximum=255),
        _nonce(nonce),
        state,
        _uint(controller_us, "controller state monotonic clock"),
        _uint(ttl_count, "controller state TTL count"),
        reason,
    )


class ControllerWatchdogSession:
    """One nonce-bound watchdog session with a single heartbeat worker."""

    def __init__(
        self,
        send_line: Callable[[str], None],
        on_fatal: Callable[[str], None],
        *,
        requested_lease_ms: int,
        nonce: str | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if isinstance(requested_lease_ms, bool) or not isinstance(requested_lease_ms, int):
            raise ValueError("requested watchdog lease must be an integer")
        if not MIN_LEASE_MS <= requested_lease_ms <= MAX_LEASE_MS:
            raise ValueError(
                f"requested watchdog lease must be between {MIN_LEASE_MS} and {MAX_LEASE_MS} ms"
            )
        self.send_line = send_line
        self.on_fatal = on_fatal
        self.requested_lease_ms = requested_lease_ms
        self.nonce = _nonce(nonce if nonce is not None else secrets.token_hex(16))
        self.monotonic_ns = monotonic_ns
        self.capabilities: ControllerCapabilities | None = None
        self.final_ttl_count: int | None = None
        self._condition = threading.Condition()
        self._acks: dict[tuple[str, int], ControllerAck] = {}
        self._expected_ack: tuple[str, int] | None = None
        self._state: ControllerState | None = None
        self._sequence = 0
        self._armed = False
        self._disarming = False
        self._stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._fatal = False
        self._fatal_reason: str | None = None
        self._heartbeats_sent = 0
        self._heartbeats_acked = 0

    def snapshot(self) -> dict[str, object]:
        caps = self.capabilities
        with self._condition:
            return {
                "mode": "watchdog_v1_experimental",
                "protocol_version": PROTOCOL_VERSION,
                "session_nonce": self.nonce,
                "requested_lease_ms": self.requested_lease_ms,
                "firmware_version": caps.firmware_version if caps else None,
                "negotiated_timeout_ms": caps.watchdog_timeout_ms if caps else None,
                "features": sorted(caps.features) if caps else [],
                "armed": self._armed,
                "final_ttl_count": self.final_ttl_count,
                "final_state": self._state.state if self._state else None,
                "final_state_reason": self._state.reason if self._state else None,
                "heartbeats_sent": self._heartbeats_sent,
                "heartbeats_acked": self._heartbeats_acked,
                "fatal_reason": self._fatal_reason,
                "host_implementation_qualified": False,
                "firmware_qualified": False,
            }

    def ingest(self, line: str) -> bool:
        """Consume a protocol reply; return False for unrelated legacy telemetry."""

        try:
            caps = parse_capabilities(line)
            ack = parse_ack(line)
            state = parse_state(line)
        except ValueError as exc:
            if line.startswith(
                (
                    "CONTROLLER_CAPS,",
                    "ACK_ARM,",
                    "ACK_HEARTBEAT,",
                    "ACK_DISARM,",
                    "CONTROLLER_STATE,",
                )
            ):
                self._fail(f"malformed watchdog controller reply: {exc}")
                return True
            return False
        if caps is not None:
            with self._condition:
                if caps.session_nonce == self.nonce:
                    self.capabilities = caps
                    self._condition.notify_all()
            return True
        if ack is not None:
            with self._condition:
                key = (ack.kind, ack.sequence)
                if ack.session_nonce == self.nonce and key == self._expected_ack:
                    self._acks.clear()
                    self._acks[key] = ack
                    self._condition.notify_all()
            return True
        if state is not None:
            unexpected = False
            with self._condition:
                if state.session_nonce == self.nonce:
                    self._state = state
                    unexpected = state.state == "WATCHDOG_TRIPPED" or (
                        state.state == "DISARMED" and self._armed and not self._disarming
                    )
                    self._condition.notify_all()
            if unexpected:
                self._fail(
                    "controller reported an unsafe state while armed: "
                    f"{state.state} ({state.reason})"
                )
            return True
        return False

    def negotiate(self, *, timeout_s: float) -> ControllerCapabilities:
        timeout_s = _host_wait_seconds(timeout_s)
        self.send_line(
            f"HELLO,{PROTOCOL_VERSION},{self.nonce},{self.requested_lease_ms}"
        )
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self.capabilities is None and not self._fatal:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            caps = self.capabilities
        if caps is None:
            raise RuntimeError("watchdog capability negotiation timed out")
        if caps.protocol_version != PROTOCOL_VERSION:
            raise RuntimeError(f"unsupported controller watchdog protocol {caps.protocol_version}")
        missing = REQUIRED_FEATURES - caps.features
        if missing:
            raise RuntimeError(
                "controller is missing required watchdog features: " + ", ".join(sorted(missing))
            )
        if caps.watchdog_timeout_ms > self.requested_lease_ms:
            raise RuntimeError("controller watchdog timeout exceeds the requested lease")
        return caps

    def arm(self, fps: int, *, timeout_s: float) -> ControllerAck:
        timeout_s = _host_wait_seconds(timeout_s)
        if self.capabilities is None:
            raise RuntimeError("watchdog session was not negotiated")
        if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
            raise ValueError("controller FPS must be a positive integer")
        sequence = self._next_sequence()
        self._send_expecting_ack(
            "ARM",
            sequence,
            f"ARM,{PROTOCOL_VERSION},{self.nonce},{fps},{sequence}",
        )
        ack = self._wait_ack("ARM", sequence, timeout_s)
        with self._condition:
            self._armed = True
        self._stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="squeakview-controller-heartbeat",
        )
        self._heartbeat_thread.start()
        return ack

    def disarm(self, *, timeout_s: float) -> ControllerAck:
        timeout_s = _host_wait_seconds(timeout_s)
        self._request_stop()
        thread = self._heartbeat_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout_s)
        sequence = self._next_sequence()
        with self._condition:
            self._state = None
            self._disarming = True
        self._send_expecting_ack(
            "DISARM",
            sequence,
            f"DISARM,{PROTOCOL_VERSION},{self.nonce},{sequence}",
        )
        # A heartbeat/protocol fault starts ordered shutdown, but it is not a
        # reason to abandon the bounded wait for independent inactive-state
        # evidence.  The serial reader deliberately remains alive for this.
        ack = self._wait_ack(
            "DISARM",
            sequence,
            timeout_s,
            abort_on_fatal=False,
            abort_on_stop=False,
        )
        with self._condition:
            deadline = time.monotonic() + timeout_s
            while self._state is None or self._state.state != "DISARMED":
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            state = self._state
            if state is None or state.state != "DISARMED":
                raise RuntimeError("controller did not confirm DISARMED state")
            if state.ttl_count != ack.ttl_count:
                raise RuntimeError("DISARM acknowledgement and state TTL counts disagree")
            self._armed = False
            self._disarming = False
            self.final_ttl_count = ack.ttl_count
        return ack

    def stop_worker(self) -> None:
        self._request_stop()
        thread = self._heartbeat_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def _next_sequence(self) -> int:
        with self._condition:
            if self._sequence >= MAX_SEQUENCE:
                raise RuntimeError("controller sequence space exhausted")
            self._sequence += 1
            return self._sequence

    def _wait_ack(
        self,
        kind: str,
        sequence: int,
        timeout_s: float,
        *,
        abort_on_fatal: bool = True,
        abort_on_stop: bool = True,
    ) -> ControllerAck:
        timeout_s = _host_wait_seconds(timeout_s)
        deadline = time.monotonic() + timeout_s
        key = (kind, sequence)
        with self._condition:
            while (
                key not in self._acks
                and (not self._fatal or not abort_on_fatal)
                and (not self._stop.is_set() or not abort_on_stop)
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(remaining)
            ack = self._acks.pop(key, None)
            if self._expected_ack == key:
                self._expected_ack = None
        if ack is None:
            raise RuntimeError(f"ACK_{kind} was not received for sequence {sequence}")
        if ack.protocol_version != PROTOCOL_VERSION:
            raise RuntimeError(f"ACK_{kind} used unsupported protocol version")
        return ack

    def _request_stop(self) -> None:
        """Wake an acknowledgement waiter as well as setting the worker flag."""

        with self._condition:
            self._stop.set()
            self._condition.notify_all()

    def _send_expecting_ack(
        self, kind: str, sequence: int, command: str
    ) -> None:
        key = (kind, sequence)
        with self._condition:
            if self._expected_ack is not None:
                raise RuntimeError("another controller acknowledgement is still pending")
            self._acks.clear()
            self._expected_ack = key
        try:
            self.send_line(command)
        except Exception:
            with self._condition:
                if self._expected_ack == key:
                    self._expected_ack = None
            raise

    def _heartbeat_loop(self) -> None:
        assert self.capabilities is not None
        interval_s = self.capabilities.watchdog_timeout_ms / 3000.0
        missed = 0
        next_send = time.monotonic() + interval_s
        while not self._stop.wait(max(0.0, next_send - time.monotonic())):
            try:
                sequence = self._next_sequence()
                self._send_expecting_ack(
                    "HEARTBEAT",
                    sequence,
                    f"HEARTBEAT,{PROTOCOL_VERSION},{self.nonce},{sequence},{self.monotonic_ns()}",
                )
                with self._condition:
                    self._heartbeats_sent += 1
                # Anchor the next renewal to the actual write.  In particular,
                # an acknowledgement timeout already consumes one interval and
                # must not be followed by another interval-long sleep: that
                # would place the next renewal at the lease deadline.
                next_send = time.monotonic() + interval_s
            except Exception as exc:
                if not self._stop.is_set():
                    self._fail(f"controller heartbeat write failed: {exc}")
                return
            try:
                self._wait_ack(
                    "HEARTBEAT",
                    sequence,
                    max(0.0, next_send - time.monotonic()),
                )
                with self._condition:
                    self._heartbeats_acked += 1
                missed = 0
            except Exception as exc:
                missed += 1
                if missed >= 2 and not self._stop.is_set():
                    self._fail(f"controller heartbeat failed twice: {exc}")
                    return

    def _fail(self, message: str) -> None:
        with self._condition:
            if self._fatal:
                return
            self._fatal = True
            self._fatal_reason = message
            self._stop.set()
            self._condition.notify_all()
        self.on_fatal(message)


__all__ = [
    "ControllerAck",
    "ControllerCapabilities",
    "ControllerWatchdogSession",
    "ControllerState",
    "MAX_LEASE_MS",
    "MAX_HOST_WAIT_S",
    "MIN_LEASE_MS",
    "PROTOCOL_VERSION",
    "REQUIRED_FEATURES",
    "parse_ack",
    "parse_capabilities",
    "parse_state",
]
