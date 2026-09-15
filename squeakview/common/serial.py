from __future__ import annotations

"""Serial helpers shared across capture/operator layers."""

import atexit
import csv
import os
import smtplib
import ssl
import tempfile
import threading
import time
import uuid
from email.message import EmailMessage
from pathlib import Path
from typing import Callable, Iterable

from squeakview.common.failure_injection import FailurePlan
from squeakview.common.controller_watchdog import (
    ControllerCapabilities,
    ControllerWatchdogSession,
    parse_capabilities as parse_controller_capabilities,
)

try:
    import serial  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    serial = None

SERIAL_HEADER = [
    "eventType",
    "unixTime",
    "rp2040Time",
    "side",
    "count",
    "duration",
    "latency",
    "value",
    "context",
    "reason",
    "hostUnixNs",
    "hostMonotonicNs",
    "rawLine",
]
MAX_SERIAL_LINE_BYTES = 64 * 1024
MAX_BUFFERED_SERIAL_ROWS = 1024
CLEAR_JAM_ACK = "ACK_CLEAR_JAM"
CLEAR_JAM_NACK_FEED_ACTIVE = "NACK,CLEAR_JAM,FEED_ACTIVE"
CLEAR_JAM_NACK_NOT_JAMMED = "NACK,CLEAR_JAM,NOT_JAMMED"
_CLEAR_JAM_RESPONSES = frozenset(
    {
        CLEAR_JAM_ACK,
        CLEAR_JAM_NACK_FEED_ACTIVE,
        CLEAR_JAM_NACK_NOT_JAMMED,
    }
)


def _is_feeder_jam_protocol_line(line: str) -> bool:
    """Keep latch state observable when routine serial logging is disabled."""

    return (
        line == "FEED_JAM"
        or line.startswith("FEED_JAM,")
        or line == "NACK,FEED,JAMMED"
        or line in _CLEAR_JAM_RESPONSES
    )


def have_pyserial() -> bool:
    return serial is not None


def timestamp() -> str:
    return time.strftime("%H:%M:%S")


class SerialHandle:
    """Threaded serial reader/writer for Arduino telemetry."""

    def __init__(
        self,
        port: str,
        baud: int,
        emit_fn: Callable[[str], None],
        on_fatal: Callable[[str], None] | None = None,
        failure_plan: FailurePlan | None = None,
    ):
        self.emit = emit_fn
        self.on_fatal = on_fatal
        self.failure_plan = (
            failure_plan
            if failure_plan is not None
            and failure_plan.target == "serial_controller"
            else None
        )
        self.port = port
        self.baud = baud
        self.ser = None
        self.last_error: str | None = None
        self._fatal_error: str | None = None
        self._fatal_lock = threading.Lock()
        self._capabilities_lock = threading.Lock()
        self._controller_capabilities: ControllerCapabilities | None = None
        self._watchdog_session: ControllerWatchdogSession | None = None
        self._write_lock = threading.Lock()
        self._ttl_lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self._ttl_seen = threading.Event()
        self._stop_ack_seen = threading.Event()
        self._stop_ack_count: int | None = None
        self._clear_jam_lock = threading.Lock()
        self._clear_jam_pending = False
        self._clear_jam_response_seen = threading.Event()
        self._clear_jam_response: str | None = None
        self._clear_jam_error: str | None = None
        self._csv_lock = threading.Lock()
        self._csv_writer: csv.writer | None = None
        self._csv_file = None
        self._buffer_rows: list[list[str]] = []
        self._max_buffered_rows = MAX_BUFFERED_SERIAL_ROWS
        self._max_line_bytes = MAX_SERIAL_LINE_BYTES
        self._csv_ready = False
        self._tmp_csv_path: str | None = None
        self._tmp_opened = False
        self._flush_every = 25
        self._row_count = 0
        self._read_count = 0
        self._write_count = 0
        self._closed = True
        # Allow silencing serial logs in the terminal; still record CSV.
        self._emit_serial_logs = os.environ.get("SQUEAKVIEW_SERIAL_LOG", "1") != "0"
        # Alert phrase and state for optional email notifications.
        self._alert_phrase = (os.environ.get("SQUEAKVIEW_SERIAL_ALERT_PHRASE") or "Feeder jammed").strip()
        self._alert_warned = False
        self._alert_lock = threading.Lock()
        self._alert_inflight = False
        self._alert_suppressed = 0
        self._atexit_registered = False
        self._register_atexit()

    def _register_atexit(self) -> None:
        if self._atexit_registered:
            return
        atexit.register(self.close)
        self._atexit_registered = True

    def _unregister_atexit(self) -> None:
        if not self._atexit_registered:
            return
        atexit.unregister(self.close)
        self._atexit_registered = False

    @property
    def fatal_error(self) -> str | None:
        """Return the first runtime integrity failure, if one occurred."""

        with self._fatal_lock:
            return self._fatal_error

    @property
    def controller_capabilities(self) -> ControllerCapabilities | None:
        """Latest valid passive capability advertisement, if one was observed."""

        with self._capabilities_lock:
            return self._controller_capabilities

    def _report_fatal(self, message: str) -> None:
        """Latch and publish one fatal serial-integrity failure.

        The callback runs outside all serial and CSV locks so a lifecycle owner
        may safely initiate shutdown.  Later errors remain secondary and cannot
        replace the first causal failure.
        """

        self._latch_fatal(message, stop_reader=True)

    def _report_watchdog_fatal(self, message: str) -> None:
        """Fail the run while preserving the reader for an ordered DISARM.

        Protocol/heartbeat failure is not itself proof that reads are broken.
        Keeping the reader alive gives finalization a chance to persist the
        DISARM acknowledgement and inactive-state evidence.  A concrete reader
        failure still uses ``_report_fatal`` and stops the pump immediately.
        """

        self._latch_fatal(message, stop_reader=False)

    def _latch_fatal(self, message: str, *, stop_reader: bool) -> None:
        detail = str(message).strip() or "unknown serial integrity failure"
        with self._fatal_lock:
            if self._fatal_error is not None:
                return
            self._fatal_error = detail
            self.last_error = detail
        if stop_reader:
            self._stop.set()
        self._interrupt_clear_jam(detail)
        self.emit(f"[{timestamp()}] [SER] FATAL: {detail}")
        if self.on_fatal is not None:
            try:
                self.on_fatal(detail)
            except Exception as exc:
                self.emit(f"[{timestamp()}] [SER] fatal callback error: {exc}")

    def _open_csv(self, path: Path) -> None:
        """Open CSV at the given path and flush any buffered lines."""
        try:
            is_empty = (not path.exists()) or path.stat().st_size == 0
        except Exception:
            is_empty = True
        f = open(path, "a", newline="", buffering=1)
        writer = csv.writer(f)
        if is_empty:
            writer.writerow(SERIAL_HEADER)
        with self._csv_lock:
            self._row_count = 0
            self._csv_file = f
            self._csv_writer = writer
            self._csv_ready = True
            if self._buffer_rows:
                self._csv_writer.writerows(self._buffer_rows)
                self._row_count += len(self._buffer_rows)
                self._buffer_rows.clear()
            self._csv_file.flush()

    def _open_temp_csv(self) -> None:
        basename = f"serial_{int(time.time())}_{uuid.uuid4().hex[:6]}.csv"
        self._tmp_csv_path = str(Path(tempfile.gettempdir()) / basename)
        self.emit(f"[{timestamp()}] [SER] Temp CSV opened: {self._tmp_csv_path}")
        self._open_csv(Path(self._tmp_csv_path))
        self._tmp_opened = True

    def set_csv_path(self, run_dir: Path) -> bool:
        """Adopt the temp CSV into the official run dir by atomic rename."""
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            dst_path = run_dir / "serial.csv"

            with self._csv_lock:
                if self._csv_file:
                    self._csv_file.flush()
                    self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None
                self._csv_ready = False

            if self._tmp_opened and self._tmp_csv_path and Path(self._tmp_csv_path).exists():
                try:
                    Path(self._tmp_csv_path).replace(dst_path)
                    self.emit(f"[{timestamp()}] [SER] Moved temp CSV → {dst_path}")
                except Exception as exc:
                    self.emit(f"[{timestamp()}] [SER] Move failed ({exc}); will append directly to {dst_path}")

            self._open_csv(dst_path)

            self._tmp_opened = False
            self._tmp_csv_path = None
            return True
        except Exception as exc:  # pragma: no cover - filesystem edge cases
            self._report_fatal(f"serial CSV setup failed for {run_dir}: {exc}")
            return False

    def open(self, run_dir: Path | None = None) -> bool:
        # A handle may be closed and deliberately reused.  Keep emergency
        # interpreter cleanup active only while the object owns resources;
        # otherwise every completed run remains strongly held by ``atexit``.
        self._register_atexit()
        self.last_error = None
        with self._fatal_lock:
            self._fatal_error = None
        with self._capabilities_lock:
            self._controller_capabilities = None
        if serial is None:
            self.last_error = "pyserial is not installed"
            self.emit(f"[{timestamp()}] [SER] pyserial not installed.")
            self._unregister_atexit()
            return False
        try:
            self.emit(f"[{timestamp()}] [SER] Opening {self.port} @ {self.baud} …")
            self.ser = serial.Serial(
                self.port,
                self.baud,
                timeout=0.05,
                write_timeout=1.0,
            )
            self._closed = False
            self._stop.clear()
            with self._ttl_lock:
                self._ttl_seen.clear()
            self._stop_ack_seen.clear()
            self._stop_ack_count = None
            with self._clear_jam_lock:
                self._clear_jam_pending = False
                self._clear_jam_response = None
                self._clear_jam_error = None
                self._clear_jam_response_seen.clear()
            if run_dir is not None:
                if not self.set_csv_path(run_dir):
                    raise OSError(self.fatal_error or "serial CSV setup failed")
            else:
                self._open_temp_csv()
            self._thread = threading.Thread(target=self._pump, daemon=True)
            self._thread.start()
            return True
        except Exception as exc:
            self.last_error = str(exc) or type(exc).__name__
            self.emit(f"[{timestamp()}] [SER] ERROR opening serial: {exc}")
            try:
                if self.ser and getattr(self.ser, "is_open", False):
                    self.ser.close()
            except Exception:
                pass
            self.ser = None
            self._closed = True
            self._unregister_atexit()
            return False

    def _pump(self) -> None:
        failure: str | None = None
        try:
            buf = b""
            while not self._stop.is_set():
                ser = self.ser
                if ser is None or not getattr(ser, "is_open", False):
                    failure = "serial reader stopped because the port closed unexpectedly"
                    break
                try:
                    chunk = ser.read(256)
                    if not chunk:
                        continue
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if len(line) > self._max_line_bytes:
                            raise ValueError(
                                "serial line exceeded bounded size of "
                                f"{self._max_line_bytes} bytes"
                            )
                        try:
                            s = line.decode(errors="replace").strip()
                        except Exception:
                            s = str(line)
                        if not s:
                            continue
                        if (
                            self.failure_plan is not None
                            and self.failure_plan.kind == "read_error"
                            and self._read_count >= self.failure_plan.after_frames
                        ):
                            raise OSError(
                                "qualification-injected serial reader failure"
                            )
                        self._read_count += 1
                        if self._emit_serial_logs or _is_feeder_jam_protocol_line(s):
                            self.emit(f"[{timestamp()}] 【SER】 {s}")
                        # Recording readiness requires evidence of an actual
                        # rising trigger edge, not arbitrary CAMERA_* status or
                        # configuration telemetry.
                        if s == "CAMERA_HIGH" or s.startswith("CAMERA_HIGH,"):
                            with self._ttl_lock:
                                self._ttl_seen.set()
                        self._write_csv_line(s)
                        if self._stop.is_set():
                            break
                        try:
                            session = self._watchdog_session
                            consumed = session.ingest(s) if session is not None else False
                            capabilities = (
                                None if consumed else parse_controller_capabilities(s)
                            )
                        except ValueError as exc:
                            self.emit(
                                f"[{timestamp()}] [SER] WARN: ignored malformed "
                                f"controller capability advertisement: {exc}"
                            )
                        else:
                            if capabilities is not None:
                                with self._capabilities_lock:
                                    self._controller_capabilities = capabilities
                        if s == "ACK_STOP" or s.startswith("ACK_STOP,"):
                            fields = s.split(",")
                            try:
                                self._stop_ack_count = int(float(fields[4]))
                            except (IndexError, TypeError, ValueError):
                                self._stop_ack_count = None
                            self._stop_ack_seen.set()
                        self._ingest_clear_jam_response(s)
                        self._maybe_send_alert(s)
                    if len(buf) > self._max_line_bytes:
                        raise ValueError(
                            "serial input without a newline exceeded bounded size of "
                            f"{self._max_line_bytes} bytes"
                        )
                except Exception as exc:
                    if self._stop.is_set():
                        break
                    failure = f"serial reader failed on {self.port}: {exc}"
                    break
        finally:
            if failure is not None:
                self._report_fatal(failure)
            elif not self._stop.is_set():
                self._report_fatal("serial reader exited unexpectedly")
            self.emit(f"[{timestamp()}] [SER] reader thread exit")

    def _write_csv_line(self, line: str) -> None:
        self._write_csv_fields(line.split(","), line)

    def _write_csv_fields(self, fields: list[str], raw_line: str) -> None:
        row = list(fields)
        serial_field_count = len(SERIAL_HEADER) - 3
        if len(row) < serial_field_count:
            row.extend([""] * (serial_field_count - len(row)))
        elif len(row) > serial_field_count:
            row = row[: serial_field_count - 1] + [",".join(row[serial_field_count - 1 :])]
        row.extend([str(time.time_ns()), str(time.monotonic_ns()), raw_line])
        failure: str | None = None
        with self._csv_lock:
            if self._csv_ready and self._csv_writer:
                try:
                    if (
                        self.failure_plan is not None
                        and self.failure_plan.kind == "ledger_write_error"
                        and self._row_count >= self.failure_plan.after_frames
                    ):
                        raise OSError(
                            "qualification-injected serial ledger write failure"
                        )
                    self._csv_writer.writerow(row)
                    self._row_count += 1
                    if self._row_count % self._flush_every == 0 and self._csv_file:
                        self._csv_file.flush()
                except Exception as exc:
                    failure = f"serial CSV write/flush failed: {exc}"
            else:
                if len(self._buffer_rows) >= self._max_buffered_rows:
                    failure = (
                        "serial CSV was unavailable and its bounded pre-ledger "
                        f"buffer reached {self._max_buffered_rows} rows"
                    )
                else:
                    self._buffer_rows.append(row)
        if failure is not None:
            self._report_fatal(failure)

    def log_marker(self, marker: str) -> None:
        """Write a non-serial marker row into the CSV for later alignment."""
        try:
            self._write_csv_fields(
                ["MARKER", "", "", "", "", "", "", "", "HOST", marker],
                f"MARKER,{marker}",
            )
            if self._emit_serial_logs:
                self.emit(f"[{timestamp()}] 【SER】 MARKER,{marker}")
        except Exception:
            pass

    def _send_line(self, text: str, *, marker_before: str | None = None) -> None:
        """Write and flush a controller command, or raise if delivery fails.

        Controller commands participate in the scientific run lifecycle.  A
        caller must never interpret a logged write error as a successful START
        or STOP command.
        """

        if not self.ser or not self.ser.is_open:
            message = f"cannot send {text!r}: serial port is not open"
            self.emit(f"[{timestamp()}] [SER] {message}")
            raise RuntimeError(message)
        try:
            command = text.strip().upper()
            if (
                self.failure_plan is not None
                and self.failure_plan.kind == "write_error"
                and self._write_count >= self.failure_plan.after_frames
            ):
                raise OSError("qualification-injected serial command write failure")
            if command == "STOP":
                # Clear immediately before writing so a fast controller reply
                # cannot race ahead of wait_for_stop_ack().
                self._stop_ack_seen.clear()
                self._stop_ack_count = None
            with self._write_lock:
                starts_trigger = command.startswith("START") or command.startswith(
                    "ARM,"
                )
                # Serialize the clear/write boundary with CAMERA_HIGH ingest.
                # A delayed pre-command edge therefore cannot race between the
                # clear and the command write and falsely satisfy readiness.
                with self._ttl_lock:
                    if starts_trigger:
                        self._ttl_seen.clear()
                    if marker_before is not None:
                        # The marker and controller write share the same edge-ingest
                        # lock. A fast controller response therefore cannot be
                        # persisted ahead of the marker that defines its epoch.
                        self.log_marker(marker_before)
                    self.emit(f"[{timestamp()}] 【SER→】 {text}")
                    self.ser.write((text + "\n").encode())
                    self.ser.flush()
                    self._write_count += 1
        except Exception as exc:
            self.emit(f"[{timestamp()}] [SER] write error: {exc}")
            raise RuntimeError(f"serial write failed for {text!r}: {exc}") from exc

    def send_line(self, text: str) -> None:
        self._send_line(text)

    def send_start(self, fps: int) -> None:
        """Persist the legacy START epoch boundary before the atomic write."""

        if type(fps) is not int or fps <= 0:
            raise ValueError("controller FPS must be a positive integer")
        self._send_line(f"START,{fps}", marker_before="START_SENT")

    def _ingest_clear_jam_response(self, line: str) -> bool:
        """Publish one exact CLEAR_JAM reply from the existing reader."""

        if line not in _CLEAR_JAM_RESPONSES:
            return False
        with self._clear_jam_lock:
            if self._clear_jam_pending and not self._clear_jam_response_seen.is_set():
                self._clear_jam_response = line
                self._clear_jam_response_seen.set()
        return True

    def _interrupt_clear_jam(self, error: str) -> None:
        with self._clear_jam_lock:
            if self._clear_jam_pending and not self._clear_jam_response_seen.is_set():
                self._clear_jam_error = str(error).strip() or "serial connection lost"
                self._clear_jam_response_seen.set()

    def clear_feeder_jam(self, *, timeout_s: float = 2.0) -> str:
        """Send one CLEAR_JAM command and await its exact response.

        This method is called by a backend worker. The existing serial reader
        fulfills the acknowledgement event; no second reader is created.
        """

        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)):
            raise ValueError("CLEAR_JAM timeout must be numeric")
        timeout = float(timeout_s)
        if not 0.0 < timeout <= 30.0:
            raise ValueError("CLEAR_JAM timeout must be greater than zero and at most 30 seconds")
        with self._clear_jam_lock:
            if self._clear_jam_pending:
                raise RuntimeError("CLEAR_JAM is already awaiting a controller response")
            self._clear_jam_pending = True
            self._clear_jam_response = None
            self._clear_jam_error = None
            self._clear_jam_response_seen.clear()
        try:
            self._send_line("CLEAR_JAM")
            if not self._clear_jam_response_seen.wait(timeout=timeout):
                raise TimeoutError(
                    f"controller did not respond to CLEAR_JAM within {timeout:g}s"
                )
            with self._clear_jam_lock:
                response = self._clear_jam_response
                error = self._clear_jam_error
            if error is not None:
                raise ConnectionError(
                    f"serial connection was lost while clearing the feeder jam: {error}"
                )
            if response not in _CLEAR_JAM_RESPONSES:
                raise RuntimeError("CLEAR_JAM completed without a recognized response")
            return response
        finally:
            with self._clear_jam_lock:
                self._clear_jam_pending = False

    def wait_for_ttl(self, timeout_s: float = 3.0) -> bool:
        self.emit(f"[{timestamp()}] [SER] Waiting for camera TTL line (timeout {timeout_s:.1f}s) …")
        hit = self._ttl_seen.wait(timeout=timeout_s)
        self.emit(
            f"[{timestamp()}] [SER] "
            f"{'TTL detected.' if hit else 'TTL not detected within timeout — startup handshake failed.'}"
        )
        return hit

    def wait_for_stop_ack(self, timeout_s: float = 2.0) -> bool:
        """Wait until ACK_STOP has been persisted to the serial ledger."""
        self.emit(f"[{timestamp()}] [SER] Waiting for ACK_STOP (timeout {timeout_s:.1f}s) …")
        hit = self._stop_ack_seen.wait(timeout=timeout_s)
        self.emit(
            f"[{timestamp()}] [SER] "
            f"{'ACK_STOP received.' if hit else 'ACK_STOP not received within timeout — draining anyway.'}"
        )
        return hit

    def negotiate_watchdog_v1(
        self, *, requested_lease_ms: int, timeout_s: float = 2.0
    ) -> ControllerCapabilities:
        """Negotiate one experimental nonce-bound session, or raise fail closed."""

        if self._watchdog_session is not None:
            raise RuntimeError("controller watchdog negotiation was already attempted")
        session = ControllerWatchdogSession(
            self.send_line,
            self._report_watchdog_fatal,
            requested_lease_ms=requested_lease_ms,
        )
        self._watchdog_session = session
        capabilities = session.negotiate(timeout_s=timeout_s)
        with self._capabilities_lock:
            self._controller_capabilities = capabilities
        return capabilities

    def arm_watchdog_v1(self, fps: int, *, timeout_s: float = 2.0) -> None:
        session = self._watchdog_session
        if session is None:
            raise RuntimeError("controller watchdog session was not negotiated")
        session.arm(fps, timeout_s=timeout_s)

    def disarm_watchdog_v1(self, *, timeout_s: float = 2.0) -> None:
        session = self._watchdog_session
        if session is None:
            raise RuntimeError("controller watchdog session was not negotiated")
        session.disarm(timeout_s=timeout_s)

    @property
    def watchdog_snapshot(self) -> dict[str, object] | None:
        session = self._watchdog_session
        return session.snapshot() if session is not None else None

    @property
    def stop_ack_count(self) -> int | None:
        """Final controller TTL count carried by the latest ACK_STOP row."""
        if self._watchdog_session is not None:
            return self._watchdog_session.final_ttl_count
        return self._stop_ack_count

    # ---- Alerts -------------------------------------------------------
    def _maybe_send_alert(self, line: str) -> None:
        """Start at most one best-effort email worker for matching telemetry.

        Serial input is not flow controlled by SMTP.  Coalescing matches while
        one delivery is in flight keeps a noisy controller from creating an
        unbounded number of threads without delaying the serial ledger.
        """
        phrase = self._alert_phrase
        if not phrase:
            return
        if phrase.lower() not in line.lower():
            return
        with self._alert_lock:
            if self._alert_inflight:
                self._alert_suppressed += 1
                return
            self._alert_inflight = True
        try:
            threading.Thread(
                target=self._run_email_alert,
                args=(line,),
                daemon=True,
                name="squeakview-serial-alert",
            ).start()
        except Exception:
            with self._alert_lock:
                self._alert_inflight = False
            raise

    def _run_email_alert(self, line: str) -> None:
        try:
            self._send_email_alert(line)
        finally:
            with self._alert_lock:
                suppressed = self._alert_suppressed
                self._alert_suppressed = 0
                self._alert_inflight = False
            if suppressed:
                self.emit(
                    f"[{timestamp()}] [SER] coalesced {suppressed} additional "
                    "serial alert match(es) while email delivery was active"
                )

    def _send_email_alert(self, line: str) -> None:
        """Send a minimal SMTP email using env vars; best-effort and non-fatal."""
        host = os.environ.get("SQUEAKVIEW_ALERT_EMAIL_HOST")
        to_addr = os.environ.get("SQUEAKVIEW_ALERT_EMAIL_TO")
        if not host or not to_addr:
            if not self._alert_warned:
                self.emit(f"[{timestamp()}] [SER] alert skipped (set SQUEAKVIEW_ALERT_EMAIL_HOST and ..._TO)")
                self._alert_warned = True
            return
        try:
            port = int(os.environ.get("SQUEAKVIEW_ALERT_EMAIL_PORT", "587"))
        except Exception:
            port = 587
        user = os.environ.get("SQUEAKVIEW_ALERT_EMAIL_USER")
        password = os.environ.get("SQUEAKVIEW_ALERT_EMAIL_PASS")
        from_addr = os.environ.get("SQUEAKVIEW_ALERT_EMAIL_FROM", user or to_addr)
        use_tls = os.environ.get("SQUEAKVIEW_ALERT_EMAIL_TLS", "1") != "0"
        subject = os.environ.get("SQUEAKVIEW_ALERT_EMAIL_SUBJECT", "SqueakView serial alert")

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = to_addr
        msg.set_content(f"Serial alert detected at {timestamp()}:\n\n{line}\n")

        try:
            with smtplib.SMTP(host, port, timeout=10) as server:
                if use_tls:
                    server.starttls(context=ssl.create_default_context())
                if user and password:
                    server.login(user, password)
                server.send_message(msg)
            self.emit(f"[{timestamp()}] [SER] alert email sent to {to_addr}")
        except Exception as exc:
            self.emit(f"[{timestamp()}] [SER] alert email failed: {exc}")

    def close(self) -> None:
        if (
            self._closed
            and self.ser is None
            and self._csv_file is None
            and not (self._thread and self._thread.is_alive())
        ):
            self._unregister_atexit()
            return
        self.emit(f"[{timestamp()}] [SER] closing …")
        self._interrupt_clear_jam("serial port closed")
        if self._watchdog_session is not None:
            self._watchdog_session.stop_worker()
        self._stop.set()
        failure: str | None = None
        # Close the port before joining so a driver read that ignores the
        # configured timeout is actively unblocked.  The ledger remains open
        # until the reader has had a chance to finish its final callback.
        try:
            if self.ser and getattr(self.ser, "is_open", False):
                self.ser.close()
        except Exception as exc:
            failure = f"serial port close failed: {exc}"
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=1.0)
                if self._thread.is_alive() and failure is None:
                    failure = "serial reader did not stop after the port was closed"
            except Exception as exc:
                if failure is None:
                    failure = f"serial reader join failed: {exc}"
        with self._csv_lock:
            if self._csv_file:
                try:
                    self._csv_file.flush()
                    self._csv_file.close()
                except Exception as exc:
                    csv_failure = f"serial CSV close/flush failed: {exc}"
                    failure = f"{failure}; {csv_failure}" if failure else csv_failure
            still_temp = self._tmp_opened and self._tmp_csv_path
            tmp_path = self._tmp_csv_path
            self._csv_file = None
            self._csv_writer = None
            self._csv_ready = False
            self._tmp_opened = False
            self._tmp_csv_path = None
        if failure is not None:
            self._report_fatal(failure)
        if still_temp and tmp_path:
            self.emit(
                f"[{timestamp()}] [SER] Run dir unknown at stop. Temp CSV kept here:\n{tmp_path}"
            )
        self.ser = None
        self._closed = True
        self._unregister_atexit()


def iter_lines(buffer: Iterable[str]) -> Iterable[str]:
    for line in buffer:
        yield line.rstrip("\n")
