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
]


def have_pyserial() -> bool:
    return serial is not None


def timestamp() -> str:
    return time.strftime("%H:%M:%S")


class SerialHandle:
    """Threaded serial reader/writer for Arduino telemetry."""

    def __init__(self, port: str, baud: int, emit_fn: Callable[[str], None]):
        self.emit = emit_fn
        self.port = port
        self.baud = baud
        self.ser = None
        self._thread = None
        self._stop = threading.Event()
        self._ttl_seen = threading.Event()
        self._csv_lock = threading.Lock()
        self._csv_writer: csv.writer | None = None
        self._csv_file = None
        self._buffer_rows: list[list[str]] = []
        self._csv_ready = False
        self._tmp_csv_path: str | None = None
        self._tmp_opened = False
        self._flush_every = 25
        self._row_count = 0
        # Allow silencing serial logs in the terminal; still record CSV.
        self._emit_serial_logs = os.environ.get("SQUEAKVIEW_SERIAL_LOG", "1") != "0"
        # Alert phrase and state for optional email notifications.
        self._alert_phrase = (os.environ.get("SQUEAKVIEW_SERIAL_ALERT_PHRASE") or "Feeder jammed").strip()
        self._alert_warned = False
        atexit.register(self.close)

    def _open_csv(self, path: Path) -> None:
        """Open CSV at the given path and flush any buffered lines."""
        try:
            is_empty = (not path.exists()) or path.stat().st_size == 0
        except Exception:
            is_empty = True
        f = open(path, "a", newline="", buffering=1)
        writer = csv.writer(f)
        if is_empty:
            try:
                writer.writerow(SERIAL_HEADER)
            except Exception:
                pass
        with self._csv_lock:
            self._row_count = 0
            self._csv_file = f
            self._csv_writer = writer
            self._csv_ready = True
            if self._buffer_rows:
                try:
                    self._csv_writer.writerows(self._buffer_rows)
                    self._row_count += len(self._buffer_rows)
                except Exception as exc:
                    self.emit(f"[{timestamp()}] [SER] CSV buffer replay error: {exc}")
                self._buffer_rows.clear()
            try:
                self._csv_file.flush()
            except Exception:
                pass

    def _open_temp_csv(self) -> None:
        basename = f"serial_{int(time.time())}_{uuid.uuid4().hex[:6]}.csv"
        self._tmp_csv_path = str(Path(tempfile.gettempdir()) / basename)
        self.emit(f"[{timestamp()}] [SER] Temp CSV opened: {self._tmp_csv_path}")
        self._open_csv(Path(self._tmp_csv_path))
        self._tmp_opened = True

    def set_csv_path(self, run_dir: Path) -> None:
        """Adopt the temp CSV into the official run dir by atomic rename."""
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            dst_path = run_dir / "serial.csv"

            with self._csv_lock:
                if self._csv_file:
                    try:
                        self._csv_file.flush()
                    except Exception:
                        pass
                    try:
                        self._csv_file.close()
                    except Exception:
                        pass
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
        except Exception as exc:  # pragma: no cover - filesystem edge cases
            self.emit(f"[{timestamp()}] [SER] Could not adopt CSV into run dir: {exc}")

    def open(self, run_dir: Path | None = None) -> bool:
        if serial is None:
            self.emit(f"[{timestamp()}] [SER] pyserial not installed.")
            return False
        try:
            self.emit(f"[{timestamp()}] [SER] Opening {self.port} @ {self.baud} …")
            self.ser = serial.Serial(self.port, self.baud, timeout=0.05)
            self._stop.clear()
            self._ttl_seen.clear()
            self._thread = threading.Thread(target=self._pump, daemon=True)
            self._thread.start()
            if run_dir is not None:
                self.set_csv_path(run_dir)
            else:
                self._open_temp_csv()
            return True
        except Exception as exc:
            self.emit(f"[{timestamp()}] [SER] ERROR opening serial: {exc}")
            self.ser = None
            return False

    def _pump(self) -> None:
        try:
            buf = b""
            while not self._stop.is_set():
                ser = self.ser
                if ser is None or not getattr(ser, "is_open", False):
                    break
                try:
                    chunk = ser.read(256)
                    if not chunk:
                        continue
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        try:
                            s = line.decode(errors="replace").strip()
                        except Exception:
                            s = str(line)
                        if not s:
                            continue
                        if self._emit_serial_logs:
                            self.emit(f"[{timestamp()}] 【SER】 {s}")
                        if s.startswith("CAMERA_"):
                            self._ttl_seen.set()
                        self._write_csv_line(s)
                        self._maybe_send_alert(s)
                except Exception as exc:
                    if self._stop.is_set():
                        break
                    if "Bad file descriptor" in str(exc):
                        break
                    self.emit(f"[{timestamp()}] [SER] read error: {exc}")
                    time.sleep(0.05)
        finally:
            self.emit(f"[{timestamp()}] [SER] reader thread exit")

    def _write_csv_line(self, line: str) -> None:
        row = line.split(",")
        with self._csv_lock:
            if self._csv_ready and self._csv_writer:
                try:
                    self._csv_writer.writerow(row)
                    self._row_count += 1
                    if self._row_count % self._flush_every == 0 and self._csv_file:
                        try:
                            self._csv_file.flush()
                        except Exception:
                            pass
                except Exception as exc:
                    self.emit(f"[{timestamp()}] [SER] CSV write error: {exc}")
            else:
                self._buffer_rows.append(row)

    def log_marker(self, marker: str) -> None:
        """Write a non-serial marker row into the CSV for later alignment."""
        try:
            self._write_csv_line(f"MARKER,{marker},{timestamp()}")
            if self._emit_serial_logs:
                self.emit(f"[{timestamp()}] 【SER】 MARKER,{marker}")
        except Exception:
            pass

    def send_line(self, text: str) -> None:
        if not self.ser or not self.ser.is_open:
            self.emit(f"[{timestamp()}] [SER] cannot send, port not open")
            return
        try:
            self.emit(f"[{timestamp()}] 【SER→】 {text}")
            self.ser.write((text + "\n").encode())
            self.ser.flush()
        except Exception as exc:
            self.emit(f"[{timestamp()}] [SER] write error: {exc}")

    def wait_for_ttl(self, timeout_s: float = 3.0) -> bool:
        self.emit(f"[{timestamp()}] [SER] Waiting for camera TTL line (timeout {timeout_s:.1f}s) …")
        hit = self._ttl_seen.wait(timeout=timeout_s)
        self.emit(
            f"[{timestamp()}] [SER] {'TTL detected.' if hit else 'TTL not detected within timeout — continuing anyway.'}"
        )
        return hit

    # ---- Alerts -------------------------------------------------------
    def _maybe_send_alert(self, line: str) -> None:
        """Fire an email alert if the configured phrase is present; non-blocking."""
        phrase = self._alert_phrase
        if not phrase:
            return
        if phrase.lower() not in line.lower():
            return
        threading.Thread(target=self._send_email_alert, args=(line,), daemon=True).start()

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
        self.emit(f"[{timestamp()}] [SER] closing …")
        self._stop.set()
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=0.5)
            except Exception:
                pass
        try:
            if self.ser and getattr(self.ser, "is_open", False):
                self.ser.close()
        except Exception:
            pass
        with self._csv_lock:
            if self._csv_file:
                try:
                    self._csv_file.flush()
                    self._csv_file.close()
                except Exception:
                    pass
            still_temp = self._tmp_opened and self._tmp_csv_path
            tmp_path = self._tmp_csv_path
            self._csv_file = None
            self._csv_writer = None
            self._csv_ready = False
            self._tmp_opened = False
            self._tmp_csv_path = None
        if still_temp and tmp_path:
            self.emit(
                f"[{timestamp()}] [SER] Run dir unknown at stop. Temp CSV kept here:\n{tmp_path}"
            )
        self.ser = None


def iter_lines(buffer: Iterable[str]) -> Iterable[str]:
    for line in buffer:
        yield line.rstrip("\n")
