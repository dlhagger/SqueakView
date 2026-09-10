"""Line-aware console-to-file mirroring for the GUI launcher."""

from __future__ import annotations

import atexit
import re
import threading
from pathlib import Path
from typing import TextIO


DEFAULT_LOG_MAX_BYTES = 32 * 1024 * 1024
DEFAULT_PENDING_MAX_CHARS = 64 * 1024


class LineBufferedLogMirror:
    """Mirror complete stdout/stderr lines to a log, filtering serial chatter.

    This is unrelated to the GStreamer camera tee. Output from ``print`` may
    arrive as separate message and newline writes, so filtering individual
    writes leaves orphaned blank lines in long-running GUI logs.
    """

    _SER_PAT = re.compile(r"\bCAMERA_(LOW|HIGH)\b")

    def __init__(
        self,
        path: Path,
        stream: TextIO,
        *,
        max_bytes: int = DEFAULT_LOG_MAX_BYTES,
        max_pending_chars: int = DEFAULT_PENDING_MAX_CHARS,
    ):
        self._stream = stream
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", buffering=1)
        self._max_bytes = max(0, int(max_bytes))
        self._bytes_written = path.stat().st_size
        self._file_capped = self._bytes_written >= self._max_bytes
        self._max_pending_chars = max(1, int(max_pending_chars))
        self._pending = ""
        self._discard_until_newline = False
        self._lock = threading.RLock()
        self._closed = False
        self._atexit_registered = True
        atexit.register(self.close)

    def _write_file(self, text: str) -> None:
        if self._file_capped or self._closed:
            return
        encoded = text.encode("utf-8", errors="replace")
        if self._bytes_written + len(encoded) <= self._max_bytes:
            try:
                self._fh.write(encoded.decode("utf-8"))
                self._bytes_written += len(encoded)
            except Exception:
                pass
            return
        marker = "[SQUEAKVIEW] operator log size limit reached\n"
        marker_size = len(marker.encode("utf-8"))
        if self._bytes_written + marker_size <= self._max_bytes:
            try:
                self._fh.write(marker)
                self._bytes_written += marker_size
            except Exception:
                pass
        self._file_capped = True
        try:
            self._stream.write(
                "[SQUEAKVIEW] operator file log capped; console output continues\n"
            )
        except Exception:
            pass

    def write(self, data: str) -> int:
        written = len(data)
        with self._lock:
            # A terminal or IDE host can disappear during a multi-day run.
            # Console mirroring is best-effort; the durable file mirror must
            # continue and must never turn shutdown logging into an exception.
            try:
                self._stream.write(data)
            except (BrokenPipeError, OSError, ValueError):
                pass
            if self._closed:
                return written
            if self._discard_until_newline:
                _discarded, separator, data = data.partition("\n")
                if not separator:
                    return written
                self._discard_until_newline = False
            self._pending += data
            while "\n" in self._pending:
                line, self._pending = self._pending.split("\n", 1)
                complete_line = line + "\n"
                if not self._SER_PAT.search(line):
                    self._write_file(complete_line)
            if len(self._pending) > self._max_pending_chars:
                fragment = self._pending[: self._max_pending_chars]
                self._pending = ""
                self._discard_until_newline = True
                if not self._SER_PAT.search(fragment):
                    self._write_file(fragment + "… [line truncated]\n")
        return written

    def flush(self) -> None:
        with self._lock:
            if not self._closed and self._pending:
                if self._SER_PAT.search(self._pending):
                    self._discard_until_newline = True
                else:
                    self._write_file(self._pending)
                self._pending = ""
            if not self._closed:
                try:
                    self._fh.flush()
                except Exception:
                    pass
            try:
                self._stream.flush()
            except Exception:
                pass

    @property
    def underlying_stream(self) -> TextIO:
        return self._stream

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Flush/close only the owned file; the supplied console stays open."""

        with self._lock:
            if self._closed:
                return
            # Preserve a final partial diagnostic before closing ownership.
            if self._pending:
                if not self._SER_PAT.search(self._pending):
                    self._write_file(self._pending)
                self._pending = ""
            try:
                self._fh.flush()
            except Exception:
                pass
            try:
                self._fh.close()
            except Exception:
                pass
            self._closed = True
        if self._atexit_registered:
            try:
                atexit.unregister(self.close)
            except Exception:
                pass
            self._atexit_registered = False

    def __getattr__(self, name: str):
        """Preserve stream attributes such as encoding, fileno, and isatty."""
        return getattr(self._stream, name)
