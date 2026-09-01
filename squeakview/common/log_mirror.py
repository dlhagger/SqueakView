"""Line-aware console-to-file mirroring for the GUI launcher."""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import TextIO


class LineBufferedLogMirror:
    """Mirror complete stdout/stderr lines to a log, filtering serial chatter.

    This is unrelated to the GStreamer camera tee. Output from ``print`` may
    arrive as separate message and newline writes, so filtering individual
    writes leaves orphaned blank lines in long-running GUI logs.
    """

    _SER_PAT = re.compile(r"\bCAMERA_(LOW|HIGH)\b")

    def __init__(self, path: Path, stream: TextIO):
        self._stream = stream
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", buffering=1)
        self._pending = ""
        self._discard_until_newline = False
        self._lock = threading.RLock()

    def write(self, data: str) -> int:
        written = len(data)
        with self._lock:
            self._stream.write(data)
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
                    try:
                        self._fh.write(complete_line)
                    except Exception:
                        pass
        return written

    def flush(self) -> None:
        with self._lock:
            if self._pending:
                if self._SER_PAT.search(self._pending):
                    self._discard_until_newline = True
                else:
                    try:
                        self._fh.write(self._pending)
                    except Exception:
                        pass
                self._pending = ""
            try:
                self._fh.flush()
            except Exception:
                pass
            try:
                self._stream.flush()
            except Exception:
                pass

    def __getattr__(self, name: str):
        """Preserve stream attributes such as encoding, fileno, and isatty."""
        return getattr(self._stream, name)
