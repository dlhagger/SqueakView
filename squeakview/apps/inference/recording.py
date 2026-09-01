"""Recording-branch admission and bounded telemetry operators."""
from __future__ import annotations

import atexit
from collections import OrderedDict
import csv
import threading
import time
from pathlib import Path
from typing import Callable

from pyservicemaker import BufferOperator


class RecordingAdmissionOperator(BufferOperator):
    """Durably record buffers that enter the non-leaky recording branch."""

    HEADERS = ["stream_id", "record_frame_index", "pts_ns", "observer_monotonic_ns"]

    def __init__(
        self,
        path: Path,
        stream_id: int,
        telemetry: "RecordingPathTelemetry | None" = None,
    ):
        super().__init__()
        self.path = path
        self.stream_id = int(stream_id)
        self._file = path.open("w", newline="", buffering=1)
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADERS)
        self._count = 0
        self._lock = threading.Lock()
        self._closed = False
        self._telemetry = telemetry
        atexit.register(self.close)

    def handle_buffer(self, buffer) -> bool:
        with self._lock:
            if self._closed:
                return False
            self._writer.writerow(
                [self.stream_id, self._count, int(buffer.timestamp), time.monotonic_ns()]
            )
            if self._telemetry is not None:
                self._telemetry.admit(int(buffer.timestamp))
            self._count += 1
        return True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._file.flush()
            self._file.close()


class RecordingPathTelemetry:
    """Sample recording backlog and encoder latency with bounded state."""

    HEADERS = [
        "host_unix_ns",
        "host_monotonic_ns",
        "stream_id",
        "event",
        "pts_ns",
        "egress_timestamp_ns",
        "encoder_correlation",
        "queue_wait_ms",
        "encoder_latency_ms",
        "waiting_for_record_admission",
        "encoder_in_flight",
        "max_waiting_since_sample",
        "max_encoder_in_flight_since_sample",
        "pending_evictions",
    ]

    def __init__(
        self,
        path: Path,
        stream_id: int,
        *,
        sample_interval_s: float = 1.0,
        warning_depth: int = 24,
        fatal_depth: int | None = None,
        max_pending: int = 4096,
        on_fatal: Callable[[str], None] | None = None,
    ):
        self.path = path
        self.stream_id = int(stream_id)
        self.sample_interval_ns = max(1, int(sample_interval_s * 1_000_000_000))
        self.warning_depth = max(1, int(warning_depth))
        self.fatal_depth = (
            max(self.warning_depth + 1, int(fatal_depth))
            if fatal_depth is not None
            else None
        )
        self.max_pending = max(self.warning_depth, int(max_pending))
        self.on_fatal = on_fatal
        self._file = path.open("w", newline="", buffering=1)
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.HEADERS)
        self._source_pending: OrderedDict[int, int] = OrderedDict()
        self._pending: OrderedDict[int, int] = OrderedDict()
        self._lock = threading.Lock()
        self._last_sample_ns = time.monotonic_ns()
        self._max_source_depth = 0
        self._max_encoder_depth = 0
        self._evictions = 0
        self._warning_active = False
        self._fatal_reported = False
        self._closed = False
        atexit.register(self.close)

    def _write(
        self,
        event: str,
        now_ns: int,
        pts_ns: int | str,
        egress_timestamp_ns="",
        encoder_correlation="",
        queue_wait_ms="",
        encoder_latency_ms="",
    ) -> None:
        self._writer.writerow(
            [
                time.time_ns(),
                now_ns,
                self.stream_id,
                event,
                pts_ns,
                egress_timestamp_ns,
                encoder_correlation,
                queue_wait_ms,
                encoder_latency_ms,
                len(self._source_pending),
                len(self._pending),
                self._max_source_depth,
                self._max_encoder_depth,
                self._evictions,
            ]
        )

    def _sample_if_due(
        self,
        now_ns: int,
        pts_ns: int,
        queue_wait_ms="",
        encoder_latency_ms="",
        egress_timestamp_ns="",
        encoder_correlation="",
    ) -> str | None:
        source_depth = len(self._source_pending)
        encoder_depth = len(self._pending)
        self._max_source_depth = max(self._max_source_depth, source_depth)
        self._max_encoder_depth = max(self._max_encoder_depth, encoder_depth)
        warning = max(source_depth, encoder_depth) >= self.warning_depth
        if warning != self._warning_active:
            self._warning_active = warning
            self._write(
                "backpressure_enter" if warning else "backpressure_exit",
                now_ns,
                pts_ns,
                egress_timestamp_ns,
                encoder_correlation,
                queue_wait_ms,
                encoder_latency_ms,
            )
        fatal_message = None
        if (
            self.fatal_depth is not None
            and source_depth >= self.fatal_depth
            and not self._fatal_reported
        ):
            self._fatal_reported = True
            fatal_message = (
                f"recording queue backlog reached {source_depth} frames "
                f"(fatal threshold {self.fatal_depth}) on stream {self.stream_id}"
            )
            self._write("backpressure_fatal", now_ns, pts_ns)
        if now_ns - self._last_sample_ns >= self.sample_interval_ns:
            self._write(
                "sample",
                now_ns,
                pts_ns,
                egress_timestamp_ns,
                encoder_correlation,
                queue_wait_ms,
                encoder_latency_ms,
            )
            self._last_sample_ns = now_ns
            self._max_source_depth = source_depth
            self._max_encoder_depth = encoder_depth
        return fatal_message

    def source(self, pts_ns: int) -> None:
        now_ns = time.monotonic_ns()
        fatal_message = None
        with self._lock:
            if self._closed:
                return
            self._source_pending[int(pts_ns)] = now_ns
            while len(self._source_pending) > self.max_pending:
                self._source_pending.popitem(last=False)
                self._evictions += 1
            fatal_message = self._sample_if_due(now_ns, int(pts_ns))
        if fatal_message is not None and self.on_fatal is not None:
            self.on_fatal(fatal_message)

    def admit(self, pts_ns: int) -> None:
        now_ns = time.monotonic_ns()
        with self._lock:
            if self._closed:
                return
            source_ns = self._source_pending.pop(int(pts_ns), None)
            queue_wait_ms = (
                f"{(now_ns - source_ns) / 1_000_000.0:.6f}"
                if source_ns is not None
                else ""
            )
            self._pending[int(pts_ns)] = now_ns
            while len(self._pending) > self.max_pending:
                self._pending.popitem(last=False)
                self._evictions += 1
            self._sample_if_due(now_ns, int(pts_ns), queue_wait_ms=queue_wait_ms)

    def egress(self, egress_timestamp_ns: int) -> None:
        now_ns = time.monotonic_ns()
        with self._lock:
            if self._closed:
                return
            output_timestamp = int(egress_timestamp_ns)
            input_pts_ns = output_timestamp
            admitted_ns = self._pending.pop(output_timestamp, None)
            correlation = "pts" if admitted_ns is not None else ""
            if admitted_ns is None and self._pending:
                # x264 is configured with bframes=0 and both lookaheads at 0,
                # so encoded access units retain input order even when the
                # parser exposes a rewritten timestamp through ServiceMaker.
                input_pts_ns, admitted_ns = self._pending.popitem(last=False)
                correlation = "fifo"
            elif admitted_ns is None:
                correlation = "unmatched"
            encoder_latency_ms = (
                f"{(now_ns - admitted_ns) / 1_000_000.0:.6f}"
                if admitted_ns is not None
                else ""
            )
            self._sample_if_due(
                now_ns,
                input_pts_ns,
                encoder_latency_ms=encoder_latency_ms,
                egress_timestamp_ns=output_timestamp,
                encoder_correlation=correlation,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._write("closed", time.monotonic_ns(), "")
            self._file.flush()
            self._file.close()


class RecordingEgressOperator(BufferOperator):
    def __init__(self, telemetry: RecordingPathTelemetry):
        super().__init__()
        self.telemetry = telemetry

    def handle_buffer(self, buffer) -> bool:
        self.telemetry.egress(int(buffer.timestamp))
        return True


class RecordingIngressOperator(BufferOperator):
    def __init__(self, telemetry: RecordingPathTelemetry):
        super().__init__()
        self.telemetry = telemetry

    def handle_buffer(self, buffer) -> bool:
        self.telemetry.source(int(buffer.timestamp))
        return True
