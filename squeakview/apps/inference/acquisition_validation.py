"""Scientific integrity validation for camera diagnostics and transport counters."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from squeakview.common.bounded_csv import BoundedCsvError, bounded_csv_lines


TRANSPORT_COUNTERS = (
    "stream_incomplete_frames",
    "stream_lost_frames",
    "stream_dropped_frames",
)


@dataclass(frozen=True, slots=True)
class AcquisitionValidationResult:
    errors_file_present: bool
    camera_telemetry_present: bool
    errors_schema_valid: bool
    camera_telemetry_schema_valid: bool
    camera_telemetry_sample_rows: int
    event_counts: tuple[tuple[str, int], ...]
    transport_counter_maxima: tuple[tuple[str, int], ...]
    transport_counter_samples: tuple[tuple[str, int], ...]
    expected_stream_ids: tuple[int, ...]
    observed_stream_ids: tuple[int, ...]
    telemetry_rows_by_stream: tuple[tuple[int, int], ...]
    transport_counter_samples_by_stream: tuple[
        tuple[int, tuple[tuple[str, int], ...]], ...
    ]
    unexpected_stream_rows: int
    invalid_diagnostic_rows: int

    @property
    def passed(self) -> bool:
        maxima = dict(self.transport_counter_maxima)
        samples = dict(self.transport_counter_samples)
        per_stream_samples = {
            stream_id: dict(counters)
            for stream_id, counters in self.transport_counter_samples_by_stream
        }
        return (
            self.errors_file_present
            and self.camera_telemetry_present
            and self.errors_schema_valid
            and self.camera_telemetry_schema_valid
            and self.camera_telemetry_sample_rows > 0
            and self.observed_stream_ids == self.expected_stream_ids
            and self.unexpected_stream_rows == 0
            and all(samples[name] > 0 for name in TRANSPORT_COUNTERS)
            and all(
                per_stream_samples[stream_id][name] > 0
                for stream_id in self.expected_stream_ids
                for name in TRANSPORT_COUNTERS
            )
            and self.invalid_diagnostic_rows == 0
            and sum(count for _, count in self.event_counts) == 0
            and sum(maxima.values()) == 0
        )

    def to_dict(self) -> dict:
        event_counts = dict(self.event_counts)
        maxima = dict(self.transport_counter_maxima)
        samples = dict(self.transport_counter_samples)
        samples_by_stream = {
            stream_id: dict(counters)
            for stream_id, counters in self.transport_counter_samples_by_stream
        }
        return {
            "schema_version": "1.0",
            "policy": "no_camera_gaps_crc_metadata_failures_or_transport_loss",
            "errors_file_present": self.errors_file_present,
            "camera_telemetry_present": self.camera_telemetry_present,
            "errors_schema_valid": self.errors_schema_valid,
            "camera_telemetry_schema_valid": self.camera_telemetry_schema_valid,
            "camera_telemetry_sample_rows": self.camera_telemetry_sample_rows,
            "event_counts": event_counts,
            "event_total": sum(event_counts.values()),
            "transport_counter_maxima": maxima,
            "transport_counter_samples": samples,
            "expected_stream_ids": list(self.expected_stream_ids),
            "observed_stream_ids": list(self.observed_stream_ids),
            "telemetry_rows_by_stream": dict(self.telemetry_rows_by_stream),
            "transport_counter_samples_by_stream": samples_by_stream,
            "unexpected_stream_rows": self.unexpected_stream_rows,
            "invalid_diagnostic_rows": self.invalid_diagnostic_rows,
            "passed": self.passed,
        }


def validate_acquisition_integrity(
    run_dir: Path, camera_count: int = 1
) -> AcquisitionValidationResult:
    """Reject camera gaps, metadata/CRC failures, and transport loss counters."""

    if isinstance(camera_count, bool) or not isinstance(camera_count, int):
        raise ValueError("camera_count must be an integer")
    if camera_count < 1:
        raise ValueError("camera_count must be positive")
    expected_stream_ids = tuple(range(camera_count))

    diagnostics = Path(run_dir) / "diagnostics"
    errors_path = diagnostics / "errors.csv"
    camera_path = diagnostics / "camera.csv"
    event_counts: dict[str, int] = {}
    invalid_rows = 0
    errors_schema_valid = False
    if errors_path.is_file():
        try:
            reader = csv.DictReader(bounded_csv_lines(errors_path))
            fieldnames = reader.fieldnames or []
            errors_schema_valid = (
                "event_type" in fieldnames
                and len(fieldnames) == len(set(fieldnames))
            )
            for row in reader:
                event_type = str(row.get("event_type") or "").strip()
                if not event_type:
                    invalid_rows += 1
                    continue
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
        except (OSError, csv.Error, BoundedCsvError):
            invalid_rows += 1

    transport_maxima = {name: 0 for name in TRANSPORT_COUNTERS}
    transport_samples = {name: 0 for name in TRANSPORT_COUNTERS}
    rows_by_stream = {stream_id: 0 for stream_id in expected_stream_ids}
    samples_by_stream = {
        stream_id: {name: 0 for name in TRANSPORT_COUNTERS}
        for stream_id in expected_stream_ids
    }
    unexpected_stream_rows = 0
    camera_schema_valid = False
    camera_sample_rows = 0
    if camera_path.is_file():
        try:
            reader = csv.DictReader(bounded_csv_lines(camera_path))
            fieldnames = reader.fieldnames or []
            camera_schema_valid = (
                {"stream_id", *TRANSPORT_COUNTERS}.issubset(fieldnames)
                and len(fieldnames) == len(set(fieldnames))
            )
            for row in reader:
                camera_sample_rows += 1
                try:
                    stream_id = int(str(row.get("stream_id") or ""))
                except ValueError:
                    invalid_rows += 1
                    continue
                if stream_id not in rows_by_stream:
                    unexpected_stream_rows += 1
                    continue
                rows_by_stream[stream_id] += 1
                for name in TRANSPORT_COUNTERS:
                    raw = row.get(name)
                    if raw in (None, ""):
                        continue
                    try:
                        text = raw if isinstance(raw, str) else ""
                        if not text or not text.isascii() or not text.isdigit():
                            raise ValueError("transport counter is not an unsigned integer")
                        value = int(text)
                        transport_maxima[name] = max(transport_maxima[name], value)
                        transport_samples[name] += 1
                        samples_by_stream[stream_id][name] += 1
                    except (TypeError, ValueError):
                        invalid_rows += 1
        except (OSError, csv.Error, BoundedCsvError):
            invalid_rows += 1

    return AcquisitionValidationResult(
        errors_file_present=errors_path.is_file(),
        camera_telemetry_present=camera_path.is_file(),
        errors_schema_valid=errors_schema_valid,
        camera_telemetry_schema_valid=camera_schema_valid,
        camera_telemetry_sample_rows=camera_sample_rows,
        event_counts=tuple(sorted(event_counts.items())),
        transport_counter_maxima=tuple(transport_maxima.items()),
        transport_counter_samples=tuple(transport_samples.items()),
        expected_stream_ids=expected_stream_ids,
        observed_stream_ids=tuple(
            stream_id
            for stream_id, row_count in rows_by_stream.items()
            if row_count > 0
        ),
        telemetry_rows_by_stream=tuple(rows_by_stream.items()),
        transport_counter_samples_by_stream=tuple(
            (stream_id, tuple(samples.items()))
            for stream_id, samples in samples_by_stream.items()
        ),
        unexpected_stream_rows=unexpected_stream_rows,
        invalid_diagnostic_rows=invalid_rows,
    )
