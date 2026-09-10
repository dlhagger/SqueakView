"""Bounded-memory qualification of completed SqueakView run evidence."""

from __future__ import annotations

import csv
import hashlib
import math
import os
import stat
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from squeakview.common import run_context
from squeakview.common.bounded_input import (
    read_stable_regular_file,
    read_stable_regular_file_with_identity,
)
from squeakview.common.bounded_csv import BoundedCsvError, bounded_csv_lines
from squeakview.common.diagnostics.evidence_identity import (
    capture_source_evidence,
    source_evidence_errors,
    stable_file_identity,
    stable_file_matches_identity,
)
from squeakview.common.recording_evidence import (
    capture_recording_evidence,
    recording_evidence_complete,
    recording_evidence_matches,
    recording_evidence_same_content,
)
from squeakview.common.diagnostics.debug_overhead import (
    deepstream_latency_evidence_for_run,
)


QUALIFICATION_SCHEMA_VERSION = "1.0"
MAX_LIMITS_BYTES = 1 << 20
TERMINAL_SUCCESS_STATES = {"post_run_complete", "analysis_complete", "finalized"}
TERMINAL_FAILURE_STATES = {"failed", "finalization_failed", "analysis_failed"}
MAX_TASK_CONFIG_BYTES = 1024 * 1024
_SYSTEM_MAX_FIELDS = (
    "ram_pct",
    "swap_pct",
    "cpu_util_mean_pct",
    "cpu_util_max_pct",
    "gpu_util_pct",
    "emc_util_pct",
    "emc_clock_pct_of_max",
    "temp_max_c",
    "vdd_in_current_mw",
    "vdd_cpu_gpu_cv_current_mw",
    "vdd_soc_current_mw",
)
_REQUIRED_RESOURCE_METRICS = (
    "ram_pct",
    "swap_pct",
    "cpu_util_mean_pct",
    "cpu_util_max_pct",
    "gpu_util_pct",
    "temp_max_c",
    "vdd_in_current_mw",
)
_EMC_RESOURCE_METRICS = ("emc_util_pct", "emc_clock_pct_of_max")
_RECORDING_MAX_FIELDS = (
    "queue_wait_ms",
    "encoder_latency_ms",
    "waiting_for_record_admission",
    "encoder_in_flight",
    "max_waiting_since_sample",
    "max_encoder_in_flight_since_sample",
    "pending_evictions",
)
_KNOWN_RECORDING_EVENTS = {
    "sample",
    "backpressure_enter",
    "backpressure_exit",
    "backpressure_fatal",
    "closed",
}
_RECORDING_REQUIRED_FIELDS = {
    "host_unix_ns",
    "host_monotonic_ns",
    "stream_id",
    "event",
    "pts_ns",
    "egress_timestamp_ns",
    "encoder_correlation",
    *_RECORDING_MAX_FIELDS,
}
_REQUIRED_PLATFORM_FIELDS = (
    "device_model",
    "machine",
    "kernel",
    "python",
    "python_executable",
    "jetson_linux_release",
    "deepstream_build",
    "nvpmodel",
)
_REQUIRED_PLATFORM_PACKAGES = (
    "nvidia-l4t-core",
    "deepstream-9.1",
    "cuda-toolkit-13-2",
    "libcudnn9-cuda-13",
    "libnvinfer10",
    "libgstreamer1.0-0",
    "libspinnaker",
    "ffmpeg",
)
_MODEL_SHA256_FIELDS = (
    "model_manifest_sha256",
    "pose_sidecar_sha256",
    "onnx_sha256",
    "config_sha256",
    "engine_sha256",
)
_REQUIRED_LIMIT_FIELDS = {
    "telemetry": (
        "expected_interval_ms",
        "min_coverage_ratio",
        "max_gap_s",
        "max_malformed_rows",
        "max_invalid_rows",
        "max_throttle_samples",
    ),
    "resources": (
        "max_ram_pct",
        "max_swap_pct",
        "max_cpu_util_mean_pct",
        "max_cpu_util_max_pct",
        "max_gpu_util_pct",
        "max_emc_util_pct",
        "max_emc_clock_pct_of_max",
        "max_temp_max_c",
        "max_vdd_in_current_mw",
    ),
    "recording": (
        "max_queue_wait_ms",
        "max_encoder_latency_ms",
        "max_waiting_for_record_admission",
        "max_encoder_in_flight",
        "max_pending_evictions",
        "max_backpressure_fatal_events",
    ),
}
_OPTIONAL_VALIDATED_LIMIT_FIELDS = {"resources.max_emc_util_pct"}
_LIMITS_ROOT_KEYS = {
    "schema_version",
    "profile_id",
    "validated",
    "description",
    *_REQUIRED_LIMIT_FIELDS,
}
_INTEGER_LIMIT_FIELDS = {
    "telemetry.expected_interval_ms",
    "telemetry.max_malformed_rows",
    "telemetry.max_invalid_rows",
    "telemetry.max_throttle_samples",
    "recording.max_waiting_for_record_admission",
    "recording.max_encoder_in_flight",
    "recording.max_pending_evictions",
    "recording.max_backpressure_fatal_events",
}


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys at every depth."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _unknown_keys(value: dict[Any, Any], allowed: set[str]) -> list[Any]:
    return [key for key in value if not isinstance(key, str) or key not in allowed]


def _strict_limits_schema_error(payload: dict[str, Any]) -> str | None:
    unknown = _unknown_keys(payload, _LIMITS_ROOT_KEYS)
    if unknown:
        return f"qualification limits root has unknown keys: {unknown!r}"
    if type(payload.get("schema_version")) is not str or payload["schema_version"] != QUALIFICATION_SCHEMA_VERSION:
        return (
            "qualification limits schema_version must be the string "
            f"{QUALIFICATION_SCHEMA_VERSION!r}"
        )
    if not isinstance(payload.get("profile_id"), str) or not payload["profile_id"].strip():
        return "qualification limits profile_id must be a non-empty string"
    if type(payload.get("validated")) is not bool:
        return "qualification limits validated must be boolean"
    if "description" in payload and (
        not isinstance(payload["description"], str) or not payload["description"].strip()
    ):
        return "qualification limits description must be a non-empty string"

    validated = payload["validated"]
    for section_name, field_names in _REQUIRED_LIMIT_FIELDS.items():
        section = payload.get(section_name)
        if section is None and not validated:
            continue
        if not isinstance(section, dict):
            return f"qualification limits {section_name} must be a mapping"
        unknown = _unknown_keys(section, set(field_names))
        if unknown:
            return (
                f"qualification limits {section_name} has unknown keys: {unknown!r}"
            )
        if validated:
            missing = {
                field_name
                for field_name in field_names
                if field_name not in section
                and f"{section_name}.{field_name}"
                not in _OPTIONAL_VALIDATED_LIMIT_FIELDS
            }
            if missing:
                return (
                    f"qualification limits {section_name} is missing keys: "
                    f"{sorted(missing)!r}"
                )
        for field_name, value in section.items():
            qualified = f"{section_name}.{field_name}"
            if value is None and (
                not validated or qualified in _OPTIONAL_VALIDATED_LIMIT_FIELDS
            ):
                continue
            if qualified in _INTEGER_LIMIT_FIELDS:
                if type(value) is not int:
                    return f"qualification limits {qualified} must be an integer"
                number = float(value)
            else:
                if type(value) not in (int, float):
                    return f"qualification limits {qualified} must be numeric"
                number = float(value)
            if not math.isfinite(number):
                return f"qualification limits {qualified} must be finite"
            if number < 0:
                return f"qualification limits {qualified} must be nonnegative"
            if qualified == "telemetry.expected_interval_ms" and value < 500:
                return (
                    "qualification limits telemetry.expected_interval_ms "
                    "must be an integer >= 500"
                )
            if qualified == "telemetry.min_coverage_ratio" and not 0 < number <= 1:
                return (
                    "qualification limits telemetry.min_coverage_ratio "
                    "must be > 0 and <= 1"
                )
    return None


def _read_yaml(path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    payload, error, _identity = _read_yaml_with_identity(path)
    return payload, error


def _read_yaml_with_identity(
    path: Path | None,
) -> tuple[dict[str, Any] | None, str | None, dict[str, object] | None]:
    if path is None:
        return None, "qualification limits were not provided", None
    identity: dict[str, object] | None = None
    try:
        raw, identity = read_stable_regular_file_with_identity(
            path, max_bytes=MAX_LIMITS_BYTES, label="qualification limits"
        )
        text = raw.decode("utf-8", errors="strict")
        payload = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError:
        return None, "qualification limits must be strict UTF-8", identity
    except ValueError as exc:
        return None, str(exc), identity
    except (OSError, yaml.YAMLError) as exc:
        return None, f"qualification limits could not be read: {exc}", identity
    if not isinstance(payload, dict):
        return None, "qualification limits must contain a mapping", identity
    schema_error = _strict_limits_schema_error(payload)
    if schema_error:
        return None, schema_error, identity
    return payload, None, identity


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _optional_int(value: object) -> int | None:
    number = _optional_float(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _csv_nonnegative_int(value: object) -> int | None:
    """Parse an exact base-10 nonnegative integer from a CSV cell."""

    if not isinstance(value, str):
        return None
    text = value.strip()
    return int(text) if text.isascii() and text.isdigit() else None


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return None


def _validated_limits_errors(limits: dict[str, Any]) -> list[str]:
    """Return schema errors for a profile claiming validated pass/fail limits."""

    errors: list[str] = []
    for section_name, field_names in _REQUIRED_LIMIT_FIELDS.items():
        section = limits.get(section_name)
        if not isinstance(section, dict):
            errors.append(f"{section_name} must be a mapping")
            continue
        for field_name in field_names:
            qualified = f"{section_name}.{field_name}"
            if (
                qualified in _OPTIONAL_VALIDATED_LIMIT_FIELDS
                and section.get(field_name) is None
            ):
                continue
            value = _optional_float(section.get(field_name))
            if value is None:
                errors.append(f"{qualified} must be finite")
            elif value < 0:
                errors.append(f"{qualified} must be nonnegative")

    telemetry = limits.get("telemetry")
    if isinstance(telemetry, dict):
        interval = _optional_int(telemetry.get("expected_interval_ms"))
        if interval is None or interval < 500:
            errors.append("telemetry.expected_interval_ms must be an integer >= 500")
        coverage = _optional_float(telemetry.get("min_coverage_ratio"))
        if coverage is None or not 0 < coverage <= 1:
            errors.append("telemetry.min_coverage_ratio must be > 0 and <= 1")
    return errors


def _valid_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _platform_provenance_complete(platform: object) -> bool:
    if not isinstance(platform, dict):
        return False
    if not all(_nonempty_string(platform.get(name)) for name in _REQUIRED_PLATFORM_FIELDS):
        return False
    packages = platform.get("packages")
    return isinstance(packages, dict) and all(
        _nonempty_string(packages.get(name)) for name in _REQUIRED_PLATFORM_PACKAGES
    )


def _native_identity_valid(value: object) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("available") is True
        and _valid_sha256(value.get("sha256"))
        and isinstance(value.get("size_bytes"), int)
        and not isinstance(value.get("size_bytes"), bool)
        and value["size_bytes"] > 0
    )


_EFFECTIVE_RUNTIME_ARTIFACTS = frozenset(
    {
        "deepstream_config", "pose_sidecar", "class_labels",
        "keypoint_labels", "onnx", "engine", "custom_parser",
    }
)
_RUN_LOCAL_EFFECTIVE_RUNTIME_ARTIFACTS = frozenset(
    {"deepstream_config", "pose_sidecar", "class_labels", "keypoint_labels"}
)


def _effective_runtime_identity_valid(
    run_dir: Path,
    inference: dict[str, Any],
    native_plugins: dict[str, Any],
) -> bool:
    """Validate the exact inference artifacts bound at model-loaded readiness."""

    effective = inference.get("effective_runtime")
    model = inference.get("model_package")
    if (
        not isinstance(effective, dict)
        or set(effective) != _EFFECTIVE_RUNTIME_ARTIFACTS
        or not isinstance(model, dict)
    ):
        return False
    if any(
        not _native_identity_valid(identity)
        or not _nonempty_string(identity.get("path"))
        for identity in effective.values()
        if isinstance(identity, dict)
    ) or any(not isinstance(identity, dict) for identity in effective.values()):
        return False
    if effective["onnx"].get("sha256") != model.get("onnx_sha256"):
        return False
    if effective["engine"].get("sha256") != model.get("engine_sha256"):
        return False
    selected_parser = native_plugins.get("deepstream_yolo_parser")
    if not isinstance(selected_parser, dict) or any(
        effective["custom_parser"].get(name) != selected_parser.get(name)
        for name in ("path", "size_bytes", "sha256")
    ):
        return False
    root = Path(run_dir).resolve()
    for name in _RUN_LOCAL_EFFECTIVE_RUNTIME_ARTIFACTS:
        recorded = effective[name]
        path = Path(str(recorded["path"])).resolve()
        if not path.is_relative_to(root):
            return False
        current = stable_file_identity(path, max_bytes=MAX_TASK_CONFIG_BYTES * 4)
        if any(
            current.get(field) != recorded.get(field)
            for field in ("path", "available", "size_bytes", "sha256")
        ):
            return False
    return True


def _preflight_evidence_valid(value: object) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("schema_version") == "3.0"
        and value.get("passed") is True
        and value.get("skipped") is False
        and value.get("ffprobe_available") is True
        and value.get("video_decode_validated") is True
        and value.get("new_streammux_validated") is True
        and value.get("automatic_suspend_disabled") is True
        and isinstance(value.get("output_size_bytes"), int)
        and not isinstance(value.get("output_size_bytes"), bool)
        and value["output_size_bytes"] > 0
        and _valid_sha256(value.get("output_sha256"))
    )


def _duration_seconds(status: dict[str, Any]) -> float | None:
    start = status.get("started_at")
    end = status.get("capture_closed_at") or status.get("stopped_at")
    if not start or not end:
        return None
    try:
        duration = (datetime.fromisoformat(str(end)) - datetime.fromisoformat(str(start))).total_seconds()
    except ValueError:
        return None
    return duration if duration >= 0 else None


def _scan_system(path: Path, expected_interval_ms: int, capture_duration_s: float | None) -> dict[str, Any]:
    maxima: dict[str, float | None] = {name: None for name in _SYSTEM_MAX_FIELDS}
    valid_counts: dict[str, int] = {name: 0 for name in _SYSTEM_MAX_FIELDS}
    report: dict[str, Any] = {
        "path": str(path),
        "present": path.is_file(),
        "schema_valid": False,
        "sample_count": 0,
        "ok_rows": 0,
        "partial_rows": 0,
        "malformed_rows": 0,
        "invalid_rows": 0,
        "first_monotonic_ns": None,
        "last_monotonic_ns": None,
        "observed_duration_s": None,
        "expected_interval_ms": expected_interval_ms,
        "expected_samples": None,
        "coverage_ratio": None,
        "longest_gap_s": None,
        "throttle_sample_count": 0,
        "thermal_status_valid_count": 0,
        "thermal_status_coverage_ratio": None,
        "throttle_duration_s_approx": 0.0,
        "resource_maxima": maxima,
        "resource_valid_counts": valid_counts,
        "resource_coverage_ratios": {name: None for name in _SYSTEM_MAX_FIELDS},
    }
    if not path.is_file():
        return report

    previous_ns: int | None = None
    previous_throttled = False
    first_ns: int | None = None
    last_ns: int | None = None
    longest_gap_ns = 0
    try:
        reader = csv.DictReader(bounded_csv_lines(path))
        required = {"host_monotonic_ns", "parse_status", "thermal_throttled"}
        report["schema_valid"] = required.issubset(reader.fieldnames or ())
        for row in reader:
            report["sample_count"] += 1
            row_invalid = None in row
            status = str(row.get("parse_status") or "").strip().lower()
            if status == "ok":
                report["ok_rows"] += 1
            elif status == "partial":
                report["partial_rows"] += 1
            else:
                report["malformed_rows"] += 1

            monotonic_ns = _csv_nonnegative_int(row.get("host_monotonic_ns"))
            if monotonic_ns is None:
                row_invalid = True
            else:
                if first_ns is None:
                    first_ns = monotonic_ns
                    previous_ns = monotonic_ns
                    last_ns = monotonic_ns
                elif previous_ns is not None:
                    gap_ns = monotonic_ns - previous_ns
                    if gap_ns < 0:
                        report["invalid_rows"] += 1
                    else:
                        longest_gap_ns = max(longest_gap_ns, gap_ns)
                        if previous_throttled:
                            report["throttle_duration_s_approx"] += gap_ns / 1_000_000_000.0
                        previous_ns = monotonic_ns
                        last_ns = monotonic_ns

            throttled = _optional_bool(row.get("thermal_throttled"))
            if throttled is None:
                row_invalid = True
            else:
                report["thermal_status_valid_count"] += 1
            previous_throttled = throttled is True
            if previous_throttled:
                report["throttle_sample_count"] += 1

            values = {name: _optional_float(row.get(name)) for name in maxima if name != "swap_pct"}
            swap_used = _optional_float(row.get("swap_used_mb"))
            swap_total = _optional_float(row.get("swap_total_mb"))
            values["swap_pct"] = (
                swap_used / swap_total * 100.0
                if swap_used is not None and swap_total is not None and swap_total > 0
                else None
            )
            for name, value in values.items():
                if value is not None and value >= 0:
                    valid_counts[name] += 1
                    current = maxima[name]
                    maxima[name] = value if current is None else max(current, value)
                elif name in _REQUIRED_RESOURCE_METRICS:
                    row_invalid = True
            # JetPack releases do not expose one stable EMC-utilization field.
            # JP 7.2.1 tegrastats omits EMC utilization, while the platform
            # sampler still supplies the measurable clock/max-clock ratio.
            # Require at least one independently observed EMC signal rather
            # than invalidating an otherwise complete row for an unavailable
            # vendor field.
            if not any(values[name] is not None and values[name] >= 0 for name in _EMC_RESOURCE_METRICS):
                row_invalid = True
            if row_invalid:
                report["invalid_rows"] += 1
    except (OSError, csv.Error, BoundedCsvError):
        report["invalid_rows"] += 1
        return report

    report["first_monotonic_ns"] = first_ns
    report["last_monotonic_ns"] = last_ns
    if first_ns is not None and last_ns is not None:
        report["observed_duration_s"] = (last_ns - first_ns) / 1_000_000_000.0
        report["longest_gap_s"] = longest_gap_ns / 1_000_000_000.0
    if previous_throttled:
        report["throttle_duration_s_approx"] += expected_interval_ms / 1000.0
    report["throttle_duration_s_approx"] = round(
        float(report["throttle_duration_s_approx"]), 6
    )
    if capture_duration_s is not None:
        expected = max(1, math.floor(capture_duration_s * 1000.0 / expected_interval_ms) + 1)
        report["expected_samples"] = expected
        report["coverage_ratio"] = report["sample_count"] / expected
    if report["sample_count"]:
        sample_count = report["sample_count"]
        report["thermal_status_coverage_ratio"] = (
            report["thermal_status_valid_count"] / sample_count
        )
        report["resource_coverage_ratios"] = {
            name: count / sample_count for name, count in valid_counts.items()
        }
    return report


MAX_QUALIFICATION_CAMERA_COUNT = 64


def _scan_recording_file(
    path: Path, *, expected_stream_ids: frozenset[int]
) -> dict[str, Any]:
    maxima: dict[str, float | None] = {name: None for name in _RECORDING_MAX_FIELDS}
    report: dict[str, Any] = {
        "path": str(path),
        "present": path.is_file(),
        "schema_valid": False,
        "row_count": 0,
        "invalid_rows": 0,
        "event_counts": {name: 0 for name in sorted(_KNOWN_RECORDING_EVENTS)},
        "unknown_event_count": 0,
        "unexpected_stream_id_count": 0,
        "stream_ids": [],
        "maxima": maxima,
    }
    if not path.is_file():
        return report
    try:
        reader = csv.DictReader(bounded_csv_lines(path))
        fieldnames = reader.fieldnames or []
        report["schema_valid"] = (
            _RECORDING_REQUIRED_FIELDS.issubset(fieldnames)
            and len(fieldnames) == len(set(fieldnames))
        )
        # Only configured stream IDs are retained. Unexpected values are
        # counted as invalid evidence without allowing an attacker-controlled
        # telemetry column to grow this set for the duration of a long scan.
        stream_ids: set[int] = set()
        for row in reader:
            report["row_count"] += 1
            row_invalid = None in row
            event = str(row.get("event") or "").strip()
            if event in _KNOWN_RECORDING_EVENTS:
                report["event_counts"][event] += 1
            else:
                report["unknown_event_count"] += 1
                row_invalid = True
            host_monotonic_ns = _csv_nonnegative_int(row.get("host_monotonic_ns"))
            if host_monotonic_ns is None:
                row_invalid = True
            stream_id = _csv_nonnegative_int(row.get("stream_id"))
            if stream_id is None:
                row_invalid = True
            elif stream_id not in expected_stream_ids:
                report["unexpected_stream_id_count"] += 1
                row_invalid = True
            else:
                stream_ids.add(stream_id)
            for name in maxima:
                raw = row.get(name)
                value = _optional_float(raw)
                if value is not None and value >= 0:
                    current = maxima[name]
                    maxima[name] = value if current is None else max(current, value)
                elif name not in {"queue_wait_ms", "encoder_latency_ms"} or str(
                    raw or ""
                ).strip():
                    row_invalid = True
            pts = row.get("pts_ns")
            if event == "closed":
                if str(pts or "").strip():
                    row_invalid = True
            elif _csv_nonnegative_int(pts) is None:
                row_invalid = True
            egress = row.get("egress_timestamp_ns")
            if str(egress or "").strip() and _csv_nonnegative_int(egress) is None:
                row_invalid = True
            correlation = str(row.get("encoder_correlation") or "").strip()
            if correlation not in {"", "pts", "fifo", "unmatched"}:
                row_invalid = True
            if row_invalid:
                report["invalid_rows"] += 1
        report["stream_ids"] = sorted(stream_ids)
    except (OSError, csv.Error, BoundedCsvError):
        report["invalid_rows"] += 1
    return report


def _scan_recording(
    run_dir: Path, *, expected_camera_count: int | None
) -> dict[str, Any]:
    diagnostics = run_dir / "diagnostics"
    valid_expected_count = (
        expected_camera_count
        if type(expected_camera_count) is int
        and 1 <= expected_camera_count <= MAX_QUALIFICATION_CAMERA_COUNT
        else None
    )
    expected_stream_ids = frozenset(range(valid_expected_count or 0))
    # A conforming run owns exactly one recording telemetry file per camera.
    # Retain at most that many reports and inspect only one additional directory
    # entry to prove overflow. When the manifest count is unusable, retain one
    # file solely to produce bounded diagnostic evidence.
    file_limit = valid_expected_count or 1
    paths: list[Path] = []
    file_count = 0
    file_count_overflow = False
    for path in diagnostics.glob("recording*.csv"):
        file_count += 1
        if len(paths) < file_limit:
            paths.append(path)
            continue
        file_count_overflow = True
        break
    paths.sort()
    files = [
        _scan_recording_file(path, expected_stream_ids=expected_stream_ids)
        for path in paths
    ]
    aggregate_maxima: dict[str, float | None] = {name: None for name in _RECORDING_MAX_FIELDS}
    event_counts = {name: 0 for name in sorted(_KNOWN_RECORDING_EVENTS)}
    stream_ids: set[int] = set()
    for report in files:
        for name, value in report["maxima"].items():
            if value is not None:
                current = aggregate_maxima[name]
                aggregate_maxima[name] = value if current is None else max(current, value)
        for name, count in report["event_counts"].items():
            event_counts[name] += count
        stream_ids.update(report["stream_ids"])
    return {
        "present": bool(files),
        # When overflow is true this is a lower bound, which is sufficient to
        # fail closed without enumerating an unbounded directory.
        "file_count": file_count,
        "file_count_overflow": file_count_overflow,
        "retained_file_count": len(files),
        "schema_valid": bool(files) and all(item["schema_valid"] for item in files),
        "row_count": sum(item["row_count"] for item in files),
        "invalid_rows": sum(item["invalid_rows"] for item in files),
        "unknown_event_count": sum(item["unknown_event_count"] for item in files),
        "unexpected_stream_id_count": sum(
            item["unexpected_stream_id_count"] for item in files
        ),
        "stream_ids": sorted(stream_ids),
        "event_counts": event_counts,
        "maxima": aggregate_maxima,
        "files": files,
    }


def _all_zero(values: object) -> bool | None:
    if not isinstance(values, dict) or not values:
        return None
    try:
        return all(int(value) == 0 for value in values.values())
    except (TypeError, ValueError):
        return False


def _task_config_snapshot_valid(run_dir: Path, manifest: dict[str, Any]) -> bool:
    identity = manifest.get("task_config")
    if not isinstance(identity, dict):
        return False
    original_path = identity.get("original_path")
    relative_path = identity.get("snapshot_path")
    expected_hash = identity.get("sha256")
    expected_size = _optional_int(identity.get("size_bytes"))
    if (
        not isinstance(original_path, str)
        or not original_path.strip()
        or not isinstance(relative_path, str)
        or relative_path != "config/task.yaml"
        or not _valid_sha256(expected_hash)
        or expected_size is None
        or expected_size < 0
        or expected_size > MAX_TASK_CONFIG_BYTES
    ):
        return False
    path = run_dir / relative_path
    try:
        raw = read_stable_regular_file(
            path, max_bytes=MAX_TASK_CONFIG_BYTES, label="task config snapshot"
        )
        if len(raw) != expected_size:
            return False
        return hashlib.sha256(raw).hexdigest() == expected_hash
    except (OSError, ValueError):
        return False


def _integrity_gates(
    status: dict[str, Any],
    manifest: dict[str, Any],
    run_dir: Path,
    *,
    current_recording_evidence: object = None,
    alignment_summary: object = None,
) -> dict[str, bool | None]:
    state = str(status.get("state") or "")
    recording = status.get("recording_validation")
    acquisition = status.get("acquisition_integrity")
    reconciliation = status.get("capture_reconciliation")
    serial_enabled = bool((manifest.get("serial") or {}).get("enabled"))
    trigger_enabled = (manifest.get("capture") or {}).get("trigger_on") is True
    alignment_required = serial_enabled and trigger_enabled
    capture_exit_code = status.get("capture_exit_code")
    parsed_exit_code = _optional_int(capture_exit_code)
    inference = (
        manifest.get("inference")
        if isinstance(manifest.get("inference"), dict)
        else {}
    )
    inference_enabled = inference.get("enabled")
    preview_enabled = inference.get("preview_enabled")
    model = (
        inference.get("model_package")
        if isinstance(inference.get("model_package"), dict)
        else {}
    )
    observability = (
        manifest.get("observability")
        if isinstance(manifest.get("observability"), dict)
        else {}
    )
    platform = manifest.get("platform")
    git = manifest.get("git") if isinstance(manifest.get("git"), dict) else {}
    native_plugins = (
        manifest.get("native_plugins")
        if isinstance(manifest.get("native_plugins"), dict)
        else {}
    )
    preflight_completed = observability.get("preflight_skipped") is False
    production_debug_disabled = (
        observability.get("deepstream_debug_profile") is False
    )
    topology = (
        manifest.get("process_topology")
        if isinstance(manifest.get("process_topology"), dict)
        else {}
    )
    durable_acquisition_owner = (
        topology.get("acquisition_owner") == "durable_supervisor"
    )
    manifest_preflight = manifest.get("preflight")
    status_preflight = status.get("preflight")
    backend_preflight_verified = (
        _preflight_evidence_valid(manifest_preflight)
        and manifest_preflight == status_preflight
    )
    terminal_success = (
        True
        if state in TERMINAL_SUCCESS_STATES
        else False
        if state
        else None
    )

    def alignment_valid() -> bool:
        if status.get("alignment_validated") is not True:
            return False
        if not isinstance(alignment_summary, dict):
            return False
        alignment = alignment_summary.get("frame_alignment")
        counts = alignment_summary.get("counts")
        validation = alignment_summary.get("validation")
        markers = alignment_summary.get("markers")
        marker_indices = alignment.get("marker_indices") if isinstance(alignment, dict) else None
        required_markers = (
            "START_SENT", "CAPTURE_STOP_REQUESTED", "STOP_SENT", "CAPTURE_STOP_DONE"
        )
        recorded_frames = counts.get("recorded_frames") if isinstance(counts, dict) else None
        epoch_highs = (
            alignment.get("controller_high_events_in_epoch")
            if isinstance(alignment, dict)
            else None
        )
        return bool(
            alignment_summary.get("schema_version") == "2.0"
            and alignment_summary.get("run_dir") == str(run_dir)
            and alignment_summary.get("start_marker_seen") is True
            and isinstance(markers, dict)
            and set(markers) == set(required_markers)
            and isinstance(marker_indices, dict)
            and set(marker_indices) == set(required_markers)
            and all(type(marker_indices[name]) is int for name in required_markers)
            and len({marker_indices[name] for name in required_markers})
            == len(required_markers)
            and [marker_indices[name] for name in required_markers]
            == sorted(marker_indices[name] for name in required_markers)
            and isinstance(alignment, dict)
            and alignment.get("method")
            == "first_recorded_frame_to_first_camera_high_after_start_sent"
            and alignment.get("epoch_markers_complete") is True
            and alignment.get("validated") is True
            and type(recorded_frames) is int
            and recorded_frames > 0
            and epoch_highs == recorded_frames
            and alignment.get("validated_pairs") == recorded_frames
            and alignment.get("controller_high_events_unmatched") == 0
            and alignment.get("shutdown_tail_high_events_unmatched") == 0
            and isinstance(alignment.get("boundary_tail_policy"), str)
            and bool(alignment["boundary_tail_policy"].strip())
            and isinstance(validation, dict)
            and validation.get("video_frame_count_matches_frames_csv") is True
        )

    def recording_validation_valid() -> bool:
        if (
            not isinstance(recording, dict)
            or recording.get("schema_version") != "2.0"
            or recording.get("passed") is not True
            or recording.get("evidence_unchanged_during_validation") is not True
        ):
            return False
        cameras = recording.get("cameras")
        capture = manifest.get("capture")
        camera_count = capture.get("num_cameras") if isinstance(capture, dict) else None
        return bool(
            isinstance(cameras, list)
            and cameras
            and type(camera_count) is int
            and len(cameras) == camera_count
            and all(
                isinstance(camera, dict)
                and camera.get("stream_id") == stream_id
                and camera.get("exists") is True
                and camera.get("nonzero_frame_count") is True
                and camera.get("source_count_matches") is True
                and camera.get("frame_count_matches") is True
                and isinstance(camera.get("frame_count_method"), str)
                and camera["frame_count_method"].startswith("full_decode_")
                and type(camera.get("source_frames")) is int
                and camera["source_frames"] > 0
                and camera.get("record_admitted_frames") == camera["source_frames"]
                and camera.get("video_frames") == camera["source_frames"]
                for stream_id, camera in enumerate(cameras)
            )
        )

    def reconciliation_counts_valid() -> bool | None:
        capture = manifest.get("capture")
        camera_count = capture.get("num_cameras") if isinstance(capture, dict) else None
        cameras = recording.get("cameras") if isinstance(recording, dict) else None
        if not isinstance(reconciliation, dict) or type(camera_count) is not int:
            return None
        if not isinstance(cameras, list) or len(cameras) != camera_count:
            return False
        maps = {
            name: reconciliation.get(name)
            for name in (
                "source_frames", "record_admitted_frames", "source_not_recorded_frames"
            )
        }
        if any(not isinstance(value, dict) for value in maps.values()):
            return False
        expected_keys = {str(index) for index in range(camera_count)}
        normalized = {
            name: {str(key): value for key, value in value.items()}
            for name, value in maps.items()
        }
        if any(set(value) != expected_keys for value in normalized.values()):
            return False
        return all(
            type(normalized["source_frames"][str(index)]) is int
            and normalized["source_frames"][str(index)] > 0
            and normalized["record_admitted_frames"][str(index)]
            == normalized["source_frames"][str(index)]
            and normalized["source_not_recorded_frames"][str(index)] == 0
            and isinstance(cameras[index], dict)
            and cameras[index].get("source_frames")
            == normalized["source_frames"][str(index)]
            and cameras[index].get("record_admitted_frames")
            == normalized["record_admitted_frames"][str(index)]
            for index in range(camera_count)
        )

    def preview_attribution_valid() -> bool:
        report = status.get("preview_attribution")
        capture = manifest.get("capture")
        camera_count = capture.get("num_cameras") if isinstance(capture, dict) else None
        cameras = report.get("cameras") if isinstance(report, dict) else None
        if (
            not isinstance(report, dict)
            or report.get("schema_version") != "1.0"
            or report.get("required") is not True
            or report.get("passed") is not True
            or type(camera_count) is not int
            or not isinstance(cameras, list)
            or len(cameras) != camera_count
        ):
            return False
        expected_names = {
            f"preview_{boundary}{'' if stream_id == 0 else f'_cam{stream_id}'}.csv"
            for stream_id in range(camera_count)
            for boundary in ("admission", "delivery")
        }
        observed_names: set[str] = set()
        try:
            with os.scandir(run_dir / "diagnostics") as entries:
                for entry in entries:
                    if not Path(entry.name).match("preview_*.csv"):
                        continue
                    if len(observed_names) >= len(expected_names) + 1:
                        return False
                    observed_names.add(entry.name)
        except OSError:
            return False
        if observed_names != expected_names:
            return False
        for stream_id, camera in enumerate(cameras):
            if not isinstance(camera, dict) or camera.get("stream_id") != stream_id:
                return False
            admitted = _optional_int(camera.get("admitted_frames"))
            delivered = _optional_int(camera.get("delivered_frames"))
            shed = _optional_int(camera.get("shed_frames"))
            unmatched = _optional_int(camera.get("unmatched_delivery_frames"))
            if (
                admitted is None
                or admitted <= 0
                or delivered is None
                or shed is None
                or unmatched != 0
                or delivered + shed != admitted
            ):
                return False
            suffix = "" if stream_id == 0 else f"_cam{stream_id}"
            for boundary in ("admission", "delivery"):
                expected = camera.get(f"{boundary}_identity")
                current = stable_file_identity(
                    run_dir / "diagnostics" / f"preview_{boundary}{suffix}.csv"
                )
                if not isinstance(expected, dict) or any(
                    expected.get(field) != current.get(field)
                    for field in ("path", "available", "size_bytes", "sha256")
                ):
                    return False
        return True
    return {
        "manifest_schema_supported": (
            str(manifest.get("schema_version")) == "2.0"
            if manifest
            else None
        ),
        "production_eligible": (
            manifest.get("production_eligible") is True
            and status.get("production_eligible") is True
            and not manifest.get("failure_injection")
            and not status.get("failure_injection")
            and preflight_completed
            and backend_preflight_verified
            and production_debug_disabled
            and durable_acquisition_owner
            if manifest
            else None
        ),
        "durable_acquisition_owner": (
            durable_acquisition_owner if manifest else None
        ),
        "task_config_snapshot_valid": (
            _task_config_snapshot_valid(run_dir, manifest) if manifest else None
        ),
        "preflight_completed": preflight_completed if manifest else None,
        "backend_preflight_verified": (
            backend_preflight_verified if manifest else None
        ),
        "system_telemetry_not_degraded": (
            status.get("system_telemetry_degraded") is not True
            if status
            else None
        ),
        "production_debug_profile_disabled": (
            production_debug_disabled if manifest else None
        ),
        "platform_provenance_complete": (
            _platform_provenance_complete(platform) if manifest else None
        ),
        "git_commit_recorded": (
            _nonempty_string(git.get("commit")) if manifest else None
        ),
        "git_worktree_clean": (git.get("dirty") is False if manifest else None),
        "flir_native_plugin_identified": (
            _native_identity_valid(native_plugins.get("flir_gstreamer_source"))
            if manifest
            else None
        ),
        "inference_native_plugin_identified_when_required": (
            True
            if inference_enabled is False
            else _native_identity_valid(native_plugins.get("deepstream_yolo_parser"))
            if inference_enabled is True
            else None
        ),
        "engine_identity_validated_when_required": (
            True
            if inference_enabled is False
            else (
                model.get("model_manifest_schema") == 3
                and isinstance(model.get("engine_build_identity"), dict)
                and bool(model.get("engine_build_identity"))
            )
            if inference_enabled is True
            else None
        ),
        "model_content_identity_validated_when_required": (
            True
            if inference_enabled is False
            else all(_valid_sha256(model.get(name)) for name in _MODEL_SHA256_FIELDS)
            if inference_enabled is True
            else None
        ),
        "effective_runtime_identity_validated_when_required": (
            True
            if inference_enabled is False
            else _effective_runtime_identity_valid(
                run_dir, inference, native_plugins
            )
            if inference_enabled is True
            else None
        ),
        "terminal_success_state": terminal_success,
        "capture_exit_zero": parsed_exit_code == 0 if parsed_exit_code is not None else None,
        "recording_validation_passed": (
            recording_validation_valid()
            if isinstance(recording, dict)
            else None
        ),
        "recording_evidence_unchanged": (
            recording_evidence_same_content(
                recording.get("evidence"),
                current_recording_evidence,
                int((manifest.get("capture") or {}).get("num_cameras")),
            )
            if isinstance(recording, dict)
            and type((manifest.get("capture") or {}).get("num_cameras")) is int
            and 1 <= (manifest.get("capture") or {}).get("num_cameras") <= 64
            else None
        ),
        "preview_attribution_complete_when_required": (
            True
            if preview_enabled is False
            else (
                preview_attribution_valid()
            )
            if preview_enabled is True
            else None
        ),
        "acquisition_integrity_passed": (
            bool(acquisition.get("passed"))
            if isinstance(acquisition, dict) and "passed" in acquisition
            else None
        ),
        "source_not_recorded_zero": (
            reconciliation_counts_valid()
        ),
        "alignment_validated_when_required": (
            alignment_valid() if alignment_required else True
        ),
        "overall_validation_passed": (
            bool(status.get("overall_validation_passed"))
            if "overall_validation_passed" in status
            else None
        ),
    }


def _factors(manifest: dict[str, Any]) -> dict[str, Any]:
    capture = manifest.get("capture") if isinstance(manifest.get("capture"), dict) else {}
    inference = manifest.get("inference") if isinstance(manifest.get("inference"), dict) else {}
    platform = manifest.get("platform") if isinstance(manifest.get("platform"), dict) else {}
    model = inference.get("model_package") if isinstance(inference.get("model_package"), dict) else {}
    return {
        "width": capture.get("width"),
        "height": capture.get("height"),
        "fps": capture.get("fps"),
        "camera_count": capture.get("num_cameras"),
        "pixel_format": capture.get("pixel_format"),
        "trigger_enabled": capture.get("trigger_on"),
        "trigger_on": capture.get("trigger_on"),
        "trigger_activation": capture.get("trigger_activation"),
        "arduino_fps": capture.get("arduino_fps"),
        "exposure_us": capture.get("exposure_us"),
        "bitrate_kbps": inference.get("bitrate_kbps"),
        "serial_enabled": (manifest.get("serial") or {}).get("enabled")
        if isinstance(manifest.get("serial"), dict)
        else None,
        "serial_port": (manifest.get("serial") or {}).get("port")
        if isinstance(manifest.get("serial"), dict)
        else None,
        "serial_baud": (manifest.get("serial") or {}).get("baud")
        if isinstance(manifest.get("serial"), dict)
        else None,
        "inference_enabled": inference.get("enabled"),
        "preview_enabled": inference.get(
            "preview_enabled", bool(inference.get("preview_sockets"))
        ),
        "model_name": model.get("name"),
        "nvpmodel": platform.get("nvpmodel"),
        "git_commit": (manifest.get("git") or {}).get("commit")
        if isinstance(manifest.get("git"), dict)
        else None,
        "git_dirty": (manifest.get("git") or {}).get("dirty")
        if isinstance(manifest.get("git"), dict)
        else None,
        "production_eligible": manifest.get("production_eligible"),
    }


def _limit_checks(
    limits: dict[str, Any], system: dict[str, Any], recording: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    checks: list[dict[str, Any]] = []
    incomplete: list[str] = []

    def maximum(name: str, observed: object, limit: object) -> None:
        if limit is None:
            return
        value = _optional_float(observed)
        threshold = _optional_float(limit)
        if threshold is None:
            incomplete.append(f"limit {name} is invalid")
            return
        if value is None:
            incomplete.append(f"measurement {name} is unavailable")
            return
        checks.append(
            {
                "name": name,
                "operator": "<=",
                "observed": value,
                "limit": threshold,
                "passed": value <= threshold,
            }
        )

    def minimum(name: str, observed: object, limit: object) -> None:
        if limit is None:
            return
        value = _optional_float(observed)
        threshold = _optional_float(limit)
        if threshold is None:
            incomplete.append(f"limit {name} is invalid")
            return
        if value is None:
            incomplete.append(f"measurement {name} is unavailable")
            return
        checks.append(
            {
                "name": name,
                "operator": ">=",
                "observed": value,
                "limit": threshold,
                "passed": value >= threshold,
            }
        )

    telemetry_limits = limits.get("telemetry") if isinstance(limits.get("telemetry"), dict) else {}
    minimum(
        "telemetry.coverage_ratio",
        system.get("coverage_ratio"),
        telemetry_limits.get("min_coverage_ratio"),
    )
    minimum(
        "telemetry.thermal_status_coverage_ratio",
        system.get("thermal_status_coverage_ratio"),
        telemetry_limits.get("min_coverage_ratio"),
    )
    maximum("telemetry.longest_gap_s", system.get("longest_gap_s"), telemetry_limits.get("max_gap_s"))
    maximum(
        "telemetry.malformed_rows",
        system.get("malformed_rows"),
        telemetry_limits.get("max_malformed_rows"),
    )
    maximum("telemetry.invalid_rows", system.get("invalid_rows"), telemetry_limits.get("max_invalid_rows"))
    maximum(
        "telemetry.throttle_sample_count",
        system.get("throttle_sample_count"),
        telemetry_limits.get("max_throttle_samples"),
    )

    resource_limits = limits.get("resources") if isinstance(limits.get("resources"), dict) else {}
    for key, limit in resource_limits.items():
        if key.startswith("max_"):
            metric = key.removeprefix("max_")
            minimum(
                f"telemetry.{metric}_coverage_ratio",
                system["resource_coverage_ratios"].get(metric),
                telemetry_limits.get("min_coverage_ratio"),
            )
            maximum(f"resources.{metric}", system["resource_maxima"].get(metric), limit)

    recording_limits = limits.get("recording") if isinstance(limits.get("recording"), dict) else {}
    for key, limit in recording_limits.items():
        if not key.startswith("max_"):
            continue
        metric = key.removeprefix("max_")
        if metric == "backpressure_fatal_events":
            observed = recording["event_counts"].get("backpressure_fatal")
        else:
            observed = recording["maxima"].get(metric)
        maximum(f"recording.{metric}", observed, limit)
    return checks, incomplete


def qualify_run(
    run_dir: Path,
    *,
    limits_path: Path | None,
    output_path: Path | None = None,
    allow_debug_profile: bool = False,
) -> dict[str, Any]:
    """Evaluate existing evidence and atomically write a derived summary only."""

    run_dir = Path(run_dir).resolve()
    manifest, manifest_identity = run_context.read_json_required_with_identity(
        run_dir / run_context.RUN_MANIFEST_FILENAME
    )
    status, status_identity = run_context.read_json_required_with_identity(
        run_dir / run_context.RUN_STATUS_FILENAME
    )
    limits, limits_error, limits_identity = _read_yaml_with_identity(
        Path(limits_path) if limits_path is not None else None
    )
    expected_interval_ms = 1000
    if limits is not None:
        telemetry_limits = limits.get("telemetry")
        if isinstance(telemetry_limits, dict):
            interval = _optional_int(telemetry_limits.get("expected_interval_ms"))
            if interval is not None:
                expected_interval_ms = max(500, interval)

    capture_duration_s = _duration_seconds(status)
    system = _scan_system(
        run_dir / "diagnostics" / "system.csv",
        expected_interval_ms,
        capture_duration_s,
    )
    capture = manifest.get("capture") if isinstance(manifest.get("capture"), dict) else {}
    camera_count = _optional_int(capture.get("num_cameras"))
    serial_config = manifest.get("serial")
    serial_enabled = (
        serial_config.get("enabled") is True
        if isinstance(serial_config, dict)
        else False
    )
    trigger_enabled = capture.get("trigger_on") is True
    alignment_required = serial_enabled and trigger_enabled
    alignment_summary: dict[str, Any] | None = None
    alignment_identity: dict[str, object] | None = None
    if alignment_required:
        try:
            alignment_summary, alignment_identity = (
                run_context.read_json_required_with_identity(
                run_dir / "alignment_summary.json"
                )
            )
        except (OSError, ValueError):
            alignment_summary = None
            alignment_identity = None
    current_recording_evidence = (
        capture_recording_evidence(run_dir, camera_count)
        if camera_count is not None and 1 <= camera_count <= MAX_QUALIFICATION_CAMERA_COUNT
        else None
    )
    recording = _scan_recording(
        run_dir,
        expected_camera_count=camera_count,
    )
    gates = _integrity_gates(
        status,
        manifest,
        run_dir,
        current_recording_evidence=current_recording_evidence,
        alignment_summary=alignment_summary,
    )
    observability = (
        manifest.get("observability")
        if isinstance(manifest.get("observability"), dict)
        else {}
    )
    debug_latency_evidence = None
    if observability.get("deepstream_debug_profile") is True:
        debug_latency_evidence = deepstream_latency_evidence_for_run(
            run_dir, manifest
        )
        gates["debug_latency_evidence_complete"] = bool(
            debug_latency_evidence.get("readable")
            and debug_latency_evidence.get("recognized")
            and not debug_latency_evidence.get("truncated")
        )
    qualification_exceptions: list[str] = []
    if allow_debug_profile and gates.get("production_eligible") is False:
        manifest_reasons = manifest.get("production_disqualifiers")
        status_reasons = status.get("production_disqualifiers")
        debug_only = (
            isinstance(observability, dict)
            and observability.get("deepstream_debug_profile") is True
            and manifest.get("failure_injection") is None
            and manifest_reasons == ["deepstream_debug_profile"]
            and status_reasons == ["deepstream_debug_profile"]
        )
        if debug_only and gates.get("debug_latency_evidence_complete") is True:
            gates["production_eligible"] = True
            gates["production_debug_profile_disabled"] = True
            qualification_exceptions.append("deepstream_debug_profile")
    failed_reasons: list[str] = []
    incomplete_reasons: list[str] = []

    state = str(status.get("state") or "")
    if state in TERMINAL_FAILURE_STATES:
        failed_reasons.append(f"run ended in failure state {state}")
    for name, passed in gates.items():
        if passed is False:
            failed_reasons.append(f"frame-integrity gate failed: {name}")
        elif passed is None:
            incomplete_reasons.append(f"frame-integrity gate is unavailable: {name}")

    if not system["present"] or not system["schema_valid"] or system["sample_count"] == 0:
        incomplete_reasons.append("bounded system telemetry is missing or unusable")
    if not recording["present"] or not recording["schema_valid"] or recording["row_count"] == 0:
        incomplete_reasons.append("recording-path telemetry is missing or unusable")
    if (
        camera_count is None
        or camera_count < 1
        or camera_count > MAX_QUALIFICATION_CAMERA_COUNT
    ):
        incomplete_reasons.append("recording telemetry camera count is unavailable")
    else:
        expected_streams = list(range(camera_count))
        recording["expected_camera_count"] = camera_count
        recording["expected_stream_ids"] = expected_streams
        if recording["file_count"] != camera_count:
            failed_reasons.append(
                "recording telemetry file coverage failed: expected "
                f"{camera_count}, observed {recording['file_count']}"
            )
        if recording["file_count_overflow"]:
            failed_reasons.append(
                "recording telemetry file enumeration exceeded the configured "
                f"camera count ({camera_count})"
            )
        if recording["stream_ids"] != expected_streams:
            failed_reasons.append(
                "recording telemetry stream coverage failed: expected "
                f"{expected_streams}, observed {recording['stream_ids']}"
            )
        if any(len(item["stream_ids"]) != 1 for item in recording["files"]):
            failed_reasons.append(
                "recording telemetry files must each contain exactly one stream"
            )
    if recording["unknown_event_count"]:
        failed_reasons.append(
            "recording telemetry contains unknown events: "
            f"{recording['unknown_event_count']}"
        )
    if recording["unexpected_stream_id_count"]:
        failed_reasons.append(
            "recording telemetry contains unexpected stream IDs: "
            f"{recording['unexpected_stream_id_count']} rows"
        )
    if recording["invalid_rows"]:
        failed_reasons.append(
            "recording telemetry contains invalid rows: "
            f"{recording['invalid_rows']}"
        )

    checks: list[dict[str, Any]] = []
    limits_declared_validated = bool(limits and limits.get("validated") is True)
    limits_validation_errors = (
        _validated_limits_errors(limits)
        if limits is not None and limits_declared_validated
        else []
    )
    limits_validated = limits_declared_validated and not limits_validation_errors
    if limits_error:
        incomplete_reasons.append(limits_error)
    elif limits_validation_errors:
        incomplete_reasons.extend(
            f"validated limits schema error: {error}"
            for error in limits_validation_errors
        )
    elif not limits_validated:
        incomplete_reasons.append("qualification limits are measurement-only and not validated")
    else:
        checks, limit_incomplete = _limit_checks(limits, system, recording)
        incomplete_reasons.extend(limit_incomplete)
        failed_reasons.extend(
            f"qualification limit failed: {check['name']}"
            for check in checks
            if not check["passed"]
        )

    if failed_reasons:
        result = "failed"
    elif incomplete_reasons:
        result = "incomplete"
    else:
        result = "passed"

    source_evidence = (
        {
            "available": False,
            "error": (
                "recording telemetry file set exceeds the configured camera "
                "count; source hashing was skipped to preserve bounded evaluation"
            ),
            "recording_telemetry": [],
        }
        if recording["file_count_overflow"]
        else capture_source_evidence(
            run_dir,
            limits_path,
            manifest_snapshot=(manifest, manifest_identity),
            status_identity=status_identity,
            limits_identity=limits_identity,
            recording_evidence_snapshot=current_recording_evidence,
            alignment_identity=alignment_identity,
        )
    )

    summary = {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "result": result,
        "run_id": manifest.get("run_id") or status.get("run_id") or run_dir.name,
        "run_directory": str(run_dir),
        "capture_duration_s": capture_duration_s,
        "factors": _factors(manifest),
        "limits": {
            "path": str(Path(limits_path).resolve()) if limits_path is not None else None,
            "input_error": limits_error,
            "profile_id": limits.get("profile_id") if limits else None,
            "schema_version": limits.get("schema_version") if limits else None,
            "validated": limits_validated,
            "declared_validated": limits_declared_validated,
            "validation_errors": limits_validation_errors,
        },
        "frame_integrity_gates": gates,
        "system_telemetry": system,
        "recording_telemetry": recording,
        "debug_latency_evidence": debug_latency_evidence,
        "limit_checks": checks,
        "qualification_exceptions": qualification_exceptions,
        "failed_reasons": failed_reasons,
        "incomplete_reasons": incomplete_reasons,
        "capture_validity_unchanged": True,
        "source_evidence": source_evidence,
    }
    destination = output_path or (run_dir / "qualification_summary.json")
    core_sources = (
        (run_dir / run_context.RUN_MANIFEST_FILENAME, manifest_identity),
        (run_dir / run_context.RUN_STATUS_FILENAME, status_identity),
    )
    if any(
        not stable_file_matches_identity(
            path, identity, max_bytes=run_context.MAX_RUN_METADATA_BYTES
        )
        for path, identity in core_sources
    ):
        raise ValueError("qualification manifest/status changed during evaluation")
    if limits_identity is not None and not stable_file_matches_identity(
        Path(limits_path), limits_identity, max_bytes=MAX_LIMITS_BYTES
    ):
        raise ValueError("qualification limits changed during evaluation")
    if not recording["file_count_overflow"]:
        evidence_errors = source_evidence_errors(
            run_dir,
            source_evidence,
            declared_limits_path=(
                str(Path(limits_path).resolve()) if limits_path is not None else None
            ),
            verify_recording_content=False,
        )
        if evidence_errors:
            raise ValueError(
                "qualification source evidence changed during evaluation: "
                + "; ".join(evidence_errors)
            )
    if (
        camera_count is not None
        and current_recording_evidence is not None
        and recording_evidence_complete(current_recording_evidence, camera_count)
        and not recording_evidence_matches(
            run_dir, current_recording_evidence, camera_count
        )
    ):
        raise ValueError(
            "primary recording artifacts changed during qualification evaluation"
        )
    run_context.atomic_write_json(Path(destination), summary)
    return summary
