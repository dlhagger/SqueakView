"""Compare paired production and DeepStream-debug qualification runs."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from squeakview.common import run_context
from squeakview.common.bounded_input import read_stable_regular_file
from squeakview.common.diagnostics.evidence_identity import (
    acquisition_identity_errors,
    canonical_acquisition_identity,
    canonical_runtime_identity,
    runtime_identity_errors,
    source_evidence_errors,
    stable_file_identity,
    stable_file_matches_identity,
)


_MATCHING_FACTORS = (
    "width",
    "height",
    "fps",
    "camera_count",
    "pixel_format",
    "trigger_enabled",
    "inference_enabled",
    "preview_enabled",
    "model_name",
    "nvpmodel",
    "git_commit",
)

_MODEL_IDENTITY_FIELDS = (
    "model_manifest_sha256",
    "pose_sidecar_sha256",
    "onnx_sha256",
    "config_sha256",
    "engine_sha256",
)
_METRICS = {
    "recording.queue_wait_ms": ("recording_telemetry", "maxima", "queue_wait_ms"),
    "recording.encoder_latency_ms": (
        "recording_telemetry",
        "maxima",
        "encoder_latency_ms",
    ),
    "recording.encoder_in_flight": (
        "recording_telemetry",
        "maxima",
        "encoder_in_flight",
    ),
    "system.cpu_util_mean_pct": (
        "system_telemetry",
        "resource_maxima",
        "cpu_util_mean_pct",
    ),
    "system.gpu_util_pct": (
        "system_telemetry",
        "resource_maxima",
        "gpu_util_pct",
    ),
    "system.temp_max_c": ("system_telemetry", "resource_maxima", "temp_max_c"),
    "system.vdd_in_current_mw": (
        "system_telemetry",
        "resource_maxima",
        "vdd_in_current_mw",
    ),
}
_FINITE_NUMBER = rb"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_LATENCY_PATTERNS = {
    "frame": re.compile(
        rb"\bsource\s+id\s*=\s*\d+.*\bframe_num\s*=\s*\d+.*"
        rb"\bframe\s+latency\s*=\s*(" + _FINITE_NUMBER + rb")\s*\(ms\)",
        re.IGNORECASE,
    ),
    "component": re.compile(
        rb"\bcomp\s+name\s*=\s*\S+.*\bcomponent[_ ]latency\s*=\s*"
        rb"(" + _FINITE_NUMBER + rb")",
        re.IGNORECASE,
    ),
}
_FPS_LINE = re.compile(rb"\*\*FPS:\s*(.*)", re.IGNORECASE)
_FPS_VALUE = re.compile(
    rb"(?:^|\s)(" + _FINITE_NUMBER + rb")\s*\(", re.IGNORECASE
)
_LOG_LIMIT_MARKER = b"[SQUEAKVIEW] diagnostic log size limit reached"
_REQUIRED_DEBUG_PROBES = ("measure_latency_probe", "measure_fps_probe")
DEFAULT_LOG_SCAN_MAX_BYTES = 64 * 1024 * 1024
MAX_THRESHOLD_BYTES = 1 << 20
THRESHOLD_SCHEMA_VERSION = "1.0"
_TERMINAL_SUCCESS_STATES = {"post_run_complete", "analysis_complete", "finalized"}
_THRESHOLD_ROOT_KEYS = {
    "schema_version",
    "profile_id",
    "approved",
    "description",
    "metrics",
}
_THRESHOLD_BOUND_KEYS = {"max_increase", "max_percent_increase"}


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


def _unknown_keys(value: Mapping[Any, Any], allowed: set[str]) -> list[Any]:
    return [key for key in value if not isinstance(key, str) or key not in allowed]


def _threshold_schema_errors(thresholds: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    unknown = _unknown_keys(thresholds, _THRESHOLD_ROOT_KEYS)
    if unknown:
        errors.append(f"threshold document has unknown keys: {unknown!r}")
    if (
        type(thresholds.get("schema_version")) is not str
        or thresholds.get("schema_version") != THRESHOLD_SCHEMA_VERSION
    ):
        errors.append(
            f"threshold schema_version must be the string {THRESHOLD_SCHEMA_VERSION!r}"
        )
    profile_id = thresholds.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id.strip():
        errors.append("threshold profile_id must be a non-empty string")
    approved = thresholds.get("approved")
    if type(approved) is not bool:
        errors.append("threshold approved must be boolean")
    if "description" in thresholds and (
        not isinstance(thresholds["description"], str)
        or not thresholds["description"].strip()
    ):
        errors.append("threshold description must be a non-empty string")
    metrics = thresholds.get("metrics")
    if metrics is None and approved is False:
        return errors
    if not isinstance(metrics, Mapping):
        errors.append("threshold metrics must be a mapping")
        return errors
    unknown_metrics = _unknown_keys(metrics, set(_METRICS))
    if unknown_metrics:
        errors.append(f"threshold metrics has unknown keys: {unknown_metrics!r}")
    if approved is True:
        missing = set(_METRICS) - metrics.keys()
        if missing:
            errors.append(f"threshold metrics are missing: {sorted(missing)!r}")
    for metric_name, policy in metrics.items():
        if metric_name not in _METRICS:
            continue
        if not isinstance(policy, Mapping):
            errors.append(f"threshold for metric {metric_name} must be a mapping")
            continue
        unknown_bounds = _unknown_keys(policy, _THRESHOLD_BOUND_KEYS)
        if unknown_bounds:
            errors.append(
                f"threshold {metric_name} has unknown keys: {unknown_bounds!r}"
            )
        present_bound = False
        for bound_name in _THRESHOLD_BOUND_KEYS:
            value = policy.get(bound_name)
            if value is None and approved is False:
                continue
            if value is None:
                continue
            present_bound = True
            if type(value) not in (int, float):
                errors.append(f"threshold {metric_name}.{bound_name} must be numeric")
                continue
            if not math.isfinite(float(value)) or float(value) < 0:
                errors.append(
                    f"threshold {metric_name}.{bound_name} must be finite and nonnegative"
                )
        if approved is True and not present_bound:
            errors.append(f"threshold for metric {metric_name} has no bound")
    return errors


def load_debug_thresholds(path: Path) -> dict[str, Any]:
    """Load one bounded, strictly typed debug-overhead threshold document."""

    try:
        raw = read_stable_regular_file(
            path, max_bytes=MAX_THRESHOLD_BYTES, label="threshold document"
        )
        text = raw.decode("utf-8", errors="strict")
        payload = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as exc:
        raise ValueError("threshold document must be strict UTF-8") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"threshold document could not be read: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("threshold document must be a mapping")
    errors = _threshold_schema_errors(payload)
    if errors:
        raise ValueError("; ".join(errors))
    return payload


def _read_evidence(
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, object]]]:
    summary, summary_identity = run_context.read_json_required_with_identity(
        Path(run_dir) / "qualification_summary.json"
    )
    manifest, manifest_identity = run_context.read_json_required_with_identity(
        Path(run_dir) / run_context.RUN_MANIFEST_FILENAME
    )
    if not summary:
        raise ValueError(f"qualification summary is missing: {run_dir}")
    if not manifest:
        raise ValueError(f"run manifest is missing: {run_dir}")
    return summary, manifest, {
        "qualification_summary": summary_identity,
        "run_manifest": manifest_identity,
    }


def _debug_enabled(manifest: Mapping[str, Any]) -> bool | None:
    observability = manifest.get("observability")
    if not isinstance(observability, Mapping):
        return None
    value = observability.get("deepstream_debug_profile")
    return value if isinstance(value, bool) else None


def _number(mapping: Mapping[str, Any], *path: str) -> float | None:
    value: Any = mapping
    for name in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _model_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    inference = manifest.get("inference")
    model = inference.get("model_package") if isinstance(inference, Mapping) else None
    if not isinstance(model, Mapping):
        return {}
    return {
        **{name: model.get(name) for name in _MODEL_IDENTITY_FIELDS},
        "engine_build_identity": model.get("engine_build_identity"),
    }


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def scan_deepstream_latency_log(
    path: Path,
    *,
    max_bytes: int = DEFAULT_LOG_SCAN_MAX_BYTES,
    declared_log_limit: int | None = None,
) -> dict[str, Any]:
    """Boundedly identify NVIDIA frame/component latency and Service Maker FPS.

    Only documented labels are recognized. Their numeric values are parsed so
    negative or non-finite latency evidence fails closed.
    """

    path = Path(path)
    result: dict[str, Any] = {
        "path": str(path),
        "present": path.is_file(),
        "readable": False,
        "bytes_scanned": 0,
        "frame_latency_records": 0,
        "component_latency_records": 0,
        "invalid_latency_records": 0,
        "fps_records": 0,
        "invalid_fps_records": 0,
        "frame_latency_max_ms": None,
        "component_latency_max_ms": None,
        "fps_min": None,
        "fps_max": None,
        "frame_latency_recognized": False,
        "component_latency_recognized": False,
        "fps_recognized": False,
        "recognized": False,
        "truncated": False,
        "error": None,
    }
    if not result["present"]:
        return result
    scan_limit = max(1, int(max_bytes))
    try:
        size = path.stat().st_size
        result["truncated"] = bool(
            declared_log_limit is not None
            and declared_log_limit > 0
            and size >= declared_log_limit
        )
        with path.open("rb") as handle:
            while result["bytes_scanned"] < scan_limit:
                remaining = scan_limit - int(result["bytes_scanned"])
                chunk = handle.readline(min(64 * 1024, remaining))
                if not chunk:
                    break
                result["bytes_scanned"] += len(chunk)
                if _LOG_LIMIT_MARKER in chunk:
                    result["truncated"] = True
                for name, pattern in _LATENCY_PATTERNS.items():
                    for match in pattern.finditer(chunk):
                        value = float(match.group(1))
                        if not math.isfinite(value) or value < 0:
                            result["invalid_latency_records"] += 1
                            continue
                        result[f"{name}_latency_records"] += 1
                        maximum_name = f"{name}_latency_max_ms"
                        current = result[maximum_name]
                        result[maximum_name] = (
                            value if current is None else max(current, value)
                        )
                fps_line = _FPS_LINE.search(chunk)
                if fps_line:
                    matches = _FPS_VALUE.findall(fps_line.group(1))
                    if not matches:
                        result["invalid_fps_records"] += 1
                    for raw_value in matches:
                        value = float(raw_value)
                        if not math.isfinite(value) or value < 0:
                            result["invalid_fps_records"] += 1
                            continue
                        result["fps_records"] += 1
                        current_min = result["fps_min"]
                        current_max = result["fps_max"]
                        result["fps_min"] = (
                            value if current_min is None else min(current_min, value)
                        )
                        result["fps_max"] = (
                            value if current_max is None else max(current_max, value)
                        )
            if handle.read(1):
                result["truncated"] = True
        result["readable"] = True
    except OSError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["frame_latency_recognized"] = bool(result["frame_latency_records"])
    result["component_latency_recognized"] = bool(
        result["component_latency_records"]
    )
    result["fps_recognized"] = bool(result["fps_records"])
    result["recognized"] = bool(
        result["frame_latency_recognized"]
        and result["component_latency_recognized"]
        and not result["invalid_latency_records"]
        and result["fps_recognized"]
        and not result["invalid_fps_records"]
    )
    return result


def deepstream_latency_evidence_for_run(
    run_dir: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve the run-scoped diagnostic path and scan it without path escape."""

    run_dir = Path(run_dir).resolve()
    observability = manifest.get("observability")
    probes = (
        observability.get("deepstream_debug_probes")
        if isinstance(observability, Mapping)
        else None
    )
    probe_modules_complete = bool(
        isinstance(probes, Mapping)
        and all(
            isinstance((identity := probes.get(name)), Mapping)
            and identity.get("available") is True
            and type(identity.get("size_bytes")) is int
            and identity["size_bytes"] > 0
            and _valid_sha256(identity.get("sha256"))
            for name in _REQUIRED_DEBUG_PROBES
        )
    )
    declared_log = (
        observability.get("deepstream_log")
        if isinstance(observability, Mapping)
        else None
    )
    relative_log = (
        Path(declared_log)
        if isinstance(declared_log, str) and declared_log
        else Path("diagnostics/deepstream.log")
    )
    log_path = (run_dir / relative_log).resolve()
    if not log_path.is_relative_to(run_dir):
        return {
            "path": str(log_path),
            "present": False,
            "readable": False,
            "bytes_scanned": 0,
            "frame_latency_records": 0,
            "component_latency_records": 0,
            "invalid_latency_records": 0,
            "fps_records": 0,
            "invalid_fps_records": 0,
            "frame_latency_max_ms": None,
            "component_latency_max_ms": None,
            "fps_min": None,
            "fps_max": None,
            "frame_latency_recognized": False,
            "component_latency_recognized": False,
            "fps_recognized": False,
            "recognized": False,
            "probe_modules_complete": probe_modules_complete,
            "truncated": False,
            "error": "declared DeepStream log path escapes the run directory",
        }
    declared_limit = (
        observability.get("deepstream_log_max_bytes")
        if isinstance(observability, Mapping)
        else None
    )
    result = scan_deepstream_latency_log(
        log_path,
        declared_log_limit=(
            int(declared_limit)
            if isinstance(declared_limit, int) and not isinstance(declared_limit, bool)
            else None
        ),
    )
    result["probe_modules_complete"] = probe_modules_complete
    result["recognized"] = bool(
        result.get("recognized") and probe_modules_complete
    )
    return result


def _threshold_checks(
    deltas: Mapping[str, Mapping[str, float | None]],
    thresholds: Mapping[str, Any] | None,
) -> tuple[bool, list[dict[str, Any]], list[str]]:
    if thresholds is None:
        return False, [], ["approved debug-overhead thresholds were not supplied"]
    errors = _threshold_schema_errors(thresholds)
    approved = thresholds.get("approved") is True
    if not approved:
        errors.append("debug-overhead thresholds are not explicitly approved")
    metric_limits = thresholds.get("metrics")
    if not isinstance(metric_limits, Mapping):
        return False, [], errors
    checks: list[dict[str, Any]] = []
    for name in _METRICS:
        policy = metric_limits.get(name)
        if not isinstance(policy, Mapping):
            continue
        bounds = {
            "increase": policy.get("max_increase"),
            "percent_increase": policy.get("max_percent_increase"),
        }
        if all(value is None for value in bounds.values()):
            continue
        for observed_name, raw_limit in bounds.items():
            if raw_limit is None:
                continue
            limit = _number({"value": raw_limit}, "value")
            observed = deltas.get(name, {}).get(observed_name)
            if limit is None or limit < 0:
                continue
            if observed is None:
                errors.append(f"measurement {name}.{observed_name} is unavailable")
                continue
            checks.append(
                {
                    "name": f"{name}.{observed_name}",
                    "operator": "<=",
                    "observed": observed,
                    "limit": limit,
                    "passed": observed <= limit,
                }
            )
    return approved and not errors, checks, errors


def _qualification_limits_identity(summary: Mapping[str, Any]) -> dict[str, Any] | None:
    limits = summary.get("limits")
    if not isinstance(limits, Mapping):
        return None
    schema_version = limits.get("schema_version")
    profile_id = limits.get("profile_id")
    validated = limits.get("validated")
    source_evidence = summary.get("source_evidence")
    limits_file = (
        source_evidence.get("limits")
        if isinstance(source_evidence, Mapping)
        else None
    )
    limits_sha256 = (
        limits_file.get("sha256") if isinstance(limits_file, Mapping) else None
    )
    if (
        type(schema_version) is not str
        or schema_version != "1.0"
        or not isinstance(profile_id, str)
        or not profile_id.strip()
        or validated is not True
        or not _valid_sha256(limits_sha256)
    ):
        return None
    return {
        "schema_version": schema_version,
        "profile_id": profile_id,
        "validated": validated,
        "sha256": limits_sha256,
    }


def compare_debug_overhead(
    baseline_dir: Path,
    debug_dir: Path,
    *,
    thresholds: Mapping[str, Any] | None = None,
    duration_tolerance_seconds: float = 5.0,
    duration_tolerance_percent: float = 1.0,
    expected_case_id: str | None = None,
) -> dict[str, Any]:
    """Return bounded paired-run deltas, failing closed on incomparable evidence."""

    baseline_dir = Path(baseline_dir).resolve()
    debug_dir = Path(debug_dir).resolve()
    baseline, baseline_manifest, baseline_read_identities = _read_evidence(
        baseline_dir
    )
    debug, debug_manifest, debug_read_identities = _read_evidence(debug_dir)
    baseline_status, baseline_status_identity = (
        run_context.read_json_required_with_identity(
        baseline_dir / run_context.RUN_STATUS_FILENAME
        )
    )
    debug_status, debug_status_identity = run_context.read_json_required_with_identity(
        debug_dir / run_context.RUN_STATUS_FILENAME
    )
    baseline_read_identities["run_status"] = baseline_status_identity
    debug_read_identities["run_status"] = debug_status_identity
    mismatches: list[str] = []

    for label, status in (("baseline", baseline_status), ("debug", debug_status)):
        state = status.get("state") if isinstance(status, Mapping) else None
        if state not in _TERMINAL_SUCCESS_STATES:
            mismatches.append(
                f"{label} run is not successfully terminal: state={state!r}"
            )
    for label, manifest in (
        ("baseline", baseline_manifest),
        ("debug", debug_manifest),
    ):
        topology = manifest.get("process_topology")
        owner = (
            topology.get("acquisition_owner")
            if isinstance(topology, Mapping)
            else None
        )
        if owner != "durable_supervisor":
            mismatches.append(
                f"{label} acquisition owner is not durable_supervisor: {owner!r}"
            )

    baseline_binding = baseline_manifest.get("qualification")
    debug_binding = debug_manifest.get("qualification")
    for label, binding, status in (
        ("baseline", baseline_binding, baseline_status),
        ("debug", debug_binding, debug_status),
    ):
        if not isinstance(binding, Mapping):
            mismatches.append(f"{label} qualification case binding is missing")
            continue
        if status.get("qualification") != binding:
            mismatches.append(
                f"{label} manifest/status qualification bindings differ"
            )
        if not isinstance(binding.get("matrix_id"), str) or not binding["matrix_id"].strip():
            mismatches.append(f"{label} qualification matrix_id is missing")
        if not _valid_sha256(binding.get("matrix_sha256")):
            mismatches.append(f"{label} qualification matrix_sha256 is invalid")
        if not isinstance(binding.get("case_id"), str) or not binding["case_id"].strip():
            mismatches.append(f"{label} qualification case_id is missing")
        if not isinstance(binding.get("expected_factors"), Mapping):
            mismatches.append(f"{label} qualification expected_factors are missing")
    if isinstance(baseline_binding, Mapping) and isinstance(debug_binding, Mapping):
        comparable_binding_fields = (
            "matrix_id",
            "matrix_sha256",
            "case_id",
            "expected_factors",
        )
        if any(
            baseline_binding.get(name) != debug_binding.get(name)
            for name in comparable_binding_fields
        ):
            mismatches.append("qualification case binding differs between runs")
        if (
            expected_case_id is not None
            and baseline_binding.get("case_id") != expected_case_id
        ):
            mismatches.append(
                f"qualification case_id does not match requested case: "
                f"{baseline_binding.get('case_id')!r} != {expected_case_id!r}"
            )

    if baseline.get("result") != "passed":
        mismatches.append("baseline qualification did not pass")
    if debug.get("result") != "passed":
        mismatches.append("debug qualification did not pass")
    if baseline.get("qualification_exceptions") != []:
        mismatches.append(
            "baseline qualification_exceptions must be exactly empty"
        )
    if debug.get("qualification_exceptions") != ["deepstream_debug_profile"]:
        mismatches.append(
            "debug qualification_exceptions must be exactly ['deepstream_debug_profile']"
        )
    baseline_limits = baseline.get("limits")
    debug_limits = debug.get("limits")
    mismatches.extend(
        f"baseline {error}"
        for error in source_evidence_errors(
            baseline_dir,
            baseline.get("source_evidence"),
            declared_limits_path=(
                baseline_limits.get("path")
                if isinstance(baseline_limits, Mapping)
                else None
            ),
        )
    )
    mismatches.extend(
        f"debug {error}"
        for error in source_evidence_errors(
            debug_dir,
            debug.get("source_evidence"),
            declared_limits_path=(
                debug_limits.get("path")
                if isinstance(debug_limits, Mapping)
                else None
            ),
        )
    )
    if _debug_enabled(baseline_manifest) is not False:
        mismatches.append("baseline run is not explicitly debug-profile off")
    if _debug_enabled(debug_manifest) is not True:
        mismatches.append("debug run is not explicitly debug-profile on")

    baseline_limits_identity = _qualification_limits_identity(baseline)
    debug_limits_identity = _qualification_limits_identity(debug)
    if baseline_limits_identity is None:
        mismatches.append("baseline validated qualification limits identity is missing")
    if debug_limits_identity is None:
        mismatches.append("debug validated qualification limits identity is missing")
    if (
        baseline_limits_identity is not None
        and debug_limits_identity is not None
        and baseline_limits_identity != debug_limits_identity
    ):
        mismatches.append("qualification limits identity differs")

    baseline_identity = _model_identity(baseline_manifest)
    debug_identity = _model_identity(debug_manifest)
    for name in _MODEL_IDENTITY_FIELDS:
        if not _valid_sha256(baseline_identity.get(name)) or not _valid_sha256(
            debug_identity.get(name)
        ):
            mismatches.append(f"model identity {name} is missing or invalid")
        elif baseline_identity[name] != debug_identity[name]:
            mismatches.append(f"model identity {name} differs")
    if baseline_identity.get("engine_build_identity") != debug_identity.get(
        "engine_build_identity"
    ):
        mismatches.append("model identity engine_build_identity differs")

    baseline_runtime_identity = canonical_runtime_identity(baseline_manifest)
    debug_runtime_identity = canonical_runtime_identity(debug_manifest)
    mismatches.extend(
        f"baseline {error}"
        for error in runtime_identity_errors(baseline_runtime_identity)
    )
    mismatches.extend(
        f"debug {error}"
        for error in runtime_identity_errors(debug_runtime_identity)
    )
    if (
        baseline_runtime_identity is not None
        and debug_runtime_identity is not None
        and baseline_runtime_identity != debug_runtime_identity
    ):
        mismatches.append("runtime identity differs")

    baseline_config_identity = canonical_acquisition_identity(
        baseline_manifest,
        run_context.read_json(baseline_dir / "diagnostics/camera_runtime.json"),
    )
    debug_config_identity = canonical_acquisition_identity(
        debug_manifest,
        run_context.read_json(debug_dir / "diagnostics/camera_runtime.json"),
    )
    mismatches.extend(
        f"baseline {error}"
        for error in acquisition_identity_errors(baseline_config_identity)
    )
    mismatches.extend(
        f"debug {error}"
        for error in acquisition_identity_errors(debug_config_identity)
    )
    if (
        baseline_config_identity is not None
        and debug_config_identity is not None
        and baseline_config_identity != debug_config_identity
    ):
        mismatches.append("acquisition/task configuration identity differs")

    baseline_factors = baseline.get("factors")
    debug_factors = debug.get("factors")
    if not isinstance(baseline_factors, Mapping) or not isinstance(
        debug_factors, Mapping
    ):
        mismatches.append("qualification factors are missing")
    else:
        for name in _MATCHING_FACTORS:
            if baseline_factors.get(name) != debug_factors.get(name):
                mismatches.append(
                    f"factor {name} differs: {baseline_factors.get(name)!r} != "
                    f"{debug_factors.get(name)!r}"
                )

    baseline_duration = _number(baseline, "capture_duration_s")
    debug_duration = _number(debug, "capture_duration_s")
    duration_absolute_tolerance = float(duration_tolerance_seconds)
    duration_percent_tolerance = float(duration_tolerance_percent)
    if not math.isfinite(duration_absolute_tolerance) or duration_absolute_tolerance < 0:
        mismatches.append("duration absolute tolerance must be finite and nonnegative")
        duration_absolute_tolerance = 0.0
    if not math.isfinite(duration_percent_tolerance) or duration_percent_tolerance < 0:
        mismatches.append("duration percent tolerance must be finite and nonnegative")
        duration_percent_tolerance = 0.0
    allowed_duration_delta = None
    duration_delta = None
    if baseline_duration is None or debug_duration is None:
        mismatches.append("capture duration is unavailable")
    else:
        duration_delta = abs(debug_duration - baseline_duration)
        allowed_duration_delta = max(
            duration_absolute_tolerance,
            max(baseline_duration, debug_duration)
            * duration_percent_tolerance
            / 100.0,
        )
        if duration_delta > allowed_duration_delta:
            mismatches.append(
                f"capture duration differs by {duration_delta:.6g}s; "
                f"allowed {allowed_duration_delta:.6g}s"
            )

    latency_evidence = deepstream_latency_evidence_for_run(
        debug_dir, debug_manifest
    )
    if not latency_evidence["readable"]:
        mismatches.append("debug latency log is missing or unreadable")
    elif latency_evidence["truncated"]:
        mismatches.append("debug latency log is truncated")
    elif not latency_evidence["recognized"]:
        mismatches.append(
            "debug latency log lacks complete NVIDIA frame/component latency, "
            "Service Maker FPS, or probe-module provenance"
        )

    deltas: dict[str, dict[str, float | None]] = {}
    for name, path in _METRICS.items():
        baseline_value = _number(baseline, *path)
        debug_value = _number(debug, *path)
        if baseline_value is None or debug_value is None:
            mismatches.append(f"metric {name} is unavailable")
            continue
        delta = debug_value - baseline_value
        percent_delta = delta / baseline_value * 100.0 if baseline_value != 0 else None
        deltas[name] = {
            "baseline": baseline_value,
            "debug": debug_value,
            "delta": delta,
            "percent_delta": percent_delta,
            "increase": max(0.0, delta),
            "percent_increase": (
                max(0.0, percent_delta) if percent_delta is not None else None
            ),
        }

    thresholds_approved, threshold_checks, threshold_errors = _threshold_checks(
        deltas, thresholds
    )
    thresholds_passed = bool(threshold_checks) and all(
        check["passed"] for check in threshold_checks
    )
    for label, root, identities in (
        ("baseline", baseline_dir, baseline_read_identities),
        ("debug", debug_dir, debug_read_identities),
    ):
        for name, filename in (
            ("qualification_summary", "qualification_summary.json"),
            ("run_manifest", run_context.RUN_MANIFEST_FILENAME),
            ("run_status", run_context.RUN_STATUS_FILENAME),
        ):
            if not stable_file_matches_identity(
                root / filename,
                identities[name],
                max_bytes=run_context.MAX_RUN_METADATA_BYTES,
            ):
                mismatches.append(
                    f"{label} {name} changed while debug evidence was evaluated"
                )
    comparable = not mismatches
    if not comparable or (thresholds_approved and not thresholds_passed):
        result = "failed"
    elif not thresholds_approved:
        result = "incomplete"
    else:
        result = "passed"

    return {
        "schema_version": "2.0",
        "result": result,
        "baseline_run": str(baseline_dir),
        "debug_run": str(debug_dir),
        "qualification_case": baseline_binding if comparable else None,
        "comparable": comparable,
        "mismatches": mismatches,
        "model_identity": baseline_identity if comparable else None,
        "runtime_identity": baseline_runtime_identity if comparable else None,
        "acquisition_config_identity": (
            baseline_config_identity if comparable else None
        ),
        "qualification_limits_identity": (
            baseline_limits_identity if comparable else None
        ),
        "duration_comparison": {
            "baseline_seconds": baseline_duration,
            "debug_seconds": debug_duration,
            "absolute_delta_seconds": duration_delta,
            "allowed_delta_seconds": allowed_duration_delta,
            "policy": {
                "absolute_tolerance_seconds": duration_absolute_tolerance,
                "percent_tolerance": duration_percent_tolerance,
                "combination": "maximum",
            },
        },
        "debug_latency_evidence": latency_evidence,
        "metric_deltas": deltas,
        "thresholds": {
            "approved": thresholds_approved,
            "profile_id": thresholds.get("profile_id") if thresholds else None,
            "errors": threshold_errors,
            "checks": threshold_checks,
            "passed": thresholds_passed if thresholds_approved else None,
        },
        "artifact_references": {
            "baseline": {
                "run_directory": str(baseline_dir),
                "qualification_summary": baseline_read_identities[
                    "qualification_summary"
                ],
                "run_manifest": baseline_read_identities["run_manifest"],
                "run_status": baseline_read_identities["run_status"],
                "source_evidence": baseline.get("source_evidence"),
            },
            "debug": {
                "run_directory": str(debug_dir),
                "qualification_summary": debug_read_identities[
                    "qualification_summary"
                ],
                "run_manifest": debug_read_identities["run_manifest"],
                "run_status": debug_read_identities["run_status"],
                "source_evidence": debug.get("source_evidence"),
            },
        },
    }


__all__ = [
    "MAX_THRESHOLD_BYTES",
    "compare_debug_overhead",
    "deepstream_latency_evidence_for_run",
    "load_debug_thresholds",
    "scan_deepstream_latency_log",
]
