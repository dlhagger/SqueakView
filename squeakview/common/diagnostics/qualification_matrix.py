"""Versioned, bounded qualification-matrix evaluation."""

from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from squeakview.common import run_context
from squeakview.common.bounded_input import read_stable_regular_file
from squeakview.common.diagnostics.qualification import (
    MAX_QUALIFICATION_CAMERA_COUNT,
    qualify_run,
)
from squeakview.common.diagnostics.evidence_identity import (
    acquisition_identity_errors,
    canonical_acquisition_identity,
    canonical_runtime_identity,
    runtime_identity_errors,
    stable_file_identity,
)


MATRIX_SCHEMA_VERSION = "1.0"
MAX_MATRIX_BYTES = 1 << 20
MAX_ASSIGNMENT_BYTES = 1 << 20
MAX_MATRIX_CASES = 4096
QUALIFICATION_CASE_ENV = "SQUEAKVIEW_QUALIFICATION_CASE_ID"
QUALIFICATION_MATRIX_ENV = "SQUEAKVIEW_QUALIFICATION_MATRIX"
_TOP_LEVEL_KEYS = {
    "schema_version",
    "matrix_id",
    "description",
    "capture_profiles",
    "durations",
    "inference_enabled",
    "preview_enabled",
    "power_modes",
    "provenance",
    "notes",
}
_CAPTURE_KEYS = {
    "id", "width", "height", "fps", "camera_count", "pixel_format",
    "trigger_on", "trigger_activation", "arduino_fps", "exposure_us",
    "bitrate_kbps", "serial_enabled", "serial_port", "serial_baud",
}
_DURATION_KEYS = {"id", "minimum_seconds"}
_SUPPORTED_POWER_MODES = {"25W", "MAXN_SUPER"}
_MODEL_SHA256_FIELDS = (
    "model_manifest_sha256",
    "pose_sidecar_sha256",
    "onnx_sha256",
    "config_sha256",
    "engine_sha256",
)
_PROVENANCE_BOOLEAN_KEYS = (
    "require_clean_git",
    "require_same_git_commit",
    "require_same_model_identity",
    "require_production_eligible",
    "require_same_device_identity",
    "require_case_binding",
    "require_same_acquisition_identity",
)
_PROVENANCE_KEYS = {
    *_PROVENANCE_BOOLEAN_KEYS,
    "expected_git_commit",
    "expected_model_identity",
}
_MODEL_IDENTITY_KEYS = {
    "name",
    "model_manifest_schema",
    *_MODEL_SHA256_FIELDS,
    "engine_build_identity",
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


def _reject_unknown_keys(value: Mapping[Any, Any], allowed: set[str], context: str) -> None:
    unknown = [key for key in value if not isinstance(key, str) or key not in allowed]
    if unknown:
        raise ValueError(f"qualification matrix {context} has unknown keys: {unknown!r}")


def _nonempty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"qualification matrix {context} must be a non-empty string")
    return value


def _bounded_integer(value: object, context: str, *, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(
            f"qualification matrix {context} must be an integer from 1 to {maximum}"
        )
    return value


def _validate_unique(values: list[Any], context: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"qualification matrix {context} entries must be unique")


def _matrix_case_count(matrix: Mapping[str, Any]) -> int:
    return (
        len(matrix["capture_profiles"])
        * len(matrix["durations"])
        * len(matrix["inference_enabled"])
        * len(matrix["preview_enabled"])
        * len(matrix["power_modes"])
    )


def load_matrix(path: Path) -> dict[str, Any]:
    try:
        raw = read_stable_regular_file(
            path, max_bytes=MAX_MATRIX_BYTES, label="qualification matrix"
        )
        text = raw.decode("utf-8", errors="strict")
        payload = yaml.load(text, Loader=_UniqueKeySafeLoader)
    except UnicodeDecodeError as exc:
        raise ValueError("qualification matrix must be strict UTF-8") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"qualification matrix could not be read: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("qualification matrix must contain a mapping")
    _reject_unknown_keys(payload, _TOP_LEVEL_KEYS, "root")
    if type(payload.get("schema_version")) is not str or payload["schema_version"] != MATRIX_SCHEMA_VERSION:
        raise ValueError(
            f"qualification matrix schema_version must be the string {MATRIX_SCHEMA_VERSION!r}"
        )
    _nonempty_string(payload.get("matrix_id"), "matrix_id")
    if "description" in payload:
        _nonempty_string(payload["description"], "description")
    if "notes" in payload and (
        not isinstance(payload["notes"], list)
        or any(not isinstance(note, str) or not note.strip() for note in payload["notes"])
    ):
        raise ValueError("qualification matrix notes must be a list of non-empty strings")
    for name in (
        "capture_profiles",
        "durations",
        "inference_enabled",
        "preview_enabled",
        "power_modes",
    ):
        if not isinstance(payload.get(name), list) or not payload[name]:
            raise ValueError(f"qualification matrix {name} must be a non-empty list")

    capture_ids: list[str] = []
    for index, capture in enumerate(payload["capture_profiles"]):
        context = f"capture_profiles[{index}]"
        if not isinstance(capture, dict):
            raise ValueError(f"qualification matrix {context} must be a mapping")
        _reject_unknown_keys(capture, _CAPTURE_KEYS, context)
        missing = _CAPTURE_KEYS - capture.keys()
        if missing:
            raise ValueError(f"qualification matrix {context} is missing keys: {sorted(missing)!r}")
        capture_ids.append(_nonempty_string(capture["id"], f"{context}.id"))
        _bounded_integer(capture["width"], f"{context}.width", maximum=65535)
        _bounded_integer(capture["height"], f"{context}.height", maximum=65535)
        _bounded_integer(capture["fps"], f"{context}.fps", maximum=1000)
        _bounded_integer(
            capture["camera_count"],
            f"{context}.camera_count",
            maximum=MAX_QUALIFICATION_CAMERA_COUNT,
        )
        _nonempty_string(capture["pixel_format"], f"{context}.pixel_format")
        if type(capture["trigger_on"]) is not bool:
            raise ValueError(f"qualification matrix {context}.trigger_on must be boolean")
        _nonempty_string(capture["trigger_activation"], f"{context}.trigger_activation")
        _bounded_integer(capture["arduino_fps"], f"{context}.arduino_fps", maximum=1000)
        exposure = capture["exposure_us"]
        if (
            type(exposure) not in (int, float)
            or not math.isfinite(float(exposure))
            or float(exposure) <= 0
        ):
            raise ValueError(
                f"qualification matrix {context}.exposure_us must be finite and positive"
            )
        _bounded_integer(capture["bitrate_kbps"], f"{context}.bitrate_kbps", maximum=1_000_000)
        if type(capture["serial_enabled"]) is not bool:
            raise ValueError(f"qualification matrix {context}.serial_enabled must be boolean")
        _nonempty_string(capture["serial_port"], f"{context}.serial_port")
        _bounded_integer(capture["serial_baud"], f"{context}.serial_baud", maximum=100_000_000)
    _validate_unique(capture_ids, "capture profile ids")

    duration_ids: list[str] = []
    for index, duration in enumerate(payload["durations"]):
        context = f"durations[{index}]"
        if not isinstance(duration, dict):
            raise ValueError(f"qualification matrix {context} must be a mapping")
        _reject_unknown_keys(duration, _DURATION_KEYS, context)
        missing = _DURATION_KEYS - duration.keys()
        if missing:
            raise ValueError(f"qualification matrix {context} is missing keys: {sorted(missing)!r}")
        duration_ids.append(_nonempty_string(duration["id"], f"{context}.id"))
        _bounded_integer(
            duration["minimum_seconds"],
            f"{context}.minimum_seconds",
            maximum=315_576_000,
        )
    _validate_unique(duration_ids, "duration ids")

    for name in ("inference_enabled", "preview_enabled"):
        values = payload[name]
        if any(type(value) is not bool for value in values):
            raise ValueError(f"qualification matrix {name} entries must be booleans")
        _validate_unique(values, name)
    power_modes = payload["power_modes"]
    if any(type(value) is not str or value not in _SUPPORTED_POWER_MODES for value in power_modes):
        raise ValueError(
            "qualification matrix power_modes entries must be one of: 25W, MAXN_SUPER"
        )
    _validate_unique(power_modes, "power_modes")

    case_count = _matrix_case_count(payload)
    if case_count > MAX_MATRIX_CASES:
        raise ValueError(
            f"qualification matrix expands to {case_count} cases; limit is {MAX_MATRIX_CASES}"
        )
    provenance = payload.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("qualification matrix provenance must be a mapping")
    _reject_unknown_keys(provenance, _PROVENANCE_KEYS, "provenance")
    provenance.setdefault("require_same_device_identity", True)
    provenance.setdefault("require_case_binding", True)
    provenance.setdefault("require_same_acquisition_identity", True)
    payload["provenance"] = provenance
    for name in _PROVENANCE_BOOLEAN_KEYS:
        if name in provenance and not isinstance(provenance[name], bool):
            raise ValueError(f"qualification matrix provenance.{name} must be boolean")
    expected_commit = provenance.get("expected_git_commit")
    if expected_commit is not None and (
        not isinstance(expected_commit, str) or not expected_commit.strip()
    ):
        raise ValueError(
            "qualification matrix provenance.expected_git_commit must be a non-empty string"
        )
    expected_model = provenance.get("expected_model_identity")
    if expected_model is not None:
        if not isinstance(expected_model, dict) or not expected_model:
            raise ValueError(
                "qualification matrix provenance.expected_model_identity must be a non-empty mapping"
            )
        _reject_unknown_keys(
            expected_model,
            _MODEL_IDENTITY_KEYS,
            "provenance.expected_model_identity",
        )
        for name, value in expected_model.items():
            if name == "model_manifest_schema":
                if type(value) is not int or value < 1:
                    raise ValueError(
                        "qualification matrix provenance.expected_model_identity."
                        "model_manifest_schema must be a positive integer"
                    )
            elif name == "engine_build_identity":
                if not isinstance(value, dict):
                    raise ValueError(
                        "qualification matrix provenance.expected_model_identity."
                        "engine_build_identity must be a mapping"
                    )
            else:
                _nonempty_string(
                    value, f"provenance.expected_model_identity.{name}"
                )

    case_ids = [case["case_id"] for case in expand_cases(payload)]
    _validate_unique(case_ids, "expanded case ids")
    return payload


def load_assignments(path: Path) -> dict[str, str | None]:
    """Load a bounded case-to-run checklist; null means not yet assigned."""

    try:
        raw = read_stable_regular_file(
            path,
            max_bytes=MAX_ASSIGNMENT_BYTES,
            label="qualification assignments",
        )
        payload = yaml.load(
            raw.decode("utf-8", errors="strict"), Loader=_UniqueKeySafeLoader
        )
    except UnicodeDecodeError as exc:
        raise ValueError("qualification assignments must be strict UTF-8") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"qualification assignments could not be read: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("qualification assignments must contain a mapping")
    for case_id, run_dir in payload.items():
        _nonempty_string(case_id, "assignment case ID")
        if run_dir is not None:
            _nonempty_string(run_dir, f"assignment {case_id!r} run directory")
    return dict(payload)


def expand_cases(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    case_count = _matrix_case_count(matrix)
    if case_count > MAX_MATRIX_CASES:
        raise ValueError(
            f"qualification matrix expands to {case_count} cases; limit is {MAX_MATRIX_CASES}"
        )
    cases: list[dict[str, Any]] = []
    dimensions = product(
        matrix["capture_profiles"],
        matrix["durations"],
        matrix["inference_enabled"],
        matrix["preview_enabled"],
        matrix["power_modes"],
    )
    for capture, duration, inference, preview, power in dimensions:
        case_id = "--".join(
            (
                str(capture["id"]),
                str(duration["id"]),
                "infer-on" if inference else "infer-off",
                "preview-on" if preview else "preview-off",
                str(power).lower().replace("_", "-"),
            )
        )
        cases.append(
            {
                "case_id": case_id,
                "capture": dict(capture),
                "duration": dict(duration),
                "inference_enabled": bool(inference),
                "preview_enabled": bool(preview),
                "power_mode": str(power),
            }
        )
    return cases


def _factor_mismatches(case: Mapping[str, Any], summary: Mapping[str, Any]) -> list[str]:
    factors = summary.get("factors")
    if not isinstance(factors, Mapping):
        return ["qualification summary factors are missing"]
    expected = {
        "width": case["capture"]["width"],
        "height": case["capture"]["height"],
        "fps": case["capture"]["fps"],
        "camera_count": case["capture"]["camera_count"],
        "pixel_format": case["capture"]["pixel_format"],
        "trigger_on": case["capture"]["trigger_on"],
        "trigger_activation": case["capture"]["trigger_activation"],
        "arduino_fps": case["capture"]["arduino_fps"],
        "exposure_us": case["capture"]["exposure_us"],
        "bitrate_kbps": case["capture"]["bitrate_kbps"],
        "serial_enabled": case["capture"]["serial_enabled"],
        "serial_port": case["capture"]["serial_port"],
        "serial_baud": case["capture"]["serial_baud"],
        "inference_enabled": case["inference_enabled"],
        "preview_enabled": case["preview_enabled"],
    }
    mismatches = [
        f"{name}: expected {value!r}, observed {factors.get(name)!r}"
        for name, value in expected.items()
        if factors.get(name) != value
    ]
    observed_power_mode = normalize_power_mode(factors.get("nvpmodel"))
    if observed_power_mode != case["power_mode"]:
        mismatches.append(
            f"nvpmodel: expected exact named mode {case['power_mode']!r}, "
            f"observed {factors.get('nvpmodel')!r}"
        )
    duration = summary.get("capture_duration_s")
    minimum = float(case["duration"]["minimum_seconds"])
    if (
        type(duration) not in (int, float)
        or not math.isfinite(float(duration))
        or float(duration) < minimum
    ):
        mismatches.append(
            f"duration: expected at least {minimum:.0f}s, observed {duration!r}"
        )
    return mismatches


def normalize_power_mode(value: object) -> str | None:
    """Extract only a complete, supported nvpmodel mode name."""

    if not isinstance(value, str):
        return None
    names: list[str] = []
    for line in value.splitlines():
        candidate = line.strip()
        for prefix in ("NV Power Mode:", "NVP Power Mode:"):
            if candidate.startswith(prefix):
                names.append(candidate[len(prefix) :].strip())
                break
    return names[0] if len(names) == 1 and names[0] in _SUPPORTED_POWER_MODES else None


def expected_case_factors(case: Mapping[str, Any]) -> dict[str, Any]:
    """Return the immutable factors a bound startup must satisfy."""

    capture = case["capture"]
    return {
        "capture_profile_id": capture["id"],
        "width": capture["width"],
        "height": capture["height"],
        "fps": capture["fps"],
        "camera_count": capture["camera_count"],
        "pixel_format": capture["pixel_format"],
        "trigger_on": capture["trigger_on"],
        "trigger_activation": capture["trigger_activation"],
        "arduino_fps": capture["arduino_fps"],
        "exposure_us": capture["exposure_us"],
        "bitrate_kbps": capture["bitrate_kbps"],
        "serial_enabled": capture["serial_enabled"],
        "serial_port": capture["serial_port"],
        "serial_baud": capture["serial_baud"],
        "minimum_duration_seconds": case["duration"]["minimum_seconds"],
        "duration_id": case["duration"]["id"],
        "inference_enabled": case["inference_enabled"],
        "preview_enabled": case["preview_enabled"],
        "power_mode": case["power_mode"],
    }


def _same_typed_value(observed: object, expected: object) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return observed.keys() == expected.keys() and all(
            _same_typed_value(observed[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _same_typed_value(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def resolve_qualification_case_binding(
    config: object,
    device_context: Mapping[str, object],
    *,
    default_matrix_path: Path,
    environ: Mapping[str, str],
) -> dict[str, Any] | None:
    """Validate an opt-in matrix case against effective startup state."""

    raw_case_id = environ.get(QUALIFICATION_CASE_ENV)
    raw_matrix_value = environ.get(QUALIFICATION_MATRIX_ENV)
    if raw_case_id is None:
        if raw_matrix_value is not None:
            raise ValueError(
                f"{QUALIFICATION_MATRIX_ENV} requires {QUALIFICATION_CASE_ENV}; "
                "refusing an accidentally unbound qualification run"
            )
        return None
    if not isinstance(raw_case_id, str) or not raw_case_id.strip():
        raise ValueError(
            f"{QUALIFICATION_CASE_ENV} must be a non-empty canonical case ID"
        )
    case_id = raw_case_id.strip()
    if raw_matrix_value is not None and (
        not isinstance(raw_matrix_value, str) or not raw_matrix_value.strip()
    ):
        raise ValueError(
            f"{QUALIFICATION_MATRIX_ENV} must be a non-empty matrix path"
        )
    raw_matrix_path = raw_matrix_value.strip() if raw_matrix_value is not None else ""
    matrix_path = Path(raw_matrix_path) if raw_matrix_path else Path(default_matrix_path)
    before = stable_file_identity(matrix_path, max_bytes=MAX_MATRIX_BYTES)
    if before.get("available") is not True:
        raise ValueError(f"qualification matrix identity is unavailable: {before.get('error')}")
    matrix = load_matrix(matrix_path)
    after = stable_file_identity(matrix_path, max_bytes=MAX_MATRIX_BYTES)
    if before != after:
        raise ValueError("qualification matrix changed while binding the run")
    matches = [case for case in expand_cases(matrix) if case["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"unknown qualification case ID: {case_id!r}")
    case = matches[0]
    expected = expected_case_factors(case)
    observed = {
        "width": getattr(config, "width", None),
        "height": getattr(config, "height", None),
        "fps": getattr(config, "fps", None),
        "camera_count": getattr(config, "num_cameras", None),
        "pixel_format": getattr(config, "pixel_format", None),
        "trigger_on": getattr(config, "trigger_on", None),
        "trigger_activation": getattr(config, "trigger_activation", None),
        "arduino_fps": getattr(config, "arduino_fps", None),
        "exposure_us": getattr(config, "exposure_us", None),
        "bitrate_kbps": getattr(config, "bitrate", None),
        "serial_enabled": getattr(config, "serial_enabled", None),
        "serial_port": getattr(config, "serial_port", None),
        "serial_baud": getattr(config, "serial_baud", None),
        "inference_enabled": getattr(config, "inference_enabled", None),
        "preview_enabled": getattr(config, "preview_enabled", None),
        "power_mode": normalize_power_mode(device_context.get("nvpmodel")),
    }
    mismatches = [
        f"{name}: expected {expected[name]!r}, observed {observed[name]!r}"
        for name in observed
        if not _same_typed_value(observed[name], expected[name])
    ]
    if mismatches:
        raise ValueError(
            "qualification case does not match effective startup: "
            + "; ".join(mismatches)
        )
    return {
        "matrix_id": matrix["matrix_id"],
        "matrix_path": str(matrix_path.resolve()),
        "matrix_sha256": before["sha256"],
        "case_id": case_id,
        "expected_factors": expected,
    }


def _model_identity(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    inference = manifest.get("inference")
    if not isinstance(inference, Mapping):
        return None
    model = inference.get("model_package")
    if not isinstance(model, Mapping):
        return None
    # Paths identify where a package happened to be installed, not its content.
    # These fields identify the selected configuration, TensorRT plan, and the
    # runtime/device identity against which that plan was validated.
    return {
        "name": model.get("name"),
        "model_manifest_schema": model.get("model_manifest_schema"),
        "model_manifest_sha256": model.get("model_manifest_sha256"),
        "pose_sidecar_sha256": model.get("pose_sidecar_sha256"),
        "onnx_sha256": model.get("onnx_sha256"),
        "config_sha256": model.get("config_sha256"),
        "engine_sha256": model.get("engine_sha256"),
        "engine_build_identity": model.get("engine_build_identity"),
    }


def _canonical_identity(identity: Mapping[str, Any] | None) -> str | None:
    if identity is None:
        return None
    return json.dumps(dict(identity), sort_keys=True, separators=(",", ":"))


def _valid_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _provenance_evidence(
    run_dir: Path,
    case: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    matrix_id: str,
    matrix_sha256: str,
) -> tuple[dict[str, Any], list[str]]:
    # Missing/invalid metadata becomes an empty bounded snapshot here so the
    # matrix can report factor/provenance mismatches for every cell instead of
    # aborting the complete report.
    manifest = run_context.read_json(
        Path(run_dir) / run_context.RUN_MANIFEST_FILENAME
    )
    status = run_context.read_json(
        Path(run_dir) / run_context.RUN_STATUS_FILENAME
    )
    git = manifest.get("git") if isinstance(manifest.get("git"), Mapping) else {}
    observability = (
        manifest.get("observability")
        if isinstance(manifest.get("observability"), Mapping)
        else {}
    )
    identity = _model_identity(manifest)
    device_identity = canonical_runtime_identity(manifest)
    camera_runtime = run_context.read_json(
        Path(run_dir) / "diagnostics/camera_runtime.json"
    )
    acquisition_identity = canonical_acquisition_identity(manifest, camera_runtime)
    qualification_binding = manifest.get("qualification")
    status_qualification_binding = status.get("qualification")
    evidence = {
        "production_eligible": manifest.get("production_eligible"),
        "preflight_skipped": observability.get("preflight_skipped"),
        "deepstream_debug_profile": observability.get(
            "deepstream_debug_profile"
        ),
        "git_commit": git.get("commit"),
        "git_dirty": git.get("dirty"),
        "model_identity": identity,
        "device_identity": device_identity,
        "acquisition_identity": acquisition_identity,
        "qualification": qualification_binding,
        "status_qualification": status_qualification_binding,
    }
    mismatches: list[str] = []
    if policy.get("require_same_device_identity") is True:
        mismatches.extend(runtime_identity_errors(device_identity))
    if policy.get("require_same_acquisition_identity") is True:
        mismatches.extend(acquisition_identity_errors(acquisition_identity))
    if policy.get("require_case_binding") is True:
        expected_binding = {
            "matrix_id": matrix_id,
            "matrix_sha256": matrix_sha256,
            "case_id": case["case_id"],
            "expected_factors": expected_case_factors(case),
        }
        for source, binding in (
            ("manifest qualification", qualification_binding),
            ("status qualification", status_qualification_binding),
        ):
            if not isinstance(binding, Mapping):
                mismatches.append(f"{source} case binding is missing")
                continue
            for name, expected in expected_binding.items():
                if not _same_typed_value(binding.get(name), expected):
                    mismatches.append(
                        f"{source}.{name}: expected {expected!r}, observed "
                        f"{binding.get(name)!r}"
                    )
    if policy.get("require_production_eligible") is True:
        if evidence["production_eligible"] is not True:
            mismatches.append(
                "production_eligible: expected True, observed "
                f"{evidence['production_eligible']!r}"
            )
        if evidence["preflight_skipped"] is not False:
            mismatches.append(
                "preflight_skipped: expected False, observed "
                f"{evidence['preflight_skipped']!r}"
            )
        if evidence["deepstream_debug_profile"] is not False:
            mismatches.append(
                "deepstream_debug_profile: expected False, observed "
                f"{evidence['deepstream_debug_profile']!r}"
            )
    if policy.get("require_clean_git") is True:
        if evidence["git_dirty"] is not False:
            mismatches.append(
                f"git_dirty: expected False, observed {evidence['git_dirty']!r}"
            )

    commit = evidence["git_commit"]
    if policy.get("require_same_git_commit") is True and not commit:
        mismatches.append("git_commit: required value is missing")
    expected_commit = policy.get("expected_git_commit")
    if expected_commit is not None and commit != expected_commit:
        mismatches.append(
            f"git_commit: expected {expected_commit!r}, observed {commit!r}"
        )

    if bool(case.get("inference_enabled")):
        if policy.get("require_same_model_identity") is True:
            if identity is None:
                mismatches.append("model_identity: required value is missing")
            else:
                required_identity = (
                    "name",
                    "engine_build_identity",
                    *_MODEL_SHA256_FIELDS,
                )
                missing = [
                    name for name in required_identity if identity.get(name) in (None, "")
                ]
                if missing:
                    mismatches.append(
                        "model_identity: required values are missing: "
                        + ", ".join(missing)
                    )
                invalid_hashes = [
                    name
                    for name in _MODEL_SHA256_FIELDS
                    if not _valid_sha256(identity.get(name))
                ]
                if invalid_hashes:
                    mismatches.append(
                        "model_identity: invalid SHA-256 values: "
                        + ", ".join(invalid_hashes)
                    )
                if identity.get("model_manifest_schema") != 3:
                    mismatches.append(
                        "model_identity.model_manifest_schema: expected 3, observed "
                        f"{identity.get('model_manifest_schema')!r}"
                    )
        expected_model = policy.get("expected_model_identity")
        if isinstance(expected_model, Mapping):
            if identity is None:
                mismatches.append("model_identity: expected identity but none was recorded")
            else:
                for name, expected in expected_model.items():
                    if identity.get(name) != expected:
                        mismatches.append(
                            f"model_identity.{name}: expected {expected!r}, "
                            f"observed {identity.get(name)!r}"
                        )
    return evidence, mismatches


def _enforce_shared_provenance(
    reports: list[dict[str, Any]], policy: Mapping[str, Any]
) -> None:
    def fail_all(items: list[dict[str, Any]], message: str) -> None:
        for item in items:
            mismatches = item.setdefault("provenance_mismatches", [])
            if message not in mismatches:
                mismatches.append(message)
            item["result"] = "failed"

    assigned = [item for item in reports if item.get("run_directory")]
    if policy.get("require_same_git_commit") is True:
        commits = {
            item.get("provenance", {}).get("git_commit") for item in assigned
        }
        if len(commits) > 1:
            observed = sorted(repr(value) for value in commits)
            fail_all(
                assigned,
                "git_commit differs across assigned cells: " + ", ".join(observed),
            )

    if policy.get("require_same_model_identity") is True:
        inferred = [item for item in assigned if item.get("inference_enabled") is True]
        identities = {
            _canonical_identity(item.get("provenance", {}).get("model_identity"))
            for item in inferred
        }
        if len(identities) > 1:
            fail_all(inferred, "model_identity differs across inference-enabled cells")

    if policy.get("require_same_device_identity") is True:
        identities = {
            _canonical_identity(item.get("provenance", {}).get("device_identity"))
            for item in assigned
        }
        if len(identities) > 1:
            fail_all(assigned, "device_identity differs across assigned cells")

    if policy.get("require_same_acquisition_identity") is True:
        identities = {
            _canonical_identity(
                item.get("provenance", {}).get("acquisition_identity")
            )
            for item in assigned
        }
        if len(identities) > 1:
            fail_all(
                assigned,
                "acquisition_identity differs across assigned cells",
            )


def qualify_matrix(
    matrix_path: Path,
    assignments: Mapping[str, Path | str | None],
    *,
    limits_path: Path,
    output_path: Path,
    qualifier: Callable[..., dict[str, Any]] = qualify_run,
) -> dict[str, Any]:
    matrix_identity = stable_file_identity(matrix_path, max_bytes=MAX_MATRIX_BYTES)
    if matrix_identity.get("available") is not True:
        raise ValueError(
            f"qualification matrix identity is unavailable: {matrix_identity.get('error')}"
        )
    matrix = load_matrix(matrix_path)
    current_identity = stable_file_identity(matrix_path, max_bytes=MAX_MATRIX_BYTES)
    if current_identity != matrix_identity:
        raise ValueError("qualification matrix changed while evaluating assignments")
    provenance_policy = matrix.get("provenance", {})
    reports: list[dict[str, Any]] = []
    assigned_dirs: set[str] = set()
    cases = expand_cases(matrix)
    case_ids = {case["case_id"] for case in cases}
    unknown_assignments = [
        case_id
        for case_id in assignments
        if not isinstance(case_id, str) or case_id not in case_ids
    ]
    if unknown_assignments:
        raise ValueError(
            "qualification assignments contain unknown case IDs: "
            f"{unknown_assignments!r}"
        )
    invalid_paths = [
        case_id
        for case_id, run_dir in assignments.items()
        if run_dir is not None
        and (
            not isinstance(run_dir, (str, Path))
            or not str(run_dir).strip()
        )
    ]
    if invalid_paths:
        raise ValueError(
            "qualification assignments contain invalid run directories for: "
            f"{invalid_paths!r}"
        )
    for case in cases:
        case_id = case["case_id"]
        raw_run_dir = assignments.get(case_id)
        if raw_run_dir is None:
            reports.append({**case, "result": "incomplete", "reason": "run not assigned"})
            continue
        run_dir = str(Path(raw_run_dir).resolve())
        if run_dir in assigned_dirs:
            reports.append(
                {**case, "run_directory": run_dir, "result": "failed", "reason": "run reused by multiple cells"}
            )
            continue
        assigned_dirs.add(run_dir)
        summary = qualifier(Path(run_dir), limits_path=limits_path)
        summary_identity = stable_file_identity(
            Path(run_dir) / "qualification_summary.json",
            max_bytes=run_context.MAX_RUN_METADATA_BYTES,
        )
        mismatches = _factor_mismatches(case, summary)
        provenance, provenance_mismatches = _provenance_evidence(
            Path(run_dir),
            case,
            provenance_policy,
            matrix_id=str(matrix["matrix_id"]),
            matrix_sha256=str(matrix_identity["sha256"]),
        )
        result = (
            "failed"
            if mismatches or provenance_mismatches
            else str(summary.get("result") or "incomplete")
        )
        reports.append(
            {
                **case,
                "run_directory": run_dir,
                "qualification_result": summary.get("result"),
                "qualification_summary_identity": summary_identity,
                "result": result,
                "factor_mismatches": mismatches,
                "provenance": provenance,
                "provenance_mismatches": provenance_mismatches,
            }
        )
    _enforce_shared_provenance(reports, provenance_policy)
    overall = (
        "failed"
        if any(item["result"] == "failed" for item in reports)
        else "incomplete"
        if any(item["result"] != "passed" for item in reports)
        else "passed"
    )
    report = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "matrix_id": matrix.get("matrix_id"),
        "matrix_sha256": matrix_identity["sha256"],
        "provenance_policy": provenance_policy,
        "result": overall,
        "case_count": len(reports),
        "passed_count": sum(item["result"] == "passed" for item in reports),
        "failed_count": sum(item["result"] == "failed" for item in reports),
        "incomplete_count": sum(item["result"] == "incomplete" for item in reports),
        "cases": reports,
    }
    run_context.atomic_write_json(Path(output_path), report)
    return report


__all__ = [
    "MATRIX_SCHEMA_VERSION",
    "MAX_ASSIGNMENT_BYTES",
    "QUALIFICATION_CASE_ENV",
    "QUALIFICATION_MATRIX_ENV",
    "expand_cases",
    "load_assignments",
    "load_matrix",
    "expected_case_factors",
    "normalize_power_mode",
    "qualify_matrix",
    "resolve_qualification_case_binding",
]
