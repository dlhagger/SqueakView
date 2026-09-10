"""Bounded inventory and verification for qualification campaign archives."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any, Mapping

from squeakview.common import run_context
from squeakview.common.bounded_input import read_json_object
from squeakview.common.diagnostics.evidence_identity import stable_file_identity
from squeakview.common.diagnostics.evidence_identity import source_evidence_errors
from squeakview.common.recording_evidence import recording_evidence_complete
from squeakview.common.diagnostics.qualification import (
    MAX_LIMITS_BYTES,
    TERMINAL_SUCCESS_STATES,
    _read_yaml,
)
from squeakview.common.diagnostics.qualification_matrix import (
    MAX_ASSIGNMENT_BYTES,
    MAX_MATRIX_BYTES,
    expand_cases,
    load_assignments,
    load_matrix,
)


SCHEMA_VERSION = "1.0"
MAX_REPORT_BYTES = 16 << 20
MAX_INVENTORY_BYTES = 64 << 20
MAX_CAMPAIGN_FILES = 16_384
MAX_RELATIVE_PATH_BYTES = 4096
MAX_DIRECTORY_DEPTH = 16
_CONTROL_NAMES = ("matrix", "assignments", "report", "limits")
_REPORT_KEYS = {
    "schema_version", "matrix_id", "matrix_sha256", "provenance_policy",
    "result", "case_count", "passed_count", "failed_count",
    "incomplete_count", "cases",
}
_INVENTORY_KEYS = {"schema_version", "matrix_id", "controls", "run_count", "artifact_count", "runs"}
_CONTROL_RECORD_KEYS = {"source_path", "archive_path", "size_bytes", "sha256"}
_RUN_RECORD_KEYS = {"case_id", "source_run_directory", "archive_path", "terminal_state", "artifacts"}
_ARTIFACT_RECORD_KEYS = {"path", "size_bytes", "sha256"}


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields are invalid")


def _require_content_record(value: Mapping[str, Any], label: str) -> None:
    size = value.get("size_bytes")
    digest = value.get("sha256")
    if type(size) is not int or size < 0:
        raise ValueError(f"{label} size is invalid")
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} SHA-256 is invalid")


def _content_identity(path: Path, *, max_bytes: int | None = None) -> dict[str, object]:
    if path.is_symlink():
        raise ValueError(f"archive evidence must not be a symlink: {path}")
    identity = stable_file_identity(path, max_bytes=max_bytes)
    if identity.get("available") is not True:
        raise ValueError(f"could not identify archive evidence {path}: {identity.get('error')}")
    return {
        "size_bytes": identity["size_bytes"],
        "sha256": identity["sha256"],
    }


def _walk_regular_files(root: Path) -> list[Path]:
    result: list[Path] = []
    pending: list[tuple[Path, int]] = [(root, 0)]
    entry_count = 0
    while pending:
        directory, depth = pending.pop()
        if depth > MAX_DIRECTORY_DEPTH:
            raise ValueError(f"run artifact tree exceeds depth limit: {directory}")
        try:
            entries = os.scandir(directory)
        except OSError as exc:
            raise ValueError(f"run artifact directory cannot be read: {exc}") from exc
        with entries:
            for entry in entries:
                entry_count += 1
                if entry_count > MAX_CAMPAIGN_FILES:
                    raise ValueError("campaign artifact tree entry count exceeds limit")
                path = Path(entry.path)
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode):
                    raise ValueError(f"run artifact tree contains symlink: {path}")
                if stat.S_ISDIR(metadata.st_mode):
                    pending.append((path, depth + 1))
                elif stat.S_ISREG(metadata.st_mode):
                    result.append(path)
                else:
                    raise ValueError(f"run artifact is not a regular file: {path}")
    result.sort(key=lambda path: path.relative_to(root).as_posix())
    return result


def _safe_relative(value: object) -> Path:
    if not isinstance(value, str) or not value or len(value.encode()) > MAX_RELATIVE_PATH_BYTES:
        raise ValueError("archive inventory contains an invalid relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise ValueError(f"archive inventory path escapes archive root: {value!r}")
    return relative


def _stat_snapshot(path: Path) -> tuple[int, int, int, int, int]:
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"campaign evidence is no longer a regular file: {path}")
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _same_content_identity(left: object, right: object) -> bool:
    return (
        isinstance(left, Mapping)
        and isinstance(right, Mapping)
        and left.get("available", True) is True
        and right.get("available", True) is True
        and all(
            left.get(field) == right.get(field)
            for field in ("size_bytes", "sha256")
        )
    )


def _recording_evidence_matches_inventory(
    run_dir: Path, recorded: object, artifacts: list[dict[str, object]]
) -> bool:
    if not isinstance(recorded, Mapping):
        return False
    camera_count = recorded.get("camera_count")
    if type(camera_count) is not int or not recording_evidence_complete(
        recorded, camera_count
    ):
        return False
    actual = {str(item["path"]): item for item in artifacts}
    for camera in recorded["cameras"]:
        for kind in ("video", "capture_ledger", "admission_ledger"):
            expected = camera[kind]
            try:
                path = Path(str(expected["path"])).resolve()
                relative = path.relative_to(run_dir).as_posix()
            except (KeyError, OSError, ValueError):
                return False
            observed = actual.get(relative)
            if not _same_content_identity(expected, observed):
                return False
    return True


def build_campaign_inventory(
    *, matrix_path: Path, assignments_path: Path, report_path: Path,
    limits_path: Path, output_path: Path,
) -> dict[str, Any]:
    """Validate a campaign and atomically publish a non-mutating inventory."""

    controls = {
        "matrix": Path(matrix_path).absolute(),
        "assignments": Path(assignments_path).absolute(),
        "report": Path(report_path).absolute(),
        "limits": Path(limits_path).absolute(),
    }
    limits, limits_error = _read_yaml(controls["limits"])
    if limits is None or limits_error is not None:
        raise ValueError(f"qualification limits are invalid: {limits_error}")
    control_identities = {
        "matrix": _content_identity(controls["matrix"], max_bytes=MAX_MATRIX_BYTES),
        "assignments": _content_identity(controls["assignments"], max_bytes=MAX_ASSIGNMENT_BYTES),
        "report": _content_identity(controls["report"], max_bytes=MAX_REPORT_BYTES),
        "limits": _content_identity(controls["limits"], max_bytes=MAX_LIMITS_BYTES),
    }
    matrix = load_matrix(controls["matrix"])
    assignments = load_assignments(controls["assignments"])
    report = read_json_object(
        controls["report"], max_bytes=MAX_REPORT_BYTES, label="matrix report"
    )
    _require_exact_keys(report, _REPORT_KEYS, "matrix report")
    cases = expand_cases(matrix)
    case_ids = [str(case["case_id"]) for case in cases]
    if set(assignments) != set(case_ids):
        raise ValueError("assignments must contain every and only canonical matrix case")
    if report.get("matrix_id") != matrix.get("matrix_id"):
        raise ValueError("matrix report matrix_id does not match matrix")
    if report.get("matrix_sha256") != control_identities["matrix"]["sha256"]:
        raise ValueError("matrix report matrix_sha256 does not match matrix bytes")
    if report.get("result") != "passed":
        raise ValueError("only a passed matrix report can be inventoried")
    report_cases = report.get("cases")
    if not isinstance(report_cases, list) or len(report_cases) != len(case_ids):
        raise ValueError("matrix report does not contain the complete case set")
    report_by_id: dict[str, Mapping[str, Any]] = {}
    for item in report_cases:
        if not isinstance(item, Mapping) or not isinstance(item.get("case_id"), str):
            raise ValueError("matrix report contains an invalid case")
        case_id = str(item["case_id"])
        if case_id in report_by_id or case_id not in assignments:
            raise ValueError("matrix report contains duplicate or unknown case IDs")
        report_by_id[case_id] = item
    if set(report_by_id) != set(case_ids):
        raise ValueError("matrix report case IDs do not match matrix")
    declared_counts = (
        report.get("case_count"), report.get("passed_count"),
        report.get("failed_count"), report.get("incomplete_count"),
    )
    expected_counts = (len(case_ids), len(case_ids), 0, 0)
    if any(type(value) is not int for value in declared_counts) or declared_counts != expected_counts:
        raise ValueError("passed matrix report declared counts are inconsistent")
    for case_id, item in report_by_id.items():
        if item.get("result") != "passed" or item.get("qualification_result") != "passed":
            raise ValueError(f"matrix report case did not pass: {case_id}")
        for field in ("factor_mismatches", "provenance_mismatches"):
            if item.get(field) != []:
                raise ValueError(f"matrix report case has {field}: {case_id}")
        summary_identity = item.get("qualification_summary_identity")
        if (
            not isinstance(summary_identity, Mapping)
            or summary_identity.get("available") is not True
        ):
            raise ValueError(
                f"matrix report case lacks qualification summary identity: {case_id}"
            )

    used_runs: set[Path] = set()
    run_records: list[dict[str, Any]] = []
    run_file_sets: dict[Path, tuple[Path, ...]] = {}
    source_snapshots: dict[Path, tuple[int, int, int, int, int]] = {}
    total_files = 0
    for case_id in case_ids:
        assigned = assignments[case_id]
        if assigned is None:
            raise ValueError(f"qualification case is missing a run: {case_id}")
        assigned_path = Path(assigned).absolute()
        if assigned_path.is_symlink():
            raise ValueError(f"assigned run must not be a symlink: {assigned_path}")
        run_dir = assigned_path.resolve()
        if run_dir in used_runs:
            raise ValueError(f"qualification run is reused: {run_dir}")
        used_runs.add(run_dir)
        if not run_dir.is_dir() or run_dir.is_symlink():
            raise ValueError(f"assigned run is missing or not a real directory: {run_dir}")
        report_run = report_by_id[case_id].get("run_directory")
        if not isinstance(report_run, str) or Path(report_run).resolve() != run_dir:
            raise ValueError(f"matrix report run does not match assignment: {case_id}")
        for required in ("run_manifest.json", "run_status.json", "qualification_summary.json"):
            if not (run_dir / required).is_file():
                raise ValueError(f"assigned run is missing {required}: {run_dir}")
        status = read_json_object(
            run_dir / "run_status.json",
            max_bytes=run_context.MAX_RUN_METADATA_BYTES,
            label="run_status.json",
        )
        if status.get("state") not in TERMINAL_SUCCESS_STATES:
            raise ValueError(f"assigned run is not successfully finalized: {run_dir}")
        summary = read_json_object(
            run_dir / "qualification_summary.json",
            max_bytes=run_context.MAX_RUN_METADATA_BYTES,
            label="qualification_summary.json",
        )
        if summary.get("result") != "passed":
            raise ValueError(f"assigned run qualification did not pass: {run_dir}")
        if Path(str(summary.get("run_directory") or "")).resolve() != run_dir:
            raise ValueError(f"qualification summary run directory is inconsistent: {run_dir}")
        summary_limits = summary.get("limits")
        if (
            not isinstance(summary_limits, Mapping)
            or summary_limits.get("validated") is not True
            or not isinstance(summary_limits.get("path"), str)
            or Path(summary_limits["path"]).resolve() != controls["limits"].resolve()
        ):
            raise ValueError(f"qualification summary limits are inconsistent: {run_dir}")
        source_evidence = summary.get("source_evidence")
        recorded_limits = (
            source_evidence.get("limits")
            if isinstance(source_evidence, Mapping)
            else None
        )
        if not _same_content_identity(recorded_limits, control_identities["limits"]):
            raise ValueError(f"qualification summary limits identity is stale: {run_dir}")
        stale_errors = source_evidence_errors(
            run_dir,
            source_evidence,
            declared_limits_path=str(controls["limits"].resolve()),
            verify_recording_content=False,
        )
        if stale_errors:
            raise ValueError(
                f"qualification summary source evidence is stale for {run_dir}: "
                + "; ".join(stale_errors)
            )
        files = _walk_regular_files(run_dir)
        run_file_sets[run_dir] = tuple(files)
        total_files += len(files)
        if total_files > MAX_CAMPAIGN_FILES:
            raise ValueError("campaign artifact count exceeds bounded limit")
        artifacts = []
        for path in files:
            relative = path.relative_to(run_dir).as_posix()
            if len(relative.encode()) > MAX_RELATIVE_PATH_BYTES:
                raise ValueError("run artifact relative path exceeds limit")
            source_snapshots[path] = _stat_snapshot(path)
            artifacts.append({"path": relative, **_content_identity(path)})
        if not _recording_evidence_matches_inventory(
            run_dir,
            source_evidence.get("recording_artifacts")
            if isinstance(source_evidence, Mapping)
            else None,
            artifacts,
        ):
            raise ValueError(
                f"qualification summary recording evidence is stale: {run_dir}"
            )
        summary_artifact = next(
            (
                item
                for item in artifacts
                if item.get("path") == "qualification_summary.json"
            ),
            None,
        )
        if not _same_content_identity(
            report_by_id[case_id].get("qualification_summary_identity"),
            summary_artifact,
        ):
            raise ValueError(
                f"matrix report qualification summary is stale: {run_dir}"
            )
        run_records.append({
            "case_id": case_id,
            "source_run_directory": str(run_dir),
            "archive_path": f"runs/{case_id}",
            "terminal_state": status["state"],
            "artifacts": artifacts,
        })

    # Detect control-file changes after all potentially long artifact hashing.
    for name, path in controls.items():
        maximum = {"matrix": MAX_MATRIX_BYTES, "assignments": MAX_ASSIGNMENT_BYTES,
                   "report": MAX_REPORT_BYTES, "limits": MAX_LIMITS_BYTES}[name]
        if _content_identity(path, max_bytes=maximum) != control_identities[name]:
            raise ValueError(f"campaign control evidence changed while inventorying: {name}")
    # Each individual hash is stable while its file is open.  Rewalk and
    # compare inode/size/time identities so earlier files and directory sets
    # cannot change unnoticed while later multi-gigabyte videos are hashed.
    for run_dir, original_files in run_file_sets.items():
        current_files = tuple(_walk_regular_files(run_dir))
        if current_files != original_files:
            raise ValueError(f"campaign run artifact set changed while inventorying: {run_dir}")
        for path in current_files:
            if _stat_snapshot(path) != source_snapshots[path]:
                raise ValueError(f"campaign run evidence changed while inventorying: {path}")
    raw_output = Path(output_path).absolute()
    if raw_output.is_symlink():
        raise ValueError("campaign inventory output must not be a symlink")
    if raw_output.exists():
        raise ValueError("campaign inventory output already exists; choose a new versioned path")
    output = raw_output.resolve()
    if any(output == run or output.is_relative_to(run) for run in used_runs):
        raise ValueError("campaign inventory output must be outside assigned run directories")
    inventory = {
        "schema_version": SCHEMA_VERSION,
        "matrix_id": matrix["matrix_id"],
        "controls": {
            name: {
                "source_path": str(controls[name]),
                "archive_path": f"control/{name}-{controls[name].name}",
                **control_identities[name],
            }
            for name in _CONTROL_NAMES
        },
        "run_count": len(run_records),
        "artifact_count": total_files,
        "runs": run_records,
    }
    run_context.atomic_write_json(output, inventory)
    return inventory


def verify_campaign_archive(*, archive_root: Path, inventory_path: Path) -> dict[str, Any]:
    """Verify an already-copied archive tree without modifying it."""

    if Path(inventory_path).absolute().is_symlink():
        raise ValueError("campaign inventory must not be a symlink")
    raw_root = Path(archive_root).absolute()
    if raw_root.is_symlink():
        raise ValueError("archive root must not be a symlink")
    root = raw_root.resolve()
    if not root.is_dir():
        raise ValueError("archive root must be a real directory")
    inventory = read_json_object(
        inventory_path, max_bytes=MAX_INVENTORY_BYTES, label="campaign inventory"
    )
    if inventory.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported campaign inventory schema_version")
    _require_exact_keys(inventory, _INVENTORY_KEYS, "campaign inventory")
    if not isinstance(inventory.get("matrix_id"), str) or not inventory[
        "matrix_id"
    ].strip():
        raise ValueError("campaign inventory matrix_id is invalid")
    controls = inventory.get("controls")
    runs = inventory.get("runs")
    if not isinstance(controls, Mapping) or set(controls) != set(_CONTROL_NAMES):
        raise ValueError("campaign inventory control set is invalid")
    if (
        not isinstance(runs, list)
        or not runs
        or len(runs) > MAX_CAMPAIGN_FILES
    ):
        raise ValueError("campaign inventory run set is invalid")
    run_count = inventory.get("run_count")
    artifact_count = inventory.get("artifact_count")
    if type(run_count) is not int or run_count < 0:
        raise ValueError("campaign inventory run_count is invalid")
    if type(artifact_count) is not int or artifact_count < 0:
        raise ValueError("campaign inventory artifact_count is invalid")
    expected: dict[str, Mapping[str, Any]] = {}
    for name in _CONTROL_NAMES:
        record = controls[name]
        if not isinstance(record, Mapping):
            raise ValueError("campaign inventory control identity is invalid")
        _require_exact_keys(record, _CONTROL_RECORD_KEYS, f"control {name}")
        source_path = record.get("source_path")
        if (
            not isinstance(source_path, str)
            or not source_path.strip()
            or not Path(source_path).is_absolute()
        ):
            raise ValueError(f"control {name} source path is invalid")
        _require_content_record(record, f"control {name}")
        archive_path = _safe_relative(record.get("archive_path")).as_posix()
        if archive_path in expected:
            raise ValueError(
                f"campaign inventory contains duplicate path: {archive_path}"
            )
        expected[archive_path] = record
    case_ids: set[str] = set()
    run_prefixes: set[str] = set()
    for run in runs:
        if not isinstance(run, Mapping):
            raise ValueError("campaign inventory run entry is invalid")
        _require_exact_keys(run, _RUN_RECORD_KEYS, "campaign inventory run")
        if (
            not isinstance(run.get("case_id"), str)
            or not isinstance(run.get("source_run_directory"), str)
            or not run["source_run_directory"].strip()
            or not Path(run["source_run_directory"]).is_absolute()
            or run.get("terminal_state") not in TERMINAL_SUCCESS_STATES
        ):
            raise ValueError("campaign inventory run metadata is invalid")
        case_id = run["case_id"]
        if not case_id or case_id in case_ids:
            raise ValueError("campaign inventory contains a duplicate or empty case_id")
        case_ids.add(case_id)
        prefix = _safe_relative(run.get("archive_path"))
        prefix_text = prefix.as_posix()
        if prefix_text in run_prefixes:
            raise ValueError(
                f"campaign inventory contains duplicate run archive path: {prefix_text}"
            )
        run_prefixes.add(prefix_text)
        artifacts = run.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("campaign inventory artifact list is invalid")
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                raise ValueError("campaign inventory artifact identity is invalid")
            _require_exact_keys(artifact, _ARTIFACT_RECORD_KEYS, "campaign artifact")
            _require_content_record(artifact, "campaign artifact")
            relative = prefix / _safe_relative(artifact.get("path"))
            key = relative.as_posix()
            if key in expected:
                raise ValueError(f"campaign inventory contains duplicate path: {key}")
            expected[key] = artifact
            if len(expected) > MAX_CAMPAIGN_FILES + len(_CONTROL_NAMES):
                raise ValueError("campaign inventory file count exceeds limit")
    if run_count != len(runs) or artifact_count != len(expected) - len(_CONTROL_NAMES):
        raise ValueError("campaign inventory declared counts are invalid")
    actual_files = _walk_regular_files(root)
    actual = {path.relative_to(root).as_posix(): path for path in actual_files}
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))[:10]
        extra = sorted(set(actual) - set(expected))[:10]
        raise ValueError(f"archive file set differs: missing={missing!r}, extra={extra!r}")
    snapshots = {path: _stat_snapshot(path) for path in actual_files}
    for relative, record in expected.items():
        identity = _content_identity(actual[relative])
        if identity.get("size_bytes") != record.get("size_bytes") or identity.get(
            "sha256"
        ) != record.get("sha256"):
            raise ValueError(f"archive artifact identity mismatch: {relative}")
    # An individually stable hash does not prove the files hashed earlier
    # remained unchanged while later multi-gigabyte artifacts were read.
    # Rewalk and compare identities before publishing a campaign-level result.
    final_files = _walk_regular_files(root)
    if final_files != actual_files:
        raise ValueError("archive file set changed while verifying")
    for path in final_files:
        if _stat_snapshot(path) != snapshots[path]:
            raise ValueError(f"archive artifact changed while verifying: {path}")
    return {"schema_version": SCHEMA_VERSION, "result": "verified", "file_count": len(expected)}


__all__ = ["build_campaign_inventory", "verify_campaign_archive"]
