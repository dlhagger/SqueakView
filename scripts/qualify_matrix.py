#!/usr/bin/env python3
"""Evaluate assigned run directories against the full qualification matrix."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from squeakview.common.diagnostics.qualification_matrix import (  # noqa: E402
    expand_cases,
    load_assignments,
    load_matrix,
    qualify_matrix,
)
from squeakview.common.bounded_input import read_stable_regular_file  # noqa: E402
from squeakview.common.diagnostics.qualification import (  # noqa: E402
    TERMINAL_FAILURE_STATES,
    TERMINAL_SUCCESS_STATES,
)
from squeakview.common.run_context import atomic_write_text  # noqa: E402


MAX_RUN_METADATA_BYTES = 4 << 20


def _next_case_worksheet(
    matrix: dict,
    assignments: dict[str, str | None],
    *,
    matrix_path: Path,
) -> dict:
    """Return the next unassigned case without modifying the checklist."""

    cases = expand_cases(matrix)
    case_ids = {case["case_id"] for case in cases}
    unknown = sorted(set(assignments) - case_ids)
    if unknown:
        raise ValueError(
            f"qualification assignments contain unknown case IDs: {unknown!r}"
        )
    assigned_count = sum(
        assignments.get(case["case_id"]) is not None for case in cases
    )
    next_case = next(
        (case for case in cases if assignments.get(case["case_id"]) is None),
        None,
    )
    worksheet: dict = {
        "schema_version": "1.0",
        "matrix_id": matrix["matrix_id"],
        "matrix_path": str(Path(matrix_path).resolve()),
        "progress": {
            "assigned": assigned_count,
            "remaining": len(cases) - assigned_count,
            "total": len(cases),
        },
        "next_case": next_case,
        "required_minimum_seconds": (
            next_case["duration"]["minimum_seconds"] if next_case else None
        ),
        "required_environment": None,
        "launch_command": None,
    }
    if next_case is not None:
        environment = {
            "SQUEAKVIEW_QUALIFICATION_CASE_ID": next_case["case_id"],
            "SQUEAKVIEW_QUALIFICATION_MATRIX": str(Path(matrix_path).resolve()),
            "SQUEAKVIEW_DISABLE_PREVIEW": (
                "0" if next_case["preview_enabled"] else "1"
            ),
            "SQUEAKVIEW_DEEPSTREAM_DEBUG_PROFILE": "0",
            "SQUEAKVIEW_VIDEO_VALIDATION_FULL_DECODE": "1",
        }
        worksheet["required_environment"] = environment
        worksheet["launch_command"] = shlex.join(
            [
                *(f"{name}={value}" for name, value in environment.items()),
                "bash",
                "squeakview.sh",
            ]
        )
    return worksheet


def _load_run_metadata(path: Path) -> dict:
    """Read a small stable JSON object and reject duplicate object fields."""

    raw = read_stable_regular_file(
        path, max_bytes=MAX_RUN_METADATA_BYTES, label=path.name
    )

    def unique_object(pairs: list[tuple[str, object]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path.name} contains duplicate field {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name} is not valid strict UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _write_new_assignment_checklist(path: Path, matrix: dict) -> None:
    """Create, but never replace, a complete null-valued assignment checklist."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {case["case_id"]: None for case in expand_cases(matrix)}
    encoded = yaml.safe_dump(payload, sort_keys=False).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and fails if the checklist exists.
        os.link(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _record_assignment(
    assignments_path: Path,
    matrix: dict,
    case_id: str,
    run_dir: Path,
) -> None:
    """Atomically record one completed-run directory in an existing checklist."""

    assignments = load_assignments(assignments_path)
    valid_ids = {case["case_id"] for case in expand_cases(matrix)}
    unknown_ids = sorted(set(assignments) - valid_ids)
    if unknown_ids:
        raise ValueError(
            f"qualification assignments contain unknown case IDs: {unknown_ids!r}"
        )
    if case_id not in valid_ids:
        raise ValueError(f"unknown qualification case ID: {case_id!r}")
    resolved = run_dir.resolve()
    if not resolved.is_dir():
        raise ValueError(f"assigned run directory does not exist: {resolved}")
    metadata: dict[str, dict] = {}
    for filename in ("run_manifest.json", "run_status.json"):
        artifact = resolved / filename
        if not artifact.exists():
            raise ValueError(f"assigned run is missing {filename}: {resolved}")
        metadata[filename] = _load_run_metadata(artifact)
    state = metadata["run_status.json"].get("state")
    terminal_states = TERMINAL_SUCCESS_STATES | TERMINAL_FAILURE_STATES
    if state not in terminal_states:
        raise ValueError(
            f"assigned run is not terminal: state={state!r}, expected one of "
            f"{sorted(terminal_states)!r}"
        )
    duplicate = [
        existing_id
        for existing_id, existing_path in assignments.items()
        if existing_id != case_id
        and existing_path is not None
        and Path(existing_path).resolve() == resolved
    ]
    if duplicate:
        raise ValueError(
            f"run directory is already assigned to case {duplicate[0]!r}: {resolved}"
        )
    assignments[case_id] = str(resolved)
    atomic_write_text(assignments_path, yaml.safe_dump(assignments, sort_keys=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "assignments",
        nargs="?",
        type=Path,
        help="YAML mapping/checklist of case IDs to run directories",
    )
    parser.add_argument("--matrix", type=Path, default=ROOT / "qualification/matrix.v1.yaml")
    parser.add_argument("--limits", type=Path, default=ROOT / "qualification/limits.v1.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "qualification/matrix_report.json")
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--list-cases", action="store_true", help="print canonical matrix cases and exit"
    )
    action.add_argument(
        "--init-assignments",
        type=Path,
        metavar="PATH",
        help="create a complete null-valued assignment checklist without overwriting",
    )
    action.add_argument(
        "--next-case",
        action="store_true",
        help=(
            "print the first unassigned case, progress, and launch environment "
            "without changing the checklist"
        ),
    )
    action.add_argument(
        "--assign",
        nargs=2,
        metavar=("CASE_ID", "RUN_DIR"),
        help="atomically assign one completed run in the assignments checklist",
    )
    args = parser.parse_args()
    try:
        if args.list_cases:
            matrix = load_matrix(args.matrix)
            cases = expand_cases(matrix)
            print(json.dumps({"matrix_id": matrix["matrix_id"], "case_count": len(cases), "cases": cases}, indent=2))
            return 0
        if args.init_assignments is not None:
            matrix = load_matrix(args.matrix)
            _write_new_assignment_checklist(args.init_assignments, matrix)
            print(json.dumps({"result": "created", "assignments": str(args.init_assignments), "case_count": len(expand_cases(matrix))}, indent=2))
            return 0
        if args.next_case:
            if args.assignments is None:
                raise ValueError("the assignments path is required with --next-case")
            matrix = load_matrix(args.matrix)
            assignments = load_assignments(args.assignments)
            worksheet = _next_case_worksheet(
                matrix,
                assignments,
                matrix_path=args.matrix,
            )
            print(json.dumps(worksheet, indent=2, sort_keys=True))
            return 0
        if args.assign is not None:
            if args.assignments is None:
                raise ValueError("the assignments path is required with --assign")
            matrix = load_matrix(args.matrix)
            _record_assignment(
                args.assignments, matrix, args.assign[0], Path(args.assign[1])
            )
            print(json.dumps({"result": "assigned", "case_id": args.assign[0], "run_directory": str(Path(args.assign[1]).resolve())}, indent=2))
            return 0
        if args.assignments is None:
            raise ValueError(
                "assignments path is required (or use --list-cases/--init-assignments)"
            )
        assignments = load_assignments(args.assignments)
        report = qualify_matrix(
            args.matrix,
            assignments,
            limits_path=args.limits,
            output_path=args.output,
        )
    except Exception as exc:
        print(json.dumps({"result": "incomplete", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return {"passed": 0, "failed": 1, "incomplete": 2}[report["result"]]


if __name__ == "__main__":
    raise SystemExit(main())
