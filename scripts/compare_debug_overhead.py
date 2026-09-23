#!/usr/bin/env python3
"""Compare matched debug-off and debug-on SqueakView qualification runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from squeakview.common import run_context
from squeakview.common.diagnostics.debug_overhead import (
    compare_debug_overhead,
    load_debug_thresholds,
)
from squeakview.project import AppPaths, open_project


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_run", type=Path)
    parser.add_argument("debug_run", type=Path)
    parser.add_argument(
        "--project",
        type=Path,
        default=(
            Path(os.environ["SQUEAKVIEW_PROJECT"])
            if os.environ.get("SQUEAKVIEW_PROJECT")
            else None
        ),
        help="required SqueakView project owning both runs and qualification state",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--case-id",
        help="require both runs to carry this backend-enforced qualification case ID",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        help="explicitly approved version-1 overhead thresholds YAML",
    )
    parser.add_argument(
        "--duration-tolerance-seconds",
        type=float,
        default=5.0,
        help="minimum allowed absolute duration difference (default: 5 seconds)",
    )
    parser.add_argument(
        "--duration-tolerance-percent",
        type=float,
        default=1.0,
        help="allowed duration difference as a percentage of the longer run (default: 1%%)",
    )
    args = parser.parse_args(argv)
    try:
        if args.project is None:
            raise ValueError("--project or SQUEAKVIEW_PROJECT is required")
        project = open_project(args.project)
        AppPaths.discover().validate_for_project(project.paths)
        baseline_run = project.paths.resolve_path(
            args.baseline_run,
            within=project.paths.runs,
            must_exist=True,
        )
        debug_run = project.paths.resolve_path(
            args.debug_run,
            within=project.paths.runs,
            must_exist=True,
        )
        thresholds = None
        if args.thresholds is not None:
            thresholds_path = project.paths.resolve_path(
                args.thresholds,
                within=project.paths.qualification,
                must_exist=True,
            )
            thresholds = load_debug_thresholds(thresholds_path)
        if args.case_id is not None and not args.case_id.strip():
            raise ValueError("--case-id must be non-empty")
        output_path = (
            project.paths.resolve_path(
                args.output,
                within=project.paths.qualification,
            )
            if args.output is not None
            else None
        )
        report = compare_debug_overhead(
            baseline_run,
            debug_run,
            thresholds=thresholds,
            duration_tolerance_seconds=args.duration_tolerance_seconds,
            duration_tolerance_percent=args.duration_tolerance_percent,
            expected_case_id=args.case_id,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"result": "incomplete", "error": str(exc)}, indent=2))
        return 2
    if output_path is not None:
        run_context.atomic_write_json(output_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return {"passed": 0, "failed": 1, "incomplete": 2}[report["result"]]


if __name__ == "__main__":
    raise SystemExit(main())
