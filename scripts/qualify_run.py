#!/usr/bin/env python3
"""Create a bounded-memory qualification summary for one completed run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from squeakview.common.diagnostics.qualification import qualify_run  # noqa: E402
from squeakview.project import AppPaths, open_project  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--project",
        type=Path,
        default=(
            Path(os.environ["SQUEAKVIEW_PROJECT"])
            if os.environ.get("SQUEAKVIEW_PROJECT")
            else None
        ),
        help="SqueakView project owning the run and qualification limits",
    )
    parser.add_argument(
        "--limits",
        type=Path,
        default=None,
        help="versioned qualification limits YAML",
    )
    parser.add_argument("--output", type=Path, help="override qualification_summary.json path")
    parser.add_argument(
        "--allow-debug-profile",
        action="store_true",
        help=(
            "allow only the explicitly marked DeepStream debug profile to bypass "
            "the production-eligibility gate for paired overhead qualification"
        ),
    )
    args = parser.parse_args(argv)
    try:
        if args.project is None:
            raise ValueError("--project or SQUEAKVIEW_PROJECT is required")
        project = open_project(args.project)
        AppPaths.discover().validate_for_project(project.paths)
        run_dir = project.paths.resolve_path(
            args.run_dir,
            within=project.paths.runs,
            must_exist=True,
        )
        limits_path = project.paths.resolve_path(
            args.limits or (project.paths.qualification / "limits.v1.yaml"),
            within=project.paths.qualification,
            must_exist=True,
        )
        output_path = None
        if args.output is not None:
            output_path = project.paths.resolve_path(args.output, within=run_dir)
        summary = qualify_run(
            run_dir,
            limits_path=limits_path,
            output_path=output_path,
            allow_debug_profile=args.allow_debug_profile,
        )
    except Exception as exc:
        print(json.dumps({"result": "incomplete", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return {"passed": 0, "failed": 1, "incomplete": 2}[summary["result"]]


if __name__ == "__main__":
    raise SystemExit(main())
