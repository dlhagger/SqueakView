#!/usr/bin/env python3
"""Validate frame, object, video, and controller timing for one run.

The current validator is bounded-memory and writes one compact
``alignment_summary.json``. Canonical CSVs remain unchanged in the run root.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from squeakview.common import run_context  # noqa: E402
from squeakview.common.run_context import atomic_write_json  # noqa: E402
from squeakview.project import AppPaths, open_project  # noqa: E402


def build_alignment(
    run_dir: Path,
    out_dir: Path,
    *,
    objects_path: Path | None = None,
    video_validation: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Run the bounded-memory validator without copying canonical CSVs."""
    try:
        from scripts.align_run_outputs_streaming import build_alignment as streaming_build
    except ModuleNotFoundError as exc:
        if exc.name != "scripts":
            raise
        # When this file is invoked directly, Python places ``scripts/`` rather
        # than the repository root on sys.path.
        from align_run_outputs_streaming import build_alignment as streaming_build

    return streaming_build(
        run_dir,
        out_dir,
        objects_path=objects_path,
        video_validation=video_validation,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="Run directory. Defaults to the active project's latest run.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=(Path(os.environ["SQUEAKVIEW_PROJECT"]) if os.environ.get("SQUEAKVIEW_PROJECT") else None),
        help="project used to resolve the latest run when run_dir is omitted",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Summary directory. Defaults to the run directory.",
    )
    parser.add_argument(
        "--objects",
        type=Path,
        default=None,
        help="Object CSV to validate. Defaults to <run_dir>/objects.csv.",
    )
    return parser.parse_args()


def validation_passed(summary: dict[str, Any]) -> bool:
    """Return whether every standalone alignment check passed."""
    validation = summary.get("validation", {})
    return (
        summary.get("frame_alignment", {}).get("validated") is True
        and validation.get("video_frame_count_matches_frames_csv") is True
        and all(
            int(validation.get(field) or 0) == 0
            for field in (
                "objects_missing_frame_count",
                "object_mapping_failed_rows",
                "object_ts_mismatch_count",
                "object_pts_mismatch_count",
            )
        )
    )


def main() -> int:
    args = parse_args()
    project = open_project(args.project) if args.project is not None else None
    if project is not None:
        AppPaths.discover().validate_for_project(project.paths)
    run_dir = args.run_dir or (
        run_context.latest_run_dir(project.paths.runs) if project is not None else None
    )
    if run_dir is None:
        raise SystemExit("No run_dir given and no project latest-run marker is available")
    run_dir = run_dir.resolve()
    if project is not None:
        run_dir = project.paths.resolve_path(
            run_dir,
            within=project.paths.runs,
            must_exist=True,
        )
    out_dir = (args.out_dir or run_dir).resolve()
    summary = build_alignment(
        run_dir,
        out_dir,
        objects_path=args.objects.resolve() if args.objects else None,
    )
    atomic_write_json(out_dir / "alignment_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if validation_passed(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
