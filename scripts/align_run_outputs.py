#!/usr/bin/env python3
"""Validate frame, object, video, and controller timing for one run.

The current validator is bounded-memory and writes one compact
``alignment_summary.json``. Canonical CSVs remain unchanged in the run root.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _latest_run_dir() -> Path | None:
    marker = _repo_root() / "runs" / ".latest_run"
    try:
        text = marker.read_text().strip()
    except FileNotFoundError:
        return None
    if not text:
        return None
    path = Path(text)
    return path if path.exists() else None


def build_alignment(
    run_dir: Path,
    out_dir: Path,
    *,
    objects_path: Path | None = None,
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
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        help="Run directory. Defaults to runs/.latest_run.",
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


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir or _latest_run_dir()
    if run_dir is None:
        raise SystemExit("No run_dir given and runs/.latest_run is missing")
    run_dir = run_dir.resolve()
    out_dir = (args.out_dir or run_dir).resolve()
    summary = build_alignment(
        run_dir,
        out_dir,
        objects_path=args.objects.resolve() if args.objects else None,
    )
    (out_dir / "alignment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
