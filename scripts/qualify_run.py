#!/usr/bin/env python3
"""Create a bounded-memory qualification summary for one completed run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from squeakview.common.diagnostics.qualification import qualify_run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--limits",
        type=Path,
        default=ROOT / "qualification" / "limits.v1.yaml",
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
    args = parser.parse_args()
    try:
        summary = qualify_run(
            args.run_dir,
            limits_path=args.limits,
            output_path=args.output,
            allow_debug_profile=args.allow_debug_profile,
        )
    except Exception as exc:
        print(json.dumps({"result": "incomplete", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return {"passed": 0, "failed": 1, "incomplete": 2}[summary["result"]]


if __name__ == "__main__":
    raise SystemExit(main())
