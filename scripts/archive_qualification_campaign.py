#!/usr/bin/env python3
"""Inventory or verify a qualification campaign without copying run data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from squeakview.common.diagnostics.campaign_archive import (  # noqa: E402
    build_campaign_inventory,
    verify_campaign_archive,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path, metavar="ARCHIVE_ROOT")
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--assignments", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--limits", type=Path)
    args = parser.parse_args()
    try:
        if args.verify is not None:
            if any((args.matrix, args.assignments, args.report, args.limits)):
                raise ValueError("--verify cannot be combined with campaign source arguments")
            result = verify_campaign_archive(
                archive_root=args.verify, inventory_path=args.inventory
            )
        else:
            missing = [name for name in ("matrix", "assignments", "report", "limits") if getattr(args, name) is None]
            if missing:
                raise ValueError("inventory creation requires: " + ", ".join(missing))
            result = build_campaign_inventory(
                matrix_path=args.matrix, assignments_path=args.assignments,
                report_path=args.report, limits_path=args.limits,
                output_path=args.inventory,
            )
    except (OSError, ValueError) as exc:
        print(json.dumps({"result": "failed", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
