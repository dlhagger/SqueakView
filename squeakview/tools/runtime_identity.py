"""Isolated device identity worker; never imported by the operator GUI."""
from __future__ import annotations
import argparse
import json
import signal
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-parent-pid", type=int, required=True)
    args = parser.parse_args()
    from squeakview.apps.operator.backend.parent_death_exec import arm_parent_death_signal
    arm_parent_death_signal(args.expected_parent_pid, signal.SIGKILL)
    from squeakview.model_package import _probe_runtime_engine_identity
    try:
        payload = {"identity": _probe_runtime_engine_identity()}
        code = 0
    except Exception as exc:
        payload = {"error": str(exc)[:4096]}
        code = 1
    args.output.write_text(json.dumps(payload, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
