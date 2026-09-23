from __future__ import annotations

"""Isolated command-line worker for project-owned TensorRT model construction."""

import argparse
import ctypes
import json
import os
import signal
import sys
import traceback
from pathlib import Path
from typing import Sequence

from squeakview.model_builder import BuildSpec, build_model_package


EVENT_PREFIX = "@@SQUEAKVIEW_MODEL_BUILD@@"
PR_SET_PDEATHSIG = 1


def _event(stage: str, message: str, **extra: object) -> None:
    payload = {"stage": stage, "message": message, **extra}
    print(
        EVENT_PREFIX + json.dumps(payload, sort_keys=True, allow_nan=False),
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-default",
        action="store_true",
        help="publish the package without selecting it as the project default",
    )
    return parser


def _configure_worker_lifetime() -> None:
    """Bind this isolated process group to its Project Setup parent on Linux."""

    if os.name != "posix":
        return
    parent_pid = os.getppid()
    try:
        os.setsid()
    except OSError:
        pass
    if os.getpgrp() != os.getpid() or os.getsid(0) != os.getpid():
        raise RuntimeError("model builder could not create an isolated process session")

    def terminate_group(signum: int, _frame: object) -> None:
        signal.signal(signum, signal.SIG_DFL)
        os.killpg(os.getpgrp(), signum)

    signal.signal(signal.SIGTERM, terminate_group)
    signal.signal(signal.SIGINT, terminate_group)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    except (AttributeError, OSError):
        # QProcess cancellation still targets the complete process group. The
        # parent-death binding is an additional Linux fail-safe.
        pass
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        # The Qt launcher cancels this complete process group. This prevents an
        # exporter or trtexec descendant from surviving cancellation or GUI loss.
        _configure_worker_lifetime()
        result = build_model_package(
            BuildSpec(
                project_root=args.project,
                source_name=args.source,
                model_name=args.model_name,
                overwrite=args.overwrite,
                set_as_default=not args.no_default,
            ),
            emit=_event,
        )
    except Exception as exc:
        _event("failed", " ".join(str(exc).split()) or type(exc).__name__)
        traceback.print_exc(file=sys.stderr)
        return 2
    _event(
        "result",
        "Model build completed successfully",
        model_name=result.model_name,
        package_root=str(result.package_root),
        config=str(result.config),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["EVENT_PREFIX", "main"]
