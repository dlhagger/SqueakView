from __future__ import annotations

"""Linux exec wrapper that terminates an acquisition child with its parent."""

import argparse
import ctypes
import os
import signal
import sys
from collections.abc import Callable, Sequence


PR_SET_PDEATHSIG = 1


class ParentChangedError(RuntimeError):
    """The expected supervisor died before parent-death containment was armed."""


def _linux_prctl(option: int, value: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.prctl(option, value, 0, 0, 0)
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def arm_parent_death_signal(
    expected_parent_pid: int,
    signal_number: int,
    *,
    prctl: Callable[[int, int], None] = _linux_prctl,
    getppid: Callable[[], int] = os.getppid,
) -> None:
    """Arm PDEATHSIG and close the fork-to-prctl parent-death race."""

    if sys.platform != "linux":
        raise RuntimeError("parent-death containment requires Linux")
    if expected_parent_pid <= 1:
        raise ValueError("expected parent PID must be greater than 1")
    if signal_number <= 0:
        raise ValueError("parent-death signal must be positive")
    prctl(PR_SET_PDEATHSIG, signal_number)
    actual_parent_pid = getppid()
    if actual_parent_pid != expected_parent_pid:
        raise ParentChangedError(
            "supervisor parent changed before containment was armed: "
            f"expected {expected_parent_pid}, observed {actual_parent_pid}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-parent-pid", type=int, required=True)
    parser.add_argument("--signal", type=int, default=int(signal.SIGINT))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        print("parent-death wrapper requires a command", file=sys.stderr)
        return 125
    try:
        arm_parent_death_signal(args.expected_parent_pid, args.signal)
        os.execvpe(command[0], command, os.environ.copy())
    except Exception as exc:
        print(f"parent-death wrapper failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 125
    return 125  # pragma: no cover - execvpe does not return


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PR_SET_PDEATHSIG",
    "ParentChangedError",
    "arm_parent_death_signal",
    "main",
]
