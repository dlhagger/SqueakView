from __future__ import annotations

"""Interpret supervised capture-process output without owning run lifecycle.

The capture child shares human-readable logs and versioned protocol records on
stdout.  Keeping interpretation here makes the protocol independently
testable, while the operator manager remains responsible for performing state
transitions and shutdown.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from squeakview.common.child_events import EVENT_PREFIX, decode_child_event


ActionKind = Literal["none", "ready", "fatal", "capture_closed"]


@dataclass(frozen=True, slots=True)
class CaptureOutputAction:
    """One validated lifecycle action derived from a child output line."""

    kind: ActionKind = "none"
    run_dir: Path | None = None
    warning: str | None = None
    error: str | None = None
    discover_run_dir: bool = False


def _validated_run_dir(
    raw: object,
    *,
    expected_run_dir: Path | None,
    is_directory: Callable[[Path], bool],
) -> tuple[Path | None, str | None]:
    if not isinstance(raw, str) or not raw.strip():
        return None, "capture protocol record is missing a valid run_dir"
    candidate = Path(raw.strip())
    if not is_directory(candidate):
        return None, f"capture protocol run_dir is not an existing directory: {candidate}"
    if expected_run_dir is not None:
        try:
            matches = candidate.resolve() == expected_run_dir.resolve()
        except OSError as exc:
            return None, f"capture protocol run_dir could not be resolved: {exc}"
        if not matches:
            return None, (
                "capture protocol run_dir does not match the prepared run: "
                f"reported={candidate}, expected={expected_run_dir}"
            )
    return candidate, None


def interpret_capture_output(
    message: str,
    *,
    expected_run_dir: Path | None,
    recording_started: bool,
    is_directory: Callable[[Path], bool] = Path.is_dir,
) -> CaptureOutputAction:
    """Translate one child-output line into a fail-closed lifecycle action.

    Any line carrying the protocol prefix is treated as a protocol record.  A
    malformed/future record therefore cannot fall through to legacy text
    readiness and accidentally arm a scientific acquisition.
    """

    event = decode_child_event(message)
    if EVENT_PREFIX in message and event is None:
        return CaptureOutputAction(
            kind="fatal",
            error="capture emitted an invalid or unsupported structured event",
        )

    if event is not None:
        run_dir, error = _validated_run_dir(
            event.payload.get("run_dir"),
            expected_run_dir=expected_run_dir,
            is_directory=is_directory,
        )
        if error is not None:
            return CaptureOutputAction(kind="fatal", error=error)
        if event.type == "pipeline_ready":
            return CaptureOutputAction(kind="ready", run_dir=run_dir)
        if event.type == "fatal":
            detail = event.payload.get("error")
            if not isinstance(detail, str) or not detail.strip():
                detail = "capture child reported a fatal error without details"
            return CaptureOutputAction(kind="fatal", run_dir=run_dir, error=detail.strip())
        return CaptureOutputAction(kind="capture_closed", run_dir=run_dir)

    lower = message.lower()

    if "[ready] inference playing" in lower and not recording_started:
        return CaptureOutputAction(
            kind="none",
            warning=(
                "ignored legacy text readiness marker; controller START requires "
                "a validated structured pipeline_ready event"
            ),
        )

    return CaptureOutputAction()


__all__ = ["CaptureOutputAction", "interpret_capture_output"]
