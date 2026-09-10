from __future__ import annotations

"""Explicitly gated qualification-only failure plans."""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from squeakview.common.bounded_input import read_stable_regular_file


FAILURE_PLAN_SCHEMA_VERSION = "1.0"
FAILURE_INJECTION_GATE = "SQUEAKVIEW_ENABLE_FAILURE_INJECTION"
MAX_FAILURE_PLAN_BYTES = 64 * 1024
_REQUIRED_FIELDS = frozenset({"schema_version", "target", "kind", "after_frames"})
_OPTIONAL_FIELDS = frozenset({"stream_id", "delay_us"})
SUPPORTED_FAILURES = {
    "flir_source": frozenset(
        {"source_read", "source_incomplete", "capture_ledger_write"}
    ),
    "record_queue": frozenset({"stall"}),
    "encoder": frozenset({"error"}),
    "muxer": frozenset({"error"}),
    "filesink": frozenset({"error", "disk_full"}),
    "serial_controller": frozenset(
        {"read_error", "write_error", "ledger_write_error"}
    ),
    "shutdown": frozenset(
        {"stop_ack_timeout", "capture_exit_unconfirmed", "finalizer_timeout"}
    ),
}


@dataclass(frozen=True, slots=True)
class FailurePlan:
    schema_version: str
    target: str
    kind: str
    after_frames: int
    stream_id: int = 0
    delay_us: int = 0

    def as_manifest(self) -> dict[str, object]:
        return asdict(self)


def validate_failure_plan(plan: FailurePlan) -> None:
    if plan.schema_version != FAILURE_PLAN_SCHEMA_VERSION:
        raise ValueError(
            f"failure plan schema_version must be {FAILURE_PLAN_SCHEMA_VERSION}"
        )
    if plan.target not in SUPPORTED_FAILURES:
        raise ValueError(f"unsupported failure target: {plan.target}")
    if plan.kind not in SUPPORTED_FAILURES[plan.target]:
        raise ValueError(
            f"unsupported {plan.target} failure kind: {plan.kind}"
        )
    if plan.after_frames < 1:
        raise ValueError("failure plan after_frames must be at least 1")
    if (
        plan.target == "filesink"
        and plan.kind == "disk_full"
        and plan.after_frames != 1
    ):
        raise ValueError(
            "filesink disk_full is immediate-only and requires after_frames=1"
        )
    if plan.stream_id < 0:
        raise ValueError("failure plan stream_id cannot be negative")
    if plan.target == "record_queue" and not 1 <= plan.delay_us <= 10_000_000:
        raise ValueError(
            "record_queue stall requires delay_us between 1 and 10000000"
        )
    if plan.target != "record_queue" and plan.delay_us != 0:
        raise ValueError("delay_us is only valid for a record_queue stall")
    if os.environ.get(FAILURE_INJECTION_GATE) != "1":
        raise ValueError(
            f"failure injection requires {FAILURE_INJECTION_GATE}=1"
        )


def load_failure_plan(path: Path) -> FailurePlan:
    path = Path(path)
    try:
        encoded = read_stable_regular_file(
            path, max_bytes=MAX_FAILURE_PLAN_BYTES, label="failure plan"
        )
        raw = encoded.decode("utf-8", errors="strict")

        def exact_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"failure plan contains duplicate field: {key}")
                result[key] = value
            return result

        payload = json.loads(raw, object_pairs_hook=exact_object)
    except ValueError as exc:
        if str(exc).startswith("failure plan"):
            raise
        raise ValueError(f"failure plan could not be read: {exc}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failure plan could not be read: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("failure plan must contain a JSON object")
    fields = set(payload)
    missing = sorted(_REQUIRED_FIELDS - fields)
    unknown = sorted(fields - _REQUIRED_FIELDS - _OPTIONAL_FIELDS)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise ValueError("failure plan fields are invalid; " + "; ".join(details))
    for name in ("schema_version", "target", "kind"):
        if not isinstance(payload[name], str):
            raise ValueError(f"failure plan {name} must be a string")
    for name in ("after_frames", "stream_id", "delay_us"):
        value = payload.get(name, 0)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"failure plan {name} must be an integer")
    plan = FailurePlan(
        schema_version=payload["schema_version"],
        target=payload["target"],
        kind=payload["kind"],
        after_frames=payload["after_frames"],
        stream_id=payload.get("stream_id", 0),
        delay_us=payload.get("delay_us", 0),
    )
    validate_failure_plan(plan)
    return plan


__all__ = [
    "FAILURE_INJECTION_GATE",
    "FAILURE_PLAN_SCHEMA_VERSION",
    "MAX_FAILURE_PLAN_BYTES",
    "FailurePlan",
    "load_failure_plan",
    "validate_failure_plan",
]
