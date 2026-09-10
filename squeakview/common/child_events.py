from __future__ import annotations

"""Versioned machine-readable events emitted by capture subprocesses.

Human-readable child output continues to share stdout with these records.  The
prefix lets the operator identify protocol records without coupling lifecycle
decisions to log wording.
"""

import json
from dataclasses import dataclass
from typing import Any, Mapping

from squeakview.common.immutable import deep_freeze


EVENT_PREFIX = "SQUEAKVIEW_EVENT "
SCHEMA_VERSION = 1
SUPPORTED_EVENT_TYPES = frozenset({"pipeline_ready", "capture_closed", "fatal"})
MAX_CHILD_EVENT_CHARS = 64 * 1024
_RESERVED_FIELDS = frozenset({"schema_version", "type"})


@dataclass(frozen=True, slots=True)
class ChildEvent:
    schema_version: int
    type: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", deep_freeze(dict(self.payload)))


def encode_child_event(event_type: str, **payload: Any) -> str:
    """Encode one supported event as a single stdout-safe line."""

    if event_type not in SUPPORTED_EVENT_TYPES:
        raise ValueError(f"unsupported child event type: {event_type}")
    reserved = _RESERVED_FIELDS.intersection(payload)
    if reserved:
        raise ValueError(
            "child event payload uses reserved field(s): "
            + ", ".join(sorted(reserved))
        )
    record = {
        "schema_version": SCHEMA_VERSION,
        "type": event_type,
        **payload,
    }
    try:
        encoded = json.dumps(
            record,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"child event payload is not strict JSON: {exc}") from exc
    if len(encoded) > MAX_CHILD_EVENT_CHARS:
        raise ValueError("child event exceeds its bounded encoded size")
    return EVENT_PREFIX + encoded


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate child event field: {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite child event number: {value}")


def decode_child_event(line: str) -> ChildEvent | None:
    """Decode a supported event embedded in a supervised child-output line.

    Invalid JSON, future schema versions, unknown event types, and ordinary
    human logs return ``None``.  Child output must never be able to crash the
    supervising process.
    """

    marker = line.find(EVENT_PREFIX)
    if marker < 0:
        return None
    raw = line[marker + len(EVENT_PREFIX) :].strip()
    if len(raw) > MAX_CHILD_EVENT_CHARS:
        return None
    try:
        record = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (TypeError, ValueError, RecursionError):
        return None
    if not isinstance(record, dict):
        return None
    if record.get("schema_version") != SCHEMA_VERSION:
        return None
    event_type = record.get("type")
    if not isinstance(event_type, str) or event_type not in SUPPORTED_EVENT_TYPES:
        return None
    payload = {
        key: value
        for key, value in record.items()
        if key not in {"schema_version", "type"}
    }
    try:
        return ChildEvent(SCHEMA_VERSION, event_type, payload)
    except (TypeError, ValueError, RecursionError):
        return None
