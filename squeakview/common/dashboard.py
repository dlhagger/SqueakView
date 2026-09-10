from __future__ import annotations

"""Shared parsing helpers for the behavior dashboard."""

import math
import re
import time
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

CSV_FIELDS = [
    "event",
    "unix_us",
    "micros64",
    "side",
    "count",
    "duration_us",
    "latency_us",
    "value",
    "context",
    "reason",
]

START_TOK = {"START", "ON", "DOWN", "PRESS", "ARRIVAL"}
END_TOK = {"END", "OFF", "UP", "RELEASE", "RETRIEVAL"}
DASHBOARD_EVENT_SCHEMA_VERSION = "1.0"
MAX_DASHBOARD_RAW_CHARS = 4096
MAX_DASHBOARD_FIELD_CHARS = 1024
_DASHBOARD_PAYLOAD_KEYS = {
    "schema_version",
    "raw_line",
    "event_uc",
    "side_uc",
    "unix_sec",
    "micros_sec",
    "count",
    "duration_us",
    "latency_us",
    "value",
    "context",
    "reason",
    "truncated",
}


def _bounded_text(value: object, maximum: int) -> tuple[str, bool]:
    text = value if isinstance(value, str) else str(value or "")
    if len(text) <= maximum:
        return text, False
    return text[:maximum], True


def _finite_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


@dataclass(frozen=True, slots=True)
class DashboardEvent:
    """One immutable, JSON-safe controller event for live presentation."""

    raw_line: str
    event_uc: str
    side_uc: str = ""
    unix_sec: float | None = None
    micros_sec: float | None = None
    count: str = ""
    duration_us: str = ""
    latency_us: str = ""
    value: str = ""
    context: str = ""
    reason: str = ""
    truncated: bool = False
    schema_version: str = DASHBOARD_EVENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DASHBOARD_EVENT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported dashboard event schema: {self.schema_version!r}"
            )
        if type(self.truncated) is not bool:
            raise ValueError("dashboard event truncated must be boolean")
        for name, maximum in (
            ("raw_line", MAX_DASHBOARD_RAW_CHARS),
            ("event_uc", MAX_DASHBOARD_FIELD_CHARS),
            ("side_uc", MAX_DASHBOARD_FIELD_CHARS),
            ("count", MAX_DASHBOARD_FIELD_CHARS),
            ("duration_us", MAX_DASHBOARD_FIELD_CHARS),
            ("latency_us", MAX_DASHBOARD_FIELD_CHARS),
            ("value", MAX_DASHBOARD_FIELD_CHARS),
            ("context", MAX_DASHBOARD_FIELD_CHARS),
            ("reason", MAX_DASHBOARD_FIELD_CHARS),
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) > maximum:
                raise ValueError(
                    f"dashboard event {name} must be a string of at most {maximum} characters"
                )
        if not self.event_uc:
            raise ValueError("dashboard event event_uc must not be empty")
        for name in ("unix_sec", "micros_sec"):
            value = getattr(self, name)
            if value is not None and _finite_or_none(value) is None:
                raise ValueError(f"dashboard event {name} must be finite or null")

    @classmethod
    def parse(cls, raw: str, *, host_time: float | None = None) -> DashboardEvent | None:
        """Parse and sanitize one raw local serial/dashboard line exactly once."""

        if not isinstance(raw, str) or not raw:
            return None
        raw_line, raw_truncated = _bounded_text(raw.strip(), MAX_DASHBOARD_RAW_CHARS)
        if not raw_line:
            return None
        source = raw_line
        if source.startswith("[Arduino] "):
            source = source[len("[Arduino] ") :].lstrip()
        parts = [part.strip() for part in source.split(",", 9)]
        truncated = raw_truncated
        values: dict[str, str] = {}
        if len(parts) >= len(CSV_FIELDS):
            for name, part in zip(CSV_FIELDS, parts[: len(CSV_FIELDS)]):
                values[name], clipped = _bounded_text(part, MAX_DASHBOARD_FIELD_CHARS)
                truncated |= clipped
            event_uc = values["event"].upper()
            side_uc = values["side"].upper()
            if side_uc == "LD":
                side_uc = "L"
            elif side_uc == "RD":
                side_uc = "R"
            unix_value = _to_num(values.get("unix_us"))
            micros_value = _to_num(values.get("micros64"))
            unix_sec = unix_value / 1e6 if math.isfinite(unix_value) else None
            micros_sec = micros_value / 1e6 if math.isfinite(micros_value) else None
        else:
            event_uc, clipped = _bounded_text(
                source.upper(), MAX_DASHBOARD_FIELD_CHARS
            )
            truncated |= clipped
            side_uc = (
                "L"
                if re.search(r"\bLEFT\b|\bL\b", event_uc)
                else "R"
                if re.search(r"\bRIGHT\b|\bR\b", event_uc)
                else ""
            )
            now = time.time() if host_time is None else host_time
            unix_sec = _finite_or_none(now)
            micros_sec = None
        if not event_uc:
            return None
        return cls(
            raw_line=raw_line,
            event_uc=event_uc,
            side_uc=side_uc,
            unix_sec=unix_sec,
            micros_sec=micros_sec,
            count=values.get("count", ""),
            duration_us=values.get("duration_us", ""),
            latency_us=values.get("latency_us", ""),
            value=values.get("value", ""),
            context=values.get("context", ""),
            reason=values.get("reason", ""),
            truncated=truncated,
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "raw_line": self.raw_line,
            "event_uc": self.event_uc,
            "side_uc": self.side_uc,
            "unix_sec": self.unix_sec,
            "micros_sec": self.micros_sec,
            "count": self.count,
            "duration_us": self.duration_us,
            "latency_us": self.latency_us,
            "value": self.value,
            "context": self.context,
            "reason": self.reason,
            "truncated": self.truncated,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> DashboardEvent:
        if not isinstance(payload, Mapping) or set(payload) != _DASHBOARD_PAYLOAD_KEYS:
            raise ValueError("dashboard event payload must contain exact version-1 fields")
        if type(payload["schema_version"]) is not str:
            raise ValueError("dashboard event schema_version must be a string")
        for name in _DASHBOARD_PAYLOAD_KEYS - {
            "schema_version",
            "unix_sec",
            "micros_sec",
            "truncated",
        }:
            if type(payload[name]) is not str:
                raise ValueError(f"dashboard event {name} must be a string")
        for name in ("unix_sec", "micros_sec"):
            value = payload[name]
            if value is not None and type(value) not in (int, float):
                raise ValueError(f"dashboard event {name} must be numeric or null")
        return cls(**dict(payload))

    def as_legacy_mapping(self) -> Dict[str, object]:
        return {
            "event_uc": self.event_uc,
            "side_uc": self.side_uc,
            "unix_sec": self.unix_sec if self.unix_sec is not None else math.nan,
            "micros_sec": self.micros_sec if self.micros_sec is not None else math.nan,
            "count": self.count,
            "duration_us": self.duration_us,
            "latency_us": self.latency_us,
            "value": self.value,
            "context": self.context,
            "reason": self.reason,
        }


def _to_num(value: Optional[str], default: float = math.nan) -> float:
    if value is None:
        return default
    try:
        value = value.strip()
    except AttributeError:
        return default
    if not value or value.lower() == "nan":
        return default
    try:
        return float(value)
    except Exception:
        return default


def parse_line(raw: str) -> Optional[Dict[str, object]]:
    event = DashboardEvent.parse(raw)
    return None if event is None else event.as_legacy_mapping()


def is_start_event(data: Dict[str, object]) -> bool:
    event_uc = str(data.get("event_uc", ""))
    context = str(data.get("context", "")).strip().upper()
    reason = str(data.get("reason", "")).strip().upper()
    if reason in START_TOK or context in START_TOK:
        return True
    return any(tok in event_uc for tok in ("_START", " START", " ON", " DOWN", " PRESS", " ARRIVAL", "_ARRIVAL"))


def is_end_event(data: Dict[str, object]) -> bool:
    event_uc = str(data.get("event_uc", ""))
    context = str(data.get("context", "")).strip().upper()
    reason = str(data.get("reason", "")).strip().upper()
    if reason in END_TOK or context in END_TOK:
        return True
    return any(tok in event_uc for tok in ("_END", " END", " OFF", " UP", " RELEASE", " RETRIEVAL", "_RETRIEVAL"))


def choose_event_time(data: Dict[str, object]) -> float:
    unix_sec = float(data.get("unix_sec", math.nan))
    if unix_sec == unix_sec and 1.5778368e9 <= unix_sec <= 4.1024448e9:
        return unix_sec
    return time.time()
