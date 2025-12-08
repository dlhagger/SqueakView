from __future__ import annotations

"""Shared parsing helpers for the behavior dashboard."""

import math
import re
import time
from typing import Dict, Optional

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
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("[Arduino] "):
        s = s[len("[Arduino] ") :].lstrip()

    parts = [p.strip() for p in s.split(",")]
    if len(parts) >= 10:
        parts = parts[:10]
        data = dict(zip(CSV_FIELDS, parts))
        event_uc = data["event"].strip().upper()
        data["event_uc"] = event_uc
        side_raw = (data.get("side") or "").strip().upper()
        if side_raw == "LD":
            side_raw = "L"
        if side_raw == "RD":
            side_raw = "R"
        data["side_uc"] = side_raw
        data["unix_sec"] = _to_num(data.get("unix_us")) / 1e6 if data.get("unix_us") else math.nan
        data["micros_sec"] = _to_num(data.get("micros64")) / 1e6 if data.get("micros64") else math.nan
        return data

    event_upper = s.upper()
    if event_upper.startswith("CAMERA_"):
        return {"event_uc": event_upper, "side_uc": "", "unix_sec": time.time(), "micros_sec": math.nan}

    side = "L" if re.search(r"\bLEFT\b|\bL\b", event_upper) else (
        "R" if re.search(r"\bRIGHT\b|\bR\b", event_upper) else ""
    )
    return {"event_uc": event_upper, "side_uc": side, "unix_sec": time.time(), "micros_sec": math.nan}


def is_start_event(data: Dict[str, object]) -> bool:
    event_uc = str(data.get("event_uc", ""))
    context = str(data.get("context", "")).strip().upper()
    reason = str(data.get("reason", "")).strip().upper()
    if reason in START_TOK or context in START_TOK:
        return True
    return any(tok in event_uc for tok in ("_START", " START", " ON", " DOWN", " PRESS", " ARRIVAL"))


def is_end_event(data: Dict[str, object]) -> bool:
    event_uc = str(data.get("event_uc", ""))
    context = str(data.get("context", "")).strip().upper()
    reason = str(data.get("reason", "")).strip().upper()
    if reason in END_TOK or context in END_TOK:
        return True
    return any(tok in event_uc for tok in ("_END", " END", " OFF", " UP", " RELEASE", " RETRIEVAL"))


def choose_event_time(data: Dict[str, object]) -> float:
    unix_sec = float(data.get("unix_sec", math.nan))
    if unix_sec == unix_sec and 1.5778368e9 <= unix_sec <= 4.1024448e9:
        return unix_sec
    return time.time()
