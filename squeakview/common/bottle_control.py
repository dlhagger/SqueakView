"""Strict host decoder for the optional MouseHouse bottle-swap extension."""

import re


SWAP_ID = re.compile(r"[0-9a-f]{32}\Z")
STATES = frozenset({"ACTIVE", "PAUSED", "CALIBRATING", "FAILED"})


def parse_lick_recovery(line: str) -> dict[str, object]:
    """Decode the standard ten-field recovery events without changing swap IDs."""
    fields = line.split(",")
    states = {"START": "RECOVERING", "ATTEMPT": "RECOVERING", "RETRY": "RECOVERING",
              "READY": "READY", "FAILED": "FAILED", "STOPPED": "STOPPED"}
    if len(fields) != 10 or not fields[0].startswith("LICK_RECOVERY_"):
        raise ValueError("malformed lick recovery event")
    suffix = fields[0][len("LICK_RECOVERY_"):]
    if suffix not in states or fields[3] not in {"LD", "RD"} or not re.fullmatch(r"[A-Z0-9_]+", fields[9]):
        raise ValueError("invalid lick recovery state, side, or reason")
    numbers = []
    for index in (2, 4, 5, 6, 7):
        raw = fields[index]
        if not raw.isascii() or not raw.isdecimal() or int(raw) > (1 << 64) - 1:
            raise ValueError("invalid lick recovery counter or timestamp")
        numbers.append(int(raw))
    stamp, count, duration, attempts, identity = numbers
    if duration > stamp or attempts > 3:
        raise ValueError("invalid lick recovery gap or attempts")
    return {"state": states[suffix], "available": suffix == "READY", "reason": fields[9],
            "monotonic_us": stamp, "gap_start_us": stamp - duration,
            "duration_us": duration, "attempts": attempts, "recovery_id": identity,
            "side": "left" if fields[3] == "LD" else "right", "count": count}


def parse_bottle_state(line: str) -> dict[str, object]:
    fields = line.split(",")
    if len(fields) != 8 or fields[:2] != ["BOTTLE_STATE", "1"]:
        raise ValueError("unsupported or malformed bottle-state response")
    _, _, swap_id, state, left, right, stamp, reason = fields
    if swap_id != "none" and SWAP_ID.fullmatch(swap_id) is None:
        raise ValueError("invalid bottle swap identity")
    if state not in STATES or not reason or not re.fullmatch(r"[A-Z0-9_]+", reason):
        raise ValueError("invalid bottle state or reason")
    values = []
    for raw in (left, right, stamp):
        if not raw.isascii() or not raw.isdecimal():
            raise ValueError("bottle counters and timestamps must be unsigned integers")
        value = int(raw)
        if value > (1 << 64) - 1:
            raise ValueError("bottle counter or timestamp exceeds uint64")
        values.append(value)
    return {
        "supported": True,
        "swap_id": None if swap_id == "none" else swap_id,
        "state": state,
        "left_count": values[0],
        "right_count": values[1],
        "monotonic_us": values[2],
        "reason": reason,
    }
