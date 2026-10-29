"""Controller-clock observation gaps reconstructed from canonical sensor events."""
from __future__ import annotations

import csv
from pathlib import Path

from .bounded_csv import bounded_csv_lines
from .bottle_control import parse_lick_recovery
from .run_layout import artifact_path
from .sensor_csv import EVENT_FIELDS

MAX_RECOVERY_GAPS = 65_536


def load_lick_recovery_gaps(run_dir: Path) -> tuple[dict, ...]:
    """Deduplicate both sides/status replays; failed/crashed gaps stay open.

    Recovery IDs are session-local. A scalar monotonic timeline cannot represent
    multiple controller clocks, so ambiguous input is rejected rather than merged.
    """
    path = artifact_path(Path(run_dir), "serial.csv")
    if not path.exists():
        return ()
    gaps = {}
    clocks = set()
    for row in csv.DictReader(bounded_csv_lines(path)):
        event = row.get("eventType", "")
        if not event.startswith("LICK_RECOVERY_"):
            continue
        recovery = parse_lick_recovery(",".join(row.get(field, "") or "" for field in EVENT_FIELDS))
        identity = recovery["recovery_id"]
        if not identity:
            continue
        clock = (row.get("controller_boot_id") or "", row.get("controller_session_id") or "")
        clocks.add(clock)
        if len(clocks) != 1:
            raise ValueError("lick recovery spans multiple controller clocks")
        key = (*clock, identity)
        start = recovery["gap_start_us"]
        if key not in gaps:
            if len(gaps) >= MAX_RECOVERY_GAPS:
                raise ValueError("lick recovery gap count exceeds supported bound")
            gaps[key] = dict(source="automatic_recovery", recovery_id=identity,
                             controller_boot_id=clock[0], controller_session_id=clock[1],
                             paused_monotonic_us=start, resumed_monotonic_us=None,
                             selected_sides=["left", "right"], reason=recovery["reason"])
        gap = gaps[key]
        if gap["paused_monotonic_us"] != start:
            raise ValueError("conflicting lick recovery gap start")
        if recovery["state"] in {"READY", "STOPPED"}:
            end = recovery["monotonic_us"]
            previous = gap["resumed_monotonic_us"]
            if previous is not None and previous != end:
                raise ValueError("conflicting lick recovery gap end")
            gap["resumed_monotonic_us"] = end
    return tuple(sorted(gaps.values(), key=lambda gap: gap["paused_monotonic_us"]))


def observation_valid(gaps, monotonic_us: int) -> bool:
    return not any(gap["paused_monotonic_us"] is not None and gap["paused_monotonic_us"] <= monotonic_us and
                   (gap["resumed_monotonic_us"] is None or monotonic_us < gap["resumed_monotonic_us"])
                   for gap in gaps)
