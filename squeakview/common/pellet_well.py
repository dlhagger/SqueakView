"""Transient-delivery pellet-well warnings, separate from the feeder jam latch."""

from .dashboard import DashboardEvent

BLOCKED = "WELL_SENSOR_BLOCKED"
CLEAR = "WELL_SENSOR_CLEAR"


def is_pellet_well_status_line(line: str) -> bool:
    return line.split(",", 1)[0].strip().upper() in {BLOCKED, CLEAR, "SYSTEM_START"}


def pellet_well_blocked(event: DashboardEvent) -> bool | None:
    """Return the reported warning state, or None for unrelated telemetry."""

    if event.event_uc == BLOCKED:
        return True
    if event.event_uc == CLEAR or event.raw_line.split(",", 1)[0].strip().upper() == "SYSTEM_START":
        return False
    return None
