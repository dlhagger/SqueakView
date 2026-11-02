"""Feeder-latch message semantics shared by presentation and supervision."""

from enum import Enum

from .dashboard import DashboardEvent


class FeederJamEvent(str, Enum):
    JAMMED = "jammed"
    CLEAR_ACK = "clear_ack"
    CLEAR_FEED_ACTIVE = "clear_feed_active"
    CLEAR_NOT_JAMMED = "clear_not_jammed"


def feeder_jam_event(event: DashboardEvent) -> FeederJamEvent | None:
    """Accept exact raw replies and their normalized ten-column log form."""
    raw = event.raw_line.strip().upper()
    if event.context.strip().upper() == "CONTROL":
        # Serial presentation formats short replies into the standard schema.
        raw = event.event_uc
        if event.reason.strip():
            raw += "," + event.reason.strip().upper()
    if event.event_uc == "FEED_JAM" or raw == "NACK,FEED,JAMMED":
        return FeederJamEvent.JAMMED
    if raw == "ACK_CLEAR_JAM":
        return FeederJamEvent.CLEAR_ACK
    if raw == "NACK,CLEAR_JAM,FEED_ACTIVE":
        return FeederJamEvent.CLEAR_FEED_ACTIVE
    if raw == "NACK,CLEAR_JAM,NOT_JAMMED":
        return FeederJamEvent.CLEAR_NOT_JAMMED
    if event.event_uc == "FEED_STOP" and "FEEDER JAM" in event.reason.upper():
        return FeederJamEvent.JAMMED
    return None
