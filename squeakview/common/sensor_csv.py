"""Normalize controller messages into the existing sensor-log columns."""
from __future__ import annotations

import csv
import io
import time
import re
from pathlib import Path

from .bounded_csv import bounded_csv_lines

_LOG_FIELDS = ("unixTime", "rp2040Time", "side", "count", "duration", "latency", "value", "context", "reason")
EVENT_FIELDS = ("eventType",) + _LOG_FIELDS
_EVENT_ACKS = {"ACK_START", "ACK_START_BEHAVIOR", "ACK_STOP", "SYSTEM_START"}
_CONTROL_PREFIXES = ("ACK_", "NACK", "CLOCK_", "RTC_", "MOUSEHOUSE_", "TRANSPORT_", "WATCHDOG_", "BOTTLE_")


def _uint(value):
    return str(value).isascii() and str(value).isdigit()


def _standard_event(fields):
    return (len(fields) >= 3 and _uint(fields[1]) and _uint(fields[2])
            and (not fields[0].startswith(_CONTROL_PREFIXES) or fields[0] in _EVENT_ACKS))


def parse_controller_fields(line):
    """Accept quoted CSV and historical unquoted final reason text."""
    try:
        return next(csv.reader([line])) or [""]
    except (csv.Error, StopIteration):
        return [line]


def controller_timestamp(fields, frame=None):
    """Read schema-specific controller time; never use the host echo as time."""
    event = fields[0] if fields else ""
    if event == "CLOCK_SYNC" and len(fields) >= 5 and _uint(fields[3]) and _uint(fields[4]):
        return (int(fields[3]) + int(fields[4])) // 2, "clock_sync_midpoint"
    if event == "CLOCK_STATUS" and len(fields) >= 4 and _uint(fields[3]):
        return int(fields[3]), "rtc_anchor"
    if event == "BOTTLE_STATE" and len(fields) == 8 and _uint(fields[6]):
        return int(fields[6]), "direct_boundary"
    if event in ("CAMERA_EPOCH", "CAMERA_CHECKPOINT", "CAMERA_STOP"):
        values = dict(field.split("=", 1) for field in fields[1:] if "=" in field)
        if _uint(values.get("timestamp_us", "")):
            return int(values["timestamp_us"]), "exact_anchor"
    if _standard_event(fields):
        return int(fields[2]), "direct_event"
    if frame is not None:
        return frame.monotonic_us, "direct_record"
    return "", "unavailable"


def format_sensor_row(row):
    """Keep every message; put non-event payloads in context/reason, not clocks."""
    row = dict(row)
    fields = parse_controller_fields(row.get("rawLine", ""))
    event = fields[0]
    if event == "MARKER":
        return row
    if _standard_event(fields):
        row.update(controller_time_us=int(fields[2]), controller_time_method="direct_event")
        return row
    for name in _LOG_FIELDS:
        row[name] = ""
    structured = re.fullmatch(r"[A-Z][A-Z0-9_]*", event) is not None
    row.update(eventType=event if structured else "CONTROLLER_LOG",
               context="CONTROL" if event.startswith(_CONTROL_PREFIXES) else "DIAGNOSTIC",
               reason=",".join(fields[1:]) if structured else row.get("rawLine", ""))
    native, method = controller_timestamp(fields)
    if native != "":
        row.update(controller_time_us=native, controller_time_method=method)
    row["rp2040Time"] = row.get("controller_time_us", "")
    if event == "CLOCK_SYNC" and len(fields) >= 6 and _uint(fields[5]) and native != "":
        row.update(unixTime=int(fields[5]) - (int(fields[4]) - native), context="CLOCK")
    elif event == "CLOCK_STATUS" and len(fields) >= 4 and _uint(fields[2]):
        row.update(unixTime=fields[2], context="CLOCK")
    elif event == "BOTTLE_STATE":
        row["context"] = "BOTTLE_SWAP"
    elif event == "ACTIVE_POKE_SCHEDULE":
        values = dict(field.split("=", 1) for field in fields[2:] if "=" in field)
        row.update(unixTime=fields[1] if len(fields)>1 and _uint(fields[1]) else "",
                   side=values.get("side", ""), context="ACTIVE_POKE_SCHEDULE")
    elif event in ("CAMERA_EPOCH", "CAMERA_CHECKPOINT", "CAMERA_STOP"):
        values = dict(field.split("=", 1) for field in fields[1:] if "=" in field)
        row.update(count=values.get("count", ""), context="CAMERA_SYNC")
    return row


def make_event_record(fields, raw_line, *, frame=None, host_unix_ns=None,
                      host_monotonic_ns=None, normalize=True):
    """Shared event logger entry point for sensor, diagnostic, and host records."""
    values = list(fields)
    if len(values) > len(EVENT_FIELDS):
        values = values[:9] + [",".join(values[9:])]
    values += [""] * (len(EVENT_FIELDS) - len(values))
    row = dict(zip(EVENT_FIELDS, values))
    native, method = controller_timestamp(fields, frame)
    row.update(rawLine=raw_line,
               hostUnixNs=host_unix_ns if host_unix_ns is not None else time.time_ns(),
               hostMonotonicNs=host_monotonic_ns if host_monotonic_ns is not None else time.monotonic_ns(),
               controller_time_us=native, controller_time_method=method,
               controller_boot_id=frame.boot_id if frame else "",
               controller_session_id=frame.session_id if frame else "",
               controller_sequence=frame.sequence if frame else "")
    return format_sensor_row(row) if normalize else row


def render_event_record(row):
    """The common ten-field terminal serializer for every event source."""
    output = io.StringIO()
    csv.writer(output, lineterminator="").writerow([
        str(row.get(name, "")).replace("\r", "\\r").replace("\n", "\\n")
        for name in EVENT_FIELDS])
    return output.getvalue()


def normalize_existing_sensor_csv(path: Path, header):
    """Format rows adopted from the pre-run temporary CSV without dropping any."""
    if not path.exists() or not path.stat().st_size:
        return
    from .run_outputs import _replacement
    with _replacement(path) as target:
        reader = csv.DictReader(bounded_csv_lines(path))
        columns = list(dict.fromkeys(list(reader.fieldnames or []) + list(header)))
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        for row in reader:
            writer.writerow(format_sensor_row(row))


def format_controller_log(message: str, frame=None) -> str:
    """Render controller RX/TX/status messages with the common ten-field logger.

    Presentation only: protocol bytes and rawLine are never rewritten here.
    Keep the existing direction tags used by the dashboard and log mirror.
    """
    if "【SER】" in message:
        prefix, payload = message.split("【SER】", 1)
        row = make_event_record(parse_controller_fields(payload.strip()), payload.strip(), frame=frame)
        prefix += "【SER】 "
    else:
        command = "【SER→】" in message
        tag = "【SER→】" if command else "[SER]"
        if tag in message:
            prefix, payload = message.split(tag, 1)
            prefix += tag + " "
        else:
            prefix, payload = f"[{time.strftime('%H:%M:%S')}] [SER] ", message
        row = make_event_record([
            "HOST_COMMAND" if command else "HOST_LOG", time.time_ns() // 1000,
            "", "", "", "", "", "", "TX" if command else "HOST", payload.strip(),
        ], payload.strip(), normalize=False)
    return prefix + render_event_record(row)
