"""Bottle swap state and audit records stored in the primary weigh-in table."""
from __future__ import annotations

import csv
import hashlib
from datetime import datetime
import json
from pathlib import Path
import time

from .bounded_csv import bounded_csv_lines
from .run_layout import artifact_path


def load_swap_state(run_dir: Path):
    """Reconstruct the latest committed swap state from streaming CSV records."""
    path = artifact_path(run_dir, "weigh_in.csv")
    if not path.exists():
        return None
    state, bottles, gaps = None, {}, {}
    for row in csv.DictReader(bounded_csv_lines(path)):
        kind = row.get("record_type")
        if kind not in ("swap_state", "bottle_state", "swap_gap"):
            continue
        payload = json.loads(row["details_json"])
        if kind == "swap_state":
            state = payload
        elif kind == "bottle_state":
            bottles[row["bottle_id"]] = payload
        else:
            gaps[row["swap_id"]] = payload
    if state is None:
        return None
    state["current"] = {item["bottle"]["side"]: item["bottle"] for item in bottles.values()
                        if item["status"] == "current"}
    state["history"] = [item["bottle"] for item in sorted(
        (item for item in bottles.values() if item["status"] == "closed"),
        key=lambda item: item["history_index"])]
    state["gaps"] = list(gaps.values())
    if set(state["current"]) != {"left", "right"}:
        raise ValueError("Weigh-in swap state is missing current bottle identities")
    return state


def _record(run_dir, kind, *, controller_time_us=None, **values):
    from .run_outputs import ClockMapping, _json
    now = values.pop("host_unix_ns", None)
    if now is None:
        now = time.time_ns()
    monotonic = values.pop("host_monotonic_ns", time.monotonic_ns())
    subject = _json(artifact_path(run_dir, "run_manifest.json")).get("mouse_id") or _json(
        artifact_path(run_dir, "run_status.json")).get("mouse_id") or ""
    row = dict(record_type=kind, mouse_id=subject, entered_at_unix_ns=now,
               entered_at_iso=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now / 1e9)),
               host_unix_ns=now, host_monotonic_ns=monotonic, **values)
    mapping = ClockMapping(run_dir)
    if controller_time_us is not None:
        row.update(controller_time_us=controller_time_us, controller_time_method="direct_boundary",
                   controller_boot_id=mapping.boot, controller_session_id=mapping.session)
    return mapping.apply(row)


def _details(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def save_swap_state(run_dir: Path, state):
    """Atomically append changed bottle/gap records and their transaction state."""
    from .run_outputs import append_weigh_records
    rows = []
    for status, entries in (("closed", state["history"]), ("current", state["current"].values())):
        for index, bottle in enumerate(entries):
            end = bottle["end_count"]
            count = max(0, (state["global_counts"][bottle["side"]] if end is None else end) - bottle["start_count"])
            initial, final = bottle["initial_weight_g"], bottle["final_weight_g"]
            intake = None if initial is None or final is None else initial - final
            payload = dict(bottle=bottle, status=status, history_index=index if status == "closed" else None,
                           lick_count=count, intake_g=intake)
            rows.append(_record(run_dir, "bottle_state", bottle_id=bottle["bottle_id"],
                side=bottle["side"], fluid=bottle["fluid"], phase=status,
                initial_weight_g=initial, final_weight_g=final, note=bottle.get("note", ""),
                start_count=bottle["start_count"], end_count=end,
                started_monotonic_us=bottle["started_monotonic_us"],
                ended_monotonic_us=bottle["ended_monotonic_us"], lick_count=count, intake_g=intake,
                details_json=_details(payload)))
    for gap in state["gaps"]:
        boundary = gap["resumed_monotonic_us"]
        if boundary is None:
            boundary = gap["paused_monotonic_us"]
        rows.append(_record(run_dir, "swap_gap", swap_id=gap["swap_id"],
            controller_time_us=boundary,
            event="resume_confirmed" if gap["resumed_monotonic_us"] is not None else "pause_confirmed",
            selected_sides=";".join(gap["selected_sides"]),
            paused_monotonic_us=gap["paused_monotonic_us"], resumed_monotonic_us=gap["resumed_monotonic_us"],
            pause_left_count=gap["pause_counts"]["left"], pause_right_count=gap["pause_counts"]["right"],
            resume_left_count=(gap["resume_counts"] or {}).get("left"),
            resume_right_count=(gap["resume_counts"] or {}).get("right"),
            details_json=_details(gap)))
    transaction = {key: value for key, value in state.items() if key not in ("current", "history", "gaps")}
    rows.append(_record(run_dir, "swap_state", swap_id=state["swap_id"], state=state["state"],
        selected_sides=";".join(state["selected_sides"]), error=state["error"],
        left_count=state["global_counts"]["left"], right_count=state["global_counts"]["right"],
        details_json=_details(transaction)))
    append_weigh_records(run_dir, rows, deduplicate=True)


def record_swap_event(run_dir: Path, record):
    from .run_outputs import append_weigh_records
    response = record.get("response", {})
    host = record.get("host_unix_ns")
    if host is None and record.get("recorded_at"):
        recorded = datetime.fromisoformat(record["recorded_at"])
        host = int(recorded.timestamp()) * 1_000_000_000 + recorded.microsecond * 1000
    row = _record(run_dir, "swap_event", event=record["event"], swap_id=record.get("swap_id"),
        host_unix_ns=host, host_monotonic_ns=record.get("host_monotonic_ns", ""),
        record_id=hashlib.sha256(_details(record).encode()).hexdigest(),
        selected_sides=";".join(record.get("selected_sides", [])),
        left_count=response.get("left_count"), right_count=response.get("right_count"),
        state=record.get("state") or response.get("state", ""),
        bottle_id=record.get("bottle_id", ""), old_weight_g=record.get("old_weight_g"),
        final_weight_g=record.get("final_weight_g"), error=record.get("error", ""),
        controller_time_us=response.get("monotonic_us"), details_json=_details(record))
    append_weigh_records(run_dir, [row], deduplicate=True)
