"""Durable bottle identities and controller-confirmed swap boundaries.

Global counters always come from the controller. A gap applies to both sensors,
while a bottle identity changes only on a selected side after confirmed resume.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
import uuid

from squeakview.common.run_context import atomic_write_json

Controller = Callable[[str, str], Mapping[str, Any]]
SIDES = ("left", "right")


class BottleSwapService:
    def __init__(self, run_dir: Path, bottles: Mapping[str, Any] | None = None):
        self.path = Path(run_dir) / "bottle_history.json"
        if self.path.exists():
            if self.path.stat().st_size > 64 * 1024 * 1024:
                raise ValueError("Bottle history exceeds the supported recovery size")
            self._data = json.loads(self.path.read_text())
            if self._data.get("schema_version") != 1:
                raise ValueError("Unsupported bottle history schema")
            # A previous process cannot establish present hardware readiness.
            self._data["available"] = False
            if self._data["state"] not in ("ACTIVE", "STOPPED"):
                self._data["state"] = "UNCONFIRMED"
        else:
            self._data = {
                "schema_version": 1, "available": False, "state": "ACTIVE",
                "swap_id": None, "selected_sides": [], "global_counts": {s: 0 for s in SIDES},
                "current": {}, "history": [], "gaps": [], "error": "",
                "pending_replacements": None,
            }
            for side in SIDES:
                values = (bottles or {}).get(side, {})
                self._data["current"][side] = self._entry(side, values, 0, None)
            self._save()

    @staticmethod
    def _fluid(value):
        fluid = str(value).strip()
        if len(fluid.encode("utf-8")) > 128:
            raise ValueError("Fluid names must be at most 128 UTF-8 bytes")
        return fluid

    @staticmethod
    def _entry(side, values, count, timestamp):
        return {
            "bottle_id": uuid.uuid4().hex, "side": side,
            "fluid": BottleSwapService._fluid(values.get("fluid", "")),
            "initial_weight_g": values.get("initial_weight_g"),
            "final_weight_g": values.get("final_weight_g"),
            "start_count": count, "end_count": None,
            "started_monotonic_us": timestamp, "ended_monotonic_us": None,
        }

    def _save(self):
        atomic_write_json(self.path, self._data)

    def _audit(self, event, **fields):
        path = self.path.with_name("bottle_history_events.jsonl")
        created = not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"event": event, "recorded_at": datetime.now(timezone.utc).isoformat(),
                  "swap_id": self._data["swap_id"], **fields}
        with path.open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if created:
            descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def observe_lick(self, side: str, count: int):
        """Accept an absolute controller count; replay cannot double count."""
        if self._data["state"] == "STOPPED":
            return
        if side not in SIDES or isinstance(count, bool) or int(count) < 0:
            raise ValueError("Invalid side or cumulative lick count")
        self._data["global_counts"][side] = max(self._data["global_counts"][side], int(count))

    def snapshot(self):
        result = deepcopy({key: value for key, value in self._data.items()
                           if key not in ("history", "gaps", "pending_replacements")})
        result["history_count"] = len(self._data["history"])
        result["run_dir"] = str(self.path.parent)
        result["resume_pending"] = self._data["pending_replacements"] is not None
        result["measurements_complete"] = all(
            entry.get("fluid") and entry.get("initial_weight_g") is not None
            and entry.get("final_weight_g") is not None
            for entry in [*self._data["current"].values(), *self._data["history"]]
        )
        result["gap_count"] = len(self._data["gaps"])
        result["history"] = deepcopy(self._data["history"][-50:])
        result["gaps"] = deepcopy(self._data["gaps"][-50:])
        result["support_status"] = "supported" if result["available"] else "unconfirmed"
        self._decorate_entries([*result["current"].values(), *result["history"]])
        # Leave room for the IPC envelope and surrounding command metadata.
        # Full history and all gaps remain durable on disk.
        while len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > 56 * 1024:
            if result["gaps"]:
                result["gaps"].pop(0)
            elif result["history"]:
                result["history"].pop(0)
            else:
                raise ValueError("Current bottle metadata exceeds IPC limit")
        result["history_truncated"] = len(result["history"]) < result["history_count"]
        result["gaps_truncated"] = len(result["gaps"]) < result["gap_count"]
        return result

    def _decorate_entries(self, entries):
        for entry in entries:
            end = entry["end_count"]
            if end is None:
                end = self._data["global_counts"][entry["side"]]
            entry["lick_count"] = entry["licks"] = max(0, end - entry["start_count"])
            initial, final = entry["initial_weight_g"], entry["final_weight_g"]
            entry["intake_g"] = None if initial is None or final is None else initial - final
            entry["complete"] = bool(entry["fluid"]) and initial is not None and final is not None
            entry["warning"] = "Final weight exceeds initial weight" if entry["intake_g"] is not None and entry["intake_g"] < 0 else ""

    def _command(self, controller, action):
        try:
            response = controller(action, self._data["swap_id"] or "")
            if not isinstance(response, Mapping):
                raise ValueError("Missing controller response")
            self.observe_controller(response)
        except Exception as exc:
            self._data["error"] = str(exc)[:1024]
            if action == "status":
                self._data["available"] = False
            else:
                self._data["state"] = "UNCONFIRMED"
            self._save()
        return self.snapshot()

    def begin(self, sides, controller: Controller):
        if isinstance(sides, str):
            sides = SIDES if sides.lower() == "both" else (sides.lower(),)
        selected = [s for s in SIDES if s in sides]
        if not selected or any(s not in SIDES for s in sides):
            raise ValueError("Select left, right, or both bottles")
        if self._data["state"] == "STOPPED":
            raise ValueError("The experiment has stopped")
        self._command(controller, "status")
        if not self._data["available"]:
            self._data["error"] = "Controller does not confirm bottle-swap support"
            return self.snapshot()
        if self._data["swap_id"] and self._data["state"] != "ACTIVE":
            if selected != self._data["selected_sides"]:
                raise ValueError("A different swap is already in progress")
            if self._data["state"] == "PAUSED":
                return self.snapshot()
            return self._command(controller, "pause")
        self._data.update(state="PAUSING", swap_id=uuid.uuid4().hex,
                          selected_sides=selected, pending_replacements=None, error="")
        self._save()  # durable intent before asking hardware to pause
        self._audit("pause_requested", selected_sides=selected)
        return self._command(controller, "pause")

    @staticmethod
    def _weight(value, required=False):
        if value is None or value == "":
            if required:
                raise ValueError("Replacement initial weight is required")
            return None
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError("Bottle weights must be finite and nonnegative")
        return number

    def resume(self, replacements, controller: Controller):
        if self._data["state"] == "ACTIVE" and self._data["pending_replacements"] is None:
            return self.snapshot()  # repeated acknowledgement does not duplicate bottles
        self._command(controller, "status")
        if self._data["state"] == "ACTIVE" and self._data["pending_replacements"] is None:
            return self.snapshot()
        if not self._data["available"]:
            raise ValueError("Controller swap status is unavailable")
        if self._data["state"] not in ("PAUSED", "FAILED"):
            raise ValueError("Controller must confirm both lick channels paused before resuming")
        cleaned = {}
        for side in self._data["selected_sides"]:
            values = replacements.get(side, {})
            fluid = self._fluid(values.get("fluid", ""))
            if not fluid:
                raise ValueError(f"{side.title()} replacement fluid is required")
            cleaned[side] = {"fluid": fluid,
                             "initial_weight_g": self._weight(values.get("initial_weight_g"), True),
                             "final_weight_g": self._weight(values.get("final_weight_g"))}
        self._data.update(pending_replacements=cleaned, state="CALIBRATING", error="")
        self._save()  # replacement metadata survives ambiguous resume delivery
        self._audit("resume_requested", replacements=cleaned)
        return self._command(controller, "resume")

    def observe_controller(self, response: Mapping[str, Any]):
        if self._data["state"] == "STOPPED":
            return self.snapshot()
        if "supported" in response:
            self._data["available"] = response["supported"] is True
        for side in SIDES:
            if response.get(f"{side}_count") is not None:
                self.observe_lick(side, response[f"{side}_count"])
        state = str(response.get("state", "")).upper()
        swap_id = response.get("swap_id")
        ours = self._data["swap_id"]
        if ours and str(swap_id) != str(ours):
            # Counts are globally meaningful; another transaction's state is not.
            self._save()
            return self.snapshot()
        timestamp = response.get("monotonic_us")
        if ours and state in ("PAUSED", "ACTIVE") and (
                timestamp is None or any(response.get(f"{s}_count") is None for s in SIDES)):
            self._data.update(state="UNCONFIRMED", error="Controller boundary is incomplete")
            self._save()
            return self.snapshot()
        boundary_counts = {
            side: int(response.get(f"{side}_count", self._data["global_counts"][side]))
            for side in SIDES
        }
        closed = any(g["swap_id"] == ours and g["resumed_monotonic_us"] is not None
                     for g in self._data["gaps"])
        if self._data["state"] == "STOPPED" or (closed and state != "ACTIVE"):
            return self.snapshot()
        if state == "PAUSED" and ours:
            if not any(g["swap_id"] == ours for g in self._data["gaps"]):
                self._data["gaps"].append({"swap_id": ours,
                    "selected_sides": list(self._data["selected_sides"]),
                    "paused_monotonic_us": timestamp, "resumed_monotonic_us": None,
                    "pause_counts": boundary_counts, "resume_counts": None})
            self._data["state"] = "PAUSED"
            self._data["error"] = ""
        elif state == "CALIBRATING" and ours:
            self._data["state"] = "CALIBRATING"
        elif state == "ACTIVE":
            pending = self._data["pending_replacements"]
            if pending and ours:
                gaps = [g for g in self._data["gaps"] if g["swap_id"] == ours]
                if not gaps:
                    self._data.update(state="UNCONFIRMED", error="Missing confirmed pause boundary")
                    self._save()
                    return self.snapshot()
                gap = gaps[-1]
                for side, values in pending.items():
                    old = self._data["current"][side]
                    old.update(final_weight_g=values["final_weight_g"],
                               end_count=gap["pause_counts"][side],
                               ended_monotonic_us=gap["paused_monotonic_us"])
                    self._data["history"].append(old)
                    self._data["current"][side] = self._entry(side, {
                        "fluid": values["fluid"], "initial_weight_g": values["initial_weight_g"]},
                        boundary_counts[side], timestamp)
                gap.update(resumed_monotonic_us=timestamp,
                           resume_counts=boundary_counts)
                self._data["pending_replacements"] = None
            # Polling may coalesce the intermediate CALIBRATING response.
            # A matching ACTIVE boundary commits a pending replacement even
            # when the last observed local state was FAILED or PAUSED.
            if pending or self._data["state"] in ("ACTIVE", "CALIBRATING", "UNCONFIRMED"):
                self._data.update(state="ACTIVE", error="")
        elif state in ("FAILED", "ERROR"):
            self._data.update(state="FAILED", error=str(response.get("reason") or "Calibration failed")[:1024])
        self._save()
        return self.snapshot()

    def update_final_weight(self, bottle_id: str, weight):
        """Correct a closed bottle measurement without changing event provenance."""
        final = self._weight(weight)
        for entry in self._data["history"]:
            if entry["bottle_id"] == bottle_id:
                self._audit("final_weight_corrected", bottle_id=bottle_id,
                            old_weight_g=entry["final_weight_g"], final_weight_g=final)
                entry["final_weight_g"] = final
                entry["measurement_updated_at"] = datetime.now(timezone.utc).isoformat()
                self._save()
                return self.snapshot()
        raise ValueError("Completed bottle was not found")

    def history_page(self, offset=0, limit=50):
        """Read complete history in bounded pages, oldest entry first."""
        if offset < 0 or not 1 <= limit <= 50:
            raise ValueError("Invalid history page")
        result = deepcopy(self._data["history"][offset:offset + limit])
        self._decorate_entries(result)
        return result

    def sync_measurements(self, bottles: Mapping[str, Any]):
        """Update current measurements without rewriting completed bottle identities."""
        updated = deepcopy(self._data["current"])
        for side in SIDES:
            values = bottles.get(side, {})
            if not isinstance(values, Mapping):
                continue
            entry = updated[side]
            for key in ("fluid", "initial_weight_g", "final_weight_g"):
                if key in values:
                    if key == "fluid":
                        entry[key] = self._fluid(values[key])
                    else:
                        entry[key] = self._weight(values[key])
        self._data["current"] = updated
        self._save()
        return self.snapshot()

    def finalize(self, bottles: Mapping[str, Any] | None = None):
        if bottles is not None:
            self.sync_measurements(bottles)
        if self._data["state"] != "STOPPED":
            interrupted = self._data["state"] != "ACTIVE" or self._data["pending_replacements"] is not None
            self._data["unresolved_swap"] = interrupted
            self._data["interrupted_state"] = self._data["state"] if interrupted else None
            if interrupted:
                # Final weight describes the removed OLD bottle even if capture
                # stops before calibration confirms any replacement as active.
                for side, values in (self._data["pending_replacements"] or {}).items():
                    self._data["current"][side]["final_weight_g"] = values["final_weight_g"]
                self._audit("swap_interrupted_by_stop", state=self._data["state"])
        self._data["state"] = "STOPPED"
        self._data["available"] = False
        self._save()
        return self.snapshot()
