from __future__ import annotations

"""Pure configuration and event semantics for the behavior dashboard.

Keeping these transformations free of Qt lets the scientific event-counting rules be
tested independently from widget construction and rendering.
"""

from dataclasses import dataclass
from typing import Any, Mapping

from squeakview.common import dashboard as dash_util


DEFAULT_TASK_CONFIG: dict[str, Any] = {
    "task_name": "SqueakView Default",
    "schema_version": 1,
    "events": [
        {"name": "POKE", "match": {"event_contains": "POKE", "phase": "start"}, "split_by_side": True, "plot": "Poke"},
        {"name": "DRINK", "match": {"event_contains": "DRINK", "phase": "start"}, "split_by_side": True, "plot": "Drink"},
        {"name": "PELLET", "match": {"event_contains": "PELLET", "phase": "retrieval"}, "split_by_side": False, "plot": "Pellet"},
        {"name": "WELL_CHECK", "match": {"event_contains": "WELL_CHECK", "phase": "start"}, "split_by_side": False, "plot": "Pellet"},
    ],
    "dashboard": {
        "plots": [
            {"id": "Poke", "title": "Pokes", "series": ["POKE_L", "POKE_R"]},
            {"id": "Drink", "title": "Drinks", "series": ["DRINK_L", "DRINK_R"]},
            {"id": "Pellet", "title": "Pellet & Well", "series": ["PELLET", "WELL_CHECK"]},
        ]
    },
}


@dataclass(frozen=True)
class DashboardDefinition:
    """Validated portions of task configuration consumed by the dashboard."""

    rules: tuple[dict[str, Any], ...]
    plots: tuple[dict[str, Any], ...]
    series_order: tuple[str, ...]
    settings_panel: bool


def default_task_config() -> dict[str, Any]:
    """Return a fresh default config so callers cannot mutate the module template."""

    return {
        "task_name": DEFAULT_TASK_CONFIG["task_name"],
        "schema_version": DEFAULT_TASK_CONFIG["schema_version"],
        "events": [
            {**rule, "match": dict(rule["match"])}
            for rule in DEFAULT_TASK_CONFIG["events"]
        ],
        "dashboard": {
            "plots": [
                {**plot, "series": list(plot["series"])}
                for plot in DEFAULT_TASK_CONFIG["dashboard"]["plots"]
            ]
        },
    }


def normalize_series(name: object) -> str:
    return str(name).strip().upper()


def event_data(event: dash_util.DashboardEvent | str) -> dict[str, object] | None:
    """Expose typed events to legacy dashboard semantics at the local GUI edge."""

    parsed = (
        dash_util.DashboardEvent.parse(event)
        if isinstance(event, str)
        else event
    )
    if not isinstance(parsed, dash_util.DashboardEvent):
        return None
    return parsed.as_legacy_mapping()


def compile_task_config(config: object) -> DashboardDefinition:
    """Normalize a task config, falling back atomically when it is unusable."""

    cfg = config if isinstance(config, Mapping) else {}
    events = cfg.get("events")
    dashboard = cfg.get("dashboard")
    plots = dashboard.get("plots") if isinstance(dashboard, Mapping) else None
    if not isinstance(events, list) or not events or not isinstance(plots, list) or not plots:
        cfg = default_task_config()
        events = cfg["events"]
        dashboard = cfg["dashboard"]
        plots = dashboard["plots"]

    rules: list[dict[str, Any]] = []
    for raw_rule in events:
        if not isinstance(raw_rule, Mapping):
            continue
        name = str(raw_rule.get("name", "")).strip()
        if not name:
            continue
        raw_match = raw_rule.get("match")
        if not isinstance(raw_match, Mapping):
            raw_match = {}
        rules.append(
            {
                "name": name.upper(),
                "match": {
                    str(key): str(value).strip()
                    for key, value in raw_match.items()
                    if value is not None
                },
                "split_by_side": bool(raw_rule.get("split_by_side", False)),
                "use_count_field": bool(raw_rule.get("use_count_field", False)),
            }
        )

    normalized_plots = tuple(dict(plot) for plot in plots if isinstance(plot, Mapping))
    series_order: list[str] = []
    for plot in normalized_plots:
        raw_series = plot.get("series")
        if not isinstance(raw_series, (list, tuple)):
            continue
        for series in raw_series:
            key = normalize_series(series)
            if key and key not in series_order:
                series_order.append(key)
    for rule in rules:
        base = rule["name"]
        keys = (f"{base}_L", f"{base}_R") if rule["split_by_side"] else (base,)
        for key in keys:
            if key not in series_order:
                series_order.append(key)

    return DashboardDefinition(
        rules=tuple(rules),
        plots=normalized_plots,
        series_order=tuple(series_order),
        settings_panel=bool(dashboard.get("settings_panel", False)),
    )


def infer_pellet_mode(
    configured_mode: str,
    observed_mode: str | None,
    data: Mapping[str, object],
    event: str,
) -> str | None:
    """Return the updated observed pellet convention for one event."""

    if configured_mode != "auto" or ("PELLET" not in event and "WELL_CHECK" not in event):
        return observed_mode
    if "PELLET_RETRIEVAL" in event or ("PELLET" in event and dash_util.is_end_event(dict(data))):
        return "both" if observed_mode == "arrival" else "retrieval"
    if "PELLET_ARRIVAL" in event or ("PELLET" in event and dash_util.is_start_event(dict(data))):
        return "both" if observed_mode == "retrieval" else "arrival"
    if "WELL_CHECK" in event and dash_util.is_start_event(dict(data)) and observed_mode is None:
        return "retrieval"
    return observed_mode


def effective_pellet_mode(configured_mode: str, observed_mode: str | None) -> str:
    return configured_mode if configured_mode != "auto" else (observed_mode or "retrieval")


def rule_matches(
    data: Mapping[str, object],
    event: str,
    rule: Mapping[str, object],
    *,
    pellet_mode: str,
) -> bool:
    match = rule.get("match")
    if not isinstance(match, Mapping):
        match = {}
    contains = match.get("event_contains")
    if contains and str(contains).upper() not in event:
        return False
    equals = match.get("event_equals")
    if equals and str(equals).upper() != event:
        return False
    reason = str(data.get("reason", ""))
    reason_equals = match.get("reason_equals")
    if reason_equals and str(reason_equals).upper() != reason.upper():
        return False
    reason_contains = match.get("reason_contains")
    if reason_contains and str(reason_contains).upper() not in reason.upper():
        return False
    value_equals = match.get("value_equals")
    if value_equals is not None and str(value_equals) != str(data.get("value", "")).strip():
        return False
    side = match.get("side")
    if side and str(side).upper() != str(data.get("side_uc", "")).upper():
        return False
    phase = str(match.get("phase", "")).strip().lower()
    event_data = dict(data)
    if phase in ("start", "arrival"):
        return dash_util.is_start_event(event_data) or (
            "PELLET" in event and pellet_mode in ("retrieval", "both") and "RETRIEVAL" in event
        )
    if phase in ("end", "retrieval"):
        return dash_util.is_end_event(event_data) or (
            "PELLET" in event and pellet_mode in ("arrival", "both") and "ARRIVAL" in event
        )
    return True


def parse_int_field(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return None
        return int(float(text))
    except (TypeError, ValueError, OverflowError):
        return None


def task_settings_update(data: Mapping[str, object], event: str) -> dict[str, str]:
    """Decode task-information events into display-ready field updates."""

    updates: dict[str, str] = {}
    if event in ("TASK_INFO", "NOGO_STAGE_INFO"):
        stage = parse_int_field(data.get("count"))
        duration_us = parse_int_field(data.get("duration_us"))
        go_pct = parse_int_field(data.get("value"))
        if stage is not None:
            updates["stage"] = str(stage)
        if duration_us is not None:
            field = "hold_ms" if event == "TASK_INFO" else "nogo_ms"
            updates[field] = str(int(round(duration_us / 1000.0)))
        if event == "TASK_INFO":
            go_us = parse_int_field(data.get("latency_us"))
            if go_us is not None:
                updates["go_ms"] = str(int(round(go_us / 1000.0)))
        if go_pct is not None:
            updates["go_pct"] = str(go_pct)
        updates["reason"] = str(data.get("reason", "")).strip() or "--"
    elif event in ("SIDE_SET", "TRIAL_START"):
        side = str(data.get("side_uc", "")).strip().upper()
        if side not in ("L", "R"):
            side = str(data.get("side", "")).strip().upper()
        if side in ("L", "R"):
            updates["side"] = side
    return updates
