from __future__ import annotations

"""Qt-free plot and count-summary presentation for the live dashboard."""

from collections.abc import Mapping, Sequence
from html import escape


_SERIES_LABELS = {
    "POKE_L": "Left poke",
    "POKE_R": "Right poke",
    "DRINK_L": "Left drink",
    "DRINK_R": "Right drink",
    "PELLET": "Pellet",
    "WELL_CHECK": "Well check",
    "WELL": "Well check",
    "GO_CORRECT": "Go correct",
    "GO_INCORRECT": "Go incorrect",
    "NOGO_CORRECT": "NoGo correct",
    "NOGO_INCORRECT": "NoGo incorrect",
}

_SERIES_COLORS = {
    "POKE_L": "#37d67a",
    "POKE_R": "#6fa8ff",
    "DRINK_L": "#a29bfe",
    "DRINK_R": "#ff7eb6",
    "PELLET": "#f5a623",
    "WELL_CHECK": "#50e3c2",
    "WELL": "#50e3c2",
    "GO_CORRECT": "#2ecc71",
    "GO_INCORRECT": "#e74c3c",
    "NOGO_CORRECT": "#4aa3df",
    "NOGO_INCORRECT": "#f39c12",
}

_FALLBACK_PALETTE = (
    "#ffd166",
    "#06d6a0",
    "#118ab2",
    "#ef476f",
    "#9b5de5",
    "#f15bb5",
)
MAX_SERIES_POINTS = 20_000


def series_label(key: str) -> str:
    return _SERIES_LABELS.get(key, key.replace("_", " ").title())


def series_color(key: str, index: int) -> str:
    return _SERIES_COLORS.get(key, _FALLBACK_PALETTE[index % len(_FALLBACK_PALETTE)])


def window_bounds(now: float, window_sec: float) -> tuple[float, float]:
    half = float(window_sec) / 2.0
    return now - half, now + half


def trim_step_series(
    xs: list[float],
    ys: list[int],
    *,
    xstart: float,
) -> None:
    """Bound a mutable step series while retaining its value at the left edge."""

    discard = 0
    while len(xs) - discard >= 2 and xs[discard + 1] < xstart:
        discard += 2
    if discard:
        del xs[:discard]
        del ys[:discard]
    if xs and xs[0] < xstart:
        xs[0] = xstart


def cap_step_series(
    xs: list[float],
    ys: list[int],
    *,
    max_points: int = MAX_SERIES_POINTS,
) -> None:
    """Apply a hard display-only history cap while keeping x/y pairs aligned."""

    limit = max(2, int(max_points))
    excess = max(0, len(xs) - limit)
    # Step histories are appended as two-point transitions. Drop complete
    # transitions so the retained first point still has a matching value.
    discard = excess + (excess % 2)
    if discard:
        del xs[:discard]
        del ys[:discard]


def curve_points(
    xs: Sequence[float],
    ys: Sequence[int],
    *,
    now: float,
) -> tuple[list[float], list[int]]:
    """Return detached plot points with the last step extended to the present."""

    if not xs or not ys:
        return [], []
    plot_x = list(xs)
    plot_y = list(ys)
    if plot_x[-1] < now:
        plot_x.append(now)
        plot_y.append(plot_y[-1])
    return plot_x, plot_y


def counts_html(
    plot_series: Sequence[Sequence[str]],
    counters: Mapping[str, int],
) -> str:
    parts: list[str] = []
    series_index = 0
    for keys in plot_series:
        group: list[str] = []
        for key in keys:
            label = escape(series_label(key).upper())
            color = series_color(key, series_index)
            series_index += 1
            group.append(
                "<span style='white-space:nowrap;'>"
                f"<span style='color:#9aa7cc; font-weight:600;'>{label}</span> "
                f"<span style='color:{color}; font-weight:800;'>{counters.get(key, 0)}</span>"
                "</span>"
            )
        if group:
            parts.append("&nbsp;&nbsp;".join(group))
    heading = "<span style='color:#eef1ff; font-weight:800;'>EVENT COUNTS</span>"
    if not parts:
        return heading + "&nbsp;&nbsp;--"
    separator = "&nbsp;&nbsp;<span style='color:#46506d;'>|</span>&nbsp;&nbsp;"
    return heading + "&nbsp;&nbsp;&nbsp;" + separator.join(parts)


__all__ = [
    "MAX_SERIES_POINTS",
    "cap_step_series",
    "counts_html",
    "curve_points",
    "series_color",
    "series_label",
    "trim_step_series",
    "window_bounds",
]
