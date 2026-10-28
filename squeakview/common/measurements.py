"""Shared numeric contract for operator and backend weigh-in inputs."""
from __future__ import annotations

import math
from collections.abc import Mapping

MAX_WEIGHT_G = 100_000.0


def weight_value(value, label: str, *, positive: bool = False) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        if isinstance(value, bool):
            raise ValueError
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label} must be a number.") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number.")
    if number < 0:
        raise ValueError(f"{label} cannot be negative.")
    if positive and number == 0:
        raise ValueError(f"{label} must be greater than zero.")
    if number > MAX_WEIGHT_G:
        raise ValueError(f"{label} cannot exceed {MAX_WEIGHT_G:g} g.")
    return round(number, 6)


def validate_bottle_inputs(bottles) -> None:
    if bottles is None:
        return
    if not isinstance(bottles, Mapping):
        raise ValueError("Measurements must be an object.")
    if "mouse_weight_g" in bottles:
        weight_value(bottles["mouse_weight_g"], "Mouse weight", positive=True)
    for side in ("left", "right"):
        if side not in bottles:
            continue
        values = bottles[side]
        if not isinstance(values, Mapping):
            raise ValueError(f"{side.title()} bottle measurements must be an object.")
        for phase in ("initial", "final"):
            if f"{phase}_weight_g" in values:
                weight_value(values[f"{phase}_weight_g"], f"{side.title()} {phase} weight")
