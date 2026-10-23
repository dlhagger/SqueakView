"""Associate protocol-v2 lick events with installed bottles and observation gaps.

Timestamps must be controller monotonic microseconds (``rp2040Time`` in the v2
loader), never host receipt times. Legacy runs without history return ``None``.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BottleHistory:
    bottles: tuple[dict[str, Any], ...]
    gaps: tuple[dict[str, Any], ...]

    def observation_valid(self, monotonic_us: int) -> bool:
        stamp = int(monotonic_us)
        for gap in self.gaps:
            start, end = gap['paused_monotonic_us'], gap['resumed_monotonic_us']
            if start is not None and stamp >= start and (end is None or stamp < end):
                return False
        return True

    def bottle_for_event(self, side: str, monotonic_us: int, *, count: int | None = None):
        """Return the installed bottle, excluding the shared pause/calibration gap.

        For lick starts, optional cumulative ``count`` also checks controller
        counter boundaries. New bottle starts are exclusive in count space:
        its first lick has ``count == start_count + 1``.
        """
        side = {'LD': 'left', 'RD': 'right'}.get(side, side)
        if side not in ('left', 'right'):
            raise ValueError('Unknown lick side')
        stamp = int(monotonic_us)
        if not self.observation_valid(stamp):
            return None
        for bottle in self.bottles:
            if bottle['side'] != side:
                continue
            start, end = bottle['started_monotonic_us'], bottle['ended_monotonic_us']
            if (start is not None and stamp < start) or (end is not None and stamp >= end):
                continue
            if count is not None and (count <= bottle['start_count'] or (
                    bottle['end_count'] is not None and count > bottle['end_count'])):
                continue
            return deepcopy(bottle)
        return None


def load_bottle_history(run_dir: str | Path) -> BottleHistory | None:
    path = Path(run_dir)
    if path.is_dir():
        path /= 'bottle_history.json'
    if not path.exists():
        return None
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('Bottle history is unexpectedly large')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('schema_version') != 1:
        raise ValueError('Unsupported bottle history schema')
    bottles = [*payload['history'], *payload['current'].values()]
    for bottle in bottles:
        end = bottle['end_count']
        if end is None:
            end = payload['global_counts'][bottle['side']]
        bottle['lick_count'] = max(0, end - bottle['start_count'])
        initial, final = bottle['initial_weight_g'], bottle['final_weight_g']
        bottle['intake_g'] = None if initial is None or final is None else initial - final
    return BottleHistory(tuple(bottles), tuple(payload['gaps']))


def annotate_bottle_events(events, history: BottleHistory):
    """Return a copy of the v2 events DataFrame with bottle and gap columns.

    Unrelated sensor events retain empty assignments. An event in a shared
    gap has ``lick_observation_valid=False`` on either side; the gap is never
    interpreted as an interval of zero licking.
    """
    import pandas as pd

    result = events.copy()
    assignments, fluids, valid = [], [], []
    for row in events.to_dict('records'):
        side, stamp = row.get('side'), row.get('rp2040Time')
        if side not in ('LD', 'RD', 'left', 'right') or pd.isna(stamp):
            assignments.append(None)
            fluids.append(None)
            valid.append(None)
            continue
        count = row.get('count') if row.get('eventType') == 'DRINK_START' else None
        if count is not None and pd.isna(count):
            count = None
        bottle = history.bottle_for_event(side, int(stamp), count=None if count is None else int(count))
        assignments.append(bottle['bottle_id'] if bottle else None)
        fluids.append(bottle['fluid'] if bottle else None)
        valid.append(history.observation_valid(int(stamp)))
    result['bottle_id'] = assignments
    result['bottle_fluid'] = fluids
    result['lick_observation_valid'] = pd.array(valid, dtype='boolean')
    return result
