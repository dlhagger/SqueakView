"""Bounded source health snapshots, with read-only support for old ledgers."""
from __future__ import annotations

import json
from pathlib import Path

from .run_layout import artifact_path
from .run_context import read_json_required
from .ledger_tail import last_complete_line

COUNTERS = ('total_timeouts', 'total_incomplete', 'total_frame_gap_events', 'total_crc_failures')


def health_path(run_dir: Path, stream_id: int = 0) -> Path:
    return artifact_path(Path(run_dir), f'capture_health_cam{stream_id}.json')


def source_path(run_dir: Path, stream_id: int = 0) -> Path:
    """Select the source's durable evidence, including historical recordings."""
    path = health_path(run_dir, stream_id)
    return path if path.exists() else artifact_path(Path(run_dir), f'capture_cam{stream_id}.jsonl')


def read_capture_health(run_dir: Path, stream_id: int = 0) -> dict:
    path = health_path(run_dir, stream_id)
    if not path.exists():
        raw = last_complete_line(source_path(run_dir, stream_id), run_dir=run_dir)
        return json.loads(raw or '{}')
    payload = read_json_required(path, max_bytes=16384)
    if (type(payload.get('schema_version')) is not int or payload['schema_version'] != 1
            or type(payload.get('camera_index')) is not int or payload['camera_index'] != stream_id
            or payload.get('state') not in ('recording', 'closed')
            or type(payload.get('native_timing_validated')) is not bool):
        raise ValueError(f'invalid capture health snapshot: {path.name}')
    for name in ('source_frames', 'host_unix_ns', 'host_monotonic_ns', *COUNTERS):
        value = payload.get(name)
        if type(value) is not int or value < 0:
            raise ValueError(f'invalid capture health {name}: {path.name}')
    # Compatibility for count consumers; this derived field is never stored.
    return dict(payload, source_sequence_index=payload['source_frames'] - 1)
