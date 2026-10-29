"""Read-only controller mappings for native primary CSVs and historical exports."""
import csv
from pathlib import Path

from .bounded_csv import bounded_csv_lines
from .run_layout import artifact_path
from .run_outputs import ClockMapping
from . import run_context
from .v2_timing import read_v2_timing, camera_time


class RowTiming:
    """Load synchronization evidence once, then map any number of CSV rows."""
    def __init__(self, run_dir):
        self.clock = ClockMapping(run_dir)
        path = artifact_path(run_dir, 'run_manifest.json')
        manifest = run_context.read_json_required(path) if path.exists() else {}
        serial = manifest.get('serial', {})
        self.timing = None
        if serial.get('enabled') and serial.get('controller_protocol') == 'v2':
            self.timing = read_v2_timing(run_dir, triggered=manifest.get('capture', {}).get('trigger_on', True))
        self.anchors = () if self.timing is None else tuple({row['count']: row for row in self.timing.anchors}.values())
        self.counts = tuple(row['count'] for row in self.anchors)

    def map(self, row, *, frame=False):
        original = dict(row)
        row = self.clock.apply(dict(row))
        if frame and self.anchors:
            raw = original.get('raw_frame_index', original.get('source_sequence_index'))
            if isinstance(raw, bool) or not str(raw).isascii() or not str(raw).isdecimal():
                raise ValueError('frame row has no valid source sequence')
            value, method = camera_time(self.anchors, self.anchors[0]['count'] + int(raw), counts=self.counts)
            row.update(controller_time_us=value, controller_time_method=method,
                       controller_boot_id=self.timing.boot_id, controller_session_id=self.timing.session_id)
            for name in ('controller_time_us', 'controller_time_method', 'controller_boot_id', 'controller_session_id'):
                if name in original and str(original[name]) != str(row[name]):
                    raise ValueError(f'exported {name} disagrees with synchronization evidence')
        return row


def iter_timestamped_rows(run_dir, artifact):
    """Stream timestamped rows without rewriting scientific source files.

    Camera/controller interpolation remains labelled; native timestamps are
    retained. Native-only recordings expose unavailable controller mappings.
    """
    root = Path(run_dir)
    resolver = RowTiming(root)
    frame = artifact in ('frames.csv', 'detections.csv', 'objects.csv') or str(artifact).startswith('frames_cam')
    for row in csv.DictReader(bounded_csv_lines(artifact_path(root, artifact))):
        yield resolver.map(row, frame=frame)
