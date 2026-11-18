"""Camera-gap policy checks without camera, GPU, or vendor Python bindings."""
import csv
import gzip
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

from squeakview.apps.inference.gap_validation import validate_gaps
from squeakview.apps.inference.mp4_sample_table import read_video_timestamps
from squeakview.apps.operator.backend.capture_drain import wait_for_capture_drain
from squeakview.apps.operator.gui.run_presentation import present_finalization
from squeakview.common.capture_gaps import GapIndex
from squeakview.common.row_timing import RowTiming
from squeakview.common.v2_timing import V2Timing, verify_v2_frame_timing


def box(kind, payload):
    return struct.pack('>I4s', 8 + len(payload), kind) + payload


def timed_video(deltas=(1000, 3000, 1000), *, scale=30000, offsets=None, version=0):
    """Three sample tables; frame 2 follows a two-frame hole at 30 fps."""
    stts = box(b'stts', bytes(4) + struct.pack('>I', len(deltas)) +
               b''.join(struct.pack('>II', 1, delta) for delta in deltas))
    ctts = b'' if offsets is None else box(b'ctts', bytes([version, 0, 0, 0]) +
               struct.pack('>I', len(offsets)) + b''.join(
                   struct.pack('>Ii' if version else '>II', 1, value) for value in offsets))
    mdhd = box(b'mdhd', bytes(12) + struct.pack('>I', scale) + bytes(8))
    hdlr = box(b'hdlr', bytes(8) + b'vide')
    stsz = box(b'stsz', bytes(4) + struct.pack('>II', 1, len(deltas)))
    stsc = box(b'stsc', bytes(4) + struct.pack('>IIII', 1, 1, len(deltas), 1))
    stco = box(b'stco', bytes(4) + struct.pack('>II', 1, 8))
    return box(b'mdat', bytes(len(deltas))) + box(b'moov', box(b'trak', box(b'mdia',
        mdhd + hdlr + box(b'minf', box(b'stbl', stts + ctts + stsz + stsc + stco)))))


def write_gap(root, **changes):
    event = dict(missing_frames=2, frame_index_before=1, frame_index_after=2,
                 pts_before_ns=33_333_333, pts_after_ns=133_333_333,
                 camera_timestamp_before_ns=1_033_333_333,
                 camera_timestamp_after_ns=1_133_333_333)
    event.update(changes)
    directory = root / 'diagnostics'
    directory.mkdir(exist_ok=True)
    with (directory / 'errors.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['event_type', 'stream_id', 'expected_frame_id', 'actual_frame_id', 'details'])
        writer.writerow(['camera_frame_gap', 0, 102, 104, json.dumps(event)])


def write_health(root, **changes):
    health = dict(schema_version=1, camera_index=0, state='closed', source_frames=3,
                  host_unix_ns=1, host_monotonic_ns=1, total_timeouts=0,
                  total_incomplete=0, total_crc_failures=0, total_frame_gap_events=1,
                  total_missing_frames=2, gap_mapping_validated=True, native_timing_validated=True)
    health.update(changes)
    (root / 'capture_health_cam0.json').write_text(json.dumps(health))


class CameraGapPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        write_gap(self.root)
        write_health(self.root)
        (self.root / 'raw.mp4').write_bytes(timed_video())

    def test_accepts_only_located_gaps_preserved_in_video(self):
        report = validate_gaps(self.root, 0, 3)
        self.assertTrue(report['passed'], report)
        self.assertEqual(report['missing_frames'], 2)
        (self.root / 'raw.mp4').write_bytes(timed_video((1000, 1000, 1000)))
        report = validate_gaps(self.root, 0, 3)
        self.assertFalse(report['passed'])
        self.assertIn('MP4 timestamp', report['error'])

    def test_composition_offsets_and_signed_offsets_are_used(self):
        for version, offsets in [(0, (7, 7, 7)), (1, (-7, -7, -7))]:
            with self.subTest(version=version):
                (self.root / 'raw.mp4').write_bytes(timed_video(offsets=offsets, version=version))
                scale, times = read_video_timestamps(self.root / 'raw.mp4', {0, 2})
                self.assertEqual((scale, times), (30000, {0: offsets[0], 2: 4000 + offsets[2]}))
                self.assertTrue(validate_gaps(self.root, 0, 3)['passed'])

    def test_missing_or_inconsistent_evidence_is_not_downgraded(self):
        for changes in [dict(frame_index_after=None), dict(frame_index_after=3),
                        dict(missing_frames=1), dict(pts_after_ns=66_666_666),
                        dict(camera_timestamp_after_ns=1_000_000_000)]:
            with self.subTest(changes=changes):
                write_gap(self.root, **changes)
                self.assertFalse(validate_gaps(self.root, 0, 3)['passed'])
        write_gap(self.root)
        write_health(self.root, gap_mapping_validated=False)
        self.assertFalse(validate_gaps(self.root, 0, 3)['passed'])
        write_health(self.root, total_frame_gap_events=0)
        (self.root / 'diagnostics/errors.csv').unlink()
        self.assertFalse(validate_gaps(self.root, 0, 3)['passed'])

    def test_timestamp_lookup_rejects_truncated_tables_and_out_of_range(self):
        with self.assertRaises(ValueError):
            read_video_timestamps(self.root / 'raw.mp4', {3})
        (self.root / 'raw.mp4').write_bytes(timed_video()[:-1])
        self.assertFalse(validate_gaps(self.root, 0, 3)['passed'])

    def test_gap_index_and_controller_mapping_preserve_holes(self):
        gaps = GapIndex(self.root)
        self.assertEqual([gaps.trigger_offset(0, i) for i in range(4)], [0, 1, 4, 5])
        timing = V2Timing(1, 2, 1, 4, 1000, 134332, (
            dict(count=1, timestamp_us=1000), dict(count=5, timestamp_us=134332)), 4)
        with mock.patch('squeakview.common.row_timing.ClockMapping') as clock:
            clock.return_value.apply.side_effect = dict
            resolver = RowTiming(self.root)
            resolver.timing = timing
            resolver.anchors = timing.anchors
            resolver.counts = (1, 5)
            row = resolver.map(dict(raw_frame_index=2, stream_id=0), frame=True)
        self.assertEqual(row['controller_time_us'], 134332)

    def test_gap_mapping_reads_retained_compressed_evidence(self):
        from squeakview.common.run_layout import initialize_layout
        initialize_layout(self.root, compact_metadata=True)
        source = self.root / 'diagnostics/errors.csv'
        destination = self.root / '_metadata/recovery/diagnostics/errors.csv.gz'
        destination.parent.mkdir(parents=True)
        with gzip.open(destination, 'wb') as output:
            output.write(source.read_bytes())
        source.unlink()
        self.assertEqual(GapIndex(self.root).trigger_offset(0, 2), 4)

    def test_v2_validation_accepts_accounted_holes_and_rejects_shifted_export(self):
        timing = V2Timing(1, 2, 1, 4, 1000, 134332, (
            dict(count=1, timestamp_us=1000), dict(count=5, timestamp_us=134332)), 4)
        rows = [dict(stream_id=0, source_sequence_index=i, raw_frame_index=i,
                     camera_timestamp_ns=1_000_000_000 + ns, missing_frames_before=missing,
                     controller_time_us=us, controller_time_method=method,
                     controller_boot_id=1, controller_session_id=2)
                for i, ns, missing, us, method in [(0, 0, 0, 1000, 'exact_anchor'),
                (1, 33_333_333, 0, 34333, 'piecewise_interpolated'),
                (2, 133_333_333, 2, 134332, 'exact_anchor')]]
        def write():
            with (self.root / 'frames.csv').open('w') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader(); w.writerows(rows)
        with mock.patch('squeakview.common.v2_timing.read_v2_timing', return_value=timing):
            write()
            self.assertTrue(verify_v2_frame_timing(self.root, triggered=True, camera_count=1)['passed'])
            rows[-1]['controller_time_us'] = 67666
            write()
            with self.assertRaisesRegex(ValueError, 'mapping disagrees'):
                verify_v2_frame_timing(self.root, triggered=True, camera_count=1)

    def test_drain_requires_recorded_plus_known_missing_equal_controller(self):
        (self.root / 'frames.csv').write_text('stream_id,camera_serial,deepstream_frame_number,source_sequence_index\n0,camera,2,2\n')
        (self.root / 'record_admission.csv').write_text('stream_id,record_frame_index,pts_ns\n0,2,133333333\n')
        for ttl, passed in [(5, True), (6, False), (3, False)]:
            now = [0.0]
            def sleep(duration):
                now[0] += duration
            result = wait_for_capture_drain(self.root, 1, expected_ttl_count=ttl,
                quiet_s=0.1, timeout_s=0.3, poll_s=0.05, monotonic=lambda: now[0], sleep=sleep)
            self.assertEqual(result.passed, passed, result)

    def test_optional_failures_or_validated_gaps_display_completed_with_warning(self):
        for extra in [dict(recording_warnings=['2 missing frames']), dict(live_services={'state': 'degraded'})]:
            result = present_finalization(dict(state='finalized', recording_validation={'passed': True}, **extra))
            self.assertTrue(result.passed)
            self.assertEqual(result.health_level, 'warning')

    def test_legacy_controller_alignment_accounts_for_only_the_located_triggers(self):
        from squeakview.apps.inference.legacy_alignment import build_alignment
        with (self.root / 'frames.csv').open('w') as f:
            w = csv.writer(f)
            w.writerow(['stream_id', 'raw_frame_index', 'camera_frame_id', 'camera_timestamp_ns', 'pts_ns'])
            for index, identity, ns in [(0, 100, 0), (1, 101, 33_333_333), (2, 104, 133_333_333)]:
                w.writerow([0, index, identity, 1_000_000_000 + ns, ns])
        with (self.root / 'serial.csv').open('w') as f:
            w = csv.writer(f)
            w.writerow(['eventType', 'reason', 'count', 'rp2040Time'])
            w.writerow(['MARKER', 'START_SENT', '', ''])
            for count in range(1, 6):
                w.writerow(['CAMERA_HIGH', '', count, 100_000 + (count - 1) * 33333])
            for name in ['CAPTURE_STOP_REQUESTED', 'STOP_SENT', 'CAPTURE_STOP_DONE']:
                w.writerow(['MARKER', name, '', ''])
        result = build_alignment(self.root, self.root / 'analysis', include_objects=False,
                                 video_validation=dict(count=3, method='mp4_sample_table', error=None))
        self.assertTrue(result['frame_alignment']['validated'], result)
        self.assertEqual(result['frame_alignment']['known_missing_camera_frames'], 2)
        (self.root / 'raw.mp4').write_bytes(timed_video((1000, 1000, 1000)))
        result = build_alignment(self.root, self.root / 'analysis', include_objects=False,
                                 video_validation=dict(count=3, method='mp4_sample_table', error=None))
        self.assertFalse(result['frame_alignment']['validated'])


if __name__ == '__main__':
    unittest.main()
