"""No-JSONL recording lifecycle, recovery, integrity failures and compatibility."""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from squeakview.common import capture_health, run_context, recording_evidence
from squeakview.common.run_layout import artifact_path
from squeakview.apps.inference import post_run
from squeakview.apps.inference.capture_reconciliation import iter_native_capture_payloads
from squeakview.apps.operator.backend.capture_drain import ledger_frame_counts, wait_for_capture_drain
from squeakview.apps.operator.gui.run_presentation import scan_capture_health
from tests.support.recordings import create_recording


class CaptureHealthTests(unittest.TestCase):
    def create(self, directory, mode='native'):
        return create_recording(Path(directory), mode, health=True)

    def update(self, root, **changes):
        path = capture_health.health_path(root)
        value = json.loads(path.read_text())
        value.update(changes)
        path.write_text(json.dumps(value))

    def test_free_and_triggered_stop_keep_primary_clocks_without_capture_jsonl(self):
        for mode in ('native', 'v2-free', 'v2-triggered'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = self.create(directory, mode)
                before = (root/'frames.csv').read_bytes()
                args = SimpleNamespace(run_dir=root, camera_count=1, enable_infer=False, align=False,
                                       full_analysis=False, materialize_timestamps=False, compact_metadata=False)
                with mock.patch.object(post_run, 'parse_args', return_value=args):
                    self.assertEqual(post_run.main(), 0)
                self.assertEqual((root/'frames.csv').read_bytes(), before)
                self.assertFalse(artifact_path(root, 'capture_cam0.jsonl').exists())
                self.assertLess(capture_health.health_path(root).stat().st_size, 1024)
                self.assertEqual(ledger_frame_counts(root, 1), (3, 3))
                self.assertTrue(wait_for_capture_drain(root, 1, expected_ttl_count=3,
                    quiet_s=.001, timeout_s=.2, poll_s=.005).passed)
                self.assertEqual(scan_capture_health(root).frames, 3)

    def test_day_length_health_count_keeps_stop_reads_bounded(self):
        import struct
        from tests.test_mp4_sample_table import _fixture, _box
        with tempfile.TemporaryDirectory() as directory:
            root = self.create(directory)
            count = 24 * 60 * 60 * 30
            self.update(root, source_frames=count)
            path = root/'frames.csv'
            with path.open() as f:
                reader = csv.DictReader(f); header = reader.fieldnames; row = next(reader)
            row.update(source_sequence_index=count-1, raw_frame_index=count-1)
            with path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header); writer.writeheader(); writer.writerow(row)
            artifact_path(root, 'record_admission.csv').write_text(
                f'stream_id,record_frame_index,pts_ns\n0,{count-1},100\n')
            movie = _fixture(samples=count)
            ftyp_end = struct.unpack('>I', movie[:4])[0]
            mdat_end = ftyp_end + struct.unpack('>I', movie[ftyp_end:ftyp_end+4])[0]
            (root/'raw.mp4').write_bytes(movie[:ftyp_end] + _box(b'mdat', b'\0'*count) + movie[mdat_end:])
            before = path.read_bytes()
            # Explicit full analysis and primary-row writers must remain unused.
            with mock.patch.object(post_run, 'finalize_run', side_effect=AssertionError('full scan')), \
                 mock.patch.object(post_run, '_write_payload', side_effect=AssertionError('row rewrite')):
                result = post_run.fast_finalize_run(root, camera_count=1)
            self.assertTrue(result.validation_passed)
            self.assertEqual(result.recorded_total, count)
            self.assertEqual(before, path.read_bytes())
            self.assertFalse(artifact_path(root, 'capture_cam0.jsonl').exists())
            self.assertLess(capture_health.health_path(root).stat().st_size, 1024)

    def test_health_faults_missing_or_invalid_summary_fail_closed(self):
        for changes in ({'source_frames': 4}, {'total_incomplete': 1}, {'total_crc_failures': 1},
                        {'total_frame_gap_events': 1}, {'native_timing_validated': False},
                        {'state': 'recording'}, {'camera_index': 1}, {'source_frames': True},
                        {'total_timeouts': -1}, {'source_frames': None}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                root = self.create(directory)
                self.update(root, **changes)
                try:
                    result = post_run.fast_finalize_run(root, camera_count=1)
                except (ValueError, RuntimeError):
                    continue
                self.assertFalse(result.validation_passed)
        with tempfile.TemporaryDirectory() as directory:
            root = self.create(directory)
            capture_health.health_path(root).unlink()
            with self.assertRaises(RuntimeError):
                post_run.fast_finalize_run(root, camera_count=1)

    def test_drain_uses_latest_csv_even_when_health_checkpoint_lags(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.create(directory)
            self.update(root, source_frames=1, state='recording')
            self.assertEqual(ledger_frame_counts(root, 1), (3, 3))

    def test_full_analysis_audits_csv_without_rewriting_it_or_camera_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.create(directory)
            before = (root/'frames.csv').read_bytes()
            runtime = artifact_path(root, 'diagnostics/camera_runtime.json')
            runtime.parent.mkdir(parents=True, exist_ok=True)
            runtime.write_text('{"retained": true}')
            probe = dict(count=3, method='fixture_decode', error=None)
            with mock.patch.object(post_run, 'probe_video_frames', return_value=probe), \
                 mock.patch.object(post_run, '_validate_acquisition_integrity', return_value=({'passed': True}, True)):
                result = post_run.finalize_run(root, camera_count=1, enable_infer=False)
            self.assertTrue(result.validation_passed)
            self.assertEqual(before, (root/'frames.csv').read_bytes())
            self.assertEqual(runtime.read_text(), '{"retained": true}')
            self.assertTrue(recording_evidence.recording_evidence_complete(
                recording_evidence.capture_recording_evidence(root, 1), 1))

    def test_evidence_covers_both_frames_and_health_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.create(directory)
            for path in (root/'frames.csv', capture_health.health_path(root)):
                before = path.read_bytes()
                evidence = recording_evidence.capture_recording_evidence(root, 1)
                self.assertTrue(recording_evidence.recording_evidence_complete(evidence, 1))
                path.write_bytes(before + b' ')
                self.assertFalse(recording_evidence.recording_evidence_matches(root, evidence, 1))
                path.write_bytes(before)

    def test_explicit_csv_audit_rejects_bad_native_clock_or_identity(self):
        for field, bad_value in [('camera_timestamp_ns', '0'), ('camera_frame_id', '999'),
                                 ('host_received_unix_ns', ''), ('payload_crc_valid', '0')]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = self.create(directory)
                path = root/'frames.csv'
                with path.open() as f:
                    reader = csv.DictReader(f); header = reader.fieldnames; rows = list(reader)
                rows[1][field] = bad_value
                with path.open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=header); writer.writeheader(); writer.writerows(rows)
                with self.assertRaises(RuntimeError):
                    list(iter_native_capture_payloads(root, 0))

    def test_historical_jsonl_remains_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = create_recording(Path(directory), 'native')
            self.assertEqual(capture_health.read_capture_health(root)['source_sequence_index'], 2)
