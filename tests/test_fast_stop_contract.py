"""Stop work budgets, immutable primary rows, explicit archival, and day-length timing."""
import collections
import csv
import gzip
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from squeakview.apps.inference import post_run
from squeakview.common import run_context, run_outputs, metadata_retention, v2_timing
from squeakview.common.run_layout import artifact_path, PRIMARY_FILES
from squeakview.common.row_timing import iter_timestamped_rows
from tests.support.recordings import create_recording
from tests.test_mp4_sample_table import _fixture, _box


class FastStopContractTests(unittest.TestCase):
    def args(self, root, **extra):
        return SimpleNamespace(run_dir=root, camera_count=1, enable_infer=False,
            align=False, full_analysis=False, materialize_timestamps=False,
            compact_metadata=extra.get('compact', False))

    def test_day_length_counts_do_not_scan_rewrite_or_compress_primary_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = create_recording(Path(directory), 'native')
            count = 24 * 60 * 60 * 30
            # Represent the source-owned final counters without generating
            # millions of rows. Stop must only inspect the header and tail.
            for name, field in [('capture_cam0.jsonl', 'source_sequence_index')]:
                path = artifact_path(root, name)
                row = json.loads(path.read_text().splitlines()[-1])
                row[field] = count - 1
                path.write_text(json.dumps(row) + '\n')
            artifact_path(root, 'record_admission.csv').write_text(f'stream_id,record_frame_index,pts_ns\n0,{count-1},100\n')
            with (root/'frames.csv').open() as stream:
                reader = csv.DictReader(stream)
                header, row = reader.fieldnames, next(reader)
            row.update(source_sequence_index=count-1, raw_frame_index=count-1)
            with (root/'frames.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=header)
                writer.writeheader()
                writer.writerow(row)
            movie = _fixture(samples=count)
            ftyp_end = struct.unpack('>I', movie[:4])[0]
            mdat_end = ftyp_end + struct.unpack('>I', movie[ftyp_end:ftyp_end+4])[0]
            (root/'raw.mp4').write_bytes(movie[:ftyp_end] + _box(b'mdat', b'\0'*count) + movie[mdat_end:])
            watched = {root/name for name in PRIMARY_FILES if name != 'raw.mp4'}
            watched.update(artifact_path(root, name) for name in ('capture_cam0.jsonl', 'record_admission.csv'))
            original_open = Path.open
            reads = collections.Counter()
            class BudgetReader:
                def __init__(self, handle, path): self.handle, self.path = handle, path
                def __enter__(self): return self
                def __exit__(self, *args): return self.handle.__exit__(*args)
                def __getattr__(self, name): return getattr(self.handle, name)
                def read(self, size=-1):
                    if size < 0: raise AssertionError('Stop attempted a full-file read')
                    result = self.handle.read(size)
                    reads[self.path] += len(result)
                    return result
                def readline(self, size=-1):
                    if size < 0: raise AssertionError('Stop attempted an unbounded row read')
                    result = self.handle.readline(size)
                    reads[self.path] += len(result)
                    return result
                def __iter__(self): raise AssertionError('Stop attempted CSV iteration')
            def opened(path, mode='r', *args, **kwargs):
                if path in watched:
                    if mode != 'rb': raise AssertionError(f'Stop attempted primary mutation/scan: {path.name}: {mode}')
                    return BudgetReader(original_open(path, mode, *args, **kwargs), path)
                return original_open(path, mode, *args, **kwargs)
            before = {p: (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in watched}
            with mock.patch.object(post_run, 'parse_args', return_value=self.args(root)), \
                 mock.patch.object(Path, 'open', opened), \
                 mock.patch.object(run_outputs.sqlite3, 'connect', side_effect=AssertionError('Stop opened a timing index')), \
                 mock.patch.object(run_outputs, 'finalize_primary_tables', side_effect=AssertionError('Stop rewrote rows')), \
                 mock.patch.object(metadata_retention, 'compact_successful_run', side_effect=AssertionError('Stop compressed ledgers')):
                self.assertEqual(post_run.main(), 0)
            self.assertEqual(before, {p: (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns) for p in watched})
            self.assertLess(sum(reads.values()), 400_000)
            self.assertEqual(run_context.read_json(run_context.status_path(root))['post_run_frames'], count)

    def test_default_stop_and_read_time_controller_mapping_preserve_all_source_bytes(self):
        for mode in ('native', 'v2-free', 'v2-triggered'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = create_recording(Path(directory), mode)
                before = {name: (root/name).read_bytes() for name in PRIMARY_FILES}
                with mock.patch.object(post_run, 'parse_args', return_value=self.args(root)):
                    self.assertEqual(post_run.main(), 0)
                rows = list(iter_timestamped_rows(root, 'frames.csv'))
                self.assertEqual(len(rows), 3)
                expected = 'unavailable' if mode == 'native' else 'host_clock_estimate' if mode == 'v2-free' else 'piecewise_interpolated'
                self.assertEqual(rows[1]['controller_time_method'], expected)
                self.assertEqual(before, {name: (root/name).read_bytes() for name in PRIMARY_FILES})

    def test_explicit_compaction_can_be_rechecked_without_decompression(self):
        with tempfile.TemporaryDirectory() as directory:
            root = create_recording(Path(directory), 'native')
            with mock.patch.object(post_run, 'parse_args', return_value=self.args(root, compact=True)):
                self.assertEqual(post_run.main(), 0)
            self.assertTrue(artifact_path(root, 'capture_cam0.jsonl').suffix == '.gz')
            with mock.patch.object(post_run, 'parse_args', return_value=self.args(root)), \
                 mock.patch.object(gzip._GzipReader, 'read', side_effect=AssertionError('Stop decompressed a recovery ledger')):
                self.assertEqual(post_run.main(), 0)
            archived = artifact_path(root, 'capture_cam0.jsonl')
            with archived.open('ab') as stream: stream.write(b'changed')
            with mock.patch.object(post_run, 'parse_args', return_value=self.args(root)):
                self.assertEqual(post_run.main(), 1)

    def test_archive_failure_does_not_relabel_good_capture_as_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = create_recording(Path(directory), 'native')
            with mock.patch.object(post_run, 'parse_args', return_value=self.args(root, compact=True)), \
                 mock.patch.object(metadata_retention, '_compress', side_effect=OSError('archive unavailable')):
                self.assertEqual(post_run.main(), 4)
            status = run_context.read_json(run_context.status_path(root))
            self.assertTrue(status['overall_validation_passed'])
            self.assertEqual(status['metadata_compaction'], 'failed')

    def test_24_hour_checkpoint_stream_fits_supported_timing_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'diagnostics').mkdir()
            (root/'diagnostics/controller_v2.jsonl').write_text('verified-record projection fixture')
            count = 86402
            (root/'diagnostics/controller_v2_summary.json').write_text(json.dumps(dict(protocol='mousehouse_v2', boot_id=1, session_id=7, counts={'frames_stored': count})))
            def records(_path):
                for index in range(count):
                    kind = 'CAMERA_EPOCH' if index == 0 else 'CAMERA_STOP' if index == count-1 else 'CAMERA_CHECKPOINT'
                    yield dict(sequence=index+1, boot_id=1, session_id=7,
                        message_type={'CAMERA_EPOCH':3,'CAMERA_CHECKPOINT':4,'CAMERA_STOP':5}[kind],
                        host_unix_ns=index*10**9, host_monotonic_ns=index*10**9,
                        payload_utf8=f'{kind},count={1+30*index},timestamp_us={1000+index*1000000},period_us=33333,pulse_us=1000,health=0x3,queue=0/64,suppressed=0,reason=Periodic')
            with mock.patch.object(v2_timing, 'journal_records', side_effect=records):
                timing = v2_timing.read_v2_timing(root, triggered=True)
            self.assertEqual(len(timing.anchors), count)
            self.assertEqual(v2_timing.camera_time(timing.anchors[-2:], timing.anchors[-1]['count'])[1], 'exact_anchor')
