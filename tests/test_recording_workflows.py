"""Real finalizer CLI and analysis across the supported recording modes."""
import csv
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.support.recordings import create_recording, MODES
from squeakview.common import run_context
from squeakview.common.run_layout import artifact_path, PRIMARY_FILES
from squeakview.apps.inference.mp4_sample_table import read_video_sample_table
from data_viz.v2_alignment import load_v2_run


class RecordingWorkflowTests(unittest.TestCase):
    def test_recovery_gaps_survive_finalization_and_v2_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = create_recording(Path(directory), 'v2-triggered', recovery=True)
            result = subprocess.run([sys.executable, '-m', 'squeakview.apps.inference.post_run', str(root), '--align'], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            loaded = load_v2_run(root)
            self.assertFalse(loaded.lick_observation_complete)
            self.assertEqual(len(loaded.lick_observation_gaps), 1)
            gap = loaded.lick_observation_gaps[0]
            self.assertEqual((gap['start_time_s'], gap['end_time_s']), (.0005, .049))
            drink = loaded.events[loaded.events.eventType == 'DRINK_START']
            self.assertEqual(list(drink.lick_observation_valid), [False])

    def test_finalize_save_clear_reopen_and_analyse_each_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in MODES:
                with self.subTest(mode=mode):
                    root = create_recording(Path(directory) / mode, mode)
                    before = hashlib.sha256((root / 'raw.mp4').read_bytes()).hexdigest()
                    self.assertEqual(read_video_sample_table(root / 'raw.mp4').sample_count, 3)
                    completed = subprocess.run([sys.executable, '-m', 'squeakview.apps.inference.post_run', str(root), '--align'], capture_output=True, text=True, timeout=15)
                    self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                    self.assertEqual(set(p.name for p in root.iterdir()), set(PRIMARY_FILES) | {'_metadata'})
                    self.assertTrue(run_context.read_json(artifact_path(root, 'run_status.json'))['overall_validation_passed'])
                    self.assertEqual(before, hashlib.sha256((root / 'raw.mp4').read_bytes()).hexdigest())
                    run_context.write_bottle_artifacts(root, {'mouse_weight_g': None})
                    self.assertIsNone(run_context.bottle_setup(root)['mouse_weight_g'])
                    run_context.write_bottle_artifacts(root, {'mouse_weight_g': 26})
                    self.assertEqual(run_context.bottle_setup(root)['mouse_weight_g'], 26)
                    if mode.startswith('v2'):
                        loaded = load_v2_run(root)
                        self.assertEqual(len(loaded.frames), 3)
                        expected = {'exact_anchor', 'piecewise_interpolated'} if mode.endswith('triggered') else {'host_clock_estimate'}
                        self.assertEqual(set(loaded.frames.controller_time_method), expected)


class OfflineCompletionWorkflowTests(unittest.TestCase):
    def test_successful_replay_finalizes_each_mode_and_keeps_sources_immutable(self):
        from types import SimpleNamespace
        from unittest import mock
        from squeakview.apps.inference import offline
        from squeakview.apps.inference.pose_persistence import PoseCsvWriter
        from squeakview.common.run_outputs import finalize_primary_tables
        from squeakview.common.analysis_timing import summarize_recording_timing

        class EndMessage:
            pass

        with tempfile.TemporaryDirectory() as directory:
            for mode in MODES:
                with self.subTest(mode=mode):
                    root = create_recording(Path(directory) / mode, mode)
                    finalize_primary_tables(root, 1)
                    config = Path(directory) / 'fixture.txt'
                    parser = Path(directory) / 'parser.so'
                    config.write_text('fixture model')
                    parser.write_bytes(b'fixture parser identity')
                    model = SimpleNamespace(name='fixture', parser_library=parser, manifest_snapshot=lambda: {})
                    output = Path(directory) / (mode + '-results')
                    with mock.patch.object(offline, 'validate_model_package', return_value=model):
                        app = offline.OfflineInferenceApp(offline.OfflineConfig(root, cfg_path=config, out_dir=output))
                    writer = PoseCsvWriter(output)
                    app.observations = writer
                    class ReplayFrames:
                        def start(self, callback):
                            for index in range(3):
                                app.audit.handle_metadata(SimpleNamespace(frame_items=[SimpleNamespace(frame_number=index)]))
                            callback(EndMessage())
                        def wait(self):
                            pass
                        def stop(self):
                            raise AssertionError('clean EOS must not be stopped twice')
                    app.pipeline = ReplayFrames()
                    original = app._current_source_hashes()
                    with mock.patch.object(offline, 'EOSMessage', EndMessage):
                        self.assertEqual(app.run(), 0)
                    self.assertEqual(app._current_source_hashes(), original)
                    self.assertTrue(app.ledger._closed)
                    manifest = run_context.read_json(output / 'offline_manifest.json')
                    self.assertEqual(manifest['status'], 'complete')
                    self.assertEqual(manifest['decoded_frames'], 3)
                    validation = run_context.read_json(output / 'alignment_summary.json')['validation']
                    if mode == 'legacy-triggered':
                        self.assertTrue(validation['video_frame_count_matches_frames_csv'])
                        self.assertEqual(validation['objects_missing_frame_count'], 0)
                    else:
                        self.assertTrue(validation['passed'])

    def test_real_sdk_start_failure_exits_without_an_external_stop(self):
        program = '''
import threading
from types import SimpleNamespace
from pyservicemaker import Pipeline
from squeakview.apps.inference.offline import OfflineInferenceApp
app = object.__new__(OfflineInferenceApp)
app.pipeline = Pipeline('software-test-missing-file')
app.pipeline.add('filesrc','source',{'location':'/this-software-test-file-does-not-exist.h264'})
app.pipeline.add('fakesink','sink',{'sync':False,'async':False})
app.pipeline.link('source','sink')
app.observations = None
app.stop_event = threading.Event()
app.exit_code = 0
app.eos_received = False
app.audit = SimpleNamespace(count=0)
app.ledger = [1]
errors=[]
app._write_manifest=lambda *args:errors.append(args)
assert app._run_impl() == 1
assert errors and 'without decoder EOS' in str(errors)
print('SDK_START_FAILURE_HANDLED')
'''
        result = subprocess.run([sys.executable, '-c', program], text=True, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('SDK_START_FAILURE_HANDLED', result.stdout)
