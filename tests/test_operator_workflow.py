"""Qt → Unix supervisor → real backend → child capture → finalizer → saved weights.

The camera and hardware preflight are fixtures; lifecycle, IPC, persistence and
the independent finalizer are production code. No devices are opened.
"""
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from squeakview.apps.operator.backend import manager, supervision
from squeakview.apps.operator.backend.supervisor.server import SupervisorServer
from squeakview.common import run_context
from squeakview.common.run_layout import PRIMARY_FILES, artifact_path
from squeakview.common.analysis_timing import summarize_recording_timing
from squeakview.project import AppPaths, RuntimeContext, UserPaths, create_project
from tests.support.virtual_controller import VirtualControllerPort


class OperatorWorkflowTests(unittest.TestCase):
    def test_real_operator_start_stop_and_measurement_save(self):
        self.exercise('native')

    def test_real_operator_with_fragmented_v2_controller_and_durable_journal(self):
        self.exercise('v2-free')

    def exercise(self, mode):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = AppPaths.discover()
            project = create_project(root / 'project', name='Software workflow', app=app)
            context = RuntimeContext(app=app, project=project, user=UserPaths(config=root/'config', state=root/'state', runtime=root/'runtime', projects_parent=root/'projects'))
            server = SupervisorServer(root/'operator.sock', runtime_context=context)
            finalize = server.backend._run_capture_finalizer
            def delayed_finalizer(run):
                deadline = time.monotonic() + 15
                while not (root/'release-validation').exists():
                    if time.monotonic() >= deadline:
                        raise TimeoutError('GUI could not enter weights during validation')
                    time.sleep(.01)
                return finalize(run)
            report = root / 'result.json'
            ports = []
            def open_port(*_args, **_kwargs):
                port = VirtualControllerPort()
                ports.append(port)
                return port
            def spawn(config, emit, on_exit=None, **_kwargs):
                return supervision.spawn('tests.support.capture_worker', [str(config.run_dir), mode], emit, 'fixture-camera',
                    workspace=app.root, on_exit=on_exit, parent_death_signal=signal.SIGINT)
            preflight = manager.preflight.PreflightResult(True, 'Software fixture preflight',
                '[PASS] DeepStream new nvstreammux VIC/NVMM path works\n'
                '[PASS] Jetson H.264 full-decode validation path works (decoded 1 frame(s) with h264_nvv4l2dec)\n')
            timer = threading.Timer(40, server.request_shutdown)
            with mock.patch.object(manager.preflight, 'run_preflight', return_value=preflight), \
                 mock.patch.object(server.backend, '_run_capture_finalizer', side_effect=delayed_finalizer), \
                 mock.patch.object(manager.process, 'spawn_inference', side_effect=spawn), \
                 mock.patch('squeakview.common.serial.serial.Serial', side_effect=open_port), \
                 mock.patch.object(manager.clock_validation, 'host_time_status', return_value={'ntp_synchronized': True, 'timezone': 'UTC'}), \
                 mock.patch.dict(os.environ, {'QT_QPA_PLATFORM': 'offscreen', 'SQUEAKVIEW_DISABLE_PREVIEW': '1',
                                             'XDG_CONFIG_HOME': str(root/'xdg-config'), 'XDG_STATE_HOME': str(root/'xdg-state')}):
                timer.start()
                try:
                    code = server.serve([sys.executable, '-m', 'tests.support.gui_workflow', str(report), mode])
                finally:
                    timer.cancel()
                    server.request_shutdown()
            self.assertTrue(report.exists(), f'GUI never reported: {server.last_error}')
            result = json.loads(report.read_text())
            self.assertTrue(result['passed'], result.get('error'))
            self.assertEqual(code, 0)
            run = Path(result['run_dir'])
            self.assertEqual(set(p.name for p in run.iterdir()), set(PRIMARY_FILES) | {'_metadata'})
            self.assertEqual(run_context.bottle_setup(run)['mouse_weight_g'], 26)
            next_run = Path(result['next_run_dir'])
            self.assertEqual(set(p.name for p in next_run.iterdir()), set(PRIMARY_FILES) | {'_metadata'})
            self.assertEqual(run_context.bottle_setup(next_run)['mouse_weight_g'], 27)
            derived = root / 'analysis'
            derived.mkdir()
            self.assertTrue(summarize_recording_timing(run, derived)['validation']['passed'])
            if mode == 'v2-free':
                from data_viz.v2_alignment import load_v2_run
                loaded = load_v2_run(run)
                self.assertEqual(len(loaded.frames), 3)
                self.assertIn('DRINK_START', set(loaded.events.eventType))
                self.assertEqual(len(ports), 2)
                for port in ports:
                    self.assertTrue(any(command.startswith('ACK_EVENTS,') for command in port.commands))
                    self.assertIn('START_BEHAVIOR', port.commands)
                    self.assertIn('STOP', port.commands)
