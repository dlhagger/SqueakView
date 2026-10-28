"""Exercise bottle identities through the actual operator orchestration seams."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest import mock

from squeakview.apps.operator.backend.bottle_swap import BottleSwapService
from squeakview.apps.operator.backend.events import RunPhase
from squeakview.apps.operator.backend.manager import OperatorBackend
from squeakview.common import run_context
from squeakview.project import AppPaths, Project, ProjectMetadata, ProjectPaths, RuntimeContext, UserPaths


class FakeSerial:
    def __init__(self):
        self.counts = {'left': 7, 'right': 11}
        self.controller = None
        self.state = 'ACTIVE'
        self.swap_id = ''
        self.timestamp = 100
        self.post_resume_licks = 0

    @property
    def bottle_control_snapshot(self):
        return {'controller': deepcopy(self.controller), 'counts': dict(self.counts)}

    def exchange_bottle_control(self, action, swap_id, **_kwargs):
        self.timestamp += 100
        if action == 'pause':
            self.state, self.swap_id = 'PAUSED', swap_id
        elif action == 'resume':
            self.state = 'ACTIVE'
        self.controller = {
            'supported': True, 'state': self.state, 'swap_id': self.swap_id,
            'left_count': self.counts['left'], 'right_count': self.counts['right'],
            'monotonic_us': self.timestamp, 'reason': '',
        }
        response = deepcopy(self.controller)
        # Simulate DRINK_START already journaled when resume waiter wakes.
        if action == 'resume':
            self.counts['left'] += self.post_resume_licks
        return response


class BottleSwapRuntimeTests(unittest.TestCase):
    def test_recovering_sensors_cannot_create_a_swap_transaction(self):
        self.backend._bottle_swap.observe_lick_recovery({"state": "RECOVERING", "available": False})
        self.backend._observe_bottle_runtime()
        before = self.backend._bottle_swap.snapshot()
        with self.assertRaisesRegex(ValueError, "recovering"):
            self.backend.begin_bottle_swap(["left"])
        after = self.backend._bottle_swap.snapshot()
        self.assertEqual(after["swap_id"], before["swap_id"])
        self.assertEqual(after["current"], before["current"])

    def test_recovery_is_persisted_separately_from_bottle_identity(self):
        before = self.backend._bottle_swap.snapshot()
        recovery = {"state": "FAILED", "available": False, "recovery_id": 1, "reason": "READINESS_TIMEOUT"}
        self.backend._bottle_swap.observe_lick_recovery(recovery)
        after = self.backend._bottle_swap.snapshot()
        self.assertEqual(after["lick_recovery"], recovery)
        self.assertEqual(after["state"], before["state"])
        self.assertEqual(after["current"], before["current"])
        restored = BottleSwapService(self.run_dir, self.initial).snapshot()
        self.assertEqual(restored["lick_recovery"], recovery)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        project_root = root / 'project'
        for path in (project_root, project_root / 'runs', root / 'app'):
            path.mkdir(exist_ok=True)
        context = RuntimeContext(
            app=AppPaths.from_root(root / 'app'),
            project=Project(paths=ProjectPaths.from_existing_root(project_root), metadata=ProjectMetadata.create('Test')),
            user=UserPaths(config=root / 'config', state=root / 'state', runtime=root / 'runtime', projects_parent=root / 'projects'),
        )
        self.backend = OperatorBackend(lambda _line: None, runtime_context=context)
        self.run_dir = project_root / 'runs' / 'run-1'
        self.run_dir.mkdir()
        self.backend.state.run_dir = self.run_dir
        self.serial = FakeSerial()
        self.backend.state.serial = self.serial
        self.backend._controller_started = True
        self.initial = {'mouse_weight_g': 25, **{s: {'fluid': 'water', 'initial_weight_g': 100} for s in ('left', 'right')}}
        run_context.write_bottle_artifacts(self.run_dir, self.initial)
        self.backend._bottle_swap = BottleSwapService(self.run_dir, self.initial)
        for phase in (RunPhase.CREATED, RunPhase.STARTING, RunPhase.RECORDING):
            self.backend._state_machine.transition(phase)
        # Manifest construction depends on launched camera runtime. Keep its
        # real bottle snapshot and artifact writes while replacing launch data.
        self.manifest_write = mock.patch.object(self.backend, '_write_run_manifest', return_value=True)
        self.manifest_write.start()
        self.addCleanup(self.manifest_write.stop)

    def swap_left(self):
        self.backend.begin_bottle_swap(['left'])
        return self.backend.resume_bottle_swap({'left': {'fluid': 'sucrose', 'initial_weight_g': 90, 'final_weight_g': 80}})

    def test_ordinary_mouse_save_cannot_overwrite_replacement_bottle(self):
        self.swap_left()
        stale_form = deepcopy(self.initial)
        stale_form['mouse_weight_g'] = 26
        stale_form['left']['final_weight_g'] = 79
        summary = self.backend.save_bottle_measurements(stale_form)
        current = self.backend.bottle_swap_snapshot()['current']
        self.assertEqual(current['left']['fluid'], 'sucrose')
        self.assertEqual(current['left']['initial_weight_g'], 90)
        self.assertIsNone(current['left']['final_weight_g'])
        self.assertEqual(summary['mouse_weight_g'], 26)
        self.assertEqual(summary['sides']['left']['fluid'], 'sucrose')
        self.assertEqual(self.backend.bottle_swap_snapshot()['history'][0]['final_weight_g'], 80)

    def test_cached_post_resume_licks_are_assigned_to_new_bottle(self):
        self.serial.post_resume_licks = 3
        snapshot = self.swap_left()
        self.assertEqual(snapshot['global_counts']['left'], 10)
        self.assertEqual(snapshot['current']['left']['lick_count'], 3)
        self.assertEqual(snapshot['history'][0]['lick_count'], 7)
        self.assertEqual(snapshot['current']['right']['lick_count'], 11)
        self.assertEqual(snapshot['gaps'][0]['resume_counts']['left'], 7)

    def test_late_resume_confirmation_updates_compatibility_artifacts(self):
        self.backend.begin_bottle_swap(['left'])
        original = self.serial.exchange_bottle_control
        def pending(action, swap_id, **kwargs):
            if action == 'resume':
                self.serial.state = 'CALIBRATING'
                self.serial.controller['state'] = 'CALIBRATING'
                raise TimeoutError('late calibration response')
            return original(action, swap_id, **kwargs)
        with mock.patch.object(self.serial, 'exchange_bottle_control', side_effect=pending):
            self.backend.resume_bottle_swap({'left': {
                'fluid': 'sucrose', 'initial_weight_g': 90, 'final_weight_g': 80,
            }})
        self.serial.state = 'ACTIVE'
        self.serial.controller['state'] = 'ACTIVE'
        snapshot = self.backend.bottle_swap_snapshot()
        self.assertEqual(snapshot['state'], 'ACTIVE')
        summary = run_context.read_json(self.run_dir / run_context.BOTTLE_SUMMARY_FILENAME)
        self.assertEqual(summary['sides']['left']['fluid'], 'sucrose')
        self.assertEqual(summary['closed_bottle_count'], 1)

    def test_finalize_persists_final_counters_and_leaves_history_editable(self):
        snapshot = self.swap_left()
        self.serial.counts['left'] += 4
        self.backend._observe_bottle_runtime()
        self.backend._bottle_swap.finalize()
        self.backend._persist_bottle_swap_summary()
        self.backend._state_machine._phase = RunPhase.FINALIZED
        # Delayed old serial observations cannot reactivate finalized history.
        snapshot = self.backend.bottle_swap_snapshot()
        self.assertEqual(snapshot['state'], 'STOPPED')
        self.assertFalse(snapshot['available'])
        self.assertEqual(snapshot['current']['left']['lick_count'], 4)
        updated = self.backend.update_bottle_history_weight(snapshot['history'][0]['bottle_id'], 78)
        self.assertEqual(updated['history'][0]['intake_g'], 22)
        disk = run_context.read_json(self.run_dir / 'bottle_history.json')
        self.assertEqual(disk['state'], 'STOPPED')
        self.assertEqual(disk['global_counts']['left'], 11)
        self.assertTrue((self.run_dir / 'bottle_history_events.jsonl').exists())

    def test_stale_final_weight_form_requires_current_bottle_identity(self):
        self.swap_left()
        self.backend._state_machine._phase = RunPhase.FINALIZED
        with self.assertRaisesRegex(ValueError, 'refresh'):
            self.backend.save_bottle_measurements(self.initial)
        snapshot = self.backend.bottle_swap_snapshot()
        payload = {'mouse_weight_g': 26, 'current_bottle_ids': {s: b['bottle_id'] for s, b in snapshot['current'].items()},
                   'left': {'final_weight_g': 70}}
        summary = self.backend.save_bottle_measurements(payload)
        self.assertEqual(summary['sides']['left']['fluid'], 'sucrose')
        self.assertEqual(summary['sides']['left']['final_weight_g'], 70)

    def test_manifest_references_full_history_and_attempt_evidence(self):
        self.swap_left()
        manifest = self.backend._bottle_manifest_snapshot(self.run_dir)
        self.assertEqual(manifest['history'], 'bottle_history.json')
        self.assertEqual(manifest['history_events'], 'bottle_history_events.jsonl')
        self.assertEqual(manifest['sides']['left']['fluid'], 'sucrose')
        outputs = self.backend._run_output_snapshot(self.run_dir)
        self.assertTrue(outputs['bottle_files']['history']['exists'])
        self.assertTrue(outputs['bottle_files']['history_events']['exists'])

    def test_compact_run_keeps_initial_and_final_measurements_across_stop(self):
        import csv
        from squeakview.common.run_layout import initialize_layout
        from squeakview.apps.operator.gui.backend_proxy import launch_config_payload, launch_config_from_payload
        from squeakview.apps.operator.backend.contracts import RunRequest
        initialize_layout(self.run_dir, compact_metadata=True)
        # Exercise the same immutable request and wire round trip as GUI Start.
        config = launch_config_from_payload(launch_config_payload(RunRequest(bottles=self.initial)))
        self.backend._bottle_swap = BottleSwapService(
            self.run_dir, run_context.normalize_bottle_measurements(config.bottles))
        self.backend._write_bottle_measurements(self.run_dir, config.bottles)
        initial = run_context.bottle_summary(self.run_dir)
        self.assertEqual(initial['mouse_weight_g'], 25)
        self.assertEqual(initial['sides']['left']['initial_weight_g'], 100)
        self.assertEqual(initial['sides']['right']['fluid'], 'water')
        self.backend._bottle_swap.finalize()
        self.backend._persist_bottle_swap_summary()
        self.backend._state_machine._phase = RunPhase.FINALIZED
        stopped = run_context.bottle_summary(self.run_dir)
        self.assertEqual(stopped['sides']['left']['initial_weight_g'], 100)
        final = deepcopy(self.initial)
        final['left']['final_weight_g'] = 95
        final['right']['final_weight_g'] = 96
        summary = self.backend.save_bottle_measurements(final)
        self.assertTrue(summary['complete'])
        self.assertEqual(run_context.bottle_summary(self.run_dir)['sides']['left']['intake_g'], 5)
        with (self.run_dir / 'weigh_in.csv').open(newline='') as handle:
            measurements = [row for row in csv.DictReader(handle) if row['record_type'] == 'measurement']
        self.assertEqual(len(measurements), 5)  # Two initial, mouse, two final.
        self.assertTrue(all(row['host_unix_ns'] for row in measurements))
        self.assertFalse((self.run_dir / '_metadata/bottle_setup.json').exists())


if __name__ == '__main__':
    unittest.main()
