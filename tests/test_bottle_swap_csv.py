from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from data_viz.bottle_history import load_bottle_history
from squeakview.apps.operator.backend.bottle_swap import BottleSwapService
from squeakview.common.bottle_csv import load_swap_state
from squeakview.common.run_layout import initialize_layout, artifact_path
from squeakview.common.run_outputs import write_weigh_in, WEIGH_FIELDS
from tests.test_bottle_swap import Controller


class BottleSwapCsvTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        initialize_layout(self.root)
        self.initial = {side: {'fluid': 'water', 'initial_weight_g': 100, 'note': 'initial'}
                        for side in ('left', 'right')}
        self.service = BottleSwapService(self.root, self.initial)
        self.controller = Controller()
        self.replacement = {'fluid':'sucrose', 'initial_weight_g':90, 'final_weight_g':80, 'note':'new bottle'}

    def rows(self):
        with (self.root / 'weigh_in.csv').open(newline='') as handle:
            return list(csv.DictReader(handle))

    def swap(self, sides='left'):
        self.service.begin(sides, self.controller)
        return self.service.resume({side:self.replacement for side in self.service.snapshot()['selected_sides']}, self.controller)

    def test_complete_swap_uses_only_weigh_in_and_has_timestamped_boundaries(self):
        right_id = self.service.snapshot()['current']['right']['bottle_id']
        snapshot = self.swap()
        rows = self.rows()
        self.assertEqual(set(row['record_type'] for row in rows),
                         {'bottle_state','swap_state','swap_event','swap_gap'})
        self.assertTrue(all(row['host_unix_ns'] and row['host_monotonic_ns'] for row in rows))
        events = [row for row in rows if row['record_type']=='swap_event']
        self.assertTrue({'pause_requested','resume_requested','controller_state'} <= {row['event'] for row in events})
        confirmed = [row for row in events if row['event']=='controller_state']
        self.assertTrue(all(row['controller_time_method']=='direct_boundary' for row in confirmed))
        latest_gap = [row for row in rows if row['record_type']=='swap_gap'][-1]
        self.assertEqual(int(latest_gap['paused_monotonic_us']), snapshot['gaps'][0]['paused_monotonic_us'])
        self.assertEqual(int(latest_gap['resumed_monotonic_us']), snapshot['gaps'][0]['resumed_monotonic_us'])
        self.assertEqual(latest_gap['pause_left_count'], '12')
        self.assertEqual(latest_gap['resume_right_count'], '23')
        self.assertEqual(latest_gap['controller_time_method'], 'direct_boundary')
        self.assertEqual(int(latest_gap['controller_time_us']),snapshot['gaps'][0]['resumed_monotonic_us'])
        closed = [row for row in rows if row['record_type']=='bottle_state' and row['phase']=='closed'][-1]
        self.assertEqual(closed['fluid'],'water')
        self.assertEqual(closed['initial_weight_g'],'100')
        self.assertEqual(float(closed['final_weight_g']),80)
        self.assertEqual(float(closed['intake_g']),20)
        self.assertEqual(closed['lick_count'],'12')
        self.assertEqual(snapshot['current']['right']['bottle_id'], right_id)
        self.assertEqual(snapshot['current']['left']['note'],'new bottle')
        self.assertFalse(artifact_path(self.root,'bottle_history.json').exists())
        self.assertFalse(artifact_path(self.root,'bottle_history_events.jsonl').exists())
        restored_bottle = load_swap_state(self.root)['history'][0]
        self.assertEqual(restored_bottle['bottle_id'], closed['bottle_id'])
        self.assertEqual(restored_bottle['ended_monotonic_us'], int(closed['ended_monotonic_us']))

    def test_recovery_and_analysis_work_from_csv_without_history_sidecars(self):
        snapshot = self.swap('both')
        recovered = BottleSwapService(self.root)
        self.assertEqual(recovered.snapshot()['current'], snapshot['current'])
        self.assertEqual(recovered.snapshot()['history'], snapshot['history'])
        self.assertFalse(recovered.snapshot()['available'])
        history = load_bottle_history(self.root)
        self.assertEqual(history.bottle_for_event('left', snapshot['gaps'][0]['paused_monotonic_us']-1)['fluid'],'water')
        self.assertIsNone(history.bottle_for_event('right', snapshot['gaps'][0]['paused_monotonic_us']))
        self.assertEqual(history.bottle_for_event('left',snapshot['gaps'][0]['resumed_monotonic_us'])['fluid'],'sucrose')
        self.assertEqual(load_bottle_history(self.root/'weigh_in.csv'), history)

    def test_measurement_saves_preserve_swap_records_and_corrections(self):
        before = self.swap()
        write_weigh_in(self.root, {**before['current'], 'mouse_weight_g':25})
        measurements = [row for row in self.rows() if row['record_type']=='measurement']
        self.assertTrue(any(row['measurement_type']=='mouse' for row in measurements))
        old = before['history'][0]
        after = self.service.update_final_weight(old['bottle_id'],75)
        write_weigh_in(self.root, {**after['current'], 'mouse_weight_g':25})
        rows = self.rows()
        self.assertTrue(any(row['event']=='final_weight_corrected' and float(row['old_weight_g'])==80 for row in rows))
        corrected = load_swap_state(self.root)['history'][0]
        self.assertEqual(corrected['final_weight_g'],75)
        self.assertEqual(corrected['ended_monotonic_us'],old['ended_monotonic_us'])
        self.assertTrue(all(row in rows for row in measurements))
        self.assertEqual(len([row for row in rows if row['measurement_type']=='mouse']),1)

    def test_timeout_pending_replacement_and_interrupted_stop_survive_csv_recovery(self):
        self.service.begin('left',self.controller)
        self.controller.fail='resume'
        self.service.resume({'left':self.replacement},self.controller)
        recovered = BottleSwapService(self.root)
        self.assertEqual(recovered.snapshot()['state'],'UNCONFIRMED')
        self.assertTrue(recovered.snapshot()['resume_pending'])
        recovered.finalize()
        state = load_swap_state(self.root)
        self.assertTrue(state['unresolved_swap'])
        self.assertEqual(state['interrupted_state'],'UNCONFIRMED')
        self.assertEqual(state['current']['left']['final_weight_g'],80)
        events = [row['event'] for row in self.rows()]
        self.assertIn('controller_command_failed',events)
        self.assertIn('swap_interrupted_by_stop',events)

    def test_no_duplicate_state_rows_for_unchanged_save_or_resume(self):
        self.swap()
        before = self.rows()
        self.service._save()
        self.service.resume({'left':self.replacement},self.controller)
        self.assertEqual(self.rows(),before)
        with (self.root/'weigh_in.csv').open(newline='') as handle:
            self.assertEqual(next(csv.reader(handle)),WEIGH_FIELDS)

    def test_multiple_swaps_preserve_history_order_and_unchanged_bottles(self):
        self.swap('right')
        self.swap('left')
        self.swap('both')
        expected = self.service._data['history']
        self.assertEqual(load_swap_state(self.root)['history'],expected)
        self.assertEqual(len(expected),4)
        self.assertEqual(len(load_swap_state(self.root)['gaps']),3)

    def test_manifest_history_and_audit_point_to_weigh_in(self):
        from squeakview.apps.operator.backend.manifest import RunManifestService
        self.swap()
        inventory = RunManifestService(lambda _:None).bottle_snapshot(self.root)
        self.assertEqual(inventory['history'],'weigh_in.csv')
        self.assertEqual(inventory['history_events'],'weigh_in.csv')

    def test_old_new_layout_audit_is_imported_once_with_original_time(self):
        # Simulate an existing new-layout run from before swap consolidation.
        state = self.service._data
        (self.root/'weigh_in.csv').unlink()
        artifact_path(self.root,'bottle_history.json').write_text(json.dumps(state))
        record = {'event':'pause_requested','recorded_at':'2026-10-26T12:00:00.123456+00:00',
                  'swap_id':'old','selected_sides':['left']}
        artifact_path(self.root,'bottle_history_events.jsonl').write_text(json.dumps(record)+'\n')
        BottleSwapService(self.root)
        BottleSwapService(self.root)
        imported = [row for row in self.rows() if row['event']=='pause_requested']
        self.assertEqual(len(imported),1)
        self.assertEqual(int(imported[0]['host_unix_ns']) % 1000000000,123456000)
        self.assertEqual(imported[0]['host_monotonic_ns'],'')

    def test_failed_csv_commit_keeps_previous_recoverable_state(self):
        from copy import deepcopy
        from unittest.mock import patch
        from squeakview.common.bottle_csv import save_swap_state
        before = load_swap_state(self.root)
        changed = deepcopy(before)
        changed['state'] = 'PAUSING'
        with patch('squeakview.common.run_outputs.os.replace',side_effect=OSError('disk failure')):
            with self.assertRaisesRegex(OSError,'disk failure'):
                save_swap_state(self.root,changed)
        self.assertEqual(load_swap_state(self.root),before)
        self.assertFalse(list(self.root.glob('.weigh_in.csv.*')))

if __name__=='__main__':
    unittest.main()
