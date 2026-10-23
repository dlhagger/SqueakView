from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from squeakview.apps.operator.backend.bottle_swap import BottleSwapService


class Controller:
    def __init__(self):
        self.state = 'ACTIVE'
        self.swap_id = ''
        self.counts = {'left': 12, 'right': 23}
        self.fail = None
        self.timestamp = 100
        self.supported = True

    def __call__(self, action, swap_id):
        self.timestamp += 100
        if self.fail == action:
            raise TimeoutError('controller timeout')
        if action == 'pause':
            self.state, self.swap_id = 'PAUSED', swap_id
        elif action == 'resume':
            self.state = 'ACTIVE'
        return {'supported': self.supported, 'state': self.state,
                'swap_id': self.swap_id, 'monotonic_us': self.timestamp,
                **{f'{s}_count': n for s, n in self.counts.items()}}


class BottleSwapTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.bottles = {s: {'fluid': 'water', 'initial_weight_g': 100} for s in ('left', 'right')}
        self.service = BottleSwapService(self.path, self.bottles)
        self.controller = Controller()
        self.new = {'fluid': 'sucrose', 'initial_weight_g': 90, 'final_weight_g': 80}

    def test_left_swap_preserves_right_identity_and_session_totals(self):
        right = self.service.snapshot()['current']['right']
        paused = self.service.begin('left', self.controller)
        self.assertEqual(paused['state'], 'PAUSED')
        self.assertFalse(paused['history'])
        result = self.service.resume({'left': self.new}, self.controller)
        self.assertEqual(result['global_counts'], {'left': 12, 'right': 23})
        self.assertEqual(result['current']['left']['lick_count'], 0)
        self.assertEqual(result['current']['right']['bottle_id'], right['bottle_id'])
        self.assertEqual(result['current']['right']['lick_count'], 23)
        self.assertEqual(result['history'][0]['lick_count'], 12)
        self.assertEqual(result['history'][0]['intake_g'], 20)
        self.assertIsNotNone(result['gaps'][0]['resumed_monotonic_us'])
        self.service.observe_lick('left', 15)
        self.service.observe_lick('right', 25)
        current = self.service.snapshot()['current']
        self.assertEqual(current['left']['lick_count'], 3)
        self.assertEqual(current['right']['lick_count'], 25)

    def test_repeated_swaps_and_duplicate_resume_do_not_duplicate_bottles(self):
        for side in ('left', 'right', 'both'):
            self.service.begin(side, self.controller)
            selected = self.service.snapshot()['selected_sides']
            self.service.resume({s: self.new for s in selected}, self.controller)
            previous = self.service.snapshot()
            self.service.resume({s: self.new for s in selected}, self.controller)
            self.assertEqual(self.service.snapshot(), previous)
            for s in ('left', 'right'):
                self.controller.counts[s] += 10
        self.assertEqual(len(self.service.snapshot()['history']), 4)

    def test_resume_timeout_recovers_from_persisted_metadata(self):
        self.service.begin('left', self.controller)
        self.controller.fail = 'resume'
        result = self.service.resume({'left': self.new}, self.controller)
        self.assertEqual(result['state'], 'UNCONFIRMED')
        self.assertEqual(result['history'], [])
        recovered = BottleSwapService(self.path)
        self.assertFalse(recovered.snapshot()['available'])
        # The command reached hardware but its reply was lost.
        self.controller.fail = None
        self.controller.state = 'ACTIVE'
        result = recovered.observe_controller(self.controller('status', ''))
        self.assertEqual(result['state'], 'ACTIVE')
        self.assertEqual(result['current']['left']['fluid'], 'sucrose')
        self.assertEqual(len(result['history']), 1)
        recovered.observe_controller(self.controller('status', ''))
        self.assertEqual(len(recovered.snapshot()['history']), 1)

    def test_failed_calibration_retry_preserves_pending_replacement(self):
        self.service.begin('both', self.controller)
        self.controller.fail = 'resume'
        self.service.resume({'left': self.new, 'right': self.new}, self.controller)
        self.controller.fail = None
        result = self.service.resume({'left': self.new, 'right': self.new}, self.controller)
        self.assertEqual(result['state'], 'ACTIVE')
        self.assertEqual(len(result['history']), 2)

    def test_pause_timeout_and_missing_support_cannot_publish_success(self):
        self.controller.fail = 'pause'
        self.assertEqual(self.service.begin('left', self.controller)['state'], 'UNCONFIRMED')
        self.assertEqual(self.service.snapshot()['history'], [])
        fresh = BottleSwapService(self.path / 'other')
        self.controller.fail = None
        self.controller.supported = False
        result = fresh.begin('left', self.controller)
        self.assertFalse(result['available'])
        self.assertIsNone(result['swap_id'])

    def test_missing_final_allowed_but_replacement_validated(self):
        self.service.begin('left', self.controller)
        for replacement in ({'fluid': '', 'initial_weight_g': 20},
                            {'fluid': 'water', 'initial_weight_g': float('nan')},
                            {'fluid': 'water', 'initial_weight_g': -1},
                            {'fluid': 'water'}):
            with self.assertRaises(ValueError):
                self.service.resume({'left': replacement}, self.controller)
        result = self.service.resume({'left': {'fluid': 'water', 'initial_weight_g': 20}}, self.controller)
        self.assertFalse(result['history'][0]['complete'])

    def test_counter_replay_and_wrong_swap_cannot_reset_or_resume(self):
        self.service.begin('right', self.controller)
        self.service.observe_lick('left', 4)
        self.service.observe_controller({'state': 'ACTIVE', 'swap_id': 'wrong', 'left_count': 13})
        result = self.service.snapshot()
        self.assertEqual(result['state'], 'PAUSED')
        self.assertEqual(result['global_counts']['left'], 13)

    def test_current_measurement_save_and_finalize_preserve_closed_history(self):
        self.service.begin('left', self.controller)
        self.service.resume({'left': self.new}, self.controller)
        closed = deepcopy(self.service.snapshot()['history'])
        self.service.sync_measurements({'left': {'final_weight_g': 70}})
        result = self.service.finalize()
        self.assertEqual(result['history'], closed)
        self.assertEqual(result['current']['left']['intake_g'], 20)
        self.assertEqual(result['state'], 'STOPPED')
        with self.assertRaises(ValueError):
            self.service.begin('both', self.controller)

    def test_exact_boundaries_survive_newer_cached_lick_counts(self):
        self.service.begin('left', self.controller)
        self.service.observe_lick('left', 17)
        result = self.service.resume({'left': self.new}, self.controller)
        self.assertEqual(result['history'][0]['lick_count'], 12)
        self.assertEqual(result['current']['left']['lick_count'], 5)
        self.assertEqual(result['gaps'][0]['resume_counts']['left'], 12)

    def test_failed_hardware_calibration_can_be_retried(self):
        self.service.begin('left', self.controller)
        self.controller.fail = 'resume'
        self.service.resume({'left': self.new}, self.controller)
        self.controller.fail = None
        self.controller.state = 'FAILED'
        result = self.service.resume({'left': self.new}, self.controller)
        self.assertEqual(result['state'], 'ACTIVE')
        self.assertEqual(len(result['history']), 1)

    def test_historical_weight_correction_preserves_provenance_and_audits(self):
        self.service.begin('left', self.controller)
        self.service.resume({'left': self.new}, self.controller)
        old = self.service.snapshot()['history'][0]
        result = self.service.update_final_weight(old['bottle_id'], 75)
        updated = result['history'][0]
        self.assertEqual(updated['intake_g'], 25)
        for key in ('start_count', 'end_count', 'started_monotonic_us', 'ended_monotonic_us'):
            self.assertEqual(old[key], updated[key])
        audit = (self.path / 'bottle_history_events.jsonl').read_text()
        self.assertIn('final_weight_corrected', audit)
        self.assertEqual(self.service.history_page(), result['history'])
        with self.assertRaises(ValueError):
            self.service.update_final_weight('missing', 80)

    def test_status_timeout_revokes_availability_and_blocks_pause(self):
        self.service.observe_controller(self.controller('status', ''))
        self.controller.fail = 'status'
        result = self.service.begin('both', self.controller)
        self.assertFalse(result['available'])
        self.assertIsNone(result['swap_id'])

    def test_stopping_failed_resume_preserves_old_weight_and_unresolved_intent(self):
        old_id = self.service.snapshot()['current']['left']['bottle_id']
        self.service.begin('left', self.controller)
        self.controller.fail = 'resume'
        self.service.resume({'left': self.new}, self.controller)
        snapshot = self.service.finalize()
        self.assertTrue(snapshot['unresolved_swap'])
        self.assertEqual(snapshot['interrupted_state'], 'UNCONFIRMED')
        self.assertTrue(snapshot['resume_pending'])
        self.assertEqual(snapshot['current']['left']['bottle_id'], old_id)
        self.assertEqual(snapshot['current']['left']['final_weight_g'], 80)
        self.assertEqual(snapshot['history'], [])
        self.assertIsNone(snapshot['gaps'][0]['resumed_monotonic_us'])
        recovered = BottleSwapService(self.path).snapshot()
        self.assertTrue(recovered['unresolved_swap'])
        self.assertEqual(recovered['state'], 'STOPPED')

    def test_worst_case_unicode_snapshot_and_fifty_entry_page_fit_ipc(self):
        import json
        from squeakview.apps.operator.backend.supervisor.protocol import EventEnvelope, encode_envelope
        maximum = (1 << 63) - 1
        for entry in self.service._data['current'].values():
            entry.update(fluid='🧪' * 32, start_count=maximum, end_count=maximum,
                         started_monotonic_us=maximum, ended_monotonic_us=maximum,
                         initial_weight_g=1e308, final_weight_g=1.7e308,
                         measurement_updated_at='2026-10-23T12:34:45.123456+00:00')
        self.service._data['history'] = [deepcopy(self.service._data['current']['left']) for _ in range(50)]
        self.service._data['gaps'] = [dict(swap_id='a' * 32, selected_sides=['left', 'right'],
            paused_monotonic_us=maximum, resumed_monotonic_us=maximum,
            pause_counts=dict(left=maximum, right=maximum),
            resume_counts=dict(left=maximum, right=maximum)) for _ in range(50)]
        self.service._data['error'] = '🧪' * 1024
        snapshot = self.service.snapshot()
        self.assertLess(len(json.dumps(snapshot, ensure_ascii=False).encode()), 64 * 1024)
        page = self.service.history_page(0, 50)
        self.assertEqual(len(page), 50)
        envelope = EventEnvelope(name='bottle_history_page', sequence=1, payload={'history': page})
        self.assertLess(len(encode_envelope(envelope)), 64 * 1024)
        self.assertEqual(len(self.service._data['history']), 50)
        self.assertEqual(len(self.service._data['gaps']), 50)

    def test_fluid_length_is_bounded_in_utf8_bytes(self):
        self.service.begin('left', self.controller)
        with self.assertRaisesRegex(ValueError, 'UTF-8'):
            self.service.resume({'left': {**self.new, 'fluid': '🧪' * 33}}, self.controller)

    def test_pending_replacement_reconciles_active_after_coalesced_retry_states(self):
        for previous_state in ('FAILED', 'PAUSED'):
            with self.subTest(previous_state=previous_state):
                service = BottleSwapService(self.path / previous_state, self.bottles)
                controller = Controller()
                service.begin('left', controller)
                right_before = service.snapshot()['current']['right']
                old_left_id = service.snapshot()['current']['left']['bottle_id']
                controller.fail = 'resume'
                service.resume({'left': self.new}, controller)
                controller.fail = None
                controller.state = previous_state
                observed = service.observe_controller(controller('status', ''))
                self.assertEqual(observed['state'], previous_state)
                self.assertTrue(observed['resume_pending'])
                # An unrelated transaction cannot commit even when ACTIVE.
                service.observe_controller({**controller('status', ''),
                                            'state': 'ACTIVE', 'swap_id': 'f' * 32})
                self.assertEqual(service.snapshot()['state'], previous_state)
                self.assertEqual(service.snapshot()['history'], [])
                # Newer starts may already be cached when the ACTIVE boundary
                # is reconciled; keep its frozen counter as the bottle origin.
                service.observe_lick('left', 15)
                service.observe_lick('right', 25)
                controller.state = 'ACTIVE'
                active = controller('status', '')
                result = service.observe_controller(active)
                self.assertEqual(result['state'], 'ACTIVE')
                self.assertEqual(result['error'], '')
                self.assertFalse(result['resume_pending'])
                self.assertEqual(result['global_counts'], {'left': 15, 'right': 25})
                self.assertEqual(result['history'][0]['bottle_id'], old_left_id)
                self.assertEqual(result['history'][0]['lick_count'], 12)
                self.assertEqual(result['current']['left']['lick_count'], 3)
                self.assertEqual(result['current']['right']['bottle_id'], right_before['bottle_id'])
                self.assertEqual(result['current']['right']['initial_weight_g'], 100)
                self.assertEqual(result['current']['right']['fluid'], 'water')
                self.assertEqual(result['current']['right']['lick_count'], 25)
                self.assertEqual(len(result['history']), 1)
                left_after = result['current']['left']['bottle_id']
                duplicate = service.observe_controller(active)
                self.assertEqual(duplicate['state'], 'ACTIVE')
                self.assertEqual(duplicate['current']['left']['bottle_id'], left_after)
                self.assertEqual(len(duplicate['history']), 1)


if __name__ == '__main__':
    unittest.main()
