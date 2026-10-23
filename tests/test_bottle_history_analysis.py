from pathlib import Path
import tempfile
import unittest

import pandas as pd

from data_viz.bottle_history import annotate_bottle_events, load_bottle_history
from squeakview.apps.operator.backend.bottle_swap import BottleSwapService


class BottleHistoryAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.service = BottleSwapService(self.path, {s: {'fluid': 'water', 'initial_weight_g': 100} for s in ('left', 'right')})
        def controller(action, swap_id):
            return {'supported': True, 'swap_id': swap_id,
                    'state': 'PAUSED' if action == 'pause' else 'ACTIVE',
                    'left_count': 10, 'right_count': 20,
                    'monotonic_us': 100 if action == 'pause' else 200}
        self.service.begin('left', controller)
        self.service.resume({'left': {'fluid': 'sucrose', 'initial_weight_g': 90, 'final_weight_g': 80}},
                            lambda action, swap_id: {**controller(action, swap_id), 'state': 'PAUSED' if action == 'status' else 'ACTIVE'})
        self.history = load_bottle_history(self.path)

    def test_maps_changed_bottle_and_preserves_unchanged_identity(self):
        old = self.history.bottle_for_event('left', 90, count=10)
        new = self.history.bottle_for_event('left', 210, count=11)
        self.assertEqual(old['fluid'], 'water')
        self.assertEqual(new['fluid'], 'sucrose')
        self.assertNotEqual(old['bottle_id'], new['bottle_id'])
        self.assertEqual(self.history.bottle_for_event('right', 90)['bottle_id'],
                         self.history.bottle_for_event('right', 210)['bottle_id'])
        self.assertIsNone(self.history.bottle_for_event('left', 210, count=10))

    def test_shared_gap_is_excluded_on_both_sides_with_exact_boundaries(self):
        for side in ('left', 'right'):
            self.assertIsNotNone(self.history.bottle_for_event(side, 99))
            self.assertIsNone(self.history.bottle_for_event(side, 100))
            self.assertIsNone(self.history.bottle_for_event(side, 199))
            self.assertIsNotNone(self.history.bottle_for_event(side, 200))

    def test_dataframe_annotations_leave_nonlick_events_unassigned(self):
        frame = pd.DataFrame([
            {'eventType': 'DRINK_START', 'side': 'LD', 'rp2040Time': 90, 'count': 10},
            {'eventType': 'DRINK_START', 'side': 'RD', 'rp2040Time': 150, 'count': 20},
            {'eventType': 'DRINK_START', 'side': 'LD', 'rp2040Time': 210, 'count': 11},
            {'eventType': 'FEED_START', 'side': '', 'rp2040Time': 160},
        ])
        annotated = annotate_bottle_events(frame, self.history)
        self.assertEqual(annotated.iloc[0]['bottle_fluid'], 'water')
        self.assertFalse(annotated.iloc[1]['lick_observation_valid'])
        self.assertEqual(annotated.iloc[2]['bottle_fluid'], 'sucrose')
        self.assertTrue(pd.isna(annotated.iloc[3]['lick_observation_valid']))
        self.assertNotIn('bottle_id', frame)

    def test_legacy_missing_history_is_explicit(self):
        self.assertIsNone(load_bottle_history(self.path / 'missing.json'))


if __name__ == '__main__':
    unittest.main()
