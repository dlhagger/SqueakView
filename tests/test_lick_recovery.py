"""Recovery gaps must remain missing observations after CSV reopen/analysis."""
import csv
import tempfile
import unittest
from pathlib import Path

from squeakview.common.lick_recovery import load_lick_recovery_gaps, observation_valid
from squeakview.common.sensor_csv import make_event_record
from squeakview.common.serial import SERIAL_HEADER
from squeakview.common.run_layout import initialize_layout
from squeakview.apps.operator.backend.bottle_swap import BottleSwapService
from data_viz.bottle_history import load_bottle_history


class LickRecoveryGapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        initialize_layout(self.root, compact_metadata=True)
        self.rows = []

    def event(self, kind, stamp, duration, *, identity=1, boot=42, session=7):
        for side in ('LD', 'RD'):
            line = f'LICK_RECOVERY_{kind},12345,{stamp},{side},21,{duration},1,{identity},task,READY'
            row = make_event_record(line.split(','), line, host_unix_ns=1, host_monotonic_ns=2)
            row.update(controller_boot_id=boot, controller_session_id=session)
            self.rows.append(row)
        with (self.root / 'sensors.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=SERIAL_HEADER)
            writer.writeheader()
            writer.writerows(self.rows)

    def test_ready_retry_and_duplicate_status_preserve_exact_shared_gap(self):
        self.event('START', 1000, 0)
        self.event('RETRY', 2000, 1000)
        self.event('READY', 20000, 19000)
        self.event('READY', 20000, 19000)
        gaps = load_lick_recovery_gaps(self.root)
        self.assertEqual(len(gaps), 1)
        self.assertTrue(observation_valid(gaps, 999))
        self.assertFalse(observation_valid(gaps, 1000))
        self.assertFalse(observation_valid(gaps, 19999))
        self.assertTrue(observation_valid(gaps, 20000))

    def test_failure_stays_open_until_manual_ready_or_stop(self):
        self.event('FAILED', 11000, 10000)
        gaps = load_lick_recovery_gaps(self.root)
        self.assertIsNone(gaps[0]['resumed_monotonic_us'])
        self.assertFalse(observation_valid(gaps, 999999))
        self.event('STOPPED', 20000, 19000)
        self.assertEqual(load_lick_recovery_gaps(self.root)[0]['resumed_monotonic_us'], 20000)

    def test_history_reopen_excludes_recovery_without_replacing_bottles(self):
        service = BottleSwapService(self.root, {side: {'fluid': 'water', 'initial_weight_g': 30} for side in ('left', 'right')})
        ids = {side: bottle['bottle_id'] for side, bottle in service.snapshot()['current'].items()}
        self.event('START', 1000, 0)
        self.event('FAILED', 11000, 10000)
        self.event('READY', 20000, 19000)
        history = load_bottle_history(self.root)
        self.assertFalse(history.lick_observation_complete)
        for side in ids:
            self.assertIsNone(history.bottle_for_event(side, 5000))
            self.assertEqual(history.bottle_for_event(side, 20000)['bottle_id'], ids[side])

    def test_conflicting_intervals_and_multiple_clocks_fail_closed(self):
        self.event('START', 1000, 0)
        self.event('READY', 20000, 18000)
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            load_lick_recovery_gaps(self.root)
        self.rows = []
        self.event('START', 1000, 0)
        self.event('START', 1000, 0, boot=99)
        with self.assertRaisesRegex(ValueError, 'multiple controller clocks'):
            load_lick_recovery_gaps(self.root)

    def test_idle_status_and_legacy_missing_sensor_file_do_not_create_gaps(self):
        self.assertEqual(load_lick_recovery_gaps(self.root), ())
        self.event('READY', 1000, 0, identity=0)
        self.assertEqual(load_lick_recovery_gaps(self.root), ())
