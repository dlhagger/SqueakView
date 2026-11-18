from __future__ import annotations

import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from squeakview.common.shutdown_progress import ProgressDeadline, wait_for_completion
from squeakview.common import shutdown_policy
from squeakview.apps.operator.backend.capture_drain import wait_for_capture_drain
from squeakview.apps.inference.capture_shutdown import RecordingShutdownProgress


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def wait(self, seconds):
        self.now += seconds
        return False


class ProgressDeadlineTests(unittest.TestCase):
    def test_real_progress_allows_work_past_idle_budget(self):
        clock = Clock()
        budget = ProgressDeadline(1, 10, monotonic=clock)

        def wait(seconds):
            clock.wait(seconds)
            return clock.now >= 3

        self.assertTrue(wait_for_completion(wait, budget, progress=lambda: {'encoded': int(clock.now * 4)}))
        self.assertGreater(clock.now, budget.stall_s)
        self.assertLess(clock.now, budget.maximum_s)

    def test_continuous_progress_cannot_extend_absolute_limit(self):
        clock = Clock()
        budget = ProgressDeadline(1, 3, monotonic=clock)
        self.assertFalse(wait_for_completion(clock.wait, budget, progress=lambda: {'encoded': int(clock.now * 100)}))
        self.assertAlmostEqual(clock.now, 3)
        self.assertEqual(budget.report()['limit'], 'overall')

    def test_stalled_counters_stop_after_idle_limit(self):
        clock = Clock()
        budget = ProgressDeadline(1, 10, monotonic=clock)
        self.assertFalse(wait_for_completion(clock.wait, budget, progress=lambda: {'encoded': 100}))
        self.assertAlmostEqual(clock.now, 1)
        self.assertEqual(budget.report()['limit'], 'stall')

    def test_regressing_missing_or_malformed_counters_never_refresh_idle_budget(self):
        clock = Clock()
        budget = ProgressDeadline(1, 10, monotonic=clock)
        budget.observe({'encoded': 100})
        for sample in ({}, {'encoded': None}, {'encoded': True}, {'encoded': -1},
                       {'encoded': 99}, {'encoded': 100}, {'new_counter': 1000}):
            clock.now += .1
            budget.observe(sample)
        self.assertEqual(budget.last_progress, 0)
        clock.now = .8
        budget.observe({'encoded': 101})
        self.assertEqual(budget.last_progress, .8)

    def test_completed_writer_steps_extend_idle_limit_but_not_overall_limit(self):
        clock = Clock()
        budget = ProgressDeadline(1, 3, monotonic=clock)
        for now in (.8, 1.6, 2.4):
            clock.now = now
            budget.completed_step()
            self.assertGreater(budget.remaining_s(), 0)
        clock.now = 3
        self.assertEqual(budget.remaining_s(), 0)
        self.assertEqual(budget.report()['completed_steps'], 3)

    def test_late_observation_cannot_revive_stalled_wait(self):
        clock = Clock()
        budget = ProgressDeadline(1, 10, monotonic=clock)
        budget.observe({'encoded': 0})
        clock.now = 2
        self.assertFalse(wait_for_completion(clock.wait, budget, progress=lambda: {'encoded': 100}))
        self.assertEqual(clock.now, 2)

    def test_multiple_camera_counters_are_tracked_independently(self):
        clock = Clock()
        budget = ProgressDeadline(1, 3, monotonic=clock)
        budget.observe({'cam0': 100, 'cam1': 10})
        clock.now = .5
        budget.observe({'cam0': 90, 'cam1': 11})
        self.assertEqual(budget.last_progress, .5)
        self.assertEqual(budget.high_water, {'cam0': 100, 'cam1': 11})
        clock.now = .8
        budget.observe({'cam0': 100, 'cam1': 11})
        self.assertEqual(budget.last_progress, .5)


class LedgerProgressTests(unittest.TestCase):
    def drain(self, counts, *, target, maximum=4, sizes=lambda: (10, 10)):
        clock = self.clock
        with mock.patch('squeakview.apps.operator.backend.capture_drain.ledger_frame_counts', side_effect=counts), \
             mock.patch('squeakview.apps.operator.backend.capture_drain.ledger_size_snapshot', side_effect=lambda *_: sizes()):
            return wait_for_capture_drain(Path('/unused'), 1, expected_ttl_count=target,
                quiet_s=.1, timeout_s=.5, maximum_s=maximum, monotonic=clock,
                sleep=clock.wait, poll_s=.05)

    def setUp(self):
        self.clock = Clock()

    def test_slow_ledger_drain_can_finish_after_original_timeout(self):
        def counts(*_):
            count = min(5, 1 + int(self.clock.now / .2))
            return count, count
        with mock.patch('squeakview.apps.operator.backend.capture_drain.known_missing_frames', return_value=0):
            result = self.drain(counts, target=5)
        self.assertTrue(result.passed)
        self.assertGreater(self.clock.now, .5)

    def test_ledger_byte_growth_without_new_frames_does_not_extend_wait(self):
        result = self.drain(lambda *_: (10, 9), target=None,
                            sizes=lambda: (int(self.clock.now * 1000), 10))
        self.assertFalse(result.passed)
        self.assertAlmostEqual(self.clock.now, .5)
        self.assertIn('stall timeout', result.reason)

    def test_endlessly_advancing_unreconciled_counts_hit_overall_limit(self):
        result = self.drain(lambda *_: (int(self.clock.now * 100) + 2, int(self.clock.now * 100)),
                            target=None, maximum=2)
        self.assertFalse(result.passed)
        self.assertAlmostEqual(self.clock.now, 2)
        self.assertIn('overall timeout', result.reason)


class ShutdownMaximumTests(unittest.TestCase):
    def test_invalid_maximum_cannot_shorten_stall_budget_or_be_unbounded(self):
        for value in ('nan', 'inf', '-1', '10', '601', 'bad'):
            with self.subTest(value=value), mock.patch.dict('os.environ', {
                'SQUEAKVIEW_PIPELINE_SHUTDOWN_TIMEOUT_S': '20',
                'SQUEAKVIEW_PIPELINE_SHUTDOWN_MAX_S': value,
            }):
                self.assertEqual(shutdown_policy.pipeline_shutdown_max_s(), 120)

    def test_supervisor_allows_full_progress_budget_and_cleanup(self):
        with mock.patch.dict('os.environ', {
            'SQUEAKVIEW_PIPELINE_SHUTDOWN_TIMEOUT_S': '20',
            'SQUEAKVIEW_PIPELINE_SHUTDOWN_MAX_S': '240',
        }):
            self.assertEqual(shutdown_policy.capture_exit_grace_s(), 286)


class ProgressSamplerTests(unittest.TestCase):
    def test_blocked_filesystem_sampler_does_not_block_shutdown_owner(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        telemetry = mock.Mock()
        telemetry.shutdown_progress.return_value = (10, 9, 8)
        app = SimpleNamespace(record_telemetry=[telemetry], run_dir=Path('/unused'),
                              config=SimpleNamespace(num_cameras=1))

        def blocked_stat(*_):
            entered.set()
            release.wait(2)
            return SimpleNamespace(st_size=1)

        with mock.patch.object(Path, 'stat', blocked_stat):
            sampler = RecordingShutdownProgress(app)
            self.addCleanup(sampler.close)
            self.assertTrue(entered.wait(1))
            self.assertEqual(sampler.snapshot()['camera_0_encoded'], 8)
            sampler.close()
            self.assertFalse(release.is_set())
            release.set()
