from __future__ import annotations

import csv
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from squeakview.project.locking import OwnershipLock
from squeakview.apps.operator.backend.preflight import _run_bounded_preflight, MAX_PREFLIGHT_LINE_BYTES
from squeakview.apps.operator.backend.startup import validate_startup_policy
from squeakview.apps.operator.backend.contracts import RunRequest
from squeakview.common import run_context
from squeakview.common.measurements import MAX_WEIGHT_G, weight_value
from squeakview.common.run_layout import initialize_layout


class ReviewRegressionTests(unittest.TestCase):
    def test_bottle_weight_clear_is_visible_after_reopening(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            initialize_layout(root, compact_metadata=True)
            for final in (20, None, None, 19):
                run_context.write_bottle_artifacts(root, {"left": {"fluid": "water", "initial_weight_g": 25, "final_weight_g": final}})
                self.assertEqual(run_context.bottle_setup(root)["sides"]["left"]["final_weight_g"], final)
            with (root / "weigh_in.csv").open(newline="") as handle:
                rows = [row for row in csv.DictReader(handle) if row["side"] == "left" and row["phase"] == "final"]
            self.assertEqual([row["weight_g"] for row in rows], ["20.0", "", "19.0"])

    def test_lock_failure_releases_unpublished_handle_with_traceback_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "owner.lock"
            for target in ("os.fsync", "json.dump", "socket.gethostname"):
                lock = OwnershipLock(path, purpose="first")
                retained = []
                with mock.patch("squeakview.project.locking." + target, side_effect=OSError("write failed")):
                    try:
                        lock.acquire()
                    except OSError as error:
                        retained.append(error)
                self.assertTrue(retained)
                self.assertFalse(lock.held)
                other = OwnershipLock(path, purpose="second")
                self.assertTrue(other.acquire())
                other.release()

    def test_preflight_long_physical_line_is_bounded_and_recovers_at_newline(self):
        lines = []
        result = _run_bounded_preflight(
            [sys.executable, "-c", "import sys;sys.stdout.buffer.write(b'x'*2097152+b'\\nrecovered \\xc3\\xa9\\n')"],
            cwd="/tmp", env=dict(os.environ), timeout_s=5, on_line=lines.append,
        )
        self.assertTrue(result.truncated)
        self.assertEqual(lines[-1], "recovered é")
        self.assertLessEqual(max(len(line.encode()) for line in lines), MAX_PREFLIGHT_LINE_BYTES + 30)
        self.assertEqual(sum("line truncated" in line for line in lines), 1)

    def test_weight_limits_are_enforced_before_start_and_before_writing(self):
        self.assertEqual(weight_value(MAX_WEIGHT_G, "Bottle"), MAX_WEIGHT_G)
        self.assertEqual(weight_value(0, "Bottle"), 0)
        for value in (454251, float("nan"), float("inf"), -1, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_startup_policy(RunRequest(bottles={"mouse_weight_g": value}))
                with tempfile.TemporaryDirectory() as temp:
                    with self.assertRaises(ValueError):
                        run_context.write_bottle_artifacts(Path(temp), {"mouse_weight_g": value})
                    self.assertFalse(list(Path(temp).iterdir()))

    def test_clear_and_correct_mouse_weight_preserves_timestamped_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            initialize_layout(root, compact_metadata=True)
            for value in (25, None, None, 26):
                run_context.write_bottle_artifacts(root, {"mouse_weight_g": value})
                self.assertEqual(run_context.bottle_summary(root)["mouse_weight_g"], value)
            with (root / "weigh_in.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["weight_g"] for row in rows], ["25.0", "", "26.0"])
            self.assertEqual(rows[1]["event"], "measurement_cleared")
            self.assertTrue(all(row["host_unix_ns"] and row["host_monotonic_ns"] for row in rows))
