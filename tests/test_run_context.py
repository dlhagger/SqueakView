from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from squeakview.common import run_context


class RunContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "runs"
        self.runs_patch = mock.patch.object(run_context, "RUNS_DIR", self.root)
        self.marker_patch = mock.patch.object(run_context, "RUN_MARKER", self.root / ".latest_run")
        self.runs_patch.start()
        self.marker_patch.start()

    def tearDown(self) -> None:
        self.marker_patch.stop()
        self.runs_patch.stop()
        self.temp_dir.cleanup()

    def test_create_run_dir_slugifies_names_and_retries_collision(self) -> None:
        base = self.root / "Experiment_One" / "Mouse_1"
        base.mkdir(parents=True)
        first_id = "Mouse_1_2026-01-02_03-04-05_010101"
        (base / first_id).mkdir()
        with (
            mock.patch.object(run_context.time, "strftime", return_value="2026-01-02_03-04-05"),
            mock.patch.object(run_context.os, "urandom", side_effect=[b"\x01" * 3, b"\x02" * 3]),
        ):
            path, run_id = run_context.create_run_dir(
                experiment_name="Experiment One",
                mouse_id="Mouse 1",
            )

        self.assertEqual(run_id, "Mouse_1_2026-01-02_03-04-05_020202")
        self.assertEqual(path, base / run_id)
        self.assertEqual(run_context.latest_run_dir(), path)

    def test_status_history_keeps_first_lifecycle_timestamps(self) -> None:
        run_dir = self.root / "run"
        timestamps = iter(
            [
                "2026-01-01T00:00:00",
                "2026-01-01T00:00:01",
                "2026-01-01T00:00:02",
                "2026-01-01T00:00:03",
            ]
        )
        with mock.patch.object(run_context, "_now_iso", side_effect=lambda: next(timestamps)):
            run_context.write_status(run_dir, "created")
            run_context.write_status(run_dir, "starting")
            run_context.write_status(run_dir, "recording")
            run_context.write_status(run_dir, "recording", note="duplicate ready marker")

        status = run_context.read_json(run_dir / run_context.RUN_STATUS_FILENAME)
        self.assertEqual(status["created_at"], "2026-01-01T00:00:00")
        self.assertEqual(status["starting_at"], "2026-01-01T00:00:01")
        self.assertEqual(status["started_at"], "2026-01-01T00:00:02")
        self.assertEqual(status["updated_at"], "2026-01-01T00:00:03")
        self.assertEqual(len(status["history"]), 4)
        self.assertFalse(list(run_dir.glob(".*.tmp")))

    def test_storage_check_rejects_insufficient_free_space(self) -> None:
        usage = run_context.shutil._ntuple_diskusage(total=1_000, used=900, free=100)
        with mock.patch.object(run_context.shutil, "disk_usage", return_value=usage):
            with self.assertRaisesRegex(OSError, "low on free space"):
                run_context.assert_runs_dir_ready(min_free_bytes=101)
        self.assertFalse(list(self.root.glob(".write_test_*")))

    def test_bottle_summary_rejects_invalid_numbers_and_calculates_intake(self) -> None:
        summary = run_context.build_bottle_summary(
            {
                "left": {"fluid": " water ", "initial_weight_g": "10.5", "final_weight_g": "8.25"},
                "right": {"fluid": "sucrose", "initial_weight_g": -1, "final_weight_g": "bad"},
            },
            updated_at="now",
        )

        self.assertEqual(summary["sides"]["left"]["fluid"], "water")
        self.assertEqual(summary["sides"]["left"]["intake_g"], 2.25)
        self.assertIsNone(summary["sides"]["right"]["initial_weight_g"])
        self.assertFalse(summary["complete"])
        self.assertIn("right.initial_weight_g", summary["missing_fields"])

    def test_unchanged_bottle_rows_preserve_entry_timestamps(self) -> None:
        run_dir = self.root / "run"
        bottles = {
            "left": {"fluid": "water", "initial_weight_g": 10, "final_weight_g": 9},
            "right": {"fluid": "water", "initial_weight_g": 12, "final_weight_g": 11},
        }
        with (
            mock.patch.object(run_context, "_now_iso", return_value="first"),
            mock.patch.object(run_context.time, "time_ns", return_value=100),
        ):
            run_context.write_bottle_artifacts(run_dir, bottles)
        with (
            mock.patch.object(run_context, "_now_iso", return_value="second"),
            mock.patch.object(run_context.time, "time_ns", return_value=200),
        ):
            summary = run_context.write_bottle_artifacts(run_dir, bottles)

        with (run_dir / run_context.BOTTLE_MEASUREMENTS_FILENAME).open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertTrue(summary["complete"])
        self.assertEqual({row["entered_at_iso"] for row in rows}, {"first"})
        self.assertEqual({row["entered_at_unix_ns"] for row in rows}, {"100"})


if __name__ == "__main__":
    unittest.main()
