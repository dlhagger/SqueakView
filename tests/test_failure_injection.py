from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from squeakview.common.failure_injection import MAX_FAILURE_PLAN_BYTES, load_failure_plan
from squeakview.apps.inference.recording import RecordingStallOperator


class FailureInjectionTests(unittest.TestCase):
    def test_plan_requires_explicit_environment_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "plan.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "target": "flir_source",
                        "kind": "source_read",
                        "after_frames": 10,
                    }
                )
            )
            with mock.patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "requires"):
                    load_failure_plan(path)

    def test_valid_source_plan_is_immutable_and_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "plan.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "target": "flir_source",
                        "kind": "capture_ledger_write",
                        "after_frames": 25,
                        "stream_id": 1,
                    }
                )
            )
            with mock.patch.dict(
                os.environ,
                {"SQUEAKVIEW_ENABLE_FAILURE_INJECTION": "1"},
            ):
                plan = load_failure_plan(path)

        self.assertEqual(plan.after_frames, 25)
        self.assertEqual(plan.stream_id, 1)
        self.assertEqual(plan.as_manifest()["kind"], "capture_ledger_write")

    def test_queue_stall_requires_bounded_delay(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "plan.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "target": "record_queue",
                        "kind": "stall",
                        "after_frames": 1,
                        "delay_us": 0,
                    }
                )
            )
            with mock.patch.dict(
                os.environ,
                {"SQUEAKVIEW_ENABLE_FAILURE_INJECTION": "1"},
            ):
                with self.assertRaisesRegex(ValueError, "delay_us"):
                    load_failure_plan(path)

    def test_queue_stall_starts_after_exact_requested_frame_count(self) -> None:
        sleeps: list[float] = []
        operator = RecordingStallOperator(3, 250_000, sleep=sleeps.append)

        for _ in range(3):
            self.assertTrue(operator.handle_buffer(object()))
        self.assertEqual(sleeps, [])

        self.assertTrue(operator.handle_buffer(object()))
        self.assertEqual(sleeps, [0.25])

    def test_disk_full_is_explicitly_immediate_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ,
            {"SQUEAKVIEW_ENABLE_FAILURE_INJECTION": "1"},
        ):
            path = Path(temp_dir) / "plan.json"
            base = {
                "schema_version": "1.0",
                "target": "filesink",
                "kind": "disk_full",
            }
            path.write_text(json.dumps({**base, "after_frames": 2}))
            with self.assertRaisesRegex(
                ValueError, "immediate-only.*after_frames=1"
            ):
                load_failure_plan(path)

            path.write_text(json.dumps({**base, "after_frames": 1}))
            plan = load_failure_plan(path)

        self.assertEqual(plan.after_frames, 1)

    def test_plan_rejects_coerced_boolean_and_string_integers(self) -> None:
        base = {
            "schema_version": "1.0",
            "target": "flir_source",
            "kind": "source_read",
            "after_frames": 1,
        }
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ,
            {"SQUEAKVIEW_ENABLE_FAILURE_INJECTION": "1"},
        ):
            path = Path(temp_dir) / "plan.json"
            for invalid in (True, "1", 1.0):
                payload = dict(base, after_frames=invalid)
                path.write_text(json.dumps(payload))
                with self.subTest(value=invalid), self.assertRaisesRegex(
                    ValueError, "after_frames must be an integer"
                ):
                    load_failure_plan(path)

    def test_plan_rejects_missing_unknown_and_duplicate_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(
            os.environ,
            {"SQUEAKVIEW_ENABLE_FAILURE_INJECTION": "1"},
        ):
            path = Path(temp_dir) / "plan.json"
            path.write_text(
                '{"schema_version":"1.0","target":"flir_source",'
                '"kind":"source_read","kind":"source_incomplete",'
                '"after_frames":1}'
            )
            with self.assertRaisesRegex(ValueError, "duplicate field: kind"):
                load_failure_plan(path)

            path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "target": "flir_source",
                        "kind": "source_read",
                        "unexpected": 1,
                    }
                )
            )
            with self.assertRaisesRegex(
                ValueError, "missing fields: after_frames; unknown fields: unexpected"
            ):
                load_failure_plan(path)

    def test_plan_read_is_size_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "plan.json"
            path.write_bytes(b" " * (MAX_FAILURE_PLAN_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "exceeds"):
                load_failure_plan(path)


if __name__ == "__main__":
    unittest.main()
