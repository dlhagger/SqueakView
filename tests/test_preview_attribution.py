from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from squeakview.apps.inference import post_run
from squeakview.apps.inference import preview_attribution
from squeakview.apps.inference.preview_attribution import (
    HEADERS,
    PreviewBoundaryOperator,
    boundary_path,
    reconcile_preview,
)
from squeakview.common import run_context


class PreviewAttributionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name) / "run"
        (self.run_dir / "diagnostics").mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write(self, boundary: str, rows: list[tuple[int, int, int]]) -> None:
        with boundary_path(self.run_dir, 0, boundary).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADERS)
            for sequence, camera_frame_id, pts_ns in rows:
                writer.writerow(
                    [boundary, 0, sequence, camera_frame_id, pts_ns, sequence + 100]
                )

    def test_reconciles_intentional_shedding_by_source_identity(self) -> None:
        self._write("admission", [(10, 110, 1000), (11, 111, 2000), (12, 112, 3000)])
        self._write("delivery", [(10, 110, 1000), (12, 112, 3000)])

        report = reconcile_preview(self.run_dir, 1, required=True)

        self.assertTrue(report["passed"])
        camera = report["cameras"][0]
        self.assertEqual(camera["admitted_frames"], 3)
        self.assertEqual(camera["delivered_frames"], 2)
        self.assertEqual(camera["shed_frames"], 1)
        self.assertEqual(camera["unmatched_delivery_frames"], 0)
        self.assertTrue(camera["admission_identity"]["available"])
        self.assertTrue(camera["delivery_identity"]["available"])

    def test_mismatched_delivery_identity_fails_attribution_only(self) -> None:
        self._write("admission", [(10, 110, 1000)])
        self._write("delivery", [(10, 999, 1000)])

        report = reconcile_preview(self.run_dir, 1, required=True)

        self.assertFalse(report["passed"])
        self.assertEqual(report["cameras"][0]["unmatched_delivery_frames"], 1)
        self.assertEqual(
            report["policy"],
            "preview_shedding_is_observed_but_never_invalidates_recording",
        )

    def test_missing_ledgers_are_bounded_failed_evidence_not_an_exception(self) -> None:
        report = reconcile_preview(self.run_dir, 1, required=True)

        self.assertFalse(report["passed"])
        self.assertEqual(report["cameras"], [])
        self.assertGreaterEqual(len(report["errors"]), 1)

    def test_extra_ledger_and_invalid_camera_count_fail_closed(self) -> None:
        self._write("admission", [(1, 41, 1000)])
        self._write("delivery", [(1, 41, 1000)])
        (self.run_dir / "diagnostics" / "preview_stale.csv").write_text("stale\n")

        extra = reconcile_preview(self.run_dir, 1, required=True)
        invalid = reconcile_preview(self.run_dir, 65, required=True)

        self.assertFalse(extra["passed"])
        self.assertTrue(any("not exact" in error for error in extra["errors"]))
        self.assertFalse(invalid["passed"])
        self.assertIn("camera_count", invalid["errors"][0])

    def test_reconciliation_rejects_ledger_changed_during_parse(self) -> None:
        self._write("admission", [(1, 41, 1000)])
        self._write("delivery", [(1, 41, 1000)])
        original = preview_attribution._iter_boundary

        def mutate_after_read(path: Path, stream_id: int, boundary: str):
            yield from original(path, stream_id, boundary)
            if boundary == "admission":
                path.write_text(path.read_text() + "changed\n")

        with mock.patch.object(
            preview_attribution, "_iter_boundary", side_effect=mutate_after_read
        ):
            report = reconcile_preview(self.run_dir, 1, required=True)

        self.assertFalse(report["passed"])
        self.assertTrue(any("changed during" in error for error in report["errors"]))

    def test_preview_disabled_requires_no_boundary_artifacts(self) -> None:
        report = reconcile_preview(self.run_dir, 1, required=False)

        self.assertTrue(report["passed"])
        self.assertFalse(report["required"])

    def test_boundary_operator_persists_native_source_identity(self) -> None:
        path = boundary_path(self.run_dir, 0, "admission")
        operator = PreviewBoundaryOperator(path, 0, "admission", meta_type=7)
        payload = {"source_sequence_index": 12, "camera_frame_id": 112}
        user_meta = SimpleNamespace(get_user_data_json=lambda: payload)
        frame = SimpleNamespace(
            source_id=0,
            pad_index=0,
            buffer_pts=3000,
            user_meta_items=lambda meta_type: iter([user_meta]) if meta_type == 7 else iter(()),
        )

        operator.handle_metadata(SimpleNamespace(frame_items=[frame]))
        operator.close()

        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["source_sequence_index"], "12")
        self.assertEqual(rows[0]["camera_frame_id"], "112")
        self.assertEqual(rows[0]["pts_ns"], "3000")

    def test_boundary_operator_contains_native_iterator_failure(self) -> None:
        path = boundary_path(self.run_dir, 0, "delivery")
        operator = PreviewBoundaryOperator(path, 0, "delivery", meta_type=7)

        def failed_iterator(_meta_type):
            raise RuntimeError("native metadata iterator failed")

        frame = SimpleNamespace(
            source_id=0,
            pad_index=0,
            buffer_pts=3000,
            user_meta_items=failed_iterator,
        )

        operator.handle_metadata(SimpleNamespace(frame_items=[frame]))
        operator.close()

        self.assertIsNone(operator.error)
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["source_sequence_index"], "")

    def test_boundary_writer_failure_is_latched_without_per_frame_retries(self) -> None:
        path = boundary_path(self.run_dir, 0, "delivery")
        operator = PreviewBoundaryOperator(path, 0, "delivery", meta_type=7)
        payload = {"source_sequence_index": 12, "camera_frame_id": 112}
        user_meta = SimpleNamespace(get_user_data_json=lambda: payload)
        frame = SimpleNamespace(
            source_id=0,
            pad_index=0,
            buffer_pts=3000,
            user_meta_items=lambda _meta_type: iter([user_meta]),
        )
        writer = mock.Mock()
        writer.writerow.side_effect = OSError("preview ledger full")
        operator._writer = writer

        operator.handle_metadata(SimpleNamespace(frame_items=[frame]))
        operator.handle_metadata(SimpleNamespace(frame_items=[frame]))
        operator.close()

        self.assertIn("preview ledger full", operator.error or "")
        writer.writerow.assert_called_once()

    def test_missing_preview_evidence_never_invalidates_recording(self) -> None:
        payload = {
            "camera_index": 0,
            "source_sequence_index": 0,
            "camera_frame_id": 41,
            "gst_pts_ns": 0,
            "host_received_monotonic_ns": 1,
            "host_received_unix_ns": 2,
            "actual_fps": 30.0,
            "telemetry_sample": True,
            "stream_incomplete_frames": 0,
            "stream_lost_frames": 0,
            "stream_dropped_frames": 0,
        }
        (self.run_dir / "capture_cam0.jsonl").write_text(json.dumps(payload) + "\n")
        (self.run_dir / "record_admission.csv").write_text(
            "stream_id,record_frame_index,pts_ns,observer_monotonic_ns\n0,0,0,3\n"
        )
        (self.run_dir / "raw.mp4").write_bytes(b"video")
        (self.run_dir / "run_manifest.json").write_text(
            json.dumps({"inference": {"preview_enabled": True}})
        )
        with mock.patch.object(
            post_run,
            "probe_video_frames",
            return_value={"count": 1, "method": "full_decode_ffmpeg", "error": None},
        ):
            result = post_run.finalize_run(
                self.run_dir, camera_count=1, enable_infer=False
            )

        status = run_context.read_json(self.run_dir / "run_status.json")
        self.assertTrue(result.validation_passed)
        self.assertTrue(status["overall_validation_passed"])
        self.assertFalse(status["preview_attribution"]["passed"])


if __name__ == "__main__":
    unittest.main()
