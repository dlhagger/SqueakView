from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

from squeakview.apps.inference import (
    acquisition_validation,
    capture_reconciliation,
    inference_reconciliation,
    recording_validation,
)


class CaptureReconciliationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def _write_capture(self, frame_ids: list[int]) -> Path:
        path = self.root / "capture_cam0.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "camera_index": 0,
                        "source_sequence_index": sequence,
                        "camera_frame_id": frame_id,
                        "gst_pts_ns": sequence * 10,
                    }
                )
                + "\n"
                for sequence, frame_id in enumerate(frame_ids)
            )
        )
        return path

    def test_camera_identity_gap_is_a_fatal_integrity_error(self) -> None:
        path = self._write_capture([41, 43])

        with self.assertRaisesRegex(RuntimeError, "camera frame IDs are not contiguous"):
            list(capture_reconciliation.iter_capture_payloads(path, 0))

    def test_capture_ledger_rejects_coerced_duplicate_and_oversized_values(self) -> None:
        valid = {
            "camera_index": 0,
            "source_sequence_index": 0,
            "camera_frame_id": 41,
            "gst_pts_ns": 0,
        }
        path = self.root / "capture_cam0.jsonl"
        for raw in (
            json.dumps({**valid, "camera_index": False}),
            json.dumps({**valid, "source_sequence_index": 0.0}),
            '{"camera_index":0,"camera_index":0,"source_sequence_index":0,'
            '"camera_frame_id":41,"gst_pts_ns":0}',
            json.dumps({**valid, "padding": "x" * capture_reconciliation.MAX_LEDGER_LINE_BYTES}),
        ):
            with self.subTest(raw=raw[:40]):
                path.write_text(raw + "\n")
                with self.assertRaises(RuntimeError):
                    list(capture_reconciliation.iter_capture_payloads(path, 0))

    def test_admission_ledger_rejects_noncanonical_integer_and_duplicate_header(self) -> None:
        path = self.root / "record_admission.csv"
        for contents in (
            "stream_id,record_frame_index,pts_ns\n+0,0,0\n",
            "stream_id,record_frame_index,pts_ns,pts_ns\n0,0,0,0\n",
            "stream_id,record_frame_index,pts_ns\n0,0," + "1" * 65536 + "\n",
        ):
            with self.subTest(contents=contents[:50]):
                path.write_text(contents)
                with self.assertRaises(RuntimeError):
                    list(capture_reconciliation.iter_admission_pts(path, 0))

    def test_recorded_payloads_reconciles_in_streaming_order(self) -> None:
        self._write_capture([41, 42, 43])
        (self.root / "record_admission.csv").write_text(
            "stream_id,record_frame_index,pts_ns\n0,0,0\n0,1,20\n"
        )
        stats = capture_reconciliation.StreamStats()

        payloads = list(capture_reconciliation.recorded_payloads(self.root, 0, stats))

        self.assertEqual(
            [payload["source_sequence_index"] for payload in payloads], [0, 2]
        )
        self.assertEqual(stats.source_frames, 3)
        self.assertEqual(stats.recorded_frames, 2)
        self.assertEqual(stats.unmatched_admissions, 0)

    def test_duplicate_pts_is_rejected_as_ambiguous_frame_identity(self) -> None:
        path = self.root / "capture_cam0.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "camera_index": 0,
                        "source_sequence_index": sequence,
                        "camera_frame_id": 41 + sequence,
                        "gst_pts_ns": 0,
                    }
                )
                + "\n"
                for sequence in range(2)
            )
        )

        with self.assertRaisesRegex(RuntimeError, "strictly increasing"):
            list(capture_reconciliation.iter_capture_payloads(path, 0))


class InferenceReconciliationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_orphan_identity_fails_immutable_result(self) -> None:
        connection = inference_reconciliation.open_index(self.root / "index.sqlite")
        try:
            connection.execute(
                "INSERT INTO recorded VALUES (?, ?, ?, ?)", (0, 0, 0, "41")
            )
            ledger = self.root / "frames.csv"
            ledger.write_text("stream_id,source_sequence_index\n0,0\n0,2\n")
            inference_reconciliation.index_inference_frames(connection, ledger)

            result = inference_reconciliation.summarize_inference_admission(
                connection, {0: 1}
            )
        finally:
            connection.close()

        self.assertFalse(result.passed)
        self.assertEqual(result.to_dict()["orphan_inference_frames"], 1)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.orphan_frames = 0  # type: ignore[misc]

    def test_duplicate_inference_identity_is_rejected(self) -> None:
        connection = inference_reconciliation.open_index(self.root / "index.sqlite")
        ledger = self.root / "frames.csv"
        ledger.write_text("stream_id,source_sequence_index\n0,1\n0,1\n")
        try:
            with self.assertRaisesRegex(RuntimeError, "duplicate or out of order"):
                inference_reconciliation.index_inference_frames(connection, ledger)
        finally:
            connection.close()

    def test_inference_identity_requires_strict_bounded_schema(self) -> None:
        invalid_ledgers = (
            (
                "stream_id,stream_id,source_sequence_index\n0,0,1\n",
                "duplicate column names",
            ),
            (
                "stream_id,source_sequence_index\n+0,1\n",
                "invalid inference frame ledger",
            ),
            (
                "stream_id,source_sequence_index\n64,1\n",
                "exceeds supported camera count",
            ),
            (
                "stream_id,source_sequence_index\n0," + "1" * (256 * 1024) + "\n",
                "CSV record exceeds",
            ),
        )
        for index, (contents, message) in enumerate(invalid_ledgers):
            with self.subTest(message=message):
                connection = inference_reconciliation.open_index(
                    self.root / f"index-{index}.sqlite"
                )
                ledger = self.root / f"frames-{index}.csv"
                ledger.write_text(contents)
                try:
                    with self.assertRaisesRegex(RuntimeError, message):
                        inference_reconciliation.index_inference_frames(
                            connection, ledger
                        )
                finally:
                    connection.close()


class RecordingValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)

    def tearDown(self) -> None:
        self._temp.cleanup()

    @staticmethod
    def _probe(_path: Path) -> dict[str, object]:
        return {"count": 2, "method": "test", "error": None}

    def test_all_counts_and_nonempty_video_are_required(self) -> None:
        (self.root / "raw.mp4").write_bytes(b"video")
        (self.root / "capture_cam0.jsonl").write_text("capture\n")
        (self.root / "record_admission.csv").write_text(
            "stream_id,record_frame_index,pts_ns\n0,0,0\n0,1,10\n"
        )

        passed = recording_validation.validate_recordings(
            self.root, 1, {0: 2}, {0: 2}, probe=self._probe
        )
        source_gap = recording_validation.validate_recordings(
            self.root, 1, {0: 3}, {0: 2}, probe=self._probe
        )

        self.assertTrue(passed.passed)
        self.assertFalse(source_gap.passed)
        self.assertFalse(source_gap.to_dict()["cameras"][0]["source_count_matches"])
        with self.assertRaises(dataclasses.FrozenInstanceError):
            passed.cameras = ()  # type: ignore[misc]

    def test_mutation_during_video_probe_invalidates_the_recording(self) -> None:
        video = self.root / "raw.mp4"
        video.write_bytes(b"video")
        (self.root / "capture_cam0.jsonl").write_text("capture\n")
        (self.root / "record_admission.csv").write_text("admission\n")

        def mutate_during_probe(path: Path) -> dict[str, object]:
            path.write_bytes(b"other")
            return {"count": 2, "method": "test", "error": None}

        result = recording_validation.validate_recordings(
            self.root, 1, {0: 2}, {0: 2}, probe=mutate_during_probe
        )

        self.assertFalse(result.passed)
        self.assertFalse(result.evidence_unchanged_during_validation)
        self.assertFalse(result.to_dict()["passed"])


class AcquisitionValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        (self.root / "diagnostics").mkdir()

    def tearDown(self) -> None:
        self._temp.cleanup()

    def test_zero_transport_counters_and_no_events_pass(self) -> None:
        diagnostics = self.root / "diagnostics"
        (diagnostics / "errors.csv").write_text("event_type\n")
        (diagnostics / "camera.csv").write_text(
            "stream_id,stream_incomplete_frames,stream_lost_frames,stream_dropped_frames\n"
            "0,0,0,0\n"
        )

        result = acquisition_validation.validate_acquisition_integrity(self.root)

        self.assertTrue(result.passed)
        self.assertTrue(result.to_dict()["passed"])
        self.assertEqual(result.to_dict()["observed_stream_ids"], [0])

    def test_two_camera_telemetry_requires_and_accepts_both_streams(self) -> None:
        diagnostics = self.root / "diagnostics"
        (diagnostics / "errors.csv").write_text("event_type\n")
        (diagnostics / "camera.csv").write_text(
            "stream_id,stream_incomplete_frames,stream_lost_frames,stream_dropped_frames\n"
            "0,0,0,0\n"
            "1,0,0,0\n"
        )

        result = acquisition_validation.validate_acquisition_integrity(
            self.root, camera_count=2
        )

        self.assertTrue(result.passed)
        report = result.to_dict()
        self.assertEqual(report["expected_stream_ids"], [0, 1])
        self.assertEqual(report["observed_stream_ids"], [0, 1])
        self.assertEqual(report["telemetry_rows_by_stream"], {0: 1, 1: 1})

    def test_two_camera_telemetry_missing_second_stream_fails_closed(self) -> None:
        diagnostics = self.root / "diagnostics"
        (diagnostics / "errors.csv").write_text("event_type\n")
        (diagnostics / "camera.csv").write_text(
            "stream_id,stream_incomplete_frames,stream_lost_frames,stream_dropped_frames\n"
            "0,0,0,0\n"
        )

        result = acquisition_validation.validate_acquisition_integrity(
            self.root, camera_count=2
        )

        self.assertFalse(result.passed)
        report = result.to_dict()
        self.assertEqual(report["observed_stream_ids"], [0])
        self.assertEqual(report["telemetry_rows_by_stream"], {0: 1, 1: 0})
        self.assertEqual(
            report["transport_counter_samples_by_stream"][1],
            {
                "stream_incomplete_frames": 0,
                "stream_lost_frames": 0,
                "stream_dropped_frames": 0,
            },
        )

    def test_missing_counter_sample_fails_closed(self) -> None:
        diagnostics = self.root / "diagnostics"
        (diagnostics / "errors.csv").write_text("event_type\n")
        (diagnostics / "camera.csv").write_text(
            "stream_id,stream_incomplete_frames,stream_lost_frames,stream_dropped_frames\n"
            "0,0,,0\n"
        )

        result = acquisition_validation.validate_acquisition_integrity(self.root)

        self.assertFalse(result.passed)
        self.assertEqual(
            result.to_dict()["transport_counter_samples"]["stream_lost_frames"], 0
        )

    def test_fractional_transport_counter_fails_closed(self) -> None:
        diagnostics = self.root / "diagnostics"
        (diagnostics / "errors.csv").write_text("event_type\n")
        (diagnostics / "camera.csv").write_text(
            "stream_id,stream_incomplete_frames,stream_lost_frames,stream_dropped_frames\n"
            "0,0.5,0,0\n"
        )

        result = acquisition_validation.validate_acquisition_integrity(self.root)

        self.assertFalse(result.passed)
        self.assertEqual(result.invalid_diagnostic_rows, 1)


if __name__ == "__main__":
    unittest.main()
