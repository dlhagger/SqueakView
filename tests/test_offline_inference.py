from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from squeakview.apps.inference.offline import FrameAuditOperator, _load_frame_ledger
from squeakview.apps.inference.pose_pipeline import (
    FramePoseStore, ObservationOperator, PoseClass, PoseSchema,
)


class OfflineInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_frame_ledger_maps_decoded_ordinal_to_authoritative_identity(self) -> None:
        path = self.root / "frames.csv"
        path.write_text(
            "stream_id,source_sequence_index,raw_frame_index,camera_frame_id,"
            "camera_timestamp_ns,pts_ns,source_width,source_height\n"
            "0,7,7,20338,1000000,0,1440,1080\n"
            "0,8,8,20339,1033333,33333333,1440,1080\n"
        )
        ledger, width, height = _load_frame_ledger(path)
        self.assertEqual((width, height), (1440, 1080))
        self.assertEqual(ledger[0]["source_sequence_index"], 7)
        self.assertEqual(ledger[1]["camera_frame_id"], 20339)

    def test_frame_audit_rejects_noncontiguous_decoder_sequence(self) -> None:
        audit = FrameAuditOperator(3)
        audit.handle_metadata(SimpleNamespace(frame_items=[SimpleNamespace(frame_number=0)]))
        with self.assertRaisesRegex(RuntimeError, "not contiguous"):
            audit.handle_metadata(SimpleNamespace(frame_items=[SimpleNamespace(frame_number=2)]))

    def test_observation_writer_marks_offline_ledger_mapping(self) -> None:
        schema = PoseSchema(
            version=2, input_width=640, input_height=640, output_layer="output0",
            keypoint_names=("nose",),
            classes=(PoseClass(0, "mouse", 0.25, True, (0,)),),
            keypoint_threshold=0.5,
        )
        operator = ObservationOperator(
            self.root, schema, store=FramePoseStore(), flir_meta_type=None,
            frame_ledger={0: {
                "source_sequence_index": 7, "camera_frame_id": 20338,
                "camera_timestamp_ns": 1000000, "pts_ns": 0,
            }},
            mapping_method="offline_video_ledger", source_name="offline_raw_mp4",
        )
        rect = SimpleNamespace(left=1.0, top=2.0, width=3.0, height=4.0)
        object_meta = SimpleNamespace(
            class_id=0, object_id=42, label="", rect_params=rect, tracker_confidence=0.8,
        )
        frame = SimpleNamespace(
            frame_number=0, source_id=0, pad_index=0, buffer_pts=999,
            source_width=1440, source_height=1080, pipeline_width=1440, pipeline_height=1080,
            object_items=[object_meta],
        )
        operator.handle_metadata(SimpleNamespace(frame_items=[frame]))
        operator.close()
        self.assertFalse((self.root / "detections.csv").exists())
        with (self.root / "objects.csv").open(newline="") as handle:
            obj = next(csv.DictReader(handle))
        self.assertEqual(obj["source_sequence_index"], "7")
        self.assertEqual(obj["camera_frame_id"], "20338")
        self.assertEqual(obj["gst_pts_ns"], "0")


if __name__ == "__main__":
    unittest.main()
