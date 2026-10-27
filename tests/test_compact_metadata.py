from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from squeakview.common import run_context
from squeakview.common.metadata_retention import compact_successful_run
from squeakview.common.run_layout import artifact_path, initialize_layout, open_artifact
from squeakview.apps.operator.backend.contracts import RunRequest
from squeakview.apps.operator.backend.manifest import RunManifestContext, RunManifestService


class CompactMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        initialize_layout(self.root, compact_metadata=True)

    def write(self, name, data):
        path = artifact_path(self.root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def status(self):
        return run_context.read_json_required(run_context.status_path(self.root))

    def test_manifest_updates_preserve_intervening_status_and_acquisition(self):
        run_context.write_status(self.root, "starting")
        service = RunManifestService(lambda _message: None)
        context = RunManifestContext(
            config=RunRequest(), application_root=self.root, project_root=self.root,
            project_id="test", project_name="test", runs_root=self.root,
            created_at="2026-10-27T12:00:00", storage={}, model_snapshot=None,
            device_context=None,
        )
        self.assertTrue(service.write(self.root, context, required=True))
        self.assertEqual(self.status()["state"], "starting")
        original_capture = self.status()["capture"]
        run_context.write_status(self.root, "finalized", overall_validation_passed=True)

        def inventory(_root):
            run_context.update_status(self.root, latest_controller_evidence="preserved")
            return {"frames": {"rows": 10}}

        self.assertTrue(service.write(self.root, context, required=True, output_snapshot=inventory))
        saved = self.status()
        self.assertEqual(saved["state"], "finalized")
        self.assertEqual(saved["latest_controller_evidence"], "preserved")
        self.assertEqual(saved["capture"], original_capture)
        self.assertEqual(saved["actual_outputs"]["frames"]["rows"], 10)
        self.assertFalse((self.root / "_metadata/run_status.json").exists())
        self.assertEqual(saved["recording"]["validation"], "_metadata/run_manifest.json:recording_validation")

    def test_status_history_does_not_duplicate_reports(self):
        run_context.write_status(self.root, "post_run_complete", outputs={"frames": 3},
                                 recording_validation={"passed": True})
        saved = self.status()
        self.assertEqual(saved["actual_outputs"], {"frames": 3})
        self.assertEqual(saved["recording_validation"], {"passed": True})
        self.assertNotIn("recording_validation", saved["history"][0])
        self.assertNotIn("outputs", saved)

    def test_bottle_views_come_from_weigh_in_without_json_caches(self):
        run_context.write_bottle_artifacts(self.root, {
            "mouse_weight_g": 25.5,
            "left": {"fluid": "water", "initial_weight_g": 100, "final_weight_g": 98},
            "right": {"fluid": "sucrose", "initial_weight_g": 90, "final_weight_g": 87},
        })
        summary = run_context.bottle_summary(self.root)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["sides"]["left"]["intake_g"], 2)
        self.assertEqual(run_context.bottle_setup(self.root)["mouse_weight_g"], 25.5)
        self.assertTrue((self.root / "weigh_in.csv").exists())
        self.assertFalse(list((self.root / "_metadata").glob("bottle*.json")))

    def test_compression_round_trip_idempotence_and_primary_files_unchanged(self):
        payload = b'{"source_sequence_index":1}\n' * 1000
        names = ["capture_cam0.jsonl", "record_admission.csv", "inference/frames.csv"]
        originals = [self.write(name, payload) for name in names]
        primary = self.write("frames.csv", b"frame_index\n0\n")
        run_context.write_status(self.root, "post_run_complete", overall_validation_passed=True)
        compact_successful_run(self.root, 1)
        for name, original in zip(names, originals):
            compressed = artifact_path(self.root, name)
            self.assertFalse(original.exists())
            self.assertIn("recovery", compressed.parts)
            with open_artifact(compressed) as handle:
                self.assertEqual(handle.read(), payload)
            entry = self.status()["metadata_retention"]["artifacts"][name]
            self.assertEqual(entry["uncompressed_sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(primary.read_bytes(), b"frame_index\n0\n")
        compact_successful_run(self.root, 1)
        self.assertTrue(self.status()["metadata_retention"]["complete"])

    def test_validation_failure_debug_and_qualification_retain_plain_ledgers(self):
        path = self.write("capture_cam0.jsonl", b"evidence\n")
        for updates in (
            {"overall_validation_passed": False},
            {"overall_validation_passed": True, "observability": {"deepstream_debug_profile": True}},
            {"observability": {}, "qualification": {"case": "test"}},
            {"qualification": None, "recording_validation": {"evidence": {"sha256": "hash"}}},
        ):
            run_context.write_status(self.root, "post_run_complete", **updates)
            compact_successful_run(self.root, 1)
            self.assertEqual(path.read_bytes(), b"evidence\n")

    def test_failed_round_trip_keeps_original_and_removes_temporary(self):
        original = self.write("capture_cam0.jsonl", b"evidence\n")
        run_context.write_status(self.root, "post_run_complete", overall_validation_passed=True)
        with patch("squeakview.common.metadata_retention._digest", return_value=("wrong", 0)):
            with self.assertRaises(RuntimeError):
                compact_successful_run(self.root, 1)
        self.assertEqual(original.read_bytes(), b"evidence\n")
        self.assertFalse(list((self.root / "_metadata/recovery").iterdir()))

    def test_finalized_frame_header_and_compressed_ledger_counts_remain_readable(self):
        import csv
        from squeakview.apps.inference.frame_audit import FrameCsvOperator
        from squeakview.apps.inference.post_run import _durable_frame_manifest_counts, _durable_ledger_counts
        from squeakview.common.run_outputs import TIMING_FIELDS
        header = list(FrameCsvOperator.HEADERS)
        header += [key for key in TIMING_FIELDS if key not in header]
        with (self.root / "frames.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            writer.writerow({"stream_id": 0, "source_sequence_index": 41})
        self.write("capture_cam0.jsonl", b'{"source_sequence_index":41}\n')
        self.write("record_admission.csv", b"stream_id,record_frame_index,pts_ns\n0,41,100\n")
        run_context.write_status(self.root, "post_run_complete", overall_validation_passed=True)
        compact_successful_run(self.root, 1)
        self.assertEqual(_durable_frame_manifest_counts(self.root, 1), {0: 42})
        self.assertEqual(_durable_ledger_counts(self.root, 1), {0: (42, 42)})


if __name__ == "__main__":
    unittest.main()
