from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from squeakview.common.recording_evidence import (
    capture_recording_evidence,
    recording_evidence_complete,
    recording_evidence_matches,
    recording_evidence_metadata_matches,
)


class RecordingEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name) / "run"
        self.run_dir.mkdir()
        (self.run_dir / "raw.mp4").write_bytes(b"video")
        (self.run_dir / "capture_cam0.jsonl").write_bytes(b"capture\n")
        (self.run_dir / "record_admission.csv").write_bytes(b"admission\n")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_complete_snapshot_matches_unchanged_artifacts(self) -> None:
        evidence = capture_recording_evidence(self.run_dir, 1)

        self.assertTrue(recording_evidence_complete(evidence, 1))
        self.assertTrue(recording_evidence_matches(self.run_dir, evidence, 1))
        self.assertTrue(
            recording_evidence_metadata_matches(self.run_dir, evidence, 1)
        )

    def test_video_or_ledger_mutation_invalidates_finalization_evidence(self) -> None:
        for name in ("raw.mp4", "capture_cam0.jsonl", "record_admission.csv"):
            with self.subTest(name=name):
                path = self.run_dir / name
                original = path.read_bytes()
                evidence = capture_recording_evidence(self.run_dir, 1)
                path.write_bytes(original + b"changed")

                self.assertFalse(
                    recording_evidence_matches(self.run_dir, evidence, 1)
                )
                path.write_bytes(original)

    def test_symlink_and_extra_artifact_fail_closed(self) -> None:
        outside = Path(self.temporary.name) / "outside.mp4"
        outside.write_bytes(b"video")
        (self.run_dir / "raw.mp4").unlink()
        (self.run_dir / "raw.mp4").symlink_to(outside)

        symlinked = capture_recording_evidence(self.run_dir, 1)

        self.assertFalse(recording_evidence_complete(symlinked, 1))
        (self.run_dir / "raw.mp4").unlink()
        (self.run_dir / "raw.mp4").write_bytes(b"video")
        (self.run_dir / "raw_stale.mp4").write_bytes(b"stale")

        extra = capture_recording_evidence(self.run_dir, 1)

        self.assertFalse(recording_evidence_complete(extra, 1))


if __name__ == "__main__":
    unittest.main()
