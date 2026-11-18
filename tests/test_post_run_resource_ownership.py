from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from squeakview.apps.inference import post_run
from squeakview.common import run_context
from squeakview.common.run_layout import artifact_path
from tests.support.recordings import create_recording


class PostRunResourceOwnershipTests(unittest.TestCase):
    def test_inference_index_failure_preserves_valid_raw_capture_and_cleans_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            create_recording(run_dir, 'native', health=True)
            with (
                mock.patch.object(post_run, "_open_index", wraps=post_run._open_index),
                mock.patch.object(
                    post_run,
                    "_index_inference_frames",
                    side_effect=RuntimeError("invalid inference ledger"),
                ),
            ):
                result = post_run.finalize_run(run_dir, camera_count=1, enable_infer=True)

            self.assertTrue(result.validation_passed)
            self.assertTrue(any('invalid inference ledger' in warning for warning in result.warnings))
            self.assertEqual(list(run_dir.glob(".post_run.*")), [])
            status = run_context.read_json(artifact_path(run_dir, "run_status.json"))
            self.assertTrue(status['recording_validation_passed'])
            self.assertFalse(status['inference_admission']['passed'])


if __name__ == "__main__":
    unittest.main()
