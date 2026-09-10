from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from squeakview.apps.inference import post_run
from squeakview.common import run_context


class PostRunResourceOwnershipTests(unittest.TestCase):
    def test_inference_index_failure_closes_database_and_removes_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            run_dir.mkdir()
            index = mock.Mock()
            with (
                mock.patch.object(post_run, "_open_index", return_value=index),
                mock.patch.object(
                    post_run,
                    "_index_inference_frames",
                    side_effect=RuntimeError("invalid inference ledger"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "invalid inference ledger"):
                    post_run.finalize_run(
                        run_dir,
                        camera_count=1,
                        enable_infer=True,
                    )

            index.close.assert_called_once_with()
            self.assertEqual(list(run_dir.glob(".post_run.*")), [])
            status = run_context.read_json(run_dir / "run_status.json")
            self.assertEqual(status["state"], "finalization_failed")
            self.assertIn("invalid inference ledger", status["error"])


if __name__ == "__main__":
    unittest.main()
