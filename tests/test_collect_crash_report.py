from pathlib import Path
import tempfile
import unittest
from unittest import mock

from squeakview.tools.collect_crash_report import collect, tail


class CrashReportTests(unittest.TestCase):
    def test_tail_is_bounded_and_preserves_invalid_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "crash.log"
            path.write_bytes(b"x" * 1000 + b"\xfflast")
            result = tail(path, 8)
            self.assertTrue(result["truncated"])
            self.assertLessEqual(len(result["tail"]), 8)
            self.assertIn("last", result["tail"])

    def test_collection_handles_missing_files_and_unavailable_kernel_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "project/runs/test"
            run.mkdir(parents=True)
            (run / "run_status.json").write_text('{"state":"recording"}')
            with mock.patch("squeakview.tools.collect_crash_report.subprocess.run", side_effect=PermissionError("no journal access")):
                report = collect(root / "missing", root / "project")
            self.assertEqual(report["logs"], [])
            self.assertIn("no journal access", report["kernel"]["error"])
            self.assertEqual(len(report["runs"]), 1)
            self.assertIn("recording", report["runs"][0]["status"]["tail"])
            self.assertTrue(all("error" in item for item in report["runs"][0]["diagnostics"]))
