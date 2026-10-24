from __future__ import annotations

from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest


class CrashDiagnosticsTests(unittest.TestCase):
    def run_worker(self, path, body):
        code = (
            "import resource; resource.setrlimit(resource.RLIMIT_CORE,(0,0))\n"
            "from pathlib import Path\n"
            "from squeakview.common.diagnostics.crash import install_crash_diagnostics\n"
            f"p=install_crash_diagnostics('test',Path({str(path)!r})); assert p is not None\n"
            + body
        )
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)

    def test_fatal_native_signal_preserves_stacks_in_private_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "crash.log"
            result = self.run_worker(path, "import os,signal; os.kill(os.getpid(),signal.SIGABRT)\n")
            self.assertEqual(result.returncode, -signal.SIGABRT)
            text = path.read_text()
            self.assertIn("Fatal Python error: Aborted", text)
            self.assertIn("role=test", text)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_main_and_thread_exceptions_survive_missing_console(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "crash.log"
            result = self.run_worker(path,
                "import threading,sys\n"
                "sys.stderr=open('/dev/null','w')\n"
                "def worker(): raise RuntimeError('worker lost')\n"
                "t=threading.Thread(target=worker,name='diagnostic-worker'); t.start(); t.join()\n"
                "raise ValueError('main lost')\n")
            self.assertNotEqual(result.returncode, 0)
            text = path.read_text()
            self.assertIn("UNHANDLED THREAD EXCEPTION (diagnostic-worker)", text)
            self.assertIn("worker lost", text)
            self.assertIn("UNHANDLED MAIN/Qt CALLBACK EXCEPTION", text)
            self.assertIn("main lost", text)

    def test_installation_failure_is_best_effort(self):
        result = subprocess.run([sys.executable, "-c",
            "from pathlib import Path; from squeakview.common.diagnostics.crash import install_crash_diagnostics; "
            "assert install_crash_diagnostics('test',Path('/dev/null/crash.log')) is None"],
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_repeated_install_keeps_original_handler_descriptor(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "crash.log"
            result = self.run_worker(path,
                "assert install_crash_diagnostics('again',Path('/dev/null/ignored')) == p\n")
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
