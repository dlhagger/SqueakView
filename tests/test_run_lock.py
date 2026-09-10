from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from squeakview.apps.operator.backend.run_lock import AcquisitionLock


class AcquisitionLockTests(unittest.TestCase):
    def test_excludes_second_owner_and_can_be_reacquired(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".acquisition.lock"
            first = AcquisitionLock(path)
            second = AcquisitionLock(path)

            self.assertTrue(first.acquire())
            self.assertTrue(first.held)
            self.assertFalse(second.acquire())
            self.assertIn("pid=", path.read_text())

            first.release()
            self.assertTrue(second.acquire())
            second.release()

    def test_release_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lock = AcquisitionLock(Path(temp_dir) / "lock")
            lock.release()
            self.assertTrue(lock.acquire())
            lock.release()
            lock.release()
            self.assertFalse(lock.held)


if __name__ == "__main__":
    unittest.main()
