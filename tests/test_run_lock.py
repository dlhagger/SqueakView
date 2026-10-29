from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from squeakview.apps.operator.backend.run_lock import AcquisitionLock
from squeakview.project.locking import OwnershipLock


class AcquisitionLockTests(unittest.TestCase):
    def test_excludes_second_owner_and_can_be_reacquired(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".acquisition.lock"
            first = AcquisitionLock(path)
            second = AcquisitionLock(path)

            self.assertTrue(first.acquire())
            self.assertTrue(first.held)
            self.assertFalse(second.acquire())
            self.assertIn('"pid":', path.read_text())

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

    def test_refuses_symlink_lock_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "target"
            target.write_text("preserve")
            link = root / "lock"
            link.symlink_to(target)

            with self.assertRaises(OSError):
                AcquisitionLock(link).acquire()

            self.assertEqual(target.read_text(), "preserve")


class OwnershipLockPersistenceTests(unittest.TestCase):

    def test_lock_failure_releases_unpublished_handle_with_traceback_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "owner.lock"
            for target in ("os.fsync", "json.dump", "socket.gethostname"):
                lock = OwnershipLock(path, purpose="first")
                retained = []
                with mock.patch("squeakview.project.locking." + target, side_effect=OSError("write failed")):
                    try:
                        lock.acquire()
                    except OSError as error:
                        retained.append(error)
                self.assertTrue(retained)
                self.assertFalse(lock.held)
                other = OwnershipLock(path, purpose="second")
                self.assertTrue(other.acquire())
                other.release()


if __name__ == "__main__":
    unittest.main()
