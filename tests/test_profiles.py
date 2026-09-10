from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from squeakview.common import profiles


class ProfileStoreBoundedInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.store = profiles.ProfileStore(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_listing_tolerates_invalid_duplicate_and_oversized_profiles(self) -> None:
        (self.store.experiments_dir / "valid.json").write_text(
            '{"name":"Valid","slug":"valid","config":{}}'
        )
        (self.store.experiments_dir / "duplicate.json").write_text(
            '{"name":"first","name":"second"}'
        )
        (self.store.experiments_dir / "large.json").write_bytes(
            b" " * (profiles.MAX_PROFILE_BYTES + 1)
        )

        loaded = self.store.list_experiments()

        self.assertEqual([item.slug for item in loaded], ["valid"])

    def test_save_is_atomic_and_rejects_oversized_payload(self) -> None:
        saved = self.store.save_experiment(
            profiles.ExperimentProfile("Study", "study", {"fps": 30})
        )
        self.assertTrue(saved.is_file())
        self.assertFalse(list(saved.parent.glob(".*.tmp")))

        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.store.save_experiment(
                profiles.ExperimentProfile(
                    "Large", "large", {"padding": "x" * profiles.MAX_PROFILE_BYTES}
                )
            )
        self.assertFalse((self.store.experiments_dir / "large.json").exists())

    def test_listing_caps_profile_file_reads(self) -> None:
        paths = [
            self.store.experiments_dir / f"profile-{index}.json"
            for index in range(profiles.MAX_PROFILE_FILES + 5)
        ]
        iterator = iter(paths)
        with (
            mock.patch.object(Path, "glob", return_value=iterator),
            mock.patch.object(profiles, "read_json_object", return_value={}) as read,
        ):
            self.store.list_experiments()

        self.assertEqual(read.call_count, profiles.MAX_PROFILE_FILES)

    def test_save_rejects_new_file_when_store_is_at_capacity(self) -> None:
        with (
            mock.patch.object(Path, "exists", return_value=False),
            mock.patch.object(
                Path,
                "glob",
                return_value=iter(Path(f"profile-{index}.json") for index in range(profiles.MAX_PROFILE_FILES)),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "file limit"):
                self.store.save_subject(profiles.SubjectProfile("New", "new"))


if __name__ == "__main__":
    unittest.main()
