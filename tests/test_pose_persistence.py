from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from squeakview.apps.inference.pose_persistence import (
    KEYPOINT_HEADERS,
    OBJECT_HEADERS,
    PoseCsvWriter,
)


class PoseCsvWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_writes_exact_versioned_ledgers_and_closes_idempotently(self) -> None:
        writer = PoseCsvWriter(self.root)
        object_row = list(range(len(OBJECT_HEADERS)))
        keypoint_row = list(range(len(KEYPOINT_HEADERS)))

        writer.write_object(object_row)
        writer.write_keypoints([keypoint_row])
        writer.close()
        writer.close()

        with (self.root / "objects.csv").open(newline="") as handle:
            objects = list(csv.reader(handle))
        with (self.root / "keypoints.csv").open(newline="") as handle:
            keypoints = list(csv.reader(handle))
        self.assertEqual(objects[0], OBJECT_HEADERS)
        self.assertEqual(objects[1], [str(value) for value in object_row])
        self.assertEqual(keypoints[0], KEYPOINT_HEADERS)
        self.assertEqual(keypoints[1], [str(value) for value in keypoint_row])

    def test_rejects_schema_mismatch_before_writing(self) -> None:
        writer = PoseCsvWriter(self.root)
        with self.assertRaisesRegex(ValueError, "expected 24"):
            writer.write_object(["too", "short"])
        with self.assertRaisesRegex(ValueError, "expected 18"):
            writer.write_keypoints([["too", "short"]])
        writer.close()

        with (self.root / "objects.csv").open(newline="") as handle:
            self.assertEqual(len(list(csv.reader(handle))), 1)
        with (self.root / "keypoints.csv").open(newline="") as handle:
            self.assertEqual(len(list(csv.reader(handle))), 1)

    def test_write_after_close_fails_closed(self) -> None:
        writer = PoseCsvWriter(self.root)
        writer.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            writer.write_object(list(range(len(OBJECT_HEADERS))))


if __name__ == "__main__":
    unittest.main()
