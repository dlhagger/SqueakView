from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from squeakview.common.bounded_csv import BoundedCsvError, bounded_csv_lines


class BoundedCsvLinesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_yields_strict_utf8_lines_at_limit(self) -> None:
        path = self.root / "evidence.csv"
        path.write_bytes(b"abc\n" + b"x" * 4)

        self.assertEqual(
            list(bounded_csv_lines(path, max_record_bytes=4)),
            ["abc\n", "xxxx"],
        )

    def test_rejects_oversized_unterminated_record(self) -> None:
        path = self.root / "evidence.csv"
        path.write_bytes(b"header\n" + b"x" * 9)

        iterator = bounded_csv_lines(path, max_record_bytes=8)
        self.assertEqual(next(iterator), "header\n")
        with self.assertRaisesRegex(BoundedCsvError, "exceeds 8 byte limit"):
            next(iterator)

    def test_rejects_invalid_utf8(self) -> None:
        path = self.root / "evidence.csv"
        path.write_bytes(b"header\n\xff\n")

        iterator = bounded_csv_lines(path)
        self.assertEqual(next(iterator), "header\n")
        with self.assertRaisesRegex(BoundedCsvError, "not strict UTF-8"):
            next(iterator)

    def test_rejects_invalid_limit(self) -> None:
        path = self.root / "evidence.csv"
        path.write_text("a\n")

        for invalid in (0, -1, True, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    next(bounded_csv_lines(path, max_record_bytes=invalid))


if __name__ == "__main__":
    unittest.main()
