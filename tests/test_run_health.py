from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from squeakview.apps.operator.gui.run_health import read_latest_system_telemetry


class RunHealthTests(unittest.TestCase):
    def test_reads_latest_complete_row_with_quoted_raw_line(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "system.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["schema_version", "sample_index", "ram_pct", "raw_line"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "schema_version": "1.0",
                        "sample_index": 0,
                        "ram_pct": 20,
                        "raw_line": "RAM 20, GPU 5",
                    }
                )
                writer.writerow(
                    {
                        "schema_version": "1.0",
                        "sample_index": 1,
                        "ram_pct": 21,
                        "raw_line": "RAM 21, GPU 6",
                    }
                )

            row = read_latest_system_telemetry(path)

        assert row is not None
        self.assertEqual(row["sample_index"], "1")
        self.assertEqual(row["raw_line"], "RAM 21, GPU 6")

    def test_ignores_incomplete_concurrent_tail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "system.csv"
            path.write_bytes(
                b"schema_version,sample_index,ram_pct\n"
                b"1.0,0,20\n"
                b"1.0,1,"
            )

            row = read_latest_system_telemetry(path)

        assert row is not None
        self.assertEqual(row["sample_index"], "0")

    def test_missing_or_header_only_file_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "system.csv"
            self.assertIsNone(read_latest_system_telemetry(path))
            path.write_text("schema_version,sample_index\n")
            self.assertIsNone(read_latest_system_telemetry(path))


if __name__ == "__main__":
    unittest.main()
