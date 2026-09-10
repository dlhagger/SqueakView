from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from squeakview.apps.inference.mp4_sample_table import read_video_sample_table


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


def _fixture(*, samples: int = 3, timing_samples: int | None = None, bad_offset: bool = False) -> bytes:
    ftyp = _box(b"ftyp", b"isom\0\0\0\0isom")
    mdat = _box(b"mdat", b"\0" * 32)
    chunk_offset = 1 if bad_offset else len(ftyp) + 8
    stsz = _box(
        b"stsz",
        b"\0\0\0\0" + struct.pack(">II", 1, samples),
    )
    stts = _box(
        b"stts",
        b"\0\0\0\0"
        + struct.pack(">I", 1)
        + struct.pack(">II", samples if timing_samples is None else timing_samples, 1),
    )
    stsc = _box(
        b"stsc",
        b"\0\0\0\0" + struct.pack(">I", 1) + struct.pack(">III", 1, samples, 1),
    )
    stco = _box(
        b"stco",
        b"\0\0\0\0" + struct.pack(">II", 1, chunk_offset),
    )
    stbl = _box(b"stbl", stsz + stts + stsc + stco)
    minf = _box(b"minf", stbl)
    hdlr = _box(b"hdlr", b"\0\0\0\0" + b"\0\0\0\0" + b"vide")
    mdia = _box(b"mdia", hdlr + minf)
    trak = _box(b"trak", mdia)
    moov = _box(b"moov", trak)
    return ftyp + mdat + moov


class Mp4SampleTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "raw.mp4"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_matching_video_tables_return_exact_sample_count(self) -> None:
        self.path.write_bytes(_fixture(samples=7))

        table = read_video_sample_table(self.path)

        self.assertEqual(table.sample_count, 7)
        self.assertEqual(table.timing_sample_count, 7)
        self.assertEqual(table.chunk_sample_count, 7)
        self.assertEqual(table.chunk_count, 1)

    def test_disagreeing_timing_table_is_rejected(self) -> None:
        self.path.write_bytes(_fixture(samples=7, timing_samples=6))

        with self.assertRaisesRegex(ValueError, "sample tables disagree"):
            read_video_sample_table(self.path)

    def test_chunk_outside_media_data_is_rejected(self) -> None:
        self.path.write_bytes(_fixture(samples=7, bad_offset=True))

        with self.assertRaisesRegex(ValueError, "outside every media-data box"):
            read_video_sample_table(self.path)

    def test_truncated_box_is_rejected(self) -> None:
        self.path.write_bytes(struct.pack(">I4s", 100, b"moov") + b"short")

        with self.assertRaisesRegex(ValueError, "invalid MP4 box size"):
            read_video_sample_table(self.path)


if __name__ == "__main__":
    unittest.main()
