"""The canonical v2 journal remains durable without retaining every event in RAM."""
from dataclasses import replace
import gc
import json
from pathlib import Path
import sqlite3
import tempfile
import tracemalloc
import unittest
from unittest import mock

from squeakview.common import controller_protocol_v2 as v2


def frame(sequence, *, boot_id=9, payload=None, flags=v2.FLAG_RELIABLE):
    return v2.decode_frame(v2.encode_frame(
        message_type=v2.MessageType.EVENT, flags=flags, boot_id=boot_id,
        session_id=3, sequence=sequence, monotonic_us=sequence,
        payload=payload if payload is not None else f"EVENT,{sequence}".encode(),
    )[:-1])


class JournalDiskIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "controller_v2.jsonl"
        self.journal = v2.DurableJournal(self.path)
        self.addCleanup(self.journal.close)

    def store(self, value):
        return self.journal.store(value, host_unix_ns=1, host_monotonic_ns=2)

    def test_fifteen_thousand_records_keep_python_and_page_cache_bounded(self):
        with mock.patch.object(v2.os, "fsync", new=lambda _fd: None):
            tracemalloc.start()
            try:
                for index in range(1, 1001):
                    self.store(frame(index, payload=(f"{index:08}" + "x" * 350).encode()))
                gc.collect()
                baseline = tracemalloc.get_traced_memory()[0]
                for index in range(1001, 15001):
                    self.store(frame(index, payload=(f"{index:08}" + "x" * 350).encode()))
                gc.collect()
                growth = tracemalloc.get_traced_memory()[0] - baseline
                self.assertLess(growth, 512 * 1024)
            finally:
                tracemalloc.stop()
            self.assertEqual(self.journal._index.execute("PRAGMA cache_size").fetchone()[0], -512)
            self.assertEqual(self.journal._index.execute("PRAGMA synchronous").fetchone()[0], 0)
            self.assertEqual(self.journal._index_path.parent, self.path.parent)
            self.assertGreater(self.journal._index_path.stat().st_size, 512 * 1024)
            self.assertEqual(self.journal.contiguous(9), 15000)
            old = frame(1, payload=("00000001" + "x" * 350).encode())
            self.assertEqual(self.store(old), (v2.StoreResult.DUPLICATE, 15000))
            self.assertEqual(self.store(replace(old, flags=old.flags | v2.FLAG_RETRANSMISSION)), (v2.StoreResult.DUPLICATE, 15000))
            self.assertEqual(self.store(replace(old, payload=b"changed")), (v2.StoreResult.CONFLICT, 15000))

    def test_json_fsync_precedes_index_update_and_watermark(self):
        observed = []
        def sync(_fd):
            observed.append(self.journal._index.execute("SELECT count(*) FROM records").fetchone()[0])
            saved = json.loads(self.path.read_text())
            self.assertEqual(saved["sequence"], 1)
        with mock.patch.object(v2.os, "fsync", side_effect=sync) as fsync:
            self.assertEqual(self.store(frame(1)), (v2.StoreResult.NEW, 1))
        self.assertEqual(observed, [0])
        fsync.assert_called_once()

    def test_failed_fsync_does_not_publish_index_or_watermark(self):
        with mock.patch.object(v2.os, "fsync", side_effect=OSError("disk flush failed")):
            with self.assertRaisesRegex(OSError, "disk flush"):
                self.store(frame(1))
        self.assertEqual(self.journal.contiguous(9), 0)
        self.assertEqual(self.journal._index.execute("SELECT count(*) FROM records").fetchone()[0], 0)

    def test_gaps_and_uint64_keys_survive_reopen(self):
        boot = (1 << 64) - 1
        self.journal.close()
        self.journal = v2.DurableJournal(self.path, allow_midstream_start=True)
        self.addCleanup(self.journal.close)
        start = (1 << 63) + 42
        self.assertEqual(self.store(frame(start, boot_id=boot))[1], start)
        self.assertEqual(self.store(frame(start + 2, boot_id=boot))[1], start)
        self.assertEqual(self.store(frame(start + 1, boot_id=boot))[1], start + 2)
        self.assertEqual(self.journal.start_sequence(boot), start)
        self.journal.close()
        self.journal = v2.DurableJournal(self.path, allow_midstream_start=True)
        self.addCleanup(self.journal.close)
        self.assertEqual(self.journal.contiguous(boot), start + 2)
        self.assertEqual(self.store(frame(start, boot_id=boot))[0], v2.StoreResult.DUPLICATE)
        self.assertEqual(self.store(frame((1 << 64) - 1, boot_id=boot))[1], start + 2)

    def test_midstream_reopen_preserves_original_baseline_after_old_replay(self):
        self.journal.close()
        self.journal = v2.DurableJournal(self.path, allow_midstream_start=True)
        self.addCleanup(self.journal.close)
        self.assertEqual(self.store(frame(42))[1], 42)
        self.assertEqual(self.store(frame(1))[1], 42)
        self.assertEqual(self.journal.start_sequence(9), 1)
        self.journal.close()
        self.journal = v2.DurableJournal(self.path, allow_midstream_start=True)
        self.addCleanup(self.journal.close)
        self.assertEqual(self.journal.contiguous(9), 42)
        self.assertEqual(self.journal.start_sequence(9), 1)

    def test_cache_error_after_canonical_fsync_requires_reopen(self):
        connection = self.journal._index
        class BrokenIndex:
            def execute(self, statement, parameters=()):
                if statement.startswith("INSERT INTO records"):
                    raise sqlite3.DatabaseError("index disk failure")
                return connection.execute(statement, parameters)
            def __getattr__(self, name):
                return getattr(connection, name)
        self.journal._index = BrokenIndex()
        with self.assertRaisesRegex(sqlite3.DatabaseError, "index disk failure"):
            self.store(frame(1))
        with self.assertRaisesRegex(RuntimeError, "reopen"):
            self.store(frame(2))
        with self.assertRaisesRegex(RuntimeError, "reopen"):
            self.journal.contiguous(9)
        self.journal.close()
        self.journal = v2.DurableJournal(self.path)
        self.addCleanup(self.journal.close)
        self.assertEqual(self.journal.contiguous(9), 1)
        self.assertEqual(self.store(frame(1))[0], v2.StoreResult.DUPLICATE)

    def test_many_boots_and_maximum_sequence_remain_exact_after_reopen(self):
        self.journal.close()
        self.journal = v2.DurableJournal(self.path, allow_midstream_start=True)
        self.addCleanup(self.journal.close)
        maximum = (1 << 64) - 1
        with mock.patch.object(v2.os, "fsync", new=lambda _fd: None):
            for boot in range(1, 1001):
                self.assertEqual(self.store(frame(1, boot_id=boot))[1], 1)
            self.assertEqual(self.store(frame(maximum, boot_id=maximum))[1], maximum)
        self.journal.close()
        self.journal = v2.DurableJournal(self.path, allow_midstream_start=True)
        self.addCleanup(self.journal.close)
        self.assertEqual(self.journal.contiguous(maximum), maximum)
        self.assertEqual(self.journal.start_sequence(maximum), maximum)
        self.assertEqual(self.store(frame(1, boot_id=1, payload=b"conflicting old event"))[0], v2.StoreResult.CONFLICT)
        self.assertEqual(self.journal.contiguous(1000), 1)

    def test_cache_lookup_corruption_never_appends_or_returns_a_watermark(self):
        self.store(frame(1))
        connection = self.journal._index
        size = self.path.stat().st_size
        class CorruptIndex:
            def execute(self, *_args, **_kwargs):
                raise sqlite3.DatabaseError("database disk image is malformed")
            def __getattr__(self, name):
                return getattr(connection, name)
        self.journal._index = CorruptIndex()
        with self.assertRaisesRegex(sqlite3.DatabaseError, "malformed"):
            self.store(frame(2))
        self.assertEqual(self.path.stat().st_size, size)
        with self.assertRaisesRegex(RuntimeError, "reopen"):
            self.journal.start_sequence(9)

    def test_invalid_canonical_journal_does_not_leave_an_index_file(self):
        bad = self.path.parent / "invalid.jsonl"
        bad.write_text("not JSON\n")
        with self.assertRaises(v2.ProtocolV2Error):
            v2.DurableJournal(bad)
        self.assertEqual(list(self.path.parent.glob(".invalid.jsonl.index-*")), [])

    def test_disposable_index_removed_on_close_and_canonical_rebuilds_it(self):
        self.store(frame(1))
        index_path = self.journal._index_path
        self.assertTrue(index_path.exists())
        self.journal.close()
        self.assertFalse(index_path.exists())
        index_path.write_bytes(b"stale cache is not canonical evidence")
        self.journal = v2.DurableJournal(self.path)
        self.addCleanup(self.journal.close)
        self.assertNotEqual(self.journal._index_path, index_path)
        self.assertEqual(self.journal.contiguous(9), 1)


if __name__ == "__main__":
    unittest.main()
