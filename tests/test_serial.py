from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from squeakview.common import serial as serial_util


class SerialCsvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp_dir.name) / "run"
        self.logs: list[str] = []
        self.handle = serial_util.SerialHandle("/dev/test", 115200, self.logs.append)

    def tearDown(self) -> None:
        self.handle.close()
        self.temp_dir.cleanup()

    def rows(self) -> list[dict[str, str]]:
        path = self.run_dir / "serial.csv"
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))

    def test_rows_buffer_until_run_directory_is_available(self) -> None:
        with (
            mock.patch.object(serial_util.time, "time_ns", return_value=101),
            mock.patch.object(serial_util.time, "monotonic_ns", return_value=202),
        ):
            self.handle._write_csv_line("POKE_START,10,20,L,1,30,40,50,Eligible,nan")
        self.assertEqual(len(self.handle._buffer_rows), 1)

        self.handle.set_csv_path(self.run_dir)
        self.handle.close()

        rows = self.rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["eventType"], "POKE_START")
        self.assertEqual(rows[0]["side"], "L")
        self.assertEqual(rows[0]["hostUnixNs"], "101")
        self.assertEqual(rows[0]["hostMonotonicNs"], "202")
        self.assertEqual(rows[0]["rawLine"], "POKE_START,10,20,L,1,30,40,50,Eligible,nan")

    def test_short_and_extra_rows_keep_a_stable_csv_schema(self) -> None:
        self.handle.set_csv_path(self.run_dir)
        with (
            mock.patch.object(serial_util.time, "time_ns", return_value=11),
            mock.patch.object(serial_util.time, "monotonic_ns", return_value=12),
        ):
            self.handle._write_csv_line("SYSTEM_START,123,456")
            self.handle._write_csv_line("EVENT,1,2,L,3,4,5,6,CTX,reason,with,commas")
        self.handle.close()

        rows = self.rows()
        self.assertEqual(set(rows[0]), set(serial_util.SERIAL_HEADER))
        self.assertEqual(rows[0]["eventType"], "SYSTEM_START")
        self.assertEqual(rows[0]["side"], "")
        self.assertEqual(rows[0]["reason"], "")
        self.assertEqual(rows[1]["reason"], "reason,with,commas")
        self.assertEqual(rows[1]["rawLine"], "EVENT,1,2,L,3,4,5,6,CTX,reason,with,commas")

    def test_host_marker_uses_reason_field_and_raw_line(self) -> None:
        self.handle.set_csv_path(self.run_dir)
        self.handle.log_marker("CAPTURE_STOP_REQUESTED")
        self.handle.close()

        row = self.rows()[0]
        self.assertEqual(row["eventType"], "MARKER")
        self.assertEqual(row["context"], "HOST")
        self.assertEqual(row["reason"], "CAPTURE_STOP_REQUESTED")
        self.assertEqual(row["rawLine"], "MARKER,CAPTURE_STOP_REQUESTED")

    def test_open_failure_preserves_original_error(self) -> None:
        fake_serial_module = mock.Mock()
        fake_serial_module.Serial.side_effect = PermissionError(13, "Permission denied", "/dev/test")
        with mock.patch.object(serial_util, "serial", fake_serial_module):
            self.assertFalse(self.handle.open(self.run_dir))

        self.assertIn("Permission denied", self.handle.last_error or "")
        self.assertTrue(any("ERROR opening serial" in line for line in self.logs))

    def test_ack_stop_is_persisted_before_wait_releases(self) -> None:
        self.handle.set_csv_path(self.run_dir)

        class FakeSerialPort:
            is_open = True

            def __init__(self, handle: serial_util.SerialHandle):
                self.handle = handle
                self.reads = 0

            def read(self, _size: int) -> bytes:
                self.reads += 1
                if self.reads == 1:
                    return b"ACK_STOP,1,2,nan,3,4,5,6,Eligible,nan\n"
                self.handle._stop.set()
                return b""

        self.handle.ser = FakeSerialPort(self.handle)
        self.handle._closed = False
        self.handle._pump()

        self.assertTrue(self.handle.wait_for_stop_ack(timeout_s=0))
        self.handle.close()
        rows = self.rows()
        self.assertEqual(rows[-1]["eventType"], "ACK_STOP")
        self.assertEqual(rows[-1]["count"], "3")
        self.assertEqual(self.handle.stop_ack_count, 3)

    def test_send_line_raises_when_port_is_not_open(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "serial port is not open"):
            self.handle.send_line("START,30")

    def test_start_clears_stale_ttl_before_writing_and_flushes(self) -> None:
        port = mock.Mock()
        port.is_open = True
        self.handle.ser = port
        self.handle._ttl_seen.set()

        self.handle.send_line("START,30")

        self.assertFalse(self.handle._ttl_seen.is_set())
        port.write.assert_called_once_with(b"START,30\n")
        port.flush.assert_called_once_with()

    def test_send_line_raises_when_device_write_fails(self) -> None:
        port = mock.Mock()
        port.is_open = True
        port.write.side_effect = OSError("device disconnected")
        self.handle.ser = port

        with self.assertRaisesRegex(RuntimeError, "device disconnected"):
            self.handle.send_line("START,30")

        self.assertTrue(any("write error" in line for line in self.logs))


if __name__ == "__main__":
    unittest.main()
