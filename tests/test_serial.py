from __future__ import annotations

import csv
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from squeakview.common import serial as serial_util
from squeakview.common.failure_injection import FailurePlan


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

    def use_fatal_callback(self) -> list[str]:
        self.handle.close()
        failures: list[str] = []
        self.handle = serial_util.SerialHandle(
            "/dev/test", 115200, self.logs.append, on_fatal=failures.append
        )
        return failures

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

            def close(self) -> None:
                self.is_open = False

        self.handle.ser = FakeSerialPort(self.handle)
        self.handle._closed = False
        self.handle._pump()

        self.assertTrue(self.handle.wait_for_stop_ack(timeout_s=0))
        self.handle.close()
        rows = self.rows()
        self.assertEqual(rows[-1]["eventType"], "ACK_STOP")
        self.assertEqual(rows[-1]["count"], "3")
        self.assertEqual(self.handle.stop_ack_count, 3)
        self.assertIsNone(self.handle.fatal_error)

    def test_clear_feeder_jam_writes_exact_command_and_returns_ack(self) -> None:
        class ReplyingPort:
            is_open = True

            def __init__(self, handle: serial_util.SerialHandle) -> None:
                self.handle = handle
                self.writes: list[bytes] = []

            def write(self, payload: bytes) -> None:
                self.writes.append(payload)
                self.handle._ingest_clear_jam_response("ACK_CLEAR_JAM")

            def flush(self) -> None:
                pass

            def close(self) -> None:
                self.is_open = False

        port = ReplyingPort(self.handle)
        self.handle.ser = port
        self.handle._closed = False

        self.assertEqual(self.handle.clear_feeder_jam(), "ACK_CLEAR_JAM")
        self.assertEqual(port.writes, [b"CLEAR_JAM\n"])

    def test_clear_feeder_jam_returns_each_firmware_nack(self) -> None:
        for reply in (
            "NACK,CLEAR_JAM,FEED_ACTIVE",
            "NACK,CLEAR_JAM,NOT_JAMMED",
        ):
            with self.subTest(reply=reply):
                handle = serial_util.SerialHandle("/dev/test", 115200, self.logs.append)

                class ReplyingPort:
                    is_open = True

                    def write(self, _payload: bytes) -> None:
                        handle._ingest_clear_jam_response(reply)

                    def flush(self) -> None:
                        pass

                    def close(self) -> None:
                        self.is_open = False

                handle.ser = ReplyingPort()
                handle._closed = False
                try:
                    self.assertEqual(handle.clear_feeder_jam(), reply)
                finally:
                    handle.close()

    def test_clear_feeder_jam_timeout_and_disconnect_are_failures(self) -> None:
        class SilentPort:
            is_open = True

            def write(self, _payload: bytes) -> None:
                pass

            def flush(self) -> None:
                pass

            def close(self) -> None:
                self.is_open = False

        self.handle.ser = SilentPort()
        self.handle._closed = False
        with self.assertRaises(TimeoutError):
            self.handle.clear_feeder_jam(timeout_s=0.01)

        def disconnect() -> None:
            time.sleep(0.01)
            self.handle._interrupt_clear_jam("USB disconnected")

        thread = threading.Thread(target=disconnect)
        thread.start()
        with self.assertRaisesRegex(ConnectionError, "USB disconnected"):
            self.handle.clear_feeder_jam(timeout_s=1.0)
        thread.join()

    def test_clear_feeder_jam_rejects_duplicate_pending_command(self) -> None:
        wrote = threading.Event()

        class SilentPort:
            is_open = True

            def __init__(self) -> None:
                self.writes: list[bytes] = []

            def write(self, payload: bytes) -> None:
                self.writes.append(payload)
                wrote.set()

            def flush(self) -> None:
                pass

            def close(self) -> None:
                self.is_open = False

        port = SilentPort()
        self.handle.ser = port
        self.handle._closed = False
        result: list[str] = []

        def first_request() -> None:
            result.append(self.handle.clear_feeder_jam(timeout_s=1.0))

        thread = threading.Thread(target=first_request)
        thread.start()
        self.assertTrue(wrote.wait(1.0))
        with self.assertRaisesRegex(RuntimeError, "already awaiting"):
            self.handle.clear_feeder_jam(timeout_s=1.0)
        self.handle._ingest_clear_jam_response("ACK_CLEAR_JAM")
        thread.join(1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(result, ["ACK_CLEAR_JAM"])
        self.assertEqual(port.writes, [b"CLEAR_JAM\n"])

    def test_clear_feeder_jam_write_failure_is_not_success(self) -> None:
        class BrokenPort:
            is_open = True

            def write(self, _payload: bytes) -> None:
                raise OSError("write failed")

            def flush(self) -> None:
                pass

            def close(self) -> None:
                self.is_open = False

        self.handle.ser = BrokenPort()
        self.handle._closed = False
        with self.assertRaisesRegex(RuntimeError, "serial write failed"):
            self.handle.clear_feeder_jam()

    def test_jam_protocol_reaches_event_path_when_routine_logging_is_disabled(self) -> None:
        class JamPort:
            is_open = True

            def __init__(self, handle: serial_util.SerialHandle) -> None:
                self.handle = handle
                self.sent = False

            def read(self, _size: int) -> bytes:
                if not self.sent:
                    self.sent = True
                    return (
                        b"CAMERA_HIGH,1,2,nan,1\n"
                        b"FEED_JAM,10,20,nan,1,69420,69420,69420,Feeding,stuck\n"
                    )
                self.handle._stop.set()
                return b""

            def close(self) -> None:
                self.is_open = False

        self.handle._emit_serial_logs = False
        self.handle.ser = JamPort(self.handle)
        self.handle._closed = False
        self.handle._pump()

        self.assertFalse(any("CAMERA_HIGH" in line for line in self.logs))
        self.assertTrue(any("FEED_JAM" in line for line in self.logs))

    def test_only_camera_high_confirms_trigger_readiness(self) -> None:
        self.handle.set_csv_path(self.run_dir)

        class StatusThenEdgePort:
            is_open = True

            def __init__(self, handle: serial_util.SerialHandle):
                self.handle = handle
                self.reads = 0

            def read(self, _size: int) -> bytes:
                self.reads += 1
                if self.reads == 1:
                    return b"CAMERA_CONFIG,30\nCAMERA_LOW,1\n"
                if self.reads == 2:
                    return b"CAMERA_HIGH,2\n"
                self.handle._stop.set()
                return b""

            def close(self) -> None:
                self.is_open = False

        self.handle.ser = StatusThenEdgePort(self.handle)
        self.handle._closed = False
        self.handle._pump()

        self.assertTrue(self.handle.wait_for_ttl(timeout_s=0))
        self.assertEqual(
            [row["eventType"] for row in self.rows()],
            ["CAMERA_CONFIG", "CAMERA_LOW", "CAMERA_HIGH"],
        )

    def test_non_edge_camera_lines_do_not_confirm_trigger_readiness(self) -> None:
        self.handle.set_csv_path(self.run_dir)

        class StatusPort:
            is_open = True

            def __init__(self, handle: serial_util.SerialHandle):
                self.handle = handle
                self.sent = False

            def read(self, _size: int) -> bytes:
                if not self.sent:
                    self.sent = True
                    return b"CAMERA_CONFIG,30\nCAMERA_LOW,1\n"
                self.handle._stop.set()
                return b""

            def close(self) -> None:
                self.is_open = False

        self.handle.ser = StatusPort(self.handle)
        self.handle._closed = False
        self.handle._pump()

        self.assertFalse(self.handle.wait_for_ttl(timeout_s=0))

    def test_capability_parser_is_strict_and_legacy_lines_are_ignored(self) -> None:
        parsed = serial_util.parse_controller_capabilities(
            "CONTROLLER_CAPS,1,0123456789abcdef0123456789abcdef,watchdog-1.2.3,1500,"
            "lease_watchdog;trigger_low_failsafe"
        )

        self.assertEqual(parsed.protocol_version, 1)
        self.assertEqual(parsed.firmware_version, "watchdog-1.2.3")
        self.assertEqual(parsed.watchdog_timeout_ms, 1500)
        self.assertEqual(
            parsed.features, frozenset({"lease_watchdog", "trigger_low_failsafe"})
        )
        self.assertIsNone(
            serial_util.parse_controller_capabilities("CAMERA_HIGH,1,2,nan,1")
        )
        for malformed in (
            "CONTROLLER_CAPS,1,0123456789abcdef0123456789abcdef,firmware,99,lease_watchdog",
            "CONTROLLER_CAPS,one,firmware,1500,lease_watchdog",
            "CONTROLLER_CAPS,1,0123456789abcdef0123456789abcdef,firmware,1500,duplicate;duplicate",
            "CONTROLLER_CAPS,1,0123456789abcdef0123456789abcdef,firmware,1500,feature,extra",
        ):
            with self.subTest(malformed=malformed):
                with self.assertRaises(ValueError):
                    serial_util.parse_controller_capabilities(malformed)

    def test_pump_records_passive_capabilities_without_changing_legacy_flow(self) -> None:
        self.handle.set_csv_path(self.run_dir)

        class CapabilityPort:
            is_open = True

            def __init__(self, handle: serial_util.SerialHandle):
                self.handle = handle
                self.sent = False

            def read(self, _size: int) -> bytes:
                if not self.sent:
                    self.sent = True
                    return (
                        b"CONTROLLER_CAPS,1,0123456789abcdef0123456789abcdef,"
                        b"watchdog-1.2.3,1500,"
                        b"lease_watchdog;trigger_low_failsafe\n"
                        b"CAMERA_HIGH,1,2,nan,1\n"
                    )
                self.handle._stop.set()
                return b""

            def close(self) -> None:
                self.is_open = False

        self.handle.ser = CapabilityPort(self.handle)
        self.handle._closed = False
        self.handle._pump()

        self.assertEqual(self.handle.controller_capabilities.protocol_version, 1)
        self.assertTrue(self.handle.wait_for_ttl(timeout_s=0))
        self.assertEqual(
            [row["eventType"] for row in self.rows()],
            ["CONTROLLER_CAPS", "CAMERA_HIGH"],
        )
        self.assertIsNone(self.handle.fatal_error)

    def test_malformed_capability_advertisement_is_logged_not_fatal(self) -> None:
        self.handle.set_csv_path(self.run_dir)

        class MalformedPort:
            is_open = True

            def __init__(self, handle: serial_util.SerialHandle):
                self.handle = handle
                self.sent = False

            def read(self, _size: int) -> bytes:
                if not self.sent:
                    self.sent = True
                    return b"CONTROLLER_CAPS,broken\n"
                self.handle._stop.set()
                return b""

        self.handle.ser = MalformedPort(self.handle)
        self.handle._closed = False
        self.handle._pump()

        self.assertIsNone(self.handle.controller_capabilities)
        self.assertIsNone(self.handle.fatal_error)
        self.assertTrue(any("ignored malformed" in line for line in self.logs))

    def test_runtime_reader_failure_is_fatal_and_is_not_retried(self) -> None:
        failures = self.use_fatal_callback()
        self.handle.set_csv_path(self.run_dir)

        port = mock.Mock()
        port.is_open = True
        port.read.side_effect = OSError("USB device disconnected")
        self.handle.ser = port
        self.handle._closed = False
        self.handle._stop.clear()

        self.handle._pump()

        self.assertEqual(port.read.call_count, 1)
        self.assertEqual(len(failures), 1)
        self.assertIn("USB device disconnected", failures[0])
        self.assertEqual(self.handle.fatal_error, failures[0])
        self.assertTrue(self.handle._stop.is_set())

    def test_watchdog_fault_keeps_reader_alive_for_ordered_disarm(self) -> None:
        failures = self.use_fatal_callback()
        self.handle._stop.clear()

        self.handle._report_watchdog_fatal("two heartbeat acknowledgements missed")

        self.assertEqual(failures, ["two heartbeat acknowledgements missed"])
        self.assertEqual(self.handle.fatal_error, failures[0])
        self.assertFalse(self.handle._stop.is_set())

    def test_unexpected_reader_exit_from_closed_port_is_fatal(self) -> None:
        failures = self.use_fatal_callback()
        self.handle.set_csv_path(self.run_dir)

        port = mock.Mock()
        port.is_open = False
        self.handle.ser = port
        self.handle._closed = False
        self.handle._stop.clear()

        self.handle._pump()

        self.assertEqual(len(failures), 1)
        self.assertIn("port closed unexpectedly", failures[0])

    def test_csv_write_failure_latches_one_fatal_error(self) -> None:
        failures = self.use_fatal_callback()
        self.handle.set_csv_path(self.run_dir)
        writer = mock.Mock()
        writer.writerow.side_effect = OSError("No space left on device")
        self.handle._csv_writer = writer

        self.handle._write_csv_line(
            "CAMERA_HIGH,1,2,nan,1,3,4,5,Eligible,nan"
        )
        self.handle._write_csv_line(
            "CAMERA_LOW,1,2,nan,1,3,4,5,Eligible,nan"
        )

        self.assertEqual(writer.writerow.call_count, 2)
        self.assertEqual(len(failures), 1)
        self.assertIn("No space left on device", failures[0])
        self.assertEqual(self.handle.fatal_error, failures[0])
        self.assertTrue(self.handle._stop.is_set())

    def test_csv_periodic_flush_failure_is_fatal(self) -> None:
        failures = self.use_fatal_callback()
        self.handle.set_csv_path(self.run_dir)
        original_file = self.handle._csv_file

        class FailingFlushFile:
            def flush(self) -> None:
                raise OSError("I/O error")

            def close(self) -> None:
                original_file.close()

        self.handle._csv_file = FailingFlushFile()
        self.handle._flush_every = 1

        self.handle._write_csv_line(
            "CAMERA_HIGH,1,2,nan,1,3,4,5,Eligible,nan"
        )

        self.assertEqual(len(failures), 1)
        self.assertIn("CSV write/flush failed", failures[0])
        self.assertIn("I/O error", failures[0])
        self.assertTrue(self.handle._stop.is_set())

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

    def test_send_start_persists_epoch_marker_before_controller_write(self) -> None:
        order: list[str] = []
        port = mock.Mock()
        port.is_open = True
        port.write.side_effect = lambda _payload: order.append("write")
        self.handle.ser = port
        original_marker = self.handle.log_marker

        def marker(name: str) -> None:
            order.append(name)
            original_marker(name)

        self.handle.log_marker = marker
        self.handle.send_start(30)

        self.assertEqual(order, ["START_SENT", "write"])
        port.write.assert_called_once_with(b"START,30\n")

    def test_send_line_raises_when_device_write_fails(self) -> None:
        port = mock.Mock()
        port.is_open = True
        port.write.side_effect = OSError("device disconnected")
        self.handle.ser = port

        with self.assertRaisesRegex(RuntimeError, "device disconnected"):
            self.handle.send_line("START,30")

        self.assertTrue(any("write error" in line for line in self.logs))

    def test_injected_ledger_failure_occurs_after_exact_row_count(self) -> None:
        self.handle.close()
        failures: list[str] = []
        self.handle = serial_util.SerialHandle(
            "/dev/test",
            115200,
            self.logs.append,
            on_fatal=failures.append,
            failure_plan=FailurePlan(
                "1.0", "serial_controller", "ledger_write_error", 1
            ),
        )
        self.handle.set_csv_path(self.run_dir)

        self.handle._write_csv_line("CAMERA_HIGH,1")
        self.handle._write_csv_line("CAMERA_LOW,2")

        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(len(failures), 1)
        self.assertIn("qualification-injected", failures[0])

    def test_injected_command_failure_occurs_after_exact_write_count(self) -> None:
        self.handle.close()
        self.handle = serial_util.SerialHandle(
            "/dev/test",
            115200,
            self.logs.append,
            failure_plan=FailurePlan(
                "1.0", "serial_controller", "write_error", 1
            ),
        )
        port = mock.Mock()
        port.is_open = True
        self.handle.ser = port

        self.handle.send_line("START,30")
        with self.assertRaisesRegex(RuntimeError, "qualification-injected"):
            self.handle.send_line("STOP")

        port.write.assert_called_once_with(b"START,30\n")

    def test_injected_reader_failure_occurs_after_exact_line_count(self) -> None:
        self.handle.close()
        failures: list[str] = []
        self.handle = serial_util.SerialHandle(
            "/dev/test",
            115200,
            self.logs.append,
            on_fatal=failures.append,
            failure_plan=FailurePlan(
                "1.0", "serial_controller", "read_error", 1
            ),
        )
        self.handle.set_csv_path(self.run_dir)

        class TwoLinePort:
            is_open = True

            def __init__(self) -> None:
                self.sent = False

            def read(self, _size: int) -> bytes:
                if not self.sent:
                    self.sent = True
                    return b"CAMERA_HIGH,1\nCAMERA_LOW,2\n"
                return b""

            def close(self) -> None:
                self.is_open = False

        self.handle.ser = TwoLinePort()
        self.handle._closed = False
        self.handle._stop.clear()
        self.handle._pump()

        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(len(failures), 1)
        self.assertIn("qualification-injected serial reader", failures[0])

    def test_reader_fails_closed_when_unterminated_line_exceeds_bound(self) -> None:
        failures = self.use_fatal_callback()
        self.handle.set_csv_path(self.run_dir)
        self.handle._max_line_bytes = 300

        class UnterminatedPort:
            is_open = True

            def read(self, _size: int) -> bytes:
                return b"x" * 256

        self.handle.ser = UnterminatedPort()
        self.handle._closed = False
        self.handle._stop.clear()
        self.handle._pump()

        self.assertEqual(len(failures), 1)
        self.assertIn("without a newline exceeded bounded size", failures[0])

    def test_preledger_row_buffer_is_bounded_and_fails_closed(self) -> None:
        failures = self.use_fatal_callback()
        self.handle._max_buffered_rows = 2

        self.handle._write_csv_line("CAMERA_HIGH,1")
        self.handle._write_csv_line("CAMERA_LOW,2")
        self.handle._write_csv_line("CAMERA_HIGH,3")

        self.assertEqual(len(self.handle._buffer_rows), 2)
        self.assertEqual(len(failures), 1)
        self.assertIn("bounded pre-ledger buffer reached 2 rows", failures[0])

    def test_serial_email_alert_worker_is_single_flight(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def blocked_delivery(_line: str) -> None:
            started.set()
            self.assertTrue(release.wait(timeout=2.0))

        with mock.patch.object(
            self.handle, "_send_email_alert", side_effect=blocked_delivery
        ) as delivery:
            self.handle._maybe_send_alert("Feeder jammed once")
            self.assertTrue(started.wait(timeout=1.0))
            for index in range(20):
                self.handle._maybe_send_alert(f"Feeder jammed again {index}")
            self.assertEqual(delivery.call_count, 1)
            self.assertEqual(self.handle._alert_suppressed, 20)
            release.set()
            deadline = time.monotonic() + 2.0
            while self.handle._alert_inflight and time.monotonic() < deadline:
                time.sleep(0.01)

        self.assertFalse(self.handle._alert_inflight)
        self.assertEqual(self.handle._alert_suppressed, 0)
        self.assertTrue(any("coalesced 20" in line for line in self.logs))

    def test_closed_serial_handle_does_not_remain_registered_at_exit(self) -> None:
        self.assertTrue(self.handle._atexit_registered)

        self.handle.close()

        self.assertFalse(self.handle._atexit_registered)

    def test_reused_serial_handle_restores_exit_cleanup_registration(self) -> None:
        self.handle.close()
        self.assertFalse(self.handle._atexit_registered)

        with mock.patch.object(serial_util.atexit, "register") as register:
            self.handle._register_atexit()

        self.assertTrue(self.handle._atexit_registered)
        register.assert_called_once_with(self.handle.close)

    def test_close_unblocks_port_before_joining_reader(self) -> None:
        order: list[str] = []

        class Port:
            is_open = True

            def close(self) -> None:
                order.append("port_close")
                self.is_open = False

        class ReaderThread:
            def is_alive(self) -> bool:
                return True

            def join(self, timeout: float) -> None:
                self.timeout = timeout
                order.append("thread_join")

        self.handle.ser = Port()
        self.handle._thread = ReaderThread()
        self.handle._closed = False

        self.handle.close()

        self.assertEqual(order, ["port_close", "thread_join"])
        self.assertIn("reader did not stop", self.handle.fatal_error or "")

    def test_port_close_error_is_a_fatal_integrity_failure(self) -> None:
        failures = self.use_fatal_callback()
        port = mock.Mock()
        port.is_open = True
        port.close.side_effect = OSError("USB close failed")
        self.handle.ser = port
        self.handle._closed = False

        self.handle.close()

        self.assertEqual(len(failures), 1)
        self.assertIn("serial port close failed", failures[0])
        self.assertIn("USB close failed", failures[0])


if __name__ == "__main__":
    unittest.main()
