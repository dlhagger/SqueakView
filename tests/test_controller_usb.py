"""Recovery cannot reset an owned controller or silently select another board."""
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from squeakview.common import controller_usb as usb
from squeakview.common.serial import SerialHandle


class ControllerUsbTests(unittest.TestCase):
    def test_present_port_never_invokes_privileged_helper(self):
        with mock.patch.object(usb.subprocess, 'run') as run:
            self.assertEqual(usb.prepare_port('/dev/null', lambda _: None), '/dev/null')
            run.assert_not_called()

    def test_recovery_retargets_only_an_unambiguous_tty_not_stable_identity(self):
        board = SimpleNamespace(device='/dev/ttyACM7')
        with mock.patch.object(usb.USB_RECOVERY_HELPER.__class__, 'is_file', return_value=True), \
                mock.patch.object(usb, 'supported_ports', return_value=[board]), \
                mock.patch.object(usb.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'USB_READY')) as run:
            self.assertEqual(usb.prepare_port('/dev/ttyACM987654', lambda _: None), board.device)
            self.assertEqual(run.call_args.args[0], ['sudo', '-n', str(usb.USB_RECOVERY_HELPER)])
            with self.assertRaisesRegex(RuntimeError, 'Selected controller did not return'):
                usb.prepare_port('/dev/serial/by-id/usb-Adafruit_Feather_RP2040_Adalogger_MISSING-if00', lambda _: None)
            with mock.patch.object(usb, 'supported_ports', return_value=[board, board]):
                with self.assertRaisesRegex(RuntimeError, 'ambiguous'):
                    usb.prepare_port('/dev/ttyACM987654', lambda _: None)
            count = run.call_count
            self.assertEqual(usb.prepare_port('/dev/ttyUSB987654', lambda _: None), '/dev/ttyUSB987654')
            self.assertEqual(run.call_count, count)

    def test_failed_recovery_aborts(self):
        with mock.patch.object(Path, 'is_file', return_value=True), \
                mock.patch.object(usb.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1, 'USB_FAILED')):
            with self.assertRaisesRegex(RuntimeError, 'recovery failed'):
                usb.prepare_port('/dev/ttyACM987654', lambda _: None)

    def test_missing_guard_is_an_error_once_helper_is_installed(self):
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(usb, 'SESSION_LOCK', Path(directory) / 'missing'), \
                mock.patch.object(Path, 'exists', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'guard missing'):
                usb.acquire_session_guard()

    def test_guard_excludes_other_owners_until_descriptor_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = Path(directory) / 'lock'
            guard.touch()
            info = guard.stat()
            with mock.patch.object(usb, 'SESSION_LOCK', guard), \
                    mock.patch.object(usb.os, 'fstat', return_value=SimpleNamespace(st_uid=0, st_mode=info.st_mode)):
                descriptor = usb.acquire_session_guard()
                try:
                    with self.assertRaises(BlockingIOError):
                        usb.acquire_session_guard()
                finally:
                    os.close(descriptor)
                descriptor = usb.acquire_session_guard()
                os.close(descriptor)

    def test_open_failure_releases_guard_even_without_serial_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            descriptor = os.open(Path(directory) / 'lock', os.O_CREAT | os.O_RDWR, 0o600)
            handle = SerialHandle('/dev/test', 115200, lambda _: None)
            with mock.patch.object(usb, 'acquire_session_guard', return_value=descriptor), \
                    mock.patch('squeakview.common.serial.serial.Serial', side_effect=OSError('disconnected')):
                self.assertFalse(handle.open())
            self.assertIsNone(handle._usb_guard_fd)
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_idle_confirmation_rejects_stale_reply_and_busy_controller(self):
        handle = SerialHandle('/dev/test', 115200, lambda _: None)
        self.addCleanup(handle.close)
        def reply(sequence, sent_ns, **_):
            return f'CLOCK_SYNC,{sequence},{sent_ns},10,11,10000,RTC_VALID', sent_ns + 1000
        with mock.patch.object(handle, 'exchange_time_sync', side_effect=reply):
            handle.confirm_idle()
        for response in ('CLOCK_SYNC,0,1,10,11,10000,RTC_VALID', 'NACK,TIME_SYNC,DEVICE_BUSY'):
            with mock.patch.object(handle, 'exchange_time_sync', return_value=(response, 2)):
                with self.assertRaises(RuntimeError):
                    handle.confirm_idle()

    def test_installed_helper_source_matches_firmware_checkout(self):
        root = Path(__file__).resolve().parents[1]
        canonical = root.parent / 'SqueakView_Firmware/src/squeakview_firmware/usb_recovery.py'
        if not canonical.is_file():
            self.skipTest('companion firmware checkout is unavailable')
        self.assertEqual((root / 'scripts/jetson_usb_recovery.py').read_bytes(), canonical.read_bytes())
