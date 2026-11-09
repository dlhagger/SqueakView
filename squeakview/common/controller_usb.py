"""Bounded pre-run USB recovery and exclusion for the dedicated controller port."""
from __future__ import annotations

import fcntl
import os
import re
from pathlib import Path
import stat
import subprocess

from serial.tools import list_ports

USB_RECOVERY_HELPER = Path('/usr/local/libexec/squeakview-usb-recovery')
SESSION_LOCK = Path('/run/lock/squeakview-controller-usb.lock')


def supported_ports():
    return [port for port in list_ports.comports()
            if (port.vid, port.pid) == (0x239a, 0x815d)]


def is_controller_port(path: str) -> bool:
    resolved = Path(path).resolve()
    return any(Path(port.device).resolve() == resolved for port in supported_ports())


def acquire_session_guard():
    try:
        descriptor = os.open(SESSION_LOCK, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        if USB_RECOVERY_HELPER.exists():
            raise RuntimeError('USB session guard missing; rerun scripts/setup_jetson.sh')
        return None  # Existing installations without automated USB recovery.
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != 0:
            raise RuntimeError('USB session guard must be a root-owned regular file')
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def prepare_port(requested: str, emit) -> str:
    """Caller must own the acquisition lock before invoking recovery."""
    if Path(requested).is_char_device():
        return requested
    if re.fullmatch(r'/dev/ttyACM\d+|/dev/serial/by-id/usb-Adafruit_Feather_RP2040_Adalogger_[A-Za-z0-9_-]+-if00', requested) is None:
        # Custom transports keep their existing preflight/open behavior. The
        # Jetson USB-C reset is restricted to the dedicated controller paths.
        return requested
    if not USB_RECOVERY_HELPER.is_file():
        emit('[USB] Automatic recovery is not installed; rerun scripts/setup_jetson.sh to enable it.')
        return requested
    emit('[USB] Controller missing; checking and recovering the dedicated USB-C port.')
    result = subprocess.run(['sudo', '-n', str(USB_RECOVERY_HELPER)],
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, timeout=55, check=False)
    for line in result.stdout.splitlines():
        emit(f'[USB] {line}')
    if result.returncode:
        raise RuntimeError('Controller USB recovery failed; rerun setup or inspect the USB connection.')
    if Path(requested).is_char_device():
        return requested
    # A by-id selection identifies one specific board and must never fall back
    # to a different one. ttyACM numbering, however, can change on enumeration.
    if str(Path(requested).parent) == '/dev/serial/by-id':
        raise RuntimeError(f'Selected controller did not return: {requested}')
    ports = supported_ports()
    if len(ports) != 1:
        raise RuntimeError('Recovered controller is ambiguous or absent; select its stable by-id port.')
    emit(f'[USB] Controller re-enumerated as {ports[0].device}')
    return str(ports[0].device)
