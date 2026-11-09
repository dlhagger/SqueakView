#!/usr/bin/python3 -I
"""On-demand USB-C enumeration recovery for a dedicated Jetson Orin Nano rig.

Uses the system Python standard library; never opens the controller serial port.
Run only while the rig is idle, before uploading or starting acquisition.
"""

import argparse
import fcntl
import grp
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time


ROLE = Path('/sys/class/usb_role/usb2-0-role-switch/role')
USB_PORT = Path('/sys/bus/usb/devices/1-1')
INITIAL_WAIT = 2
RECOVERY_WAIT = 15
ATTEMPTS = 3
SESSION_LOCK = Path('/run/lock/squeakview-controller-usb.lock')


def acquire_session_guard():
    """Share the acquisition owner's USB exclusion lock; never reset in use."""
    descriptor = os.open(SESSION_LOCK, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o660)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != 0:
            raise RuntimeError('USB session guard must be a root-owned regular file')
        os.fchown(descriptor, 0, grp.getgrnam('dialout').gr_gid)
        os.fchmod(descriptor, 0o660)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Controller session is active; refusing USB recovery') from exc
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def log(message):
    print(message, flush=True)


class Controller:
    def __init__(self, serial=None):
        self.serial = serial
        self.device = Path('/dev/serial/by-id') / (
            f'usb-Adafruit_Feather_RP2040_Adalogger_{serial or "*"}-if00'
        )

    def ready(self):
        if self.serial is None:
            matches = [path for path in self.device.parent.glob(self.device.name)
                       if path.is_char_device()]
            if len(matches) > 1:
                raise RuntimeError('Multiple controllers found; use --serial to select one')
            if matches:
                self.device = matches[0]
                return True
            return False
        return self.device.is_char_device()

    def wait(self, seconds):
        deadline = time.monotonic() + seconds
        while True:
            if self.ready():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.5, remaining))

    def reset_role(self):
        # Recheck immediately before touching USB; a healthy device is a no-op.
        if self.ready():
            return
        if ROLE.read_text().strip() != 'host':
            raise RuntimeError('USB-C is not in host mode; refusing to force its role')
        if USB_PORT.exists():
            identity = tuple((USB_PORT / name).read_text().strip()
                             for name in ('idVendor', 'idProduct', 'serial'))
            if identity[:2] != ('239a', '815d') or (
                    self.serial is not None and identity[2] != self.serial):
                raise RuntimeError('USB-C is occupied by another device; refusing reset')
        # This affects USB-C only, not the shared xHCI driver or USB-A hub.
        try:
            ROLE.write_text('none\n')
            time.sleep(1)
        finally:
            ROLE.write_text('host\n')


def recover(controller, check_only=False):
    if controller.ready():
        log(f'USB_READY already present: {controller.device}')
        return 0
    if check_only:
        log(f'USB_NOT_READY missing: {controller.device}')
        return 1
    log(f'Waiting up to {INITIAL_WAIT}s for {controller.device}')
    if controller.wait(INITIAL_WAIT):
        log('USB_READY enumerated without recovery')
        return 0
    for attempt in range(1, ATTEMPTS + 1):
        log(f'USB_RECOVERY attempt={attempt}/{ATTEMPTS} role=host->none->host')
        controller.reset_role()
        if controller.wait(RECOVERY_WAIT):
            log(f'USB_READY recovered attempt={attempt}')
            return 0
    log(f'USB_FAILED expected controller absent after {ATTEMPTS} recovery attempts')
    return 1


def serial_number(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', value):
        raise argparse.ArgumentTypeError('invalid controller USB serial number')
    return value


def interrupted(signum, frame):
    raise RuntimeError(f'interrupted by signal {signum}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial', type=serial_number,
                        help='optional: select a specific controller by USB serial number')
    parser.add_argument('--check', action='store_true', help='read-only presence check')
    args = parser.parse_args(argv)
    controller = Controller(args.serial)
    try:
        if args.check or controller.ready():
            return recover(controller, check_only=True)
        if os.geteuid() != 0:
            raise RuntimeError('Recovery requires root; rerun with sudo. No USB reset performed.')
        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        descriptor = os.open('/run/lock/mousehouse-usb-recovery.lock',
                             os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, 'r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            guard = acquire_session_guard()
            try:
                return recover(controller)
            finally:
                os.close(guard)
    except (OSError, RuntimeError) as exc:
        log(f'USB_FAILED {exc}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
