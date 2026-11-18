"""Compare optional output progress with actual work available in mailboxes."""
from __future__ import annotations

import fcntl
import os
import struct


LIVE_MAGIC = 0x53514C4956450001


def mailbox_sequence(path) -> int | None:
    """Read the small native header without waiting for the publisher's lock."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        header = os.read(fd, 24)
        if len(header) != 24:
            return None
        magic, sequence, closed = struct.unpack('=QQQ', header)
        return sequence if magic == LIVE_MAGIC and not closed else None
    finally:
        os.close(fd)


class LiveProgressWatchdog:
    def __init__(self, cameras: int, timeout_s: float = 10.0):
        self.timeout_s = timeout_s
        self.inputs = [0] * cameras
        self.outputs = [0] * cameras
        self.consumed_inputs = [0] * cameras
        self.pending_since: list[float | None] = [None] * cameras

    def check(self, inputs, outputs, now: float) -> str | None:
        if (len(inputs) != len(self.outputs) or not isinstance(outputs, list) or len(outputs) != len(self.outputs)
                or any(type(value) is not int or value < 0 for value in outputs)):
            return 'live worker reported invalid frame progress'
        for i, output in enumerate(outputs):
            source = inputs[i]
            if type(source) is int and source >= 0:
                self.inputs[i] = max(self.inputs[i], source)
            if output < self.outputs[i]:
                return f'live frame counter reset on stream {i}'
            if output > self.outputs[i]:
                self.outputs[i] = output
                self.consumed_inputs[i] = self.inputs[i]
                self.pending_since[i] = None
            if self.inputs[i] > self.consumed_inputs[i]:
                if self.pending_since[i] is None:
                    self.pending_since[i] = now
                if now - self.pending_since[i] >= self.timeout_s:
                    return (f'live processing stopped advancing on stream {i} for {self.timeout_s:.1f}s '
                            f'despite available camera frames; outputs={output}')
        return None
