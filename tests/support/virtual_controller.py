"""Byte-port fixture; production reader, CRC decoder and durable journal run normally."""
import queue
import time

from squeakview.common import controller_protocol_v2 as protocol


class VirtualControllerPort:
    def __init__(self):
        self.incoming = queue.Queue()
        self.commands = []
        self.is_open = True
        self.sequence = 0
        self.session = 0
        self.count = 0

    def read(self, size):
        try:
            return self.incoming.get(timeout=.01)
        except queue.Empty:
            return b''

    def reply(self, payload, *, event=False):
        self.sequence += 1
        stamp = time.monotonic_ns() // 1000
        if callable(payload):
            payload = payload(stamp)
        frame = protocol.encode_frame(message_type=1 if event else 2,
            flags=protocol.FLAG_RELIABLE, boot_id=12345, session_id=self.session,
            sequence=self.sequence, monotonic_us=stamp, payload=payload.encode())
        # Exercise fragmented reads rather than handing the decoder a whole frame.
        for index in range(0, len(frame), 7):
            self.incoming.put(frame[index:index+7])

    def write(self, data):
        text = data.decode().strip()
        self.commands.append(text)
        fields = text.split(',')
        if text == 'PROTO,2':
            self.incoming.put(b'partial-tail\0')
        elif fields[0] == 'TIME_SYNC':
            self.reply(lambda stamp: f'CLOCK_SYNC,{fields[1]},{fields[2]},{stamp},{stamp},{time.time_ns() // 1000},RTC_VALID')
        elif text == 'START_BEHAVIOR':
            self.session = 7
            self.reply(lambda stamp: f'ACK_START_BEHAVIOR,1,{stamp},nan,0,0,69420,69420,task,nan', event=True)
            self.count = 1
            self.reply(lambda stamp: f'DRINK_START,1,{stamp},LD,1,69420,69420,69420,task,nan', event=True)
        elif text == 'BOTTLE_STATUS':
            self.reply(lambda stamp: f'BOTTLE_STATE,1,none,ACTIVE,{self.count},0,{stamp},OK')
        elif text == 'STOP':
            self.reply(lambda stamp: f'ACK_STOP,1,{stamp},nan,0,69420,69420,69420,task,nan', event=True)
        return len(data)

    def flush(self):
        pass

    def close(self):
        self.is_open = False
