from __future__ import annotations

import json
from pathlib import Path
import queue
import tempfile
import unittest
from unittest import mock

from squeakview.common import controller_protocol_v2 as v2
from squeakview.common.serial import SerialHandle


class StartupV2HandshakeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.handle = SerialHandle('/dev/test',115200,lambda _:None)
        self.assertTrue(self.handle.set_csv_path(self.root))
        self.addCleanup(self.handle.close)
        self.commands = []
        self.handle._queue_line = lambda command, **_:self.commands.append(command)

    def encode(self, sequence, text, *, boot=99, session=0, flags=v2.FLAG_RELIABLE,
               kind=v2.MessageType.COMMAND_RESULT):
        return v2.encode_frame(message_type=kind,flags=flags,boot_id=boot,
            session_id=session,sequence=sequence,monotonic_us=2000,payload=text.encode())

    def begin(self):
        self.handle._protocol_mode='v2_resync'
        self.handle._protocol_v2_pending=True
        self.handle._protocol_v2_probe=(123,1000)

    def proof(self, sequence=42, **kwargs):
        return self.encode(sequence,'CLOCK_SYNC,123,1000,100,110,1000000,RTC_VALID',**kwargs)

    def persist(self):
        while not self.handle._v2_queue.empty():
            frame,unix,mono=self.handle._v2_queue.get_nowait()
            try:
                self.handle._persist_v2_frame(frame,unix,mono)
            finally:
                self.handle._v2_queue.task_done()

    def test_partial_old_frame_and_stale_ack_cannot_establish_baseline(self):
        self.begin()
        old=self.encode(40,'ACK_PROTO,2')
        stale=self.encode(41,'CLOCK_SYNC,122,1000,100,110,1000000,RTC_VALID')
        self.handle._ingest_serial_bytes(b'partial-tail\x00'+old+stale+self.proof())
        self.assertEqual(self.handle._protocol_mode,'v2')
        self.assertEqual(self.handle._v2_queue.qsize(),1)
        self.assertFalse(self.handle._protocol_v2_seen.is_set())
        self.assertEqual(self.commands,[])
        self.assertEqual(self.handle._v2_journal.path.read_text(),'')
        self.persist()
        self.assertTrue(self.handle._protocol_v2_seen.is_set())
        self.assertEqual(self.handle._v2_journal.start_sequence(99),42)
        self.assertEqual(self.commands,['ACK_EVENTS,99,42'])
        counts=self.handle.protocol_v2_snapshot['counts']
        self.assertEqual(counts['startup_discarded_fragments'],1)
        self.assertEqual(counts['startup_discarded_frames'],2)
        self.assertEqual(counts['crc_or_framing_errors'],0)
        summary=json.loads((self.root/'diagnostics/controller_v2_summary.json').read_text())
        self.assertEqual(summary['startup_handshake']['method'],'fresh_idle_time_sync_echo')
        self.assertTrue(summary['startup_handshake']['durable'])

    def test_fragmented_proof_and_following_sensor_frame_are_preserved(self):
        self.begin()
        data=b'partial-tail\x00'+self.proof()+self.encode(43,'POKE_START,1000000,2000,L,1',session=1,kind=v2.MessageType.EVENT)
        for byte in data:
            self.handle._ingest_serial_bytes(bytes([byte]))
        self.persist()
        records=[json.loads(line) for line in self.handle._v2_journal.path.read_text().splitlines()]
        self.assertEqual([row['sequence'] for row in records],[42,43])
        self.assertEqual(self.handle._v2_journal.contiguous(99),43)

    def test_no_echo_and_wrong_echo_timeout_without_acknowledgements(self):
        def send(command):
            if command.startswith('TIME_SYNC,'):
                self.handle._ingest_serial_bytes(b'partial-tail\x00'+self.encode(41,'CLOCK_SYNC,123,999,100,110,1000000,RTC_VALID'))
        with mock.patch.object(self.handle,'_send_line',side_effect=send):
            with self.assertRaisesRegex(TimeoutError,'fresh idle'):
                self.handle.negotiate_protocol_v2(timeout_s=.01)
        self.assertEqual(self.commands,[])
        self.assertEqual(self.handle._v2_journal.path.read_text(),'')
        self.assertFalse(self.handle._protocol_v2_pending)
        self.assertIsNone(self.handle._protocol_v2_probe)
        self.assertEqual(self.handle._protocol_mode,'v1')

    def test_fresh_reply_must_be_idle_reliable_and_well_formed(self):
        for options in ({'session':1},{'flags':0},{'sequence':0},{'kind':v2.MessageType.EVENT}):
            with self.subTest(options=options):
                self.handle._v1_input.clear()
                self.handle._protocol_v2_error=None
                self.handle._protocol_v2_seen.clear()
                self.begin()
                data=self.proof(**options)
                self.handle._ingest_serial_bytes(data)
                self.assertIsNotNone(self.handle._protocol_v2_error)
                self.assertTrue(self.handle._protocol_v2_seen.is_set())
                self.assertEqual(self.handle._v2_queue.qsize(),0)
        self.handle._v1_input.clear()
        self.handle._protocol_v2_error=None
        self.begin()
        self.handle._ingest_serial_bytes(self.encode(42,'CLOCK_SYNC,123,1000,200,100,1000000,RTC_VALID'))
        self.assertIn('invalid startup handshake',self.handle._protocol_v2_error)

    def test_busy_and_integrity_faults_stop_startup(self):
        for text,flags in [('NACK,TIME_SYNC,DEVICE_BUSY',v2.FLAG_RELIABLE),
                           ('ACK_PROTO,2',v2.FLAG_RELIABLE|v2.FLAG_INTEGRITY_LATCHED)]:
            with self.subTest(text=text):
                self.handle._v1_input.clear()
                self.handle._protocol_v2_error=None
                self.begin()
                self.handle._ingest_serial_bytes(self.encode(41,text,flags=flags))
                self.assertTrue(self.handle._protocol_v2_seen.is_set())
                self.assertIsNotNone(self.handle._protocol_v2_error)
                self.assertEqual(self.handle._v2_queue.qsize(),0)
                self.assertEqual(self.commands,[])

    def test_missing_sequence_after_boundary_still_requests_replay(self):
        self.begin()
        self.handle._ingest_serial_bytes(self.proof()+self.encode(44,'POKE_START,1000000,2000,L,1',session=1,kind=v2.MessageType.EVENT))
        self.persist()
        self.assertEqual(self.handle._v2_journal.contiguous(99),42)
        self.assertIn('RESEND_EVENTS,99,43',self.commands)

    def test_corruption_after_boundary_is_not_hidden_as_startup_noise(self):
        self.begin()
        self.handle._ingest_serial_bytes(self.proof()+b'corrupt\x00')
        self.assertEqual(self.handle.protocol_v2_snapshot['counts']['crc_or_framing_errors'],1)
        self.assertEqual(self.handle.protocol_v2_snapshot['counts']['startup_discarded_fragments'],0)

    def test_disk_failure_cannot_complete_or_acknowledge_handshake(self):
        self.begin()
        self.handle._ingest_serial_bytes(self.proof())
        self.handle._v2_worker_stop.set()
        with mock.patch.object(self.handle._v2_journal,'store',side_effect=OSError('disk full')):
            self.handle._v2_persistence_pump()
        self.assertTrue(self.handle._protocol_v2_seen.is_set())
        self.assertIn('disk full',self.handle._protocol_v2_error)
        self.assertEqual(self.commands,[])

    def test_large_initial_noise_is_iterative_and_memory_bounded(self):
        self.begin()
        self.handle._ingest_serial_bytes(b'\x00'*2000+self.proof())
        self.assertEqual(self.handle._v2_queue.qsize(),1)
        self.assertEqual(self.handle.protocol_v2_snapshot['counts']['startup_discarded_fragments'],2000)
        self.assertEqual(self.handle._v1_input,bytearray())

    def test_public_negotiation_handles_cold_and_warm_controller(self):
        for cold in (True,False):
            with self.subTest(cold=cold):
                # Each public negotiation owns an empty run-local journal.
                self.handle._v2_journal.close()
                self.handle._v2_journal=v2.DurableJournal(self.root/f'journal-{cold}.jsonl',allow_midstream_start=True)
                self.handle._protocol_mode='v1'
                self.handle._v2_boot_id=None
                self.commands.clear()
                def send(command):
                    if command=='PROTO,2':
                        self.handle._ingest_serial_bytes(b'RTC_TYPE,DS3231\nACK_PROTO,2\n' if cold else b'partial-tail\x00'+self.encode(40,'ACK_PROTO,2'))
                        self.assertFalse(self.handle._protocol_v2_seen.is_set())
                    elif command.startswith('TIME_SYNC,'):
                        _,nonce,sent=command.split(',')
                        reply=self.encode(42,f'CLOCK_SYNC,{nonce},{sent},100,110,1000000,RTC_VALID')
                        self.handle._ingest_serial_bytes(reply)
                        self.persist()
                with mock.patch.object(self.handle,'_send_line',side_effect=send):
                    self.handle.negotiate_protocol_v2(timeout_s=.5)
                self.assertEqual(self.handle._protocol_mode,'v2')
                self.assertEqual(self.handle._v2_boot_id,99)
                self.assertIn('ACK_EVENTS,99,42',self.commands)

    def test_pending_negotiation_cannot_be_bypassed_before_fsync(self):
        self.begin()
        self.handle._ingest_serial_bytes(self.proof())
        self.assertEqual(self.handle._protocol_mode,'v2')
        self.assertFalse(self.handle.protocol_v2_snapshot['startup_handshake']['durable'])
        with self.assertRaisesRegex(RuntimeError,'already pending'):
            self.handle.negotiate_protocol_v2()
        self.assertFalse(self.handle._protocol_v2_seen.is_set())
        self.persist()
        self.assertTrue(self.handle.protocol_v2_snapshot['startup_handshake']['durable'])

    def test_existing_journal_cannot_be_rebased(self):
        frame=v2.decode_frame(self.encode(1,'ACK_PROTO,2')[:-1])
        self.handle._v2_journal.store(frame,host_unix_ns=1,host_monotonic_ns=2)
        with self.assertRaisesRegex(RuntimeError,'durable watermark'):
            self.handle.negotiate_protocol_v2()


class ThreadedStartupV2Tests(unittest.TestCase):
    def test_real_reader_writer_and_journal_workers_recover_warm_attach(self):
        class Port:
            is_open=True
            def __init__(self):
                self.incoming=queue.Queue()
                self.commands=[]
            def read(self,size):
                try:return self.incoming.get(timeout=.01)
                except queue.Empty:return b''
            def write(self,data):
                self.commands.append(data.decode().strip())
                if data==b'PROTO,2\n':self.incoming.put(b'partial-tail\x00')
                if data.startswith(b'TIME_SYNC,'):
                    _,nonce,sent=data.decode().strip().split(',')
                    proof=v2.encode_frame(message_type=v2.MessageType.COMMAND_RESULT,
                        flags=v2.FLAG_RELIABLE,boot_id=99,session_id=0,sequence=42,
                        monotonic_us=110,payload=f'CLOCK_SYNC,{nonce},{sent},100,110,1000000,RTC_VALID'.encode())
                    for i in range(0,len(proof),7):self.incoming.put(proof[i:i+7])
                return len(data)
            def flush(self):pass
            def close(self):self.is_open=False
        with tempfile.TemporaryDirectory() as directory:
            port=Port()
            handle=SerialHandle('/dev/test',115200,lambda _:None)
            try:
                with mock.patch('squeakview.common.serial.serial.Serial',return_value=port):
                    self.assertTrue(handle.open(Path(directory)))
                    handle.negotiate_protocol_v2(timeout_s=1)
                self.assertIsNone(handle.fatal_error)
                self.assertEqual(handle._v2_journal.contiguous(99),42)
                self.assertEqual(handle._v2_last_ack_queued.get(99),42)
            finally:
                handle.close()
            self.assertIn('ACK_EVENTS,99,42',port.commands)

if __name__=='__main__':
    unittest.main()
