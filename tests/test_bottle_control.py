from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from squeakview.common.bottle_control import parse_bottle_state, parse_lick_recovery
from squeakview.common.serial import SerialHandle
from squeakview.common import controller_protocol_v2 as v2


IDENTITY = "a" * 32


def line(state="PAUSED", identity=IDENTITY, left=7, right=11, stamp=100):
    return f"BOTTLE_STATE,1,{identity},{state},{left},{right},{stamp},READY"


class BottleControlTests(unittest.TestCase):
    def test_recovery_status_preserves_counts_and_bottle_transaction(self):
        handle = self.handle()
        handle._ingest_bottle_line(line("ACTIVE", left=21, right=16))
        for event, stamp, gap, attempt in (("START", 1000, 0, 0), ("ATTEMPT", 1010, 10, 1),
                                           ("RETRY", 2000, 1000, 1), ("READY", 800000, 799000, 2)):
            for side, count in (("LD", 21), ("RD", 16)):
                handle._ingest_bottle_line(f"LICK_RECOVERY_{event},1,{stamp},{side},{count},{gap},{attempt},1,task,READY")
            snapshot = handle.bottle_control_snapshot
            self.assertEqual(snapshot["counts"], {"left": 21, "right": 16})
            self.assertEqual(snapshot["controller"]["swap_id"], IDENTITY)
            self.assertEqual(snapshot["controller"]["state"], "ACTIVE")
            self.assertEqual(snapshot["lick_recovery"]["available"], event == "READY")
            self.assertEqual(snapshot["lick_recovery"]["gap_start_us"], 1000)
        handle._ingest_bottle_line("ACK_START,1,2,nan,30,nan,nan,nan,task,nan")
        self.assertIsNone(handle.bottle_control_snapshot["lick_recovery"])

    def test_recovery_decoder_rejects_malformed_fields(self):
        valid = "LICK_RECOVERY_FAILED,1,1000,LD,21,100,3,1,task,READINESS_TIMEOUT"
        self.assertEqual(parse_lick_recovery(valid)["state"], "FAILED")
        for malformed in (valid.replace("FAILED", "BOGUS"), valid.replace(",LD,", ",LEFT,"),
                          valid.replace(",100,3,", ",100,4,"), valid.replace(",100,3,", ",1001,3,"),
                          valid.replace(",21,", ",-1,")):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                parse_lick_recovery(malformed)

    def test_recovery_events_use_standard_csv_fields_and_direct_timestamps(self):
        from squeakview.common.sensor_csv import make_event_record, EVENT_FIELDS
        text = "LICK_RECOVERY_READY,12345,1000,LD,21,900,1,1,task,READY"
        row = make_event_record(text.split(","), text, host_unix_ns=5, host_monotonic_ns=6)
        self.assertEqual(row["eventType"], "LICK_RECOVERY_READY")
        self.assertEqual(row["controller_time_us"], 1000)
        self.assertEqual(row["controller_time_method"], "direct_event")
        self.assertEqual(row["count"], "21")
        self.assertTrue(all(field in row for field in EVENT_FIELDS))

    def handle(self):
        handle = SerialHandle("/unused", 115200, lambda _line: None)
        handle._protocol_mode = "v2"
        self.addCleanup(handle.close)
        return handle

    def test_state_decoder_rejects_invalid_versions_ids_and_numbers(self):
        state = parse_bottle_state(line())
        self.assertEqual(state["left_count"], 7)
        self.assertEqual(state["swap_id"], IDENTITY)
        for text in (line().replace(",1,", ",2,", 1), line(identity="bad"),
                     line(left=-1), line(stamp=1 << 64), line(state="BOGUS")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_bottle_state(text)

    def test_resume_waits_for_matching_terminal_state(self):
        handle = self.handle()
        def send(_command):
            handle._ingest_bottle_line(line("ACTIVE", "b" * 32))
            handle._ingest_bottle_line(line("CALIBRATING"))
            handle._ingest_bottle_line(line("ACTIVE", stamp=200))
        with mock.patch.object(handle, "send_line", side_effect=send):
            response = handle.exchange_bottle_control("resume", IDENTITY, timeout_s=0.1)
        self.assertEqual(response["monotonic_us"], 200)

    def test_old_cached_pause_cannot_acknowledge_new_request(self):
        handle = self.handle()
        handle._ingest_bottle_line(line())
        with mock.patch.object(handle, "send_line"), self.assertRaises(TimeoutError):
            handle.exchange_bottle_control("pause", IDENTITY, timeout_s=0.01)

    def test_stop_cancels_calibration_wait(self):
        handle = self.handle()
        with mock.patch.object(handle, "send_line"), self.assertRaisesRegex(RuntimeError, "Stop"):
            handle.exchange_bottle_control("resume", IDENTITY, cancelled=lambda: True)

    def test_lick_counts_deduplicate_and_reset_at_session_start(self):
        handle = self.handle()
        for count in (7, 7, 3, 9):
            handle._ingest_bottle_line(f"DRINK_START,1,2,LD,{count},nan,nan,nan,task,nan")
        self.assertEqual(handle.bottle_control_snapshot["counts"]["left"], 9)
        handle._ingest_bottle_line("ACK_START,1,2,nan,30,nan,nan,nan,task,nan")
        self.assertEqual(handle.bottle_control_snapshot["counts"], {"left": 0, "right": 0})

    def test_durable_journal_precedes_exposing_pause(self):
        handle = self.handle()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal.jsonl"
            handle._v2_journal = v2.DurableJournal(path)
            self.addCleanup(handle._v2_journal.close)
            frame = v2.decode_frame(v2.encode_frame(
                message_type=2, flags=1, boot_id=42, session_id=1,
                sequence=1, monotonic_us=100, payload=line().encode(),
            )[:-1])
            original = handle._ingest_bottle_line
            def ingest(text):
                self.assertIn('"sequence":1', path.read_text())
                original(text)
            with mock.patch.object(handle, "_ingest_bottle_line", side_effect=ingest), mock.patch.object(handle, "_queue_line"):
                handle._persist_v2_frame(frame, 1000, 2000)
            self.assertEqual(handle.bottle_control_snapshot["controller"]["state"], "PAUSED")

    def test_behavior_start_requires_ack_and_resets_session_counts(self):
        handle = self.handle()
        handle._bottle_counts['left'] = 99
        def send(command, **_kwargs):
            self.assertEqual(command, 'START_BEHAVIOR')
            handle._handle_semantic_line(
                'ACK_START_BEHAVIOR,100,200,nan,0,0,nan,nan,task,nan', received_ns=300,
            )
        with mock.patch.object(handle, '_send_line', side_effect=send):
            handle.send_behavior_start(timeout_s=0.1)
        self.assertEqual(handle.bottle_control_snapshot['counts'], {'left': 0, 'right': 0})

    def test_behavior_start_rejects_unsupported_firmware_and_missing_ack(self):
        handle = self.handle()
        def nack(_command, **_kwargs):
            handle._handle_semantic_line('NACK,UNKNOWN,UNKNOWN_COMMAND', received_ns=1)
        with mock.patch.object(handle, '_send_line', side_effect=nack):
            with self.assertRaisesRegex(RuntimeError, 'unsupported'):
                handle.send_behavior_start(timeout_s=0.1)
        with mock.patch.object(handle, '_send_line'):
            with self.assertRaises(TimeoutError):
                handle.send_behavior_start(timeout_s=0.01)


if __name__ == "__main__":
    unittest.main()
