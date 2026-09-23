from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from squeakview.common import controller_protocol_v2 as v2
from squeakview.common import serial as serial_util
from squeakview.tools import mousehouse_protocol_v2_test as qualification


class ProtocolV2CodecTests(unittest.TestCase):
    def frame(self, *, sequence: int = 1, flags: int = v2.FLAG_RELIABLE) -> bytes:
        return v2.encode_frame(
            message_type=v2.MessageType.COMMAND_RESULT,
            flags=flags,
            boot_id=0x0102030405060708,
            session_id=7,
            sequence=sequence,
            monotonic_us=123456789,
            payload=b"ACK_FEED",
        )

    def test_documented_crc_vector(self) -> None:
        self.assertEqual(v2.crc32(b"123456789"), 0xCBF43926)

    def test_documented_example_decodes(self) -> None:
        encoded = bytes.fromhex(
            "04 02 02 01 0a 08 07 06 05 04 03 02 01 07 01 01 "
            "02 2a 01 01 01 01 01 01 05 15 cd 5b 07 01 01 01 "
            "02 08 0d 41 43 4b 5f 46 45 45 44 49 38 a8 0f 00"
        )
        frame = v2.decode_frame(encoded[:-1])
        self.assertEqual(frame.boot_id, 0x0102030405060708)
        self.assertEqual(frame.sequence, 42)
        self.assertEqual(frame.text, "ACK_FEED")

    def test_round_trip_payload_with_zero_bytes(self) -> None:
        encoded = v2.encode_frame(
            message_type=8,
            flags=1,
            boot_id=2,
            session_id=0,
            sequence=9,
            monotonic_us=10,
            payload=b"a\x00b",
        )
        self.assertEqual(v2.decode_frame(encoded[:-1]).payload, b"a\x00b")

    def test_stream_decoder_handles_fragmented_and_combined_reads(self) -> None:
        first, second = self.frame(sequence=1), self.frame(sequence=2)
        decoder = v2.StreamDecoder()
        frames, errors = decoder.feed(first[:5])
        self.assertEqual((frames, errors), ([], []))
        frames, errors = decoder.feed(first[5:] + second)
        self.assertEqual([frame.sequence for frame in frames], [1, 2])
        self.assertEqual(errors, [])

    def test_stream_decoder_rejects_crc_and_recovers_at_delimiter(self) -> None:
        broken = bytearray(self.frame(sequence=1))
        broken[-3] ^= 0x01
        decoder = v2.StreamDecoder()
        frames, errors = decoder.feed(bytes(broken) + self.frame(sequence=2))
        self.assertEqual([frame.sequence for frame in frames], [2])
        self.assertEqual(len(errors), 1)
        self.assertIn("CRC mismatch", errors[0])

    def test_disconnect_discard_drops_partial_frame(self) -> None:
        decoder = v2.StreamDecoder()
        encoded = self.frame(sequence=1)
        decoder.feed(encoded[:10])
        decoder.discard_through_delimiter()
        frames, errors = decoder.feed(encoded[10:] + self.frame(sequence=2))
        self.assertEqual([frame.sequence for frame in frames], [2])
        self.assertEqual(errors, [])

    def test_camera_and_status_payloads_follow_firmware_layout(self) -> None:
        camera = v2.parse_camera_payload(
            "CAMERA_CHECKPOINT,count=30,timestamp_us=1000,period_us=33333,"
            "pulse_us=1000,health=0x00000003,queue=2/5,suppressed=0,reason=Periodic"
        )
        self.assertEqual(camera["count"], "30")
        status = v2.parse_transport_status(
            "TRANSPORT_STATUS,b=1,m=3,n=1,q=2/64,h=5,o=1,a=5,t=5,k=3,"
            "x=0,l=none/0,u=0,c=0,p=0,z=0,r=0,v=0,f=0,d=50"
        )
        self.assertEqual(status["q"], "2/64")


class ProtocolV2JournalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "controller_v2.jsonl"

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def decoded(sequence: int, *, payload: bytes = b"ACK") -> v2.Frame:
        encoded = v2.encode_frame(
            message_type=2,
            flags=v2.FLAG_RELIABLE,
            boot_id=88,
            session_id=1,
            sequence=sequence,
            monotonic_us=sequence * 10,
            payload=payload,
        )
        return v2.decode_frame(encoded[:-1])

    def test_contiguous_advances_only_after_gap_is_filled(self) -> None:
        journal = v2.DurableJournal(self.path)
        try:
            self.assertEqual(journal.store(self.decoded(2), host_unix_ns=1, host_monotonic_ns=2)[1], 0)
            self.assertEqual(journal.store(self.decoded(1), host_unix_ns=3, host_monotonic_ns=4)[1], 2)
        finally:
            journal.close()

    def test_retransmission_is_duplicate_but_conflicting_payload_fails(self) -> None:
        journal = v2.DurableJournal(self.path)
        original = self.decoded(1)
        try:
            self.assertEqual(journal.store(original, host_unix_ns=1, host_monotonic_ns=2)[0], v2.StoreResult.NEW)
            retransmitted = v2.decode_frame(
                v2.encode_frame(
                    message_type=2,
                    flags=v2.FLAG_RELIABLE | v2.FLAG_RETRANSMISSION,
                    boot_id=88,
                    session_id=1,
                    sequence=1,
                    monotonic_us=10,
                    payload=b"ACK",
                )[:-1]
            )
            self.assertEqual(journal.store(retransmitted, host_unix_ns=3, host_monotonic_ns=4)[0], v2.StoreResult.DUPLICATE)
            self.assertEqual(journal.store(self.decoded(1, payload=b"NACK"), host_unix_ns=5, host_monotonic_ns=6)[0], v2.StoreResult.CONFLICT)
        finally:
            journal.close()

    def test_journal_recovers_contiguous_state_after_reopen(self) -> None:
        journal = v2.DurableJournal(self.path)
        journal.store(self.decoded(1), host_unix_ns=1, host_monotonic_ns=2)
        journal.close()
        reopened = v2.DurableJournal(self.path)
        try:
            self.assertEqual(reopened.contiguous(88), 1)
        finally:
            reopened.close()

    def test_run_local_journal_can_begin_mid_boot_and_preserve_baseline(self) -> None:
        journal = v2.DurableJournal(self.path, allow_midstream_start=True)
        try:
            result, contiguous = journal.store(
                self.decoded(42),
                host_unix_ns=1,
                host_monotonic_ns=2,
            )
            self.assertEqual(result, v2.StoreResult.NEW)
            self.assertEqual(contiguous, 42)
            self.assertEqual(journal.start_sequence(88), 42)
        finally:
            journal.close()

        reopened = v2.DurableJournal(self.path, allow_midstream_start=True)
        try:
            self.assertEqual(reopened.contiguous(88), 42)
            self.assertEqual(reopened.start_sequence(88), 42)
        finally:
            reopened.close()


class SerialProtocolV2BoundaryTests(unittest.TestCase):
    def test_negotiation_preserves_binary_bytes_after_ack_newline(self) -> None:
        handle = serial_util.SerialHandle("/dev/test", 115200, lambda _line: None)
        handle._protocol_v2_pending = True
        binary = v2.encode_frame(
            message_type=2,
            flags=1,
            boot_id=9,
            session_id=0,
            sequence=1,
            monotonic_us=2,
            payload=b"ACK_PROTO,2",
        )
        handle._ingest_serial_bytes(b"ACK_PROTO,2\n" + binary)
        queued, _unix, _monotonic = handle._v2_queue.get_nowait()
        self.assertEqual(handle._protocol_mode, "v2")
        self.assertEqual(queued.text, "ACK_PROTO,2")

    def test_ingest_counts_retransmission_before_deduplication(self) -> None:
        handle = serial_util.SerialHandle("/dev/test", 115200, lambda _line: None)
        handle._protocol_mode = "v2"
        encoded = v2.encode_frame(
            message_type=v2.MessageType.EVENT,
            flags=v2.FLAG_RELIABLE | v2.FLAG_RETRANSMISSION,
            boot_id=9,
            session_id=1,
            sequence=3,
            monotonic_us=4,
            payload=b"POKE_START",
        )

        handle._ingest_serial_bytes(encoded)

        counts = handle.protocol_v2_snapshot["counts"]
        self.assertIsInstance(counts, dict)
        assert isinstance(counts, dict)
        self.assertEqual(counts["frames_received"], 1)
        self.assertEqual(
            counts["retransmissions_received"],
            1,
        )

    def test_reconnect_binary_boundary_requests_from_durable_watermark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            handle = serial_util.SerialHandle("/dev/test", 115200, lambda _line: None)
            handle._v2_journal = v2.DurableJournal(Path(directory) / "journal.jsonl")
            first = v2.decode_frame(
                v2.encode_frame(
                    message_type=2,
                    flags=1,
                    boot_id=9,
                    session_id=0,
                    sequence=1,
                    monotonic_us=2,
                    payload=b"ACK",
                )[:-1]
            )
            handle._v2_journal.store(first, host_unix_ns=1, host_monotonic_ns=2)
            handle._v2_boot_id = 9
            handle._protocol_mode = "v2_resync"
            commands: list[str] = []
            with mock.patch.object(
                handle, "_queue_line", side_effect=lambda command, **_kwargs: commands.append(command)
            ):
                handle._ingest_v2_resync_bytes(b"discarded-partial\x00")
            self.assertEqual(handle._protocol_mode, "v2")
            self.assertEqual(commands, ["ACK_EVENTS,9,1", "RESEND_EVENTS,9,2"])
            handle._v2_journal.close()

    def test_new_handle_rejoins_already_active_v2_without_dropping_ack(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            handle = serial_util.SerialHandle("/dev/test", 115200, lambda _line: None)
            handle._v2_journal = v2.DurableJournal(Path(directory) / "journal.jsonl")
            handle._protocol_mode = "v2_resync"
            handle._protocol_v2_pending = True
            encoded = v2.encode_frame(
                message_type=v2.MessageType.COMMAND_RESULT,
                flags=v2.FLAG_RELIABLE,
                boot_id=99,
                session_id=0,
                sequence=1,
                monotonic_us=2,
                payload=b"ACK_PROTO,2",
            )

            handle._ingest_serial_bytes(encoded)
            frame, host_unix_ns, host_monotonic_ns = handle._v2_queue.get_nowait()
            assert frame is not None
            with mock.patch.object(handle, "_queue_line"):
                handle._persist_v2_frame(frame, host_unix_ns, host_monotonic_ns)

            self.assertEqual(handle._protocol_mode, "v2")
            self.assertFalse(handle._protocol_v2_pending)
            self.assertTrue(handle._protocol_v2_seen.is_set())
            self.assertEqual(handle.protocol_v2_snapshot["boot_id"], 99)
            handle._v2_journal.close()

    def test_new_handle_fails_closed_when_attached_midframe(self) -> None:
        handle = serial_util.SerialHandle("/dev/test", 115200, lambda _line: None)
        handle._protocol_mode = "v2_resync"
        handle._protocol_v2_pending = True

        handle._ingest_serial_bytes(b"partial-frame-tail\x00")

        self.assertTrue(handle._protocol_v2_seen.is_set())
        self.assertFalse(handle._protocol_v2_pending)
        self.assertIn("safe run-local sequence baseline", handle._protocol_v2_error or "")

    def test_reconnect_v1_ack_handles_controller_reboot_boundary(self) -> None:
        handle = serial_util.SerialHandle("/dev/test", 115200, lambda _line: None)
        handle._protocol_mode = "v2_resync"
        handle._ingest_v2_resync_bytes(b"SYSTEM_START,1,2\nACK_PROTO,2\n")
        self.assertEqual(handle._protocol_mode, "v2")

    def test_durable_store_precedes_ack_queueing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            handle = serial_util.SerialHandle("/dev/test", 115200, lambda _line: None)
            handle._v2_journal = v2.DurableJournal(Path(directory) / "journal.jsonl")
            frame = v2.decode_frame(
                v2.encode_frame(
                    message_type=8,
                    flags=v2.FLAG_RELIABLE,
                    boot_id=5,
                    session_id=0,
                    sequence=1,
                    monotonic_us=3,
                    payload=b"DIAGNOSTIC",
                )[:-1]
            )

            def assert_durable(*_args, **_kwargs) -> None:
                self.assertIn('"sequence":1', (Path(directory) / "journal.jsonl").read_text())

            with mock.patch.object(handle, "_queue_line", side_effect=assert_durable):
                handle._persist_v2_frame(frame, 10, 11)
            handle._v2_journal.close()


class ProtocolV2QualificationTests(unittest.TestCase):
    class ReplayHandle:
        def __init__(self) -> None:
            self.ack_withheld = True
            self.duplicates = 0
            self.retransmissions = 0
            self.flags = [v2.FLAG_RELIABLE]
            self.calls: list[str] = []
            self.fatal_error: str | None = None

        @property
        def protocol_v2_snapshot(self) -> dict[str, object]:
            return {
                "boot_id": 99,
                "counts": {
                    "duplicates": self.duplicates,
                    "retransmissions_received": self.retransmissions,
                },
            }

        def protocol_v2_contiguous_sequence(self) -> int:
            return 20

        def protocol_v2_observations(
            self,
        ) -> tuple[dict[int, int], list[tuple[int, str, int, int]]]:
            return {}, [
                (int(v2.MessageType.EVENT), "EVENT", index + 1, flags)
                for index, flags in enumerate(self.flags)
            ]

        def request_protocol_v2_replay(self, from_sequence: int) -> None:
            self.calls.append(f"replay:{from_sequence}:held={self.ack_withheld}")
            self.duplicates += 1
            self.retransmissions += 1
            self.flags.append(v2.FLAG_RELIABLE | v2.FLAG_RETRANSMISSION)

        def set_protocol_v2_ack_withheld(self, withheld: bool) -> None:
            self.ack_withheld = withheld
            self.calls.append(f"withheld:{withheld}")

    def test_clock_correction_requires_explicit_cli_opt_in(self) -> None:
        self.assertFalse(qualification._parse_args([]).correct_clock)
        self.assertTrue(
            qualification._parse_args(["--correct-clock"]).correct_clock
        )

    def test_replay_is_requested_before_ack_hold_is_released(self) -> None:
        handle = self.ReplayHandle()
        checks: dict[str, object] = {}

        probe = qualification._finish_ack_hold(  # type: ignore[arg-type]
            handle,
            checks,
            request_resend=True,
        )

        self.assertEqual(
            handle.calls,
            ["replay:18:held=True", "withheld:False"],
        )
        self.assertEqual(
            probe,
            {
                "boot_id": 99,
                "from_sequence": 18,
                "durable_sequence_at_request": 20,
            },
        )
        self.assertIs(checks["replay_duplicate_received"], True)
        self.assertIs(checks["retransmission_flag_seen"], True)

    def test_ack_hold_is_released_when_replay_probe_raises(self) -> None:
        handle = self.ReplayHandle()
        handle.protocol_v2_contiguous_sequence = lambda: 0  # type: ignore[method-assign]

        with self.assertRaisesRegex(RuntimeError, "durable v2 record"):
            qualification._finish_ack_hold(  # type: ignore[arg-type]
                handle,
                {},
                request_resend=True,
            )

        self.assertFalse(handle.ack_withheld)
        self.assertEqual(handle.calls, ["withheld:False"])


if __name__ == "__main__":
    unittest.main()
