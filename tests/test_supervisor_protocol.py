from __future__ import annotations

import dataclasses
import threading
import time
import unittest
from unittest import mock

from squeakview.apps.operator.backend.supervisor.protocol import (
    MAX_FRAME_BYTES,
    CommandEnvelope,
    EventEnvelope,
    FrameTooLargeError,
    NewlineFrameDecoder,
    ProtocolError,
    SocketEnvelopeReader,
    SocketEnvelopeWriter,
    decode_envelope,
    encode_envelope,
    send_envelope,
)
from squeakview.common.dashboard import DashboardEvent
from squeakview.common.dashboard import MAX_DASHBOARD_RAW_CHARS


class _MemorySocket:
    """Minimal connected-socket double for sandbox-independent framing tests."""

    def __init__(self) -> None:
        self.buffer = bytearray()

    def sendall(self, data: bytes) -> None:
        self.buffer.extend(data)

    def recv(self, size: int) -> bytes:
        chunk = bytes(self.buffer[:size])
        del self.buffer[:size]
        return chunk


class SupervisorProtocolTest(unittest.TestCase):
    def test_command_round_trip_is_immutable_and_detached_from_source(self) -> None:
        source = {"config": {"cameras": ["a", "b"]}}
        command = CommandEnvelope("start", "req-1", source)
        source["config"]["cameras"].append("c")

        decoded = decode_envelope(encode_envelope(command))

        self.assertEqual(decoded, command)
        self.assertEqual(decoded.payload["config"]["cameras"], ("a", "b"))
        with self.assertRaises(TypeError):
            decoded.payload["new"] = True
        with self.assertRaises(dataclasses.FrozenInstanceError):
            decoded.name = "stop"

    def test_event_round_trip_preserves_sequence_and_correlation(self) -> None:
        event = EventEnvelope(
            "command.completed",
            42,
            {"ok": True},
            request_id="req-1",
        )

        self.assertEqual(decode_envelope(encode_envelope(event)), event)

    def test_dashboard_event_payload_round_trip_is_typed_and_json_safe(self) -> None:
        dashboard = DashboardEvent.parse(
            "POKE_START,1000000,2000000,LD,3,4,5,6,ON,reason"
        )
        assert dashboard is not None

        envelope = decode_envelope(
            encode_envelope(EventEnvelope("dashboard", 1, dashboard.to_payload()))
        )
        restored = DashboardEvent.from_payload(envelope.payload)

        self.assertEqual(restored, dashboard)
        self.assertEqual(restored.side_uc, "L")
        self.assertEqual(restored.unix_sec, 1.0)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            restored.event_uc = "CHANGED"

    def test_dashboard_event_sanitization_keeps_ipc_frame_bounded(self) -> None:
        dashboard = DashboardEvent.parse("EVENT," + "🐁" * (64 * 1024))
        assert dashboard is not None

        frame = encode_envelope(EventEnvelope("dashboard", 1, dashboard.to_payload()))

        self.assertTrue(dashboard.truncated)
        self.assertEqual(len(dashboard.raw_line), MAX_DASHBOARD_RAW_CHARS)
        self.assertLessEqual(len(frame), MAX_FRAME_BYTES)

    def test_event_may_be_unsolicited(self) -> None:
        event = EventEnvelope("phase.changed", 0, {"phase": "recording"})

        self.assertIsNone(decode_envelope(encode_envelope(event)).request_id)

    def test_decoder_handles_fragmented_and_coalesced_frames(self) -> None:
        first = CommandEnvelope("ping", "one")
        second = EventEnvelope("pong", 1, request_id="one")
        decoder = NewlineFrameDecoder()
        wire = encode_envelope(first) + encode_envelope(second)

        self.assertEqual(decoder.feed(wire[:7]), ())
        self.assertEqual(decoder.pending_bytes, 7)
        self.assertEqual(decoder.feed(wire[7:]), (first, second))
        self.assertEqual(decoder.pending_bytes, 0)

    def test_socket_helpers_preserve_multiple_buffered_frames(self) -> None:
        connection = _MemorySocket()
        first = CommandEnvelope("stop", "one")
        second = CommandEnvelope("shutdown", "two")
        reader = SocketEnvelopeReader()

        send_envelope(connection, first)
        send_envelope(connection, second)

        self.assertEqual(reader.receive(connection), first)
        self.assertEqual(reader.receive(connection), second)

    def test_persistent_socket_writer_serializes_concurrent_frames(self) -> None:
        class OverlapDetectingSocket(_MemorySocket):
            def __init__(self) -> None:
                super().__init__()
                self.active_writes = 0
                self.overlapped = False
                self.state_lock = threading.Lock()

            def sendall(self, data: bytes) -> None:
                with self.state_lock:
                    self.active_writes += 1
                    self.overlapped |= self.active_writes > 1
                time.sleep(0.005)
                super().sendall(data)
                with self.state_lock:
                    self.active_writes -= 1

        connection = OverlapDetectingSocket()
        writer = SocketEnvelopeWriter(connection)
        first = CommandEnvelope("ping", "one")
        second = EventEnvelope("log", 1)
        threads = [
            threading.Thread(target=writer.send, args=(envelope,))
            for envelope in (first, second)
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertFalse(connection.overlapped)
        decoded = NewlineFrameDecoder().feed(bytes(connection.buffer))
        self.assertEqual(len(decoded), 2)
        self.assertEqual({item.name for item in decoded}, {"ping", "log"})

    def test_writer_deadline_is_independent_of_socket_read_timeout(self) -> None:
        connection = mock.Mock()
        connection.send.side_effect = BlockingIOError
        writer = SocketEnvelopeWriter(connection, send_timeout_s=0.01)

        started = time.monotonic()
        with (
            mock.patch(
                "squeakview.apps.operator.backend.supervisor.protocol.select.select",
                return_value=([], [], []),
            ),
            self.assertRaisesRegex(TimeoutError, "delivery exceeded"),
        ):
            writer.send(EventEnvelope("log", 1, {"message": "blocked"}))

        self.assertLess(time.monotonic() - started, 0.2)
        connection.settimeout.assert_not_called()

    def test_clean_socket_eof_is_distinct_from_protocol_failure(self) -> None:
        with self.assertRaises(EOFError):
            SocketEnvelopeReader().receive(_MemorySocket())

    def test_socket_eof_rejects_truncated_frame(self) -> None:
        connection = _MemorySocket()
        connection.sendall(b'{"version":1')

        with self.assertRaisesRegex(ProtocolError, "incomplete"):
            SocketEnvelopeReader().receive(connection)

    def test_encoded_frame_has_fixed_size_limit(self) -> None:
        command = CommandEnvelope("start", "large", {"blob": "x" * MAX_FRAME_BYTES})

        with self.assertRaises(FrameTooLargeError):
            encode_envelope(command)

    def test_unterminated_stream_cannot_grow_to_frame_limit(self) -> None:
        decoder = NewlineFrameDecoder()

        with self.assertRaises(FrameTooLargeError):
            decoder.feed(b"x" * MAX_FRAME_BYTES)
        self.assertEqual(decoder.pending_bytes, 0)
        with self.assertRaisesRegex(ProtocolError, "cannot continue"):
            decoder.feed(b"{}\n")

    def test_decode_requires_exactly_one_newline_terminated_frame(self) -> None:
        command = encode_envelope(CommandEnvelope("ping", "one"))

        with self.assertRaisesRegex(ProtocolError, "terminator"):
            decode_envelope(command[:-1])
        with self.assertRaisesRegex(ProtocolError, "exactly one"):
            decode_envelope(command + command)

    def test_invalid_utf8_and_nonobject_json_are_rejected(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "UTF-8"):
            decode_envelope(b"\xff\n")
        with self.assertRaisesRegex(ProtocolError, "JSON object"):
            decode_envelope(b"[]\n")

    def test_duplicate_json_fields_are_rejected(self) -> None:
        frame = (
            b'{"version":1,"version":1,"type":"command","name":"ping",'
            b'"request_id":"one","payload":{}}\n'
        )

        with self.assertRaisesRegex(ProtocolError, "duplicate"):
            decode_envelope(frame)

    def test_unknown_or_missing_fields_are_rejected(self) -> None:
        unknown = (
            b'{"version":1,"type":"command","name":"ping",'
            b'"request_id":"one","payload":{},"extra":true}\n'
        )
        missing = b'{"version":1,"type":"command","name":"ping","payload":{}}\n'

        with self.assertRaisesRegex(ProtocolError, "exactly"):
            decode_envelope(unknown)
        with self.assertRaisesRegex(ProtocolError, "exactly"):
            decode_envelope(missing)

    def test_unsupported_version_and_envelope_type_are_rejected(self) -> None:
        wrong_version = (
            b'{"version":2,"type":"command","name":"ping",'
            b'"request_id":"one","payload":{}}\n'
        )
        wrong_type = (
            b'{"version":1,"type":"response","name":"ping",'
            b'"request_id":"one","payload":{}}\n'
        )

        with self.assertRaisesRegex(ProtocolError, "unsupported"):
            decode_envelope(wrong_version)
        with self.assertRaisesRegex(ProtocolError, "command.*event"):
            decode_envelope(wrong_type)

    def test_identifiers_and_sequences_are_strictly_validated(self) -> None:
        for factory in (
            lambda: CommandEnvelope("UPPER", "one"),
            lambda: CommandEnvelope("ping", "contains space"),
            lambda: EventEnvelope("phase", -1),
            lambda: EventEnvelope("phase", True),
        ):
            with self.subTest(factory=factory):
                with self.assertRaises(ProtocolError):
                    factory()

    def test_payload_must_contain_only_strict_json_values(self) -> None:
        for payload in (
            [],
            {"bad": float("nan")},
            {"bad": object()},
            {1: "non-string key"},
            {"huge": 1 << 80},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ProtocolError):
                    CommandEnvelope("ping", "one", payload)

    def test_payload_rejects_invalid_unicode_and_excessive_nesting(self) -> None:
        nested: object = None
        for _ in range(34):
            nested = [nested]

        for payload in ({"bad": "\ud800"}, {"nested": nested}):
            with self.subTest(payload=payload):
                with self.assertRaises(ProtocolError):
                    CommandEnvelope("ping", "one", payload)


if __name__ == "__main__":
    unittest.main()
