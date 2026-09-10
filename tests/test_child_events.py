from __future__ import annotations

import unittest

from squeakview.common.child_events import (
    EVENT_PREFIX,
    ChildEvent,
    decode_child_event,
    encode_child_event,
)


class ChildEventTests(unittest.TestCase):
    def test_supported_event_round_trip(self) -> None:
        line = encode_child_event(
            "pipeline_ready", run_dir="/tmp/run", ready_origin="sink0"
        )

        self.assertEqual(
            decode_child_event(f"[12:00:00] Inference {line}"),
            ChildEvent(
                schema_version=1,
                type="pipeline_ready",
                payload={"ready_origin": "sink0", "run_dir": "/tmp/run"},
            ),
        )

    def test_human_and_malformed_lines_are_not_events(self) -> None:
        samples = (
            "[READY] wording is not a protocol",
            EVENT_PREFIX + "not-json",
            EVENT_PREFIX + "[]",
            EVENT_PREFIX + '{"schema_version":1}',
            EVENT_PREFIX + '{"schema_version":2,"type":"pipeline_ready"}',
            EVENT_PREFIX + '{"schema_version":1,"type":"unknown"}',
        )
        for sample in samples:
            with self.subTest(sample=sample):
                self.assertIsNone(decode_child_event(sample))

    def test_unknown_event_cannot_be_encoded(self) -> None:
        with self.assertRaises(ValueError):
            encode_child_event("unknown")

    def test_reserved_or_nonfinite_payload_cannot_be_encoded(self) -> None:
        for payload in ({"type": "fatal"}, {"schema_version": 2}, {"value": float("nan")}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                encode_child_event("pipeline_ready", **payload)

    def test_duplicate_nonfinite_and_deep_records_are_rejected(self) -> None:
        samples = (
            EVENT_PREFIX
            + '{"schema_version":1,"type":"pipeline_ready","type":"fatal"}',
            EVENT_PREFIX
            + '{"schema_version":1,"type":"pipeline_ready","value":NaN}',
            EVENT_PREFIX + "[" * 2000 + "]" * 2000,
        )
        for sample in samples:
            with self.subTest(sample=sample[:100]):
                self.assertIsNone(decode_child_event(sample))

    def test_decoded_payload_is_recursively_immutable(self) -> None:
        event = decode_child_event(
            EVENT_PREFIX
            + '{"schema_version":1,"type":"pipeline_ready",'
            '"nested":{"values":[1,2]}}'
        )

        self.assertIsNotNone(event)
        with self.assertRaises(TypeError):
            event.payload["nested"]["new"] = True


if __name__ == "__main__":
    unittest.main()
