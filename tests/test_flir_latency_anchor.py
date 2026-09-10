from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native" / "flir_gst_source" / "src" / "gstflirspinsrc.cpp"


class FlirLatencyAnchorSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE.read_text(encoding="utf-8")

    def test_anchor_is_debug_gated_and_uses_real_source_identity(self) -> None:
        self.assertIn(
            'environment_value_is_truthy("NVDS_ENABLE_LATENCY_MEASUREMENT")',
            self.source,
        )
        self.assertIn("if (self->latency_reference_enabled)", self.source)
        self.assertIn("GST_ELEMENT_NAME(self)", self.source)
        self.assertNotIn('nvds_add_reference_timestamp_meta(buffer, "', self.source)

    def test_anchor_uses_acquisition_sequence_after_timestamps_before_delivery(self) -> None:
        duration = self.source.index("GST_BUFFER_DURATION(buffer) = self->frame_duration;")
        anchor = self.source.index("nvds_add_reference_timestamp_meta(", duration)
        sequence = self.source.index("static_cast<guint>(self->frame_count)", anchor)
        delivery = self.source.index("*out_buffer = buffer;", sequence)
        increment = self.source.index("++self->frame_count;", sequence)
        self.assertLess(duration, anchor)
        self.assertLess(anchor, sequence)
        self.assertLess(sequence, increment)
        self.assertLess(increment, delivery)

    def test_anchor_fails_before_narrowing_unrepresentable_sequence(self) -> None:
        anchor = self.source.index("nvds_add_reference_timestamp_meta(")
        guard = self.source.rfind("self->frame_count > G_MAXUINT", 0, anchor)
        failure = self.source.rfind("return GST_FLOW_ERROR;", 0, anchor)
        self.assertGreater(guard, 0)
        self.assertGreater(failure, guard)


if __name__ == "__main__":
    unittest.main()
