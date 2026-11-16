"""Source drain requests never call the vendor's GIL-holding stop binding."""
import unittest
from unittest import mock

from squeakview.apps.inference.capture_pipeline import CapturePipeline, Pipeline


class CapturePipelineTests(unittest.TestCase):
    def make_pipeline(self, nodes):
        with mock.patch.object(Pipeline, "__init__", return_value=None):
            pipeline = CapturePipeline("test")
        self.enterContext(mock.patch.object(Pipeline, "__getitem__", side_effect=nodes.__getitem__))
        self.enterContext(mock.patch.object(Pipeline, "add", return_value=pipeline))
        vendor_stop = self.enterContext(mock.patch.object(Pipeline, "stop", side_effect=AssertionError("unsafe stop")))
        self.addCleanup(vendor_stop.assert_not_called)
        return pipeline

    def test_all_sources_are_drained_without_vendor_stop(self):
        nodes = {name: mock.Mock() for name in ("camera0", "camera1")}
        for node in nodes.values():
            node.get.side_effect = [False, True, True]
        pipeline = self.make_pipeline(nodes)
        pipeline.add("queue", "queue")
        for name in nodes:
            pipeline.add("flirspinsrc", name)
        pipeline.stop()
        pipeline.stop()
        for node in nodes.values():
            self.assertEqual(node.set.call_args_list, [mock.call({"graceful-eos": True})] * 2)

    def test_one_failed_source_does_not_skip_other_sources(self):
        first, second = mock.Mock(), mock.Mock()
        first.get.return_value = False
        second.get.side_effect = [False, True]
        pipeline = self.make_pipeline({"a": first, "b": second})
        pipeline.add("flirspinsrc", "a").add("flirspinsrc", "b")
        first.set.side_effect = OSError("unavailable")
        with self.assertRaisesRegex(RuntimeError, "a: unavailable"):
            pipeline.stop()
        second.set.assert_called_once_with({"graceful-eos": True})

    def test_missing_control_rejected_before_start(self):
        node = mock.Mock()
        node.get.return_value = None
        pipeline = self.make_pipeline({"camera": node})
        with self.assertRaisesRegex(RuntimeError, "rebuild"):
            pipeline.add("flirspinsrc", "camera")

    def test_unacknowledged_stop_is_a_failure(self):
        node = mock.Mock()
        node.get.return_value = False
        pipeline = self.make_pipeline({"camera": node})
        pipeline.add("flirspinsrc", "camera")
        with self.assertRaisesRegex(RuntimeError, "did not acknowledge"):
            pipeline.stop()
