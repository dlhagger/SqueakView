"""Source drain requests never call the vendor's GIL-holding stop binding."""
import unittest
from unittest import mock

from tests.support.vendor_module import load_inference_module
module = load_inference_module('capture_pipeline')
CapturePipeline, Pipeline = module.CapturePipeline, module.Pipeline


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
            node.get.side_effect = [False, True, True, True]
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
        first.get.side_effect = [False, True]
        second.get.side_effect = [False, True, True]
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
        node.get.side_effect = [False, True, False]
        pipeline = self.make_pipeline({"camera": node})
        pipeline.add("flirspinsrc", "camera")
        with self.assertRaisesRegex(RuntimeError, "did not acknowledge"):
            pipeline.stop()

    def test_wait_without_eos_never_confirms_closure(self):
        pipeline = self.make_pipeline({})
        with mock.patch.object(Pipeline, 'wait'):
            with self.assertRaisesRegex(RuntimeError, 'EOS'):
                pipeline.wait()
        self.assertFalse(pipeline.closure_confirmed)

    def test_eos_without_null_transition_never_confirms_closure(self):
        node = mock.Mock()
        node.get.side_effect = [False, True, False]
        pipeline = self.make_pipeline({'camera': node})
        pipeline.add('flirspinsrc', 'camera')
        pipeline._eos_seen.set()
        with mock.patch.object(Pipeline, 'wait'):
            with self.assertRaisesRegex(RuntimeError, 'NULL transition'):
                pipeline.wait()
        self.assertFalse(pipeline.closure_confirmed)

    def test_start_tracks_eos_and_wait_checks_all_sources(self):
        nodes = {name: mock.Mock() for name in ('a', 'b')}
        for node in nodes.values():
            node.get.side_effect = [False, True, True]
        pipeline = self.make_pipeline(nodes)
        for name in nodes:
            pipeline.add('flirspinsrc', name)
        class EOS: pass
        callback = mock.Mock()
        with mock.patch.object(Pipeline, 'start') as start, mock.patch.object(module, 'EOSMessage', EOS):
            pipeline.start(callback)
            message = EOS()
            start.call_args.args[0](message)
        callback.assert_called_once_with(message)
        with mock.patch.object(Pipeline, 'wait'):
            pipeline.wait()
        self.assertTrue(pipeline.closure_confirmed)

    def test_stale_null_verification_property_is_rejected_at_build(self):
        node = mock.Mock()
        node.get.side_effect = [False, None]
        pipeline = self.make_pipeline({'camera': node})
        with self.assertRaisesRegex(RuntimeError, 'NULL verification'):
            pipeline.add('flirspinsrc', 'camera')
