from __future__ import annotations

import csv
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from squeakview.apps.inference.offline import (
    FrameAuditOperator,
    OfflineConfig,
    OfflineInferenceApp,
    _supervise_worker,
    _worker_command,
    _load_frame_ledger,
    _assert_direct_run_file,
)
from squeakview.apps.inference.pose_pipeline import (
    FramePoseStore, ObservationOperator, PoseClass, PoseSchema,
)


class OfflineInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_frame_ledger_maps_decoded_ordinal_to_authoritative_identity(self) -> None:
        path = self.root / "frames.csv"
        path.write_text(
            "stream_id,source_sequence_index,raw_frame_index,camera_frame_id,"
            "camera_timestamp_ns,pts_ns,source_width,source_height\n"
            "0,0,0,20338,1000000,0,1440,1080\n"
            "0,1,1,20339,1033333,33333333,1440,1080\n"
        )
        ledger, width, height = _load_frame_ledger(path)
        try:
            self.assertEqual((width, height), (1440, 1080))
            self.assertEqual(ledger[0]["source_sequence_index"], 0)
            self.assertEqual(ledger[1]["camera_frame_id"], 20339)
        finally:
            ledger.close()

    def test_frame_ledger_is_record_bounded_and_strict(self) -> None:
        header = (
            "stream_id,source_sequence_index,raw_frame_index,camera_frame_id,"
            "camera_timestamp_ns,pts_ns,source_width,source_height\n"
        )
        for body, message in (
            ("0,1.5,1,2,3,4,1440,1080\n", "unsigned integer"),
            ("0,1,1,2,3,4,1440,1080," + "x" * (64 * 1024), "record exceeds"),
        ):
            with self.subTest(message=message):
                path = self.root / f"invalid-{message.replace(' ', '-')}.csv"
                path.write_text(header + body)
                with self.assertRaisesRegex(ValueError, message):
                    _load_frame_ledger(path)

    def test_frame_ledger_rejects_identity_regression_and_dimension_change(self) -> None:
        header = (
            "stream_id,source_sequence_index,raw_frame_index,camera_frame_id,"
            "camera_timestamp_ns,pts_ns,source_width,source_height\n"
        )
        for second, message in (
            ("0,0,1,3,4,5,1440,1080\n", "not contiguous"),
            ("0,1,1,3,4,5,640,480\n", "dimensions change"),
        ):
            path = self.root / f"invalid-{message.replace(' ', '-')}.csv"
            path.write_text(header + "0,0,0,2,3,4,1440,1080\n" + second)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _load_frame_ledger(path)

    def test_frame_ledger_rejects_raw_index_frame_id_and_clock_gaps(self) -> None:
        header = (
            "stream_id,source_sequence_index,raw_frame_index,camera_frame_id,"
            "camera_timestamp_ns,pts_ns,source_width,source_height\n"
        )
        first = "0,0,0,20,1000,0,1440,1080\n"
        for second, message in (
            ("0,1,2,21,2000,10,1440,1080\n", "raw frame index"),
            ("0,1,1,22,2000,10,1440,1080\n", "FrameIDs"),
            ("0,1,1,21,1000,10,1440,1080\n", "timestamps"),
            ("0,1,1,21,2000,0,1440,1080\n", "PTS"),
        ):
            path = self.root / f"invalid-offline-{message.replace(' ', '-')}"
            path.write_text(header + first + second)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _load_frame_ledger(path)

    def test_frame_audit_rejects_noncontiguous_decoder_sequence(self) -> None:
        audit = FrameAuditOperator(3)
        audit.handle_metadata(SimpleNamespace(frame_items=[SimpleNamespace(frame_number=0)]))
        with self.assertRaisesRegex(RuntimeError, "not contiguous"):
            audit.handle_metadata(SimpleNamespace(frame_items=[SimpleNamespace(frame_number=2)]))

    def test_constructor_does_not_predecode_long_recording(self) -> None:
        run_dir = self.root / "run"
        run_dir.mkdir()
        (run_dir / "raw.mp4").write_bytes(b"video")
        (run_dir / "frames.csv").write_text("frames")
        (run_dir / "serial.csv").write_text("serial")
        config = self.root / "model.txt"
        parser = self.root / "parser.so"
        config.write_text("model")
        parser.write_bytes(b"parser")
        ledger = mock.MagicMock()
        ledger.__len__.return_value = 1
        model = SimpleNamespace(
            parser_library=parser,
            name="model",
            manifest_snapshot=lambda: {},
        )

        out_dir = self.root / "offline-output"
        with (
            mock.patch(
                "squeakview.apps.inference.offline.validate_model_package",
                return_value=model,
            ),
            mock.patch(
                "squeakview.apps.inference.offline._load_frame_ledger",
                return_value=(ledger, 1440, 1080),
            ),
        ):
            app = OfflineInferenceApp(
                OfflineConfig(run_dir, cfg_path=config, out_dir=out_dir)
            )

        self.assertEqual(app.audit.expected, 1)
        ledger.close.assert_not_called()
        app.ledger.close()

    def test_offline_inputs_reject_symlinks_and_run_directory_escape(self) -> None:
        run_dir = self.root / "run-local"
        run_dir.mkdir()
        outside = self.root / "outside.mp4"
        outside.write_bytes(b"video")
        linked = run_dir / "raw.mp4"
        linked.symlink_to(outside)

        with self.assertRaisesRegex(ValueError, "direct run-local regular file"):
            _assert_direct_run_file(run_dir.resolve(), linked)

    def test_run_wrapper_always_closes_disk_backed_ledger(self) -> None:
        app = object.__new__(OfflineInferenceApp)
        app.ledger = mock.Mock()

        with (
            mock.patch.object(
                OfflineInferenceApp, "_run_impl", side_effect=RuntimeError("failed")
            ),
            self.assertRaisesRegex(RuntimeError, "failed"),
        ):
            app.run()

        app.ledger.close.assert_called_once_with()

    def test_supervisor_command_selects_isolated_worker(self) -> None:
        command = _worker_command(
            OfflineConfig(
                self.root / "run", cfg_path=self.root / "model.txt",
                out_dir=self.root / "derived",
            )
        )
        self.assertIn(
            "squeakview.apps.operator.backend.parent_death_exec", command
        )
        self.assertIn("--worker", command)
        self.assertIn("--cfg", command)
        self.assertIn("--out-dir", command)

    def test_supervisor_force_terminates_stuck_native_teardown(self) -> None:
        worker = mock.Mock(pid=1234, returncode=None)
        worker.poll.return_value = None
        handlers: dict[int, object] = {}

        def install(signum, handler):
            if callable(handler):
                handlers[signum] = handler

        def wait(*, timeout):
            del timeout
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            raise subprocess.TimeoutExpired("offline-worker", 0.25)

        worker.wait.side_effect = wait

        def terminate(target) -> None:
            self.assertIs(target, worker)
            worker.returncode = -signal.SIGKILL
            worker.poll.return_value = worker.returncode

        with (
            mock.patch("squeakview.apps.inference.offline.signal.getsignal", return_value=None),
            mock.patch("squeakview.apps.inference.offline.signal.signal", side_effect=install),
            mock.patch("squeakview.apps.inference.offline.os.killpg") as kill_group,
            mock.patch(
                "squeakview.apps.inference.offline.time.monotonic",
                side_effect=(10.0, 12.0),
            ),
            mock.patch(
                "squeakview.apps.inference.offline._terminate_process_group",
                side_effect=terminate,
            ) as terminate_group,
        ):
            result = _supervise_worker(
                OfflineConfig(self.root / "run"),
                shutdown_timeout_s=1.0,
                popen_factory=lambda *_args, **_kwargs: worker,
            )

        self.assertEqual(result, 128 + signal.SIGKILL)
        kill_group.assert_called_once_with(1234, signal.SIGTERM)
        terminate_group.assert_called_once_with(worker)

    def test_pose_handoff_fails_closed_at_bounded_frame_capacity(self) -> None:
        store = FramePoseStore(max_frames=2)
        for frame_number in range(2):
            store.put(0, frame_number, {"detection_index": 0})

        with self.assertRaisesRegex(RuntimeError, "bounded frame capacity"):
            store.put(0, 2, {"detection_index": 0})

        store.discard(0, 0)
        self.assertEqual(store.put(0, 2, {"detection_index": 0}), "SQPOSE:2:0")

    def test_pose_handoff_rejects_duplicate_and_excess_observations(self) -> None:
        store = FramePoseStore(max_observations_per_frame=2)
        store.put(3, 7, {"detection_index": 0})
        with self.assertRaisesRegex(RuntimeError, "duplicate pose detection index"):
            store.put(3, 7, {"detection_index": 0})
        store.put(3, 7, {"detection_index": 1})
        with self.assertRaisesRegex(RuntimeError, "per-frame observation capacity"):
            store.put(3, 7, {"detection_index": 2})

    def test_observation_writer_marks_offline_ledger_mapping(self) -> None:
        schema = PoseSchema(
            version=2, input_width=640, input_height=640, output_layer="output0",
            keypoint_names=("nose",),
            classes=(PoseClass(0, "mouse", 0.25, True, (0,)),),
            keypoint_threshold=0.5,
        )
        operator = ObservationOperator(
            self.root, schema, store=FramePoseStore(), flir_meta_type=None,
            frame_ledger={0: {
                "source_sequence_index": 7, "camera_frame_id": 20338,
                "camera_timestamp_ns": 1000000, "pts_ns": 0,
            }},
            mapping_method="offline_video_ledger", source_name="offline_raw_mp4",
        )
        rect = SimpleNamespace(left=1.0, top=2.0, width=3.0, height=4.0)
        object_meta = SimpleNamespace(
            class_id=0, object_id=42, label="", rect_params=rect, tracker_confidence=0.8,
        )
        frame = SimpleNamespace(
            frame_number=0, source_id=0, pad_index=0, buffer_pts=999,
            source_width=1440, source_height=1080, pipeline_width=1440, pipeline_height=1080,
            object_items=[object_meta],
        )
        operator.handle_metadata(SimpleNamespace(frame_items=[frame]))
        operator.close()
        self.assertFalse((self.root / "detections.csv").exists())
        with (self.root / "objects.csv").open(newline="") as handle:
            obj = next(csv.DictReader(handle))
        self.assertEqual(obj["source_sequence_index"], "7")
        self.assertEqual(obj["camera_frame_id"], "20338")
        self.assertEqual(obj["gst_pts_ns"], "0")


if __name__ == "__main__":
    unittest.main()
