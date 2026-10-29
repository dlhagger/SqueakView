from __future__ import annotations

import csv
import json
import tempfile
import threading
import tracemalloc
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from squeakview.apps.inference import service_maker_runner as runner
from squeakview.apps.inference import (
    contracts,
    debug_instrumentation,
    frame_audit,
    main as inference_main,
    pipeline_builder,
    post_run,
    video_probe,
)
from squeakview.apps.inference.pose_pipeline import PoseClass, PoseSchema, decode_yolo26_rows, load_pose_schema
from squeakview.common import run_context
from squeakview.common.capture_policy import capture_buffer_policy
from squeakview.common.child_events import decode_child_event
from squeakview.common.failure_injection import FailurePlan


class FakePipeline:
    def __init__(self, name: str):
        self.name = name
        self.nodes: dict[str, tuple[str, dict]] = {}
        self.links: list[tuple] = []
        self.attachments: list[tuple] = []
        self.stopped = False
        self.waited = False
        self.started = False

    def add(self, type_name: str, name: str, properties: dict | None = None):
        self.nodes[name] = (type_name, properties or {})
        return self

    def link(self, *args):
        self.links.append(args)
        return self

    def attach(self, target: str, what, name="", tips="", properties=None):
        self.attachments.append((target, what, name, tips, properties))
        return self

    def stop(self):
        self.stopped = True
        return self

    def start(self, _callback):
        self.started = True
        return self

    def wait(self):
        self.waited = True
        return self



class ServiceMakerCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        labels = self.root / "classes.txt"
        labels.write_text("mouse\nbottle\n")
        self.infer_config = self.root / "infer.txt"
        self.infer_config.write_text(
            "[property]\n"
            "batch-size=1\n"
            "labelfile-path=classes.txt\n"
        )
        keypoints = self.root / "keypoints.txt"
        keypoints.write_text("nose\nbottle_tip\n")
        self.pose_sidecar = self.root / "infer.pose.json"
        self.pose_sidecar.write_text(json.dumps({
            "schema_version": 2,
            "task": "pose",
            "postprocess": "pyservicemaker_yolo26_pose_v1",
            "output_layer": "output0",
            "input_width": 640,
            "input_height": 640,
            "letterbox": "symmetric",
            "end2end": True,
            "keypoint_labels_path": str(keypoints),
            "keypoint_count": 2,
            "keypoint_dims": 3,
            "keypoint_threshold": 0.5,
            "classes": [
                {"id": 0, "name": "mouse", "threshold": 0.25, "track": True, "keypoint_indices": [0]},
                {"id": 1, "name": "bottle", "threshold": 0.25, "track": False, "keypoint_indices": [1]},
            ],
        }, indent=2) + "\n")


    def tearDown(self) -> None:
        self.temp_dir.cleanup()


    def config(self, **overrides) -> runner.InferenceConfig:
        values = {
            "cfg_path": self.infer_config,
            "run_dir": self.root / "run",
            "width": 640,
            "height": 480,
            "fps": 30,
            "bitrate": 4000,
        }
        values.update(overrides)
        return runner.InferenceConfig(**values)


    def build(self, config: runner.InferenceConfig) -> tuple[runner.ServiceMakerApp, FakePipeline]:
        app = runner.ServiceMakerApp(
            config,
            pipeline_factory=FakePipeline,
            probe_factory=lambda name, operator: (name, operator),
        )
        pipeline = app.build()
        return app, pipeline


