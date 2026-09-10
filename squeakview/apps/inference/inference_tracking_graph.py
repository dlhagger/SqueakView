"""Leaky inference-input and optional pose/tracking graph construction."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from squeakview.common.capture_policy import (
    capture_buffer_policy,
    leaky_inference_queue_properties,
)

from .contracts import InferenceConfig, load_class_names
from .frame_audit import FrameCsvOperator
from .pose_pipeline import (
    FramePoseStore,
    ObservationOperator,
    Yolo26PoseTensorOperator,
    load_pose_schema,
)


def add_inference_input_branch(
    pipeline,
    config: InferenceConfig,
    index: int,
    tee: str,
) -> None:
    """Add the downstream-leaky camera-to-mux branch for one camera."""

    buffer_policy = capture_buffer_policy(config.fps)
    infer_queue = f"infer_queue{index}"
    infer_caps = f"infer_caps{index}"
    pipeline.add(
        "queue", infer_queue, leaky_inference_queue_properties(buffer_policy)
    )
    pipeline.add(
        "nvvideoconvert",
        f"infer_convert{index}",
        {
            "compute-hw": 2,
            "copy-hw": 2,
            # The new nvstreammux does not transform its inputs.  Pin the
            # Jetson VIC output to surface-array NVMM instead of inheriting a
            # DeepStream default that may change between releases.
            "nvbuf-memory-type": 4,
        },
    )
    pipeline.add(
        "capsfilter",
        infer_caps,
        {
            "caps": (
                f"video/x-raw(memory:NVMM),format=NV12,width={config.width},"
                f"height={config.height}"
            )
        },
    )
    pipeline.link(tee, infer_queue, f"infer_convert{index}", infer_caps)
    pipeline.link((infer_caps, "mux"), ("", "sink_%u"))


def add_inference_tracking_branch(
    pipeline,
    config: InferenceConfig,
    run_dir: Path,
    frames: FrameCsvOperator,
    *,
    probe_factory: Callable[[str, object], object],
) -> tuple[list[str], ObservationOperator | None]:
    """Add optional inference/tracking and return the shared output tail."""

    tail = ["mux"]
    if not config.enable_infer:
        return tail, None

    assert config.cfg_path is not None
    class_names = load_class_names(Path(config.cfg_path))
    pose_schema = load_pose_schema(Path(config.cfg_path), class_names)
    pose_store = FramePoseStore()
    pipeline.add(
        "nvinfer",
        "infer",
        {
            "config-file-path": str(Path(config.cfg_path).resolve()),
            "batch-size": config.num_cameras,
            "filter-out-class-ids": ";".join(
                str(item.class_id) for item in pose_schema.classes
            ),
        },
    )
    tail.append("infer")
    pipeline.attach(
        "infer",
        probe_factory(
            "yolo26_pose", Yolo26PoseTensorOperator(pose_schema, pose_store)
        ),
    )

    tracker_config = (
        Path(__file__).resolve().parents[3] / "configs" / "tracker_mouse_nvdcf.yml"
    )
    pipeline.add(
        "nvtracker",
        "tracker",
        {
            "tracker-width": 640,
            "tracker-height": 480,
            "ll-lib-file": (
                "/opt/nvidia/deepstream/deepstream/lib/"
                "libnvds_nvmultiobjecttracker.so"
            ),
            "ll-config-file": str(tracker_config),
            "operate-on-class-ids": ";".join(
                str(item.class_id) for item in pose_schema.classes if item.track
            ),
            "display-tracking-id": False,
            "tracking-id-reset-mode": 3,
        },
    )
    tail.append("tracker")
    observations = ObservationOperator(
        Path(run_dir),
        pose_schema,
        store=pose_store,
        flir_meta_type=frames.meta_type,
    )
    pipeline.attach("tracker", probe_factory("observations", observations))
    return tail, observations


__all__ = ["add_inference_input_branch", "add_inference_tracking_branch"]
