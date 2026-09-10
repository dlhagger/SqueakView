"""Lifecycle-neutral orchestration for the scientific capture graph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .contracts import InferenceConfig
from .frame_audit import FrameCsvOperator
from .inference_tracking_graph import (
    add_inference_input_branch,
    add_inference_tracking_branch,
)
from .pose_pipeline import ObservationOperator
from .preview_attribution import PreviewBoundaryOperator
from .preview_output_graph import add_output_branch
from . import debug_instrumentation
from .recording import RecordingAdmissionOperator, RecordingPathTelemetry
from .source_recording_graph import (
    CameraRecordingBranch,
    add_camera_recording_branch,
    camera_source_properties,
    flir_pixel_format,
)


@dataclass(frozen=True, slots=True)
class PipelineResources:
    """Pipeline plus every scientific resource whose lifecycle must be closed."""

    pipeline: object
    ready_origin: str
    frames: FrameCsvOperator
    observations: ObservationOperator | None
    admissions: tuple[RecordingAdmissionOperator, ...]
    recording_telemetry: tuple[RecordingPathTelemetry, ...]
    preview_boundaries: tuple[PreviewBoundaryOperator, ...]


def build_pipeline(
    config: InferenceConfig,
    run_dir: Path,
    primary_raw_video: Path,
    *,
    pipeline_factory: Callable[[str], object],
    probe_factory: Callable[[str, object], object],
    on_recording_fault: Callable[[str], None],
) -> PipelineResources:
    """Build the exact audited graph and return explicit lifecycle ownership."""

    run_dir = Path(run_dir)
    admissions: list[RecordingAdmissionOperator] = []
    telemetry: list[RecordingPathTelemetry] = []
    preview_boundaries: list[PreviewBoundaryOperator] = []
    frames: FrameCsvOperator | None = None
    observations: ObservationOperator | None = None
    try:
        pipeline = pipeline_factory("squeakview")
        pipeline.add(
            "nvstreammux",
            "mux",
            {
                "batch-size": int(config.num_cameras),
                "batched-push-timeout": max(
                    10_000, int(1_000_000 / int(config.fps))
                ),
                "sync-inputs": bool(config.num_cameras > 1),
                # NVIDIA requires max-latency > one frame interval whenever
                # new-nvstreammux input synchronization is enabled.
                "max-latency": (
                    int(2_000_000_000 / int(config.fps))
                    if config.num_cameras > 1
                    else 0
                ),
            },
        )
        for index in range(config.num_cameras):
            recording = add_camera_recording_branch(
                pipeline,
                config,
                run_dir,
                primary_raw_video,
                index,
                probe_factory=probe_factory,
                on_recording_fault=on_recording_fault,
                register_admission=admissions.append,
                register_telemetry=telemetry.append,
            )
            add_inference_input_branch(pipeline, config, index, recording.tee)

        inference_dir = run_dir / "inference"
        inference_dir.mkdir(parents=True, exist_ok=True)
        frames = FrameCsvOperator(
            inference_dir / "frames.csv",
            write_audit_sidecars=False,
            max_cameras=config.num_cameras,
        )
        pipeline.attach("mux", probe_factory("frames", frames))
        tail, observations = add_inference_tracking_branch(
            pipeline,
            config,
            run_dir,
            frames,
            probe_factory=probe_factory,
        )
        if debug_instrumentation.profile_enabled():
            # Both probes remain downstream of each camera's leaky inference
            # queue, so qualification logging cannot backpressure recording.
            debug_instrumentation.attach_debug_probes(pipeline, tail[-1])
        ready_origin = add_output_branch(
            pipeline,
            config,
            tail,
            run_dir=run_dir,
            meta_type=frames.meta_type,
            probe_factory=probe_factory,
            register_boundary=preview_boundaries.append,
        )
        return PipelineResources(
            pipeline=pipeline,
            ready_origin=ready_origin,
            frames=frames,
            observations=observations,
            admissions=tuple(admissions),
            recording_telemetry=tuple(telemetry),
            preview_boundaries=tuple(preview_boundaries),
        )
    except Exception:
        closers = []
        if observations is not None:
            closers.append(observations.close)
        closers.extend(admission.close for admission in admissions)
        closers.extend(recorder.close for recorder in telemetry)
        closers.extend(boundary.close for boundary in preview_boundaries)
        if frames is not None:
            closers.append(frames.close)
        for close in closers:
            try:
                close()
            except Exception:
                pass
        raise


# Compatibility exports preserve callers that imported branch builders from
# this former monolithic module while implementation ownership stays narrow.
__all__ = [
    "CameraRecordingBranch",
    "PipelineResources",
    "add_camera_recording_branch",
    "add_inference_input_branch",
    "add_inference_tracking_branch",
    "add_output_branch",
    "build_pipeline",
    "camera_source_properties",
    "flir_pixel_format",
]
