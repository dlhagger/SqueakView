"""FLIR source and loss-intolerant recording graph construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from squeakview.common.capture_policy import (
    capture_buffer_policy,
    non_leaky_record_queue_properties,
)

from .contracts import InferenceConfig
from .recording import (
    RecordingAdmissionOperator,
    RecordingEgressOperator,
    RecordingIngressOperator,
    RecordingPathTelemetry,
    RecordingStallOperator,
)


def flir_pixel_format(value: str | None) -> str:
    """Normalize the GStreamer alias to the FLIR/Spinnaker pixel format."""

    pixel_format = str(value or "Mono8").strip()
    return "Mono8" if pixel_format.upper() == "GRAY8" else pixel_format or "Mono8"


def camera_source_properties(
    config: InferenceConfig,
    run_dir: Path,
    index: int,
) -> dict[str, object]:
    """Return the exact audited properties for one FLIR source element."""

    buffer_policy = capture_buffer_policy(config.fps)
    properties: dict[str, object] = {
        "camera-index": index,
        "width": int(config.width),
        "height": int(config.height),
        "fps": int(config.fps),
        "pixel-format": flir_pixel_format(config.pixel_format),
        "trigger": bool(config.trigger_on),
        "trigger-activation": (
            "falling"
            if str(config.trigger_activation).lower().startswith("fall")
            else "rising"
        ),
        "exposure-us": (
            -1.0 if config.exposure_us is None else float(config.exposure_us)
        ),
        "gain": -1.0 if config.gain is None else float(config.gain),
        "drop-incomplete": False,
        "buffer-handling": "OldestFirst",
        "stream-buffer-count": buffer_policy.source_transport_buffers,
        "capture-log-path": str(Path(run_dir) / f"capture_cam{index}.jsonl"),
        "metadata-profile": "scientific",
        "max-consecutive-timeouts": 0 if config.trigger_on else 10,
    }
    if config.camera_serials:
        properties["camera-serial"] = config.camera_serials[index]
    if (
        config.failure_plan is not None
        and config.failure_plan.target == "flir_source"
        and config.failure_plan.stream_id == index
    ):
        properties["fault-after-frames"] = config.failure_plan.after_frames
        properties["fault-kind"] = config.failure_plan.kind
    return properties


@dataclass(frozen=True, slots=True)
class CameraRecordingBranch:
    """Owned operators and connection points created for one camera."""

    source: str
    source_caps: str
    tee: str
    admission: RecordingAdmissionOperator
    telemetry: RecordingPathTelemetry


def add_camera_recording_branch(
    pipeline,
    config: InferenceConfig,
    run_dir: Path,
    primary_raw_video: Path,
    index: int,
    *,
    probe_factory: Callable[[str, object], object],
    on_recording_fault: Callable[[str], None],
    register_admission: Callable[[RecordingAdmissionOperator], None],
    register_telemetry: Callable[[RecordingPathTelemetry], None],
) -> CameraRecordingBranch:
    """Add one FLIR source, tee, and complete loss-intolerant recording branch."""

    run_dir = Path(run_dir)
    buffer_policy = capture_buffer_policy(config.fps)
    source = f"flirsrc{index}"
    source_caps = f"source_caps{index}"
    tee = f"camera_tee{index}"
    record_queue = f"record_queue{index}"
    raw_path = primary_raw_video if index == 0 else run_dir / f"raw_cam{index}.mp4"
    failure = config.failure_plan
    failure_applies = failure is not None and failure.stream_id == index

    pipeline.add(
        "flirspinsrc", source, camera_source_properties(config, run_dir, index)
    )
    pipeline.add(
        "capsfilter",
        source_caps,
        {
            "caps": (
                f"video/x-raw,format=GRAY8,width={config.width},"
                f"height={config.height},framerate={config.fps}/1"
            )
        },
    )
    pipeline.add("tee", tee).link(source, source_caps, tee)

    pipeline.add(
        "queue", record_queue, non_leaky_record_queue_properties(buffer_policy)
    )
    admission_path = (
        run_dir / "record_admission.csv"
        if index == 0
        else run_dir / f"record_admission_cam{index}.csv"
    )
    diagnostics_dir = run_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = diagnostics_dir / (
        "recording.csv" if index == 0 else f"recording_cam{index}.csv"
    )
    telemetry = RecordingPathTelemetry(
        telemetry_path,
        index,
        warning_depth=buffer_policy.record_warning_frames,
        fatal_depth=buffer_policy.record_failure_frames,
        on_fatal=on_recording_fault,
    )
    register_telemetry(telemetry)
    pipeline.attach(
        source_caps,
        probe_factory(f"record_ingress{index}", RecordingIngressOperator(telemetry)),
    )
    admission = RecordingAdmissionOperator(admission_path, index, telemetry)
    register_admission(admission)
    pipeline.attach(
        record_queue,
        probe_factory(f"record_admission{index}", admission),
    )
    pipeline.add(
        "x264enc",
        f"record_encoder{index}",
        {
            "tune": 4,
            "speed-preset": 1,
            "bitrate": int(config.bitrate),
            "key-int-max": int(config.fps),
            # Match NVIDIA's tuned Orin Nano software-encode GOP: one
            # reference frame and no adaptive quantization.  Keep these
            # explicit so an x264 preset/default change cannot add hidden
            # temporal work to the loss-intolerant recording branch.
            "ref": 1,
            "option-string": "aq-mode=0",
            "bframes": 0,
            "rc-lookahead": 0,
            "sync-lookahead": 0,
            "sliced-threads": False,
            "vbv-buf-capacity": 100,
            "qos": False,
        },
    )
    pipeline.add("h264parse", f"record_parser{index}")
    pipeline.attach(
        f"record_parser{index}",
        probe_factory(f"record_egress{index}", RecordingEgressOperator(telemetry)),
    )
    pipeline.add("mp4mux", f"record_muxer{index}")
    pipeline.add(
        "filesink",
        f"record_sink{index}",
        {
            "location": (
                "/dev/full"
                if failure_applies
                and failure.target == "filesink"
                and failure.kind == "disk_full"
                else str(raw_path)
            ),
            "qos": False,
            "sync": False,
        },
    )
    record_chain: list[str] = [tee, record_queue]
    if failure_applies and failure.target == "record_queue":
        fault_name = f"fault_record_queue{index}"
        pipeline.add("identity", fault_name)
        pipeline.attach(
            fault_name,
            probe_factory(
                f"fault_record_queue_stall{index}",
                RecordingStallOperator(failure.after_frames, failure.delay_us),
            ),
        )
        record_chain.append(fault_name)
    if failure_applies and failure.target == "encoder":
        fault_name = f"fault_encoder{index}"
        pipeline.add("identity", fault_name, {"error-after": failure.after_frames})
        record_chain.append(fault_name)
    record_chain.append(f"record_encoder{index}")
    if failure_applies and failure.target == "muxer":
        fault_name = f"fault_muxer{index}"
        pipeline.add("identity", fault_name, {"error-after": failure.after_frames})
        record_chain.append(fault_name)
    record_chain.extend((f"record_parser{index}", f"record_muxer{index}"))
    if (
        failure_applies
        and failure.target == "filesink"
        and failure.kind == "error"
    ):
        fault_name = f"fault_filesink{index}"
        pipeline.add("identity", fault_name, {"error-after": failure.after_frames})
        record_chain.append(fault_name)
    record_chain.append(f"record_sink{index}")
    pipeline.link(*record_chain)
    return CameraRecordingBranch(source, source_caps, tee, admission, telemetry)


__all__ = [
    "CameraRecordingBranch",
    "add_camera_recording_branch",
    "camera_source_properties",
    "flir_pixel_format",
]
