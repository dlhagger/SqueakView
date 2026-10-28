"""Protocol-aware derived timing reports; scientific source files are read-only."""
from __future__ import annotations
import csv
from pathlib import Path
from . import run_context
from .bounded_csv import bounded_csv_lines
from .run_layout import artifact_path


def summarize_recording_timing(run_dir: Path, output_dir: Path, *, objects_path=None, video_validation=None) -> dict:
    manifest_path = artifact_path(run_dir, "run_manifest.json")
    manifest = run_context.read_json_required(manifest_path) if manifest_path.exists() else {}
    serial = manifest.get("serial", {})
    capture = manifest.get("capture", {})
    enabled = serial.get("enabled", True)
    triggered = capture.get("trigger_on", True)
    protocol = serial.get("controller_protocol", "legacy")
    if type(enabled) is not bool or type(triggered) is not bool:
        raise ValueError("recording modes must be boolean")
    if enabled and protocol not in {"legacy", "watchdog_v1_experimental", "v2"}:
        raise ValueError("unsupported controller protocol")
    if enabled and protocol != "v2" and triggered:
        from squeakview.apps.inference.legacy_alignment import build_alignment
        return build_alignment(run_dir, output_dir, objects_path=objects_path,
                               video_validation=video_validation, include_objects=objects_path is not None)
    camera_count = capture.get("num_cameras", 1)
    if type(camera_count) is not int or not 1 <= camera_count <= 64:
        raise ValueError("invalid camera count in recording manifest")
    if enabled and serial.get("controller_protocol") == "v2":
        from .v2_timing import verify_v2_frame_timing
        report = verify_v2_frame_timing(run_dir, triggered=triggered, camera_count=camera_count)
    else:
        cameras = []
        for stream in range(camera_count):
            path = artifact_path(run_dir, "frames.csv" if stream == 0 else f"frames_cam{stream}.csv")
            count = 0
            previous = None
            for row in csv.DictReader(bounded_csv_lines(path)):
                if (int(row["stream_id"]) != stream or int(row["source_sequence_index"]) != count or int(row["raw_frame_index"]) != count):
                    raise ValueError("native frame identities are not contiguous")
                stamp = int(row["camera_timestamp_ns"])
                if stamp < 0 or (previous is not None and stamp <= previous):
                    raise ValueError("native camera clock is not strictly increasing")
                previous = stamp
                count += 1
            if not count:
                raise ValueError("recording has no native frame timestamps")
            cameras.append({"stream_id": stream, "frames": count})
        report = {"passed": True, "method": "camera_native", "cameras": cameras}
    if video_validation is not None and camera_count == 1 and video_validation.get("count") != report["cameras"][0]["frames"]:
        raise ValueError("decoded video/frame timing counts disagree")
    summary = {"schema_version": "3.0", "run_dir": str(Path(run_dir).resolve()),
               "timing": report, "counts": {"recorded_frames": sum(camera["frames"] for camera in report["cameras"])},
               "validation": {"passed": True, "controller_trigger_alignment_required": bool(enabled and triggered)}}
    run_context.atomic_write_json(Path(output_dir) / "alignment_summary.json", summary)
    return summary
