"""Scientific validation of ground-truth recording artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from squeakview.common import run_context
from squeakview.common.recording_evidence import (
    MAX_RECORDING_CAMERAS,
    capture_recording_evidence,
    recording_evidence_complete,
    recording_evidence_matches,
)


VideoProbe = Callable[[Path], Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class CameraRecordingResult:
    stream_id: int
    video: str
    exists: bool
    source_frames: int
    record_admitted_frames: int
    source_count_matches: bool
    nonzero_frame_count: bool
    video_frames: int | None
    frame_count_matches: bool | None
    frame_count_method: object
    frame_count_error: object
    frame_count_warning: object

    @property
    def passed(self) -> bool:
        return (
            self.exists
            and self.nonzero_frame_count
            and self.source_count_matches
            and self.frame_count_matches is True
        )

    def to_dict(self) -> dict:
        return {
            "stream_id": self.stream_id,
            "video": self.video,
            "exists": self.exists,
            "source_frames": self.source_frames,
            "record_admitted_frames": self.record_admitted_frames,
            "source_count_matches": self.source_count_matches,
            "nonzero_frame_count": self.nonzero_frame_count,
            "video_frames": self.video_frames,
            "frame_count_matches": self.frame_count_matches,
            "frame_count_method": self.frame_count_method,
            "frame_count_error": self.frame_count_error,
            "frame_count_warning": self.frame_count_warning,
        }


@dataclass(frozen=True, slots=True)
class RecordingValidationResult:
    cameras: tuple[CameraRecordingResult, ...]
    evidence: Mapping[str, object]
    evidence_unchanged_during_validation: bool

    @property
    def passed(self) -> bool:
        return (
            bool(self.cameras)
            and all(camera.passed for camera in self.cameras)
            and recording_evidence_complete(self.evidence, len(self.cameras))
            and self.evidence_unchanged_during_validation
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": "2.0",
            "policy": (
                "every_source_frame_must_be_record_admitted_and_present_in_"
                "ground_truth_video"
            ),
            "cameras": [camera.to_dict() for camera in self.cameras],
            "evidence": dict(self.evidence),
            "evidence_unchanged_during_validation": (
                self.evidence_unchanged_during_validation
            ),
            "passed": self.passed,
        }


def validate_recordings(
    run_dir: Path,
    camera_count: int,
    source_counts: Mapping[int, int],
    recorded_counts: Mapping[int, int],
    *,
    probe: VideoProbe,
    evidence: Mapping[str, object] | None = None,
) -> RecordingValidationResult:
    if type(camera_count) is not int or not 1 <= camera_count <= MAX_RECORDING_CAMERAS:
        raise ValueError(
            f"camera_count must be an integer from 1 through {MAX_RECORDING_CAMERAS}"
        )
    run_dir = Path(run_dir).resolve()
    artifacts = run_context.run_artifacts(run_dir)
    initial_evidence = (
        capture_recording_evidence(run_dir, camera_count)
        if evidence is None
        else evidence
    )
    cameras: list[CameraRecordingResult] = []
    for stream_id in range(camera_count):
        video_path = (
            artifacts.raw_video
            if stream_id == 0
            else run_dir / f"raw_cam{stream_id}.mp4"
        )
        source_frames = source_counts.get(stream_id, 0)
        admitted_frames = recorded_counts.get(stream_id, 0)
        probe_result = probe(video_path)
        raw_video_frames = probe_result["count"]
        video_frames = (
            raw_video_frames
            if type(raw_video_frames) is int and raw_video_frames >= 0
            else None
        )
        exists = video_path.is_file() and video_path.stat().st_size > 0
        cameras.append(
            CameraRecordingResult(
                stream_id=stream_id,
                video=video_path.name,
                exists=exists,
                source_frames=source_frames,
                record_admitted_frames=admitted_frames,
                source_count_matches=source_frames == admitted_frames,
                nonzero_frame_count=(
                    source_frames > 0
                    and admitted_frames > 0
                    and video_frames is not None
                    and video_frames > 0
                ),
                video_frames=video_frames,
                frame_count_matches=(
                    video_frames == admitted_frames if video_frames is not None else None
                ),
                frame_count_method=probe_result["method"],
                frame_count_error=(
                    probe_result["error"]
                    if video_frames is not None or raw_video_frames is None
                    else "decoder returned a non-integer or negative frame count"
                ),
                frame_count_warning=probe_result.get("warning"),
            )
        )
    unchanged = recording_evidence_matches(
        run_dir, initial_evidence, camera_count
    )
    return RecordingValidationResult(tuple(cameras), initial_evidence, unchanged)
