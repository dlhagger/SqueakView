"""Scientific capture buffering policy shared by runtime and run manifests.

The recording queue is intentionally bounded but non-leaky.  Reaching the
failure threshold stops acquisition before the queue itself becomes full; it
must never be "fixed" by enabling a leaky queue.
"""

from __future__ import annotations

from dataclasses import dataclass


RECORD_QUEUE_SECONDS = 4
RECORD_BACKPRESSURE_WARNING_SECONDS = 1
RECORD_BACKPRESSURE_FATAL_SECONDS = 3
SOURCE_TRANSPORT_BUFFER_SECONDS = 2

MIN_RECORD_QUEUE_FRAMES = 120
MIN_RECORD_WARNING_FRAMES = 24
MIN_RECORD_FATAL_FRAMES = 90
MIN_SOURCE_TRANSPORT_BUFFERS = 64

INFERENCE_QUEUE_FRAMES = 32
PREVIEW_QUEUE_FRAMES = 1
DOWNSTREAM_LEAKY = 2
NON_LEAKY = 0


@dataclass(frozen=True, slots=True)
class CaptureBufferPolicy:
    """Frame capacities for one camera at a configured acquisition rate."""

    source_transport_buffers: int
    record_queue_frames: int
    record_warning_frames: int
    record_failure_frames: int
    inference_queue_frames: int = INFERENCE_QUEUE_FRAMES
    preview_queue_frames: int = PREVIEW_QUEUE_FRAMES

    def __post_init__(self) -> None:
        if not 0 < self.record_warning_frames < self.record_failure_frames:
            raise ValueError("recording warning threshold must precede failure threshold")
        if self.record_failure_frames >= self.record_queue_frames:
            raise ValueError("recording failure threshold must precede queue capacity")


def capture_buffer_policy(fps: int) -> CaptureBufferPolicy:
    """Return the audited buffering policy for a positive integer frame rate."""

    frame_rate = int(fps)
    if frame_rate <= 0:
        raise ValueError("fps must be greater than zero")
    return CaptureBufferPolicy(
        source_transport_buffers=max(
            MIN_SOURCE_TRANSPORT_BUFFERS,
            frame_rate * SOURCE_TRANSPORT_BUFFER_SECONDS,
        ),
        record_queue_frames=max(
            MIN_RECORD_QUEUE_FRAMES,
            frame_rate * RECORD_QUEUE_SECONDS,
        ),
        record_warning_frames=max(
            MIN_RECORD_WARNING_FRAMES,
            frame_rate * RECORD_BACKPRESSURE_WARNING_SECONDS,
        ),
        record_failure_frames=max(
            MIN_RECORD_FATAL_FRAMES,
            frame_rate * RECORD_BACKPRESSURE_FATAL_SECONDS,
        ),
    )


def non_leaky_record_queue_properties(policy: CaptureBufferPolicy) -> dict[str, int]:
    """Return explicit GStreamer properties for the loss-intolerant branch."""

    return {
        "max-size-buffers": policy.record_queue_frames,
        "max-size-bytes": 0,
        "max-size-time": 0,
        "leaky": NON_LEAKY,
        "flush-on-eos": False,
    }


def leaky_inference_queue_properties(policy: CaptureBufferPolicy) -> dict[str, int]:
    """Return bounded latest-frame flow control for optional inference."""

    return {
        "max-size-buffers": policy.inference_queue_frames,
        "max-size-bytes": 0,
        "max-size-time": 0,
        "leaky": DOWNSTREAM_LEAKY,
    }
