"""Bounded startup confirmation from each camera's native frame evidence."""

from pathlib import Path
import time
from typing import Callable

from squeakview.common.capture_health import health_path, read_capture_health


FIRST_FRAME_TIMEOUT_S = 10.0


def wait_for_first_frames(
    run_dir: Path,
    camera_count: int,
    *,
    deadline: float,
    cancelled: Callable[[], bool],
    capture_running: Callable[[], bool],
    wait: Callable[[float], object],
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[int, int]:
    """Confirm actual images without timing out the pre-trigger preparation.

    Native health is atomically replaced and bounded in size. A positive count
    follows successful image/timing validation and frame-manifest publication.
    Missing files and zero counts mean waiting; malformed/closed evidence fails.
    """
    pending = set(range(camera_count))
    counts: dict[int, int] = {}

    def check_active() -> None:
        if cancelled():
            raise RuntimeError("startup cancelled while waiting for camera frames")
        if not capture_running():
            raise RuntimeError("capture exited before first-frame confirmation")

    while True:
        check_active()
        if monotonic() >= deadline:
            raise TimeoutError(
                "first-frame deadline expired; no validated frame from camera(s): "
                + ", ".join(str(index) for index in sorted(pending))
            )
        for index in sorted(pending):
            # Current capture always writes native health. Do not interpret an
            # absent snapshot as legacy evidence or accept an unrelated log.
            if not health_path(run_dir, index).exists():
                continue
            snapshot = read_capture_health(run_dir, index)
            if snapshot['state'] != 'recording':
                raise RuntimeError(f"camera {index} closed during startup")
            count = snapshot['source_frames']
            if count > 0:
                if not snapshot['native_timing_validated']:
                    raise RuntimeError(f"camera {index} has no validated native timing")
                counts[index] = count
        pending.difference_update(counts)
        check_active()
        if not pending and monotonic() < deadline:
            return counts
        wait(min(0.05, max(0.0, deadline - monotonic())))
