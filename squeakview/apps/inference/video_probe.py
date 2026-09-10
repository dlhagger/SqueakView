"""Bounded full-decode validation of recorded scientific video."""

from __future__ import annotations

import json
import math
import os
import select
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

from .mp4_sample_table import read_video_sample_table


MAX_VALIDATOR_OUTPUT_BYTES = 64 * 1024
MAX_VALIDATOR_LINE_BYTES = 4096
# Leave one hour of the supervisor's default six-hour finalization window for
# hashing, acquisition validation, status persistence, and orderly teardown.
DEFAULT_VIDEO_VALIDATION_TIMEOUT_S = 5 * 60 * 60
JETSON_H264_DECODER = "h264_nvv4l2dec"
GSTREAMER_H264_DECODER = "nvv4l2decoder"
GSTREAMER_METHOD = "full_decode_gstreamer_nvv4l2decoder"
STRUCTURAL_METHOD = "mp4_sample_table_plus_full_h264_parse"
VALIDATION_OUTPUT_FILTER = "settb=1/1000000,setpts=N"
VALIDATION_PROGRESS_PERIOD_S = 2
DecodeProgressCallback = Callable[[int], None]


def _timeout_seconds() -> float:
    try:
        value = float(
            os.environ.get(
                "SQUEAKVIEW_VIDEO_VALIDATION_TIMEOUT_S",
                str(DEFAULT_VIDEO_VALIDATION_TIMEOUT_S),
            )
        )
    except (TypeError, ValueError):
        value = float(DEFAULT_VIDEO_VALIDATION_TIMEOUT_S)
    if not math.isfinite(value):
        value = float(DEFAULT_VIDEO_VALIDATION_TIMEOUT_S)
    return min(max(value, 1.0), 24 * 60 * 60)


def _terminate_group(worker: subprocess.Popen) -> None:
    try:
        os.killpg(worker.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        worker.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(worker.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        worker.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _decode_command(ffmpeg: str, path: Path) -> list[str]:
    """Build the explicit Jetson decode-to-null command.

    NVIDIA's reduced JetPack 7.2.1 FFmpeg package also advertises the desktop
    CUVID decoder, but Jetson does not provide ``libnvcuvid.so``.  Never leave
    decoder or null-output encoder selection to FFmpeg's registration order.
    The synthetic output timestamps are intentionally generated from decoded
    frame order; capture timestamps are validated independently in the frame
    and controller ledgers.
    """

    return [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-c:v",
        JETSON_H264_DECODER,
        "-stats_period",
        str(VALIDATION_PROGRESS_PERIOD_S),
        "-progress",
        "pipe:1",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vf",
        VALIDATION_OUTPUT_FILTER,
        "-c:v",
        "rawvideo",
        "-f",
        "null",
        "-",
    ]


def _full_decode(
    ffmpeg: str,
    path: Path,
    timeout_s: float,
    *,
    progress_callback: DecodeProgressCallback | None = None,
) -> dict[str, object]:
    """Decode every frame to a null sink and parse bounded progress output."""

    command = _decode_command(ffmpeg, path)
    try:
        worker = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        return {"count": None, "method": None, "error": f"ffmpeg start failed: {exc}"}
    assert worker.stdout is not None
    descriptor = worker.stdout.fileno()
    os.set_blocking(descriptor, False)
    deadline = time.monotonic() + timeout_s
    pending = bytearray()
    retained = bytearray()
    frame_count: int | None = None
    progress_end = False
    truncated = False

    def consume(line: bytes) -> None:
        nonlocal frame_count, progress_end, truncated
        if len(line) > MAX_VALIDATOR_LINE_BYTES:
            truncated = True
            return
        text = line.decode("utf-8", errors="replace").strip()
        if text.startswith("frame="):
            raw = text.partition("=")[2].strip()
            if raw.isascii() and raw.isdigit():
                frame_count = int(raw)
        elif text.startswith("progress="):
            progress_end = text == "progress=end"
            if frame_count is not None and progress_callback is not None:
                progress_callback(frame_count)
        elif "=" in text and text.partition("=")[0] in {
            "fps", "stream_0_0_q", "bitrate", "total_size", "out_time_us",
            "out_time_ms", "out_time", "dup_frames", "drop_frames", "speed",
        }:
            # Routine periodic progress is intentionally not retained; an
            # 88-hour validation must use constant memory.
            return
        elif text:
            available = MAX_VALIDATOR_OUTPUT_BYTES - len(retained)
            if available > 0:
                retained.extend(line[:available])
            if len(line) > available:
                truncated = True

    try:
        while True:
            if time.monotonic() >= deadline:
                _terminate_group(worker)
                return {
                    "count": None,
                    "method": "full_decode_ffmpeg",
                    "error": f"full decode timed out after {timeout_s:.1f}s",
                }
            readable, _, _ = select.select([descriptor], [], [], 0.2)
            chunk = b""
            if readable:
                try:
                    chunk = os.read(descriptor, 64 * 1024)
                except BlockingIOError:
                    pass
            if chunk:
                pending.extend(chunk)
                while b"\n" in pending:
                    line, _, remainder = pending.partition(b"\n")
                    pending = bytearray(remainder)
                    consume(line)
                if len(pending) > MAX_VALIDATOR_LINE_BYTES:
                    truncated = True
                    pending.clear()
            elif worker.poll() is not None:
                # Drain anything made readable between poll and process exit.
                try:
                    tail = os.read(descriptor, 64 * 1024)
                except BlockingIOError:
                    tail = b""
                if tail:
                    pending.extend(tail)
                    continue
                break
        if pending:
            consume(bytes(pending))
        return_code = worker.wait(timeout=1)
    except Exception:
        _terminate_group(worker)
        raise
    finally:
        worker.stdout.close()

    detail = retained.decode("utf-8", errors="replace").strip()
    if return_code != 0:
        return {
            "count": None,
            "method": "full_decode_ffmpeg",
            "error": detail or f"ffmpeg exited with code {return_code}",
        }
    if truncated:
        return {
            "count": None,
            "method": "full_decode_ffmpeg",
            "error": "ffmpeg validation output exceeded its bounded limit",
        }
    if not progress_end or frame_count is None:
        return {
            "count": None,
            "method": "full_decode_ffmpeg",
            "error": "ffmpeg completed without a final decoded-frame count",
        }
    return {"count": frame_count, "method": "full_decode_ffmpeg", "error": None}


def _gstreamer_command(
    path: Path, output_fd: int, *, parse_only: bool = False
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "squeakview.apps.inference.gstreamer_video_probe",
        "--output-fd",
        str(output_fd),
    ]
    if parse_only:
        command.append("--parse-only")
    command.append(str(path))
    return command


def _full_decode_gstreamer(
    path: Path,
    timeout_s: float,
    *,
    progress_callback: DecodeProgressCallback | None = None,
) -> dict[str, object]:
    """Run the NVMM decoder in an isolated process and require a clean EOS."""

    return _run_gstreamer_validator(
        path,
        timeout_s,
        method=GSTREAMER_METHOD,
        parse_only=False,
        progress_callback=progress_callback,
    )


def _full_parse_gstreamer(
    path: Path,
    timeout_s: float,
    *,
    progress_callback: DecodeProgressCallback | None = None,
) -> dict[str, object]:
    """Parse every H.264 access unit in an isolated process through clean EOS."""

    return _run_gstreamer_validator(
        path,
        timeout_s,
        method=STRUCTURAL_METHOD,
        parse_only=True,
        progress_callback=progress_callback,
    )


def _run_gstreamer_validator(
    path: Path,
    timeout_s: float,
    *,
    method: str,
    parse_only: bool,
    progress_callback: DecodeProgressCallback | None,
) -> dict[str, object]:
    """Run one bounded GStreamer validator child and consume its small protocol."""

    read_fd, write_fd = os.pipe()
    try:
        worker = subprocess.Popen(
            _gstreamer_command(path, write_fd, parse_only=parse_only),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=(write_fd,),
            start_new_session=True,
        )
    except OSError as exc:
        os.close(read_fd)
        os.close(write_fd)
        return {
            "count": None,
            "method": method,
            "error": f"GStreamer validator start failed: {exc}",
        }
    os.close(write_fd)
    descriptor = read_fd
    os.set_blocking(descriptor, False)
    deadline = time.monotonic() + timeout_s
    pending = bytearray()
    retained = bytearray()
    result: dict[str, object] | None = None
    truncated = False

    def consume(line: bytes) -> None:
        nonlocal result, truncated
        if len(line) > MAX_VALIDATOR_LINE_BYTES:
            truncated = True
            return
        try:
            payload = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            available = MAX_VALIDATOR_OUTPUT_BYTES - len(retained)
            if available > 0:
                retained.extend(line[:available] + b"\n")
            if len(line) > available:
                truncated = True
            return
        if not isinstance(payload, dict):
            truncated = True
            return
        if payload.get("type") == "progress":
            frames = payload.get("frames")
            if type(frames) is int and frames >= 0 and progress_callback is not None:
                progress_callback(frames)
        elif payload.get("type") == "result":
            result = payload

    try:
        while True:
            if time.monotonic() >= deadline:
                _terminate_group(worker)
                return {
                    "count": None,
                    "method": method,
                    "error": f"validation timed out after {timeout_s:.1f}s",
                }
            readable, _, _ = select.select([descriptor], [], [], 0.2)
            chunk = b""
            if readable:
                try:
                    chunk = os.read(descriptor, 64 * 1024)
                except BlockingIOError:
                    pass
            if chunk:
                pending.extend(chunk)
                while b"\n" in pending:
                    line, _, remainder = pending.partition(b"\n")
                    pending = bytearray(remainder)
                    consume(line)
                if len(pending) > MAX_VALIDATOR_LINE_BYTES:
                    truncated = True
                    pending.clear()
            elif worker.poll() is not None:
                try:
                    tail = os.read(descriptor, 64 * 1024)
                except BlockingIOError:
                    tail = b""
                if tail:
                    pending.extend(tail)
                    continue
                break
        if pending:
            consume(bytes(pending))
        return_code = worker.wait(timeout=1)
    except Exception:
        _terminate_group(worker)
        raise
    finally:
        os.close(descriptor)

    detail = retained.decode("utf-8", errors="replace").strip()
    if truncated:
        return {
            "count": None,
            "method": method,
            "error": "GStreamer validation output exceeded its bounded limit",
        }
    if result is None:
        return {
            "count": None,
            "method": method,
            "error": detail or f"GStreamer validator exited with code {return_code}",
        }
    count = result.get("count")
    error = result.get("error")
    if return_code != 0 or type(count) is not int or count < 0 or error is not None:
        return {
            "count": None,
            "method": method,
            "error": str(error or detail or f"GStreamer validator exited with code {return_code}"),
        }
    if progress_callback is not None:
        progress_callback(count)
    return {"count": count, "method": method, "error": None}


def _ab_verify_enabled() -> bool:
    return os.environ.get("SQUEAKVIEW_VIDEO_VALIDATION_AB_VERIFY", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _full_decode_enabled() -> bool:
    return os.environ.get(
        "SQUEAKVIEW_VIDEO_VALIDATION_FULL_DECODE", ""
    ).strip().lower() in {"1", "true", "yes", "on"}


def _structural_probe(
    path: Path,
    timeout_s: float,
    *,
    progress_callback: DecodeProgressCallback | None,
) -> dict[str, object]:
    """Reconcile MP4 sample tables with a complete H.264 parse to clean EOS."""

    try:
        table = read_video_sample_table(path)
    except (OSError, ValueError) as exc:
        return {
            "count": None,
            "method": STRUCTURAL_METHOD,
            "error": f"MP4 sample-table validation failed: {exc}",
        }
    parsed = _full_parse_gstreamer(
        path,
        timeout_s,
        progress_callback=progress_callback,
    )
    if parsed.get("error") is not None:
        return parsed
    if parsed.get("count") != table.sample_count:
        return {
            "count": None,
            "method": STRUCTURAL_METHOD,
            "error": (
                "MP4 sample count and parsed H.264 access-unit count disagree: "
                f"samples={table.sample_count}, parsed={parsed.get('count')}"
            ),
        }
    return {
        "count": table.sample_count,
        "method": STRUCTURAL_METHOD,
        "error": None,
        "mp4_timing_samples": table.timing_sample_count,
        "mp4_chunk_samples": table.chunk_sample_count,
        "mp4_chunks": table.chunk_count,
    }


def _decode_with_fallback(
    path: Path,
    timeout_s: float,
    *,
    progress_callback: DecodeProgressCallback | None,
) -> dict[str, object]:
    """Fully decode with GStreamer, retaining FFmpeg fallback and A/B mode."""

    validation_started = time.monotonic()
    primary = _full_decode_gstreamer(
        path,
        timeout_s,
        progress_callback=progress_callback,
    )
    ffmpeg = shutil.which("ffmpeg")
    if primary.get("error") is None and not _ab_verify_enabled():
        return primary
    if ffmpeg is None:
        if primary.get("error") is None:
            return {
                "count": None,
                "method": f"{GSTREAMER_METHOD}_ab_verification",
                "error": "FFmpeg is unavailable for requested A/B verification",
            }
        return primary
    remaining_s = timeout_s - (time.monotonic() - validation_started)
    if remaining_s < 1.0:
        if primary.get("error") is None:
            return {
                "count": None,
                "method": f"{GSTREAMER_METHOD}_ab_verification",
                "error": "no validation time remained for FFmpeg A/B verification",
            }
        return primary
    fallback = _full_decode(
        ffmpeg,
        path,
        remaining_s,
        progress_callback=progress_callback,
    )
    if primary.get("error") is None:
        if fallback.get("error") is not None:
            return {
                "count": None,
                "method": f"{GSTREAMER_METHOD}_ab_verification",
                "error": f"FFmpeg A/B decode failed: {fallback['error']}",
            }
        if fallback.get("count") != primary.get("count"):
            return {
                "count": None,
                "method": f"{GSTREAMER_METHOD}_ab_verification",
                "error": (
                    "decoder A/B frame-count mismatch: "
                    f"GStreamer={primary.get('count')}, FFmpeg={fallback.get('count')}"
                ),
            }
        return {
            "count": primary["count"],
            "method": f"{GSTREAMER_METHOD}_ffmpeg_verified",
            "error": None,
        }
    if fallback.get("error") is None:
        return {
            "count": fallback["count"],
            "method": "full_decode_ffmpeg_fallback",
            "error": None,
            "warning": f"GStreamer primary decode failed: {primary.get('error')}",
        }
    return {
        "count": None,
        "method": "full_decode_gstreamer_then_ffmpeg",
        "error": (
            f"GStreamer decode failed: {primary.get('error')}; "
            f"FFmpeg fallback failed: {fallback.get('error')}"
        ),
    }


def probe_video_frames(
    path: Path,
    *,
    progress_callback: DecodeProgressCallback | None = None,
    expected_frames: int | None = None,
    force_full_decode: bool = False,
) -> dict[str, object]:
    """Validate a recording structurally, escalating anomalies to full decode."""

    path = Path(path)
    if not path.is_file():
        reason = "video file is missing"
        return {"count": None, "method": None, "error": reason}
    timeout_s = _timeout_seconds()
    validation_started = time.monotonic()
    structural = _structural_probe(
        path,
        timeout_s,
        progress_callback=progress_callback,
    )
    structural_mismatch = (
        expected_frames is not None
        and structural.get("count") != expected_frames
    )
    full_required = force_full_decode or _full_decode_enabled() or _ab_verify_enabled()
    if structural.get("error") is None and not structural_mismatch and not full_required:
        return structural

    anomaly = structural.get("error")
    if anomaly is None and structural_mismatch:
        anomaly = (
            "structural frame-count mismatch: "
            f"recording={structural.get('count')}, expected={expected_frames}"
        )
    remaining_s = timeout_s - (time.monotonic() - validation_started)
    if remaining_s < 1.0:
        return {
            "count": None,
            "method": structural.get("method"),
            "error": f"{anomaly or 'full decode required'}; no validation time remained",
        }
    decoded = _decode_with_fallback(
        path,
        remaining_s,
        progress_callback=progress_callback,
    )
    if anomaly is not None:
        if decoded.get("error") is None:
            existing_warning = decoded.get("warning")
            decoded["warning"] = "; ".join(
                part
                for part in (
                    f"Fast structural validation anomaly: {anomaly}",
                    str(existing_warning) if existing_warning else "",
                )
                if part
            )
        else:
            decoded["error"] = f"{anomaly}; full-decode escalation failed: {decoded['error']}"
    return decoded


def video_frame_count(path: Path) -> int | None:
    count = probe_video_frames(path).get("count")
    return int(count) if count is not None else None


def decode_self_test(path: Path, *, expected_frames: int = 1) -> tuple[bool, str]:
    """Exercise the same real decoder used by post-run validation."""

    result = probe_video_frames(
        path,
        expected_frames=expected_frames,
        force_full_decode=True,
    )
    count = result.get("count")
    if count == expected_frames and result.get("error") is None:
        return True, f"decoded {count} frame(s) with {result.get('method')}"
    return False, str(result.get("error") or f"decoded {count!r}; expected {expected_frames}")


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Exercise SqueakView's authoritative Jetson video decoder"
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--expect-frames", type=int, default=1)
    args = parser.parse_args(argv)
    if args.expect_frames < 1:
        parser.error("--expect-frames must be positive")
    passed, detail = decode_self_test(args.video, expected_frames=args.expect_frames)
    print(detail, flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
