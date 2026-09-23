"""Bounded recording validation and automatic controller alignment."""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Mapping

from squeakview.common import run_context
from squeakview.common import qualification_barrier
from squeakview.common.recording_evidence import (
    MAX_RECORDING_CAMERAS,
    capture_recording_evidence,
)

from .frame_audit import FrameCsvOperator
from . import (
    acquisition_validation,
    inference_reconciliation,
    preview_attribution,
    recording_validation,
)
from .capture_reconciliation import (
    StreamStats,
    iter_admission_pts as _iter_admission_pts,
    iter_capture_payloads as _iter_capture_payloads,
    payload_sort_key as _payload_sort_key,
    recorded_payloads as _recorded_payloads,
)
from .inference_reconciliation import (
    index_inference_frames as _index_inference_frames,
    open_index as _open_index,
)
from .video_probe import probe_video_frames


PROGRESS_FILENAME = "post_run_progress.json"
MAX_LEDGER_TAIL_BYTES = 65_536


def _timestamp() -> str:
    return time.strftime("%H:%M:%S")


def _update_progress(run_dir: Path, **updates) -> None:
    path = Path(run_dir) / PROGRESS_FILENAME
    payload = run_context.read_json(path)
    payload.update(updates)
    run_context.atomic_write_json(path, payload)


def _video_progress_payload(
    decoded_frames: int,
    expected_frames: int,
    started_monotonic: float,
    *,
    now_monotonic: float | None = None,
) -> dict[str, object]:
    """Build bounded, operator-facing full-decode progress and ETA state."""

    now = time.monotonic() if now_monotonic is None else now_monotonic
    elapsed_s = max(0.0, now - started_monotonic)
    decoded = max(0, int(decoded_frames))
    expected = max(0, int(expected_frames))
    rate_fps = decoded / elapsed_s if elapsed_s > 0.0 else 0.0
    remaining = max(0, expected - decoded)
    eta_s = remaining / rate_fps if rate_fps > 0.0 else None
    percent = min(100.0, decoded * 100.0 / expected) if expected else None
    return {
        "video_frames_decoded": decoded,
        "video_frames_expected": expected,
        "video_validation_elapsed_s": round(elapsed_s, 1),
        "video_validation_rate_fps": round(rate_fps, 1),
        "video_validation_eta_s": None if eta_s is None else round(eta_s, 1),
        "video_validation_percent": None if percent is None else round(percent, 1),
    }


@dataclass(slots=True)
class FinalizationResult:
    source_counts: dict[int, int]
    recorded_counts: dict[int, int]
    recorded_total: int
    validation_passed: bool
    video_validation: dict[str, object]
    warnings: list[str] = field(default_factory=list)


def _write_payload(operator: FrameCsvOperator, payload: dict) -> None:
    stream_id = int(payload.get("camera_index") or 0)
    sequence = int(payload.get("source_sequence_index") or 0)
    pts_ns = int(payload.get("gst_pts_ns") or 0)
    user_meta = SimpleNamespace(get_user_data_json=lambda payload=payload: payload)
    frame_meta = SimpleNamespace(
        frame_number=sequence,
        source_id=stream_id,
        pad_index=stream_id,
        buffer_pts=pts_ns,
        user_meta_items=lambda _meta_type, user_meta=user_meta: iter([user_meta]),
    )
    operator.handle_metadata(SimpleNamespace(frame_items=[frame_meta]))


def _summarize_inference_admission(
    connection,
    recorded_counts: dict[int, int],
) -> dict:
    return inference_reconciliation.summarize_inference_admission(
        connection, recorded_counts
    ).to_dict()


def _durable_replace(source: Path, destination: Path) -> None:
    """Atomically replace one artifact and persist its directory entry."""

    with source.open("rb") as handle:
        os.fsync(handle.fileno())
    source.replace(destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _promote(temp_dir: Path, run_dir: Path, paths: dict[str, str]) -> None:
    """Durably promote reproducible artifacts with restart-safe semantics.

    Status remains finalizing until the complete set is promoted and validated.
    If interrupted between replacements, rerunning finalization reconstructs
    and replaces every artifact.
    """

    for source_name, destination_name in paths.items():
        source = temp_dir / source_name
        if source.exists():
            destination = run_dir / destination_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            _durable_replace(source, destination)


def _validate_recordings(
    run_dir: Path,
    camera_count: int,
    source_counts: dict[int, int],
    recorded_counts: dict[int, int],
    *,
    evidence: dict[str, object] | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[dict, bool]:
    def probe(path: Path) -> dict[str, object]:
        return probe_video_frames(path, progress_callback=progress_callback)

    result = recording_validation.validate_recordings(
        run_dir,
        camera_count,
        source_counts,
        recorded_counts,
        probe=probe,
        evidence=evidence,
    )
    return result.to_dict(), result.passed


def _validate_acquisition_integrity(
    run_dir: Path, camera_count: int = 1
) -> tuple[dict, bool]:
    result = acquisition_validation.validate_acquisition_integrity(
        run_dir, camera_count
    )
    return result.to_dict(), result.passed


def _last_complete_line(path: Path) -> str | None:
    """Read one bounded final ledger record without scanning the whole file."""

    try:
        with Path(path).open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            if end <= 0:
                return None
            read_size = min(end, MAX_LEDGER_TAIL_BYTES)
            handle.seek(end - read_size)
            data = handle.read(read_size)
    except OSError:
        return None
    lines = data.splitlines()
    if data and not data.endswith((b"\n", b"\r")):
        lines = lines[:-1]
    for line in reversed(lines):
        text = line.decode("utf-8", errors="strict").strip()
        if text:
            return text
    return None


def _durable_ledger_counts(run_dir: Path, camera_count: int) -> dict[int, tuple[int, int]]:
    """Return capture/admission totals from their monotonic final indices."""

    counts: dict[int, tuple[int, int]] = {}
    for stream_id in range(camera_count):
        capture_path = run_dir / f"capture_cam{stream_id}.jsonl"
        admission_path = run_dir / (
            "record_admission.csv"
            if stream_id == 0
            else f"record_admission_cam{stream_id}.csv"
        )
        capture_line = _last_complete_line(capture_path)
        admission_line = _last_complete_line(admission_path)
        try:
            capture_payload = json.loads(capture_line or "")
            capture_index = capture_payload["source_sequence_index"]
            if type(capture_index) is not int or capture_index < 0:
                raise ValueError("invalid source_sequence_index")
            capture_count = capture_index + 1
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"cannot read final capture metadata count for camera {stream_id}: {exc}"
            ) from exc
        try:
            fields = next(csv.reader([admission_line or ""]))
            record_index_text = fields[1]
            if not record_index_text.isascii() or not record_index_text.isdecimal():
                raise ValueError("invalid record_frame_index")
            admission_count = int(record_index_text) + 1
        except (IndexError, StopIteration, ValueError) as exc:
            raise RuntimeError(
                f"cannot read final recording-admission count for camera {stream_id}: {exc}"
            ) from exc
        counts[stream_id] = (capture_count, admission_count)
    return counts


def _durable_frame_manifest_counts(run_dir: Path, camera_count: int) -> dict[int, int]:
    """Read source-owned canonical manifest totals from final row indices."""

    counts: dict[int, int] = {}
    for stream_id in range(camera_count):
        path = run_dir / (
            "frames.csv" if stream_id == 0 else f"frames_cam{stream_id}.csv"
        )
        line = _last_complete_line(path)
        try:
            fields = next(csv.reader([line or ""], strict=True))
            if len(fields) != len(FrameCsvOperator.HEADERS):
                raise ValueError(
                    f"expected {len(FrameCsvOperator.HEADERS)} fields, got {len(fields)}"
                )
            if fields[0] != str(stream_id):
                raise ValueError(f"unexpected stream_id {fields[0]!r}")
            sequence_text = fields[3]
            if not sequence_text.isascii() or not sequence_text.isdecimal():
                raise ValueError("invalid source_sequence_index")
            counts[stream_id] = int(sequence_text) + 1
        except (csv.Error, IndexError, StopIteration, ValueError) as exc:
            raise RuntimeError(
                "cannot read final canonical frame manifest count for camera "
                f"{stream_id}: {exc}"
            ) from exc
    return counts


def _bounded_acquisition_integrity(run_dir: Path, camera_count: int) -> tuple[dict, bool]:
    """Validate source-owned cumulative fault counters from final ledger rows."""

    cameras: list[dict[str, object]] = []
    for stream_id in range(camera_count):
        line = _last_complete_line(run_dir / f"capture_cam{stream_id}.jsonl")
        try:
            payload = json.loads(line or "")
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"cannot read final acquisition counters for camera {stream_id}: {exc}"
            ) from exc
        counters: dict[str, int] = {}
        for name in (
            "total_incomplete",
            "total_frame_gap_events",
            "total_crc_failures",
        ):
            value = payload.get(name)
            if type(value) is not int or value < 0:
                raise RuntimeError(
                    f"final acquisition counter {name} is unavailable or invalid "
                    f"for camera {stream_id}; rebuild the FLIR source plugin"
                )
            counters[name] = value
        cameras.append(
            {
                "stream_id": stream_id,
                **counters,
                "passed": not any(counters.values()),
            }
        )
    passed = bool(cameras) and all(camera["passed"] is True for camera in cameras)
    return (
        {
            "schema_version": "2.0",
            "validation_tier": "source_cumulative_counters",
            "policy": "no_incomplete_frames_frame_gaps_or_payload_crc_failures",
            "cameras": cameras,
            "passed": passed,
        },
        passed,
    )


def _video_metadata_identity(path: Path) -> tuple[int, int, int, int, int]:
    metadata = Path(path).stat()
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def fast_finalize_run(run_dir: Path, *, camera_count: int) -> FinalizationResult:
    """Validate the live frame manifest against admission and MP4 samples.

    The source writes ``frames.csv`` during acquisition, matching the original
    SqueakView lifecycle. Shutdown performs bounded tail reads and a direct MP4
    sample-table read; it never reconstructs metadata or decodes video.
    """

    run_dir = Path(run_dir).resolve()
    if type(camera_count) is not int or not 1 <= camera_count <= MAX_RECORDING_CAMERAS:
        raise ValueError(
            f"camera_count must be an integer from 1 through {MAX_RECORDING_CAMERAS}"
        )
    run_context.write_status(
        run_dir,
        "finalizing",
        stage="recording_count_validation",
        error=None,
    )
    run_context.atomic_write_json(
        run_dir / PROGRESS_FILENAME,
        {"stage": "recording_count_validation", "frames_processed": 0},
    )
    try:
        counts = _durable_ledger_counts(run_dir, camera_count)
        manifest_counts = _durable_frame_manifest_counts(run_dir, camera_count)
        prior_status = run_context.read_json_required(
            run_dir / run_context.RUN_STATUS_FILENAME
        )
        expected_ttl = prior_status.get("expected_ttl_count")
        if expected_ttl is not None and (
            type(expected_ttl) is not int or expected_ttl < 0
        ):
            raise RuntimeError("persisted expected_ttl_count is invalid")

        cameras: list[dict[str, object]] = []
        source_counts: dict[int, int] = {}
        recorded_counts: dict[int, int] = {}
        for stream_id in range(camera_count):
            source_count, admitted_count = counts[stream_id]
            manifest_count = manifest_counts[stream_id]
            source_counts[stream_id] = source_count
            recorded_counts[stream_id] = admitted_count
            video_path = run_dir / (
                "raw.mp4" if stream_id == 0 else f"raw_cam{stream_id}.mp4"
            )
            before = _video_metadata_identity(video_path)
            probe = probe_video_frames(video_path, expected_frames=admitted_count)
            after = _video_metadata_identity(video_path)
            raw_video_count = probe.get("count")
            video_count = (
                raw_video_count
                if type(raw_video_count) is int and raw_video_count >= 0
                else None
            )
            unchanged = before == after
            source_matches = source_count == admitted_count
            manifest_matches = manifest_count == admitted_count
            controller_matches = expected_ttl is None or source_count == expected_ttl
            count_matches = video_count == admitted_count if video_count is not None else None
            error = probe.get("error")
            if not unchanged:
                error = "recording changed during sample-count validation"
            elif not controller_matches:
                error = (
                    f"camera metadata count {source_count} does not match "
                    f"controller count {expected_ttl}"
                )
            cameras.append(
                {
                    "stream_id": stream_id,
                    "video": video_path.name,
                    "exists": before[2] > 0,
                    "source_frames": source_count,
                    "record_admitted_frames": admitted_count,
                    "frame_manifest_frames": manifest_count,
                    "source_count_matches": source_matches,
                    "frame_manifest_count_matches": manifest_matches,
                    "controller_count_matches": controller_matches,
                    "nonzero_frame_count": bool(
                        source_count > 0 and admitted_count > 0 and (video_count or 0) > 0
                    ),
                    "video_frames": video_count,
                    "frame_count_matches": count_matches,
                    "frame_count_method": probe.get("method"),
                    "frame_count_error": error,
                    "frame_count_warning": probe.get("warning"),
                    "video_unchanged_during_validation": unchanged,
                    "video_size_bytes": before[2],
                }
            )

        recording_passed = bool(cameras) and all(
            camera["exists"] is True
            and camera["source_count_matches"] is True
            and camera["frame_manifest_count_matches"] is True
            and camera["controller_count_matches"] is True
            and camera["nonzero_frame_count"] is True
            and camera["frame_count_matches"] is True
            and camera["frame_count_error"] is None
            and camera["video_unchanged_during_validation"] is True
            for camera in cameras
        )
        recording_report = {
            "schema_version": "3.0",
            "validation_tier": "shutdown_fast_count",
            "policy": (
                "mp4_sample_count_equals_live_frame_manifest_and_durable_"
                "capture_admission_counts"
            ),
            "cameras": cameras,
            "passed": recording_passed,
        }
        integrity_report, integrity_passed = _bounded_acquisition_integrity(
            run_dir, camera_count
        )
        passed = recording_passed and integrity_passed
        reconciliation = {
            "schema_version": "2.0",
            "validation_tier": "durable_final_indices",
            "source_frames": source_counts,
            "record_admitted_frames": recorded_counts,
            "frame_manifest_frames": manifest_counts,
            "source_not_recorded_frames": {
                stream_id: source_counts[stream_id] - recorded_counts[stream_id]
                for stream_id in source_counts
            },
            "policy": (
                "final source, non-leaky recording admission, and live canonical "
                "frame-manifest indices must agree"
            ),
        }
        total = sum(recorded_counts.values())
        run_context.write_status(
            run_dir,
            "post_run_complete",
            stage="recording_count_validation_complete",
            error=None,
            post_run_frames=total,
            recording_validation_passed=recording_passed,
            recording_validation=recording_report,
            acquisition_integrity=integrity_report,
            overall_validation_passed=passed,
            capture_reconciliation=reconciliation,
            analysis_complete=False,
            alignment_validated=False,
        )
        _update_progress(
            run_dir,
            stage="complete" if passed else "failed",
            frames_processed=total,
            recording_validation_passed=recording_passed,
            overall_validation_passed=passed,
        )
        level = "PASS" if passed else "ERROR"
        print(
            f"[{_timestamp()}] [POST-RUN] {level}: MP4/metadata count validation "
            f"completed for {total} frame(s)",
            flush=True,
        )
        primary = cameras[0]
        return FinalizationResult(
            source_counts,
            recorded_counts,
            total,
            passed,
            {
                "count": primary.get("video_frames"),
                "method": primary.get("frame_count_method"),
                "error": primary.get("frame_count_error"),
            },
        )
    except Exception as exc:
        run_context.write_status(run_dir, "finalization_failed", error=str(exc))
        _update_progress(
            run_dir,
            stage="failed",
            overall_validation_passed=False,
            error=str(exc),
        )
        raise


def finalize_run(
    run_dir: Path,
    *,
    camera_count: int,
    enable_infer: bool,
) -> FinalizationResult:
    run_dir = Path(run_dir).resolve()
    if type(camera_count) is not int or not 1 <= camera_count <= MAX_RECORDING_CAMERAS:
        raise ValueError(
            f"camera_count must be an integer from 1 through {MAX_RECORDING_CAMERAS}"
        )
    # A finalizer may be rerun after a recoverable validation failure.  Clear
    # the prior active error as soon as recovery begins; write_status merges
    # with the durable document so omitting this would leave a stale failure
    # attached to a healthy retry.
    run_context.write_status(
        run_dir,
        "finalizing",
        stage="capture_reconciliation",
        error=None,
    )
    qualification_barrier.wait_at_barrier(
        run_dir, "finalizer:capture_reconciliation"
    )
    run_context.atomic_write_json(
        run_dir / PROGRESS_FILENAME,
        {"stage": "capture_reconciliation", "frames_processed": 0},
    )
    initial_recording_evidence = capture_recording_evidence(
        run_dir, camera_count
    )
    temp_dir = Path(tempfile.mkdtemp(prefix=".post_run.", dir=run_dir))
    stats = {stream_id: StreamStats() for stream_id in range(camera_count)}
    index = None
    operator = None
    try:
        index = _open_index(temp_dir / "post_run.sqlite")
        if enable_infer:
            _index_inference_frames(index, run_dir / "inference" / "frames.csv")
        operator = FrameCsvOperator(
            temp_dir / "frames.csv",
            meta_type=0,
            audit_dir=temp_dir / "diagnostics",
            max_cameras=camera_count,
        )
    except Exception as exc:
        if index is not None:
            index.close()
        shutil.rmtree(temp_dir, ignore_errors=True)
        run_context.write_status(run_dir, "finalization_failed", error=str(exc))
        run_context.atomic_write_json(
            run_dir / PROGRESS_FILENAME,
            {"stage": "failed", "frames_processed": 0, "error": str(exc)},
        )
        raise
    assert index is not None
    assert operator is not None
    processed = 0
    last_progress = time.monotonic()
    recorded_insert_batch: list[tuple[int, int, int, str]] = []
    try:
        iterators = [
            _recorded_payloads(run_dir, stream_id, stats[stream_id])
            for stream_id in range(camera_count)
        ]
        for payload in heapq.merge(*iterators, key=_payload_sort_key):
            if enable_infer:
                inferred = index.execute(
                    "SELECT 1 FROM inferred WHERE stream_id=? AND source_sequence_index=?",
                    (
                        int(payload.get("camera_index") or 0),
                        int(payload.get("source_sequence_index") or 0),
                    ),
                ).fetchone()
                payload["inference_admitted"] = int(inferred is not None)
            _write_payload(operator, payload)
            stream_id = int(payload.get("camera_index") or 0)
            sequence = int(payload.get("source_sequence_index") or 0)
            camera_frame_id = payload.get("camera_frame_id")
            recorded_insert_batch.append(
                (
                    processed,
                    stream_id,
                    sequence,
                    "" if camera_frame_id is None else str(camera_frame_id),
                )
            )
            processed += 1
            if len(recorded_insert_batch) >= 10_000:
                with index:
                    index.executemany(
                        "INSERT INTO recorded VALUES (?, ?, ?, ?)",
                        recorded_insert_batch,
                    )
                recorded_insert_batch.clear()
            now = time.monotonic()
            if processed % 100_000 == 0 or now - last_progress >= 5.0:
                run_context.atomic_write_json(
                    run_dir / PROGRESS_FILENAME,
                    {
                        "stage": "capture_reconciliation",
                        "frames_processed": processed,
                    },
                )
                last_progress = now
        if recorded_insert_batch:
            with index:
                index.executemany(
                    "INSERT INTO recorded VALUES (?, ?, ?, ?)",
                    recorded_insert_batch,
                )
        operator.close()

        unmatched = sum(item.unmatched_admissions for item in stats.values())
        if unmatched:
            raise RuntimeError(
                f"{unmatched} recording admissions have no matching source metadata"
            )
        source_counts = {
            stream_id: item.source_frames for stream_id, item in stats.items()
        }
        recorded_counts = {
            stream_id: item.recorded_frames for stream_id, item in stats.items()
        }
        reconciliation = {
            "schema_version": "1.0",
            "source_frames": source_counts,
            "record_admitted_frames": recorded_counts,
            "source_not_recorded_frames": {
                stream_id: source_counts[stream_id] - recorded_counts[stream_id]
                for stream_id in source_counts
            },
            "policy": (
                "frames.csv contains only buffers admitted to the non-leaky "
                "recording branch"
            ),
        }
        inference_summary = None
        if enable_infer:
            run_context.write_status(run_dir, "finalizing", stage="inference_admission")
            qualification_barrier.wait_at_barrier(
                run_dir, "finalizer:inference_admission"
            )
            inference_summary = _summarize_inference_admission(index, recorded_counts)
        _promote(
            temp_dir,
            run_dir,
            {
                "frames.csv": "frames.csv",
                "diagnostics/camera_runtime.json": "diagnostics/camera_runtime.json",
                "diagnostics/camera.csv": "diagnostics/camera.csv",
                "diagnostics/errors.csv": "diagnostics/errors.csv",
            },
        )
        run_context.write_status(run_dir, "finalizing", stage="recording_validation")
        qualification_barrier.wait_at_barrier(
            run_dir, "finalizer:recording_validation"
        )
        video_validation_started = time.monotonic()
        video_expected_total = sum(recorded_counts.values())
        video_decoded_by_stream = {
            stream_id: 0 for stream_id in range(camera_count)
        }
        active_video_stream = 0
        last_video_progress_write = video_validation_started

        _update_progress(
            run_dir,
            stage="recording_validation",
            **_video_progress_payload(
                0, video_expected_total, video_validation_started
            ),
        )

        def report_video_progress(decoded_frames: int) -> None:
            nonlocal last_video_progress_write
            video_decoded_by_stream[active_video_stream] = max(
                video_decoded_by_stream[active_video_stream],
                max(0, int(decoded_frames)),
            )
            decoded_total = sum(video_decoded_by_stream.values())
            now = time.monotonic()
            # Progress is transient presentation state, not scientific
            # evidence. Bound crash-resistant writes during multi-day runs
            # while still updating the operator often enough to show life.
            if (
                decoded_total != video_expected_total
                and now - last_video_progress_write < 5.0
            ):
                return
            _update_progress(
                run_dir,
                stage="recording_validation",
                video_stream_id=active_video_stream,
                **_video_progress_payload(
                    decoded_total,
                    video_expected_total,
                    video_validation_started,
                    now_monotonic=now,
                ),
            )
            last_video_progress_write = now

        def probe_with_stream_progress(path: Path) -> dict[str, object]:
            nonlocal active_video_stream
            active_video_stream = (
                0
                if path.name == "raw.mp4"
                else int(path.stem.removeprefix("raw_cam"))
            )
            return probe_video_frames(
                path,
                progress_callback=report_video_progress,
                expected_frames=recorded_counts[active_video_stream],
            )

        result = recording_validation.validate_recordings(
            run_dir,
            camera_count,
            source_counts,
            recorded_counts,
            probe=probe_with_stream_progress,
            evidence=initial_recording_evidence,
        )
        report, recording_passed = result.to_dict(), result.passed
        integrity_report, integrity_passed = _validate_acquisition_integrity(
            run_dir, camera_count
        )
        # The finalizer does not make recording validity depend on optional
        # preview evidence. A missing/corrupt manifest is rejected later by
        # qualification, while recording reconciliation remains recoverable.
        manifest = run_context.read_json(
            run_dir / run_context.RUN_MANIFEST_FILENAME
        )
        inference_manifest = (
            manifest.get("inference")
            if isinstance(manifest.get("inference"), dict)
            else {}
        )
        preview_required = inference_manifest.get("preview_enabled") is True
        preview_report = preview_attribution.reconcile_preview(
            run_dir, camera_count, required=preview_required
        )
        inference_passed = bool(
            inference_summary is None or inference_summary.get("passed") is True
        )
        passed = recording_passed and integrity_passed and inference_passed
        _update_progress(
            run_dir,
            stage=("recording_validation_complete" if passed else "recording_validation_failed"),
            frames_processed=processed,
            recording_validation_passed=recording_passed,
            overall_validation_passed=False if not passed else None,
        )
        run_context.write_status(
            run_dir,
            "post_run_complete",
            error=None,
            post_run_frames=processed,
            recording_validation_passed=recording_passed,
            recording_validation=report,
            acquisition_integrity=integrity_report,
            overall_validation_passed=passed,
            capture_reconciliation=reconciliation,
            inference_admission=inference_summary,
            preview_attribution=preview_report,
        )
        level = "PASS" if passed else "ERROR"
        print(
            f"[{_timestamp()}] [POST-RUN] {level}: finalized {processed} frames; "
            f"validation={report['cameras']}",
            flush=True,
        )
        primary_camera = report["cameras"][0]
        video_validation = {
            "count": primary_camera.get("video_frames"),
            "method": primary_camera.get("frame_count_method"),
            "error": primary_camera.get("frame_count_error"),
        }
        return FinalizationResult(
            source_counts,
            recorded_counts,
            processed,
            passed,
            video_validation,
        )
    except Exception as exc:
        run_context.write_status(run_dir, "finalization_failed", error=str(exc))
        run_context.atomic_write_json(
            run_dir / PROGRESS_FILENAME,
            {"stage": "failed", "frames_processed": processed, "error": str(exc)},
        )
        raise
    finally:
        try:
            operator.close()
        finally:
            try:
                index.close()
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)


def align_run(
    run_dir: Path,
    *,
    video_validation: Mapping[str, object] | None = None,
) -> dict:
    """Build the compact alignment/validation summary without expanded CSV caches."""

    from scripts.align_run_outputs_streaming import build_alignment

    run_dir = Path(run_dir).resolve()
    temp_dir = Path(tempfile.mkdtemp(prefix=".analysis.", dir=run_dir))
    try:
        run_context.write_status(run_dir, "analyzing", stage="streaming_alignment")
        qualification_barrier.wait_at_barrier(
            run_dir, "finalizer:streaming_alignment"
        )
        _update_progress(run_dir, stage="alignment", overall_validation_passed=None)
        summary = build_alignment(
            run_dir,
            temp_dir,
            video_validation=video_validation,
            include_objects=False,
        )
        run_context.atomic_write_json(run_dir / "alignment_summary.json", summary)
        counts = summary.get("counts", {})
        validation = summary.get("validation", {})
        frame_alignment = summary.get("frame_alignment", {})
        failures: list[str] = []
        if summary.get("schema_version") != "2.0":
            failures.append("alignment evidence schema")
        if frame_alignment.get("validated") is not True:
            failures.append("frame/controller alignment")
        if (
            frame_alignment.get("epoch_markers_complete") is not True
            or summary.get("start_marker_seen") is not True
        ):
            failures.append("controller trigger epoch")
        if (
            frame_alignment.get("controller_high_events_unmatched") != 0
            or frame_alignment.get("shutdown_tail_high_events_unmatched") != 0
            or frame_alignment.get("controller_high_events_in_epoch")
            != counts.get("recorded_frames")
        ):
            failures.append("controller trigger/frame bijection")
        if validation.get("video_frame_count_matches_frames_csv") is not True:
            failures.append("video/frame count")
        for key in (
            "objects_missing_frame_count",
            "object_mapping_failed_rows",
            "object_ts_mismatch_count",
            "object_pts_mismatch_count",
        ):
            if int(validation.get(key) or 0) != 0:
                failures.append(key)
        if int(counts.get("frame_gaps_detected") or 0) != 0:
            failures.append("frame gaps")
        if failures:
            raise RuntimeError(f"alignment validation failed: {', '.join(failures)}")
        _update_progress(
            run_dir,
            stage="complete",
            alignment_validation_passed=True,
            overall_validation_passed=True,
        )
        run_context.write_status(
            run_dir,
            "analysis_complete",
            stage="complete",
            error=None,
            analysis_complete=True,
            alignment_summary="alignment_summary.json",
            alignment_validated=summary.get("frame_alignment", {}).get("validated"),
        )
        return summary
    except Exception as exc:
        run_context.write_status(run_dir, "analysis_failed", error=str(exc))
        _update_progress(
            run_dir,
            stage="failed",
            alignment_validation_passed=False,
            overall_validation_passed=False,
            error=str(exc),
        )
        raise
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def cleanup_successful_run(run_dir: Path, camera_count: int) -> None:
    """Remove transient progress state while preserving scientific provenance."""

    run_dir = Path(run_dir)
    _ = camera_count  # Retained for compatibility with existing callers.
    (run_dir / PROGRESS_FILENAME).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--camera-count", type=int, default=1)
    parser.add_argument(
        "--full-analysis",
        action="store_true",
        help=(
            "recover canonical frame metadata from verbose ledgers before "
            "validation; normal capture writes frames.csv live"
        ),
    )
    parser.add_argument("--enable-infer", action="store_true")
    parser.add_argument("--align", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.full_analysis:
            result = finalize_run(
                args.run_dir,
                camera_count=max(1, int(args.camera_count)),
                enable_infer=bool(args.enable_infer),
            )
        else:
            result = fast_finalize_run(
                args.run_dir,
                camera_count=max(1, int(args.camera_count)),
            )
    except Exception as exc:
        print(f"[{_timestamp()}] [POST-RUN] ERROR: {exc}", flush=True)
        return 1
    if args.align and result.validation_passed:
        try:
            summary = align_run(
                args.run_dir,
                video_validation=result.video_validation,
            )
            print(
                f"[{_timestamp()}] [POST-RUN] alignment complete: "
                f"{summary.get('counts', {})}",
                flush=True,
            )
        except Exception as exc:
            print(f"[{_timestamp()}] [POST-RUN] alignment failed: {exc}", flush=True)
            return 3
    elif result.validation_passed:
        _update_progress(
            args.run_dir,
            stage="complete",
            overall_validation_passed=True,
        )
    if result.validation_passed:
        cleanup_successful_run(args.run_dir, max(1, int(args.camera_count)))
    return 0 if result.validation_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
