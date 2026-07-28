"""Bounded-memory post-run capture reconciliation and recording validation."""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

from squeakview.common import run_context

from .service_maker_runner import FrameCsvOperator, ServiceMakerApp


PROGRESS_FILENAME = "post_run_progress.json"


def _timestamp() -> str:
    return time.strftime("%H:%M:%S")


def _update_progress(run_dir: Path, **updates) -> None:
    path = Path(run_dir) / PROGRESS_FILENAME
    payload = run_context.read_json(path)
    payload.update(updates)
    run_context.atomic_write_json(path, payload)


@dataclass(slots=True)
class StreamStats:
    source_frames: int = 0
    recorded_frames: int = 0
    unmatched_admissions: int = 0


@dataclass(slots=True)
class FinalizationResult:
    source_counts: dict[int, int]
    recorded_counts: dict[int, int]
    recorded_total: int
    validation_passed: bool
    warnings: list[str] = field(default_factory=list)


def _iter_capture_payloads(path: Path, stream_id: int) -> Iterator[dict]:
    if not path.exists():
        return
    previous_pts: int | None = None
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"invalid capture ledger {path.name}:{line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"invalid capture ledger object in {path.name}:{line_number}"
                )
            payload.setdefault("camera_index", stream_id)
            pts_ns = int(payload.get("gst_pts_ns") or 0)
            if previous_pts is not None and pts_ns < previous_pts:
                raise RuntimeError(
                    f"capture ledger is not PTS ordered at {path.name}:{line_number}"
                )
            previous_pts = pts_ns
            yield payload


def _iter_admission_pts(path: Path) -> Iterator[int]:
    if not path.exists():
        raise RuntimeError(f"recording admission ledger is missing: {path.name}")
    previous_pts: int | None = None
    with path.open(newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), 2):
            try:
                pts_ns = int(row["pts_ns"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"invalid recording admission {path.name}:{line_number}"
                ) from exc
            if previous_pts is not None and pts_ns < previous_pts:
                raise RuntimeError(
                    f"recording admissions are not PTS ordered at {path.name}:{line_number}"
                )
            previous_pts = pts_ns
            yield pts_ns


def _recorded_payloads(
    run_dir: Path,
    stream_id: int,
    stats: StreamStats,
) -> Iterator[dict]:
    capture_path = run_dir / f"capture_cam{stream_id}.jsonl"
    admission_path = run_dir / (
        "record_admission.csv"
        if stream_id == 0
        else f"record_admission_cam{stream_id}.csv"
    )
    admissions = _iter_admission_pts(admission_path)
    admission = next(admissions, None)
    for payload in _iter_capture_payloads(capture_path, stream_id):
        stats.source_frames += 1
        pts_ns = int(payload.get("gst_pts_ns") or 0)
        while admission is not None and admission < pts_ns:
            stats.unmatched_admissions += 1
            admission = next(admissions, None)
        if admission == pts_ns:
            stats.recorded_frames += 1
            admission = next(admissions, None)
            yield payload
    while admission is not None:
        stats.unmatched_admissions += 1
        admission = next(admissions, None)


def _payload_sort_key(payload: dict) -> tuple[int, int, int]:
    return (
        int(payload.get("host_received_monotonic_ns") or 0),
        int(payload.get("camera_index") or 0),
        int(payload.get("source_sequence_index") or 0),
    )


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


def _open_index(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        "CREATE TABLE recorded("
        "ordinal INTEGER PRIMARY KEY, stream_id INTEGER NOT NULL, "
        "source_sequence_index INTEGER NOT NULL, camera_frame_id TEXT)"
    )
    connection.execute(
        "CREATE TABLE inferred("
        "stream_id INTEGER NOT NULL, source_sequence_index INTEGER NOT NULL, "
        "PRIMARY KEY(stream_id, source_sequence_index)) WITHOUT ROWID"
    )
    return connection


def _index_inference_frames(connection: sqlite3.Connection, path: Path) -> None:
    if not path.exists():
        return
    with path.open(newline="") as handle, connection:
        rows: list[tuple[int, int]] = []
        for line_number, row in enumerate(csv.DictReader(handle), 2):
            try:
                rows.append(
                    (int(row["stream_id"]), int(row["source_sequence_index"]))
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"invalid inference frame ledger {path.name}:{line_number}"
                ) from exc
            if len(rows) >= 10_000:
                connection.executemany(
                    "INSERT OR IGNORE INTO inferred VALUES (?, ?)", rows
                )
                rows.clear()
        if rows:
            connection.executemany(
                "INSERT OR IGNORE INTO inferred VALUES (?, ?)", rows
            )


def _summarize_inference_admission(
    connection: sqlite3.Connection,
    recorded_counts: dict[int, int],
) -> dict:
    admitted_counts = {stream_id: 0 for stream_id in recorded_counts}
    query = (
        "SELECT r.stream_id, COUNT(i.stream_id) "
        "FROM recorded r LEFT JOIN inferred i "
        "ON i.stream_id=r.stream_id "
        "AND i.source_sequence_index=r.source_sequence_index "
        "GROUP BY r.stream_id"
    )
    for stream_id, admitted in connection.execute(query):
        admitted_counts[int(stream_id)] = int(admitted)
    return {
        "schema_version": "1.0",
        "policy": "capture_non_leaky_inference_leaky_downstream",
        "captured_frames": recorded_counts,
        "inference_admitted_frames": admitted_counts,
        "inference_skipped_frames": {
            stream_id: recorded_counts[stream_id] - admitted_counts.get(stream_id, 0)
            for stream_id in recorded_counts
        },
    }


def _promote(temp_dir: Path, run_dir: Path, paths: dict[str, str]) -> None:
    for source_name, destination_name in paths.items():
        source = temp_dir / source_name
        if source.exists():
            destination = run_dir / destination_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)


def _validate_recordings(
    run_dir: Path,
    camera_count: int,
    source_counts: dict[int, int],
    recorded_counts: dict[int, int],
) -> tuple[dict, bool]:
    artifacts = run_context.run_artifacts(run_dir)
    cameras: list[dict] = []
    failed = False
    for stream_id in range(camera_count):
        video_path = (
            artifacts.raw_video
            if stream_id == 0
            else run_dir / f"raw_cam{stream_id}.mp4"
        )
        source_frames = source_counts.get(stream_id, 0)
        record_admitted_frames = recorded_counts.get(stream_id, 0)
        probe = ServiceMakerApp._video_frame_probe(video_path)
        video_frames = probe["count"]
        exists = video_path.is_file() and video_path.stat().st_size > 0
        source_count_matches = source_frames == record_admitted_frames
        frame_count_matches = (
            video_frames == record_admitted_frames
            if video_frames is not None
            else None
        )
        if not exists or not source_count_matches or frame_count_matches is not True:
            failed = True
        cameras.append(
            {
                "stream_id": stream_id,
                "video": video_path.name,
                "exists": exists,
                "source_frames": source_frames,
                "record_admitted_frames": record_admitted_frames,
                "source_count_matches": source_count_matches,
                "video_frames": video_frames,
                "frame_count_matches": frame_count_matches,
                "frame_count_method": probe["method"],
                "frame_count_error": probe["error"],
            }
        )
    report = {
        "schema_version": "1.0",
        "policy": (
            "every_source_frame_must_be_record_admitted_and_present_in_ground_truth_video"
        ),
        "cameras": cameras,
        "passed": not failed,
    }
    return report, not failed


def finalize_run(
    run_dir: Path,
    *,
    camera_count: int,
    enable_infer: bool,
) -> FinalizationResult:
    run_dir = Path(run_dir).resolve()
    run_context.write_status(run_dir, "finalizing", stage="capture_reconciliation")
    run_context.atomic_write_json(
        run_dir / PROGRESS_FILENAME,
        {"stage": "capture_reconciliation", "frames_processed": 0},
    )
    temp_dir = Path(tempfile.mkdtemp(prefix=".post_run.", dir=run_dir))
    stats = {stream_id: StreamStats() for stream_id in range(camera_count)}
    index = _open_index(temp_dir / "post_run.sqlite")
    if enable_infer:
        _index_inference_frames(index, run_dir / "inference" / "frames.csv")
    operator = FrameCsvOperator(
        temp_dir / "frames.csv",
        meta_type=0,
        audit_dir=temp_dir / "diagnostics",
    )
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
        report, passed = _validate_recordings(
            run_dir, camera_count, source_counts, recorded_counts
        )
        _update_progress(
            run_dir,
            stage=("recording_validation_complete" if passed else "recording_validation_failed"),
            frames_processed=processed,
            recording_validation_passed=passed,
            overall_validation_passed=False if not passed else None,
        )
        run_context.write_status(
            run_dir,
            "post_run_complete",
            post_run_frames=processed,
            recording_validation_passed=passed,
            recording_validation=report,
            capture_reconciliation=reconciliation,
            inference_admission=inference_summary,
        )
        level = "PASS" if passed else "ERROR"
        print(
            f"[{_timestamp()}] [POST-RUN] {level}: finalized {processed} frames; "
            f"validation={report['cameras']}",
            flush=True,
        )
        return FinalizationResult(
            source_counts, recorded_counts, processed, passed
        )
    except Exception as exc:
        run_context.write_status(run_dir, "finalization_failed", error=str(exc))
        run_context.atomic_write_json(
            run_dir / PROGRESS_FILENAME,
            {"stage": "failed", "frames_processed": processed, "error": str(exc)},
        )
        raise
    finally:
        operator.close()
        index.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


def align_run(run_dir: Path) -> dict:
    """Build the compact alignment/validation summary without expanded CSV caches."""

    from scripts.align_run_outputs_streaming import build_alignment

    run_dir = Path(run_dir).resolve()
    temp_dir = Path(tempfile.mkdtemp(prefix=".analysis.", dir=run_dir))
    try:
        run_context.write_status(run_dir, "analyzing", stage="streaming_alignment")
        _update_progress(run_dir, stage="alignment", overall_validation_passed=None)
        summary = build_alignment(run_dir, temp_dir)
        summary["outputs"] = {}
        run_context.atomic_write_json(run_dir / "alignment_summary.json", summary)
        counts = summary.get("counts", {})
        validation = summary.get("validation", {})
        failures: list[str] = []
        if summary.get("frame_alignment", {}).get("validated") is not True:
            failures.append("frame/controller alignment")
        if validation.get("video_frame_count_matches_frames_csv") is False:
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
    """Remove recovery-only ledgers after validation and alignment have passed."""

    run_dir = Path(run_dir)
    for index in range(max(1, int(camera_count))):
        (run_dir / f"capture_cam{index}.jsonl").unlink(missing_ok=True)
        admission = (
            run_dir / "record_admission.csv"
            if index == 0
            else run_dir / f"record_admission_cam{index}.csv"
        )
        admission.unlink(missing_ok=True)
    shutil.rmtree(run_dir / "inference", ignore_errors=True)
    (run_dir / PROGRESS_FILENAME).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--camera-count", type=int, default=1)
    parser.add_argument("--enable-infer", action="store_true")
    parser.add_argument("--align", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = finalize_run(
            args.run_dir,
            camera_count=max(1, int(args.camera_count)),
            enable_infer=bool(args.enable_infer),
        )
    except Exception as exc:
        print(f"[{_timestamp()}] [POST-RUN] ERROR: {exc}", flush=True)
        return 1
    if args.align and result.validation_passed:
        try:
            summary = align_run(args.run_dir)
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
