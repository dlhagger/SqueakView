from __future__ import annotations

"""Run-manifest provenance and output-inventory services.

This module deliberately has no acquisition lifecycle responsibilities.  It
builds and persists a point-in-time description from state owned by the
operator backend, and preserves acquisition provenance when a completed run is
opened later to add bottle measurements.
"""

import copy
import hashlib
import os
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from squeakview.apps.operator.backend.contracts import RunRequest
from squeakview.common import run_context
from squeakview.common.capture_policy import capture_buffer_policy
from squeakview.common.device_context import device_context_snapshot, file_identity
from squeakview.common.storage_policy import resolve_storage_reserve_policy


LogCallback = Callable[[str], None]
SnapshotCallback = Callable[[Path], dict[str, Any]]

DURABLE_SUPERVISOR_OWNER = "durable_supervisor"
IN_PROCESS_DEV_OWNER = "in_process_dev"
_ACQUISITION_OWNERS = frozenset(
    {DURABLE_SUPERVISOR_OWNER, IN_PROCESS_DEV_OWNER}
)
TASK_CONFIG_SNAPSHOT_PATH = Path("config/task.yaml")
MAX_TASK_CONFIG_BYTES = 1024 * 1024
RUN_MANIFEST_SCHEMA_VERSION = "3.0"


def _preflight_evidence_valid(value: Mapping[str, object] | None) -> bool:
    digest = value.get("output_sha256") if isinstance(value, Mapping) else None
    size = value.get("output_size_bytes") if isinstance(value, Mapping) else None
    return bool(
        isinstance(value, Mapping)
        and value.get("schema_version") == "3.0"
        and value.get("passed") is True
        and value.get("skipped") is False
        and value.get("ffprobe_available") is True
        and value.get("video_decode_validated") is True
        and value.get("new_streammux_validated") is True
        and value.get("automatic_suspend_disabled") is True
        and isinstance(size, int)
        and not isinstance(size, bool)
        and size > 0
        and isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )


def snapshot_task_config(run_dir: Path, source: Path) -> dict[str, object]:
    """Atomically preserve a bounded task definition inside a new run."""

    original = Path(source).expanduser().resolve(strict=True)
    source_stat = original.stat()
    if not stat.S_ISREG(source_stat.st_mode):
        raise ValueError(f"task config is not a regular file: {original}")
    if source_stat.st_size > MAX_TASK_CONFIG_BYTES:
        raise ValueError(
            "task config exceeds bounded snapshot limit of "
            f"{MAX_TASK_CONFIG_BYTES} bytes: {original}"
        )
    target = Path(run_dir) / TASK_CONFIG_SNAPSHOT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as output:
            with original.open("rb") as source_handle:
                while chunk := source_handle.read(64 * 1024):
                    size += len(chunk)
                    if size > MAX_TASK_CONFIG_BYTES:
                        raise ValueError(
                            "task config grew beyond bounded snapshot limit while copying"
                        )
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return {
        "original_path": str(original),
        "snapshot_path": TASK_CONFIG_SNAPSHOT_PATH.as_posix(),
        "size_bytes": size,
        "sha256": digest.hexdigest(),
    }


def production_disqualifiers(
    failure_plan: Mapping[str, object] | None,
    *,
    acquisition_owner: str = IN_PROCESS_DEV_OWNER,
    preflight_evidence: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return explicit reasons a run must not be treated as scientific production."""

    env = os.environ if environ is None else environ
    reasons: list[str] = []
    if acquisition_owner != DURABLE_SUPERVISOR_OWNER:
        reasons.append("in_process_acquisition_owner")
    if not _preflight_evidence_valid(preflight_evidence):
        reasons.append("preflight_unverified")
    if failure_plan is not None:
        reasons.append("failure_injection")
    if str(env.get("SQUEAKVIEW_SKIP_PREFLIGHT", "0")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        reasons.append("preflight_skipped")
    if str(env.get("SQUEAKVIEW_DEEPSTREAM_DEBUG_PROFILE", "0")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        reasons.append("deepstream_debug_profile")
    if env.get("SQUEAKVIEW_ENABLE_SUPERVISOR_FAILURE_INJECTION") == "1":
        reasons.append("supervisor_failure_injection")
    if str(env.get("SQUEAKVIEW_SUPERVISOR_FAILURE_BARRIER", "")).strip():
        reasons.append("supervisor_failure_barrier")
    return tuple(reasons)


@dataclass(frozen=True, slots=True)
class RunManifestContext:
    """Immutable references needed to describe one acquisition run."""

    config: RunRequest
    application_root: Path
    project_root: Path
    project_id: str
    project_name: str
    runs_root: Path
    created_at: str | None
    storage: Mapping[str, Any]
    model_snapshot: Mapping[str, Any] | None
    device_context: Mapping[str, object] | None
    effective_deepstream: Mapping[str, object] | None = None
    failure_plan: Mapping[str, object] | None = None
    acquisition_owner: str = IN_PROCESS_DEV_OWNER
    task_config_snapshot: Mapping[str, object] | None = None
    preflight_evidence: Mapping[str, object] | None = None
    clock_validation: Mapping[str, object] | None = None
    qualification_case: Mapping[str, object] | None = None
    controller_watchdog: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if self.acquisition_owner not in _ACQUISITION_OWNERS:
            raise ValueError(
                f"unsupported acquisition owner: {self.acquisition_owner!r}"
            )
        if self.qualification_case is not None:
            object.__setattr__(
                self,
                "qualification_case",
                copy.deepcopy(dict(self.qualification_case)),
            )
        if self.effective_deepstream is not None:
            object.__setattr__(
                self,
                "effective_deepstream",
                copy.deepcopy(dict(self.effective_deepstream)),
            )
        if self.controller_watchdog is not None:
            object.__setattr__(
                self,
                "controller_watchdog",
                copy.deepcopy(dict(self.controller_watchdog)),
            )


class RunManifestService:
    """Build and atomically persist versioned run manifests."""

    TERMINAL_STATES = frozenset(
        {
            "post_run_complete",
            "analysis_complete",
            "finalized",
            "finalization_failed",
            "analysis_failed",
            "failed",
        }
    )

    def __init__(self, emit: LogCallback) -> None:
        self._emit = emit

    @staticmethod
    def file_info(path: Path) -> dict[str, Any]:
        exists = path.exists()
        info: dict[str, Any] = {"path": str(path), "exists": exists}
        if exists:
            try:
                info["size_bytes"] = int(path.stat().st_size)
            except OSError:
                # An inventory remains useful if an artifact disappears or
                # becomes unreadable between exists() and stat().
                pass
        return info

    @staticmethod
    def git_snapshot(application_root: Path) -> dict[str, Any]:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(application_root),
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )
            dirty = subprocess.run(
                ["git", "status", "--short"],
                cwd=str(application_root),
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )
            return {
                "commit": commit.stdout.strip() if commit.returncode == 0 else None,
                "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
            }
        except (OSError, subprocess.SubprocessError):
            return {"commit": None, "dirty": None}

    def output_snapshot(self, run_dir: Path) -> dict[str, Any]:
        artifacts = run_context.run_artifacts(run_dir)
        diagnostics_dir = run_dir / "diagnostics"
        status = run_context.read_json(artifacts.status_json)
        bottle_summary = run_context.read_json(artifacts.bottle_summary_json)
        video_files = sorted(run_dir.glob("raw*.mp4"))
        return {
            "csv_files": {
                "frames": self.file_info(artifacts.frames_csv),
                "errors": self.file_info(artifacts.drop_events_csv),
                "recording": self.file_info(diagnostics_dir / "recording.csv"),
                "camera": self.file_info(diagnostics_dir / "camera.csv"),
                "system": self.file_info(diagnostics_dir / "system.csv"),
                "objects": self.file_info(artifacts.objects_csv),
                "keypoints": self.file_info(artifacts.keypoints_csv),
                "serial": (
                    self.file_info(artifacts.serial_csv)
                    if artifacts.serial_csv
                    else None
                ),
                "bottle_measurements": self.file_info(
                    artifacts.bottle_measurements_csv
                ),
            },
            "video_files": [self.file_info(path) for path in video_files],
            "recording_validation": status.get("recording_validation"),
            "capture_reconciliation": status.get("capture_reconciliation"),
            "inference_admission": status.get("inference_admission"),
            "camera_runtime": self.file_info(
                diagnostics_dir / "camera_runtime.json"
            ),
            "camera_telemetry": self.file_info(diagnostics_dir / "camera.csv"),
            "alignment_summary": self.file_info(run_dir / "alignment_summary.json"),
            "has_analysis": (run_dir / "alignment_summary.json").exists(),
            "bottle_measurements_complete": (
                bool(bottle_summary.get("complete")) if bottle_summary else False
            ),
            "bottle_files": {
                "setup": self.file_info(artifacts.bottle_setup_json),
                "measurements": self.file_info(artifacts.bottle_measurements_csv),
                "summary": self.file_info(artifacts.bottle_summary_json),
            },
        }

    @staticmethod
    def bottle_snapshot(run_dir: Path) -> dict[str, Any]:
        artifacts = run_context.run_artifacts(run_dir)
        summary = run_context.read_json(artifacts.bottle_summary_json)
        return {
            "setup": artifacts.bottle_setup_json.name,
            "measurements": artifacts.bottle_measurements_csv.name,
            "summary": artifacts.bottle_summary_json.name,
            "complete": bool(summary.get("complete")) if summary else False,
            "sides": summary.get("sides", {}) if summary else {},
        }

    def build(
        self,
        run_dir: Path,
        context: RunManifestContext,
        *,
        output_snapshot: SnapshotCallback | None = None,
        bottle_snapshot: SnapshotCallback | None = None,
    ) -> dict[str, Any]:
        cfg = context.config
        buffer_policy = capture_buffer_policy(cfg.fps)
        artifacts = run_context.run_artifacts(run_dir)
        experiment = (cfg.experiment_name or "").strip() or None
        mouse_id = (cfg.mouse_id or "").strip() or None
        runs_root = Path(context.runs_root).resolve(strict=True)
        resolved_run_dir = Path(run_dir).resolve(strict=True)
        try:
            relative_run_dir = resolved_run_dir.relative_to(runs_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"run directory is outside the active project: {resolved_run_dir}"
            ) from exc
        inventory = output_snapshot or self.output_snapshot
        bottles = bottle_snapshot or self.bottle_snapshot
        disqualifiers = production_disqualifiers(
            context.failure_plan,
            acquisition_owner=context.acquisition_owner,
            preflight_evidence=context.preflight_evidence,
        )
        if cfg.controller_protocol == "watchdog_v1_experimental":
            disqualifiers = (*disqualifiers, "controller_watchdog_unqualified")
        storage = dict(context.storage)
        storage_policy = resolve_storage_reserve_policy()
        storage["reserve_supervision"] = {
            "min_free_bytes": storage_policy.min_free_bytes,
            "check_interval_s": storage_policy.check_interval_s,
            "failure_policy": "fatal_graceful_capture_shutdown",
        }
        return {
            "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
            "run_id": run_dir.name,
            "run_directory": str(run_dir),
            "run_directory_relative": relative_run_dir,
            "created_at": context.created_at,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "application": {"root": str(context.application_root)},
            "project": {
                "id": context.project_id,
                "name": context.project_name,
                "root": str(context.project_root),
            },
            "process_topology": {
                "acquisition_owner": context.acquisition_owner,
            },
            "platform": dict(context.device_context or device_context_snapshot()),
            "native_plugins": {
                "flir_gstreamer_source": file_identity(
                    context.application_root
                    / "native/flir_gst_source/build/gstflirspinsrc.so"
                ),
                "deepstream_yolo_parser": (
                    copy.deepcopy(context.effective_deepstream["custom_parser"])
                    if context.effective_deepstream is not None
                    and isinstance(context.effective_deepstream.get("custom_parser"), Mapping)
                    else file_identity(
                        context.application_root
                        / "native/nvdsinfer_custom_impl_yolo/libnvdsinfer_custom_impl_Yolo.so"
                    )
                ),
                "application_deepstream_yolo_parser_build": file_identity(
                    context.application_root
                    / "native/nvdsinfer_custom_impl_yolo/libnvdsinfer_custom_impl_Yolo.so"
                ),
            },
            "git": self.git_snapshot(context.application_root),
            "failure_injection": (
                dict(context.failure_plan)
                if context.failure_plan is not None
                else None
            ),
            "production_eligible": not disqualifiers,
            "production_disqualifiers": list(disqualifiers),
            "qualification": (
                dict(context.qualification_case)
                if context.qualification_case is not None
                else None
            ),
            "storage": storage,
            "experiment_name": experiment,
            "mouse_id": mouse_id,
            "capture": {
                "backend": str(getattr(cfg, "capture_backend", "flir_direct")),
                "num_cameras": int(getattr(cfg, "num_cameras", 1)),
                "camera_serials": list(getattr(cfg, "camera_serials", ())),
                "width": cfg.width,
                "height": cfg.height,
                "fps": cfg.fps,
                "pixel_format": cfg.pixel_format,
                "trigger_on": cfg.trigger_on,
                "trigger_activation": cfg.trigger_activation,
                "arduino_fps": cfg.arduino_fps,
                "exposure_us": cfg.exposure_us,
                "metadata_profile": "scientific",
                "runtime_metadata": "diagnostics/camera_runtime.json",
                "frame_identity": {
                    "camera_frame_id": "FLIR chunk FrameID",
                    "stream_frame_id": "Spinnaker Image.GetFrameID acquisition-local counter",
                    "source_sequence_index": "flirspinsrc emitted-buffer counter",
                    "missing_value_policy": "null; never substitute a sequential counter",
                },
            },
            "inference": {
                "enabled": cfg.inference_enabled,
                "deepstream_config": str(cfg.ds_cfg) if cfg.ds_cfg else None,
                "portable_source_config": (
                    {
                        "path": context.model_snapshot.get("config"),
                        "sha256": context.model_snapshot.get("config_sha256"),
                    }
                    if context.model_snapshot is not None
                    else None
                ),
                "effective_runtime": (
                    copy.deepcopy(dict(context.effective_deepstream))
                    if context.effective_deepstream is not None
                    else None
                ),
                "model_package": (
                    dict(context.model_snapshot)
                    if context.model_snapshot is not None
                    else None
                ),
                "bitrate_kbps": cfg.bitrate,
                "preview_transport": "nvunixfd",
                "preview_enabled": cfg.preview_enabled,
                "preview_sockets": [str(path) for path in cfg.preview_socket_paths],
                "flow_control": "downstream-leaky; latest pending frames retained",
                "preview_attribution": {
                    "admission": "diagnostics/preview_admission[_camN].csv",
                    "delivery": "diagnostics/preview_delivery[_camN].csv",
                    "validation": "run_status.json:preview_attribution",
                    "policy": "preview_shedding_never_invalidates_recording",
                },
                "admission_field": "frames.csv:inference_admitted",
                "streammux_implementation": "new-v2-pinned",
                "streammux_migration_status": (
                    "active; run-level frame-identity qualification required"
                ),
            },
            "recording": {
                "container": "mp4",
                "file": "raw.mp4",
                "encoder": {
                    "element": "x264enc",
                    "implementation": "software",
                    "input_format": "GRAY8",
                    "speed_preset": "ultrafast",
                    "reference_frames": 1,
                    "adaptive_quantization": False,
                    "sliced_threads": False,
                    "bitrate_kbps": cfg.bitrate,
                    "rate_control": "bitrate-controlled",
                    "pixel_fidelity": "lossy",
                },
                "scientific_claim": "temporal_frame_completeness",
                "record_queue_capacity_frames": buffer_policy.record_queue_frames,
                "backpressure_warning_frames": buffer_policy.record_warning_frames,
                "backpressure_failure_frames": buffer_policy.record_failure_frames,
                "source_transport_buffer_count": buffer_policy.source_transport_buffers,
                "backpressure_telemetry": "diagnostics/recording.csv",
                "frame_manifest": artifacts.frames_csv.name,
                "drop_events": "diagnostics/errors.csv",
                "validation": "run_status.json:recording_validation",
                "camera_telemetry": "diagnostics/camera.csv",
                "objects": artifacts.objects_csv.name,
                "keypoints": artifacts.keypoints_csv.name,
                "serial": (
                    artifacts.serial_csv.name
                    if cfg.serial_enabled and artifacts.serial_csv
                    else None
                ),
            },
            "task_config": (
                dict(context.task_config_snapshot)
                if context.task_config_snapshot is not None
                else None
            ),
            "preflight": (
                dict(context.preflight_evidence)
                if context.preflight_evidence is not None
                else None
            ),
            "clock_validation": (
                dict(context.clock_validation)
                if context.clock_validation is not None
                else None
            ),
            "serial": {
                "enabled": cfg.serial_enabled,
                "port": cfg.serial_port if cfg.serial_enabled else None,
                "baud": cfg.serial_baud if cfg.serial_enabled else None,
                "alignment_required": bool(cfg.serial_enabled and cfg.trigger_on),
                "controller_protocol": cfg.controller_protocol,
                "watchdog": (
                    dict(context.controller_watchdog)
                    if context.controller_watchdog is not None
                    else None
                ),
            },
            "observability": {
                "preflight_skipped": "preflight_skipped" in disqualifiers,
                "supervisor_failure_barrier": (
                    os.environ.get("SQUEAKVIEW_SUPERVISOR_FAILURE_BARRIER", "").strip()
                    or None
                ),
                "system_telemetry": "diagnostics/system.csv",
                "system_telemetry_schema_version": "1.0",
                "system_telemetry_interval_ms": 1000,
                "deepstream_log": "diagnostics/deepstream.log",
                "deepstream_log_max_bytes": 67108864,
                "deepstream_debug_profile": os.environ.get(
                    "SQUEAKVIEW_DEEPSTREAM_DEBUG_PROFILE", "0"
                ).lower()
                in {"1", "true", "yes", "on"},
                "deepstream_debug_probes": (
                    dict(context.preflight_evidence.get("deepstream_debug_probes"))
                    if isinstance(context.preflight_evidence, Mapping)
                    and isinstance(
                        context.preflight_evidence.get("deepstream_debug_probes"),
                        Mapping,
                    )
                    else None
                ),
                "capture_validity_dependency": "best_effort; required only for qualification",
            },
            "bottles": bottles(run_dir),
            "expected_outputs": {
                "status": run_context.RUN_STATUS_FILENAME,
                "manifest": run_context.RUN_MANIFEST_FILENAME,
                "camera_runtime": "diagnostics/camera_runtime.json",
                "camera_telemetry": "diagnostics/camera.csv",
                "config_dir": "config",
                "frames": artifacts.frames_csv.name,
                "drop_events": "diagnostics/errors.csv",
                "objects": artifacts.objects_csv.name if cfg.inference_enabled else None,
                "keypoints": artifacts.keypoints_csv.name if cfg.inference_enabled else None,
                "serial": (
                    artifacts.serial_csv.name
                    if cfg.serial_enabled and artifacts.serial_csv
                    else None
                ),
                "raw_video": artifacts.raw_video.name,
                "recording_path_telemetry": "diagnostics/recording.csv",
                "preview_attribution": (
                    [
                        (
                            f"diagnostics/preview_{boundary}.csv"
                            if index == 0
                            else f"diagnostics/preview_{boundary}_cam{index}.csv"
                        )
                        for index in range(int(cfg.num_cameras))
                        for boundary in ("admission", "delivery")
                    ]
                    if cfg.preview_enabled
                    else []
                ),
                "system_telemetry": "diagnostics/system.csv",
                "alignment_summary": (
                    "alignment_summary.json"
                    if cfg.serial_enabled and cfg.trigger_on
                    else None
                ),
                "bottle_setup": artifacts.bottle_setup_json.name,
                "bottle_measurements": artifacts.bottle_measurements_csv.name,
                "bottle_summary": artifacts.bottle_summary_json.name,
            },
            "actual_outputs": inventory(run_dir),
        }

    def write(
        self,
        run_dir: Path,
        context: RunManifestContext,
        *,
        required: bool = False,
        build_manifest: Callable[[Path], dict[str, Any]] | None = None,
        output_snapshot: SnapshotCallback | None = None,
        bottle_snapshot: SnapshotCallback | None = None,
    ) -> bool:
        """Persist a manifest, retaining completed-run acquisition provenance."""

        try:
            manifest_path = run_dir / run_context.RUN_MANIFEST_FILENAME
            status_path = run_dir / run_context.RUN_STATUS_FILENAME

            def read_existing(path: Path) -> dict[str, Any]:
                try:
                    path.lstat()
                except FileNotFoundError:
                    return {}
                return run_context.read_json_required(path)

            existing = read_existing(manifest_path)
            status = read_existing(status_path)
            if existing and (
                context.created_at is None
                or status.get("state") in self.TERMINAL_STATES
            ):
                if str(existing.get("schema_version")) != RUN_MANIFEST_SCHEMA_VERSION:
                    raise ValueError(
                        "unsupported completed run manifest schema; expected "
                        f"{RUN_MANIFEST_SCHEMA_VERSION}"
                    )
                inventory = output_snapshot or self.output_snapshot
                bottles = bottle_snapshot or self.bottle_snapshot
                manifest = dict(existing)
                manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                manifest["bottles"] = bottles(run_dir)
                manifest["actual_outputs"] = inventory(run_dir)
            elif build_manifest is not None:
                manifest = build_manifest(run_dir)
            else:
                manifest = self.build(run_dir, context)
            path = run_context.write_manifest(run_dir, manifest)
            self._emit(f"[BACKEND] manifest written → {path}")
            return True
        except Exception as exc:
            self._emit(f"[BACKEND] manifest write failed: {exc}")
            if required:
                raise RuntimeError(f"run manifest persistence failed: {exc}") from exc
            return False


__all__ = [
    "DURABLE_SUPERVISOR_OWNER",
    "IN_PROCESS_DEV_OWNER",
    "MAX_TASK_CONFIG_BYTES",
    "TASK_CONFIG_SNAPSHOT_PATH",
    "RunManifestContext",
    "RunManifestService",
    "snapshot_task_config",
]
