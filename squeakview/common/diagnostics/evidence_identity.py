"""Streaming identities for the files consumed by run qualification."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path
from typing import Any, Mapping

from squeakview.common import run_context
from squeakview.common.recording_evidence import (
    capture_recording_evidence,
    recording_evidence_same_content,
)


MAX_QUALIFICATION_INPUT_BYTES = 1 << 20
MAX_RECORDING_TELEMETRY_FILES = 64
_RUN_LOCAL_EFFECTIVE_RUNTIME_FILES = (
    "deepstream_config",
    "pose_sidecar",
    "class_labels",
    "keypoint_labels",
)
_PLATFORM_FIELDS = (
    "device_model", "machine", "kernel", "python", "python_executable",
    "jetson_linux_release", "deepstream_build",
)
_REQUIRED_PACKAGES = (
    "nvidia-l4t-core", "deepstream-9.1", "cuda-toolkit-13-2",
    "libcudnn9-cuda-13", "libnvinfer10", "libgstreamer1.0-0", "libspinnaker",
)
_NATIVE_PLUGINS = ("flir_gstreamer_source", "deepstream_yolo_parser")
_CAPTURE_IDENTITY_FIELDS = (
    "backend", "num_cameras", "camera_serials", "width", "height", "fps",
    "pixel_format", "trigger_on", "trigger_activation", "arduino_fps",
    "exposure_us",
)
_CAMERA_RUNTIME_IDENTITY_FIELDS = (
    "camera_index", "camera_serial", "device_model", "firmware_version",
    "source_width", "source_height", "source_pixel_format", "actual_fps",
    "configured_exposure_us", "configured_gain_db",
    "configured_stream_buffer_count", "timestamp_increment_ns",
    "timestamp_latch_available", "enabled_chunks",
)


def stable_file_identity(path: Path, *, max_bytes: int | None = None) -> dict[str, object]:
    """Hash one regular file without following an unbounded stream."""

    unresolved = Path(path)
    resolved = unresolved.absolute()
    try:
        resolved = unresolved.resolve()
        with resolved.open("rb") as handle:
            metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("not a regular file")
            size = int(metadata.st_size)
            if max_bytes is not None and size > max_bytes:
                raise ValueError(f"file exceeds {max_bytes} byte limit")
            digest = hashlib.sha256()
            remaining = size
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise OSError("file became shorter while hashing")
                digest.update(chunk)
                remaining -= len(chunk)
            if handle.read(1):
                raise OSError("file grew while hashing")
            after = os.fstat(handle.fileno())
        final = resolved.stat()
        if (
            after.st_dev != metadata.st_dev
            or after.st_ino != metadata.st_ino
            or after.st_size != metadata.st_size
            or after.st_mtime_ns != metadata.st_mtime_ns
            or after.st_ctime_ns != metadata.st_ctime_ns
            or final.st_dev != metadata.st_dev
            or final.st_ino != metadata.st_ino
            or final.st_size != metadata.st_size
            or final.st_mtime_ns != metadata.st_mtime_ns
            or final.st_ctime_ns != metadata.st_ctime_ns
        ):
            raise OSError("file changed while hashing")
    except (OSError, ValueError) as exc:
        return {
            "path": str(resolved),
            "available": False,
            "error": str(exc),
        }
    return {
        "path": str(resolved),
        "available": True,
        "size_bytes": size,
        "mtime_ns": metadata.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def canonical_runtime_identity(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return path-independent platform/package/native-binary identity."""

    platform = manifest.get("platform")
    native = manifest.get("native_plugins")
    if not isinstance(platform, Mapping) or not isinstance(native, Mapping):
        return None
    packages = platform.get("packages")
    package_identity = (
        {name: packages[name] for name in sorted(packages)}
        if isinstance(packages, Mapping)
        and all(isinstance(name, str) for name in packages)
        else None
    )
    return {
        "platform": {
            **{name: platform.get(name) for name in _PLATFORM_FIELDS},
            "packages": package_identity,
        },
        "native_plugins": {
            name: (
                {
                    "available": value.get("available"),
                    "size_bytes": value.get("size_bytes"),
                    "sha256": value.get("sha256"),
                }
                if isinstance((value := native.get(name)), Mapping)
                else None
            )
            for name in _NATIVE_PLUGINS
        },
    }


def canonical_acquisition_identity(
    manifest: Mapping[str, Any],
    camera_runtime: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return the path-independent scientific acquisition/protocol identity."""

    capture = manifest.get("capture")
    inference = manifest.get("inference")
    serial = manifest.get("serial")
    task = manifest.get("task_config")
    cameras = camera_runtime.get("cameras") if isinstance(camera_runtime, Mapping) else None
    if not all(isinstance(value, Mapping) for value in (capture, inference, serial, task)):
        return None
    if not isinstance(cameras, list) or not all(isinstance(item, Mapping) for item in cameras):
        return None
    canonical_cameras = [
        {name: camera.get(name) for name in _CAMERA_RUNTIME_IDENTITY_FIELDS}
        for camera in cameras
    ]
    canonical_cameras.sort(
        key=lambda item: (
            str(item.get("camera_index")), str(item.get("camera_serial"))
        )
    )
    return {
        "capture": {name: capture.get(name) for name in _CAPTURE_IDENTITY_FIELDS},
        # Enabled inference and preview are deliberate matrix dimensions. The
        # encoder bitrate remains an invariant of the scientific recording.
        "recording": {"bitrate_kbps": inference.get("bitrate_kbps")},
        "serial": {
            name: serial.get(name)
            for name in ("enabled", "port", "baud", "controller_protocol")
        },
        "task_config": {
            "size_bytes": task.get("size_bytes"),
            "sha256": task.get("sha256"),
        },
        "camera_runtime": {
            "schema_version": camera_runtime.get("schema_version"),
            "metadata_type": camera_runtime.get("metadata_type"),
            "cameras": canonical_cameras,
        },
    }


def acquisition_identity_errors(identity: Mapping[str, Any] | None) -> list[str]:
    """Validate that an acquisition identity is complete enough to compare."""

    if identity is None:
        return ["acquisition_identity: required scientific identity is missing"]
    capture = identity.get("capture")
    recording = identity.get("recording")
    serial = identity.get("serial")
    task = identity.get("task_config")
    runtime = identity.get("camera_runtime")
    errors: list[str] = []
    if not isinstance(capture, Mapping):
        errors.append("acquisition_identity.capture: missing")
    else:
        missing = [name for name in _CAPTURE_IDENTITY_FIELDS if capture.get(name) is None]
        if missing:
            errors.append("acquisition_identity.capture missing: " + ", ".join(missing))
        if type(capture.get("camera_serials")) is not list:
            errors.append("acquisition_identity.capture.camera_serials: must be a list")
    bitrate = recording.get("bitrate_kbps") if isinstance(recording, Mapping) else None
    if type(bitrate) is not int or bitrate <= 0:
        errors.append("acquisition_identity.recording.bitrate_kbps: invalid")
    if not isinstance(serial, Mapping) or serial.get("enabled") is not True:
        errors.append("acquisition_identity.serial: enabled controller is required")
    elif (
        not isinstance(serial.get("port"), str)
        or not serial["port"].strip()
        or type(serial.get("baud")) is not int
        or serial["baud"] <= 0
        or serial.get("controller_protocol")
        not in {"legacy", "watchdog_v1_experimental", "v2"}
    ):
        errors.append(
            "acquisition_identity.serial: port/baud/controller protocol identity "
            "is invalid"
        )
    digest = task.get("sha256") if isinstance(task, Mapping) else None
    size = task.get("size_bytes") if isinstance(task, Mapping) else None
    if not (
        type(size) is int
        and size > 0
        and isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    ):
        errors.append("acquisition_identity.task_config: size/SHA-256 is invalid")
    cameras = runtime.get("cameras") if isinstance(runtime, Mapping) else None
    expected_count = capture.get("num_cameras") if isinstance(capture, Mapping) else None
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("schema_version") != "1.0"
        or not isinstance(runtime.get("metadata_type"), str)
        or not isinstance(cameras, list)
        or type(expected_count) is not int
        or len(cameras) != expected_count
    ):
        errors.append("acquisition_identity.camera_runtime: camera set is invalid")
    elif any(
        not isinstance(camera.get("camera_serial"), str)
        or not camera["camera_serial"].strip()
        or camera.get("camera_index") is None
        or camera.get("source_width") != capture.get("width")
        or camera.get("source_height") != capture.get("height")
        or camera.get("source_pixel_format") != capture.get("pixel_format")
        for camera in cameras
    ):
        errors.append("acquisition_identity.camera_runtime: camera identity/config is incomplete")
    elif (
        len({camera.get("camera_index") for camera in cameras}) != len(cameras)
        or len({camera.get("camera_serial") for camera in cameras}) != len(cameras)
    ):
        errors.append("acquisition_identity.camera_runtime: camera identities are not unique")
    return errors


def runtime_identity_errors(identity: Mapping[str, Any] | None) -> list[str]:
    if identity is None:
        return ["runtime_identity: platform or native plugin identity is missing"]
    platform = identity.get("platform")
    native = identity.get("native_plugins")
    errors: list[str] = []
    if not isinstance(platform, Mapping):
        errors.append("runtime_identity.platform: required identity is missing")
    else:
        missing = [
            name for name in _PLATFORM_FIELDS
            if not isinstance(platform.get(name), str) or not platform[name].strip()
        ]
        if missing:
            errors.append("runtime_identity.platform missing: " + ", ".join(missing))
        packages = platform.get("packages")
        missing = [
            name for name in _REQUIRED_PACKAGES
            if not isinstance(packages, Mapping)
            or not isinstance(packages.get(name), str)
            or not packages[name].strip()
        ]
        if missing:
            errors.append("runtime_identity.packages missing: " + ", ".join(missing))
    for name in _NATIVE_PLUGINS:
        value = native.get(name) if isinstance(native, Mapping) else None
        digest = value.get("sha256") if isinstance(value, Mapping) else None
        if not (
            isinstance(value, Mapping)
            and value.get("available") is True
            and type(value.get("size_bytes")) is int
            and value["size_bytes"] > 0
            and isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
        ):
            errors.append(f"runtime_identity.native_plugins.{name}: invalid identity")
    return errors


def capture_source_evidence(
    run_dir: Path,
    limits_path: Path | None,
    *,
    manifest_snapshot: tuple[Mapping[str, Any], Mapping[str, object]] | None = None,
    status_identity: Mapping[str, object] | None = None,
    limits_identity: Mapping[str, object] | None = None,
    recording_evidence_snapshot: object = None,
    alignment_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Identify source artifacts, retaining identities from exact parsed bytes."""

    root = Path(run_dir).resolve()
    manifest_path = root / run_context.RUN_MANIFEST_FILENAME
    if manifest_snapshot is None:
        manifest, parsed_manifest_identity = (
            run_context.read_json_required_with_identity(manifest_path)
        )
    else:
        manifest, parsed_manifest_identity = manifest_snapshot
    observability = manifest.get("observability")
    declared_log = (
        observability.get("deepstream_log")
        if isinstance(observability, Mapping)
        else None
    )
    relative_log = Path(declared_log) if isinstance(declared_log, str) and declared_log else Path("diagnostics/deepstream.log")
    log_path = (root / relative_log).resolve()
    if not log_path.is_relative_to(root):
        log_identity: dict[str, object] = {
            "path": str(log_path), "available": False,
            "error": "declared DeepStream log path escapes run directory",
        }
    else:
        log_identity = stable_file_identity(log_path)
    task_config = manifest.get("task_config")
    declared_task_snapshot = (
        task_config.get("snapshot_path")
        if isinstance(task_config, Mapping)
        else None
    )
    relative_task = (
        Path(declared_task_snapshot)
        if isinstance(declared_task_snapshot, str) and declared_task_snapshot
        else Path("config/task.yaml")
    )
    task_path = (root / relative_task).resolve()
    if not task_path.is_relative_to(root):
        task_identity: dict[str, object] = {
            "path": str(task_path),
            "available": False,
            "error": "declared task config path escapes run directory",
        }
    else:
        task_identity = stable_file_identity(
            task_path, max_bytes=MAX_QUALIFICATION_INPUT_BYTES
        )
    capture = manifest.get("capture")
    declared_camera_count = (
        capture.get("num_cameras") if isinstance(capture, Mapping) else None
    )
    recording_limit = (
        declared_camera_count
        if type(declared_camera_count) is int
        and 1 <= declared_camera_count <= MAX_RECORDING_TELEMETRY_FILES
        else MAX_RECORDING_TELEMETRY_FILES
    )
    recording_artifacts = (
        recording_evidence_snapshot
        if recording_evidence_snapshot is not None
        else capture_recording_evidence(root, declared_camera_count)
        if type(declared_camera_count) is int
        and 1 <= declared_camera_count <= MAX_RECORDING_TELEMETRY_FILES
        else None
    )
    recording_paths: list[Path] = []
    for path in (root / "diagnostics").glob("recording*.csv"):
        if len(recording_paths) >= recording_limit:
            raise ValueError(
                "recording telemetry file set exceeds bounded expected camera count"
            )
        recording_paths.append(path)
    recording_paths.sort()
    inference = manifest.get("inference")
    effective_runtime = (
        inference.get("effective_runtime")
        if isinstance(inference, Mapping)
        else None
    )
    effective_runtime_identities: dict[str, object] = {}
    for name in _RUN_LOCAL_EFFECTIVE_RUNTIME_FILES:
        declared = (
            effective_runtime.get(name)
            if isinstance(effective_runtime, Mapping)
            else None
        )
        declared_path = declared.get("path") if isinstance(declared, Mapping) else None
        if declared_path is None:
            effective_runtime_identities[name] = None
            continue
        if not isinstance(declared_path, str) or not declared_path:
            effective_runtime_identities[name] = {
                "available": False,
                "error": f"declared {name} path is invalid",
            }
            continue
        candidate = Path(declared_path)
        resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        if not resolved.is_relative_to(root):
            effective_runtime_identities[name] = {
                "path": str(resolved),
                "available": False,
                "error": f"declared {name} path escapes run directory",
            }
            continue
        effective_runtime_identities[name] = stable_file_identity(
            resolved, max_bytes=MAX_QUALIFICATION_INPUT_BYTES
        )
    preview_enabled = (
        inference.get("preview_enabled") is True
        if isinstance(inference, Mapping)
        else False
    )
    preview_attribution: list[dict[str, object]] = []
    if preview_enabled and type(declared_camera_count) is int:
        for stream_id in range(min(declared_camera_count, MAX_RECORDING_TELEMETRY_FILES)):
            suffix = "" if stream_id == 0 else f"_cam{stream_id}"
            for boundary in ("admission", "delivery"):
                preview_attribution.append(
                    stable_file_identity(
                        root / "diagnostics" / f"preview_{boundary}{suffix}.csv"
                    )
                )
    return {
        "run_manifest": dict(parsed_manifest_identity),
        "run_status": (
            dict(status_identity)
            if status_identity is not None
            else stable_file_identity(root / run_context.RUN_STATUS_FILENAME)
        ),
        "system_telemetry": stable_file_identity(root / "diagnostics/system.csv"),
        "recording_telemetry": [
            stable_file_identity(path)
            for path in recording_paths
        ],
        "recording_artifacts": recording_artifacts,
        "deepstream_log": log_identity,
        "task_config": task_identity,
        "camera_runtime": stable_file_identity(
            root / "diagnostics/camera_runtime.json",
            max_bytes=MAX_QUALIFICATION_INPUT_BYTES,
        ),
        "alignment_summary": (
            dict(alignment_identity)
            if alignment_identity is not None
            else stable_file_identity(
                root / "alignment_summary.json",
                max_bytes=MAX_QUALIFICATION_INPUT_BYTES,
            )
        ),
        "effective_runtime": effective_runtime_identities,
        "preview_attribution": preview_attribution,
        "limits": (
            (
                dict(limits_identity)
                if limits_identity is not None
                else stable_file_identity(
                    Path(limits_path).resolve(),
                    max_bytes=MAX_QUALIFICATION_INPUT_BYTES,
                )
            )
            if limits_path is not None
            else None
        ),
    }


def _same_file_identity(recorded: object, current: object) -> bool:
    if recorded is None and current is None:
        return True
    if not isinstance(recorded, Mapping) or not isinstance(current, Mapping):
        return False
    return all(
        recorded.get(name) == current.get(name)
        for name in ("path", "available", "size_bytes", "sha256")
    )


def stable_file_matches_identity(
    path: Path,
    expected: object,
    *,
    max_bytes: int | None = None,
) -> bool:
    """Return whether a stable current read has the recorded content identity."""

    return _same_file_identity(
        expected, stable_file_identity(path, max_bytes=max_bytes)
    )


def source_evidence_errors(
    run_dir: Path,
    recorded: object,
    *,
    declared_limits_path: object,
    verify_recording_content: bool = True,
) -> list[str]:
    """Report stale, missing, or substituted qualification source artifacts."""

    if not isinstance(recorded, Mapping):
        return ["qualification source evidence identity is missing"]
    limits_path = (
        Path(declared_limits_path).resolve()
        if isinstance(declared_limits_path, str) and declared_limits_path
        else None
    )
    current = capture_source_evidence(
        run_dir,
        limits_path,
        recording_evidence_snapshot=(
            recorded.get("recording_artifacts")
            if not verify_recording_content
            else None
        ),
    )
    errors: list[str] = []
    for name in (
        "run_manifest",
        "run_status",
        "system_telemetry",
        "deepstream_log",
        "task_config",
        "camera_runtime",
        "alignment_summary",
        "limits",
    ):
        if not _same_file_identity(recorded.get(name), current.get(name)):
            errors.append(f"qualification source evidence {name} is stale or invalid")
    recorded_runtime = recorded.get("effective_runtime")
    current_runtime = current.get("effective_runtime")
    if not isinstance(recorded_runtime, Mapping) or not isinstance(
        current_runtime, Mapping
    ):
        errors.append("qualification source evidence effective_runtime is stale or invalid")
    else:
        for name in _RUN_LOCAL_EFFECTIVE_RUNTIME_FILES:
            if not _same_file_identity(
                recorded_runtime.get(name), current_runtime.get(name)
            ):
                errors.append(
                    "qualification source evidence effective_runtime."
                    f"{name} is stale or invalid"
                )
    recorded_recording = recorded.get("recording_telemetry")
    current_recording = current["recording_telemetry"]
    if not isinstance(recorded_recording, list) or len(recorded_recording) != len(
        current_recording
    ):
        errors.append(
            "qualification source evidence recording telemetry set is stale or invalid"
        )
    elif any(
        not _same_file_identity(old, new)
        for old, new in zip(recorded_recording, current_recording, strict=True)
    ):
        errors.append(
            "qualification source evidence recording telemetry is stale or invalid"
        )
    recorded_preview = recorded.get("preview_attribution")
    current_preview = current.get("preview_attribution")
    if not isinstance(recorded_preview, list) or not isinstance(current_preview, list):
        errors.append(
            "qualification source evidence preview attribution is stale or invalid"
        )
    elif len(recorded_preview) != len(current_preview) or any(
        not _same_file_identity(old, new)
        for old, new in zip(recorded_preview, current_preview, strict=True)
    ):
        errors.append(
            "qualification source evidence preview attribution is stale or invalid"
        )
    if verify_recording_content:
        recorded_artifacts = recorded.get("recording_artifacts")
        current_artifacts = current.get("recording_artifacts")
        manifest = run_context.read_json(Path(run_dir) / run_context.RUN_MANIFEST_FILENAME)
        capture = manifest.get("capture") if isinstance(manifest.get("capture"), Mapping) else {}
        camera_count = capture.get("num_cameras")
        if (
            type(camera_count) is not int
            or not recording_evidence_same_content(
                recorded_artifacts, current_artifacts, camera_count
            )
        ):
            errors.append(
                "qualification source evidence recording artifacts are stale or invalid"
            )
    return errors


__all__ = [
    "MAX_QUALIFICATION_INPUT_BYTES",
    "MAX_RECORDING_TELEMETRY_FILES",
    "acquisition_identity_errors",
    "canonical_acquisition_identity",
    "canonical_runtime_identity",
    "capture_source_evidence",
    "runtime_identity_errors",
    "source_evidence_errors",
    "stable_file_identity",
    "stable_file_matches_identity",
]
