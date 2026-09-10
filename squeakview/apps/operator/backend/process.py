from __future__ import annotations

"""Subprocess helpers for the operator GUI."""

import copy
import hashlib
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from squeakview import config as squeakview_config
from squeakview.apps.operator.backend.contracts import LaunchConfig, RunRequest
from squeakview.apps.operator.backend import supervision
from squeakview.common.bounded_input import read_json_object, read_stable_regular_file
from squeakview.common.device_context import file_identity


WORKSPACE = squeakview_config.WORKSPACE

INFERENCE_ENTRY = "squeakview.apps.inference.main"
POST_RUN_ENTRY = "squeakview.apps.inference.post_run"
MAX_INFERENCE_CONFIG_BYTES = 1024 * 1024
MAX_POSE_SIDECAR_BYTES = 4 * 1024 * 1024
MAX_LABEL_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class EffectiveDeepStreamConfig:
    """Immutable identities of every model artifact consumed by capture."""

    path: Path
    artifacts: Mapping[str, Mapping[str, object]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())
        object.__setattr__(
            self,
            "artifacts",
            MappingProxyType(
                {
                    name: MappingProxyType(copy.deepcopy(dict(identity)))
                    for name, identity in self.artifacts.items()
                }
            ),
        )

    @property
    def config_identity(self) -> Mapping[str, object]:
        return self.artifacts["deepstream_config"]

    @property
    def parser_identity(self) -> Mapping[str, object]:
        return self.artifacts["custom_parser"]

    def manifest_snapshot(self) -> dict[str, object]:
        return {name: dict(identity) for name, identity in self.artifacts.items()}


def preview_socket_paths(run_dir: Path, num_cameras: int) -> tuple[Path, ...]:
    """Return short, run-unique Unix socket paths below Linux's path limit."""
    identity = str(Path(run_dir).expanduser().resolve()).encode("utf-8")
    token = hashlib.blake2s(identity, digest_size=6).hexdigest()
    return tuple(
        Path("/tmp") / f"squeakview-preview-{token}-cam{index}.sock"
        for index in range(max(1, int(num_cameras)))
    )


def _prepend_env_path(env: dict[str, str], key: str, paths: list[Path]) -> None:
    values = [str(path) for path in paths if path.exists()]
    current = os.environ.get(key, "")
    if current:
        values.append(current)
    if values:
        env[key] = os.pathsep.join(values)


def _deepstream_runtime_env() -> dict[str, str]:
    # DeepStream 9.1 still selects Stream multiplexer 2 through this
    # compatibility switch.  Pin it rather than inheriting a login-shell
    # value so acquisition and post-run replay use the same mux implementation.
    env: dict[str, str] = {"USE_NEW_NVSTREAMMUX": "yes"}
    sdk = squeakview_config.DEEPSTREAM_SDK_ROOT
    _prepend_env_path(env, "LD_LIBRARY_PATH", [sdk / "lib"])
    _prepend_env_path(env, "GST_PLUGIN_PATH", [sdk / "lib" / "gst-plugins"])
    return env


def _resolve_infer_config_path(raw: str, config_dir: Path) -> Path:
    path = Path(raw.strip().strip('"')).expanduser()
    if path.is_absolute():
        resolved = squeakview_config.resolve_workspace_path(path)
        return resolved if resolved is not None else path
    return (config_dir / path).resolve()


def _localize_deepstream_config(
    config_path: Path,
    run_dir: Path | None,
    emit: Callable[[str], None],
) -> Path:
    """Write a run-local nvinfer config with paths resolved for this clone."""
    config_path = Path(config_path).expanduser().resolve()
    if run_dir is None:
        return config_path
    try:
        config_text = read_stable_regular_file(
            config_path,
            max_bytes=MAX_INFERENCE_CONFIG_BYTES,
            label="DeepStream inference config",
        ).decode("utf-8", errors="strict")
        lines = config_text.splitlines()
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise RuntimeError(
            f"could not safely localize DeepStream config {config_path}: {exc}"
        ) from exc

    config_dir = config_path.parent
    localized_dir = Path(run_dir) / "config"
    localized_config = localized_dir / config_path.name
    class_labels_target = localized_dir / f"{config_path.stem}.classes.txt"
    class_labels_payload: str | None = None
    path_keys = {
        "onnx-file",
        "model-engine-file",
        "labelfile-path",
        "custom-lib-path",
    }
    localized_lines: list[str] = []
    changed = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            localized_lines.append(line)
            continue
        key, raw_value = line.split("=", 1)
        key_name = key.strip()
        if key_name not in path_keys:
            localized_lines.append(line)
            continue
        resolved = _resolve_infer_config_path(raw_value, config_dir)
        if key_name == "labelfile-path":
            try:
                class_labels_payload = read_stable_regular_file(
                    resolved,
                    max_bytes=MAX_LABEL_BYTES,
                    label="class labels",
                ).decode("utf-8", errors="strict")
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                raise RuntimeError(
                    f"could not safely localize class labels {resolved}: {exc}"
                ) from exc
            resolved = class_labels_target.resolve()
        new_line = f"{key}={resolved}"
        localized_lines.append(new_line)
        changed = changed or new_line != line

    sidecar_path = config_path.with_name(f"{config_path.stem}.pose.json")
    localized_sidecar: Path | None = None
    localized_sidecar_payload: str | None = None
    keypoint_labels_target = localized_dir / f"{config_path.stem}.keypoints.txt"
    keypoint_labels_payload: str | None = None
    if sidecar_path.exists():
        try:
            sidecar = read_json_object(
                sidecar_path,
                max_bytes=MAX_POSE_SIDECAR_BYTES,
                label="pose sidecar",
            )
            if sidecar.get("keypoint_labels_path"):
                kp_path = _resolve_infer_config_path(
                    str(sidecar["keypoint_labels_path"]), sidecar_path.parent
                )
                keypoint_labels_payload = read_stable_regular_file(
                    kp_path,
                    max_bytes=MAX_LABEL_BYTES,
                    label="keypoint labels",
                ).decode("utf-8", errors="strict")
                sidecar["keypoint_labels_path"] = str(
                    keypoint_labels_target.resolve()
                )
            localized_sidecar = localized_dir / sidecar_path.name
            localized_sidecar_payload = json.dumps(sidecar, indent=2) + "\n"
        except Exception as exc:
            raise RuntimeError(
                f"could not safely localize pose sidecar {sidecar_path}: {exc}"
            ) from exc

    from squeakview.common import run_context

    localized_dir.mkdir(parents=True, exist_ok=True)
    run_context.atomic_write_text(localized_config, "\n".join(localized_lines) + "\n")
    if class_labels_payload is not None:
        run_context.atomic_write_text(class_labels_target, class_labels_payload)
    if keypoint_labels_payload is not None:
        run_context.atomic_write_text(keypoint_labels_target, keypoint_labels_payload)
    if localized_sidecar is not None and localized_sidecar_payload is not None:
        run_context.atomic_write_text(localized_sidecar, localized_sidecar_payload)

    if changed:
        emit(f"【DS】 localized DeepStream config → {localized_config}")
    return localized_config


def prepare_effective_deepstream_config(
    config_path: Path,
    run_dir: Path,
    emit: Callable[[str], None],
    *,
    validated_artifacts: Mapping[str, Path],
) -> EffectiveDeepStreamConfig:
    """Validate package paths, localize small files, and bind runtime inputs."""

    from squeakview.apps.inference.contracts import read_config_value_strict

    required = {
        "deepstream_config",
        "pose_sidecar",
        "class_labels",
        "keypoint_labels",
        "onnx",
        "engine",
        "custom_parser",
    }
    if set(validated_artifacts) != required:
        raise RuntimeError(
            "validated model artifact set is incomplete or contains unknown keys"
        )
    expected = {
        name: Path(path).expanduser().resolve()
        for name, path in validated_artifacts.items()
    }
    source_config = Path(config_path).expanduser().resolve()
    if source_config != expected["deepstream_config"]:
        raise RuntimeError(
            "selected DeepStream config does not match the validated model package"
        )
    config_keys = {
        "onnx": "onnx-file",
        "engine": "model-engine-file",
        "class_labels": "labelfile-path",
        "custom_parser": "custom-lib-path",
    }
    for artifact, key in config_keys.items():
        raw = read_config_value_strict(source_config, key)
        if not raw or _resolve_infer_config_path(
            raw, source_config.parent
        ) != expected[artifact]:
            raise RuntimeError(
                f"DeepStream {artifact} path does not match the validated model package"
            )
    source_sidecar = source_config.with_name(f"{source_config.stem}.pose.json")
    if source_sidecar != expected["pose_sidecar"]:
        raise RuntimeError("pose sidecar path does not match the validated model package")
    sidecar = read_json_object(
        source_sidecar,
        max_bytes=MAX_POSE_SIDECAR_BYTES,
        label="pose sidecar",
    )
    raw_keypoints = sidecar.get("keypoint_labels_path")
    if not raw_keypoints or _resolve_infer_config_path(
        str(raw_keypoints), source_sidecar.parent
    ) != expected["keypoint_labels"]:
        raise RuntimeError("keypoint labels path does not match the validated model package")

    localized = _localize_deepstream_config(source_config, run_dir, emit)
    localized_sidecar = localized.with_name(f"{localized.stem}.pose.json")
    localized_sidecar_data = read_json_object(
        localized_sidecar,
        max_bytes=MAX_POSE_SIDECAR_BYTES,
        label="localized pose sidecar",
    )
    localized_refs = {
        artifact: _resolve_infer_config_path(
            read_config_value_strict(localized, key) or "",
            localized.parent,
        )
        for artifact, key in config_keys.items()
    }
    for artifact in ("onnx", "engine", "custom_parser"):
        if localized_refs[artifact] != expected[artifact]:
            raise RuntimeError(
                f"localized DeepStream {artifact} path diverged from the validated package"
            )
    effective_paths = {
        "deepstream_config": localized,
        "pose_sidecar": localized_sidecar,
        "class_labels": localized_refs["class_labels"],
        "keypoint_labels": _resolve_infer_config_path(
            str(localized_sidecar_data.get("keypoint_labels_path") or ""),
            localized_sidecar.parent,
        ),
        "onnx": localized_refs["onnx"],
        "engine": localized_refs["engine"],
        "custom_parser": localized_refs["custom_parser"],
    }
    limits = {
        "deepstream_config": MAX_INFERENCE_CONFIG_BYTES,
        "pose_sidecar": MAX_POSE_SIDECAR_BYTES,
        "class_labels": MAX_LABEL_BYTES,
        "keypoint_labels": MAX_LABEL_BYTES,
        "onnx": 4 * 1024 * 1024 * 1024,
        "engine": 4 * 1024 * 1024 * 1024,
        "custom_parser": 512 * 1024 * 1024,
    }
    identities = {
        name: file_identity(path, max_bytes=limits[name])
        for name, path in effective_paths.items()
    }
    for label, identity in identities.items():
        if identity.get("available") is not True:
            raise RuntimeError(
                f"could not identify {label}: "
                f"{identity.get('error', 'unknown error')}"
            )
    return EffectiveDeepStreamConfig(localized, identities)


def verify_effective_deepstream_config(
    prepared: EffectiveDeepStreamConfig,
) -> None:
    """Fail closed if any prepared runtime artifact changed before spawn."""

    limits = {
        "deepstream_config": MAX_INFERENCE_CONFIG_BYTES,
        "pose_sidecar": MAX_POSE_SIDECAR_BYTES,
        "class_labels": MAX_LABEL_BYTES,
        "keypoint_labels": MAX_LABEL_BYTES,
        "onnx": 4 * 1024 * 1024 * 1024,
        "engine": 4 * 1024 * 1024 * 1024,
        "custom_parser": 512 * 1024 * 1024,
    }
    for name, expected in prepared.artifacts.items():
        current = file_identity(Path(str(expected["path"])), max_bytes=limits[name])
        if current != dict(expected):
            raise RuntimeError(f"effective DeepStream artifact changed before capture launch: {name}")


ProcessHandle = supervision.ProcessHandle
_should_suppress_child_output = supervision.should_suppress_child_output


def _spawn(
    module: str,
    args: Sequence[str],
    emit: Callable[[str], None],
    name: str,
    extra_env: dict[str, str] | None = None,
    on_exit: Callable[[int], None] | None = None,
    output_log_path: Path | None = None,
    parent_death_signal: signal.Signals | None = None,
) -> ProcessHandle:
    return supervision.spawn(
        module,
        args,
        emit,
        name,
        workspace=WORKSPACE,
        extra_env=extra_env,
        on_exit=on_exit,
        output_log_path=output_log_path,
        parent_death_signal=parent_death_signal,
    )


def spawn_inference(
    config: LaunchConfig,
    emit: Callable[[str], None],
    on_exit: Callable[[int], None] | None = None,
    *,
    effective_config: EffectiveDeepStreamConfig | None = None,
) -> ProcessHandle:
    backend = str(getattr(config, "capture_backend", "flir_direct") or "flir_direct").lower().strip()
    if backend != "flir_direct":
        raise RuntimeError(f"SqueakView only supports capture_backend='flir_direct' (got {backend!r})")

    args: list[str] = []
    if config.ds_cfg is not None:
        if effective_config is not None:
            verify_effective_deepstream_config(effective_config)
            ds_cfg = effective_config.path
            if Path(config.ds_cfg).expanduser().resolve() != ds_cfg.resolve():
                raise RuntimeError("capture config does not match prepared DeepStream config")
        else:
            ds_cfg = squeakview_config.resolve_workspace_path(config.ds_cfg)
            if ds_cfg is not None:
                ds_cfg = _localize_deepstream_config(ds_cfg, config.run_dir, emit)
        args += ["--cfg", str(ds_cfg)]
    args += ["--capture-backend", backend]
    args += ["--num-cameras", str(max(1, int(getattr(config, "num_cameras", 1))))]
    for camera_serial in getattr(config, "camera_serials", ()):
        args += ["--camera-serial", str(camera_serial)]
    if config.pixel_format:
        args += ["--pixel-format", str(config.pixel_format)]
    args += ["--trigger", "on" if bool(getattr(config, "trigger_on", False)) else "off"]
    args += ["--trigger-activation", str(getattr(config, "trigger_activation", "rising") or "rising")]
    if config.exposure_us is not None:
        args += ["--exposure-us", str(config.exposure_us)]
    args += ["--gain", "-1"]
    if config.width:
        args += ["--width", str(config.width)]
    if config.height:
        args += ["--height", str(config.height)]
    if config.fps:
        args += ["--fps", str(config.fps)]
    args += ["--bitrate", str(config.bitrate)]
    configured_preview_sockets = config.preview_socket_paths
    if config.run_dir is not None:
        args += ["--run-dir", str(config.run_dir)]
        if config.preview_enabled and not configured_preview_sockets:
            configured_preview_sockets = preview_socket_paths(
                config.run_dir, config.num_cameras
            )
    for socket_path in configured_preview_sockets:
        if config.preview_enabled:
            args += ["--preview-socket", str(socket_path)]
    if not config.inference_enabled:
        args.append("--disable-infer")
    if config.failure_plan is not None:
        args += ["--failure-plan", str(config.failure_plan)]
    extra_env = _deepstream_runtime_env()
    debug_profile = os.environ.get("SQUEAKVIEW_DEEPSTREAM_DEBUG_PROFILE", "0").lower()
    if debug_profile in {"1", "true", "yes", "on"}:
        extra_env.update(
            {
                "NVDS_ENABLE_LATENCY_MEASUREMENT": "1",
                "NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT": "1",
            }
        )
        emit(
            "【DS】 debug latency profiling enabled; use only for qualification "
            "until acquisition overhead is validated"
        )
    if backend == "flir_direct":
        plugin_dir = squeakview_config.FLIR_GST_PLUGIN_DIR
        if plugin_dir.exists():
            existing = extra_env.get("GST_PLUGIN_PATH") or os.environ.get("GST_PLUGIN_PATH", "")
            paths = [str(plugin_dir)]
            if existing:
                paths.append(existing)
            extra_env["GST_PLUGIN_PATH"] = os.pathsep.join(paths)
            emit(f"【DS】 GST_PLUGIN_PATH includes {plugin_dir}")
        else:
            emit(f"【DS】 WARN: FLIR direct plugin build directory not found: {plugin_dir}")
    output_log_path = (
        Path(config.run_dir) / "diagnostics" / "deepstream.log"
        if config.run_dir is not None
        else None
    )
    return _spawn(
        INFERENCE_ENTRY,
        args,
        emit,
        "【DS】",
        extra_env=extra_env,
        on_exit=on_exit,
        output_log_path=output_log_path,
        parent_death_signal=signal.SIGINT,
    )


def spawn_post_run(
    run_dir: Path,
    *,
    camera_count: int,
    enable_infer: bool,
    enable_align: bool,
) -> subprocess.Popen[bytes]:
    """Start a restart-safe post-run worker in its own process session.

    Output goes directly to the run directory so the worker can continue safely
    if the GUI exits while finalization is still in progress.
    """

    args = [
        sys.executable,
        "-m",
        POST_RUN_ENTRY,
        str(Path(run_dir).resolve()),
        "--camera-count",
        str(max(1, int(camera_count))),
    ]
    if enable_infer:
        args.append("--enable-infer")
    if enable_align:
        args.append("--align")
    env = os.environ.copy()
    env.update(_deepstream_runtime_env())
    package_root = str(WORKSPACE)
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{package_root}{os.pathsep}{current_pythonpath}"
        if current_pythonpath
        else package_root
    )
    env["PYTHONUNBUFFERED"] = "1"
    log_path = Path(run_dir) / "diagnostics" / "post_run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab", buffering=0)
    try:
        return subprocess.Popen(
            args,
            cwd=str(WORKSPACE),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    finally:
        log_handle.close()
