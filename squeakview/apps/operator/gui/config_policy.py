"""Qt-free collection and validation policy for operator configuration fields."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from squeakview import model_package


@dataclass(frozen=True, slots=True)
class ConfigFields:
    width: str
    height: str
    fps: str
    bitrate: str
    arduino_fps: str
    serial_baud: str
    exposure_us: str
    pixel_format: str
    capture_backend: str
    trigger_enabled: bool
    serial_enabled: bool
    serial_port: str
    inference_enabled: bool
    ds_cfg: str
    task_cfg: str
    camera_count: int
    mouse_id: str
    experiment_mode: str
    experiment_name: str


@dataclass(frozen=True, slots=True)
class ConfigError:
    title: str
    message: str


@dataclass(frozen=True, slots=True)
class ConfigCollection:
    config: dict[str, Any] | None
    error: ConfigError | None


def _failure(title: str, message: str) -> ConfigCollection:
    return ConfigCollection(None, ConfigError(title, message))


def collect_config(
    fields: ConfigFields,
    *,
    include_mode: bool,
    resolve_path: Callable[[str], Path | None] | None = None,
    resolve_model_path: Callable[[str], Path | None] | None = None,
    resolve_task_path: Callable[[str], Path | None] | None = None,
    validate_model: Callable[[Path], object],
) -> ConfigCollection:
    """Normalize a form snapshot and fail closed on invalid acquisition inputs."""

    try:
        width = int(fields.width) or 1280
        height = int(fields.height) or 720
        fps = int(fields.fps) or 30
        bitrate = int(fields.bitrate) or 4000
        arduino_fps = int(fields.arduino_fps) or 30
        serial_baud = int(fields.serial_baud) or 115200
        exposure_us = int(fields.exposure_us) if fields.exposure_us else 10000
    except ValueError:
        return _failure(
            "Invalid input",
            "Please enter valid numeric values for size, FPS, bitrate, and baud.",
        )

    model_resolver = resolve_model_path or resolve_path
    task_resolver = resolve_task_path or resolve_path
    if model_resolver is None or task_resolver is None:
        raise TypeError("model and task path resolvers are required")
    try:
        ds_cfg = (
            model_resolver(fields.ds_cfg.strip())
            if fields.inference_enabled
            else None
        )
        task_cfg = task_resolver(fields.task_cfg.strip())
    except ValueError as exc:
        return _failure("Invalid project path", str(exc))
    result: dict[str, Any] = {
        "width": width,
        "height": height,
        "fps": fps,
        "pixel_format": fields.pixel_format or "Mono8",
        "capture_backend": fields.capture_backend,
        "trigger_on": fields.trigger_enabled,
        "exposure_us": exposure_us,
        "arduino_fps": arduino_fps,
        "serial_enabled": fields.serial_enabled,
        "serial_port": fields.serial_port.strip() or "/dev/ttyACM0",
        "serial_baud": serial_baud,
        "ds_cfg": ds_cfg,
        "inference_enabled": fields.inference_enabled,
        "task_cfg": task_cfg,
        "num_cameras": fields.camera_count,
        "bitrate": bitrate,
        "mouse_id": fields.mouse_id,
    }
    if include_mode:
        result["experiment_mode"] = fields.experiment_mode
        result["experiment_name"] = (
            fields.experiment_name if fields.experiment_mode == "existing" else ""
        )
        if fields.experiment_mode == "existing" and not fields.experiment_name:
            return _failure(
                "Experiment required",
                "Please select an experiment or switch to sandbox mode.",
            )

    if fields.inference_enabled:
        if ds_cfg is None:
            return _failure(
                "Config missing",
                "DeepStream config is required when inference is enabled.",
            )
        if not ds_cfg.exists():
            return _failure(
                "Config missing",
                f"DeepStream config not found:\n{ds_cfg}",
            )
        try:
            validate_model(ds_cfg)
        except model_package.ModelPackageError as exc:
            return _failure("Invalid model package", str(exc))

    if task_cfg is None:
        return _failure(
            "Task config required",
            "Please select a task config before starting.",
        )
    if not task_cfg.exists():
        return _failure(
            "Task config missing",
            f"Task config not found:\n{task_cfg}",
        )
    return ConfigCollection(result, None)
