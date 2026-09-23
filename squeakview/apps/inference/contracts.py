"""Immutable configuration contract for the DeepStream capture subprocess."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from squeakview.common.failure_injection import FailurePlan, validate_failure_plan
from squeakview.common.bounded_input import read_stable_regular_file


MAX_INFERENCE_CONFIG_BYTES = 1024 * 1024
MAX_CLASS_LABEL_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """CLI-compatible configuration validated before pipeline construction."""

    cfg_path: Path | None = None
    capture_backend: str = "flir_direct"
    num_cameras: int = 1
    camera_serials: tuple[str, ...] = ()
    pixel_format: str = "Mono8"
    trigger_on: bool = False
    trigger_activation: str = "rising"
    exposure_us: float | None = 10000.0
    gain: float | None = -1.0
    width: int = 1280
    height: int = 720
    fps: int = 30
    bitrate: int = 4000
    preview_sockets: tuple[Path, ...] = ()
    enable_infer: bool = True
    run_dir: Path | None = None
    failure_plan: FailurePlan | None = None


def _config_text(config_path: Path) -> str:
    return read_stable_regular_file(
        config_path,
        max_bytes=MAX_INFERENCE_CONFIG_BYTES,
        label="DeepStream inference config",
    ).decode("utf-8", errors="strict")


def read_config_value_strict(config_path: Path, key: str) -> str | None:
    """Read a key from a bounded, stable, strict-UTF-8 DeepStream config."""

    lines = _config_text(config_path).splitlines()
    prefix = key.lower()
    matches: list[str] = []
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        name, value = raw.split("=", 1)
        if name.strip().lower() == prefix:
            matches.append(value.strip().strip('"'))
    if len(matches) > 1:
        raise ValueError(
            f"DeepStream inference config contains duplicate key {key!r}"
        )
    return matches[0] if matches else None


def read_config_value(config_path: Path | None, key: str) -> str | None:
    """Compatibility read returning ``None`` for an unavailable config."""

    if config_path is None:
        return None
    try:
        return read_config_value_strict(config_path, key)
    except (OSError, UnicodeDecodeError, ValueError):
        return None


def load_class_names(config_path: Path | None) -> list[str]:
    """Load nonempty labels referenced by a DeepStream inference config."""

    raw_path = read_config_value(config_path, "labelfile-path")
    if not raw_path or config_path is None:
        return []
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    text = read_stable_regular_file(
        path, max_bytes=MAX_CLASS_LABEL_BYTES, label="class labels"
    ).decode("utf-8", errors="strict")
    return [line.strip() for line in text.splitlines() if line.strip()]


def validate_config(config: InferenceConfig) -> None:
    """Reject configurations that cannot produce the requested graph safely."""

    backend = str(config.capture_backend or "").lower().strip()
    if backend != "flir_direct":
        raise ValueError(
            "SqueakView only supports capture_backend='flir_direct' "
            f"(got {backend!r})"
        )
    if isinstance(config.num_cameras, bool) or not isinstance(config.num_cameras, int):
        raise ValueError("num_cameras must be an integer")
    if config.num_cameras < 1:
        raise ValueError("num_cameras must be at least 1")
    if config.camera_serials and len(config.camera_serials) != config.num_cameras:
        raise ValueError("camera serial count must match camera count")
    if len(set(config.camera_serials)) != len(config.camera_serials):
        raise ValueError("camera serials must be unique")
    for name in ("width", "height", "fps", "bitrate"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be greater than zero")
    if not isinstance(config.pixel_format, str) or not config.pixel_format.strip():
        raise ValueError("pixel_format must be a non-empty string")
    if config.trigger_activation not in {"rising", "falling"}:
        raise ValueError("trigger_activation must be 'rising' or 'falling'")
    if config.exposure_us is not None and (
        isinstance(config.exposure_us, bool)
        or not isinstance(config.exposure_us, (int, float))
        or not math.isfinite(float(config.exposure_us))
        or float(config.exposure_us) <= 0
    ):
        raise ValueError("exposure_us must be finite and positive or null")
    if config.gain is not None:
        invalid_gain = (
            isinstance(config.gain, bool)
            or not isinstance(config.gain, (int, float))
            or not math.isfinite(float(config.gain))
        )
        if not invalid_gain:
            gain = float(config.gain)
            invalid_gain = gain < 0 and gain != -1.0
        if invalid_gain:
            raise ValueError(
                "gain must be -1 (automatic), finite nonnegative, or null"
            )
    if config.enable_infer:
        if config.cfg_path is None:
            raise ValueError(
                "DeepStream config (--cfg) is required when inference is enabled"
            )
        if not Path(config.cfg_path).is_file():
            raise FileNotFoundError(
                f"DeepStream config does not exist: {config.cfg_path}"
            )
        batch_size = read_config_value_strict(Path(config.cfg_path), "batch-size")
        if batch_size is not None and int(batch_size) != int(config.num_cameras):
            raise ValueError(
                f"nvinfer config batch-size ({batch_size}) does not match camera count "
                f"({config.num_cameras})"
            )
    if config.preview_sockets and len(config.preview_sockets) != config.num_cameras:
        raise ValueError(
            "preview socket count "
            f"({len(config.preview_sockets)}) does not match camera count "
            f"({config.num_cameras})"
        )
    if config.failure_plan is not None:
        validate_failure_plan(config.failure_plan)
        if config.failure_plan.stream_id >= config.num_cameras:
            raise ValueError(
                "failure plan stream_id must be smaller than num_cameras"
            )


# Compatibility names retained while callers migrate to the public helpers.
_read_config_value = read_config_value
_load_class_names = load_class_names
_validate_config = validate_config


__all__ = [
    "InferenceConfig",
    "load_class_names",
    "read_config_value",
    "read_config_value_strict",
    "validate_config",
]
