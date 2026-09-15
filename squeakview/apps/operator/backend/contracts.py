from __future__ import annotations

"""Typed immutable contracts at the operator/backend boundary."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol

from squeakview.apps.operator.backend.events import BackendEvent, RunSnapshot

from squeakview import config as squeakview_config
from squeakview.common.immutable import deep_freeze


@dataclass(frozen=True, slots=True)
class RunRequest:
    capture_backend: str = "flir_direct"
    width: int | None = 1440
    height: int | None = 1080
    fps: int | None = 30
    pixel_format: str | None = "Mono8"
    trigger_on: bool = False
    trigger_activation: str = "rising"
    ds_cfg: Path | None = squeakview_config.DEFAULT_INFER_CONFIG
    inference_enabled: bool = True
    num_cameras: int = 1
    camera_serials: tuple[str, ...] = ()
    bitrate: int = 4000
    exposure_us: float | None = 10000.0
    serial_enabled: bool = True
    serial_port: str = "/dev/ttyACM0"
    serial_baud: int = 115200
    controller_protocol: str = "legacy"
    controller_watchdog_lease_ms: int = 1500
    arduino_fps: int = 30
    preview_window_id: int | None = None
    preview_enabled: bool = True
    preview_socket_paths: tuple[Path, ...] = ()
    run_dir: Path | None = None
    mouse_id: str | None = None
    experiment_name: str | None = None
    task_cfg: Path | None = None
    failure_plan: Path | None = None
    bottles: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Snapshot every caller-owned container so a running request cannot
        # change underneath the lifecycle owner. Runtime policy performs the
        # semantic validation; this boundary guarantees structural ownership.
        if isinstance(self.camera_serials, (str, bytes)):
            raise TypeError("camera_serials must be a sequence of strings")
        if isinstance(self.preview_socket_paths, (str, bytes, Path)):
            raise TypeError("preview_socket_paths must be a sequence of paths")
        object.__setattr__(self, "camera_serials", tuple(self.camera_serials))
        object.__setattr__(
            self,
            "preview_socket_paths",
            tuple(Path(path) for path in self.preview_socket_paths),
        )
        object.__setattr__(
            self,
            "bottles",
            deep_freeze(dict(self.bottles)),
        )


# Compatibility name retained while external callers migrate.
LaunchConfig = RunRequest


class BackendProtocol(Protocol):
    """Narrow typed surface consumed by the Qt presentation process."""

    @property
    def current_snapshot(self) -> RunSnapshot: ...

    @property
    def runtime_config(self) -> RunRequest: ...

    def subscribe(self, callback: Callable[[BackendEvent], None]) -> None: ...

    def start_run(self, config: RunRequest) -> bool: ...

    def stop_run(self) -> None: ...

    def clear_feeder_jam(self) -> str: ...

    def shutdown(self) -> None: ...

    def save_bottle_measurements(
        self, bottles: dict[str, object] | None, run_dir: Path | None = None
    ) -> dict[str, object]: ...


__all__ = ["BackendProtocol", "LaunchConfig", "RunRequest"]
