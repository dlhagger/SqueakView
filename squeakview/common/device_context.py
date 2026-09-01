"""Immutable Jetson software/hardware provenance for scientific run manifests."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path


_PACKAGES = (
    "nvidia-l4t-core",
    "deepstream-9.1",
    "cuda-toolkit-13-2",
    "libcudnn9-cuda-13",
    "libnvinfer10",
    "libgstreamer1.0-0",
    "libspinnaker",
)


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace").replace("\x00", "").strip() or None
    except OSError:
        return None


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def file_identity(path: Path) -> dict[str, object]:
    """Return an auditable identity for a native binary used by acquisition."""

    resolved = path.resolve()
    try:
        stat = resolved.stat()
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        return {"path": str(resolved), "available": False, "error": str(exc)}
    return {
        "path": str(resolved),
        "available": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def device_context_snapshot() -> dict[str, object]:
    """Capture exact runtime versions for one scientific run."""

    package_output = _command_output(
        ["dpkg-query", "-W", "-f=${Package}=${Version}\\n", *_PACKAGES]
    )
    packages: dict[str, str] = {}
    for line in (package_output or "").splitlines():
        name, separator, version = line.partition("=")
        if separator and name and version:
            packages[name] = version

    return {
        "device_model": _read_text(Path("/proc/device-tree/model")),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "jetson_linux_release": _read_text(Path("/etc/nv_tegra_release")),
        "deepstream_build": _read_text(
            Path("/opt/nvidia/deepstream/deepstream/version")
        ),
        "nvpmodel": _command_output(["nvpmodel", "-q"]),
        "packages": packages,
    }
