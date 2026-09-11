"""Crash-safe staging and atomic promotion for device-local model packages."""

from __future__ import annotations

import ctypes
import math
import os
import signal
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

from squeakview.model_package import ModelPackageInfo, validate_model_package


RENAME_EXCHANGE = 2
MAX_TRTEXEC_TIMEOUT_S = 60 * 60
MAX_TRTEXEC_OUTPUT_BYTES = 16 * 1024 * 1024


def run_trtexec_validation(
    engine: Path,
    *,
    executable: str = "trtexec",
    timeout_s: float = 120.0,
    max_output_bytes: int = 64 * 1024,
) -> dict[str, object]:
    """Execute one synthetic TensorRT inference with bounded captured output."""

    engine = Path(engine).resolve()
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(float(timeout_s))
        or not 0 < float(timeout_s) <= MAX_TRTEXEC_TIMEOUT_S
    ):
        raise ValueError(
            f"trtexec timeout must be finite and between 0 and "
            f"{MAX_TRTEXEC_TIMEOUT_S} seconds"
        )
    if (
        isinstance(max_output_bytes, bool)
        or not isinstance(max_output_bytes, int)
        or not 1 <= max_output_bytes <= MAX_TRTEXEC_OUTPUT_BYTES
    ):
        raise ValueError(
            f"trtexec output limit must be between 1 and "
            f"{MAX_TRTEXEC_OUTPUT_BYTES} bytes"
        )
    command = (
        executable,
        f"--loadEngine={engine}",
        "--iterations=1",
        "--warmUp=0",
        "--duration=0",
        "--infStreams=1",
    )
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    captured = bytearray()
    truncated = False

    def drain() -> None:
        nonlocal truncated
        assert process.stdout is not None
        while True:
            try:
                chunk = process.stdout.read(8192)
            except (OSError, ValueError):
                return
            if not chunk:
                return
            available = max_output_bytes - len(captured)
            if available > 0:
                captured.extend(chunk[:available])
            if len(chunk) > available:
                truncated = True

    reader = threading.Thread(target=drain, name="squeakview-trtexec-output", daemon=True)
    reader.start()
    try:
        timed_out = False
        try:
            returncode = process.wait(timeout=float(timeout_s))
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                returncode = process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                returncode = process.wait(timeout=5.0)
        reader.join(timeout=5.0)
        if reader.is_alive():
            process.stdout.close()  # type: ignore[union-attr]
            reader.join(timeout=1.0)
        output = captured.decode("utf-8", errors="replace")
        return {
            "command": list(command),
            "timeout_s": float(timeout_s),
            "max_output_bytes": int(max_output_bytes),
            "returncode": int(returncode),
            "timed_out": timed_out,
            "output_truncated": truncated,
            "output": output,
            "passed": not timed_out and returncode == 0,
        }
    finally:
        if process.stdout is not None and not process.stdout.closed:
            process.stdout.close()
        if reader.is_alive():
            reader.join(timeout=1.0)


def create_staging_package(models_dir: Path, model_name: str) -> Path:
    """Create a hidden package directory on the destination filesystem."""

    models_dir = Path(models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    if not model_name or Path(model_name).name != model_name:
        raise ValueError("model_name must be one path component")
    container = Path(
        tempfile.mkdtemp(prefix=f".{model_name}.build-", dir=models_dir)
    ).resolve()
    package = container / model_name
    package.mkdir()
    return package


def cleanup_staging_package(path: Path) -> None:
    """Remove only a hidden staging package created beside model packages."""

    path = Path(path).resolve()
    container = path.parent
    if not container.name.startswith(".") or ".build-" not in container.name:
        raise ValueError(f"refusing to remove a non-staging package: {path}")
    shutil.rmtree(container, ignore_errors=True)


def _fsync_tree(root: Path) -> None:
    for directory, _subdirs, files in os.walk(root):
        directory_path = Path(directory)
        for name in files:
            with (directory_path / name).open("rb") as handle:
                os.fsync(handle.fileno())
        descriptor = os.open(directory_path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exchange(left: Path, right: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError("atomic directory exchange is unavailable on this system")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(left),
        -100,
        os.fsencode(right),
        RENAME_EXCHANGE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def promote_model_package(
    staging_package: Path,
    destination: Path,
    *,
    config_name: str,
    overwrite: bool,
    validator: Callable[[Path], ModelPackageInfo] = validate_model_package,
) -> Path:
    """Validate and atomically publish one complete package.

    When replacing an existing package, Linux ``RENAME_EXCHANGE`` keeps either
    the old or the new complete directory visible at every instant. The old
    package is removed only after the exchange has succeeded.
    """

    staging_package = Path(staging_package).resolve()
    destination = Path(destination).resolve()
    if (
        staging_package.name != destination.name
        or staging_package.parent.parent != destination.parent
        or not staging_package.parent.name.startswith(".")
        or ".build-" not in staging_package.parent.name
    ):
        raise ValueError("staging package must be in its hidden destination-side container")
    if staging_package == destination:
        raise ValueError("staging package and destination must differ")
    if not staging_package.is_dir():
        raise FileNotFoundError(f"staging package does not exist: {staging_package}")
    config = staging_package / "configs" / config_name
    info = validator(config)
    if Path(info.root).resolve() != staging_package:
        raise ValueError("validator resolved a package outside the staging directory")
    _fsync_tree(staging_package)

    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"model package already exists and was preserved: {destination}"
            )
        _exchange(staging_package, destination)
        _fsync_directory(destination.parent)
        # The former destination now occupies the hidden staging package path.
        cleanup_staging_package(staging_package)
    else:
        os.rename(staging_package, destination)
        _fsync_directory(destination.parent)
    return destination / "configs" / config_name


__all__ = [
    "cleanup_staging_package",
    "create_staging_package",
    "promote_model_package",
    "run_trtexec_validation",
]
