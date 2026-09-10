"""Qt-free system-preflight execution and actionable failure policy."""

from __future__ import annotations

import os
import hashlib
import signal
import threading
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Callable, Mapping

from squeakview.apps.inference import debug_instrumentation


GENERIC_FAILURE = "System preflight failed. Open Operator Events for the failed check."
PREFLIGHT_EVIDENCE_SCHEMA_VERSION = "3.0"
MAX_PREFLIGHT_OUTPUT_BYTES = 256 * 1024
PREFLIGHT_TERMINATE_GRACE_S = 2.0
_DECODER_PASS_PREFIX = "[PASS] Jetson H.264 full-decode validation path works"


@dataclass(frozen=True, slots=True)
class _BoundedProcessResult:
    returncode: int
    output: str
    truncated: bool = False
    timed_out: bool = False


def _run_bounded_preflight(
    argv: list[str],
    *,
    cwd: str,
    env: Mapping[str, str],
    timeout_s: float,
    on_line: Callable[[str], None] | None = None,
) -> _BoundedProcessResult:
    """Run preflight, retaining bounded output while streaming complete lines."""

    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    retained = bytearray()
    truncated = threading.Event()

    def drain() -> None:
        assert process.stdout is not None
        pending = bytearray()

        def publish(*, final: bool = False) -> None:
            while True:
                newline = pending.find(b"\n")
                if newline < 0:
                    break
                raw = bytes(pending[:newline])
                del pending[: newline + 1]
                if on_line is not None:
                    try:
                        on_line(raw.decode("utf-8", errors="replace"))
                    except Exception:
                        pass
            if final and pending and on_line is not None:
                try:
                    on_line(bytes(pending).decode("utf-8", errors="replace"))
                except Exception:
                    pass
                pending.clear()

        while True:
            try:
                # os.read returns currently available pipe data; BufferedReader
                # read(size) may wait for the entire size and make preflight
                # appear frozen until the child exits.
                chunk = os.read(process.stdout.fileno(), 8192)
            except (OSError, ValueError):
                publish(final=True)
                return
            if not chunk:
                publish(final=True)
                return
            available = MAX_PREFLIGHT_OUTPUT_BYTES - len(retained)
            if available > 0:
                retained.extend(chunk[:available])
            if len(chunk) > available:
                truncated.set()
            pending.extend(chunk)
            publish()

    reader = threading.Thread(target=drain, name="squeakview-preflight-output")
    reader.start()
    timed_out = False
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=PREFLIGHT_TERMINATE_GRACE_S)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    finally:
        reader.join(timeout=PREFLIGHT_TERMINATE_GRACE_S)
        if process.stdout is not None and not process.stdout.closed:
            process.stdout.close()
        if reader.is_alive():
            reader.join(timeout=PREFLIGHT_TERMINATE_GRACE_S)
        if reader.is_alive():
            raise RuntimeError("preflight output reader did not stop after pipe closure")
    return _BoundedProcessResult(
        int(process.returncode),
        retained.decode("utf-8", errors="replace"),
        truncated.is_set(),
        timed_out,
    )


@dataclass(frozen=True, slots=True)
class PreflightRequest:
    capture_backend: str = "flir_direct"
    inference_enabled: bool = True
    ds_cfg: Path | None = None
    serial_enabled: bool = False
    serial_port: str = "/dev/ttyACM0"


@dataclass(frozen=True, slots=True)
class PreflightResult:
    passed: bool
    message: str
    output: str = ""
    skipped: bool = False
    output_truncated: bool = False
    timed_out: bool = False
    deepstream_debug_probes: Mapping[str, object] | None = None


def evidence_snapshot(result: PreflightResult) -> dict[str, object]:
    """Return bounded, machine-checkable evidence from one completed preflight."""

    encoded = result.output.encode("utf-8", errors="strict")
    return {
        "schema_version": PREFLIGHT_EVIDENCE_SCHEMA_VERSION,
        "passed": result.passed,
        "skipped": result.skipped,
        "output_truncated": result.output_truncated,
        "timed_out": result.timed_out,
        "ffprobe_available": "[PASS] ffprobe is installed" in result.output,
        "video_decode_validated": (
            _DECODER_PASS_PREFIX in result.output
        ),
        "new_streammux_validated": (
            "[PASS] DeepStream new nvstreammux VIC/NVMM path works"
            in result.output
        ),
        "automatic_suspend_disabled": (
            "[PASS] Automatic desktop suspend on AC power is disabled"
            in result.output
        ),
        "output_size_bytes": len(encoded),
        "output_sha256": hashlib.sha256(encoded).hexdigest(),
        "deepstream_debug_probes": (
            dict(result.deepstream_debug_probes)
            if result.deepstream_debug_probes is not None
            else None
        ),
    }


def failure_message(output: str) -> str:
    """Turn known preflight failures into an actionable GUI explanation."""

    lowered = output.lower()
    if (
        "ffmpeg/ffprobe is not installed" in lowered
        or "ffmpeg is not installed" in lowered
        or "ffprobe is not installed" in lowered
        or "command -v ffprobe" in lowered
    ):
        return (
            "FFmpeg is not installed, so SqueakView cannot validate the recorded video.\n\n"
            "Open a terminal and run:\n"
            "sudo apt install ffmpeg\n\n"
            "Then start the run again. You can also run scripts/setup_jetson.sh to install "
            "FFmpeg and configure serial-port access together."
        )
    if "h.264 full-decode validation path failed" in lowered:
        return (
            "The installed NVIDIA FFmpeg cannot fully decode SqueakView recordings "
            "with the Jetson video decoder.\n\n"
            "Run sudo apt update && sudo apt full-upgrade, reboot, and run "
            "scripts/preflight.sh again. Do not begin scientific acquisition until "
            "the decoder self-test passes."
        )
    if (
        "[fail] deepstream new nvstreammux" in lowered
        or "[fail] legacy nvstreammux" in lowered
    ):
        return (
            "DeepStream 9.1 Stream multiplexer 2 could not process the required "
            "Jetson VIC/NVMM path.\n\nRun sudo apt update && sudo apt "
            "full-upgrade, reboot, and run scripts/preflight.sh again."
        )
    if "does not have effective dialout access" in lowered:
        return (
            "This login session cannot access the serial controller.\n\n"
            "Run scripts/setup_jetson.sh, reboot the Jetson so the dialout "
            "membership becomes effective, and try again."
        )
    if "serial controller port is not present" in lowered:
        return (
            "The configured serial controller port is not present.\n\n"
            "Connect the controller, confirm its /dev/ttyACM* device, and select "
            "that port in SqueakView configuration."
        )
    return GENERIC_FAILURE


def run_preflight(
    request: PreflightRequest,
    *,
    workspace: Path,
    python_bin: str,
    emit: Callable[[str], None],
    environ: Mapping[str, str] | None = None,
    timeout_s: float = 240.0,
) -> PreflightResult:
    """Execute the repository preflight without any Qt dependencies."""

    source_env = os.environ if environ is None else environ
    debug_probes: Mapping[str, object] | None = None
    if debug_instrumentation.profile_enabled(source_env):
        try:
            debug_probes = debug_instrumentation.validate_probe_modules()
        except RuntimeError as exc:
            message = str(exc)
            emit(f"[GUI] Preflight blocked debug instrumentation: {message}")
            return PreflightResult(False, message)
    if str(source_env.get("SQUEAKVIEW_SKIP_PREFLIGHT", "0")).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        emit("[GUI] Preflight skipped via SQUEAKVIEW_SKIP_PREFLIGHT=1")
        return PreflightResult(
            True,
            "Preflight skipped",
            skipped=True,
            deepstream_debug_probes=debug_probes,
        )

    script = Path(workspace) / "scripts" / "preflight.sh"
    if not script.exists():
        emit(f"[GUI] Preflight script not found: {script}")
        return PreflightResult(
            False,
            "Required system preflight is unavailable. Restore scripts/preflight.sh "
            "before starting a scientific run.",
        )

    child_env = dict(source_env)
    child_env["PYTHON_BIN"] = python_bin
    child_env["CAPTURE_BACKEND"] = request.capture_backend.lower()
    child_env["INFERENCE_ENABLED"] = "1" if request.inference_enabled else "0"
    child_env["SERIAL_ENABLED"] = "1" if request.serial_enabled else "0"
    child_env["SERIAL_PORT"] = request.serial_port
    if request.ds_cfg is not None:
        child_env["DS_CFG"] = str(request.ds_cfg)
    try:
        streamed = [False]

        def emit_preflight_line(line: str) -> None:
            streamed[0] = True
            emit(f"[PREFLIGHT] {line}")

        emit(
            "[GUI] Running per-run Jetson multimedia and hardware-decoder checks"
        )
        completed = _run_bounded_preflight(
            ["bash", str(script)],
            cwd=str(workspace),
            env=child_env,
            timeout_s=timeout_s,
            on_line=emit_preflight_line,
        )
    except Exception as exc:
        emit(f"[GUI] Preflight execution failed: {exc}")
        return PreflightResult(False, GENERIC_FAILURE)

    output = completed.output.strip()
    if completed.truncated:
        output += "\n[FAIL] Preflight output exceeded the bounded capture limit."
    if completed.timed_out:
        output += "\n[FAIL] Preflight timed out and its process group was terminated."
    passed = completed.returncode == 0 and not completed.truncated and not completed.timed_out
    emit("[GUI] Preflight passed" if passed else "[GUI] Preflight failed")
    if not streamed[0]:
        for line in output.splitlines():
            emit(f"[PREFLIGHT] {line}")
    return PreflightResult(
        passed,
        "Preflight passed" if passed else failure_message(output),
        output,
        output_truncated=completed.truncated,
        timed_out=completed.timed_out,
        deepstream_debug_probes=debug_probes,
    )


__all__ = [
    "GENERIC_FAILURE",
    "MAX_PREFLIGHT_OUTPUT_BYTES",
    "PreflightRequest",
    "PreflightResult",
    "evidence_snapshot",
    "failure_message",
    "run_preflight",
]
