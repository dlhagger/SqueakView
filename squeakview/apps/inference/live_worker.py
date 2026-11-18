"""Optional live processing owns neither cameras, raw video, nor controller data."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import threading
import time

from squeakview.common import run_context
from squeakview.common.run_layout import artifact_path
from .contracts import InferenceConfig


def main():
    # CPU pose work yields to acquisition/encoding when the Jetson is busy.
    try:
        os.nice(5)
    except OSError:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    values = json.loads(args.config.read_text())
    values["run_dir"] = Path(values["run_dir"])
    values["cfg_path"] = Path(values["cfg_path"]) if values["cfg_path"] else None
    for key in ("live_mailboxes", "preview_sockets"):
        values[key] = tuple(Path(value) for value in values[key])
    values["camera_serials"] = tuple(values["camera_serials"])
    config = InferenceConfig(**values)
    status_path = artifact_path(config.run_dir, "diagnostics") / "live_status.json"
    stop = threading.Event()
    finished = threading.Event()
    ready = threading.Event()
    errors = []
    resources = None
    wait_thread = None
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    def status(state, error=None):
        run_context.atomic_write_json(status_path, {"state": state, "error": error,
            "heartbeat_monotonic": time.monotonic()})

    try:
        from pyservicemaker import Probe, EOSMessage, StateTransitionMessage, PipelineState
        from .capture_pipeline import CapturePipeline
        from .pipeline_builder import build_pipeline
        resources = build_pipeline(config, config.run_dir, config.run_dir / "unused.mp4",
            pipeline_factory=CapturePipeline, probe_factory=Probe,
            on_recording_fault=lambda reason: (_ for _ in ()).throw(RuntimeError(reason)))
        if config.enable_infer:
            import torch
            torch.arange(16, device="cuda").cpu()
            torch.cuda.synchronize()

        def message(event):
            if isinstance(event, StateTransitionMessage) and event.new_state == PipelineState.PLAYING and event.origin == resources.ready_origin:
                ready.set()
            if isinstance(event, EOSMessage):
                stop.set()

        def wait():
            try:
                resources.pipeline.wait()
                if not stop.is_set():
                    errors.append("live pipeline ended without EOS")
            except Exception as exc:
                errors.append(str(exc))
            finally:
                finished.set()
                stop.set()

        resources.pipeline.start(message)
        wait_thread = threading.Thread(target=wait, daemon=True)
        wait_thread.start()
        started = time.monotonic()
        while not stop.wait(1):
            if not ready.is_set() and time.monotonic() - started > 45:
                raise RuntimeError("live pipeline did not become ready")
            status("running" if ready.is_set() else "starting")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if resources is not None:
            # If vendor teardown holds the GIL, the acquisition process's
            # independent watchdog kills this optional process.
            try:
                resources.pipeline.stop()
                if wait_thread and not finished.wait(2):
                    errors.append("live pipeline did not drain")
                elif wait_thread:
                    for closer in [resources.observations, resources.frames, *resources.preview_boundaries]:
                        if closer is not None:
                            closer.close()
            except Exception as exc:
                errors.append(f"live cleanup: {exc}")
        status("degraded" if errors else "completed", "; ".join(errors) or None)
        if errors:
            print("; ".join(errors), flush=True)
            # Avoid running atexit writers against undrained native callbacks.
            os._exit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
