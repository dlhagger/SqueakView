"""Entry point for DeepStream inference runner."""
from __future__ import annotations

import argparse
from pathlib import Path

from squeakview import config as squeakview_config

from . import service_maker_runner as runner


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Jetson DeepStream live runner")
    ap.add_argument("--cfg", type=Path, default=None)
    ap.add_argument("--capture-backend", choices=["flir_direct"], default="flir_direct")
    ap.add_argument("--num-cameras", type=int, default=1)
    ap.add_argument(
        "--camera-serial",
        action="append",
        default=[],
        help="Stable FLIR serial for each camera, in stream order; repeat once per camera.",
    )
    ap.add_argument("--pixel-format", type=str, default="Mono8")
    ap.add_argument("--trigger", choices=["on", "off"], default="off")
    ap.add_argument("--trigger-activation", choices=["rising", "falling"], default="rising")
    ap.add_argument("--exposure-us", type=float, default=10000.0)
    ap.add_argument("--gain", type=float, default=-1.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bitrate", type=int, default=4000)
    ap.add_argument(
        "--preview-socket",
        type=Path,
        action="append",
        default=[],
        help="Unix socket used to publish one camera's NVMM preview frames.",
    )
    ap.add_argument("--disable-infer", action="store_true", help="Disable YOLO inference overlays.")
    ap.add_argument("--run-dir", type=Path, default=None, help="Existing run directory to reuse.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = squeakview_config.resolve_workspace_path(args.cfg) if args.cfg else None
    config = runner.InferenceConfig(
        cfg_path=cfg_path,
        capture_backend=args.capture_backend,
        num_cameras=max(1, int(args.num_cameras)),
        camera_serials=tuple(str(value).strip() for value in args.camera_serial),
        pixel_format=str(args.pixel_format),
        trigger_on=args.trigger == "on",
        trigger_activation=str(args.trigger_activation),
        exposure_us=float(args.exposure_us),
        gain=float(args.gain),
        width=args.width,
        height=args.height,
        fps=args.fps,
        bitrate=args.bitrate,
        preview_sockets=tuple(args.preview_socket),
        enable_infer=not args.disable_infer,
        run_dir=args.run_dir,
    )
    return runner.run(config)


if __name__ == "__main__":
    raise SystemExit(main())
