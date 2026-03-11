"""Entry point for DeepStream inference runner."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import runner


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Jetson DeepStream live runner")
    ap.add_argument("--sock", action="append", default=None, help="Input shm socket path (repeat for multi-camera).")
    ap.add_argument("--cfg", type=Path, default=None)
    ap.add_argument("--capture-backend", choices=["flir", "zed"], default="flir")
    ap.add_argument("--num-cameras", type=int, default=1)
    ap.add_argument("--zed-depth-enabled", action="store_true")
    ap.add_argument("--zed-depth-mode", type=str, default="NEURAL")
    ap.add_argument("--zed-depth-socket", type=str, default="/tmp/cam_depth.sock")
    ap.add_argument("--zed-depth-record", action="store_true")
    ap.add_argument("--zed-confidence-threshold", type=int, default=100)
    ap.add_argument("--zed-texture-confidence-threshold", type=int, default=100)
    ap.add_argument("--zed-depth-minimum-distance-mm", type=int, default=300)
    ap.add_argument("--zed-depth-maximum-distance-mm", type=int, default=20000)
    ap.add_argument("--zed-fill-mode", action="store_true")
    ap.add_argument("--zed-depth-stabilization", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bitrate", type=int, default=4000)
    ap.add_argument("--window-xid", type=int, default=None)
    ap.add_argument("--disable-infer", action="store_true", help="Disable YOLO inference overlays.")
    ap.add_argument("--run-dir", type=Path, default=None, help="Existing run directory to reuse.")
    ap.add_argument("--draw-skeleton", action="store_true", help="Draw pose skeleton lines.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    socks = args.sock or ["/tmp/cam.sock"]
    config = runner.InferenceConfig(
        sock=socks[0],
        socks=socks,
        cfg_path=args.cfg,
        capture_backend=args.capture_backend,
        num_cameras=max(1, int(args.num_cameras)),
        zed_depth_enabled=bool(args.zed_depth_enabled),
        zed_depth_mode=str(args.zed_depth_mode),
        zed_depth_socket=str(args.zed_depth_socket),
        zed_depth_record=bool(args.zed_depth_record),
        zed_confidence_threshold=int(args.zed_confidence_threshold),
        zed_texture_confidence_threshold=int(args.zed_texture_confidence_threshold),
        zed_depth_minimum_distance_mm=int(args.zed_depth_minimum_distance_mm),
        zed_depth_maximum_distance_mm=int(args.zed_depth_maximum_distance_mm),
        zed_fill_mode=bool(args.zed_fill_mode),
        zed_depth_stabilization=int(args.zed_depth_stabilization),
        width=args.width,
        height=args.height,
        fps=args.fps,
        bitrate=args.bitrate,
        window_xid=args.window_xid,
        enable_infer=not args.disable_infer,
        run_dir=args.run_dir,
        draw_skeleton=args.draw_skeleton,
    )
    runner.run(config)


if __name__ == "__main__":
    main()
