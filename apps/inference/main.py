"""Entry point for DeepStream inference runner."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import runner


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Jetson DeepStream live runner")
    ap.add_argument("--sock", default="/tmp/cam.sock")
    ap.add_argument("--cfg", type=Path, required=True)
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
    config = runner.InferenceConfig(
        sock=args.sock,
        cfg_path=args.cfg,
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
