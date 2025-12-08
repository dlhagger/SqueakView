"""Entry point for the capture server wrapping PySpin + GStreamer."""
from __future__ import annotations

import argparse

from . import pipeline


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Jetson capture publisher")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--pix", default="Mono8")
    ap.add_argument("--trigger", choices=["on", "off"], default="off")
    ap.add_argument("--activation", choices=["rising", "falling"], default="rising")
    ap.add_argument("--exposure-us", type=float, default=10000.0)
    ap.add_argument("--gain", type=float, default=None)
    ap.add_argument("--trig-timeout-ms", type=int, default=1000)
    ap.add_argument("--socket", default=pipeline.DEFAULT_SOCKET)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    config = pipeline.CaptureConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        pixel_format=args.pix,
        trigger_on=args.trigger == "on",
        trigger_activation=args.activation,
        exposure_us=args.exposure_us,
        gain=args.gain,
        trig_timeout_ms=args.trig_timeout_ms,
        socket_path=args.socket,
    )
    pipeline.run_capture(config)


if __name__ == "__main__":
    main()
