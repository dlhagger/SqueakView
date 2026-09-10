"""Isolated Jetson GStreamer decoder used by recording validation.

This module is intentionally executed in a child process.  A decoder/plugin
fault therefore cannot take down the durable finalizer, and its stdout is a
small machine-readable protocol consumed by :mod:`video_probe`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence, TextIO


DECODER = "nvv4l2decoder"
ELEMENT_FACTORIES = (
    "filesrc",
    "qtdemux",
    "h264parse",
    DECODER,
    "identity",
    "fakesink",
)
PROGRESS_INTERVAL_FRAMES = 300
_OUTPUT: TextIO = sys.stdout


def _emit(payload: dict[str, object]) -> None:
    print(json.dumps(payload, separators=(",", ":")), file=_OUTPUT, flush=True)


def decode(path: Path, *, parse_only: bool = False) -> int:
    """Parse or decode *path* completely and emit an exact access-unit count."""

    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
    except Exception as exc:
        _emit(
            {
                "type": "result",
                "count": None,
                "error": f"GStreamer unavailable: {exc}",
            }
        )
        return 2

    Gst.init(None)
    pipeline = Gst.Pipeline.new("squeakview-recording-validator")
    if pipeline is None:
        _emit({"type": "result", "count": None, "error": "could not create GStreamer pipeline"})
        return 2

    factories = tuple(
        factory
        for factory in ELEMENT_FACTORIES
        if not (parse_only and factory == DECODER)
    )
    elements = {
        factory: Gst.ElementFactory.make(factory, f"validator-{factory}")
        for factory in factories
    }
    missing = [factory for factory, element in elements.items() if element is None]
    if missing:
        _emit(
            {
                "type": "result",
                "count": None,
                "error": f"missing GStreamer element(s): {', '.join(missing)}",
            }
        )
        return 2

    source = elements["filesrc"]
    demux = elements["qtdemux"]
    parser = elements["h264parse"]
    decoder = elements.get(DECODER)
    counter = elements["identity"]
    sink = elements["fakesink"]
    source.set_property("location", str(path))
    parser.set_property("disable-passthrough", True)
    if decoder is not None:
        decoder.set_property("drop-frame-interval", 0)
        decoder.set_property("skip-frames", 0)
        decoder.set_property("enable-error-check", True)
    counter.set_property("silent", True)
    counter.set_property("signal-handoffs", False)
    counter.set_property("sync", False)
    sink.set_property("sync", False)
    sink.set_property("async", False)
    sink.set_property("qos", False)

    for element in elements.values():
        pipeline.add(element)
    media_linked = (
        parser.link(counter)
        if decoder is None
        else parser.link(decoder) and decoder.link(counter)
    )
    linked = source.link(demux) and media_linked and counter.link(sink)
    if not linked:
        _emit(
            {
                "type": "result",
                "count": None,
                "error": "could not link GStreamer validator",
            }
        )
        return 2

    parser_sink = parser.get_static_pad("sink")

    def link_video_pad(_demux, pad) -> None:
        if parser_sink is None or parser_sink.is_linked():
            return
        caps = pad.get_current_caps() or pad.query_caps(None)
        structure = caps.get_structure(0) if caps and caps.get_size() else None
        if structure is not None and structure.get_name() == "video/x-h264":
            pad.link(parser_sink)

    demux.connect("pad-added", link_video_pad)

    decoded = [0]
    corrupted = [0]

    def count_buffer(_pad, info):
        buffer = info.get_buffer()
        if buffer is not None:
            decoded[0] += 1
            if buffer.has_flags(Gst.BufferFlags.CORRUPTED):
                corrupted[0] += 1
            if decoded[0] % PROGRESS_INTERVAL_FRAMES == 0:
                _emit({"type": "progress", "frames": decoded[0]})
        return Gst.PadProbeReturn.OK

    counter_src = counter.get_static_pad("src")
    if counter_src is None:
        _emit({"type": "result", "count": None, "error": "identity source pad is unavailable"})
        return 2
    counter_src.add_probe(Gst.PadProbeType.BUFFER, count_buffer)

    try:
        state = pipeline.set_state(Gst.State.PLAYING)
        if state == Gst.StateChangeReturn.FAILURE:
            _emit(
                {
                    "type": "result",
                    "count": None,
                    "error": "GStreamer refused PLAYING state",
                }
            )
            return 2
        bus = pipeline.get_bus()
        message = bus.timed_pop_filtered(
            Gst.CLOCK_TIME_NONE,
            Gst.MessageType.ERROR | Gst.MessageType.EOS,
        )
        if message is None:
            _emit({"type": "result", "count": None, "error": "GStreamer ended without EOS"})
            return 2
        if message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            detail = str(error)
            if debug:
                detail = f"{detail} ({debug})"
            _emit({"type": "result", "count": None, "error": detail})
            return 2
        if corrupted[0]:
            _emit(
                {
                    "type": "result",
                    "count": None,
                    "error": f"pipeline marked {corrupted[0]} output frame(s) corrupted",
                }
            )
            return 2
        _emit({"type": "result", "count": decoded[0], "error": None})
        return 0
    finally:
        pipeline.set_state(Gst.State.NULL)
        pipeline.get_state(5 * Gst.SECOND)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import os

    parser = argparse.ArgumentParser(description="SqueakView GStreamer recording validator")
    parser.add_argument("--output-fd", type=int)
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("video", type=Path)
    args = parser.parse_args(argv)
    global _OUTPUT
    if args.output_fd is None:
        return decode(args.video, parse_only=args.parse_only)
    with os.fdopen(args.output_fd, "w", encoding="utf-8", closefd=False) as output:
        _OUTPUT = output
        return decode(args.video, parse_only=args.parse_only)


if __name__ == "__main__":
    raise SystemExit(main())
