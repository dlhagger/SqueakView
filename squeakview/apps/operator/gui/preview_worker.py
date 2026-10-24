"""Disposable native renderer. Imported only by the preview subprocess."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import sys


def _status(event, **fields):
    print(json.dumps({"event": event, **fields}), flush=True)


def _guard_parent(parent_pid: int) -> None:
    # Linux/Jetson: renderer must not outlive a crashed GUI, including while
    # stuck inside a driver. Close the parent-death registration race too.
    library = ctypes.CDLL(None, use_errno=True)
    if library.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "could not establish preview parent-death signal")
    if os.getppid() != parent_pid:
        raise RuntimeError("preview parent exited before renderer startup")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--window-id", type=int, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args(argv)
    pipeline = None
    Gst = None
    try:
        _guard_parent(args.parent_pid)
        from squeakview.common.diagnostics.crash import install_crash_diagnostics
        install_crash_diagnostics("preview")
        import gi
        gi.require_version("Gst", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import GLib, Gst, GstVideo
        Gst.init(None)
        loop = GLib.MainLoop()
        pipeline = Gst.Pipeline.new("squeakview-gui-preview")
        source = Gst.ElementFactory.make("nvunixfdsrc", "preview_ipc_source")
        queue = Gst.ElementFactory.make("queue", "preview_queue")
        transform = Gst.ElementFactory.make("nvegltransform", "preview_transform")
        sink = Gst.ElementFactory.make("nveglglessink", "preview_sink")
        if not all((pipeline, source, queue, transform, sink)):
            raise RuntimeError("required GStreamer preview elements could not be created")
        source.set_property("socket-path", args.socket)
        source.set_property("connection-attempts", -1)
        source.set_property("connection-interval", 100_000)
        source.set_property("buffer-timestamp-copy", True)
        for name, value in (("leaky", 2), ("max-size-buffers", 2),
                            ("max-size-bytes", 0), ("max-size-time", 0)):
            queue.set_property(name, value)
        for name, value in (("sync", False), ("qos", False), ("create-window", False),
                            ("force-aspect-ratio", True), ("winsys", "x11")):
            sink.set_property(name, value)
        def set_window(target):
            GstVideo.VideoOverlay.set_window_handle(target, args.window_id)
        set_window(sink)
        for element in (source, queue, transform, sink):
            pipeline.add(element)
        if not source.link(queue) or not queue.link(transform) or not transform.link(sink):
            raise RuntimeError("GStreamer preview elements could not be linked")
        bus = pipeline.get_bus()
        def sync_message(_bus, message, _data):
            if message.type == Gst.MessageType.ELEMENT:
                structure = message.get_structure()
                if structure and structure.get_name() == "prepare-window-handle":
                    set_window(message.src)
                    return Gst.BusSyncReply.DROP
            return Gst.BusSyncReply.PASS
        bus.set_sync_handler(sync_message, None)
        bus.add_signal_watch()
        ready = False
        failed = False
        def on_message(_bus, message):
            nonlocal ready, failed
            if message.type == Gst.MessageType.ERROR:
                error, debug = message.parse_error()
                failed = True
                _status("error", detail=f"{error} ({debug or ''})"[:4096])
                loop.quit()
            elif message.type == Gst.MessageType.EOS:
                _status("ended")
                loop.quit()
            elif message.type == Gst.MessageType.STATE_CHANGED and message.src == pipeline:
                _old, new, _pending = message.parse_state_changed()
                if new == Gst.State.PLAYING and not ready:
                    ready = True
                    _status("ready")
        bus.connect("message", on_message)
        def stop():
            loop.quit()
            return False
        for signum in (signal.SIGTERM, signal.SIGINT):
            GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signum, stop)
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("GStreamer preview failed to enter PLAYING")
        loop.run()
        return 1 if failed else 0
    except Exception as exc:
        _status("error", detail=f"{type(exc).__name__}: {exc}"[:4096])
        return 1
    finally:
        if pipeline is not None and Gst is not None:
            pipeline.set_state(Gst.State.NULL)
            pipeline.get_state(Gst.SECOND)


if __name__ == "__main__":
    raise SystemExit(main())
