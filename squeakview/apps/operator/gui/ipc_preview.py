from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6 import QtCore


class IpcPreviewController(QtCore.QObject):
    """Render a DeepStream DMA-BUF IPC stream into an existing Qt window."""

    ready = QtCore.Signal()
    failed = QtCore.Signal(str)
    ended = QtCore.Signal()

    def __init__(self, emit: Callable[[str], None], parent=None) -> None:
        super().__init__(parent)
        self._emit = emit
        self._pipeline = None
        self._bus = None
        self._sink = None
        self._window_id = 0
        self._ready = False
        self._Gst = None
        self._GstVideo = None
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(100)
        self._poll_timer.timeout.connect(self._poll_bus)

    @property
    def running(self) -> bool:
        return self._pipeline is not None

    def start(self, socket_path: Path, window_id: int) -> bool:
        self.stop()
        self._window_id = int(window_id)
        if self._window_id <= 0:
            self._fail("Qt preview window is not available")
            return False

        try:
            import gi

            gi.require_version("Gst", "1.0")
            gi.require_version("GstVideo", "1.0")
            from gi.repository import Gst, GstVideo

            Gst.init(None)
            self._Gst = Gst
            self._GstVideo = GstVideo
            pipeline = Gst.Pipeline.new("squeakview-gui-preview")
            source = Gst.ElementFactory.make("nvunixfdsrc", "preview_ipc_source")
            queue = Gst.ElementFactory.make("queue", "preview_queue")
            transform = Gst.ElementFactory.make("nvegltransform", "preview_transform")
            sink = Gst.ElementFactory.make("nveglglessink", "preview_sink")
            if not all((pipeline, source, queue, transform, sink)):
                raise RuntimeError("required GStreamer preview elements could not be created")

            source.set_property("socket-path", str(socket_path))
            source.set_property("connection-attempts", -1)
            source.set_property("connection-interval", 100_000)
            source.set_property("buffer-timestamp-copy", True)
            queue.set_property("leaky", 2)
            queue.set_property("max-size-buffers", 2)
            queue.set_property("max-size-bytes", 0)
            queue.set_property("max-size-time", 0)
            sink.set_property("sync", False)
            sink.set_property("qos", False)
            sink.set_property("create-window", False)
            sink.set_property("force-aspect-ratio", True)
            sink.set_property("winsys", "x11")

            for element in (source, queue, transform, sink):
                pipeline.add(element)
            if not source.link(queue) or not queue.link(transform) or not transform.link(sink):
                raise RuntimeError("GStreamer preview elements could not be linked")

            self._pipeline = pipeline
            self._sink = sink
            self._set_window_handle(sink)
            self._bus = pipeline.get_bus()
            self._bus.set_sync_handler(self._on_sync_message, None)
            self._poll_timer.start()
            result = pipeline.set_state(Gst.State.PLAYING)
            if result == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("GStreamer preview pipeline failed to enter PLAYING")
        except Exception as exc:
            self.stop()
            self._fail(str(exc))
            return False

        self._emit(f"[PREVIEW] waiting for IPC frames on {socket_path}")
        return True

    def stop(self) -> None:
        self._poll_timer.stop()
        pipeline = self._pipeline
        bus = self._bus
        self._pipeline = None
        self._bus = None
        self._sink = None
        self._ready = False
        if bus is not None:
            try:
                bus.set_sync_handler(None, None)
            except Exception:
                pass
        if pipeline is not None and self._Gst is not None:
            try:
                pipeline.set_state(self._Gst.State.NULL)
                pipeline.get_state(self._Gst.SECOND)
            except Exception:
                pass

    def _set_window_handle(self, sink) -> None:
        if sink is None or self._GstVideo is None or self._window_id <= 0:
            return
        if hasattr(sink, "set_window_handle"):
            sink.set_window_handle(self._window_id)
        else:
            self._GstVideo.VideoOverlay.set_window_handle(sink, self._window_id)

    def _on_sync_message(self, _bus, message, _data):
        Gst = self._Gst
        if Gst is None:
            return 0
        if message.type == Gst.MessageType.ELEMENT:
            structure = message.get_structure()
            if structure is not None and structure.get_name() == "prepare-window-handle":
                try:
                    self._set_window_handle(message.src)
                    return Gst.BusSyncReply.DROP
                except Exception:
                    pass
        return Gst.BusSyncReply.PASS

    def _poll_bus(self) -> None:
        Gst = self._Gst
        bus = self._bus
        pipeline = self._pipeline
        if Gst is None or bus is None or pipeline is None:
            return
        message_types = (
            Gst.MessageType.ERROR
            | Gst.MessageType.EOS
            | Gst.MessageType.STATE_CHANGED
        )
        while True:
            message = bus.timed_pop_filtered(0, message_types)
            if message is None:
                break
            if message.type == Gst.MessageType.ERROR:
                error, debug = message.parse_error()
                detail = str(error)
                if debug:
                    detail = f"{detail} ({debug})"
                self.stop()
                self._fail(detail)
                return
            if message.type == Gst.MessageType.EOS:
                self.stop()
                self.ended.emit()
                return
            if message.type == Gst.MessageType.STATE_CHANGED and message.src == pipeline:
                _old, new, _pending = message.parse_state_changed()
                if new == Gst.State.PLAYING and not self._ready:
                    self._ready = True
                    self._emit("[PREVIEW] embedded IPC preview is playing")
                    self.ready.emit()

    def _fail(self, detail: str) -> None:
        message = f"Preview unavailable: {detail}"
        self._emit(f"[PREVIEW] {message}")
        self.failed.emit(message)
