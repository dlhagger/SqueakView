from __future__ import annotations

"""Capture pipeline wrapping PySpin → GStreamer shared memory."""

import os
import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass

import PySpin
import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst

DEFAULT_SOCKET = "/tmp/cam.sock"


def log(message: str) -> None:
    print(message, flush=True)


def warn(message: str) -> None:
    print(f"[WARN] {message}", flush=True)


def set_trigger(cam, on: bool, line_index: int = 0, activation: str = "RisingEdge") -> None:
    nodemap = cam.GetNodeMap()

    def enum(name: str):
        e = PySpin.CEnumerationPtr(nodemap.GetNode(name))
        if not PySpin.IsAvailable(e) or not PySpin.IsWritable(e):
            raise RuntimeError(f"Node not available/writable: {name}")
        return e

    def enum_set(e, entry_name: str) -> None:
        ent = e.GetEntryByName(entry_name)
        if not PySpin.IsAvailable(ent) or not PySpin.IsReadable(ent):
            raise RuntimeError(f"Entry not readable: {entry_name}")
        e.SetIntValue(ent.GetValue())

    try:
        cam.EndAcquisition()
    except Exception:
        pass

    enum_set(enum("TriggerMode"), "Off")

    if on:
        enum_set(enum("TriggerSelector"), "FrameStart")

        line_sel = PySpin.CEnumerationPtr(nodemap.GetNode("LineSelector"))
        if PySpin.IsAvailable(line_sel) and PySpin.IsWritable(line_sel):
            try:
                ent = line_sel.GetEntryByName(f"Line{line_index}")
                line_sel.SetIntValue(ent.GetValue())
            except Exception:
                pass
            line_mode = PySpin.CEnumerationPtr(nodemap.GetNode("LineMode"))
            if PySpin.IsAvailable(line_mode) and PySpin.IsWritable(line_mode):
                try:
                    ent = line_mode.GetEntryByName("Input")
                    line_mode.SetIntValue(ent.GetValue())
                except Exception:
                    pass

        enum_set(enum("TriggerSource"), f"Line{line_index}")
        enum_set(enum("TriggerActivation"), "RisingEdge" if activation.lower().startswith("r") else "FallingEdge")

        exp_auto = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
        if PySpin.IsAvailable(exp_auto) and PySpin.IsWritable(exp_auto):
            try:
                ent = exp_auto.GetEntryByName("Off")
                exp_auto.SetIntValue(ent.GetValue())
            except Exception:
                pass

        overlap = PySpin.CEnumerationPtr(nodemap.GetNode("TriggerOverlap"))
        if PySpin.IsAvailable(overlap) and PySpin.IsWritable(overlap):
            try:
                ent = overlap.GetEntryByName("ReadOut")
                overlap.SetIntValue(ent.GetValue())
            except Exception:
                pass

        enum_set(enum("TriggerMode"), "On")
    else:
        enum_set(enum("TriggerMode"), "Off")

    time.sleep(0.05)


def pyspin_setup(
    width: int,
    height: int,
    fps: int,
    pix: str = "Mono8",
    trigger_on: bool = False,
    activation: str = "rising",
    exposure_us: float | None = 10000.0,
    gain: float | None = None,
):
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    if cam_list.GetSize() == 0:
        system.ReleaseInstance()
        raise RuntimeError("No cameras found")

    cam = cam_list.GetByIndex(0)
    cam.Init()
    nm = cam.GetNodeMap()

    pix_enum = PySpin.CEnumerationPtr(nm.GetNode("PixelFormat"))
    if PySpin.IsAvailable(pix_enum) and PySpin.IsWritable(pix_enum):
        try:
            ent = pix_enum.GetEntryByName(pix)
            pix_enum.SetIntValue(ent.GetValue())
        except Exception as exc:
            cam.DeInit()
            cam_list.Clear()
            system.ReleaseInstance()
            raise RuntimeError(f"Cannot set PixelFormat={pix}: {exc}")

    width_node = PySpin.CIntegerPtr(nm.GetNode("Width"))
    height_node = PySpin.CIntegerPtr(nm.GetNode("Height"))
    if PySpin.IsAvailable(width_node) and PySpin.IsWritable(width_node):
        width_node.SetValue(min(width, width_node.GetMax()))
    if PySpin.IsAvailable(height_node) and PySpin.IsWritable(height_node):
        height_node.SetValue(min(height, height_node.GetMax()))

    enable_node = PySpin.CBooleanPtr(nm.GetNode("AcquisitionFrameRateEnable"))
    fps_node = PySpin.CFloatPtr(nm.GetNode("AcquisitionFrameRate"))
    try:
        if PySpin.IsAvailable(enable_node) and PySpin.IsWritable(enable_node):
            enable_node.SetValue(True)
        if PySpin.IsAvailable(fps_node) and PySpin.IsWritable(fps_node):
            fps_node.SetValue(float(fps))
    except Exception:
        pass

    exposure_desc = "Camera default"

    if exposure_us is not None:
        exp_auto = PySpin.CEnumerationPtr(nm.GetNode("ExposureAuto"))
        if PySpin.IsAvailable(exp_auto) and PySpin.IsWritable(exp_auto):
            try:
                ent = exp_auto.GetEntryByName("Off")
                exp_auto.SetIntValue(ent.GetValue())
            except Exception:
                pass
        exp_node = PySpin.CFloatPtr(nm.GetNode("ExposureTime"))
        if PySpin.IsAvailable(exp_node) and PySpin.IsWritable(exp_node):
            try:
                exp_node.SetValue(float(exposure_us))
                exposure_desc = f"Manual {float(exposure_us):.0f}us"
            except Exception:
                pass

    if gain is not None:
        gain_auto = PySpin.CEnumerationPtr(nm.GetNode("GainAuto"))
        if PySpin.IsAvailable(gain_auto) and PySpin.IsWritable(gain_auto):
            try:
                ent = gain_auto.GetEntryByName("Off")
                gain_auto.SetIntValue(ent.GetValue())
            except Exception:
                pass
        gain_node = PySpin.CFloatPtr(nm.GetNode("Gain"))
        if PySpin.IsAvailable(gain_node) and PySpin.IsWritable(gain_node):
            try:
                gain_node.SetValue(float(gain))
            except Exception:
                pass

    set_trigger(
        cam,
        trigger_on,
        line_index=0,
        activation="RisingEdge" if activation.lower().startswith("r") else "FallingEdge",
    )

    log(
        f"[PySpin] Using PixelFormat={pix} @ {width}x{height} ~{fps} FPS | Trigger={'ON' if trigger_on else 'OFF'} | {exposure_desc}"
    )
    return system, cam_list, cam


def make_pipeline(width: int, height: int, fps: int, socket_path: str) -> Gst.Pipeline:
    Gst.init(None)
    desc = (
        "appsrc name=src is-live=true format=time do-timestamp=true caps="
        f"video/x-raw,format=GRAY8,width={width},height={height},framerate={fps}/1 ! "
        f"shmsink socket-path={socket_path} wait-for-connection=true sync=false shm-size=52428800"
    )
    return Gst.parse_launch(desc)


@contextmanager
def _signal_handler(handler):
    previous_int = signal.signal(signal.SIGINT, handler)
    previous_term = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)


@dataclass(slots=True)
class CaptureConfig:
    width: int = 1440
    height: int = 1080
    fps: int = 30
    pixel_format: str = "Mono8"
    trigger_on: bool = False
    trigger_activation: str = "rising"
    exposure_us: float | None = None
    gain: float | None = None
    trig_timeout_ms: int = 1000
    socket_path: str = DEFAULT_SOCKET


def run_capture(config: CaptureConfig) -> None:
    """Entry point invoked by the orchestrator."""

    try:
        os.unlink(config.socket_path)
    except FileNotFoundError:
        pass
    except Exception:
        pass

    system = cam_list = cam = None
    pipeline = None
    appsrc = None

    try:
        system, cam_list, cam = pyspin_setup(
            config.width,
            config.height,
            config.fps,
            pix=config.pixel_format,
            trigger_on=config.trigger_on,
            activation=config.trigger_activation,
            exposure_us=config.exposure_us,
            gain=config.gain,
        )

        pipeline = make_pipeline(config.width, config.height, config.fps, config.socket_path)
        appsrc = pipeline.get_by_name("src")
        pipeline.set_state(Gst.State.PLAYING)

        stop = False

        def on_sig(*_):
            nonlocal stop
            stop = True

        with _signal_handler(on_sig):
            cam.BeginAcquisition()
            t0 = time.time()

            while not stop:
                try:
                    image = cam.GetNextImage(config.trig_timeout_ms)
                    if image.IsIncomplete():
                        image.Release()
                        continue
                    frame = image.GetNDArray()
                    image.Release()
                except PySpin.SpinnakerException as exc:
                    if stop:
                        # Camera is shutting down; ignore spurious errors while exiting.
                        break
                    msg = str(exc) or repr(exc)
                    if "Timeout" in msg or "Timeout" in repr(exc):
                        continue
                    if "EventData" in msg or "-1011" in msg:
                        # Occurs when acquisition is being torn down; treat as benign.
                        continue
                    warn(f"GetNextImage: {exc}")
                    time.sleep(0.01)
                    continue

                try:
                    buf = Gst.Buffer.new_allocate(None, frame.nbytes, None)
                    buf.fill(0, frame.tobytes())
                    now = time.time()
                    pts_ns = int((now - t0) * 1e9)
                    buf.pts = pts_ns
                    buf.dts = pts_ns
                    buf.duration = 0
                    appsrc.emit("push-buffer", buf)
                except Exception as exc:
                    warn(f"GStreamer push-buffer: {exc}")
                    time.sleep(0.01)

    finally:
        try:
            if appsrc:
                appsrc.emit("end-of-stream")
        except Exception:
            pass

        try:
            if cam:
                cam.EndAcquisition()
        except Exception:
            pass

        try:
            if cam:
                cam.DeInit()
        except Exception:
            pass

        try:
            if cam_list:
                cam_list.Clear()
        except Exception:
            pass
        cam = None
        cam_list = None

        try:
            if system:
                system.ReleaseInstance()
        except Exception:
            pass
        system = None

        try:
            if pipeline:
                pipeline.set_state(Gst.State.NULL)
        except Exception:
            pass

        log("[INFO] Capture shutting down…")
