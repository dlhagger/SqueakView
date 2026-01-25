#!/usr/bin/env python3
import gi, sys, os
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

def main(cfg_path, width=640, height=640, num_buffers=30, batch_size=1):
    Gst.init(None)

    # GStreamer pipeline:
    # videotestsrc → raw NV12 → nvvidconv → NVMM NV12 → nvstreammux (batch=1)
    # → nvinfer (loads your config; will BUILD the engine from ONNX if missing)
    # → fakesink
    pipeline_desc = f"""
videotestsrc num-buffers={num_buffers} !
video/x-raw,format=NV12,width={width},height={height},framerate=30/1 !
nvvidconv !
video/x-raw(memory:NVMM), format=NV12, width={width}, height={height}, framerate=30/1 !
nvstreammux name=mux width={width} height={height} batch-size={batch_size} !
nvinfer config-file-path={cfg_path} batch-size={batch_size} unique-id=1 !
fakesink sync=false
"""

    # Link test source into mux.sink_0 (by name) then run through nvinfer
    pipeline_desc = pipeline_desc.replace("nvstreammux name=mux", f"mux.sink_0 nvstreammux name=mux")

    print("[build_engine] Using config:", cfg_path)
    print("[build_engine] Launching pipeline…")
    pipeline = Gst.parse_launch(pipeline_desc)

    bus = pipeline.get_bus()
    pipeline.set_state(Gst.State.PLAYING)

    try:
        while True:
            msg = bus.timed_pop_filtered(
                5 * Gst.SECOND,
                Gst.MessageType.ERROR | Gst.MessageType.EOS
            )
            if msg:
                if msg.type == Gst.MessageType.ERROR:
                    err, dbg = msg.parse_error()
                    print("[ERROR]", err, dbg)
                    break
                if msg.type == Gst.MessageType.EOS:
                    print("[build_engine] Done (EOS).")
                    break
    finally:
        pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: build_engine.py /path/to/config_infer_primary_*.txt [width height]")
        sys.exit(1)
    cfg = sys.argv[1]
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 640
    h = int(sys.argv[3]) if len(sys.argv) > 3 else 640
    main(cfg, w, h)

