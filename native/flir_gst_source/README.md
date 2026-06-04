# FLIR Spinnaker GStreamer Source

Experimental GStreamer source element for FLIR/Teledyne cameras through the
Spinnaker C++ SDK.

The first target is a source element that matches SqueakView's current FLIR
contract:

```text
flirspinsrc ! video/x-raw,format=GRAY8,width=1440,height=1080,framerate=30/1
```

It is intentionally not wired into the SqueakView GUI yet. The current Python
capture path remains unchanged.

## Build

```bash
cmake -S native/flir_gst_source -B native/flir_gst_source/build
cmake --build native/flir_gst_source/build -j
```

## Inspect

```bash
GST_PLUGIN_PATH=$PWD/native/flir_gst_source/build gst-inspect-1.0 flirspinsrc
```

## Basic Pipeline

With a camera connected:

```bash
GST_PLUGIN_PATH=$PWD/native/flir_gst_source/build \
gst-launch-1.0 -e \
  flirspinsrc camera-index=0 width=1440 height=1080 fps=30 pixel-format=Mono8 exposure-us=10000 trigger=false ! \
  video/x-raw,format=GRAY8,width=1440,height=1080,framerate=30/1 ! \
  queue ! fakesink sync=false
```

## DeepStream-Shaped Smoke Test

This was tested on the Jetson with a `FLIR Blackfly S BFS-U3-16S2M` visible to
Spinnaker:

```bash
GST_PLUGIN_PATH=$PWD/native/flir_gst_source/build \
gst-launch-1.0 -e \
  nvstreammux name=m batch-size=1 width=640 height=480 live-source=1 batched-push-timeout=33333 ! \
    fakesink sync=false \
  flirspinsrc num-buffers=5 camera-index=0 width=640 height=480 fps=30 pixel-format=Mono8 trigger=false exposure-us=10000 ! \
    video/x-raw,format=GRAY8,width=640,height=480,framerate=30/1 ! \
    nvvideoconvert compute-hw=1 copy-hw=2 ! \
    'video/x-raw(memory:NVMM),format=NV12,width=640,height=480' ! \
    queue ! m.sink_0
```

The eventual SqueakView/DeepStream shape should be:

```text
flirspinsrc camera-index=0 width=1440 height=1080 fps=30 pixel-format=Mono8 !
  video/x-raw,format=GRAY8,width=1440,height=1080,framerate=30/1 !
  nvvideoconvert compute-hw=1 copy-hw=2 !
  video/x-raw(memory:NVMM),format=NV12,width=1440,height=1080 !
  m.sink_0
```

## Current Scope

- Outputs `GRAY8` frames for DeepStream conversion to NVMM/NV12.
- Uses Spinnaker C++ `GetNextImage()`.
- Supports camera index, width, height, fps, pixel format, trigger,
  trigger activation, exposure, gain, timeout, and stream buffer handling.
- Defaults to `buffer-handling=OldestFirst` and `drop-incomplete=false` so
  scientific acquisition fails/logs instead of silently skipping frames.
- Sets `GstBuffer.offset` from Spinnaker `Image::GetFrameID()` when available
  and `GstBuffer.pts` from the camera timestamp normalized to the first frame.
- Converts non-`Mono8` Spinnaker output to `Mono8` before pushing GStreamer
  buffers.
- Copies image memory into a `GstBuffer`; zero-copy/NVMM is a later phase.

## Integration Notes

- `SqueakView` uses this element as the primary FLIR capture backend.
- Preserve raw-video recording in an upstream tee outside this source element;
  the source itself only publishes frames and timing metadata.
- Add serial-number camera selection before relying on this for stable
  multi-camera rigs; `camera-index` is enough for the first single-camera test.
- Capture frame manifests and frame-gap logs live in the Python runner.
