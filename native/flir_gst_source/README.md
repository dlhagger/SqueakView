# FLIR Spinnaker GStreamer Source

Validated GStreamer source element for FLIR/Teledyne cameras through the
Spinnaker C++ SDK.

The element implements SqueakView's current FLIR contract:

```text
flirspinsrc ! video/x-raw,format=GRAY8,width=1440,height=1080,framerate=30/1
```

SqueakView uses this element as its direct FLIR source in the PyServiceMaker
pipeline.

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
  flirspinsrc camera-index=0 width=1440 height=1080 fps=30 pixel-format=Mono8 exposure-us=10000 trigger=false stream-buffer-count=64 ! \
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
    nvvideoconvert compute-hw=2 copy-hw=2 ! \
    'video/x-raw(memory:NVMM),format=NV12,width=640,height=480' ! \
    queue ! m.sink_0
```

The production SqueakView/DeepStream shape is:

```text
flirspinsrc camera-index=0 width=1440 height=1080 fps=30 pixel-format=Mono8 !
  video/x-raw,format=GRAY8,width=1440,height=1080,framerate=30/1 !
  nvvideoconvert compute-hw=2 copy-hw=2 !
  video/x-raw(memory:NVMM),format=NV12,width=1440,height=1080 !
  m.sink_0
```

## Current Scope

- Outputs `GRAY8` frames for DeepStream conversion to NVMM/NV12.
- Uses Spinnaker C++ `GetNextImage()`.
- Supports stable camera serial selection in addition to camera index, width,
  height, fps, pixel format, trigger,
  trigger activation, exposure, gain, timeout, stream buffer handling, and an
  explicit manual Spinnaker host transport buffer count.
- Defaults to `buffer-handling=OldestFirst` and `drop-incomplete=false` so
  scientific acquisition fails/logs instead of silently skipping frames.
- Defaults to `metadata-profile=scientific`, enabling supported FLIR chunks for
  frame ID, timestamp, exposure, gain, black level, ROI, pixel format,
  sequencer state, and CRC.
- Sets `GstBuffer.offset` from FLIR chunk `FrameID`. If the chunk ID is
  unavailable, offset is `GST_BUFFER_OFFSET_NONE`; a sequential value is never
  substituted.
- Treats `Image::GetFrameID()` as an acquisition-local stream frame ID and
  records it separately from the device chunk frame ID.
- Sets `GstBuffer.pts` from the absolute camera timestamp normalized to the
  first frame, with explicit host-monotonic fallback provenance.
- Attaches `SQUEAKVIEW.FLIR.FRAME_META.v1` before `nvstreammux`. DeepStream
  transforms it to frame-level `NvDsUserMeta` for inference admission auditing.
- Writes the same payload to `capture-log-path` as line-delimited JSON before
  returning each source buffer. This temporary recovery ledger is the durable
  pre-tee audit during capture. SqueakView reconciles it with recording-branch
  admissions to build authoritative `frames.csv`, then removes it only after
  successful validation; failed finalization retains it for diagnosis.
- Captures immediate host receipt clocks, source image layout, actual exposure
  and gain, payload sizes, image status, CRC validation, frame gaps, and source
  health counters.
- Latches the device timestamp at acquisition start against bracketing host
  monotonic and Unix clocks, preserving the raw device tick value and its
  reported nanoseconds-per-tick increment.
- Samples camera temperature and transport-layer health counters about once
  per second; SqueakView writes these samples to `diagnostics/camera.csv` using
  the frame's acquisition-side host Unix and monotonic clocks. Missing source
  clocks remain empty rather than being replaced with CSV-write time.
- Uses Spinnaker's typed timeout error and supports a bounded consecutive
  timeout policy. Triggered acquisitions can set the bound to zero.
- Implements `GstBaseSrc` `unlock`/`unlock_stop` so a triggered source blocked
  in `GetNextImage()` returns `GST_FLOW_FLUSHING` during shutdown without
  requiring an additional hardware trigger.
- Converts non-`Mono8` Spinnaker output to `Mono8` before pushing GStreamer
  buffers.
- Copies image memory into a `GstBuffer`; zero-copy/NVMM is a later phase.

## Integration Notes

- `SqueakView` uses this element as the primary FLIR capture backend.
- Preserve raw-video recording in an upstream tee outside this source element;
  the source itself only publishes frames and timing metadata.
- Use `camera-serial` for stable multi-camera rigs; an empty value retains
  index-based selection for compatibility.
- Width, height, exposure, and gain requests are strict: unsupported values
  fail camera startup with the camera's supported range instead of being
  silently clamped.
- `record_admission.csv` is a temporary recovery ledger written at the
  non-leaky recording queue. It is removed after successful finalization.
- `frames.csv` is finalized by matching recording admissions to the source-side
  metadata ledger. Its `inference_admitted` column records whether each frame
  also entered the downstream-leaky inference branch.
- `diagnostics/errors.csv` records camera-ID gaps, source-to-mux gaps, CRC failures,
  and missing/invalid metadata; a header-only file means none were observed.

The direct source and recording path completed a validated 16-hour,
1,707,205-frame single-camera run at 1440×1080 and 30 FPS with no source gaps,
transport loss, recording drops, or buffer evictions.
