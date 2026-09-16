from __future__ import annotations

"""Read-only application and NVIDIA runtime paths.

Scientific data paths deliberately do not live here. They come from the
validated project selected by the durable supervisor.
"""

import os
from pathlib import Path

def _resolve_deepstream_sdk() -> Path:
    candidate = os.environ.get("SQUEAKVIEW_DEEPSTREAM_SDK")
    if candidate:
        return Path(candidate).expanduser().resolve()
    return Path("/opt/nvidia/deepstream/deepstream").resolve()


APP_ROOT = Path(__file__).resolve().parents[1]
DEEPSTREAM_SDK_ROOT = _resolve_deepstream_sdk()

NATIVE_ROOT = APP_ROOT / "native"
FLIR_GST_SOURCE_ROOT = NATIVE_ROOT / "flir_gst_source"
FLIR_GST_PLUGIN_DIR = FLIR_GST_SOURCE_ROOT / "build"
CUSTOM_YOLO_LIB = NATIVE_ROOT / "nvdsinfer_custom_impl_yolo" / "libnvdsinfer_custom_impl_Yolo.so"
