"""Location of the application-private NVIDIA TensorRT 10 build tools."""

import argparse
from pathlib import Path
import sys


TENSORRT_VERSION = "10.16.2.10"
TENSORRT_DEB_VERSION = TENSORRT_VERSION + "-1+cuda13.2"
TENSORRT_RUNTIME_PACKAGES = ("libnvinfer10", "libnvinfer-plugin10", "libnvonnxparsers10")
# NVIDIA Jetson r39.2 APT index, ARM64; keep these hashes with the version pin.
TENSORRT_TOOL_PACKAGES = {
    "python3-libnvinfer": "da1282da12166871f731bb3f058f81b76c4ef22036668111723b0116ba2cf216",
    "libnvinfer-bin": "7a7de013265b50acb1839ec1e42c2e611a323446dd09142deced5b0cea052f56",
}


def bundled_tensorrt_root() -> Path:
    return Path(sys.prefix) / "vendor" / f"tensorrt-{TENSORRT_VERSION}"


def bundled_tensorrt_python() -> Path:
    return bundled_tensorrt_root() / "usr/lib/python3.12/dist-packages"


def trtexec_executable() -> str:
    bundled = bundled_tensorrt_root() / "usr/bin/trtexec"
    if not bundled.is_file():
        raise RuntimeError(
            f"Private TensorRT {TENSORRT_VERSION} tool is missing: {bundled}. "
            "Run python3 scripts/install_tensorrt10.py, then restart SqueakView."
        )
    return str(bundled)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-packages", action="store_true", required=True)
    parser.parse_args()
    print("\n".join(f"{name}={TENSORRT_DEB_VERSION}" for name in TENSORRT_RUNTIME_PACKAGES))
