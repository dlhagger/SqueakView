#!/usr/bin/env python3
"""Extract pinned NVIDIA tools into .venv without changing JetPack packages."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from squeakview.common.tensorrt_runtime import (
    TENSORRT_VERSION as VERSION,
    TENSORRT_DEB_VERSION as DEB_VERSION,
    TENSORRT_RUNTIME_PACKAGES,
    TENSORRT_TOOL_PACKAGES as PACKAGES,
)


def main() -> None:
    if platform.machine() != "aarch64":
        raise SystemExit("This TensorRT bundle targets Jetson ARM64 only.")
    venv = ROOT / ".venv"
    python = venv / "bin/python"
    subprocess.run([str(python), "-c", "import sys; assert sys.version_info[:2] == (3, 12)"], check=True)
    sdk = Path(os.environ.get("SQUEAKVIEW_DEEPSTREAM_SDK", "/opt/nvidia/deepstream/deepstream"))
    dependencies = subprocess.check_output(
        ["readelf", "-d", str(sdk / "lib/libnvds_infer.so")], text=True,
    )
    if "Shared library: [libnvinfer.so.10]" not in dependencies:
        raise SystemExit(f"Selected SDK {sdk} does not use TensorRT 10; bundle was not installed.")
    for package in TENSORRT_RUNTIME_PACKAGES:
        version = subprocess.check_output(
            ["dpkg-query", "-W", "-f=${db:Status-Status} ${Version}", package], text=True,
        ).strip()
        if version != "installed " + DEB_VERSION:
            raise SystemExit(
                f"Required runtime: {package}={DEB_VERSION}; found {version!r}. "
                "Run scripts/setup_jetson.sh to provision the matching TensorRT 10 runtimes."
            )
    vendor = venv / "vendor"
    vendor.mkdir(exist_ok=True)
    destination = vendor / f"tensorrt-{VERSION}"
    if not destination.exists():
        with tempfile.TemporaryDirectory(prefix=".tensorrt-download-", dir=vendor) as temporary:
            work = Path(temporary)
            payload = work / "payload"
            payload.mkdir()
            for package, expected in PACKAGES.items():
                subprocess.run(["apt-get", "download", f"{package}={DEB_VERSION}"], cwd=work, check=True)
                deb = work / f"{package}_{DEB_VERSION}_arm64.deb"
                actual = hashlib.sha256(deb.read_bytes()).hexdigest()
                if actual != expected:
                    raise SystemExit(f"SHA256 mismatch for {deb.name}; bundle was not installed.")
                subprocess.run(["dpkg-deb", "-x", str(deb), str(payload)], check=True)
            (payload / "squeakview-bundle.json").write_text(
                json.dumps({"version": DEB_VERSION, "packages": PACKAGES}, indent=2) + "\n"
            )
            # Test the extracted bindings and CLI before publishing them.
            bindings = payload / "usr/lib/python3.12/dist-packages"
            subprocess.run([
                str(python), "-c",
                "import sys; sys.path.insert(0, sys.argv[1]); import tensorrt; "
                "assert tensorrt.__version__ == sys.argv[2], tensorrt.__version__",
                str(bindings), VERSION,
            ], check=True)
            subprocess.run([str(payload / "usr/bin/trtexec"), "--help"], check=True,
                           stdout=subprocess.DEVNULL)
            payload.rename(destination)
    expected_manifest = {"version": DEB_VERSION, "packages": PACKAGES}
    try:
        manifest = json.loads((destination / "squeakview-bundle.json").read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"Unreadable TensorRT bundle identity at {destination}; move this bundle "
            "directory aside and rerun this installer."
        ) from exc
    if manifest != expected_manifest:
        raise SystemExit(f"Unexpected bundle identity at {destination}; move this bundle directory aside and rerun this installer.")
    bindings = destination / "usr/lib/python3.12/dist-packages"
    required = (destination / "usr/bin/trtexec", bindings / "tensorrt/__init__.py", bindings / "tensorrt/tensorrt.so")
    if not all(path.is_file() for path in required):
        raise SystemExit(f"Incomplete TensorRT bundle: {destination}; move this bundle directory aside and rerun this installer.")
    site = venv / "lib/python3.12/site-packages"
    pth = site / "squeakview_tensorrt10.pth"
    previous = pth.read_bytes() if pth.exists() else None
    try:
        pth.write_text(str(bindings) + "\n")
        subprocess.run([
            str(python), "-c",
            "import sys, tensorrt; from pathlib import Path; "
            "assert tensorrt.__version__ == sys.argv[1], tensorrt.__version__; "
            "assert Path(tensorrt.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve()), tensorrt.__file__; "
            "print('SqueakView TensorRT:', tensorrt.__version__, tensorrt.__file__)",
            VERSION, str(bindings),
        ], check=True)
    except BaseException:
        if previous is None:
            pth.unlink(missing_ok=True)
        else:
            pth.write_bytes(previous)
        raise
    print(f"[PASS] Private TensorRT {VERSION} tools installed; system TensorRT was not modified.")


if __name__ == "__main__":
    main()
