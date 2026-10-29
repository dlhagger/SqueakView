"""Read-only dependency check with one content-verified NVIDIA metadata exception."""
from __future__ import annotations

import base64
import importlib.metadata as metadata
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

from .diagnostics.evidence_identity import stable_file_identity

# NVIDIA's 0.8.1 ARM64 wheel is named manylinux2014_aarch64, but records an
# internal sbsa tag. These are the exact upstream payload hashes, not values
# trusted from an installed RECORD. No wheel, metadata, lock or binary is edited.
VENDOR_WHEEL_SHA256 = '4dca476c50bf4780d46cd0bfbd82e2bc10a08e4fef7950917ce8d7578d22a23f'
_VENDOR_FILES = {
    'nvidia/cusparselt/lib/libcusparseLt.so.0': '-7A7v40iOGcPJCVsVFIoHCpmAWKJm4OLtNDMBtrQJ98',
    'nvidia/cusparselt/include/cusparseLt.h': 'DfhLWAaIdx14vmK87rxAiGehYcZB_gvKilB1ugQ7Y7M',
    'nvidia_cusparselt_cu13-0.8.1.dist-info/METADATA': 'NTZpuvuAy_IvEyQSGsBQSdNqaw-0Sx_tP6sAtRMjEXE',
    'nvidia_cusparselt_cu13-0.8.1.dist-info/WHEEL': '9EEww1DEpSYdq-rReTwY88G51UoHWvABWuRF6Sy_8Bs',
}


def reviewed_cusparselt_metadata() -> bool:
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        return False
    try:
        dist = metadata.distribution('nvidia-cusparselt-cu13')
        if dist.version != '0.8.1':
            return False
        requirements = metadata.requires('torch') or []
        if not any(re.match(r'nvidia-cusparselt-cu13\s*==\s*0\.8\.1(?:\s*;|$)', item) for item in requirements):
            return False
        root = Path(dist.locate_file('')).resolve(strict=True)
        for relative, expected in _VENDOR_FILES.items():
            path = root / relative
            if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
                return False
            identity = stable_file_identity(path, max_bytes=512 * 1024 * 1024)
            digest = base64.urlsafe_b64decode(expected + '=').hex()
            if identity.get('available') is not True or identity.get('sha256') != digest:
                return False
        return True
    except (OSError, ValueError, metadata.PackageNotFoundError):
        return False


def check_dependencies(*, allow_reviewed_vendor_metadata=False) -> tuple[bool, str]:
    uv = shutil.which('uv')
    if uv is None:
        candidate = Path.home() / '.local/bin/uv'
        uv = str(candidate) if candidate.is_file() else None
    if uv is None:
        return False, 'uv is required to check the project environment.'
    try:
        result = subprocess.run([uv, 'pip', 'check', '--python', sys.executable],
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f'Dependency check failed: {exc}'
    output = (result.stdout + result.stderr).strip()
    if result.returncode == 0:
        return True, output
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    expected = 'The package `nvidia-cusparselt-cu13` was built for a different platform'
    only_known = (result.returncode == 1 and lines[-2:] == ['Found 1 incompatibility', expected]
                  and all(re.fullmatch(r'Checked \d+ packages in .+', line) for line in lines[:-2]))
    if allow_reviewed_vendor_metadata and only_known and reviewed_cusparselt_metadata():
        return True, ('[WARN] Reviewed NVIDIA cuSPARSELt 0.8.1 ARM64 metadata mismatch; '
                      'exact library/header/metadata hashes verified. No files changed. '
                      'This is not GPU validation.\n' + output)
    return False, output
