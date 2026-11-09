"""Validate recorded SDK/package identity without consulting the current machine."""
from __future__ import annotations

import re
from collections.abc import Mapping


def deepstream_package_errors(platform: Mapping) -> list[str]:
    build = platform.get("deepstream_build")
    match = re.search(r"^(?:Version:\s*|DeepStreamSDK\s+)?((9\.[12])(?:\.\d+)?)(?=$|\s|-)",
                      build, re.MULTILINE) if isinstance(build, str) else None
    if match is None:
        return ["runtime_identity.deepstream_build: expected DeepStream 9.1 or 9.2"]
    selected, version = match.groups()
    name = f"deepstream-{version}"
    packages = platform.get("packages")
    package = packages.get(name) if isinstance(packages, Mapping) else None
    if not isinstance(package, str) or not package.strip():
        return [f"runtime_identity.packages missing: {name}"]
    package_match = re.match(r"^(?:\d+:)?(\d+\.\d+(?:\.\d+)?)(?:-|$)", package)
    actual = package_match.group(1) if package_match is not None else ""
    matches = actual == selected or (selected.count('.') == 1 and actual.startswith(selected + '.'))
    if not matches:
        return [f"runtime_identity.packages: {name} version does not match selected SDK"]
    return []
