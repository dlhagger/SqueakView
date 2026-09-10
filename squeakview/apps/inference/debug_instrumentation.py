"""Bounded, qualification-only DeepStream Service Maker instrumentation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from squeakview import config as squeakview_config
from squeakview.common.diagnostics.evidence_identity import stable_file_identity


DEBUG_PROFILE_ENV = "SQUEAKVIEW_DEEPSTREAM_DEBUG_PROFILE"
MAX_PROBE_MODULE_BYTES = 16 * 1024 * 1024
FPS_INTERVAL_SECONDS = 5
_PROBE_MODULES = {
    "measure_latency_probe": "libmeasure_latency_probe.so",
    "measure_fps_probe": "libmeasure_fps_probe.so",
}


def profile_enabled(environ: Mapping[str, str] | None = None) -> bool:
    source = os.environ if environ is None else environ
    return str(source.get(DEBUG_PROFILE_ENV, "0")).lower() in {
        "1", "true", "yes", "on",
    }


def probe_module_identities(
    *, sdk_root: Path | None = None
) -> dict[str, dict[str, object]]:
    """Return stable, bounded identities for NVIDIA's shipped probe modules."""

    root = Path(sdk_root or squeakview_config.DEEPSTREAM_SDK_ROOT)
    module_dir = root / "service-maker" / "modules"
    return {
        name: stable_file_identity(
            module_dir / filename, max_bytes=MAX_PROBE_MODULE_BYTES
        )
        for name, filename in _PROBE_MODULES.items()
    }


def validate_probe_modules(
    *, sdk_root: Path | None = None
) -> dict[str, dict[str, object]]:
    identities = probe_module_identities(sdk_root=sdk_root)
    invalid = [
        f"{name}: {identity.get('error', 'identity unavailable')}"
        for name, identity in identities.items()
        if identity.get("available") is not True
    ]
    if invalid:
        raise RuntimeError(
            "DeepStream debug probe modules are unavailable: " + "; ".join(invalid)
        )
    return identities


def attach_debug_probes(pipeline, target: str) -> None:
    """Attach NVIDIA probes at one downstream inference-branch node."""

    if not isinstance(target, str) or not target:
        raise ValueError("debug instrumentation target must be non-empty")
    validate_probe_modules()
    pipeline.attach(target, "measure_latency_probe", "squeakview_latency")
    pipeline.attach(
        target,
        "measure_fps_probe",
        "squeakview_fps",
        properties={"interval": FPS_INTERVAL_SECONDS},
    )


__all__ = [
    "DEBUG_PROFILE_ENV",
    "FPS_INTERVAL_SECONDS",
    "MAX_PROBE_MODULE_BYTES",
    "attach_debug_probes",
    "probe_module_identities",
    "profile_enabled",
    "validate_probe_modules",
]
