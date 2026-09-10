"""Headless and GUI-preview output graph construction."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .contracts import InferenceConfig
from .preview_attribution import PreviewBoundaryOperator, boundary_path


def add_output_branch(
    pipeline,
    config: InferenceConfig,
    tail: list[str],
    *,
    run_dir: Path,
    meta_type: int,
    probe_factory: Callable[[str, object], object],
    register_boundary: Callable[[PreviewBoundaryOperator], None],
) -> str:
    """Add headless or GUI-preview output and return its readiness origin."""

    if not config.preview_sockets:
        pipeline.add("fakesink", "sink", {"sync": False})
        tail.append("sink")
        pipeline.link(*tail)
        return "sink"

    pipeline.add("nvosdbin", "osd")
    tail.append("osd")
    pipeline.add("nvstreamdemux", "preview_demux")
    tail.append("preview_demux")
    pipeline.link(*tail)
    for index, socket_path in enumerate(config.preview_sockets):
        queue_name = f"preview_queue{index}"
        admission_name = f"preview_admission{index}"
        sink_name = f"preview_sink{index}"
        pipeline.add("identity", admission_name)
        admission = PreviewBoundaryOperator(
            boundary_path(run_dir, index, "admission"),
            index,
            "admission",
            meta_type,
        )
        register_boundary(admission)
        pipeline.attach(
            admission_name,
            probe_factory(f"preview_admission_probe{index}", admission),
        )
        pipeline.add(
            "queue",
            queue_name,
            {
                "leaky": 2,
                "max-size-buffers": 1,
                "max-size-bytes": 0,
                "max-size-time": 0,
            },
        )
        delivery = PreviewBoundaryOperator(
            boundary_path(run_dir, index, "delivery"),
            index,
            "delivery",
            meta_type,
        )
        register_boundary(delivery)
        pipeline.attach(
            queue_name,
            probe_factory(f"preview_delivery_probe{index}", delivery),
        )
        pipeline.add(
            "nvunixfdsink",
            sink_name,
            {
                "socket-path": str(socket_path),
                "sync": False,
                "async": False,
                "buffer-timestamp-copy": True,
                "qos": False,
            },
        )
        pipeline.link(("preview_demux", admission_name), (f"src_{index}", ""))
        pipeline.link(admission_name, queue_name, sink_name)
    return "preview_sink0"


__all__ = ["add_output_branch"]
