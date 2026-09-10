"""Pure presentation policy for a committed operator configuration."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from squeakview import model_package


@dataclass(frozen=True, slots=True)
class ConfigPresentation:
    """Text rendered after a configuration has been committed."""

    inference_enabled: bool
    session_text: str
    summary_html: str
    preview_info: str


def present_config(
    data: Mapping[str, Any],
    *,
    ds_cfg: Path | None,
    task_cfg: Path | None,
) -> ConfigPresentation:
    """Build deterministic, HTML-safe operator copy from resolved config data."""

    inference_enabled = bool(data.get("inference_enabled", True))
    serial_text = (
        f"{data['serial_port']} @ {data['serial_baud']}"
        if data.get("serial_enabled", True)
        else "disabled"
    )
    session_bits: list[str] = []
    if data.get("experiment_name"):
        session_bits.append(str(data["experiment_name"]))
    if data.get("mouse_id"):
        session_bits.append(f"Subject {data['mouse_id']}")
    session_text = " / ".join(session_bits) if session_bits else "No session profile"

    model_name = "Inference off"
    if ds_cfg is not None:
        try:
            model_name = model_package.validate_model_package(ds_cfg).name
        except model_package.ModelPackageError:
            model_name = f"Invalid: {ds_cfg.name}"

    rows = (
        (
            "Camera",
            f"{data['width']}×{data['height']} @ {data['fps']} FPS · "
            f"{data['pixel_format']} · {data.get('num_cameras', 1)} cam",
        ),
        (
            "Run",
            f"Trigger {'On' if data['trigger_on'] else 'Off'} · "
            f"Inference {'On' if inference_enabled else 'Off'}",
        ),
        ("Model", model_name),
        ("Task", task_cfg.name if task_cfg else "N/A"),
        ("Serial", serial_text),
        ("Session", session_text),
    )
    summary_html = (
        "<table cellspacing='0' cellpadding='2'>"
        + "".join(
            "<tr>"
            "<td style='color:#9aa7cc; font-weight:700; padding-right:10px;'>"
            f"{html.escape(label)}</td>"
            "<td style='color:#e8ecff;'>"
            f"{html.escape(str(value))}</td>"
            "</tr>"
            for label, value in rows
        )
        + "</table>"
    )
    preview_info = (
        f"{data['width']}×{data['height']} · {data['fps']} FPS · "
        f"{'Trig' if data['trigger_on'] else 'Free'} · {data['bitrate']} kbps"
    )
    return ConfigPresentation(
        inference_enabled=inference_enabled,
        session_text=session_text,
        summary_html=summary_html,
        preview_info=preview_info,
    )
