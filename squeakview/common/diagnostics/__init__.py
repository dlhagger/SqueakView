"""Bounded, Qt-free diagnostics used by acquisition and qualification tools."""

from .system_telemetry import SYSTEM_TELEMETRY_HEADERS, SystemTelemetryRecorder
from .tegrastats import parse_tegrastats_line, read_platform_metrics

__all__ = [
    "SYSTEM_TELEMETRY_HEADERS",
    "SystemTelemetryRecorder",
    "parse_tegrastats_line",
    "read_platform_metrics",
]
