"""Parse Jetson ``tegrastats`` output without Qt or NVIDIA Python bindings."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


_RAM_RE = re.compile(
    r"\bRAM\s+(?P<used>\d+)/(?P<total>\d+)MB"
    r"(?:\s+\(lfb\s+(?P<blocks>\d+)x(?P<block>\d+)MB\))?",
    re.IGNORECASE,
)
_SWAP_RE = re.compile(
    r"\bSWAP\s+(?P<used>\d+)/(?P<total>\d+)MB"
    r"(?:\s+\(cached\s+(?P<cached>\d+)MB\))?",
    re.IGNORECASE,
)
_CPU_RE = re.compile(r"\bCPU\s*\[(?P<cores>[^]]*)\]", re.IGNORECASE)
_CORE_RE = re.compile(r"(?P<util>\d+(?:\.\d+)?)%\s*@\s*(?P<freq>\d+(?:\.\d+)?)")
_GPU_RE = re.compile(
    r"\bGR3D(?:_FREQ|\s+FREQ)?\s+(?P<util>\d+(?:\.\d+)?)%"
    r"(?:\s*@\s*(?P<freq>\d+(?:\.\d+)?))?",
    re.IGNORECASE,
)
_EMC_RE = re.compile(
    r"\bEMC(?:_FREQ|\s+FREQ)?\s+(?P<util>\d+(?:\.\d+)?)%"
    r"(?:\s*@\s*(?P<freq>\d+(?:\.\d+)?))?",
    re.IGNORECASE,
)
_TEMP_RE = re.compile(
    r"\b(?P<name>cpu|gpu|tj|soc0|soc1|soc2)@"
    r"(?P<current>-?\d+(?:\.\d+)?)C(?:/-?\d+(?:\.\d+)?C)?",
    re.IGNORECASE,
)
_POWER_RE = re.compile(
    r"\b(?P<name>VDD_IN|VDD_CPU_GPU_CV|VDD_SOC)\s+"
    r"(?P<current>\d+)mW(?:/(?P<average>\d+)mW/(?P<peak>\d+)mW)?",
    re.IGNORECASE,
)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_tegrastats_line(raw_line: str) -> dict[str, object]:
    """Return normalized optional metrics from one complete tegrastats line.

    Missing fields remain ``None``. A recognizable but incomplete target-Jetson
    line is marked ``partial``; an empty or unrecognizable line is ``malformed``.
    The original line is retained so future parsers can recover vendor fields.
    """

    raw = str(raw_line).rstrip("\r\n")
    result: dict[str, object] = {
        "parse_status": "malformed",
        "ram_used_mb": None,
        "ram_total_mb": None,
        "ram_pct": None,
        "lfb_blocks": None,
        "lfb_block_mb": None,
        "swap_used_mb": None,
        "swap_total_mb": None,
        "swap_cached_mb": None,
        "cpu_online": None,
        "cpu_total": None,
        "cpu_util_mean_pct": None,
        "cpu_util_max_pct": None,
        "cpu_freq_mean_mhz": None,
        "cpu_freq_max_mhz": None,
        "gpu_util_pct": None,
        "gpu_freq_mhz": None,
        "emc_util_pct": None,
        "emc_freq_mhz": None,
        "temp_cpu_c": None,
        "temp_gpu_c": None,
        "temp_tj_c": None,
        "temp_soc0_c": None,
        "temp_soc1_c": None,
        "temp_soc2_c": None,
        "temp_max_c": None,
        "vdd_in_current_mw": None,
        "vdd_in_avg_mw": None,
        "vdd_in_peak_mw": None,
        "vdd_cpu_gpu_cv_current_mw": None,
        "vdd_cpu_gpu_cv_avg_mw": None,
        "vdd_cpu_gpu_cv_peak_mw": None,
        "vdd_soc_current_mw": None,
        "vdd_soc_avg_mw": None,
        "vdd_soc_peak_mw": None,
        "raw_line": raw,
    }
    recognized = 0

    ram = _RAM_RE.search(raw)
    if ram:
        recognized += 1
        used = int(ram.group("used"))
        total = int(ram.group("total"))
        result.update(
            ram_used_mb=used,
            ram_total_mb=total,
            ram_pct=(used / total * 100.0) if total > 0 else None,
            lfb_blocks=int(ram.group("blocks")) if ram.group("blocks") else None,
            lfb_block_mb=int(ram.group("block")) if ram.group("block") else None,
        )

    swap = _SWAP_RE.search(raw)
    if swap:
        recognized += 1
        result.update(
            swap_used_mb=int(swap.group("used")),
            swap_total_mb=int(swap.group("total")),
            swap_cached_mb=int(swap.group("cached")) if swap.group("cached") else None,
        )

    cpu = _CPU_RE.search(raw)
    if cpu:
        recognized += 1
        tokens = [token.strip() for token in cpu.group("cores").split(",")]
        samples = [_CORE_RE.search(token) for token in tokens]
        utilities = [float(match.group("util")) for match in samples if match]
        frequencies = [float(match.group("freq")) for match in samples if match]
        result.update(
            cpu_online=len(utilities),
            cpu_total=len(tokens),
            cpu_util_mean_pct=_mean(utilities),
            cpu_util_max_pct=max(utilities) if utilities else None,
            cpu_freq_mean_mhz=_mean(frequencies),
            cpu_freq_max_mhz=max(frequencies) if frequencies else None,
        )

    gpu = _GPU_RE.search(raw)
    if gpu:
        recognized += 1
        result.update(
            gpu_util_pct=float(gpu.group("util")),
            gpu_freq_mhz=float(gpu.group("freq")) if gpu.group("freq") else None,
        )

    emc = _EMC_RE.search(raw)
    if emc:
        recognized += 1
        result.update(
            emc_util_pct=float(emc.group("util")),
            emc_freq_mhz=float(emc.group("freq")) if emc.group("freq") else None,
        )

    temperatures: list[float] = []
    for match in _TEMP_RE.finditer(raw):
        recognized += 1
        name = match.group("name").lower()
        value = float(match.group("current"))
        result[f"temp_{name}_c"] = value
        temperatures.append(value)
    result["temp_max_c"] = max(temperatures) if temperatures else None

    power_prefixes = {
        "VDD_IN": "vdd_in",
        "VDD_CPU_GPU_CV": "vdd_cpu_gpu_cv",
        "VDD_SOC": "vdd_soc",
    }
    for match in _POWER_RE.finditer(raw):
        recognized += 1
        prefix = power_prefixes[match.group("name").upper()]
        result[f"{prefix}_current_mw"] = int(match.group("current"))
        result[f"{prefix}_avg_mw"] = (
            int(match.group("average")) if match.group("average") else None
        )
        result[f"{prefix}_peak_mw"] = (
            int(match.group("peak")) if match.group("peak") else None
        )

    required = ("ram_used_mb", "ram_total_mb", "cpu_online", "cpu_total", "gpu_util_pct")
    if recognized:
        result["parse_status"] = (
            "ok" if all(result[name] is not None for name in required) else "partial"
        )
    return result


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def read_platform_metrics(
    *,
    emc_root: Path = Path("/sys/class/devfreq/bwmgr"),
    throttle_paths: Iterable[Path] | None = None,
) -> dict[str, object]:
    """Read optional EMC clock and thermal-trip state exposed by Jetson sysfs."""

    current = _read_int(emc_root / "cur_freq")
    maximum = _read_int(emc_root / "max_freq")
    ratio = None
    if current is not None and maximum is not None and maximum > 0:
        ratio = current / maximum * 100.0

    paths = (
        tuple(throttle_paths)
        if throttle_paths is not None
        else tuple(Path("/sys/devices/platform").glob("*-throttle-alert/thermal_trip_event"))
    )
    readable_events = 0
    active_domains: list[str] = []
    for path in paths:
        state = _read_int(path)
        if state is None:
            continue
        readable_events += 1
        if state != 0:
            active_domains.append(path.parent.name.removesuffix("-throttle-alert"))

    return {
        "emc_freq_hz": current,
        "emc_max_freq_hz": maximum,
        "emc_clock_pct_of_max": ratio,
        "thermal_throttled": bool(active_domains) if readable_events else None,
        "thermal_throttle_domains": ";".join(sorted(active_domains)) if readable_events else None,
    }
