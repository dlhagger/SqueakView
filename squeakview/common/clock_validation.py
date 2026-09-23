"""Pre-run controller RTC validation using an already-owned serial transport."""

from __future__ import annotations

import statistics
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol, Sequence


DEFAULT_SAMPLE_COUNT = 7
DEFAULT_BEST_SAMPLE_COUNT = 3
DEFAULT_MAX_OFFSET_SECONDS = 1.5
DEFAULT_SAMPLE_INTERVAL_SECONDS = 0.1


class ClockTransport(Protocol):
    port: str

    def exchange_time_sync(
        self, sequence: int, jetson_send_ns: int, *, timeout_s: float
    ) -> tuple[str, int]: ...

    def exchange_set_rtc(
        self, unix_seconds: int, *, timeout_s: float
    ) -> str: ...


class ClockValidationError(RuntimeError):
    """Expected protocol/transport failure with a machine-readable reason."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timedatectl_property(name: str) -> str:
    try:
        result = subprocess.run(
            ["timedatectl", "show", f"--property={name}", "--value"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Could not query Jetson time status: {exc}") from exc
    value = result.stdout.strip()
    if result.returncode != 0 or not value:
        detail = result.stderr.strip() or "no value returned"
        raise RuntimeError(f"Could not query Jetson time status: {detail}")
    return value


def host_time_status(
    property_reader: Callable[[str], str] = _timedatectl_property,
) -> dict[str, Any]:
    return {
        "ntp_synchronized": property_reader("NTPSynchronized").lower() == "yes",
        "timezone": property_reader("Timezone"),
        "checked_utc": utc_now(),
    }


def parse_clock_response(
    line: str, sequence: int, sent_ns: int, received_ns: int
) -> dict[str, int | float | str]:
    fields = line.split(",")
    try:
        if (
            len(fields) != 7
            or fields[0] != "CLOCK_SYNC"
            or int(fields[1]) != sequence
            or int(fields[2]) != sent_ns
            or fields[6] not in {"RTC_VALID", "RTC_INVALID"}
        ):
            raise ValueError
        receive_us = int(fields[3])
        send_us = int(fields[4])
        controller_unix_us = int(fields[5])
    except (TypeError, ValueError) as exc:
        raise ClockValidationError(
            "MALFORMED_CLOCK_SYNC",
            f"Malformed or mismatched CLOCK_SYNC response: {line}",
        ) from exc
    if received_ns < sent_ns or receive_us < 0 or send_us < receive_us:
        raise ClockValidationError(
            "MALFORMED_CLOCK_SYNC", f"Invalid CLOCK_SYNC timing fields: {line}"
        )
    return {
        "raw": line,
        "sequence": sequence,
        "jetson_send_ns": sent_ns,
        "jetson_receive_ns": received_ns,
        "round_trip_ns": received_ns - sent_ns,
        "rp2040_receive_us": receive_us,
        "rp2040_send_us": send_us,
        "rp2040_midpoint_us": (receive_us + send_us) / 2.0,
        "jetson_midpoint_ns": (sent_ns + received_ns) / 2.0,
        "controller_unix_us": controller_unix_us,
        "rtc_status": fields[6],
    }


def sample_offset_ns(sample: Mapping[str, Any]) -> float:
    controller_midpoint_ns = float(sample["controller_unix_us"]) * 1000.0 - (
        float(sample["rp2040_send_us"]) - float(sample["rp2040_midpoint_us"])
    ) * 1000.0
    return controller_midpoint_ns - float(sample["jetson_midpoint_ns"])


def summarize_samples(
    samples: Sequence[Mapping[str, Any]],
    best_sample_count: int = DEFAULT_BEST_SAMPLE_COUNT,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("At least one clock sample is required")
    if best_sample_count <= 0:
        raise ValueError("best_sample_count must be positive")
    selected = sorted(samples, key=lambda item: int(item["round_trip_ns"]))[
        : min(best_sample_count, len(samples))
    ]
    offsets = [sample_offset_ns(sample) for sample in selected]
    return {
        "sample_count": len(samples),
        "selected_sample_count": len(selected),
        "all_rtc_valid": all(sample["rtc_status"] == "RTC_VALID" for sample in samples),
        "median_offset_ns": statistics.median(offsets),
        "median_offset_seconds": statistics.median(offsets) / 1_000_000_000.0,
        "median_round_trip_ms": statistics.median(
            float(sample["round_trip_ns"]) for sample in selected
        )
        / 1_000_000.0,
        "selected_sequences": [int(sample["sequence"]) for sample in selected],
    }


def _nack_reason(line: str, command: str) -> ClockValidationError:
    if line == f"NACK,{command},DEVICE_BUSY":
        return ClockValidationError(
            "DEVICE_BUSY", "Controller session or feeder is active; clock access was rejected"
        )
    suffix = line.removeprefix(f"NACK,{command},")
    reason = f"{command}_{suffix}" if suffix else f"{command}_NACK"
    return ClockValidationError(reason, f"Controller rejected {command}: {line}")


def collect_burst(
    transport: ClockTransport,
    *,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    interval_s: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
    starting_sequence: int = 0,
    timeout_s: float = 3.0,
    time_ns: Callable[[], int] = time.time_ns,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for index in range(sample_count):
        sequence = starting_sequence + index
        sent_ns = time_ns()
        try:
            line, received_ns = transport.exchange_time_sync(
                sequence, sent_ns, timeout_s=timeout_s
            )
        except TimeoutError as exc:
            raise ClockValidationError("TIME_SYNC_TIMEOUT", str(exc)) from exc
        except (ConnectionError, RuntimeError) as exc:
            text = str(exc)
            if text.startswith("NACK,TIME_SYNC,"):
                raise _nack_reason(text, "TIME_SYNC") from exc
            raise ClockValidationError("TIME_SYNC_TRANSPORT_ERROR", text) from exc
        if line.startswith("NACK,TIME_SYNC,"):
            raise _nack_reason(line, "TIME_SYNC")
        samples.append(parse_clock_response(line, sequence, sent_ns, received_ns))
        if index + 1 < sample_count:
            sleep(interval_s)
    return samples


def parse_set_rtc_ack(line: str, unix_seconds: int) -> dict[str, Any]:
    if line.startswith("NACK,SET_RTC,"):
        raise _nack_reason(line, "SET_RTC")
    fields = line.split(",")
    try:
        if (
            len(fields) != 4
            or fields[0] != "ACK_SET_RTC"
            or int(fields[1]) != unix_seconds
        ):
            raise ValueError
        anchor_us = int(fields[2])
        uncertainty_us = int(fields[3])
    except (TypeError, ValueError) as exc:
        raise ClockValidationError(
            "MALFORMED_SET_RTC_ACK", f"Malformed or mismatched SET_RTC response: {line}"
        ) from exc
    return {
        "raw": line,
        "unix_seconds": unix_seconds,
        "rp2040_anchor_us": anchor_us,
        "anchor_uncertainty_us": uncertainty_us,
    }


def validate_clock(
    transport: ClockTransport,
    *,
    host_status: Mapping[str, Any],
    correct: bool,
    controller_identifier: str | None = None,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    best_sample_count: int = DEFAULT_BEST_SAMPLE_COUNT,
    max_offset_seconds: float = DEFAULT_MAX_OFFSET_SECONDS,
    interval_s: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
    progress: Callable[[str, Mapping[str, Any]], None] | None = None,
    time_ns: Callable[[], int] = time.time_ns,
    time_seconds: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Validate and optionally correct the idle controller RTC, never raising."""

    record: dict[str, Any] = {
        "schema_version": "1.0",
        "gate": "MouseHouse pre-run clock validation",
        "started_utc": utc_now(),
        "host": dict(host_status),
        "controller_identifier": controller_identifier or str(transport.port),
        "configured_tolerance_seconds": max_offset_seconds,
        "required_sample_count": sample_count,
        "best_sample_count": best_sample_count,
        "correction_requested": bool(correct),
        "correction_applied": False,
        "controller_idle_verification": (
            "TIME_SYNC acceptance (firmware rejects active session/feed)"
        ),
        "set_rtc_ack": None,
        "before_samples": [],
        "after_samples": [],
    }

    def publish(state: str, **details: Any) -> None:
        record["validation_state"] = state
        if progress is not None:
            progress(state, {**record, **details})

    def finish(result: str, reason: str, detail: str = "") -> dict[str, Any]:
        record.update(
            result=result,
            reason=reason,
            detail=detail,
            completed_utc=utc_now(),
        )
        publish(
            (
                "WITHIN_TOLERANCE"
                if reason == "CLOCK_WITHIN_TOLERANCE"
                else "CORRECTED_AND_VERIFIED"
                if reason == "CLOCK_CORRECTED_AND_VERIFIED"
                else "JETSON_NTP_NOT_SYNCHRONIZED"
                if reason == "JETSON_NTP_NOT_SYNCHRONIZED"
                else "DEVICE_BUSY"
                if reason == "DEVICE_BUSY"
                else "CORRECTION_FAILED"
                if reason.startswith("SET_RTC_") or reason == "CLOCK_CORRECTION_FAILED"
                else "CORRECTION_REQUIRED"
                if reason in {"CONTROLLER_RTC_INVALID", "CLOCK_OFFSET_OUT_OF_TOLERANCE"}
                else "VALIDATION_ERROR"
            )
        )
        return record

    if not bool(host_status.get("ntp_synchronized", False)):
        return finish("FAIL", "JETSON_NTP_NOT_SYNCHRONIZED")

    try:
        publish("CHECKING")
        before_samples = collect_burst(
            transport,
            sample_count=sample_count,
            interval_s=interval_s,
            timeout_s=3.0,
            time_ns=time_ns,
            sleep=sleep,
        )
        record["before_samples"] = before_samples
        record["controller_idle_verified"] = True
        before = summarize_samples(before_samples, best_sample_count)
        record["before"] = before
        before_passed = bool(before["all_rtc_valid"]) and abs(
            float(before["median_offset_seconds"])
        ) <= max_offset_seconds
        if before_passed:
            return finish("PASS", "CLOCK_WITHIN_TOLERANCE")
        if not correct:
            reason = (
                "CONTROLLER_RTC_INVALID"
                if not before["all_rtc_valid"]
                else "CLOCK_OFFSET_OUT_OF_TOLERANCE"
            )
            return finish("FAIL", reason, "RTC correction was not authorized")

        publish("CORRECTING")
        unix_seconds = int(time_seconds() + 0.5)
        try:
            reply = transport.exchange_set_rtc(unix_seconds, timeout_s=3.0)
        except TimeoutError as exc:
            raise ClockValidationError("SET_RTC_TIMEOUT", str(exc)) from exc
        except (ConnectionError, RuntimeError) as exc:
            text = str(exc)
            if text.startswith("NACK,SET_RTC,"):
                raise _nack_reason(text, "SET_RTC") from exc
            raise ClockValidationError("SET_RTC_TRANSPORT_ERROR", text) from exc
        record["set_rtc_ack"] = parse_set_rtc_ack(reply, unix_seconds)
        record["correction_applied"] = True

        publish("CHECKING")
        after_samples = collect_burst(
            transport,
            sample_count=sample_count,
            interval_s=interval_s,
            starting_sequence=sample_count,
            timeout_s=3.0,
            time_ns=time_ns,
            sleep=sleep,
        )
        record["after_samples"] = after_samples
        after = summarize_samples(after_samples, best_sample_count)
        record["after"] = after
        passed = bool(after["all_rtc_valid"]) and abs(
            float(after["median_offset_seconds"])
        ) <= max_offset_seconds
        return finish(
            "PASS" if passed else "FAIL",
            "CLOCK_CORRECTED_AND_VERIFIED" if passed else "CLOCK_CORRECTION_FAILED",
        )
    except ClockValidationError as exc:
        return finish("FAIL", exc.reason, exc.detail)
    except Exception as exc:
        return finish("FAIL", "VALIDATION_ERROR", f"{type(exc).__name__}: {exc}")


__all__ = [
    "ClockValidationError",
    "DEFAULT_BEST_SAMPLE_COUNT",
    "DEFAULT_MAX_OFFSET_SECONDS",
    "DEFAULT_SAMPLE_COUNT",
    "collect_burst",
    "host_time_status",
    "parse_clock_response",
    "parse_set_rtc_ack",
    "sample_offset_ns",
    "summarize_samples",
    "validate_clock",
]
