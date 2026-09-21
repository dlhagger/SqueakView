from __future__ import annotations

"""Hardware qualification client for the MouseHouse protocol-v2 transport."""

import argparse
import json
import sys
import time
from pathlib import Path

from squeakview.common import clock_validation
from squeakview.common import controller_protocol_v2 as protocol_v2
from squeakview.common import serial as serial_util


def _exercise_replay(
    handle: serial_util.SerialHandle,
    checks: dict[str, object],
    *,
    timeout_s: float = 5.0,
) -> dict[str, int]:
    """Request recent, deliberately unacknowledged records and verify replay."""

    snapshot = handle.protocol_v2_snapshot
    boot_id = snapshot.get("boot_id")
    if not boot_id:
        raise RuntimeError("cannot exercise replay before learning the controller boot ID")
    counts = snapshot.get("counts")
    if not isinstance(counts, dict):
        raise RuntimeError("protocol-v2 counters are unavailable")
    contiguous = handle.protocol_v2_contiguous_sequence()
    if contiguous < 1:
        raise RuntimeError("cannot exercise replay before a durable v2 record exists")

    # The firmware retains unacknowledged reliable records, not an indefinite
    # history of records the host has already acknowledged.  This probe must be
    # issued before ACK release; selecting a recent sequence keeps it within the
    # bounded controller queue even during a busy qualification run.
    resend_from = max(1, contiguous - 2)
    duplicate_before = int(counts["duplicates"])
    retransmissions_before = int(counts.get("retransmissions_received", 0))
    handle.request_protocol_v2_replay(resend_from)

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and handle.fatal_error is None:
        current = handle.protocol_v2_snapshot
        current_counts = current.get("counts")
        if not isinstance(current_counts, dict):
            raise RuntimeError("protocol-v2 counters disappeared during replay")
        if int(current_counts["duplicates"]) > duplicate_before:
            break
        time.sleep(0.05)

    current = handle.protocol_v2_snapshot
    current_counts = current.get("counts")
    if not isinstance(current_counts, dict):
        raise RuntimeError("protocol-v2 counters disappeared after replay")
    checks["replay_duplicate_received"] = (
        int(current_counts["duplicates"]) > duplicate_before
    )
    retransmissions_after = int(current_counts.get("retransmissions_received", 0))
    checks["retransmission_flag_seen"] = (
        retransmissions_after > retransmissions_before
    )
    return {
        "boot_id": int(boot_id),
        "from_sequence": resend_from,
        "durable_sequence_at_request": contiguous,
    }


def _finish_ack_hold(
    handle: serial_util.SerialHandle,
    checks: dict[str, object],
    *,
    request_resend: bool,
) -> dict[str, int] | None:
    """Exercise replay while records remain retained, then release ACKs."""

    try:
        return _exercise_replay(handle, checks) if request_resend else None
    finally:
        handle.set_protocol_v2_ack_withheld(False)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", nargs="?", default="/dev/ttyACM0")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--output", type=Path, default=Path("protocol-v2-test"))
    parser.add_argument("--withhold-acks", type=float, default=0.0)
    parser.add_argument("--request-resend", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument(
        "--correct-clock",
        action="store_true",
        help=(
            "authorize one idle controller RTC correction when clock validation "
            "is outside tolerance"
        ),
    )
    parser.add_argument("--no-start", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--exercise-overflow-failsafe", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.fps <= 0 or args.duration <= 0 or args.withhold_acks < 0:
        raise SystemExit("fps/duration must be positive and withhold-acks non-negative")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    messages: list[str] = []

    def emit(line: str) -> None:
        messages.append(line)
        if args.verbose or "FATAL" in line or "WARN" in line:
            print(line, flush=True)

    handle = serial_util.SerialHandle(args.port, 115200, emit)
    report: dict[str, object] = {
        "schema_version": 1,
        "port": args.port,
        "fps": args.fps,
        "duration_s": args.duration,
        "started_unix_ns": time.time_ns(),
        "checks": {},
        "passed": False,
    }
    checks: dict[str, object] = report["checks"]  # type: ignore[assignment]
    started = False
    try:
        if not handle.open(output):
            raise RuntimeError(handle.last_error or "serial open failed")
        host_status = clock_validation.host_time_status()
        clock = clock_validation.validate_clock(
            handle,
            host_status=host_status,
            correct=args.correct_clock,
            controller_identifier=args.port,
        )
        report["clock_validation"] = clock
        checks["clock_preflight"] = clock.get("result") == "PASS"
        if not checks["clock_preflight"]:
            raise RuntimeError(
                f"clock preflight failed: {clock.get('reason')} {clock.get('detail', '')}"
            )

        handle.negotiate_protocol_v2(timeout_s=3.0)
        checks["v2_negotiated"] = True
        report["controller_reboot_required_before_next_v1_preflight"] = True
        if args.status:
            handle.request_protocol_v2_status()

        if not args.no_start:
            handle.send_start(args.fps)
            started = True
            checks["camera_epoch"] = handle.wait_for_protocol_v2_message(
                protocol_v2.MessageType.CAMERA_EPOCH, timeout_s=3.0
            )
            if not checks["camera_epoch"]:
                raise RuntimeError("CAMERA_EPOCH was not received")

        withhold = args.withhold_acks
        if args.request_resend and withhold <= 0:
            withhold = min(5.0, args.duration)
        if args.exercise_overflow_failsafe:
            withhold = max(withhold, args.duration)
        deadline = time.monotonic() + args.duration
        withhold_deadline = time.monotonic() + withhold
        if withhold:
            handle.set_protocol_v2_ack_withheld(True)
        replay_exercised = False
        next_status = time.monotonic()
        while time.monotonic() < deadline and handle.fatal_error is None:
            now = time.monotonic()
            if withhold and now >= withhold_deadline:
                report["replay_probe"] = _finish_ack_hold(
                    handle,
                    checks,
                    request_resend=args.request_resend,
                )
                replay_exercised = args.request_resend
                withhold = 0.0
            if args.status or args.exercise_overflow_failsafe:
                if now >= next_status:
                    handle.request_protocol_v2_status()
                    next_status = now + 1.0
            time.sleep(0.05)
        if withhold:
            if args.request_resend and handle.fatal_error is None:
                report["replay_probe"] = _finish_ack_hold(
                    handle,
                    checks,
                    request_resend=True,
                )
                replay_exercised = True
            else:
                handle.set_protocol_v2_ack_withheld(False)
            withhold = 0.0
        else:
            handle.set_protocol_v2_ack_withheld(False)

        if args.request_resend and not replay_exercised:
            checks["replay_duplicate_received"] = False
            checks["retransmission_flag_seen"] = False

        snapshot = handle.protocol_v2_snapshot
        report["before_stop"] = snapshot
        message_counts, payloads = handle.protocol_v2_observations()
        checks["periodic_checkpoint"] = bool(
            args.no_start
            or message_counts.get(int(protocol_v2.MessageType.CAMERA_CHECKPOINT), 0)
        )
        checks["no_v2_camera_edges"] = not any(
            payload.startswith("CAMERA_HIGH,") or payload.startswith("CAMERA_LOW,")
            for _kind, payload, _sequence, _flags in payloads
        )

        if started:
            camera_stop_before = message_counts.get(
                int(protocol_v2.MessageType.CAMERA_STOP), 0
            )
            handle.send_line("STOP")
            checks["stop_result"] = handle.wait_for_stop_ack(timeout_s=3.0)
            checks["camera_stop"] = handle.wait_for_protocol_v2_message(
                protocol_v2.MessageType.CAMERA_STOP,
                after_count=camera_stop_before,
                timeout_s=3.0,
            )
            started = False

        final_snapshot = handle.protocol_v2_snapshot
        report["final"] = final_snapshot
        if args.exercise_overflow_failsafe:
            checks["overflow_integrity_latched"] = bool(
                final_snapshot.get("integrity_latched")
            )
        required = [
            value for key, value in checks.items()
            if key not in {"periodic_checkpoint"} or not args.no_start
        ]
        report["passed"] = all(value is True for value in required)
        if handle.fatal_error and not args.exercise_overflow_failsafe:
            report["fatal_error"] = handle.fatal_error
            report["passed"] = False
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["passed"] = False
    finally:
        if started and handle.ser is not None and getattr(handle.ser, "is_open", False):
            try:
                handle.send_line("STOP")
            except Exception:
                pass
        handle.close()
        report["completed_unix_ns"] = time.time_ns()
        report_path = output / "protocol_v2_test_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"{'PASS' if report.get('passed') else 'FAIL'}: {report_path}",
            flush=True,
        )
    return 0 if report.get("passed") is True else 1


if __name__ == "__main__":
    sys.exit(main())
