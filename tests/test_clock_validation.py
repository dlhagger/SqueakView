from __future__ import annotations

import unittest
from typing import Any

from squeakview.common import clock_validation as clock


class FakeTransport:
    port = "/dev/test"

    def __init__(
        self,
        *,
        offset_ns: int = 100_000_000,
        after_offset_ns: int | None = None,
        rtc_status: str = "RTC_VALID",
        after_rtc_status: str = "RTC_VALID",
        rtts_ns: list[int] | None = None,
    ) -> None:
        self.offset_ns = offset_ns
        self.after_offset_ns = offset_ns if after_offset_ns is None else after_offset_ns
        self.rtc_status = rtc_status
        self.after_rtc_status = after_rtc_status
        self.rtts_ns = rtts_ns or [2_000_000] * 14
        self.time_sync_calls: list[tuple[int, int]] = []
        self.set_rtc_calls: list[int] = []
        self.time_sync_error: BaseException | None = None
        self.set_rtc_error: BaseException | None = None
        self.reply_override: str | None = None

    def exchange_time_sync(
        self, sequence: int, jetson_send_ns: int, *, timeout_s: float
    ) -> tuple[str, int]:
        self.time_sync_calls.append((sequence, jetson_send_ns))
        if self.time_sync_error is not None:
            raise self.time_sync_error
        received_ns = jetson_send_ns + self.rtts_ns[sequence]
        if self.reply_override is not None:
            return self.reply_override, received_ns
        receive_us, send_us = 100, 200
        midpoint_ns = (jetson_send_ns + received_ns) // 2
        corrected = bool(self.set_rtc_calls)
        offset = self.after_offset_ns if corrected else self.offset_ns
        status = self.after_rtc_status if corrected else self.rtc_status
        controller_unix_us = (midpoint_ns + offset + 50_000) // 1000
        return (
            f"CLOCK_SYNC,{sequence},{jetson_send_ns},{receive_us},{send_us},"
            f"{controller_unix_us},{status}",
            received_ns,
        )

    def exchange_set_rtc(self, unix_seconds: int, *, timeout_s: float) -> str:
        self.set_rtc_calls.append(unix_seconds)
        if self.set_rtc_error is not None:
            raise self.set_rtc_error
        return f"ACK_SET_RTC,{unix_seconds},123456,500000"


def validate(transport: FakeTransport, **updates: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "host_status": {
            "ntp_synchronized": True,
            "timezone": "America/New_York",
            "checked_utc": "2026-09-16T12:00:00+00:00",
        },
        "correct": False,
        "interval_s": 0,
        "time_ns": lambda: 1_800_000_000_000_000_000,
        "time_seconds": lambda: 1_800_000_000.4,
        "sleep": lambda _seconds: None,
    }
    args.update(updates)
    return clock.validate_clock(transport, **args)


class ClockValidationTests(unittest.TestCase):
    def test_ntp_synchronized_clock_within_tolerance(self) -> None:
        transport = FakeTransport(offset_ns=1_117_000_000)
        result = validate(transport)
        self.assertEqual(result["result"], "PASS")
        self.assertEqual(result["reason"], "CLOCK_WITHIN_TOLERANCE")
        self.assertEqual(len(result["before_samples"]), 7)

    def test_ntp_not_synchronized_fails_without_serial_traffic(self) -> None:
        transport = FakeTransport()
        result = validate(
            transport,
            host_status={"ntp_synchronized": False, "timezone": "UTC"},
            correct=True,
        )
        self.assertEqual(result["reason"], "JETSON_NTP_NOT_SYNCHRONIZED")
        self.assertFalse(transport.time_sync_calls)
        self.assertFalse(transport.set_rtc_calls)

    def test_invalid_rtc_requires_authorized_correction(self) -> None:
        transport = FakeTransport(rtc_status="RTC_INVALID")
        result = validate(transport)
        self.assertEqual(result["reason"], "CONTROLLER_RTC_INVALID")
        self.assertFalse(result["before"]["all_rtc_valid"])
        self.assertFalse(transport.set_rtc_calls)

    def test_offset_outside_tolerance_without_authorization(self) -> None:
        transport = FakeTransport(offset_ns=2_000_000_000)
        result = validate(transport)
        self.assertEqual(result["reason"], "CLOCK_OFFSET_OUT_OF_TOLERANCE")
        self.assertFalse(transport.set_rtc_calls)

    def test_successful_correction_is_followed_by_new_burst(self) -> None:
        transport = FakeTransport(
            offset_ns=-10_000_000_000,
            after_offset_ns=1_117_000_000,
            rtc_status="RTC_INVALID",
        )
        result = validate(transport, correct=True)
        self.assertEqual(result["reason"], "CLOCK_CORRECTED_AND_VERIFIED")
        self.assertTrue(result["correction_applied"])
        self.assertEqual(len(result["before_samples"]), 7)
        self.assertEqual(len(result["after_samples"]), 7)
        self.assertEqual([item[0] for item in transport.time_sync_calls], list(range(14)))
        self.assertEqual(transport.set_rtc_calls, [1_800_000_000])

    def test_correction_ack_followed_by_bad_verification_fails(self) -> None:
        transport = FakeTransport(
            offset_ns=4_000_000_000,
            after_offset_ns=3_000_000_000,
        )
        result = validate(transport, correct=True)
        self.assertEqual(result["reason"], "CLOCK_CORRECTION_FAILED")
        self.assertEqual(result["result"], "FAIL")

    def test_device_busy_is_distinct(self) -> None:
        transport = FakeTransport()
        transport.time_sync_error = RuntimeError("NACK,TIME_SYNC,DEVICE_BUSY")
        result = validate(transport)
        self.assertEqual(result["reason"], "DEVICE_BUSY")
        self.assertEqual(result["validation_state"], "DEVICE_BUSY")

    def test_time_sync_timeout_is_distinct(self) -> None:
        transport = FakeTransport()
        transport.time_sync_error = TimeoutError("no clock response")
        self.assertEqual(validate(transport)["reason"], "TIME_SYNC_TIMEOUT")

    def test_set_rtc_timeout_leaves_failed_record(self) -> None:
        transport = FakeTransport(offset_ns=2_000_000_000)
        transport.set_rtc_error = TimeoutError("no ack")
        result = validate(transport, correct=True)
        self.assertEqual(result["reason"], "SET_RTC_TIMEOUT")
        self.assertFalse(result["correction_applied"])

    def test_malformed_clock_response(self) -> None:
        transport = FakeTransport()
        transport.reply_override = "CLOCK_SYNC,broken"
        self.assertEqual(validate(transport)["reason"], "MALFORMED_CLOCK_SYNC")

    def test_mismatched_sequence(self) -> None:
        transport = FakeTransport()
        transport.reply_override = (
            "CLOCK_SYNC,99,1800000000000000000,100,200,1800000000001050,RTC_VALID"
        )
        self.assertEqual(validate(transport)["reason"], "MALFORMED_CLOCK_SYNC")

    def test_mismatched_echoed_send_timestamp(self) -> None:
        transport = FakeTransport()
        transport.reply_override = "CLOCK_SYNC,0,7,100,200,8,RTC_VALID"
        self.assertEqual(validate(transport)["reason"], "MALFORMED_CLOCK_SYNC")

    def test_lowest_latency_selection_and_median_math(self) -> None:
        samples = []
        for sequence, (rtt, offset) in enumerate(
            [(9, 900), (1, 100), (3, 300), (2, 200), (8, 800)]
        ):
            samples.append(
                {
                    "sequence": sequence,
                    "round_trip_ns": rtt,
                    "rtc_status": "RTC_VALID",
                    "controller_unix_us": offset / 1000,
                    "rp2040_send_us": 0,
                    "rp2040_midpoint_us": 0,
                    "jetson_midpoint_ns": 0,
                }
            )
        summary = clock.summarize_samples(samples, 3)
        self.assertEqual(summary["selected_sequences"], [1, 3, 2])
        self.assertEqual(summary["median_offset_ns"], 200)

    def test_already_passing_clock_never_receives_set_rtc(self) -> None:
        transport = FakeTransport(offset_ns=0)
        result = validate(transport, correct=True)
        self.assertEqual(result["reason"], "CLOCK_WITHIN_TOLERANCE")
        self.assertFalse(transport.set_rtc_calls)

    def test_record_contains_required_raw_evidence_and_final_reason(self) -> None:
        transport = FakeTransport(offset_ns=0)
        result = validate(transport)
        self.assertIn("started_utc", result)
        self.assertIn("completed_utc", result)
        self.assertEqual(result["controller_identifier"], "/dev/test")
        self.assertEqual(result["configured_tolerance_seconds"], 1.5)
        self.assertEqual(result["before"]["sample_count"], 7)
        self.assertEqual(len(result["before_samples"]), 7)
        self.assertIn("raw", result["before_samples"][0])
        self.assertEqual(result["reason"], "CLOCK_WITHIN_TOLERANCE")


if __name__ == "__main__":
    unittest.main()
