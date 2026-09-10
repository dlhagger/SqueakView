from __future__ import annotations

import dataclasses
import subprocess
import tempfile
import unittest
from pathlib import Path

from squeakview.apps.operator.backend import lifecycle
from squeakview.apps.operator.backend.events import RunPhase
from squeakview.common import run_context


class _Capture:
    def __init__(self, events: list[str], *, wait_fails: bool = False) -> None:
        self.events = events
        self.wait_fails = wait_fails

    def is_running(self) -> bool:
        return True

    def terminate_group_graceful(self, *_args) -> None:
        self.events.append("capture:terminate")

    def wait(self, timeout: float | None = None) -> int:
        self.events.append(f"capture:wait:{timeout}")
        if self.wait_fails:
            raise subprocess.TimeoutExpired("capture", timeout)
        return 0


class _Serial:
    stop_ack_count = 17

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def log_marker(self, marker: str) -> None:
        self.events.append(f"marker:{marker}")

    def send_line(self, text: str) -> None:
        self.events.append(f"serial:{text}")

    def wait_for_stop_ack(self, timeout_s: float = 2.0) -> bool:
        self.events.append(f"serial:ack:{timeout_s}")
        return True

    def disarm_watchdog_v1(self, *, timeout_s: float = 2.0) -> None:
        self.events.append(f"serial:disarm:{timeout_s}")

    def close(self) -> None:
        self.events.append("serial:close")


class FinalizationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp_dir.name) / "run"
        self.run_dir.mkdir()
        run_context.atomic_write_json(
            self.run_dir / run_context.RUN_STATUS_FILENAME,
            {
                "recording_validation": {"passed": True},
                "acquisition_integrity": {"passed": True},
            },
        )
        self.events: list[str] = []

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _hooks(self) -> lifecycle.FinalizationHooks:
        def transition(phase: RunPhase, *, message: str | None = None) -> None:
            del message
            self.events.append(f"phase:{phase.value}")

        def drain(_run_dir: Path, *, expected_ttl_count: int | None) -> bool:
            self.events.append(f"capture:drain:{expected_ttl_count}")
            return True

        def finalizer(_run_dir: Path) -> int:
            self.events.append("capture:validate")
            return 0

        def manifest(_run_dir: Path) -> bool:
            self.events.append("manifest:write")
            return True

        def outputs(_run_dir: Path) -> dict[str, object]:
            self.events.append("outputs:snapshot")
            return {}

        return lifecycle.FinalizationHooks(
            log=lambda message: self.events.append(f"log:{message}"),
            transition=transition,
            wait_for_capture_drain=drain,
            run_capture_finalizer=finalizer,
            write_run_manifest=manifest,
            run_output_snapshot=outputs,
            sleep=lambda seconds: self.events.append(f"sleep:{seconds}"),
        )

    def _request(self, capture: _Capture, serial: _Serial) -> lifecycle.FinalizationRequest:
        return lifecycle.FinalizationRequest(
            final_state="finalized",
            error=None,
            terminate_capture=True,
            known_capture_returncode=None,
            run_dir=self.run_dir,
            capture=capture,
            serial=serial,
            capture_running=True,
            controller_started=True,
            trigger_on=True,
            phase=RunPhase.RECORDING,
        )

    def test_contracts_are_immutable(self) -> None:
        request = self._request(_Capture(self.events), _Serial(self.events))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            request.final_state = "failed"  # type: ignore[misc]

        result = lifecycle.FinalizationResult("finalized", None, 0, True)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.error = "changed"  # type: ignore[misc]

    def test_scientific_shutdown_order_is_explicit(self) -> None:
        result = lifecycle.finalize_run(
            self._request(_Capture(self.events), _Serial(self.events)),
            self._hooks(),
        )

        self.assertFalse(result.failed)
        ordered = [
            "marker:CAPTURE_STOP_REQUESTED",
            "serial:STOP",
            "serial:ack:2.0",
            "capture:drain:17",
            "capture:terminate",
            "capture:wait:2",
            "serial:close",
            "phase:capture_closed",
            "phase:validating",
            "capture:validate",
        ]
        positions = [self.events.index(event) for event in ordered]
        self.assertEqual(positions, sorted(positions), self.events)

    def test_unconfirmed_capture_exit_skips_validation_and_fails_closed(self) -> None:
        result = lifecycle.finalize_run(
            self._request(
                _Capture(self.events, wait_fails=True),
                _Serial(self.events),
            ),
            self._hooks(),
        )

        self.assertTrue(result.failed)
        self.assertIn("did not exit", result.error or "")
        self.assertNotIn("capture:validate", self.events)
        self.assertIn("serial:close", self.events)
        self.assertNotIn("phase:finalized", self.events)

    def test_watchdog_disarm_ack_precedes_capture_drain(self) -> None:
        request = dataclasses.replace(
            self._request(_Capture(self.events), _Serial(self.events)),
            controller_protocol="watchdog_v1_experimental",
        )
        result = lifecycle.finalize_run(request, self._hooks())

        self.assertFalse(result.failed)
        self.assertNotIn("serial:STOP", self.events)
        ordered = [
            "marker:DISARM_SENT",
            "serial:disarm:2.0",
            "marker:WATCHDOG_V1_DISARM_ACKED",
            "capture:drain:17",
            "capture:terminate",
        ]
        positions = [self.events.index(event) for event in ordered]
        self.assertEqual(positions, sorted(positions), self.events)

    def test_required_controller_alignment_cannot_be_skipped(self) -> None:
        request = dataclasses.replace(
            self._request(_Capture(self.events), _Serial(self.events)),
            alignment_required=True,
        )

        result = lifecycle.finalize_run(request, self._hooks())

        self.assertTrue(result.failed)
        self.assertIn("alignment validation", result.error or "")

    def test_free_running_serial_run_finalizes_without_trigger_alignment(self) -> None:
        request = dataclasses.replace(
            self._request(_Capture(self.events), _Serial(self.events)),
            controller_started=False,
            trigger_on=False,
            alignment_required=False,
        )

        result = lifecycle.finalize_run(request, self._hooks())

        self.assertFalse(result.failed)
        self.assertEqual(result.final_state, "finalized")
        self.assertIn("serial:close", self.events)
        self.assertIn("capture:validate", self.events)


if __name__ == "__main__":
    unittest.main()
