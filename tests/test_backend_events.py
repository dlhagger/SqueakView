from __future__ import annotations

import json
import unittest

from squeakview.apps.operator.backend.events import BackendEvent, RunPhase, RunStateMachine


class RunStateMachineTests(unittest.TestCase):
    def test_event_payload_is_recursively_immutable_and_json_serializable(self) -> None:
        source = {"nested": {"values": [1, 2]}}
        event = BackendEvent("test", RunPhase.IDLE, None, payload=source)
        source["nested"]["values"].append(3)

        self.assertEqual(event.payload["nested"]["values"], (1, 2))
        with self.assertRaises(TypeError):
            event.payload["nested"]["new"] = True
        encoded = json.dumps({"payload": event.payload})
        self.assertEqual(json.loads(encoded)["payload"]["nested"]["values"], [1, 2])

    def test_complete_scientific_lifecycle(self) -> None:
        machine = RunStateMachine()
        phases = (
            RunPhase.CREATED,
            RunPhase.STARTING,
            RunPhase.RECORDING,
            RunPhase.STOPPING,
            RunPhase.DRAINING,
            RunPhase.CAPTURE_CLOSED,
            RunPhase.VALIDATING,
            RunPhase.FINALIZED,
        )
        for phase in phases:
            machine.transition(phase)
        self.assertEqual(machine.phase, RunPhase.FINALIZED)

    def test_failure_is_allowed_from_every_nonterminal_run_phase(self) -> None:
        for phase in (
            RunPhase.CREATED,
            RunPhase.STARTING,
            RunPhase.RECORDING,
            RunPhase.STOPPING,
            RunPhase.DRAINING,
            RunPhase.CAPTURE_CLOSED,
            RunPhase.VALIDATING,
        ):
            with self.subTest(phase=phase):
                machine = RunStateMachine(phase)
                machine.transition(RunPhase.FAILED)
                self.assertEqual(machine.phase, RunPhase.FAILED)

    def test_forbidden_transition_is_rejected(self) -> None:
        machine = RunStateMachine()
        with self.assertRaisesRegex(ValueError, "idle -> recording"):
            machine.transition(RunPhase.RECORDING)


if __name__ == "__main__":
    unittest.main()
