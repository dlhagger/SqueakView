from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType

from squeakview.apps.operator.backend import process
from squeakview.apps.operator.backend.contracts import RunRequest


class OperatorContractTests(unittest.TestCase):
    def test_launch_config_compatibility_name_is_immutable_run_request(self) -> None:
        request = process.LaunchConfig(fps=30)

        self.assertIsInstance(request, RunRequest)
        with self.assertRaises(FrozenInstanceError):
            request.fps = 60  # type: ignore[misc]
        self.assertEqual(replace(request, fps=60).fps, 60)

    def test_bottle_payload_is_snapshotted_at_construction(self) -> None:
        payload = {"left": {"fluid": "water"}}
        request = RunRequest(bottles=payload)

        payload["left"]["fluid"] = "changed"

        self.assertEqual(request.bottles["left"]["fluid"], "water")  # type: ignore[index]
        with self.assertRaises(TypeError):
            request.bottles["left"]["fluid"] = "mutated"  # type: ignore[index]

    def test_nested_ipc_mapping_proxies_are_snapshotted_without_pickling(self) -> None:
        payload = MappingProxyType(
            {"left": MappingProxyType({"fluid": "water"})}
        )

        request = RunRequest(bottles=payload)

        self.assertEqual(request.bottles["left"]["fluid"], "water")  # type: ignore[index]
        with self.assertRaises(TypeError):
            request.bottles["left"]["fluid"] = "changed"  # type: ignore[index]

    def test_sequence_inputs_are_normalized_and_detached_from_callers(self) -> None:
        serials = ["camera-a"]
        sockets = ["/tmp/preview.sock"]

        request = RunRequest(
            camera_serials=serials,  # type: ignore[arg-type]
            preview_socket_paths=sockets,  # type: ignore[arg-type]
        )
        serials.append("camera-b")
        sockets[0] = "/tmp/changed.sock"

        self.assertEqual(request.camera_serials, ("camera-a",))
        self.assertEqual(
            request.preview_socket_paths, (Path("/tmp/preview.sock"),)
        )


if __name__ == "__main__":
    unittest.main()
