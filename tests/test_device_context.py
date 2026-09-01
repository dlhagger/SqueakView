from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest import mock

from squeakview.common import device_context


class DeviceContextTests(unittest.TestCase):
    def test_snapshot_records_exact_package_versions_and_runtime(self) -> None:
        with (
            mock.patch.object(device_context, "_read_text", side_effect=["Orin", "R39.2.1", "9.1"]),
            mock.patch.object(
                device_context,
                "_command_output",
                side_effect=[
                    "nvidia-l4t-core=39.2.1\ndeepstream-9.1=9.1.0-1\nmissing=",
                    "NV Power Mode: MAXN_SUPER\n2",
                ],
            ),
        ):
            snapshot = device_context.device_context_snapshot()

        self.assertEqual(snapshot["device_model"], "Orin")
        self.assertEqual(snapshot["jetson_linux_release"], "R39.2.1")
        self.assertEqual(snapshot["deepstream_build"], "9.1")
        self.assertEqual(
            snapshot["packages"],
            {"nvidia-l4t-core": "39.2.1", "deepstream-9.1": "9.1.0-1"},
        )
        self.assertIn("python", snapshot)
        self.assertIn("MAXN_SUPER", str(snapshot["nvpmodel"]))

    def test_file_identity_hashes_native_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "plugin.so"
            artifact.write_bytes(b"scientific-plugin")

            identity = device_context.file_identity(artifact)

        self.assertTrue(identity["available"])
        self.assertEqual(identity["size_bytes"], 17)
        self.assertEqual(len(str(identity["sha256"])), 64)


if __name__ == "__main__":
    unittest.main()
