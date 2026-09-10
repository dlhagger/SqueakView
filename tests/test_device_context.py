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
                    "nvidia-jetpack=7.2.1-b17\nnvidia-l4t-core=39.2.1\n"
                    "deepstream-9.1=9.1.0-1\nmissing=",
                    "NV Power Mode: MAXN_SUPER\n2",
                ],
            ) as command_output,
        ):
            snapshot = device_context.device_context_snapshot()

        self.assertEqual(snapshot["device_model"], "Orin")
        self.assertEqual(snapshot["jetson_linux_release"], "R39.2.1")
        self.assertEqual(snapshot["deepstream_build"], "9.1")
        self.assertEqual(
            snapshot["packages"],
            {
                "nvidia-jetpack": "7.2.1-b17",
                "nvidia-l4t-core": "39.2.1",
                "deepstream-9.1": "9.1.0-1",
            },
        )
        package_command = command_output.call_args_list[0].args[0]
        self.assertIn("nvidia-jetpack", package_command)
        self.assertIn("ffmpeg", package_command)
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

    def test_file_identity_is_size_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "large.so"
            artifact.write_bytes(b"12345")

            identity = device_context.file_identity(artifact, max_bytes=4)

        self.assertFalse(identity["available"])
        self.assertIn("byte limit", str(identity["error"]))


if __name__ == "__main__":
    unittest.main()
