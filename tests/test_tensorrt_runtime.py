"""Private TensorRT provisioning must fail without damaging a working environment."""

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from squeakview.common import tensorrt_runtime


spec = importlib.util.spec_from_file_location(
    "tensorrt_installer", Path(__file__).resolve().parents[1] / "scripts/install_tensorrt10.py"
)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class TensorRTInstallerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.venv = self.root / ".venv"
        self.site = self.venv / "lib/python3.12/site-packages"
        self.site.mkdir(parents=True)
        self.destination = self.venv / "vendor" / f"tensorrt-{installer.VERSION}"
        self.pth = self.site / "squeakview_tensorrt10.pth"
        self.corrupt_download = False
        self.fail_activation = False
        self.sdk_abi = "10"
        self.runtime_version = installer.DEB_VERSION
        self.packages = {
            name: hashlib.sha256(name.encode()).hexdigest() for name in installer.PACKAGES
        }
        self.commands = []
        patches = (
            mock.patch.object(installer, "ROOT", self.root),
            mock.patch.object(installer, "PACKAGES", self.packages),
            mock.patch.object(installer.platform, "machine", return_value="aarch64"),
            mock.patch.object(installer.subprocess, "check_output", side_effect=self.check_output),
            mock.patch.object(installer.subprocess, "run", side_effect=self.run_command),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def check_output(self, command, **kwargs):
        if command[0] == "readelf":
            return f"Shared library: [libnvinfer.so.{self.sdk_abi}]"
        return "installed " + self.runtime_version

    def run_command(self, command, **kwargs):
        self.commands.append(command)
        if command[0] == "apt-get":
            name = command[-1].split("=", 1)[0]
            deb = Path(kwargs["cwd"]) / f"{name}_{installer.DEB_VERSION}_arm64.deb"
            deb.write_bytes(b"corrupt" if self.corrupt_download else name.encode())
        elif command[0] == "dpkg-deb":
            payload = Path(command[-1])
            for relative in (
                "usr/bin/trtexec",
                "usr/lib/python3.12/dist-packages/tensorrt/__init__.py",
                "usr/lib/python3.12/dist-packages/tensorrt/tensorrt.so",
            ):
                path = payload / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
        elif self.fail_activation and len(command) > 2 and "from pathlib import Path" in command[2]:
            raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    def test_bad_checksum_leaves_no_bundle_and_retry_can_succeed(self):
        self.corrupt_download = True
        with self.assertRaisesRegex(SystemExit, "SHA256 mismatch"):
            installer.main()
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.pth.exists())
        self.assertEqual(list((self.venv / "vendor").iterdir()), [])
        self.assertFalse(any(c[0] == "dpkg-deb" for c in self.commands))
        self.corrupt_download = False
        installer.main()
        self.assertTrue(self.destination.is_dir())
        self.assertIn(str(self.destination), self.pth.read_text())

    def test_wrong_sdk_or_runtime_fails_before_download(self):
        self.sdk_abi = "11"
        with self.assertRaisesRegex(SystemExit, "does not use TensorRT 10"):
            installer.main()
        self.sdk_abi = "10"
        self.runtime_version = "10.16.1.7-1+cuda13.2"
        with self.assertRaisesRegex(SystemExit, "Required runtime"):
            installer.main()
        self.assertFalse(any(c[0] == "apt-get" for c in self.commands))

    def test_activation_failure_restores_existing_pth_and_retry_reuses_bundle(self):
        self.pth.write_bytes(b"previous binding location\n")
        self.fail_activation = True
        with self.assertRaises(subprocess.CalledProcessError):
            installer.main()
        self.assertEqual(self.pth.read_bytes(), b"previous binding location\n")
        self.assertTrue(self.destination.exists())
        self.commands.clear()
        self.fail_activation = False
        installer.main()
        self.assertFalse(any(c[0] == "apt-get" for c in self.commands))
        self.assertIn(str(self.destination), self.pth.read_text())

    def test_activation_failure_removes_new_pth(self):
        self.fail_activation = True
        with self.assertRaises(subprocess.CalledProcessError):
            installer.main()
        self.assertFalse(self.pth.exists())

    def test_incomplete_existing_bundle_is_preserved_without_activation(self):
        installer.main()
        self.pth.write_bytes(b"previous binding location\n")
        cli = self.destination / "usr/bin/trtexec"
        cli.unlink()
        with self.assertRaisesRegex(SystemExit, "Incomplete TensorRT bundle"):
            installer.main()
        self.assertEqual(self.pth.read_bytes(), b"previous binding location\n")
        self.assertTrue(self.destination.exists())

    def test_invalid_existing_manifest_is_preserved_without_activation(self):
        installer.main()
        manifest = self.destination / "squeakview-bundle.json"
        manifest.write_text("invalid JSON")
        original = self.pth.read_bytes()
        with self.assertRaisesRegex(SystemExit, "Unreadable TensorRT bundle identity"):
            installer.main()
        self.assertEqual(manifest.read_text(), "invalid JSON")
        self.assertEqual(self.pth.read_bytes(), original)


class TensorRTRuntimeTests(unittest.TestCase):
    def test_missing_private_cli_never_selects_system_tool(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            tensorrt_runtime, "bundled_tensorrt_root", return_value=Path(temporary)
        ), mock.patch.dict("os.environ", {"PATH": "/usr/bin"}):
            with self.assertRaisesRegex(RuntimeError, "install_tensorrt10.py"):
                tensorrt_runtime.trtexec_executable()

    def test_installer_and_runtime_share_the_same_version_and_hashes(self):
        self.assertEqual(installer.VERSION, tensorrt_runtime.TENSORRT_VERSION)
        self.assertEqual(installer.DEB_VERSION, tensorrt_runtime.TENSORRT_DEB_VERSION)
        self.assertIs(installer.PACKAGES, tensorrt_runtime.TENSORRT_TOOL_PACKAGES)
