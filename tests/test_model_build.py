from __future__ import annotations

import tempfile
import unittest
import os
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from squeakview import model_build


class ModelBuildPromotionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.models = Path(self.temp_dir.name) / "models"
        self.models.mkdir()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def validator(config: Path):
        if config.read_text() != "valid\n":
            raise ValueError("invalid staged package")
        return SimpleNamespace(root=config.parent.parent)

    def stage(self, value: str = "valid\n") -> Path:
        package = model_build.create_staging_package(self.models, "mousehouse")
        (package / "configs").mkdir()
        (package / "configs" / "mousehouse.txt").write_text(value)
        return package

    def test_new_package_is_published_only_after_validation(self) -> None:
        staging = self.stage()
        destination = self.models / "mousehouse"

        config = model_build.promote_model_package(
            staging,
            destination,
            config_name="mousehouse.txt",
            overwrite=False,
            validator=self.validator,
        )

        self.assertEqual(config, destination / "configs" / "mousehouse.txt")
        self.assertEqual(config.read_text(), "valid\n")
        self.assertFalse(staging.exists())

    def test_stale_cleanup_is_limited_to_selected_model_containers(self) -> None:
        selected = self.stage()
        other = model_build.create_staging_package(self.models, "other")
        published = self.models / "mousehouse"
        published.mkdir()

        removed = model_build.cleanup_stale_staging_packages(
            self.models,
            "mousehouse",
        )

        self.assertEqual(removed, (selected.parent.resolve(),))
        self.assertFalse(selected.parent.exists())
        self.assertTrue(other.parent.is_dir())
        self.assertTrue(published.is_dir())

    def test_stale_cleanup_refuses_matching_symlink(self) -> None:
        outside = Path(self.temp_dir.name) / "outside"
        outside.mkdir()
        link = self.models / ".mousehouse.build-unsafe"
        link.symlink_to(outside, target_is_directory=True)

        with self.assertRaisesRegex(ValueError, "unsafe stale staging entry"):
            model_build.cleanup_stale_staging_packages(
                self.models,
                "mousehouse",
            )

        self.assertTrue(outside.is_dir())
        self.assertTrue(link.is_symlink())

    def test_invalid_staging_never_replaces_existing_package(self) -> None:
        destination = self.models / "mousehouse"
        (destination / "configs").mkdir(parents=True)
        existing = destination / "configs" / "mousehouse.txt"
        existing.write_text("working\n")
        staging = self.stage("invalid\n")

        with self.assertRaisesRegex(ValueError, "invalid staged package"):
            model_build.promote_model_package(
                staging,
                destination,
                config_name="mousehouse.txt",
                overwrite=True,
                validator=self.validator,
            )

        self.assertEqual(existing.read_text(), "working\n")
        self.assertTrue(staging.exists())

    def test_overwrite_false_preserves_existing_package(self) -> None:
        destination = self.models / "mousehouse"
        (destination / "configs").mkdir(parents=True)
        existing = destination / "configs" / "mousehouse.txt"
        existing.write_text("working\n")
        staging = self.stage()

        with self.assertRaisesRegex(FileExistsError, "preserved"):
            model_build.promote_model_package(
                staging,
                destination,
                config_name="mousehouse.txt",
                overwrite=False,
                validator=self.validator,
            )

        self.assertEqual(existing.read_text(), "working\n")
        self.assertTrue(staging.exists())

    def test_overwrite_atomically_exchanges_complete_directories(self) -> None:
        destination = self.models / "mousehouse"
        (destination / "configs").mkdir(parents=True)
        (destination / "configs" / "mousehouse.txt").write_text("working\n")
        staging = self.stage()

        model_build.promote_model_package(
            staging,
            destination,
            config_name="mousehouse.txt",
            overwrite=True,
            validator=self.validator,
        )

        self.assertEqual(
            (destination / "configs" / "mousehouse.txt").read_text(), "valid\n"
        )
        self.assertFalse(staging.exists())

    def test_trtexec_evidence_is_output_bounded(self) -> None:
        executable = Path(self.temp_dir.name) / "fake-trtexec"
        executable.write_text("#!/bin/sh\nprintf 'abcdefghijklmnopqrstuvwxyz'\nexit 0\n")
        executable.chmod(0o755)
        engine = Path(self.temp_dir.name) / "model.engine"
        engine.write_bytes(b"engine")

        evidence = model_build.run_trtexec_validation(
            engine, executable=os.fspath(executable), max_output_bytes=8
        )

        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["output"], "abcdefgh")
        self.assertTrue(evidence["output_truncated"])
        self.assertEqual(evidence["max_output_bytes"], 8)
        self.assertIn("--iterations=1", evidence["command"])
        self.assertIn("--infStreams=1", evidence["command"])
        self.assertNotIn("--streams=1", evidence["command"])

    def test_trtexec_stdout_handle_is_closed_after_success(self) -> None:
        executable = Path(self.temp_dir.name) / "fake-trtexec"
        executable.write_text("#!/bin/sh\nprintf 'ok\\n'\n")
        executable.chmod(0o755)
        engine = Path(self.temp_dir.name) / "model.engine"
        engine.write_bytes(b"engine")
        real_popen = subprocess.Popen
        spawned = []

        def capture_process(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            spawned.append(process)
            return process

        with mock.patch.object(
            model_build.subprocess, "Popen", side_effect=capture_process
        ):
            evidence = model_build.run_trtexec_validation(
                engine, executable=os.fspath(executable)
            )

        self.assertTrue(evidence["passed"])
        self.assertEqual(len(spawned), 1)
        self.assertIsNotNone(spawned[0].stdout)
        self.assertTrue(spawned[0].stdout.closed)

    def test_trtexec_timeout_is_fail_closed(self) -> None:
        executable = Path(self.temp_dir.name) / "slow-trtexec"
        executable.write_text("#!/bin/sh\nsleep 5\n")
        executable.chmod(0o755)
        engine = Path(self.temp_dir.name) / "model.engine"
        engine.write_bytes(b"engine")

        evidence = model_build.run_trtexec_validation(
            engine, executable=os.fspath(executable), timeout_s=0.05
        )

        self.assertFalse(evidence["passed"])
        self.assertTrue(evidence["timed_out"])

    def test_trtexec_rejects_nonfinite_or_unbounded_limits_before_spawn(self) -> None:
        engine = Path(self.temp_dir.name) / "model.engine"
        engine.write_bytes(b"engine")
        invalid = (
            {"timeout_s": math.nan},
            {"timeout_s": math.inf},
            {"timeout_s": model_build.MAX_TRTEXEC_TIMEOUT_S + 1},
            {"max_output_bytes": True},
            {"max_output_bytes": model_build.MAX_TRTEXEC_OUTPUT_BYTES + 1},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                model_build.run_trtexec_validation(engine, **kwargs)


if __name__ == "__main__":
    unittest.main()
