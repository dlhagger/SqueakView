from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from squeakview.apps.operator.backend import preflight


class PreflightServiceTests(unittest.TestCase):
    def test_missing_preflight_script_fails_closed(self) -> None:
        result = preflight.run_preflight(
            preflight.PreflightRequest(),
            workspace=Path("/missing"),
            python_bin="python3",
            emit=lambda _message: None,
            environ={},
        )

        self.assertFalse(result.passed)
        self.assertFalse(result.skipped)
        self.assertIn("Restore scripts/preflight.sh", result.message)

    def test_skip_is_explicit_and_does_not_spawn(self) -> None:
        logs: list[str] = []
        with mock.patch.object(preflight.subprocess, "run") as run:
            result = preflight.run_preflight(
                preflight.PreflightRequest(),
                workspace=Path("/missing"),
                python_bin="python3",
                emit=logs.append,
                environ={"SQUEAKVIEW_SKIP_PREFLIGHT": "1"},
            )
        self.assertTrue(result.passed)
        self.assertTrue(result.skipped)
        run.assert_not_called()

    def test_debug_profile_requires_stable_nvidia_probe_modules(self) -> None:
        with mock.patch.object(
            preflight.debug_instrumentation,
            "validate_probe_modules",
            side_effect=RuntimeError("probe identity unavailable"),
        ):
            result = preflight.run_preflight(
                preflight.PreflightRequest(),
                workspace=Path("/missing"),
                python_bin="python3",
                emit=lambda _message: None,
                environ={"SQUEAKVIEW_DEEPSTREAM_DEBUG_PROFILE": "1"},
            )

        self.assertFalse(result.passed)
        self.assertIn("probe identity unavailable", result.message)

    def test_debug_probe_identities_are_retained_in_evidence(self) -> None:
        identities = {
            "measure_latency_probe": {
                "available": True,
                "size_bytes": 10,
                "sha256": "a" * 64,
            },
            "measure_fps_probe": {
                "available": True,
                "size_bytes": 11,
                "sha256": "b" * 64,
            },
        }
        with (
            mock.patch.object(
                preflight.debug_instrumentation,
                "validate_probe_modules",
                return_value=identities,
            ),
            mock.patch.dict(
                "os.environ", {"SQUEAKVIEW_SKIP_PREFLIGHT": "1"}, clear=True
            ),
        ):
            result = preflight.run_preflight(
                preflight.PreflightRequest(),
                workspace=Path("/missing"),
                python_bin="python3",
                emit=lambda _message: None,
                environ={
                    "SQUEAKVIEW_SKIP_PREFLIGHT": "1",
                    "SQUEAKVIEW_DEEPSTREAM_DEBUG_PROFILE": "1",
                },
            )

        self.assertEqual(
            preflight.evidence_snapshot(result)["deepstream_debug_probes"],
            identities,
        )

    def test_request_is_translated_to_bounded_subprocess_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "scripts").mkdir()
            (workspace / "scripts" / "preflight.sh").write_text("#!/bin/bash\n")
            completed = preflight._BoundedProcessResult(0, "ok\n")
            with mock.patch.object(
                preflight, "_run_bounded_preflight", return_value=completed
            ) as run:
                result = preflight.run_preflight(
                    preflight.PreflightRequest(
                        capture_backend="flir_direct",
                        inference_enabled=False,
                        ds_cfg=Path("/tmp/model.txt"),
                        serial_enabled=True,
                        serial_port="/dev/ttyACM7",
                    ),
                    workspace=workspace,
                    python_bin="/usr/bin/python3",
                    emit=lambda _message: None,
                    environ={},
                )

        self.assertTrue(result.passed)
        self.assertEqual(run.call_args.kwargs["timeout_s"], 240.0)
        child_env = run.call_args.kwargs["env"]
        self.assertEqual(child_env["INFERENCE_ENABLED"], "0")
        self.assertEqual(child_env["DS_CFG"], "/tmp/model.txt")
        self.assertEqual(child_env["SERIAL_ENABLED"], "1")
        self.assertEqual(child_env["SERIAL_PORT"], "/dev/ttyACM7")

    def test_truncated_or_timed_out_preflight_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "scripts").mkdir()
            (workspace / "scripts" / "preflight.sh").write_text("#!/bin/bash\n")
            for completed in (
                preflight._BoundedProcessResult(0, "many lines", truncated=True),
                preflight._BoundedProcessResult(-15, "partial", timed_out=True),
            ):
                with mock.patch.object(
                    preflight, "_run_bounded_preflight", return_value=completed
                ):
                    result = preflight.run_preflight(
                        preflight.PreflightRequest(inference_enabled=False),
                        workspace=workspace,
                        python_bin="python3",
                        emit=lambda _message: None,
                        environ={},
                    )
                self.assertFalse(result.passed)
                self.assertIn("[FAIL]", result.output)
                self.assertEqual(result.output_truncated, completed.truncated)
                self.assertEqual(result.timed_out, completed.timed_out)

    def test_bounded_runner_drains_but_caps_child_output(self) -> None:
        result = preflight._run_bounded_preflight(
            [
                "/usr/bin/python3",
                "-c",
                f"import sys; sys.stdout.write('x' * {preflight.MAX_PREFLIGHT_OUTPUT_BYTES + 17})",
            ],
            cwd="/tmp",
            env={},
            timeout_s=5.0,
        )

        self.assertEqual(result.returncode, 0)
        self.assertTrue(result.truncated)
        self.assertEqual(len(result.output.encode()), preflight.MAX_PREFLIGHT_OUTPUT_BYTES)

    def test_bounded_runner_closes_stdout_handle_after_success(self) -> None:
        real_popen = subprocess.Popen
        spawned = []

        def capture_process(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            spawned.append(process)
            return process

        with mock.patch.object(
            preflight.subprocess, "Popen", side_effect=capture_process
        ):
            result = preflight._run_bounded_preflight(
                ["/usr/bin/python3", "-c", "print('ok')"],
                cwd="/tmp",
                env={},
                timeout_s=5.0,
            )

        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(spawned), 1)
        self.assertIsNotNone(spawned[0].stdout)
        self.assertTrue(spawned[0].stdout.closed)

    def test_bounded_runner_streams_complete_output_lines(self) -> None:
        lines: list[str] = []
        result = preflight._run_bounded_preflight(
            [
                "/usr/bin/python3",
                "-c",
                "import sys; print('first', flush=True); print('second', flush=True)",
            ],
            cwd="/tmp",
            env={},
            timeout_s=5.0,
            on_line=lines.append,
        )

        self.assertEqual(result.returncode, 0)
        self.assertEqual(lines, ["first", "second"])

    def test_bounded_runner_times_out_its_owned_process_group(self) -> None:
        result = preflight._run_bounded_preflight(
            ["/usr/bin/python3", "-c", "import time; time.sleep(10)"],
            cwd="/tmp",
            env={},
            timeout_s=0.05,
        )

        self.assertTrue(result.timed_out)
        self.assertNotEqual(result.returncode, 0)

    def test_ffmpeg_failure_is_actionable(self) -> None:
        self.assertIn(
            "sudo apt install ffmpeg",
            preflight.failure_message("[FAIL] FFmpeg/ffprobe is not installed."),
        )

    def test_unknown_failure_uses_authoritative_generic_warning(self) -> None:
        self.assertEqual(
            preflight.failure_message("[FAIL] something else"),
            preflight.GENERIC_FAILURE,
        )

    def test_serial_permission_failure_requires_reboot(self) -> None:
        message = preflight.failure_message(
            "[FAIL] This login session does not have effective dialout access."
        )
        self.assertIn("reboot", message.lower())
        self.assertIn("setup_jetson.sh", message)

    def test_missing_serial_device_is_actionable(self) -> None:
        message = preflight.failure_message(
            "[FAIL] Configured serial controller port is not present: /dev/ttyACM0"
        )
        self.assertIn("/dev/ttyACM*", message)

    def test_evidence_snapshot_is_structured_and_hashes_bounded_output(self) -> None:
        output = (
            "[PASS] ffprobe is installed (/usr/bin/ffprobe)\n"
            "[PASS] DeepStream new nvstreammux VIC/NVMM path works\n"
            "[PASS] Jetson H.264 full-decode validation path works "
            "(decoded 1 frame(s) with h264_nvv4l2dec)\n"
            "[PASS] Automatic desktop suspend on AC power is disabled"
        )

        evidence = preflight.evidence_snapshot(
            preflight.PreflightResult(True, "Preflight passed", output)
        )

        self.assertTrue(evidence["passed"])
        self.assertEqual(evidence["schema_version"], "3.0")
        self.assertTrue(evidence["ffprobe_available"])
        self.assertTrue(evidence["video_decode_validated"])
        self.assertTrue(evidence["new_streammux_validated"])
        self.assertTrue(evidence["automatic_suspend_disabled"])
        self.assertFalse(evidence["output_truncated"])
        self.assertFalse(evidence["timed_out"])
        self.assertEqual(evidence["output_size_bytes"], len(output.encode()))
        self.assertEqual(len(evidence["output_sha256"]), 64)

    def test_decoder_self_test_failure_is_actionable(self) -> None:
        message = preflight.failure_message(
            "[FAIL] Jetson H.264 full-decode validation path failed: decoder error"
        )

        self.assertIn("full-upgrade", message)
        self.assertIn("Do not begin scientific acquisition", message)

    def test_new_streammux_failure_is_actionable(self) -> None:
        message = preflight.failure_message(
            "[FAIL] DeepStream new nvstreammux VIC/NVMM self-test failed."
        )

        self.assertIn("Stream multiplexer 2", message)
        self.assertIn("full-upgrade", message)


if __name__ == "__main__":
    unittest.main()
