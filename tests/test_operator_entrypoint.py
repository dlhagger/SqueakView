from __future__ import annotations

import io
import os
import signal
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import squeakview_gui
from squeakview.apps.operator import main as operator_main
from squeakview.apps.operator.backend.supervisor import __main__ as supervisor_main
from squeakview.common.log_mirror import DEFAULT_LOG_MAX_BYTES


class OperatorSignalLifecycleTest(unittest.TestCase):
    def test_terminal_signals_schedule_guarded_window_close(self) -> None:
        app = mock.Mock()
        window = mock.Mock()
        registrations: dict[signal.Signals, object] = {}

        with (
            mock.patch.object(
                operator_main.signal,
                "signal",
                side_effect=lambda signum, callback: registrations.setdefault(
                    signum, callback
                ),
            ),
            mock.patch.object(operator_main.QtCore.QTimer, "singleShot") as single_shot,
        ):
            operator_main._install_close_signal_handlers(app, lambda: window)
            registrations[signal.SIGTERM](signal.SIGTERM, None)

        single_shot.assert_called_once_with(0, window.close)
        app.quit.assert_not_called()

    def test_signal_before_window_creation_schedules_application_quit(self) -> None:
        app = mock.Mock()
        registrations: dict[signal.Signals, object] = {}

        with (
            mock.patch.object(
                operator_main.signal,
                "signal",
                side_effect=lambda signum, callback: registrations.setdefault(
                    signum, callback
                ),
            ),
            mock.patch.object(operator_main.QtCore.QTimer, "singleShot") as single_shot,
        ):
            operator_main._install_close_signal_handlers(app, lambda: None)
            registrations[signal.SIGINT](signal.SIGINT, None)

        single_shot.assert_called_once_with(0, app.quit)

    def test_installs_hangup_handler_when_platform_supports_it(self) -> None:
        app = mock.Mock()
        with mock.patch.object(operator_main.signal, "signal") as register:
            operator_main._install_close_signal_handlers(app, lambda: None)

        registered = {call.args[0] for call in register.call_args_list}
        self.assertIn(signal.SIGINT, registered)
        self.assertIn(signal.SIGTERM, registered)
        if hasattr(signal, "SIGHUP"):
            self.assertIn(signal.SIGHUP, registered)

    def test_gui_construction_failure_exits_nonzero_instead_of_hanging(self) -> None:
        app = mock.Mock()
        splash = mock.Mock()
        holder = {}

        with mock.patch("builtins.print") as output:
            launched = operator_main._launch_main_window(
                app,
                holder,
                splash=splash,
                app_icon=None,
                window_factory=mock.Mock(side_effect=RuntimeError("socket unavailable")),
            )

        self.assertFalse(launched)
        self.assertEqual(holder, {})
        splash.close.assert_called_once_with()
        app.exit.assert_called_once_with(1)
        self.assertIn("socket unavailable", output.call_args.args[0])


class OperatorLauncherTest(unittest.TestCase):
    def test_direct_gui_refuses_log_output_inside_application_checkout(self) -> None:
        root = Path(__file__).resolve().parents[1]
        forbidden = root / ".forbidden-runtime-log"
        stderr = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            mock.patch.dict(
                os.environ,
                {
                    "HOME": temp_dir,
                    "XDG_CONFIG_HOME": str(Path(temp_dir) / "config"),
                    "XDG_STATE_HOME": str(Path(temp_dir) / "state"),
                    "XDG_RUNTIME_DIR": str(Path(temp_dir) / "runtime"),
                    squeakview_gui.LOG_ENV: str(forbidden),
                },
                clear=True,
            ),
            redirect_stderr(stderr),
        ):
            result = squeakview_gui.main()

        self.assertEqual(result, 1)
        self.assertIn("outside the application checkout", stderr.getvalue())
        self.assertFalse(forbidden.exists())

    def test_launcher_detaches_only_the_durable_supervisor(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (root / "squeakview.sh").read_text()

        supervisor = "-m squeakview.apps.operator.backend.supervisor"
        gui = '--gui-command "$PYTHON_BIN" "$ROOT/squeakview_gui.py"'
        self.assertIn(supervisor, script)
        self.assertIn('--project "$PROJECT_PATH"', script)
        self.assertIn("squeakview.apps.project_launcher", script)
        self.assertIn(gui, script)
        self.assertLess(script.index(supervisor), script.index(gui))
        self.assertNotIn("SQUEAKVIEW_ALLOW_INPROCESS_BACKEND", script)
        self.assertIn("SQUEAKVIEW_SUPERVISOR_LOGFILE", script)
        self.assertIn("SQUEAKVIEW_LAUNCH_STATUS_FILE", script)
        self.assertIn("GUI_READY", script)
        self.assertIn("SQUEAKVIEW_LAUNCH_TIMEOUT_S", script)
        self.assertIn('assert_external_log_path "Launch log directory"', script)
        self.assertIn('"$ROOT"|"$ROOT"/*', script)
        self.assertIn('"$PROJECT_PATH"|"$PROJECT_PATH"/*', script)
        self.assertNotIn('>"$SUPERVISOR_LOG_PATH"', script)

    def test_launcher_returns_nonzero_when_project_selection_fails(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            temp_dir = Path(tmp)
            fake_python = temp_dir / "python"
            fake_python.write_text("#!/bin/sh\nexit 7\n")
            fake_python.chmod(0o700)
            completed = subprocess.run(
                ["bash", str(root / "squeakview.sh")],
                cwd=root,
                env={
                    **os.environ,
                    "PYTHON_BIN": str(fake_python),
                    "SQUEAKVIEW_LAUNCH_LOG_DIR": str(temp_dir / "logs"),
                    "SQUEAKVIEW_LAUNCH_TIMEOUT_S": "2",
                },
                capture_output=True,
                text=True,
                timeout=5,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("No SqueakView project was selected", completed.stderr)

    def test_supervisor_uses_the_bounded_log_mirror(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "supervisor.log"
            console = io.StringIO()
            with (
                mock.patch.dict(
                    supervisor_main.os.environ,
                    {supervisor_main.SUPERVISOR_LOG_ENV: str(path)},
                    clear=True,
                ),
                mock.patch.object(supervisor_main.sys, "stdout", console),
                mock.patch.object(supervisor_main.sys, "stderr", console),
            ):
                mirror = supervisor_main._install_log_mirror()
                self.assertIsNotNone(mirror)
                assert mirror is not None
                mirror.write("supervisor diagnostic\n")
                mirror.flush()
                self.assertEqual(mirror._max_bytes, DEFAULT_LOG_MAX_BYTES)
                mirror._fh.close()

            self.assertEqual(path.read_text(), "supervisor diagnostic\n")

    def test_supervisor_refuses_log_output_inside_application_checkout(self) -> None:
        root = Path(__file__).resolve().parents[1]
        forbidden = root / ".forbidden-supervisor-log"
        with (
            mock.patch.dict(
                supervisor_main.os.environ,
                {supervisor_main.SUPERVISOR_LOG_ENV: str(forbidden)},
                clear=True,
            ),
            self.assertRaisesRegex(ValueError, "outside the application checkout"),
        ):
            supervisor_main._install_log_mirror()

        self.assertFalse(forbidden.exists())


class JetsonSetupScriptTest(unittest.TestCase):
    def test_setup_is_one_run_build_group_and_reboot_workflow(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (root / "scripts" / "setup_jetson.sh").read_text()

        self.assertIn('usermod -aG dialout "$TARGET_USER"', script)
        self.assertNotIn("usermod -aG dialout $USER", script)
        self.assertIn("Reboot this Jetson before starting SqueakView", script)
        self.assertIn("gstreamer1.0-tools", script)
        self.assertIn("gstreamer1.0-plugins-good", script)
        self.assertIn("gstreamer1.0-plugins-bad", script)
        self.assertIn("gstreamer1.0-plugins-ugly", script)
        self.assertIn('bash "$ROOT/scripts/build_native.sh"', script)
        self.assertIn('runuser -u "$TARGET_USER"', script)
        self.assertIn("keep native build outputs owned", script)
        self.assertIn('TARGET_USER_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"', script)
        self.assertIn('SQUEAKVIEW_PROJECTS_DIR:-$TARGET_USER_HOME/Documents/SqueakView Projects', script)
        self.assertIn("Project parent is a dangling symbolic link", script)
        self.assertIn("Project parent must be outside the application checkout", script)
        self.assertIn("Project parent may not contain the application checkout", script)
        self.assertIn("Existing project parent left unchanged", script)
        self.assertIn("Created project parent", script)

    def test_setup_verifies_both_native_runtime_outputs_and_contracts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (root / "scripts" / "build_native.sh").read_text()

        self.assertIn("gstflirspinsrc.so", script)
        self.assertIn("libnvdsinfer_custom_impl_Yolo.so", script)
        self.assertIn("ldd \"$output\"", script)
        self.assertIn("NvDsInferParseYolo26Pose", script)
        self.assertIn("capture-log-path", script)


class PreflightScriptTest(unittest.TestCase):
    def test_model_selection_is_project_rooted_without_dead_path_resolver(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = (root / "scripts" / "preflight.sh").read_text()

        self.assertIn('PROJECT_ROOT="${SQUEAKVIEW_PROJECT:-}"', script)
        self.assertIn(
            'CFG="$PROJECT_ROOT/models/${SQUEAKVIEW_MODEL_NAME}/configs/'
            '${SQUEAKVIEW_MODEL_NAME}.txt"',
            script,
        )
        self.assertNotIn("resolve_repo_path", script)


class DetachedLaunchReadinessTest(unittest.TestCase):
    def test_status_acknowledgement_requires_private_owned_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "status"
            path.write_text("")
            path.chmod(0o600)
            with mock.patch.dict(
                os.environ,
                {supervisor_main.LAUNCH_STATUS_ENV: str(path)},
                clear=True,
            ):
                supervisor_main._mark_gui_ready()
            self.assertEqual(path.read_text(), "GUI_READY\n")

            path.chmod(0o644)
            with (
                mock.patch.dict(
                    os.environ,
                    {supervisor_main.LAUNCH_STATUS_ENV: str(path)},
                    clear=True,
                ),
                self.assertRaisesRegex(RuntimeError, "private regular file"),
            ):
                supervisor_main._mark_gui_ready()


if __name__ == "__main__":
    unittest.main()
