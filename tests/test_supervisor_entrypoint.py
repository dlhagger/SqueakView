from __future__ import annotations

import importlib
import io
import signal
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock


entrypoint = importlib.import_module(
    "squeakview.apps.operator.backend.supervisor.__main__"
)


class SupervisorEntrypointTests(unittest.TestCase):
    def test_posix_handler_only_sets_event(self) -> None:
        requested = threading.Event()
        registrations = {}
        with mock.patch.object(
            entrypoint.signal,
            "signal",
            side_effect=lambda signum, callback: registrations.setdefault(
                signum, callback
            ),
        ):
            entrypoint._install_signal_handlers(requested)

        registrations[signal.SIGTERM](signal.SIGTERM, None)

        self.assertTrue(requested.is_set())

    def test_installs_bounded_supervisor_log_from_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "supervisor.log"
            stream = io.StringIO()
            with (
                mock.patch.dict(
                    "os.environ", {entrypoint.SUPERVISOR_LOG_ENV: str(path)}
                ),
                mock.patch.object(entrypoint.sys, "stdout", stream),
                mock.patch.object(entrypoint.sys, "stderr", stream),
            ):
                mirror = entrypoint._install_log_mirror()
                print("supervisor evidence")
                mirror.close()

            self.assertIn("supervisor evidence", path.read_text())

    def test_explicit_gui_argv_is_forwarded_without_a_shell(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            server = mock.Mock()
            server.serve.return_value = 0
            server.last_error = None
            server.last_warning = None
            socket_path = Path(temp_dir) / "operator.sock"
            mirror = mock.Mock()
            with (
                mock.patch.object(entrypoint, "_install_log_mirror", return_value=mirror),
                mock.patch.object(entrypoint, "SupervisorServer", return_value=server),
                mock.patch.object(entrypoint.signal, "signal"),
            ):
                result = entrypoint.main(
                    [
                        "--socket",
                        str(socket_path),
                        "--gui-command",
                        "python3",
                        "squeakview_gui.py",
                    ]
                )

            self.assertEqual(result, 0)
            server.serve.assert_called_once_with(
                ["python3", "squeakview_gui.py"],
                on_gui_ready=entrypoint._mark_gui_ready,
            )
            mirror.close.assert_called_once_with()

    def test_early_server_initialization_failure_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mirror = mock.Mock()
            with (
                mock.patch.object(entrypoint, "_install_log_mirror", return_value=mirror),
                mock.patch.object(
                    entrypoint,
                    "SupervisorServer",
                    side_effect=RuntimeError("bad socket"),
                ),
            ):
                result = entrypoint.main(
                    [
                        "--socket",
                        str(Path(temp_dir) / "operator.sock"),
                        "--gui-command",
                        "gui",
                    ]
                )

            self.assertEqual(result, 1)
            mirror.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
