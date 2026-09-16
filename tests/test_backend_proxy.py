from __future__ import annotations

import os
import queue
import socket
import stat
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from squeakview.apps.operator.backend.contracts import LaunchConfig
from squeakview.apps.operator.backend.events import RunPhase
from squeakview.apps.operator.backend.supervisor.protocol import (
    CommandEnvelope,
    EventEnvelope,
    HEARTBEAT_COMMAND_NAME,
    HEARTBEAT_REQUEST_ID,
)
from squeakview.apps.operator.gui import backend_proxy
from squeakview.apps.operator.gui.backend_proxy import (
    SupervisorBackendProxy,
    _backend_event_from_payload,
    _snapshot_from_payload,
    launch_config_from_payload,
    launch_config_payload,
)
from squeakview.common.dashboard import DashboardEvent


class _FakeSupervisor:
    def __init__(self) -> None:
        self.client = mock.Mock()
        self.commands: queue.Queue[CommandEnvelope] = queue.Queue()
        self.events: queue.Queue[EventEnvelope | BaseException] = queue.Queue()
        self.sequence = 0

    def receive(self) -> CommandEnvelope:
        return self.commands.get(timeout=1.0)

    def respond(self, command: CommandEnvelope, result: object) -> None:
        self.events.put(
            EventEnvelope(
                "command_result",
                self.sequence,
                {"ok": True, "result": result, "error": None},
                request_id=command.request_id,
            )
        )
        self.sequence += 1

    def event(self, name: str, payload: dict[str, object]) -> None:
        self.events.put(EventEnvelope(name, self.sequence, payload))
        self.sequence += 1

    def close(self) -> None:
        self.events.put(EOFError("peer closed"))


class _FakeReader:
    def __init__(self, fake: _FakeSupervisor) -> None:
        self.fake = fake

    def receive(self, _connection) -> EventEnvelope:
        item = self.fake.events.get(timeout=2.0)
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeWriter:
    def __init__(self, fake: _FakeSupervisor) -> None:
        self.fake = fake

    def send(self, envelope: CommandEnvelope) -> None:
        self.fake.commands.put(envelope)


class SupervisorBackendProxyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = _FakeSupervisor()
        self.connect = mock.patch.object(
            SupervisorBackendProxy,
            "_connect",
            return_value=self.fake.client,
        )
        self.connect.start()
        self.reader = mock.patch.object(
            backend_proxy,
            "SocketEnvelopeReader",
            return_value=_FakeReader(self.fake),
        )
        self.writer = mock.patch.object(
            backend_proxy,
            "SocketEnvelopeWriter",
            return_value=_FakeWriter(self.fake),
        )
        self.reader.start()
        self.writer.start()

    def tearDown(self) -> None:
        self.connect.stop()
        self.reader.stop()
        self.writer.stop()
        self.fake.close()

    def proxy(self, **kwargs) -> SupervisorBackendProxy:
        return SupervisorBackendProxy(Path("/unused.sock"), mock.Mock(), **kwargs)

    def test_launch_config_wire_round_trip_preserves_paths_and_immutable_bottles(self) -> None:
        source = LaunchConfig(
            ds_cfg=Path("/tmp/config.txt"),
            camera_serials=("one", "two"),
            preview_socket_paths=(Path("/tmp/preview.sock"),),
            bottles={"left": {"initial_g": 4.5}},
        )

        restored = launch_config_from_payload(launch_config_payload(source))

        self.assertEqual(restored, source)
        self.assertEqual(restored.preview_socket_paths, (Path("/tmp/preview.sock"),))

    def test_start_updates_runtime_config_before_returning(self) -> None:
        proxy = self.proxy()
        config = LaunchConfig(preview_enabled=True)
        runtime = LaunchConfig(
            run_dir=Path("/tmp/run-1"),
            preview_socket_paths=(Path("/tmp/preview-1.sock"),),
        )

        def serve() -> None:
            command = self.fake.receive()
            self.assertEqual(command.name, "start_run")
            self.fake.respond(
                command,
                {
                    "started": True,
                    "config": launch_config_payload(runtime),
                    "snapshot": {
                        "schema_version": "1.0",
                        "phase": "recording",
                        "run_dir": "/tmp/run-1",
                        "error": None,
                        "capture_running": True,
                        "finalization_in_progress": False,
                    },
                },
            )

        thread = threading.Thread(target=serve)
        thread.start()
        self.assertTrue(proxy.start_run(config))
        thread.join()

        self.assertEqual(proxy.launch_cfg, runtime)
        self.assertEqual(proxy.state.run_dir, Path("/tmp/run-1"))
        self.assertTrue(proxy.state.inference.is_running())

    def test_clear_jam_command_returns_exact_controller_response(self) -> None:
        proxy = self.proxy()

        def serve() -> None:
            command = self.fake.receive()
            self.assertEqual(command.name, "clear_feeder_jam")
            self.assertEqual(command.payload, {})
            self.fake.respond(command, {"response": "ACK_CLEAR_JAM"})

        thread = threading.Thread(target=serve)
        thread.start()
        try:
            self.assertEqual(proxy.clear_feeder_jam(), "ACK_CLEAR_JAM")
        finally:
            thread.join()

    def test_malformed_start_snapshot_closes_lease_as_uncertain_active(self) -> None:
        proxy = self.proxy()

        def serve() -> None:
            command = self.fake.receive()
            self.fake.respond(
                command,
                {
                    "started": True,
                    "config": launch_config_payload(LaunchConfig()),
                    "snapshot": {
                        "schema_version": "1.0",
                        "phase": "recording",
                        "run_dir": "/tmp/run",
                        "error": None,
                        "capture_running": "yes",
                        "finalization_in_progress": False,
                    },
                },
            )

        thread = threading.Thread(target=serve)
        thread.start()
        with self.assertRaisesRegex(ValueError, "capture_running must be boolean"):
            proxy.start_run(LaunchConfig())
        thread.join()

        self.assertTrue(proxy._closed.is_set())
        self.assertTrue(proxy.state.inference.is_running())
        self.assertTrue(proxy.finalization_in_progress)

    def test_backend_events_reach_compatible_callbacks(self) -> None:
        started = threading.Event()
        subscribed = threading.Event()
        proxy = self.proxy(on_run_started=started.set)
        received = []
        proxy.subscribe(lambda event: (received.append(event), subscribed.set()))

        self.fake.event(
            "backend_event",
            {
                "schema_version": "1.0",
                "type": "phase_changed",
                "phase": "recording",
                "run_dir": "/tmp/run-2",
                "message": None,
                "payload": {"previous_phase": "starting"},
                "host_unix_ns": 123,
            },
        )

        self.assertTrue(started.wait(1.0))
        self.assertTrue(subscribed.wait(1.0))
        self.assertEqual(received[0].phase, RunPhase.RECORDING)
        self.assertEqual(proxy.state.run_dir, Path("/tmp/run-2"))

    def test_snapshot_decoder_rejects_missing_unknown_and_coerced_fields(self) -> None:
        valid = {
            "schema_version": "1.0",
            "phase": "idle",
            "run_dir": None,
            "error": None,
            "capture_running": False,
            "finalization_in_progress": False,
        }
        for mutation in (
            lambda value: value.pop("capture_running"),
            lambda value: value.update(extra=True),
            lambda value: value.update(capture_running=0),
            lambda value: value.update(schema_version=1.0),
        ):
            payload = dict(valid)
            mutation(payload)
            with self.assertRaises(ValueError):
                _snapshot_from_payload(payload)

    def test_backend_event_decoder_requires_exact_versioned_semantics(self) -> None:
        valid = {
            "schema_version": "1.0",
            "type": "phase_changed",
            "phase": "idle",
            "run_dir": None,
            "message": None,
            "payload": {},
            "host_unix_ns": 123,
        }
        self.assertEqual(_backend_event_from_payload(valid).phase, RunPhase.IDLE)
        for mutation in (
            lambda value: value.pop("host_unix_ns"),
            lambda value: value.update(extra=True),
            lambda value: value.update(payload=[]),
            lambda value: value.update(host_unix_ns=True),
        ):
            payload = dict(valid)
            mutation(payload)
            with self.assertRaises(ValueError):
                _backend_event_from_payload(payload)

    def test_dashboard_event_is_strictly_decoded_before_callback(self) -> None:
        received: list[DashboardEvent] = []
        ready = threading.Event()
        proxy = self.proxy(
            ingest_dashboard=lambda event: (received.append(event), ready.set())
        )
        event = DashboardEvent.parse(
            "POKE_START,1000000,2000000,R,1,2,3,4,ON,ok"
        )
        assert event is not None

        self.fake.event("dashboard", event.to_payload())

        self.assertTrue(ready.wait(1.0))
        self.assertEqual(received, [event])

    def test_malformed_dashboard_payload_fails_connection_closed(self) -> None:
        proxy = self.proxy()

        self.fake.event("dashboard", {"line": "legacy raw line"})

        self.assertTrue(proxy._closed.wait(1.0))
        self.assertTrue(proxy._closed.is_set())

    def test_heartbeat_uses_exact_reserved_frame_without_pending_response(self) -> None:
        proxy = self.proxy()

        self.assertTrue(proxy.heartbeat())
        command = self.fake.receive()

        self.assertEqual(command.name, HEARTBEAT_COMMAND_NAME)
        self.assertEqual(command.request_id, HEARTBEAT_REQUEST_ID)
        self.assertEqual(dict(command.payload), {})
        self.assertEqual(proxy._pending, {})

    def test_heartbeat_enqueue_is_nonblocking_and_coalesces_at_one(self) -> None:
        proxy = object.__new__(SupervisorBackendProxy)
        proxy._closed = threading.Event()
        proxy._shutdown_requested = False
        proxy._heartbeats = queue.Queue(maxsize=1)
        proxy._pending = {}

        self.assertTrue(proxy.heartbeat())
        self.assertFalse(proxy.heartbeat())
        self.assertEqual(proxy._heartbeats.qsize(), 1)
        self.assertEqual(proxy._pending, {})

    def test_connection_loss_during_run_remains_fail_closed(self) -> None:
        failed = threading.Event()
        errors: list[str] = []
        proxy = self.proxy(
            on_run_failed=lambda error: (errors.append(error), failed.set())
        )
        self.fake.event(
            "backend_event",
            {
                "type": "phase_changed",
                "schema_version": "1.0",
                "phase": "recording",
                "run_dir": "/tmp/run-3",
                "message": None,
                "payload": {},
                "host_unix_ns": 123,
            },
        )
        self.fake.close()

        self.assertTrue(failed.wait(1.0))
        self.assertIn("connection lost", errors[0])
        self.assertTrue(proxy.state.inference.is_running())
        self.assertTrue(proxy.finalization_in_progress)

    def test_start_send_timeout_closes_transport_and_assumes_remote_active(self) -> None:
        failed = threading.Event()
        errors: list[str] = []
        proxy = self.proxy(
            on_run_failed=lambda error: (errors.append(error), failed.set())
        )
        proxy._writer.send = mock.Mock(side_effect=TimeoutError("send stalled"))

        with self.assertRaisesRegex(TimeoutError, "send stalled"):
            proxy.start_run(LaunchConfig())

        self.assertTrue(failed.wait(1.0))
        self.assertTrue(proxy._closed.is_set())
        self.assertTrue(proxy.state.inference.is_running())
        self.assertTrue(proxy.finalization_in_progress)
        self.assertIn("connection lost", errors[0])

    def test_mutation_response_timeout_closes_lease_and_assumes_remote_active(self) -> None:
        failed = threading.Event()
        errors: list[str] = []
        proxy = self.proxy(
            on_run_failed=lambda error: (errors.append(error), failed.set())
        )

        with self.assertRaisesRegex(TimeoutError, "did not complete"):
            proxy._command("start_run", {}, timeout=0.01)

        self.assertTrue(failed.wait(1.0))
        self.assertTrue(proxy._closed.is_set())
        self.assertTrue(proxy.state.inference.is_running())
        self.assertTrue(proxy.finalization_in_progress)
        self.assertIn("connection lost", errors[0])

    def test_snapshot_uses_explicit_capture_and_finalization_flags(self) -> None:
        proxy = self.proxy()

        def serve() -> None:
            command = self.fake.receive()
            self.fake.respond(
                command,
                {
                    "schema_version": "1.0",
                    "phase": "validating",
                    "run_dir": "/tmp/run-4",
                    "error": None,
                    "capture_running": False,
                    "finalization_in_progress": True,
                },
            )

        thread = threading.Thread(target=serve)
        thread.start()
        snapshot = proxy.snapshot()
        thread.join()

        self.assertEqual(snapshot.phase, RunPhase.VALIDATING)
        self.assertFalse(proxy.state.inference.is_running())
        self.assertTrue(proxy.finalization_in_progress)

    def test_stop_ack_is_async_and_waits_for_terminal_event_without_rpc_timeout(self) -> None:
        proxy = self.proxy()
        proxy._apply_phase(RunPhase.RECORDING, Path("/tmp/long-run"))
        terminal_delay = 0.15

        def serve() -> None:
            command = self.fake.receive()
            self.assertEqual(command.name, "stop_run")
            self.fake.respond(
                command,
                {
                    "accepted": True,
                    "snapshot": {
                        "schema_version": "1.0",
                        "phase": "recording",
                        "run_dir": "/tmp/long-run",
                        "error": None,
                        "capture_running": True,
                        "finalization_in_progress": False,
                    },
                },
            )
            time.sleep(terminal_delay)
            self.fake.event(
                "backend_event",
                {
                    "schema_version": "1.0",
                    "type": "phase_changed",
                    "phase": "finalized",
                    "run_dir": "/tmp/long-run",
                    "message": None,
                    "payload": {"previous_phase": "validating"},
                    "host_unix_ns": 123,
                },
            )

        thread = threading.Thread(target=serve)
        thread.start()
        started = time.monotonic()
        proxy.stop_run()
        elapsed = time.monotonic() - started
        thread.join()

        self.assertGreaterEqual(elapsed, terminal_delay)
        self.assertFalse(proxy._closed.is_set())
        self.assertEqual(proxy.current_snapshot.phase, RunPhase.FINALIZED)
        self.assertFalse(proxy.finalization_in_progress)

    def test_shutdown_response_then_eof_is_not_reported_as_connection_loss(self) -> None:
        emit = mock.Mock()
        proxy = SupervisorBackendProxy(Path("/unused.sock"), emit)

        def serve() -> None:
            command = self.fake.receive()
            self.assertEqual(command.name, "shutdown_server")
            self.fake.respond(
                command,
                {
                    "snapshot": {
                        "schema_version": "1.0",
                        "phase": "idle",
                        "run_dir": None,
                        "error": None,
                        "capture_running": False,
                        "finalization_in_progress": False,
                    }
                },
            )
            self.fake.close()

        thread = threading.Thread(target=serve)
        thread.start()
        proxy.shutdown()
        thread.join()

        self.assertFalse(
            any("connection lost" in str(call.args[0]) for call in emit.call_args_list)
        )


class SupervisorSocketValidationTest(unittest.TestCase):
    def test_connect_rejects_non_socket_and_nonprivate_socket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            regular = Path(tmp) / "not-a-socket"
            regular.touch()
            with self.assertRaisesRegex(RuntimeError, "not a Unix socket"):
                SupervisorBackendProxy._connect(regular, 0.0)

        metadata = mock.Mock(st_mode=stat.S_IFSOCK | 0o666, st_uid=os.getuid())
        connection = mock.Mock()
        with (
            mock.patch.object(Path, "lstat", return_value=metadata),
            mock.patch.object(socket, "socket", return_value=connection),
            self.assertRaisesRegex(RuntimeError, "not private"),
        ):
            SupervisorBackendProxy._connect(Path("/private/operator.sock"), 0.0)
        connection.connect.assert_not_called()

    def test_connect_accepts_private_same_user_socket(self) -> None:
        metadata = mock.Mock(st_mode=stat.S_IFSOCK | 0o600, st_uid=os.getuid())
        connection = mock.Mock()
        connection.getsockopt.return_value = struct.pack("3i", 123, os.getuid(), 456)
        path = Path("/private/operator.sock")
        with (
            mock.patch.object(Path, "lstat", return_value=metadata),
            mock.patch.object(socket, "socket", return_value=connection),
        ):
            client = SupervisorBackendProxy._connect(path, 0.0)

        self.assertIs(client, connection)
        connection.connect.assert_called_once_with(str(path))
        self.assertGreaterEqual(connection.settimeout.call_count, 2)
        self.assertEqual(
            connection.settimeout.call_args.args[0],
            backend_proxy._SOCKET_IO_TIMEOUT_SECONDS,
        )


class BackendFactoryPolicyTest(unittest.TestCase):
    def test_supervisor_socket_is_mandatory_without_explicit_dev_override(self) -> None:
        from squeakview.apps.operator.gui.main_window import _production_backend_factory

        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "squeakview.sh"):
                _production_backend_factory(mock.Mock())

    def test_explicit_dev_override_is_visible_and_uses_inprocess_backend(self) -> None:
        from squeakview.apps.operator.gui import main_window

        backend = mock.sentinel.backend
        emit = mock.Mock()
        project = mock.Mock()
        project.paths.root = Path("/tmp/test-project")
        session = mock.Mock(project=project)
        with (
            mock.patch.dict(
                os.environ,
                {backend_proxy.ALLOW_INPROCESS_BACKEND_ENV: "1"},
                clear=True,
            ),
            mock.patch.object(
                main_window, "OperatorBackend", return_value=backend
            ) as backend_constructor,
            mock.patch.object(
                main_window, "project_from_environment", return_value=project
            ),
            mock.patch.object(
                main_window.ProjectSession, "open", return_value=session
            ),
            mock.patch.object(main_window.UserPaths, "discover") as user_paths,
            mock.patch.object(main_window.AppPaths, "discover"),
            mock.patch.object(main_window, "RuntimeContext"),
        ):
            result = main_window._production_backend_factory(emit)

        self.assertIs(result, backend)
        self.assertIn("development mode", emit.call_args.args[0])
        self.assertEqual(
            backend_constructor.call_args.kwargs["acquisition_owner"],
            "in_process_dev",
        )
        user_paths.return_value.ensure.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
