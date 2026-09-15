from __future__ import annotations

import os
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from squeakview.apps.operator.backend.contracts import RunRequest
from squeakview.apps.operator.backend.events import RunPhase
from squeakview.apps.operator.backend.manager import OperatorBackend
from squeakview.apps.operator.backend.supervisor.protocol import (
    CommandEnvelope,
    EventEnvelope,
    HEARTBEAT_COMMAND_NAME,
    HEARTBEAT_REQUEST_ID,
    MAX_FRAME_BYTES,
    ProtocolError,
    encode_envelope,
)
from squeakview.apps.operator.backend.supervisor.server import (
    MAX_MUTATION_REQUESTS,
    SupervisorServer,
    _json_value,
)
from squeakview.common.dashboard import DashboardEvent


class FakeBackend:
    def __init__(self, emit, ingest, *, acquisition_owner) -> None:
        self.emit = emit
        self.ingest = ingest
        self.acquisition_owner = acquisition_owner
        self.subscriber = None
        self.phase = RunPhase.IDLE
        self.state = SimpleNamespace(run_dir=None, inference=None)
        self.launch_cfg = RunRequest(fps=30, serial_enabled=False)
        self.finalization_in_progress = False
        self.cancelled = threading.Event()
        self.start_entered = threading.Event()
        self.start_release = threading.Event()
        self.stop_entered = threading.Event()
        self.stop_release = threading.Event()
        self.block_start = False
        self.block_stop = False
        self.abort_calls: list[str] = []
        self.saved = 0
        self.saved_bottles = None

    def subscribe(self, callback) -> None:
        self.subscriber = callback

    def snapshot(self):
        capture = self.state.inference
        return SimpleNamespace(
            phase=self.phase,
            run_dir=self.state.run_dir,
            error=None,
            capture_running=bool(capture and capture.is_running()),
            finalization_in_progress=self.finalization_in_progress,
        )

    def start_run(self, config) -> bool:
        self.launch_cfg = config
        self.start_entered.set()
        if self.block_start:
            self.start_release.wait()
        if self.cancelled.is_set():
            return False
        self.phase = RunPhase.RECORDING
        return True

    def stop_run(self) -> None:
        self.phase = RunPhase.STOPPING
        self.finalization_in_progress = True
        self.stop_entered.set()
        if self.block_stop:
            self.stop_release.wait()
        self.finalization_in_progress = False
        self.phase = RunPhase.FINALIZED

    def shutdown(self) -> None:
        self.stop_run()

    def cancel_operator_lease(self) -> None:
        self.cancelled.set()
        self.start_release.set()

    def abort_run(self, error: str) -> bool:
        self.abort_calls.append(error)
        if self.phase == RunPhase.STOPPING and self.finalization_in_progress:
            # The real backend serializes abort/finalize through its
            # finalization lock. Model that ownership in this lightweight fake.
            self.stop_release.wait()
            return False
        if self.phase in {RunPhase.CREATED, RunPhase.STARTING, RunPhase.RECORDING}:
            self.phase = RunPhase.FAILED
            return True
        return False

    def save_bottle_measurements(self, bottles, run_dir=None):
        self.saved += 1
        self.saved_bottles = bottles
        return {"complete": bool(bottles), "run_dir": str(run_dir) if run_dir else None}


class SupervisorServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.socket_path = Path(self.temp_dir.name) / "operator.sock"
        self.server = SupervisorServer(self.socket_path, backend_factory=FakeBackend)
        self.backend: FakeBackend = self.server.backend

    def tearDown(self) -> None:
        self.server._stopping.set()
        self.backend.start_release.set()
        self.backend.stop_release.set()
        if self.server._loss_thread is not None:
            self.server._loss_thread.join(timeout=2)
        if self.server._async_stop_thread is not None:
            self.server._async_stop_thread.join(timeout=2)
        self.temp_dir.cleanup()

    def _worker(self) -> threading.Thread:
        worker = threading.Thread(target=self.server._command_worker, daemon=False)
        worker.start()
        return worker

    @staticmethod
    def _config(**updates) -> dict:
        values = dict(_json_value(RunRequest(fps=30, serial_enabled=False)))
        values.update(updates)
        return values

    def test_start_result_returns_effective_runtime_configuration(self) -> None:
        self.assertEqual(self.backend.acquisition_owner, "durable_supervisor")
        request = CommandEnvelope(
            "start_run",
            "start-1",
            {"config": self._config(preview_enabled=False)},
        )

        result, stop = self.server._dispatch(request)

        self.assertFalse(stop)
        self.assertTrue(result["started"])
        self.assertEqual(result["config"]["fps"], 30)
        self.assertEqual(result["config"]["preview_socket_paths"], [])

    def test_run_request_decoder_rejects_missing_and_wrong_scalar_types(self) -> None:
        missing = self._config()
        missing.pop("fps")
        wrong_bool = self._config(trigger_on=1)

        with self.assertRaisesRegex(ValueError, "missing fields: fps"):
            self.server._dispatch(CommandEnvelope("start_run", "a", missing))
        with self.assertRaisesRegex(ValueError, "trigger_on must be boolean"):
            self.server._dispatch(CommandEnvelope("start_run", "b", wrong_bool))

    def test_gui_loss_cancels_blocked_pre_created_start_and_joins_worker(self) -> None:
        self.backend.block_start = True
        worker = self._worker()
        self.server._commands.put(
            CommandEnvelope("start_run", "start-1", self._config())
        )
        self.assertTrue(self.backend.start_entered.wait(1))
        self.assertEqual(self.backend.phase, RunPhase.IDLE)

        self.server._signal_client_loss()
        self.server._loss_thread.join(2)
        worker.join(2)

        self.assertFalse(worker.is_alive())
        self.assertTrue(self.backend.cancelled.is_set())
        self.assertEqual(self.backend.abort_calls, ["operator GUI lost"])
        self.assertNotEqual(self.backend.phase, RunPhase.RECORDING)

    def test_gui_loss_waits_for_in_progress_finalization(self) -> None:
        self.backend.phase = RunPhase.RECORDING
        self.backend.block_stop = True
        worker = self._worker()
        self.server._commands.put(CommandEnvelope("stop_run", "stop-1"))
        self.assertTrue(self.backend.stop_entered.wait(1))

        self.server._signal_client_loss()
        time.sleep(0.02)
        self.assertTrue(self.server._loss_thread.is_alive())
        self.backend.stop_release.set()
        self.server._loss_thread.join(2)
        self.server._async_stop_thread.join(2)
        worker.join(2)

        self.assertFalse(worker.is_alive())
        self.assertEqual(self.backend.phase, RunPhase.FINALIZED)
        self.assertEqual(self.backend.abort_calls, ["operator GUI lost"])

    def test_stop_command_acknowledges_while_long_finalizer_is_still_running(self) -> None:
        self.backend.phase = RunPhase.RECORDING
        self.backend.block_stop = True

        started = time.monotonic()
        result, stop = self.server._dispatch(
            CommandEnvelope("stop_run", "stop-async")
        )
        elapsed = time.monotonic() - started

        self.assertFalse(stop)
        self.assertTrue(result["accepted"])
        self.assertLess(elapsed, 0.5)
        self.assertTrue(self.backend.stop_entered.wait(1))
        self.assertTrue(self.server._async_stop_thread.is_alive())

        duplicate, _ = self.server._dispatch(
            CommandEnvelope("stop_run", "stop-async-duplicate")
        )
        self.assertFalse(duplicate["accepted"])

        self.backend.stop_release.set()
        self.server._async_stop_thread.join(2)
        self.assertEqual(self.backend.phase, RunPhase.FINALIZED)

    def test_blocked_finalizer_does_not_block_stop_ack_or_progress_snapshot(self) -> None:
        self.backend.phase = RunPhase.RECORDING
        self.backend.block_stop = True
        responses: dict[str, dict] = {}
        response_ready = threading.Event()

        def enqueue(name, payload, **kwargs):
            if name == "command_result":
                responses[str(kwargs.get("request_id"))] = dict(payload)
                response_ready.set()
            return True

        with mock.patch.object(self.server, "_enqueue", side_effect=enqueue):
            worker = self._worker()
            self.server._commands.put(CommandEnvelope("stop_run", "slow-stop"))
            self.assertTrue(self.backend.stop_entered.wait(1))
            self.assertTrue(response_ready.wait(1))
            self.assertTrue(responses["slow-stop"]["ok"])
            self.assertTrue(responses["slow-stop"]["result"]["accepted"])

            response_ready.clear()
            self.server._commands.put(CommandEnvelope("snapshot", "during-stop"))
            self.assertTrue(response_ready.wait(1))
            self.assertEqual(
                responses["during-stop"]["result"]["phase"], "stopping"
            )
            self.assertTrue(
                responses["during-stop"]["result"]["finalization_in_progress"]
            )

            self.backend.stop_release.set()
            self.server._async_stop_thread.join(2)
            self.server._stopping.set()
            worker.join(2)

        self.assertFalse(worker.is_alive())
        self.assertEqual(self.backend.phase, RunPhase.FINALIZED)

    def test_duplicate_mutation_request_replays_without_reexecution(self) -> None:
        worker = self._worker()
        command = CommandEnvelope(
            "save_bottle_measurements", "save-1", {"bottles": {"a": 1}}
        )
        self.server._commands.put(command)
        self.server._commands.put(command)
        deadline = time.monotonic() + 1
        while len(self.server._mutation_ledger) < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.server._stopping.set()
        worker.join(2)

        self.assertEqual(self.backend.saved, 1)

    def test_bottle_command_preserves_nested_values_from_frozen_ipc_payload(self) -> None:
        command = CommandEnvelope(
            "save_bottle_measurements",
            "save-bottles",
            {
                "bottles": {
                    "left": {
                        "fluid": "water",
                        "initial_weight_g": 443.0,
                        "final_weight_g": 422.0,
                    },
                    "right": {
                        "fluid": "ethanol",
                        "initial_weight_g": 321.0,
                        "final_weight_g": 232.0,
                    },
                }
            },
        )

        result, stop = self.server._dispatch(command)

        self.assertFalse(stop)
        self.assertTrue(result["complete"])
        self.assertEqual(self.backend.saved_bottles["left"]["fluid"], "water")
        self.assertEqual(
            self.backend.saved_bottles["right"]["final_weight_g"],
            232.0,
        )

    def test_replay_window_evicts_oldest_and_never_blocks_shutdown(self) -> None:
        for index in range(MAX_MUTATION_REQUESTS):
            self.server._mutation_ledger[f"old-{index}"] = (
                "signature",
                {"ok": True, "result": {}, "error": None},
                False,
            )
        worker = self._worker()

        def enqueue(_name, _payload, **kwargs):
            if kwargs.get("sent") is not None:
                kwargs["sent"].set()
            return True

        with mock.patch.object(self.server, "_enqueue", side_effect=enqueue):
            self.server._commands.put(
                CommandEnvelope("shutdown_server", "shutdown-after-window")
            )
            worker.join(2)

        self.assertFalse(worker.is_alive())
        self.assertNotIn("old-0", self.server._mutation_ledger)
        self.assertIn("shutdown-after-window", self.server._mutation_ledger)
        self.assertLessEqual(len(self.server._mutation_ledger), MAX_MUTATION_REQUESTS)
        self.assertEqual(self.backend.phase, RunPhase.FINALIZED)

    def test_shutdown_discards_coalesced_later_start(self) -> None:
        worker = self._worker()
        shutdown = CommandEnvelope("shutdown_server", "shutdown-1")
        start = CommandEnvelope("start_run", "start-1", self._config())

        def enqueue(_name, _payload, **kwargs):
            if kwargs.get("sent") is not None:
                kwargs["sent"].set()
            return True

        with mock.patch.object(self.server, "_enqueue", side_effect=enqueue):
            self.server._commands.put(shutdown)
            self.server._commands.put(start)
            worker.join(2)

        self.assertFalse(worker.is_alive())
        self.assertFalse(self.backend.start_entered.is_set())
        self.assertEqual(self.backend.phase, RunPhase.FINALIZED)

    def test_callbacks_only_enqueue_and_never_write_a_socket(self) -> None:
        with mock.patch(
            "squeakview.apps.operator.backend.supervisor.server.SocketEnvelopeWriter.send"
        ) as send:
            self.server._emit_log("hello")
            event = DashboardEvent.parse("sample")
            assert event is not None
            self.server._emit_dashboard(event)

        send.assert_not_called()
        self.assertEqual(self.server._presentation_events.qsize(), 2)
        queued = self.server._presentation_events.get_nowait()
        self.assertEqual(queued.name, "log")
        queued = self.server._presentation_events.get_nowait()
        self.assertEqual(queued.name, "dashboard")
        self.assertEqual(DashboardEvent.from_payload(queued.payload), event)

    def test_maximum_serial_log_is_bounded_below_ipc_frame_limit(self) -> None:
        self.server._emit_log("🐁" * (64 * 1024))
        queued = self.server._presentation_events.get_nowait()

        frame = encode_envelope(
            EventEnvelope(queued.name, 0, queued.payload)
        )

        self.assertLessEqual(len(frame), MAX_FRAME_BYTES)
        self.assertIn("presentation truncated", queued.payload["message"])

    def test_heartbeat_renews_lease_inline_during_long_command(self) -> None:
        clock = [0.0]
        server = SupervisorServer(
            self.socket_path,
            backend_factory=FakeBackend,
            gui_lease_timeout_s=5.0,
            monotonic=lambda: clock[0],
        )
        server._renew_gui_lease()
        server._operation_lock.acquire()
        try:
            clock[0] = 4.0
            handled = server._handle_heartbeat(
                CommandEnvelope(
                    HEARTBEAT_COMMAND_NAME,
                    HEARTBEAT_REQUEST_ID,
                    {},
                )
            )
        finally:
            server._operation_lock.release()

        clock[0] = 8.0
        self.assertTrue(handled)
        self.assertFalse(server._gui_lease_expired())
        self.assertEqual(server._commands.qsize(), 0)

    def test_heartbeat_requires_reserved_id_and_empty_payload(self) -> None:
        with self.assertRaisesRegex(ProtocolError, "invalid.*heartbeat"):
            self.server._handle_heartbeat(
                CommandEnvelope(HEARTBEAT_COMMAND_NAME, "wrong-id", {})
            )
        with self.assertRaisesRegex(ProtocolError, "invalid.*heartbeat"):
            self.server._handle_heartbeat(
                CommandEnvelope(
                    HEARTBEAT_COMMAND_NAME,
                    HEARTBEAT_REQUEST_ID,
                    {"unexpected": True},
                )
            )

    def test_many_heartbeats_do_not_grow_any_server_queue_or_ledger(self) -> None:
        heartbeat = CommandEnvelope(
            HEARTBEAT_COMMAND_NAME, HEARTBEAT_REQUEST_ID, {}
        )

        for _ in range(10_000):
            self.assertTrue(self.server._handle_heartbeat(heartbeat))

        self.assertEqual(self.server._commands.qsize(), 0)
        self.assertEqual(self.server._critical_events.qsize(), 0)
        self.assertEqual(self.server._presentation_events.qsize(), 0)
        self.assertEqual(len(self.server._mutation_ledger), 0)

    def test_peer_credentials_include_pid_and_uid(self) -> None:
        peer = mock.Mock()
        peer.getsockopt.return_value = struct.pack("3i", 123, os.getuid(), 456)

        self.assertEqual(
            self.server._peer_credentials(peer), (123, os.getuid())
        )

    def test_cleanup_does_not_unlink_replaced_socket_inode(self) -> None:
        self.socket_path.write_text("old")
        old = self.socket_path.stat()
        self.server._socket_identity = (old.st_dev, old.st_ino, old.st_ctime_ns)
        self.socket_path.unlink()
        self.socket_path.write_text("replacement")

        self.server._unlink_owned_socket()

        self.assertTrue(self.socket_path.exists())

    def test_non_socket_path_is_never_replaced(self) -> None:
        self.socket_path.write_text("important")

        with self.assertRaisesRegex(RuntimeError, "non-socket"):
            self.server._prepare_listener()

        self.assertEqual(self.socket_path.read_text(), "important")

    def test_gui_launch_failure_still_closes_listener_and_cleans_socket(self) -> None:
        listener = mock.Mock()
        with (
            mock.patch.object(self.server, "_prepare_listener", return_value=listener),
            mock.patch.object(self.server, "_unlink_owned_socket") as unlink,
            mock.patch(
                "squeakview.apps.operator.backend.supervisor.server.subprocess.Popen",
                side_effect=OSError("exec failed"),
            ),
        ):
            with self.assertRaisesRegex(OSError, "exec failed"):
                self.server.serve(["gui"])

        listener.close.assert_called_once_with()
        unlink.assert_called_once_with()
        self.assertTrue(self.backend.cancelled.is_set())

    def test_unexpected_ipc_eof_returns_nonzero_while_gui_is_alive(self) -> None:
        listener = mock.Mock()
        client = mock.Mock()
        listener.accept.return_value = (client, None)
        gui = mock.Mock(pid=123)
        gui.poll.return_value = None
        with (
            mock.patch.object(self.server, "_prepare_listener", return_value=listener),
            mock.patch.object(self.server, "_peer_credentials", return_value=(123, os.getuid())),
            mock.patch(
                "squeakview.apps.operator.backend.supervisor.server.subprocess.Popen",
                return_value=gui,
            ),
            mock.patch(
                "squeakview.apps.operator.backend.supervisor.server.SocketEnvelopeReader.receive",
                side_effect=[
                    CommandEnvelope(HEARTBEAT_COMMAND_NAME, HEARTBEAT_REQUEST_ID, {}),
                    EOFError("lost"),
                ],
            ),
            mock.patch.object(self.server, "_unlink_owned_socket"),
        ):
            result = self.server.serve(["gui"])

        self.assertEqual(result, 1)
        self.assertIn("lease was lost", self.server.last_error)

    def test_launcher_acknowledgement_occurs_only_after_owned_gui_connects(self) -> None:
        listener = mock.Mock()
        client = mock.Mock()
        listener.accept.return_value = (client, None)
        gui = mock.Mock(pid=123)
        gui.poll.return_value = None
        connected = mock.Mock()
        with (
            mock.patch.object(self.server, "_prepare_listener", return_value=listener),
            mock.patch.object(self.server, "_peer_credentials", return_value=(123, os.getuid())),
            mock.patch(
                "squeakview.apps.operator.backend.supervisor.server.subprocess.Popen",
                return_value=gui,
            ),
            mock.patch(
                "squeakview.apps.operator.backend.supervisor.server.SocketEnvelopeReader.receive",
                side_effect=[
                    CommandEnvelope(HEARTBEAT_COMMAND_NAME, HEARTBEAT_REQUEST_ID, {}),
                    EOFError("lost"),
                ],
            ),
            mock.patch.object(self.server, "_unlink_owned_socket"),
        ):
            result = self.server.serve(["gui"], on_gui_ready=connected)

        self.assertEqual(result, 1)
        connected.assert_called_once_with()

    def test_expired_gui_heartbeat_fails_closed_and_returns_nonzero(self) -> None:
        listener = mock.Mock()
        client = mock.Mock()
        listener.accept.return_value = (client, None)
        gui = mock.Mock(pid=123)
        gui.poll.return_value = None
        clock = [0.0]
        server = SupervisorServer(
            self.socket_path,
            backend_factory=FakeBackend,
            gui_lease_timeout_s=1.0,
            monotonic=lambda: clock[0],
        )

        def quiet_peer(_client):
            clock[0] = 2.0
            raise TimeoutError

        with (
            mock.patch.object(server, "_prepare_listener", return_value=listener),
            mock.patch.object(server, "_peer_credentials", return_value=(123, os.getuid())),
            mock.patch(
                "squeakview.apps.operator.backend.supervisor.server.subprocess.Popen",
                return_value=gui,
            ),
            mock.patch(
                "squeakview.apps.operator.backend.supervisor.server.SocketEnvelopeReader.receive",
                side_effect=quiet_peer,
            ),
            mock.patch.object(server, "_unlink_owned_socket"),
        ):
            result = server.serve(["gui"])

        self.assertEqual(result, 1)
        self.assertEqual(
            server.last_error,
            "operator GUI main-loop heartbeat lease expired",
        )
        self.assertTrue(server.backend.cancelled.is_set())
        self.assertEqual(server.backend.abort_calls, ["operator GUI lost"])

    def test_later_error_populates_loss_reason_after_concurrent_loss_signal(self) -> None:
        self.server._signal_client_loss()
        self.server._signal_client_loss("supervisor IPC send failed")
        self.server._loss_thread.join(2)

        self.assertEqual(self.server.last_error, "supervisor IPC send failed")

    def test_backend_abort_seam_wakes_startup_and_fails_active_run(self) -> None:
        backend = OperatorBackend(lambda _line: None)
        backend._state_machine.transition(RunPhase.CREATED)
        backend._run_finalized = False
        with mock.patch.object(backend, "_finalize_run", return_value=True) as finalize:
            result = backend.abort_run("operator GUI lost")

        self.assertTrue(result)
        self.assertTrue(backend._stop_requested.is_set())
        self.assertTrue(backend._inference_ready.is_set())
        finalize.assert_called_once_with(
            final_state="failed", error="operator GUI lost"
        )

    def test_backend_abort_waits_through_nonterminal_finalization_phase(self) -> None:
        backend = OperatorBackend(lambda _line: None)
        backend._state_machine.transition(RunPhase.CREATED)
        backend._state_machine.transition(RunPhase.STARTING)
        backend._state_machine.transition(RunPhase.STOPPING)
        backend._run_finalized = False
        with mock.patch.object(backend, "_finalize_run", return_value=False) as finalize:
            backend.abort_run("operator GUI lost")

        finalize.assert_called_once_with(
            final_state="failed", error="operator GUI lost"
        )

    def test_lease_loss_during_successful_finalizer_forces_failed_terminal_state(self) -> None:
        backend = OperatorBackend(lambda _line: None)
        backend._state_machine.transition(RunPhase.CREATED)
        backend._run_finalized = False
        backend.state.run_dir = self.socket_path.parent / "run"
        backend.state.run_dir.mkdir()

        def finalize(*_args, **_kwargs):
            backend.cancel_operator_lease()
            from squeakview.apps.operator.backend.lifecycle import FinalizationResult

            return FinalizationResult("finalized", None, 0, False)

        with (
            mock.patch(
                "squeakview.apps.operator.backend.manager.lifecycle.finalize_run",
                side_effect=finalize,
            ),
            mock.patch(
                "squeakview.apps.operator.backend.manager.run_context.write_status"
            ) as write_status,
            mock.patch.object(backend, "_write_run_manifest", return_value=True),
        ):
            backend._finalize_run(final_state="finalized")

        self.assertEqual(backend.snapshot().phase, RunPhase.FAILED)
        self.assertEqual(backend.snapshot().error, "operator GUI lost")
        self.assertTrue(
            any(call.args[1] == "failed" for call in write_status.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
