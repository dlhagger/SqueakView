from __future__ import annotations

import threading
import time
import unittest

from squeakview.common.controller_watchdog import (
    REQUIRED_FEATURES,
    ControllerWatchdogSession,
    parse_ack,
    parse_capabilities,
    parse_state,
)


NONCE = "0123456789abcdef0123456789abcdef"
FEATURES = ";".join(sorted(REQUIRED_FEATURES))


class ControllerWatchdogProtocolTests(unittest.TestCase):
    def test_host_waits_reject_nonfinite_or_unbounded_values_before_send(self) -> None:
        for timeout in (float("nan"), float("inf"), -1.0, 61.0):
            sent: list[str] = []
            session = ControllerWatchdogSession(
                sent.append,
                lambda _message: None,
                requested_lease_ms=100,
                nonce=NONCE,
            )
            with self.subTest(timeout=timeout), self.assertRaisesRegex(
                ValueError, "controller host wait"
            ):
                session.negotiate(timeout_s=timeout)
            self.assertEqual(sent, [])

    def test_unsolicited_nonce_correct_ack_is_not_retained(self) -> None:
        session = ControllerWatchdogSession(
            lambda _line: None,
            lambda _message: None,
            requested_lease_ms=100,
            nonce=NONCE,
        )
        for sequence in range(10_000):
            self.assertTrue(
                session.ingest(
                    f"ACK_HEARTBEAT,1,{NONCE},{sequence},20,4"
                )
            )
        self.assertEqual(session._acks, {})

    def test_strict_nonce_bound_parsers(self) -> None:
        caps = parse_capabilities(
            f"CONTROLLER_CAPS,1,{NONCE},fw-1.2.3,900,{FEATURES}"
        )
        self.assertEqual(caps.session_nonce, NONCE)
        self.assertEqual(caps.watchdog_timeout_ms, 900)
        ack = parse_ack(f"ACK_DISARM,1,{NONCE},7,123456,88")
        self.assertEqual(ack.kind, "DISARM")
        self.assertEqual(ack.ttl_count, 88)
        state = parse_state(
            f"CONTROLLER_STATE,1,{NONCE},DISARMED,123457,88,DISARM_COMMAND"
        )
        self.assertEqual(state.state, "DISARMED")
        for line in (
            "CONTROLLER_CAPS,1,fw,900,lease_watchdog",
            f"ACK_ARM,1,WRONG,1,2",
            f"ACK_HEARTBEAT,1,{NONCE},1,2",
            f"ACK_DISARM,1,{NONCE},1,2,-1",
        ):
            with self.subTest(line=line), self.assertRaises(ValueError):
                (parse_capabilities if line.startswith("CONTROLLER") else parse_ack)(line)

    def test_full_session_orders_hello_arm_heartbeats_and_disarm(self) -> None:
        sent: list[str] = []
        holder: dict[str, ControllerWatchdogSession] = {}

        def send(line: str) -> None:
            sent.append(line)
            fields = line.split(",")
            session = holder["session"]
            if fields[0] == "HELLO":
                session.ingest(
                    f"CONTROLLER_CAPS,1,{NONCE},fw-1,100,{FEATURES}"
                )
            elif fields[0] == "ARM":
                session.ingest(f"ACK_ARM,1,{NONCE},{fields[4]},10")
            elif fields[0] == "HEARTBEAT":
                session.ingest(
                    f"ACK_HEARTBEAT,1,{NONCE},{fields[3]},20,4"
                )
            elif fields[0] == "DISARM":
                session.ingest(f"ACK_DISARM,1,{NONCE},{fields[3]},30,5")
                session.ingest(
                    f"CONTROLLER_STATE,1,{NONCE},DISARMED,31,5,DISARM_COMMAND"
                )

        failures: list[str] = []
        session = ControllerWatchdogSession(
            send, failures.append, requested_lease_ms=100, nonce=NONCE
        )
        holder["session"] = session
        session.negotiate(timeout_s=0.1)
        session.arm(30, timeout_s=0.1)
        time.sleep(0.08)
        session.disarm(timeout_s=0.2)

        self.assertEqual(sent[0], f"HELLO,1,{NONCE},100")
        self.assertEqual(sent[1], f"ARM,1,{NONCE},30,1")
        self.assertTrue(any(line.startswith("HEARTBEAT,1,") for line in sent))
        self.assertTrue(sent[-1].startswith(f"DISARM,1,{NONCE},"))
        self.assertEqual(session.final_ttl_count, 5)
        self.assertEqual(failures, [])

    def test_wrong_nonce_and_missing_features_fail_closed(self) -> None:
        holder: dict[str, ControllerWatchdogSession] = {}

        def stale_send(_line: str) -> None:
            holder["session"].ingest(
                "CONTROLLER_CAPS,1,ffffffffffffffffffffffffffffffff,fw,100,"
                + FEATURES
            )

        session = ControllerWatchdogSession(
            stale_send, lambda _message: None, requested_lease_ms=100, nonce=NONCE
        )
        holder["session"] = session
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            session.negotiate(timeout_s=0)

        def incomplete_send(_line: str) -> None:
            holder["session"].ingest(
                f"CONTROLLER_CAPS,1,{NONCE},fw,100,lease_watchdog"
            )

        session = ControllerWatchdogSession(
            incomplete_send, lambda _message: None, requested_lease_ms=100, nonce=NONCE
        )
        holder["session"] = session
        with self.assertRaisesRegex(RuntimeError, "missing required"):
            session.negotiate(timeout_s=0.1)

    def test_two_missed_heartbeat_acknowledgements_are_fatal(self) -> None:
        fatal = threading.Event()
        failures: list[str] = []
        holder: dict[str, ControllerWatchdogSession] = {}

        def fail(message: str) -> None:
            failures.append(message)
            fatal.set()

        def send(line: str) -> None:
            fields = line.split(",")
            session = holder["session"]
            if fields[0] == "HELLO":
                session.ingest(
                    f"CONTROLLER_CAPS,1,{NONCE},fw,100,{FEATURES}"
                )
            elif fields[0] == "ARM":
                session.ingest(f"ACK_ARM,1,{NONCE},{fields[4]},10")

        session = ControllerWatchdogSession(
            send, fail, requested_lease_ms=100, nonce=NONCE
        )
        holder["session"] = session
        session.negotiate(timeout_s=0.1)
        session.arm(30, timeout_s=0.1)
        self.assertTrue(fatal.wait(0.4))
        self.assertEqual(len(failures), 1)
        self.assertIn("heartbeat failed twice", failures[0])
        session.stop_worker()

    def test_missed_ack_does_not_delay_next_renewal_to_lease_deadline(self) -> None:
        heartbeat_times: list[float] = []
        second_heartbeat = threading.Event()
        holder: dict[str, ControllerWatchdogSession] = {}

        def send(line: str) -> None:
            fields = line.split(",")
            session = holder["session"]
            if fields[0] == "HELLO":
                session.ingest(
                    f"CONTROLLER_CAPS,1,{NONCE},fw,300,{FEATURES}"
                )
            elif fields[0] == "ARM":
                session.ingest(f"ACK_ARM,1,{NONCE},{fields[4]},10")
            elif fields[0] == "HEARTBEAT":
                heartbeat_times.append(time.monotonic())
                if len(heartbeat_times) == 2:
                    second_heartbeat.set()

        session = ControllerWatchdogSession(
            send, lambda _message: None, requested_lease_ms=300, nonce=NONCE
        )
        holder["session"] = session
        session.negotiate(timeout_s=0.1)
        session.arm(30, timeout_s=0.1)
        try:
            self.assertTrue(second_heartbeat.wait(0.5))
            # timeout / 3 is 100 ms. The old loop spent 100 ms waiting for
            # ACK and then slept another 100 ms, sending at the 300 ms lease
            # deadline. Leave generous scheduler margin while distinguishing
            # the intended one-interval cadence from that two-interval bug.
            self.assertLess(heartbeat_times[1] - heartbeat_times[0], 0.17)
        finally:
            session.stop_worker()

    def test_first_heartbeat_write_error_is_immediately_fatal(self) -> None:
        fatal = threading.Event()
        failures: list[str] = []
        holder: dict[str, ControllerWatchdogSession] = {}

        def fail(message: str) -> None:
            failures.append(message)
            fatal.set()

        def send(line: str) -> None:
            fields = line.split(",")
            session = holder["session"]
            if fields[0] == "HELLO":
                session.ingest(
                    f"CONTROLLER_CAPS,1,{NONCE},fw,100,{FEATURES}"
                )
            elif fields[0] == "ARM":
                session.ingest(f"ACK_ARM,1,{NONCE},{fields[4]},10")
            elif fields[0] == "HEARTBEAT":
                raise OSError("USB write failed")

        session = ControllerWatchdogSession(
            send, fail, requested_lease_ms=100, nonce=NONCE
        )
        holder["session"] = session
        session.negotiate(timeout_s=0.1)
        session.arm(30, timeout_s=0.1)
        self.assertTrue(fatal.wait(0.2))
        self.assertEqual(len(failures), 1)
        self.assertIn("heartbeat write failed", failures[0])

    def test_watchdog_trip_telemetry_is_immediately_fatal(self) -> None:
        failures: list[str] = []
        holder: dict[str, ControllerWatchdogSession] = {}

        def send(line: str) -> None:
            fields = line.split(",")
            session = holder["session"]
            if fields[0] == "HELLO":
                session.ingest(
                    f"CONTROLLER_CAPS,1,{NONCE},fw,100,{FEATURES}"
                )
            elif fields[0] == "ARM":
                session.ingest(f"ACK_ARM,1,{NONCE},{fields[4]},10")

        session = ControllerWatchdogSession(
            send, failures.append, requested_lease_ms=100, nonce=NONCE
        )
        holder["session"] = session
        session.negotiate(timeout_s=0.1)
        session.arm(30, timeout_s=0.1)
        session.ingest(
            f"CONTROLLER_STATE,1,{NONCE},WATCHDOG_TRIPPED,20,3,LEASE_EXPIRED"
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("unsafe state", failures[0])
        session.stop_worker()

    def test_fatal_session_still_waits_for_ordered_disarm_evidence(self) -> None:
        holder: dict[str, ControllerWatchdogSession] = {}
        reply_threads: list[threading.Thread] = []

        def send(line: str) -> None:
            fields = line.split(",")
            session = holder["session"]
            if fields[0] == "HELLO":
                session.ingest(
                    f"CONTROLLER_CAPS,1,{NONCE},fw,300,{FEATURES}"
                )
            elif fields[0] == "ARM":
                session.ingest(f"ACK_ARM,1,{NONCE},{fields[4]},10")
            elif fields[0] == "DISARM":
                def delayed_reply() -> None:
                    time.sleep(0.02)
                    session.ingest(
                        f"ACK_DISARM,1,{NONCE},{fields[3]},30,5"
                    )
                    session.ingest(
                        f"CONTROLLER_STATE,1,{NONCE},DISARMED,31,5,DISARM_COMMAND"
                    )

                thread = threading.Thread(target=delayed_reply)
                reply_threads.append(thread)
                thread.start()

        failures: list[str] = []
        session = ControllerWatchdogSession(
            send, failures.append, requested_lease_ms=300, nonce=NONCE
        )
        holder["session"] = session
        session.negotiate(timeout_s=0.1)
        session.arm(30, timeout_s=0.1)
        session.ingest(
            f"CONTROLLER_STATE,1,{NONCE},WATCHDOG_TRIPPED,20,3,LEASE_EXPIRED"
        )

        ack = session.disarm(timeout_s=0.2)

        for thread in reply_threads:
            thread.join()
        self.assertEqual(ack.ttl_count, 5)
        self.assertEqual(session.final_ttl_count, 5)
        self.assertEqual(len(failures), 1)

    def test_disarm_interrupts_heartbeat_ack_wait_before_sending(self) -> None:
        holder: dict[str, ControllerWatchdogSession] = {}
        heartbeat_waiting = threading.Event()

        def send(line: str) -> None:
            fields = line.split(",")
            session = holder["session"]
            if fields[0] == "DISARM":
                session.ingest(f"ACK_DISARM,1,{NONCE},{fields[3]},30,5")
                session.ingest(
                    f"CONTROLLER_STATE,1,{NONCE},DISARMED,31,5,DISARM_COMMAND"
                )

        session = ControllerWatchdogSession(
            send, lambda _message: None, requested_lease_ms=60_000, nonce=NONCE
        )
        holder["session"] = session
        session.capabilities = parse_capabilities(
            f"CONTROLLER_CAPS,1,{NONCE},fw,60000,{FEATURES}"
        )
        session._armed = True
        session._sequence = 1

        def pending_heartbeat() -> None:
            session._send_expecting_ack(
                "HEARTBEAT", 1, f"HEARTBEAT,1,{NONCE},1,0"
            )
            heartbeat_waiting.set()
            with self.assertRaisesRegex(RuntimeError, "not received"):
                session._wait_ack("HEARTBEAT", 1, 20.0)

        thread = threading.Thread(target=pending_heartbeat)
        session._heartbeat_thread = thread
        thread.start()
        self.assertTrue(heartbeat_waiting.wait(0.1))

        started = time.monotonic()
        session.disarm(timeout_s=0.2)
        elapsed = time.monotonic() - started

        thread.join(timeout=0.1)
        self.assertFalse(thread.is_alive())
        self.assertLess(elapsed, 0.15)


if __name__ == "__main__":
    unittest.main()
