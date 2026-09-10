# RP2040 Controller Watchdog Protocol and Host Contract

Status: **specification only; firmware and hardware qualification are not present in this repository.**

This document defines the controller safety boundary required before triggered
SqueakView acquisition can be called resilient to total Jetson, supervisor, USB,
or host-power failure. The current deployed host protocol is legacy `START,<fps>`
and `STOP`, with `CAMERA_*` telemetry and `ACK_STOP`. It has no negotiated lease.
The Python host therefore must not claim that an independently powered controller
will stop pulses after the Jetson dies.

Normative words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** describe the future
watchdog-capable firmware and its matching host integration.

## Safety invariants

1. Reset, boot, USB disconnect, malformed input, expired lease, and internal
   watchdog reset MUST put the physical camera-trigger output in its inactive
   electrical level before any other application action.
2. The controller MUST boot `DISARMED`. It MUST NOT emit a trigger because a USB
   port opened, a `HELLO` arrived, or a previous session was armed.
3. Only a successfully negotiated, session-bound `ARM` may enter `ARMED`.
4. While `ARMED`, failure to receive a valid heartbeat before the monotonic lease
   deadline MUST synchronously disable the trigger and latch `WATCHDOG_TRIPPED`.
5. `WATCHDOG_TRIPPED` MUST require a new session nonce and new `ARM`; a delayed
   packet from the previous session MUST NOT restart output.
6. `STOP`/`DISARM` MUST disable the trigger first, then emit its acknowledgement.
7. Host logging, GUI state, or USB presence MUST NOT be part of the controller's
   safety action. The RP2040 hardware timer/watchdog owns the deadline.

## Transport and framing

- USB CDC serial, 115200 baud, 8-N-1, newline-delimited ASCII.
- Maximum line length: 512 bytes including arguments but excluding `\n`.
- Commands and event names are uppercase and comma-separated. Fields MUST NOT
  contain commas, newlines, or control characters.
- Integers are unsigned base-10 with no sign. Nonces are 32 lowercase hexadecimal
  characters. Unknown commands receive `ERROR,<nonce>,UNKNOWN_COMMAND` and MUST
  NOT change output state.
- Duplicate commands with the same nonce and sequence number are idempotent.

The host's existing 64 KiB serial input bound remains defense in depth; qualified
firmware is held to the tighter 512-byte bound.

## Version and capability negotiation

The future host opens the port while capture remains unstarted and sends:

```text
HELLO,1,<session_nonce>,<requested_lease_ms>
```

Qualified v1 firmware replies:

```text
CONTROLLER_CAPS,1,<session_nonce>,<firmware_version>,<watchdog_timeout_ms>,lease_watchdog;trigger_low_failsafe;session_nonce;telemetry_v1
```

The reply nonce MUST exactly match the current `HELLO`; this prevents a delayed
capability advertisement from an earlier connection from satisfying negotiation.
`watchdog_timeout_ms` MUST be 100–60000 and MUST be no greater than the requested
lease. The host MUST reject a version other than `1`, a missing required feature,
an out-of-policy timeout, malformed fields, or no response within the negotiated
startup deadline. Rejection occurs before camera/controller `START` and is
production-fatal.

The host now contains an opt-in, unit-tested experimental v1 implementation. It
sends nonce-bound `HELLO`, validates the complete required feature set, waits for
nonce/sequence-bound arm/disarm acknowledgements, and renews the lease from the
Qt-free supervisor. Legacy `START`/`STOP` remains the default. The experimental
path is selected only by launching with
`SQUEAKVIEW_CONTROLLER_PROTOCOL=watchdog_v1_experimental`; its requested lease
may be set with `SQUEAKVIEW_CONTROLLER_WATCHDOG_LEASE_MS` (default 1500). It is
always recorded as `controller_watchdog_unqualified` and is therefore never
production eligible. Do not enable it against existing legacy firmware.

## Arm, heartbeat, and disarm

After DeepStream is ready and all pre-start metadata is durable, the future host
sends:

```text
ARM,1,<session_nonce>,<fps>,<sequence>
ACK_ARM,1,<session_nonce>,<sequence>,<controller_monotonic_us>
HEARTBEAT,1,<session_nonce>,<sequence>,<host_monotonic_ns>
ACK_HEARTBEAT,1,<session_nonce>,<sequence>,<controller_monotonic_us>,<ttl_count>
DISARM,1,<session_nonce>,<sequence>
ACK_DISARM,1,<session_nonce>,<sequence>,<controller_monotonic_us>,<ttl_count>
```

- `fps` MUST be within a firmware-compiled safe range and MUST match the accepted
  experiment rate.
- The first trigger MUST occur only after `ACK_ARM` has been queued for delivery.
- Heartbeat sequence numbers MUST strictly increase. Repeats MAY be acknowledged
  idempotently but MUST NOT extend the lease; older values MUST be ignored.
- The host SHOULD transmit every `watchdog_timeout_ms / 3`, using monotonic time.
  It MUST treat two missed acknowledgements or any write/read error as a fatal run
  fault and begin host-side STOP/finalization.
- The controller MUST renew the lease only after validating version, nonce,
  sequence, and full frame syntax. Arbitrary serial traffic does not renew it.
- `DISARM` and legacy `STOP` MUST both force inactive output. During migration,
  watchdog firmware MAY accept legacy `START,<fps>`, but doing so MUST remain
  production-ineligible because it has no session lease.

## Watchdog telemetry

Every transition is persisted through the existing raw `serial.csv` ledger. New
firmware emits these bounded lines:

```text
CONTROLLER_STATE,1,<nonce>,<DISARMED|ARMED|WATCHDOG_TRIPPED>,<controller_us>,<ttl_count>,<reason>
CONTROLLER_HEALTH,1,<nonce>,<controller_us>,<ttl_count>,<last_heartbeat_sequence>,<lease_remaining_ms>,<reset_cause>
```

`CONTROLLER_HEALTH` SHOULD be emitted once per second while armed and once on
each transition. `reason` and `reset_cause` are enumerated tokens, not free text.
At minimum: `BOOT`, `ARM_COMMAND`, `DISARM_COMMAND`, `LEASE_EXPIRED`,
`USB_DISCONNECT`, `INTERNAL_WATCHDOG`, and `INVALID_COMMAND`.

The experimental host records its protocol, nonce, requested/negotiated timeout,
features, arm state, and final TTL count in the run manifest. Qualification must
extend that evidence to include firmware version,
firmware binary SHA-256, negotiated timeout/features, session nonce, heartbeat
counts, final TTL count, final state, reset cause, and qualification identity.
Raw lines and host receive clocks remain in `serial.csv` for alignment.

## Host integration contract

The production host implementation, once compatible firmware exists and the
experimental implementation has passed bench qualification, MUST:

1. Negotiate before spawning/arming acquisition and reject legacy/no-response
   controllers for watchdog-required triggered runs.
2. Generate a cryptographically random nonce per run and never reuse it.
3. Start heartbeat renewal from the durable Qt-free supervisor, not the GUI and
   not the capture child. GUI loss still initiates normal ordered shutdown; total
   supervisor/Jetson loss is covered by lease expiry.
4. Use a bounded nonblocking heartbeat queue of capacity one. A missed write is a
   fault, never an invitation to accumulate pending heartbeats.
5. Confirm `ACK_ARM` and the first camera TTL before publishing `RECORDING`.
6. On stop: send `DISARM`; confirm inactive state and `ACK_DISARM`; reconcile the
   final TTL count; then drain/close capture, close serial, and validate artifacts.
7. Mark the run failed if negotiation, heartbeat, acknowledgement, telemetry,
   final-state evidence, or TTL reconciliation is absent or inconsistent.

## Fault-injection and bench procedure

Use a logic analyzer or oscilloscope on the actual trigger output and a second
independent clock/reference channel. Do not infer electrical fail-safe timing only
from host logs.

For each supported firmware binary, device revision, cable/power arrangement,
rate, and lease timeout:

1. Verify boot/reset output remains inactive for at least 60 seconds without a
   host.
2. Negotiate and arm; verify frequency, polarity, first-pulse ordering, and
   heartbeat acknowledgements.
3. Stop host heartbeats while leaving USB and both devices powered.
4. Kill the supervisor with `SIGKILL` while leaving the controller powered.
5. Disconnect USB during output-high and output-low phases.
6. Remove Jetson power while keeping the controller independently powered.
7. Inject truncated, oversized, duplicated, reordered, wrong-nonce, wrong-version,
   and high-rate command streams.
8. Reset/brown-out the RP2040 while armed and exercise its internal watchdog.
9. Verify a delayed heartbeat/ARM from the old session cannot restart triggers.
10. Repeat enough trials to cover timing phase and establish a documented upper
    confidence bound, including the maximum pulses after the last valid lease.

Firmware-only injection hooks MUST be compile-time qualification features, visibly
reported in capabilities, and forbidden in production builds. Host serial
`read_error`, `write_error`, and `ledger_write_error` plans test host fail-close
behavior but do not substitute for the electrical tests above.

## Acceptance evidence

A watchdog qualification is complete only when an immutable evidence bundle has:

- firmware source revision, reproducible binary, SHA-256, compiler/SDK versions,
  RP2040 board revision, wiring/polarity, and power topology;
- host commit, JetPack/kernel identity, exact negotiated transcript, and all
  `serial.csv`/run manifests;
- raw logic-analyzer captures with calibrated time base for every fault case;
- measured arm latency, heartbeat jitter, lease-expiry-to-inactive latency, pulse
  width/rate, final TTL count, and pulses after expiry/disconnect/power loss;
- explicit pass criteria: output reaches and remains inactive within the
  negotiated watchdog deadline, **zero new trigger pulses begin after expiry**,
  old-session traffic cannot rearm, and the host run is non-successful;
- repeated-trial counts, failures, equipment calibration, operator/date, and
  independent review sign-off.

Until those artifacts exist, SqueakView
documentation and UI must describe watchdog protection as unqualified/not
available. Successful legacy `START`/`STOP` tests do not close this safety gap.
