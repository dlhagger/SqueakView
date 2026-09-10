# Failure-injection qualification protocol

Failure injection is disabled by default and every injected run is permanently
marked non-production. Use dedicated test storage and never reuse an injected
run as scientific evidence or a sustained-matrix cell.

Create one JSON plan per run:

```json
{
  "schema_version": "1.0",
  "target": "flir_source",
  "kind": "source_read",
  "after_frames": 30,
  "stream_id": 0,
  "delay_us": 0
}
```

Launch the normal GUI from a standalone terminal with the qualification gate
and selected plan:

```bash
SQUEAKVIEW_ENABLE_FAILURE_INJECTION=1 \
SQUEAKVIEW_FAILURE_PLAN=/absolute/path/to/plan.json \
bash scripts/launch_operator.sh
```

Supported target/kind pairs are:

| Target | Kinds |
|---|---|
| `flir_source` | `source_read`, `source_incomplete`, `capture_ledger_write` |
| `record_queue` | `stall` |
| `encoder` | `error` |
| `muxer` | `error` |
| `filesink` | `error`, `disk_full` |
| `serial_controller` | `read_error`, `write_error`, `ledger_write_error` |
| `shutdown` | `stop_ack_timeout`, `capture_exit_unconfirmed`, `finalizer_timeout` |

`after_frames` is the exact number of successful frames/serial operations
before a source, recording, or serial fault begins, with one explicit
exception: `filesink/disk_full` redirects the selected sink to `/dev/full`,
which fails on its first write. It is therefore immediate-only and requires
`after_frames: 1` as a schema sentinel; it does not claim that one frame was
successfully written. Use `filesink/error` for a delayed filesink-boundary
fault. For `record_queue/stall`, set `delay_us` from 1 through 10,000,000; it is
zero for every other target. Shutdown injections occur at their named boundary
and still require `after_frames` to be at least one for the version-1 schema.

For each target, retain the plan, manifest, status, ledgers, MP4/full-decode
result, operator log, and diagnostic telemetry. Acceptance requires:

- nonzero/failure terminal status and `production_eligible: false`;
- no success claim without a concrete capture exit and post-run validation;
- no silent source-to-recording discrepancy;
- no telemetry eviction presented as frame integrity;
- bounded TERM/KILL escalation for a stuck process; and
- a subsequent uninjected short GUI run that validates normally.

Serial cases require the physical controller and cannot be substituted with a
mock for on-device acceptance. The sudden-power case is separate and follows
`HARD_POWER_LOSS_PROTOCOL.md`.
