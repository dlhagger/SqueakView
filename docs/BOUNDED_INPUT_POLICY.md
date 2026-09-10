# Bounded Configuration and Metadata Input Policy

SqueakView treats configuration and run metadata as scientific control inputs,
not as arbitrary files. Small documents are read through
`squeakview.common.bounded_input`: the reader opens a regular file, checks its
size before allocation, reads exactly that size, rejects growth or identity
change during the read, and decodes strict UTF-8. JSON objects reject duplicate
keys at every nesting level. Safe YAML mappings use the same stable read and
duplicate-key policy.

Current limits are deliberately much larger than valid repository documents:

| Input | Limit |
|---|---:|
| Failure plans | 64 KiB |
| Latest-run marker | 4 KiB |
| DeepStream inference config | 1 MiB |
| Class/keypoint labels | 1 MiB |
| Task config and snapshotted task config | 1 MiB |
| Operator experiment/subject profile | 1 MiB each, 1,024 files per type |
| Qualification limits, matrix, and assignments | 1 MiB each |
| Pose schema/model metadata | 4 MiB |
| Run status, manifest, and derived JSON metadata | 16 MiB |

Malformed, oversized, duplicate-key, non-regular, changing, or invalid-UTF-8
control documents fail closed before acquisition when they affect startup.
Inference repeats scalar checks, including finite exposure/gain and exact gain
automatic-mode semantics. Qualification and debug comparison use required
strict reads. Matrix reporting intentionally converts invalid per-run metadata
to an empty bounded snapshot so it can report a failed cell rather than aborting
the entire matrix. GUI profile listing likewise skips individual invalid files,
but its enumeration and per-file allocation remain bounded. Profile writes and
localized runtime configs use crash-resistant atomic replacement. Existing
metadata cannot be updated when its current content fails strict validation,
and JSON writers reject NaN and Infinity.

## Intentional streaming and pseudo-file exceptions

The following are not whole-file configuration reads and must not be converted
to the small-document helper:

- capture JSONL and recording-admission CSV use strict UTF-8, duplicate-safe
  parsing with a 64 KiB per-record limit; other frame/serial/telemetry CSV and
  DeepStream latency logs are processed incrementally or through bounded tail
  windows;
- video/model hashing reads fixed chunks and verifies stable identity/size;
- model artifacts and recordings are large binary inputs with their own bounded
  streaming or subprocess validation;
- `/proc`, `/sys`, `/etc/nv_tegra_release`, and device-tree scalar identity
  fields are kernel/platform pseudo-files read as small best-effort context;
- supervisor IPC and child-event JSON are already bounded by their framing or
  line-buffer limits before JSON decoding;
- offline replay reads `frames.csv` through 64 KiB-bounded strict UTF-8 records
  into a temporary disk-backed SQLite ordinal index. Full-duration replay lookup
  therefore remains bounded in memory without weakening source-frame identity.

Callers that make scientific acceptance or lifecycle decisions must use strict
read APIs and propagate failure. The tolerant `run_context.read_json` function
is retained only for presentation, optional progress, initial document
creation, and complete matrix failure reporting; it is still bounded and
duplicate-safe, returning an empty snapshot for invalid input.
