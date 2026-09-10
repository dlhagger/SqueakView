# DeepStream debug-overhead protocol

This protocol measures the acquisition overhead of SqueakView's opt-in NVIDIA
latency diagnostics. The tools only inspect completed GUI-launched runs. They do
not create, simulate, launch, resume, or modify a scientific run.

## Preconditions

- Select one canonical case with
  `.venv/bin/python scripts/qualify_matrix.py --list-cases`.
- Use one reviewed `validated: true` qualification-limits file for both runs.
- Use the same Jetson, software commit, model package, native plugins, camera,
  controller, capture configuration, preview state, named nvpmodel mode, and
  qualification case.
- Plan comparable durations. The default comparison tolerance is the larger of
  five seconds or 1% of the longer run.

The checked-in limits profile is measurement-only and cannot satisfy these
preconditions. Do not mark it validated simply to obtain a result.

The checked-in
`qualification/debug_overhead_thresholds.v1.yaml` is likewise an unapproved,
measurement-only template. Its seven exact metric keys each show the supported
`max_increase` and `max_percent_increase` bounds. Leave unused bounds null, but
an approved profile must provide at least one finite nonnegative bound for every
metric. Copy it to a separately reviewed, versioned file and set `approved: true`
only after the matched evidence has been reviewed; never approve the template in
place merely to obtain a passing comparison.

## Acquire the pair through the GUI

Export the same backend-enforced binding for both launches:

```bash
export SQUEAKVIEW_QUALIFICATION_CASE_ID='<case-id>'
# Optional only for a deliberate alternate matrix:
# export SQUEAKVIEW_QUALIFICATION_MATRIX='/absolute/path/to/matrix.yaml'
```

Launch and complete the baseline with the debug profile explicitly off:

```bash
export SQUEAKVIEW_DEEPSTREAM_DEBUG_PROFILE=0
bash scripts/launch_operator.sh
```

Wait for terminal finalization, then exit that GUI/supervisor before changing
its environment. Qualify the completed baseline:

```bash
.venv/bin/python scripts/qualify_run.py /absolute/path/to/baseline-run \
  --limits qualification/<validated-limits>.yaml
```

Launch a new GUI/supervisor with the debug profile on, reproduce the same bound
case and duration, and wait for terminal finalization:

```bash
export SQUEAKVIEW_DEEPSTREAM_DEBUG_PROFILE=1
bash scripts/launch_operator.sh
.venv/bin/python scripts/qualify_run.py /absolute/path/to/debug-run \
  --limits qualification/<validated-limits>.yaml \
  --allow-debug-profile
```

The profile also attaches NVIDIA's shipped `measure_latency_probe` and
`measure_fps_probe` downstream of the leaky inference work. Preflight requires
stable regular probe modules and records their size and SHA-256 identities.
When `NVDS_ENABLE_LATENCY_MEASUREMENT` is truthy, the raw FLIR source adds
NVIDIA's public reference-timestamp metadata immediately after setting PTS/DTS,
using the real source element name and its acquisition-local source sequence.
The production/default source path does not add this metadata. NVIDIA's API
returns no success status and documents decoder names as the standard latency
anchors, so this does not by itself prove frame-latency output for the raw-source
graph. Debug-run qualification remains incomplete/failed unless the bounded
DeepStream log has valid structured frame and component latency records, finite
Service Maker FPS evidence, and no truncation. A short GUI-launched on-device run
must demonstrate that behavior before the instrumentation is accepted.

## Compare and retain evidence

Run the Qt-free checker with an explicit case guard and durable output path:

```bash
.venv/bin/python scripts/compare_debug_overhead.py \
  /absolute/path/to/baseline-run \
  /absolute/path/to/debug-run \
  --case-id '<case-id>' \
  --thresholds qualification/<reviewed-approved-debug-overhead-thresholds>.yaml \
  --output qualification/results/<pair-id>-debug-overhead.json
```

The checker fails closed unless both runs are successfully terminal, owned by
the durable supervisor, carry identical manifest/status qualification bindings,
have opposite debug states, pass the correct qualification modes, use identical
validated limits, runtime/config/model/source identities, and meet duration and
latency-log requirements. The report retains both run paths, qualification
summary identities, and the hashed source-evidence references it consumed.

Without explicitly approved overhead thresholds, a valid comparison is
`incomplete`, not passed. Thresholds bound increases only and must cover every
reported metric. A single pair characterizes that pair; use repeated,
counterbalanced runs and scientific review before approving a general profile.
Retain the report, threshold document, limits document, matrix, and both entire
immutable run directories together.
