#!/usr/bin/env python3
"""Inspect or inject a narrowly targeted durable-supervisor qualification fault."""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from squeakview.common.bounded_input import read_stable_regular_file  # noqa: E402


GATE = "SQUEAKVIEW_ENABLE_SUPERVISOR_FAILURE_INJECTION"
CONFIRMATION = "SUPERVISOR-FAILURE-QUALIFICATION"
MAX_PROC_BYTES = 64 * 1024
MAX_STATUS_BYTES = 1024 * 1024
ACTIVE_PHASES = ("starting", "recording", "stopping")
FINALIZER_STAGES = (
    "capture_reconciliation",
    "inference_admission",
    "recording_validation",
)
ANALYSIS_STAGES = ("streaming_alignment",)
ALLOWED_PHASES = (*ACTIVE_PHASES, "finalizing", "analyzing")
SIGNALS = {
    "gui-freeze": signal.SIGSTOP,
    "gui-terminate": signal.SIGTERM,
    "supervisor-terminate": signal.SIGTERM,
}
STARTUP_BARRIERS = ("pre_capture", "after_spawn_before_ready")
FINALIZER_BARRIERS = tuple(f"finalizer:{stage}" for stage in (*FINALIZER_STAGES, *ANALYSIS_STAGES))
ALLOWED_BARRIERS = (*STARTUP_BARRIERS, *FINALIZER_BARRIERS)
MAX_WAIT_TIMEOUT_S = 300.0


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    pid: int
    ppid: int
    uid: int
    start_ticks: int
    argv: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class InjectionPlan:
    action: str
    signal: int
    supervisor: ProcessIdentity
    gui: ProcessIdentity
    capture: ProcessIdentity | None
    finalizer: ProcessIdentity | None
    target: ProcessIdentity
    run_dir: str
    phase: str
    stage: str | None
    barrier: str | None
    dry_run: bool


def _read_proc_file(path: Path, *, maximum: int = MAX_PROC_BYTES) -> bytes:
    with path.open("rb", buffering=0) as handle:
        value = handle.read(maximum + 1)
    if len(value) > maximum:
        raise ValueError(f"process metadata exceeds {maximum} bytes: {path}")
    return value


def _identity(pid: int, *, proc_root: Path = Path("/proc")) -> ProcessIdentity:
    if isinstance(pid, bool) or pid <= 1:
        raise ValueError("PID must be greater than 1")
    process_dir = proc_root / str(pid)
    stat_line = _read_proc_file(process_dir / "stat").decode("ascii", "strict")
    close = stat_line.rfind(")")
    if close < 0:
        raise ValueError(f"malformed process stat for PID {pid}")
    fields = stat_line[close + 2 :].split()
    if len(fields) < 20:
        raise ValueError(f"incomplete process stat for PID {pid}")
    argv = tuple(
        part.decode("utf-8", "strict")
        for part in _read_proc_file(process_dir / "cmdline").split(b"\0")
        if part
    )
    if not argv:
        raise ValueError(f"PID {pid} has an empty command line")
    return ProcessIdentity(
        pid=pid,
        ppid=int(fields[1]),
        uid=process_dir.stat().st_uid,
        start_ticks=int(fields[19]),
        argv=argv,
    )


def _children(parent_pid: int, *, proc_root: Path = Path("/proc")) -> tuple[ProcessIdentity, ...]:
    children: list[ProcessIdentity] = []
    with os.scandir(proc_root) as entries:
        for entry in entries:
            if not entry.name.isdecimal():
                continue
            try:
                identity = _identity(int(entry.name), proc_root=proc_root)
            except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
                continue
            if identity.ppid == parent_pid:
                children.append(identity)
    return tuple(sorted(children, key=lambda item: item.pid))


def _has_argv_sequence(argv: tuple[str, ...], expected: tuple[str, ...]) -> bool:
    width = len(expected)
    return any(argv[index : index + width] == expected for index in range(len(argv) - width + 1))


def _run_dir_arg(identity: ProcessIdentity) -> Path | None:
    try:
        index = identity.argv.index("--run-dir")
        return Path(identity.argv[index + 1]).resolve()
    except (ValueError, IndexError):
        return None


def _post_run_dir_arg(identity: ProcessIdentity) -> Path | None:
    marker = ("-m", "squeakview.apps.inference.post_run")
    for index in range(len(identity.argv) - len(marker)):
        if identity.argv[index : index + len(marker)] == marker:
            try:
                return Path(identity.argv[index + len(marker)]).resolve()
            except IndexError:
                return None
    return None


def _process_roles(
    children: tuple[ProcessIdentity, ...],
) -> tuple[
    tuple[ProcessIdentity, ...],
    tuple[ProcessIdentity, ...],
    tuple[ProcessIdentity, ...],
]:
    """Classify every direct SqueakView child without filtering by run."""

    gui = tuple(
        child
        for child in children
        if any(Path(argument).name == "squeakview_gui.py" for argument in child.argv)
    )
    capture = tuple(
        child
        for child in children
        if _has_argv_sequence(child.argv, ("-m", "squeakview.apps.inference.main"))
    )
    finalizer = tuple(
        child
        for child in children
        if _has_argv_sequence(child.argv, ("-m", "squeakview.apps.inference.post_run"))
    )
    return gui, capture, finalizer


def _environment(pid: int, *, proc_root: Path = Path("/proc")) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in _read_proc_file(proc_root / str(pid) / "environ").split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        result[key.decode("utf-8", "strict")] = value.decode("utf-8", "strict")
    return result


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate run-status key: {key}")
        result[key] = value
    return result


def _status(run_dir: Path) -> dict[str, object]:
    raw = read_stable_regular_file(
        run_dir / "run_status.json", max_bytes=MAX_STATUS_BYTES, label="run status"
    )
    value = json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=_unique_json_object)
    if not isinstance(value, dict):
        raise ValueError("run status must be a JSON object")
    return value


def build_plan(
    *,
    action: str,
    supervisor_pid: int,
    run_dir: Path,
    expected_phase: str,
    expected_stage: str | None,
    expected_supervisor_start: int | None,
    expected_target_pid: int | None,
    expected_target_start: int | None,
    execute: bool,
    confirmation: str | None,
    proc_root: Path = Path("/proc"),
    expected_barrier: str | None = None,
) -> InjectionPlan:
    """Resolve and validate one exact process tree without sending a signal."""

    if action not in SIGNALS:
        raise ValueError(f"unsupported action: {action}")
    if expected_phase not in ALLOWED_PHASES:
        raise ValueError(f"unsupported expected phase: {expected_phase}")
    if expected_barrier is not None and expected_barrier not in ALLOWED_BARRIERS:
        raise ValueError(f"unsupported expected barrier: {expected_barrier}")
    run_dir = run_dir.expanduser().resolve(strict=True)
    supervisor = _identity(supervisor_pid, proc_root=proc_root)
    if supervisor.uid != os.getuid():
        raise ValueError("supervisor UID does not match the invoking user")
    if not _has_argv_sequence(
        supervisor.argv, ("-m", "squeakview.apps.operator.backend.supervisor")
    ):
        raise ValueError("PID is not the SqueakView durable supervisor")
    if _environment(supervisor.pid, proc_root=proc_root).get(GATE) != "1":
        raise ValueError(f"supervisor was not launched with {GATE}=1")

    children = _children(supervisor.pid, proc_root=proc_root)
    gui_candidates, capture_candidates, finalizer_candidates = _process_roles(children)
    if len(gui_candidates) != 1:
        raise ValueError(f"expected exactly one supervisor-owned GUI, found {len(gui_candidates)}")
    gui = gui_candidates[0]
    capture: ProcessIdentity | None = None
    finalizer: ProcessIdentity | None = None
    if expected_barrier == "pre_capture":
        if expected_phase != "starting" or expected_stage is not None:
            raise ValueError("pre_capture barrier requires phase 'starting' and no stage")
        if capture_candidates or finalizer_candidates:
            raise ValueError("pre_capture barrier requires no capture or post-run child")
    elif expected_phase in ACTIVE_PHASES:
        if expected_stage is not None:
            raise ValueError("--expected-stage must be omitted for an active-capture phase")
        if len(capture_candidates) != 1 or finalizer_candidates:
            raise ValueError(
                "active phase requires exactly one supervisor-owned capture and no "
                f"post-run worker (found capture={len(capture_candidates)}, "
                f"post_run={len(finalizer_candidates)})"
            )
        capture = capture_candidates[0]
        if _run_dir_arg(capture) != run_dir:
            raise ValueError("supervisor-owned capture does not name the exact run directory")
    else:
        allowed_stages = FINALIZER_STAGES if expected_phase == "finalizing" else ANALYSIS_STAGES
        if expected_stage not in allowed_stages:
            raise ValueError(
                f"phase {expected_phase!r} requires --expected-stage in {allowed_stages}"
            )
        if capture_candidates or len(finalizer_candidates) != 1:
            raise ValueError(
                "post-capture phase requires no capture and exactly one supervisor-owned "
                f"post-run worker (found capture={len(capture_candidates)}, "
                f"post_run={len(finalizer_candidates)})"
            )
        finalizer = finalizer_candidates[0]
        if _post_run_dir_arg(finalizer) != run_dir:
            raise ValueError("supervisor-owned post-run worker does not name the exact run directory")
    workloads = tuple(item for item in (capture, finalizer) if item is not None)
    if gui.uid != supervisor.uid or any(item.uid != supervisor.uid for item in workloads):
        raise ValueError("supervisor, GUI, and run workload UIDs must match")

    status = _status(run_dir)
    if status.get("state") != expected_phase:
        raise ValueError(
            f"run phase changed: expected {expected_phase!r}, found {status.get('state')!r}"
        )
    observed_stage = status.get("stage")
    if expected_stage is not None and observed_stage != expected_stage:
        raise ValueError(
            f"run stage changed: expected {expected_stage!r}, found {observed_stage!r}"
        )
    observed_barrier = status.get("supervisor_failure_barrier")
    if expected_barrier is not None and (
        not isinstance(observed_barrier, dict)
        or observed_barrier.get("name") != expected_barrier
        or observed_barrier.get("state") != "waiting"
    ):
        raise ValueError(f"run is not waiting at barrier {expected_barrier!r}")
    if Path(str(status.get("run_directory", ""))).resolve() != run_dir:
        raise ValueError("run status directory does not match --run-dir")
    topology = status.get("process_topology")
    if not isinstance(topology, dict) or topology.get("acquisition_owner") != "durable_supervisor":
        raise ValueError("run is not owned by the durable supervisor")
    disqualifiers = status.get("production_disqualifiers")
    if (
        status.get("production_eligible") is not False
        or not isinstance(disqualifiers, list)
        or "supervisor_failure_injection" not in disqualifiers
    ):
        raise ValueError("run is not explicitly marked for supervisor failure qualification")

    target = supervisor if action == "supervisor-terminate" else gui
    if expected_supervisor_start is not None and supervisor.start_ticks != expected_supervisor_start:
        raise ValueError("supervisor start time changed; refusing stale PID")
    if expected_target_pid is not None and target.pid != expected_target_pid:
        raise ValueError("resolved signal target does not match --target-pid")
    if expected_target_start is not None and target.start_ticks != expected_target_start:
        raise ValueError("target start time changed; refusing stale PID")
    if execute:
        if expected_supervisor_start is None or expected_target_pid is None or expected_target_start is None:
            raise ValueError("--execute requires explicit supervisor/target PID start-time identities")
        if confirmation != CONFIRMATION:
            raise ValueError(f"--execute requires --confirm {CONFIRMATION}")
    return InjectionPlan(
        action=action,
        signal=int(SIGNALS[action]),
        supervisor=supervisor,
        gui=gui,
        capture=capture,
        finalizer=finalizer,
        target=target,
        run_dir=str(run_dir),
        phase=expected_phase,
        stage=expected_stage,
        barrier=expected_barrier,
        dry_run=not execute,
    )


def send_verified_signal(
    plan: InjectionPlan,
    *,
    proc_root: Path = Path("/proc"),
    kill=os.kill,
) -> None:
    """Recheck PID identity and phase immediately before the single signal."""

    if plan.dry_run:
        raise ValueError("cannot signal from a dry-run plan")
    current_supervisor = _identity(plan.supervisor.pid, proc_root=proc_root)
    current_target = _identity(plan.target.pid, proc_root=proc_root)
    if current_supervisor != plan.supervisor or current_target != plan.target:
        raise ValueError("process identity changed after validation; no signal sent")
    gui, capture, finalizer = _process_roles(
        _children(plan.supervisor.pid, proc_root=proc_root)
    )
    if gui != (plan.gui,):
        raise ValueError("supervisor-owned GUI relationship changed; no signal sent")
    run_dir = Path(plan.run_dir)
    if plan.barrier == "pre_capture":
        if capture or finalizer:
            raise ValueError("pre-capture workload relationship changed; no signal sent")
    elif plan.phase in ACTIVE_PHASES:
        if plan.capture is None or capture != (plan.capture,) or finalizer:
            raise ValueError("active capture workload relationship changed; no signal sent")
        if _run_dir_arg(capture[0]) != run_dir:
            raise ValueError("active capture run directory changed; no signal sent")
    else:
        if plan.finalizer is None or capture or finalizer != (plan.finalizer,):
            raise ValueError("post-run workload relationship changed; no signal sent")
        if _post_run_dir_arg(finalizer[0]) != run_dir:
            raise ValueError("post-run worker run directory changed; no signal sent")
    current_status = _status(Path(plan.run_dir))
    if current_status.get("state") != plan.phase:
        raise ValueError("run phase changed after validation; no signal sent")
    if plan.stage is not None and current_status.get("stage") != plan.stage:
        raise ValueError("run stage changed after validation; no signal sent")
    if plan.barrier is not None:
        barrier = current_status.get("supervisor_failure_barrier")
        if not isinstance(barrier, dict) or (
            barrier.get("name"), barrier.get("state")
        ) != (plan.barrier, "waiting"):
            raise ValueError("run barrier changed after validation; no signal sent")
    kill(plan.target.pid, plan.signal)


def wait_for_state(
    run_dir: Path,
    *,
    phase: str,
    stage: str | None,
    barrier: str | None,
    timeout_s: float,
    monotonic=time.monotonic,
    sleep=time.sleep,
) -> None:
    """Boundedly wait for an exact persisted phase/stage/barrier tuple."""

    if not 0 < timeout_s <= MAX_WAIT_TIMEOUT_S:
        raise ValueError(
            f"--wait-timeout must be greater than zero and at most {MAX_WAIT_TIMEOUT_S:g}"
        )
    deadline = monotonic() + timeout_s
    last_error = "status unavailable"
    while monotonic() < deadline:
        try:
            status = _status(run_dir)
            observed = status.get("supervisor_failure_barrier")
            barrier_matches = barrier is None or (
                isinstance(observed, dict)
                and observed.get("name") == barrier
                and observed.get("state") == "waiting"
            )
            if (
                status.get("state") == phase
                and (stage is None or status.get("stage") == stage)
                and barrier_matches
            ):
                return
            last_error = (
                f"observed phase={status.get('state')!r}, stage={status.get('stage')!r}, "
                f"barrier={observed!r}"
            )
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        sleep(min(0.1, max(0.0, deadline - monotonic())))
    raise ValueError(f"timed out waiting for requested run state: {last_error}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True, choices=tuple(SIGNALS))
    parser.add_argument("--supervisor-pid", required=True, type=int)
    parser.add_argument("--supervisor-start-ticks", type=int)
    parser.add_argument("--target-pid", type=int)
    parser.add_argument("--target-start-ticks", type=int)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-phase", required=True, choices=ALLOWED_PHASES)
    parser.add_argument(
        "--expected-stage", choices=(*FINALIZER_STAGES, *ANALYSIS_STAGES)
    )
    parser.add_argument("--expected-barrier", choices=ALLOWED_BARRIERS)
    parser.add_argument("--wait-timeout", type=float, default=0.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.wait_timeout:
            wait_for_state(
                args.run_dir.expanduser().resolve(),
                phase=args.expected_phase,
                stage=args.expected_stage,
                barrier=args.expected_barrier,
                timeout_s=args.wait_timeout,
            )
        plan = build_plan(
            action=args.action,
            supervisor_pid=args.supervisor_pid,
            run_dir=args.run_dir,
            expected_phase=args.expected_phase,
            expected_stage=args.expected_stage,
            expected_supervisor_start=args.supervisor_start_ticks,
            expected_target_pid=args.target_pid,
            expected_target_start=args.target_start_ticks,
            execute=args.execute,
            confirmation=args.confirm,
            expected_barrier=args.expected_barrier,
        )
        print(json.dumps(asdict(plan), indent=2, sort_keys=True))
        if args.execute:
            send_verified_signal(plan)
            print(f"Signal {plan.signal} sent to verified PID {plan.target.pid}")
        else:
            print("DRY RUN: no signal sent")
        return 0
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
