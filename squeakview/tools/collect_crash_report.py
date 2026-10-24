"""Collect bounded, read-only crash evidence; never signal a live rig."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess

from squeakview.project import AppPaths, ProjectPaths, UserPaths, validate_external_output_path


TAIL_BYTES = 64 * 1024


def tail(path: Path, limit: int = TAIL_BYTES) -> dict:
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            data = stream.read(limit)
        return {"path": str(path), "size_bytes": size, "truncated": size > limit,
                "tail": data.decode("utf-8", errors="replace")}
    except OSError as exc:
        return {"path": str(path), "error": str(exc)}


def collect(logs: Path, project: Path | None = None) -> dict:
    report = {"schema_version": 1, "collected_utc": datetime.now(timezone.utc).isoformat(),
              "hostname": platform.node(), "platform": platform.platform(),
              "python": platform.python_version(), "logs": [], "runs": [],
              "limitations": "SIGKILL does not run Python crash handlers. A -9 exit alone does not establish OOM; compare kernel evidence."}
    try:
        paths = sorted(logs.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:16]
        report["logs"] = [tail(path) for path in paths]
    except OSError as exc:
        report["log_discovery_error"] = str(exc)
    report["memory"] = tail(Path("/proc/meminfo"))
    try:
        result = subprocess.run(
            ["journalctl", "-k", "--no-pager", "-n", "150", "--grep",
             "Out of memory|oom-kill|Killed process|segfault|trap|NVRM|Xid"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5,
        )
        report["kernel"] = {"returncode": result.returncode,
            "stdout": result.stdout[-TAIL_BYTES:], "stderr": result.stderr[-8192:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["kernel"] = {"error": str(exc)}
    if project is not None:
        try:
            statuses = sorted((project / "runs").rglob("run_status.json"),
                              key=lambda p: p.stat().st_mtime, reverse=True)[:3]
            for status in statuses:
                run = {"status": tail(status)}
                run["diagnostics"] = [tail(status.parent / "diagnostics" / name)
                    for name in ("supervisor_recovery.jsonl", "gui_stalls.jsonl", "deepstream.log",
                                 "controller_v2_summary.json", "post_run.log")]
                report["runs"].append(run)
        except OSError as exc:
            report["run_discovery_error"] = str(exc)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project", type=Path)
    args = parser.parse_args(argv)
    app, user = AppPaths.discover(), UserPaths.discover()
    user.validate_for_app(app)
    project = ProjectPaths.from_existing_root(args.project) if args.project else None
    output = validate_external_output_path(args.output, app=app, project=project, label="crash report")
    report = collect(user.launch_logs, project.root if project else None)
    # Exclusive creation protects an earlier report from accidental overwrite.
    output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
