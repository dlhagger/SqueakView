"""Bounded final-record access, including explicitly archived recovery files."""
import os
from pathlib import Path

MAX_TAIL_BYTES = 65_536
SUMMARY_NAME = 'recovery_tail_summary.json'


def identity(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def last_complete_line(path, *, run_dir=None):
    path = Path(path)
    if path.suffix == '.gz':
        if run_dir is None:
            raise ValueError('Compressed ledger requires a retained final-record summary')
        from .run_layout import artifact_path
        from .run_context import read_json_required
        try:
            payload = read_json_required(artifact_path(run_dir, SUMMARY_NAME))
            item = payload['records'][str(path.relative_to(run_dir))]
            if payload['schema_version'] != 1 or item['identity'] != identity(path):
                raise ValueError('Compressed ledger changed since its final-record summary')
            line = item['last_complete_line']
            if not isinstance(line, str) or len(line.encode('utf-8')) > MAX_TAIL_BYTES:
                raise ValueError('Invalid retained final record')
            return line
        except (KeyError, OSError) as exc:
            raise ValueError('Compressed ledger lacks a matching final-record summary; use explicit --full-analysis to recover historical evidence') from exc
    with path.open('rb') as handle:
        handle.seek(0, os.SEEK_END)
        end = handle.tell()
        if end <= 0:
            return None
        handle.seek(max(0, end - MAX_TAIL_BYTES))
        data = handle.read(MAX_TAIL_BYTES)
    lines = data.splitlines()
    if data and not data.endswith((b'\n', b'\r')):
        lines = lines[:-1]
    for line in reversed(lines):
        value = line.decode('utf-8', errors='strict').strip()
        if value:
            return value
    return None


def retain_final_record(run_dir, path, line):
    from .run_layout import artifact_path
    from .run_context import read_json_required, atomic_write_json
    summary = artifact_path(run_dir, SUMMARY_NAME)
    payload = read_json_required(summary) if summary.exists() else {'schema_version': 1, 'records': {}}
    payload['records'][str(Path(path).relative_to(run_dir))] = {
        'identity': identity(path), 'last_complete_line': line,
    }
    atomic_write_json(summary, payload)
