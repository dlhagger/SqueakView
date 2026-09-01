#!/usr/bin/env bash
set -euo pipefail

# Launch the operator independently of the invoking terminal/VS Code process.
# The GUI still owns the controller and capture lifecycle, but a terminal host
# disappearing will no longer orphan only the capture subprocess.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
LOG_DIR="${SQUEAKVIEW_LAUNCH_LOG_DIR:-$ROOT/runs/logs}"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_PATH="${SQUEAKVIEW_LOGFILE:-$LOG_DIR/squeakview_gui_$TIMESTAMP.log}"

if [ ! -x "$PYTHON_BIN" ]; then
  printf '[FAIL] Python environment is unavailable: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi
if ! command -v setsid >/dev/null 2>&1; then
  printf '[FAIL] setsid is required for a terminal-independent launch.\n' >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$(dirname "$LOG_PATH")"
SQUEAKVIEW_LOGFILE="$LOG_PATH" nohup setsid \
  "$PYTHON_BIN" "$ROOT/squeakview_gui.py" </dev/null >/dev/null 2>&1 &
launcher_pid=$!

sleep 0.5
if ! kill -0 "$launcher_pid" 2>/dev/null; then
  printf '[FAIL] SqueakView exited during launch. Inspect %s\n' "$LOG_PATH" >&2
  exit 1
fi

printf '[PASS] SqueakView launched independently (PID %s).\n' "$launcher_pid"
printf '       Operator log: %s\n' "$LOG_PATH"
