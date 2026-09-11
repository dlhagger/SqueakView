#!/usr/bin/env bash
set -euo pipefail

# Launch a detached, Qt-free supervisor which owns acquisition and launches the
# mandatory GUI. GUI loss is observable and forces fail-closed run finalization;
# the supervisor never continues acquisition headlessly.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
LOG_DIR="${SQUEAKVIEW_LAUNCH_LOG_DIR:-$ROOT/runs/logs}"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
GUI_LOG_PATH="${SQUEAKVIEW_LOGFILE:-$LOG_DIR/squeakview_gui_$TIMESTAMP.log}"
SUPERVISOR_LOG_PATH="${SQUEAKVIEW_SUPERVISOR_LOGFILE:-$LOG_DIR/squeakview_supervisor_$TIMESTAMP.log}"
LAUNCH_TIMEOUT_S="${SQUEAKVIEW_LAUNCH_TIMEOUT_S:-35}"

if [ ! -x "$PYTHON_BIN" ]; then
  printf '[FAIL] Python environment is unavailable: %s\n' "$PYTHON_BIN" >&2
  printf '       Run "uv sync" from %s, then try again.\n' "$ROOT" >&2
  exit 1
fi
if ! command -v setsid >/dev/null 2>&1; then
  printf '[FAIL] setsid is required for a terminal-independent launch.\n' >&2
  exit 1
fi

cd "$ROOT"
mkdir -p "$LOG_DIR" "$(dirname "$GUI_LOG_PATH")" "$(dirname "$SUPERVISOR_LOG_PATH")"
case "$LAUNCH_TIMEOUT_S" in
  ''|*[!0-9]*)
    printf '[FAIL] SQUEAKVIEW_LAUNCH_TIMEOUT_S must be a positive integer.\n' >&2
    exit 2
    ;;
  0)
    printf '[FAIL] SQUEAKVIEW_LAUNCH_TIMEOUT_S must be greater than zero.\n' >&2
    exit 2
    ;;
esac
umask 077
LAUNCH_STATUS_PATH="$(mktemp "$LOG_DIR/.squeakview_launch.XXXXXX")"
supervisor_pid=""
launch_complete=0
launch_stopped=0
cleanup_launch_status() {
  rm -f -- "$LAUNCH_STATUS_PATH"
  if [ "$launch_complete" -eq 0 ] && [ "$launch_stopped" -eq 0 ] && [ -n "$supervisor_pid" ]; then
    stop_failed_launch
  fi
}
stop_failed_launch() {
  launch_stopped=1
  kill "$supervisor_pid" 2>/dev/null || true
  launch_stop_waited=0
  while kill -0 "$supervisor_pid" 2>/dev/null && [ "$launch_stop_waited" -lt 25 ]; do
    sleep 0.2
    launch_stop_waited=$((launch_stop_waited + 1))
  done
  if kill -0 "$supervisor_pid" 2>/dev/null; then
    kill -KILL "$supervisor_pid" 2>/dev/null || true
  fi
  wait "$supervisor_pid" 2>/dev/null || true
}
trap cleanup_launch_status EXIT
SQUEAKVIEW_LOGFILE="$GUI_LOG_PATH" \
SQUEAKVIEW_SUPERVISOR_LOGFILE="$SUPERVISOR_LOG_PATH" \
SQUEAKVIEW_LAUNCH_STATUS_FILE="$LAUNCH_STATUS_PATH" nohup setsid \
  "$PYTHON_BIN" -m squeakview.apps.operator.backend.supervisor \
  --gui-command "$PYTHON_BIN" "$ROOT/squeakview_gui.py" \
  </dev/null >/dev/null 2>&1 &
supervisor_pid=$!

launch_waited=0
while [ "$launch_waited" -lt "$LAUNCH_TIMEOUT_S" ]; do
  if [ "$(cat "$LAUNCH_STATUS_PATH" 2>/dev/null || true)" = "GUI_READY" ]; then
    break
  fi
  if ! kill -0 "$supervisor_pid" 2>/dev/null; then
    printf '[FAIL] SqueakView supervisor or GUI exited during launch. Inspect %s\n' "$SUPERVISOR_LOG_PATH" >&2
    exit 1
  fi
  sleep 1
  launch_waited=$((launch_waited + 1))
done
if [ "$(cat "$LAUNCH_STATUS_PATH" 2>/dev/null || true)" != "GUI_READY" ]; then
  printf '[FAIL] The GUI did not connect within %s second(s). Inspect %s\n' "$LAUNCH_TIMEOUT_S" "$SUPERVISOR_LOG_PATH" >&2
  stop_failed_launch
  exit 1
fi

launch_complete=1
printf '[PASS] SqueakView durable supervisor launched (PID %s).\n' "$supervisor_pid"
printf '       GUI log: %s\n' "$GUI_LOG_PATH"
printf '       Supervisor log: %s\n' "$SUPERVISOR_LOG_PATH"
