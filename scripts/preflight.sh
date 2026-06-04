#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CAPTURE_BACKEND="${CAPTURE_BACKEND:-flir_direct}"
DEEPSTREAM_SDK="${SQUEAKVIEW_DEEPSTREAM_SDK:-/opt/nvidia/deepstream/deepstream}"
FLIR_PLUGIN_DIR="$ROOT/native/flir_gst_source/build"
DEFAULT_MODEL_DIR="$ROOT/models/yolo26s_dino_10kpts_fp16"
CFG="${DS_CFG:-${SQUEAKVIEW_DS_CFG:-$DEFAULT_MODEL_DIR/configs/yolo26s_dino_10kpts_fp16.txt}}"

resolve_repo_path() {
  local raw="$1"
  if [ -z "$raw" ]; then
    return 0
  fi
  case "$raw" in
    /*)
      if [ -e "$raw" ]; then
        printf '%s\n' "$raw"
      elif [[ "$raw" == */models/* ]]; then
        printf '%s/models/%s\n' "$ROOT" "${raw#*/models/}"
      elif [[ "$raw" == */native/* ]]; then
        printf '%s/native/%s\n' "$ROOT" "${raw#*/native/}"
      elif [[ "$raw" == */tasks/* ]]; then
        printf '%s/tasks/%s\n' "$ROOT" "${raw#*/tasks/}"
      else
        printf '%s\n' "$raw"
      fi
      ;;
    *)
      printf '%s/%s\n' "$ROOT" "$raw"
      ;;
  esac
}

CFG="$(resolve_repo_path "$CFG")"
if [ ! -f "$CFG" ] && [ -d "$ROOT/models" ]; then
  FIRST_CFG="$(find "$ROOT/models" -path '*/configs/*.txt' -type f | sort | head -n 1 || true)"
  if [ -n "$FIRST_CFG" ]; then
    CFG="$FIRST_CFG"
  fi
fi
CFG_DIR="$(cd "$(dirname "$CFG")" 2>/dev/null && pwd)"
CFG_STEM="$(basename "$CFG" .txt)"
MODEL_DIR="$(cd "$CFG_DIR/.." 2>/dev/null && pwd)"
POSE_META="$CFG_DIR/$CFG_STEM.pose.json"

cfg_value() {
  local key="$1"
  awk -F= -v key="$key" '
    $1 == key {
      sub(/^[[:space:]]+/, "", $2)
      sub(/[[:space:]]+$/, "", $2)
      print $2
      exit
    }
  ' "$CFG"
}

resolve_infer_path() {
  local raw="$1"
  if [ -z "$raw" ]; then
    return 0
  fi
  case "$raw" in
    /*) resolve_repo_path "$raw" ;;
    *) printf '%s/%s\n' "$CFG_DIR" "$raw" ;;
  esac
}

ENGINE="$(resolve_infer_path "$(cfg_value model-engine-file)")"
PARSER="$(resolve_infer_path "$(cfg_value custom-lib-path)")"
CLASS_LABELS="$(resolve_infer_path "$(cfg_value labelfile-path)")"
KEYPOINT_LABELS="$MODEL_DIR/labels/labels.txt"

fail=0

pass() {
  printf '[PASS] %s\n' "$1"
}

warn() {
  printf '[WARN] %s\n' "$1"
}

check_tegrastats_memory() {
  if ! command -v tegrastats >/dev/null 2>&1; then
    return 0
  fi
  local line
  line="$(timeout 2 tegrastats 2>/dev/null | head -n 1 || true)"
  if [ -z "$line" ]; then
    return 0
  fi
  printf '[INFO] tegrastats: %s\n' "$line"
  local lfb_mb
  lfb_mb="$(printf '%s\n' "$line" | sed -n 's/.*lfb [0-9][0-9]*x\([0-9][0-9]*\)MB.*/\1/p')"
  if [ -n "$lfb_mb" ] && [ "$lfb_mb" -lt 16 ]; then
    warn "Low contiguous NvMap memory (lfb ${lfb_mb}MB). Close notebooks/other GPU apps or reboot before DeepStream if cuda/NvMap allocation fails."
  fi
}

check() {
  if "$@"; then
    pass "$*"
  else
    printf '[FAIL] %s\n' "$*"
    fail=1
  fi
}

printf '=== SqueakView Preflight ===\n'
printf 'Python: %s\n' "$PYTHON_BIN"
printf 'Capture backend: %s\n' "$CAPTURE_BACKEND"
printf 'DeepStream SDK: %s\n' "$DEEPSTREAM_SDK"
printf 'DeepStream config: %s\n' "$CFG"
printf '\n'

if [ -d "$DEEPSTREAM_SDK" ]; then
  pass "DeepStream SDK directory exists"
else
  printf '[FAIL] DeepStream SDK directory missing: %s\n' "$DEEPSTREAM_SDK"
  fail=1
fi

export LD_LIBRARY_PATH="$DEEPSTREAM_SDK/lib:${LD_LIBRARY_PATH:-}"
export GST_PLUGIN_PATH="$FLIR_PLUGIN_DIR:$DEEPSTREAM_SDK/lib/gst-plugins:${GST_PLUGIN_PATH:-}"

check command -v gst-inspect-1.0
for element in nvstreammux nvinfer nvvideoconvert nveglglessink videoconvert x264enc splitmuxsink h264parse; do
  if gst-inspect-1.0 "$element" >/dev/null 2>&1; then
    pass "GStreamer element '$element' is available"
  else
    printf '[FAIL] GStreamer element missing: %s\n' "$element"
    fail=1
  fi
done

if [ "$CAPTURE_BACKEND" = "flir_direct" ]; then
  if gst-inspect-1.0 flirspinsrc >/dev/null 2>&1; then
    pass "GStreamer element 'flirspinsrc' is available"
  else
    printf '[FAIL] GStreamer element missing: flirspinsrc\n'
    printf '       Build it with: cmake -S native/flir_gst_source -B native/flir_gst_source/build && cmake --build native/flir_gst_source/build -j\n'
    fail=1
  fi
fi

for path in "$CFG" "$POSE_META" "$ENGINE" "$PARSER" "$CLASS_LABELS" "$KEYPOINT_LABELS"; do
  if [ -e "$path" ]; then
    pass "Found $path"
  else
    printf '[FAIL] Missing %s\n' "$path"
    fail=1
  fi
done

check_tegrastats_memory

if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
import pyds
from PySide6 import QtWidgets
PY
then
  pass "Python imports for Gst, pyds, and PySide6 resolved"
else
  warn "Python imports failed for Gst, pyds, or PySide6 in this environment"
fi

if [ "$fail" -eq 0 ]; then
  printf '\nAll preflight checks passed.\n'
else
  printf '\nPreflight checks failed.\n'
fi

exit "$fail"
