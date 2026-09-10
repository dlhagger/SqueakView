#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CAPTURE_BACKEND="${CAPTURE_BACKEND:-flir_direct}"
DEEPSTREAM_SDK="${SQUEAKVIEW_DEEPSTREAM_SDK:-/opt/nvidia/deepstream/deepstream}"
FLIR_PLUGIN_DIR="$ROOT/native/flir_gst_source/build"
INFERENCE_ENABLED="${INFERENCE_ENABLED:-1}"
SERIAL_ENABLED="${SERIAL_ENABLED:-0}"
SERIAL_PORT="${SERIAL_PORT:-/dev/ttyACM0}"
CFG="${DS_CFG:-${SQUEAKVIEW_DS_CFG:-}}"
if [ -z "$CFG" ] && [ -n "${SQUEAKVIEW_MODEL_NAME:-}" ]; then
  CFG="models/${SQUEAKVIEW_MODEL_NAME}/configs/${SQUEAKVIEW_MODEL_NAME}.txt"
fi

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

case "$INFERENCE_ENABLED" in
  0|false|FALSE|no|NO|off|OFF) INFERENCE_ENABLED=0 ;;
  *) INFERENCE_ENABLED=1 ;;
esac

case "$SERIAL_ENABLED" in
  1|true|TRUE|yes|YES|on|ON) SERIAL_ENABLED=1 ;;
  *) SERIAL_ENABLED=0 ;;
esac

if [ "$INFERENCE_ENABLED" -eq 1 ] && [ -z "$CFG" ]; then
  printf '[FAIL] Inference is enabled but no model was selected.\n'
  printf '       Set DS_CFG=models/<model_name>/configs/<model_name>.txt\n'
  printf '       or set SQUEAKVIEW_MODEL_NAME=<model_name>.\n'
  exit 1
fi

CFG="$(resolve_repo_path "$CFG")"
CFG_DIR=""
CFG_STEM=""
MODEL_DIR=""
POSE_META=""
ENGINE=""
PARSER=""
CLASS_LABELS=""
KEYPOINT_LABELS=""
if [ "$INFERENCE_ENABLED" -eq 1 ]; then
  CFG_DIR="$(cd "$(dirname "$CFG")" 2>/dev/null && pwd)"
  CFG_STEM="$(basename "$CFG" .txt)"
  MODEL_DIR="$(cd "$CFG_DIR/.." 2>/dev/null && pwd)"
  POSE_META="$CFG_DIR/$CFG_STEM.pose.json"
fi

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

if [ "$INFERENCE_ENABLED" -eq 1 ]; then
  ENGINE="$(resolve_infer_path "$(cfg_value model-engine-file)")"
  PARSER="$(resolve_infer_path "$(cfg_value custom-lib-path)")"
  CLASS_LABELS="$(resolve_infer_path "$(cfg_value labelfile-path)")"
  KEYPOINT_LABELS="$MODEL_DIR/labels/labels.txt"
fi

fail=0

pass() {
  printf '[PASS] %s\n' "$1"
}

info() {
  printf '[INFO] %s\n' "$1"
}

report_tracker_accelerator() {
  local device_model=""
  if [ -r /proc/device-tree/model ]; then
    device_model="$(tr -d '\000' < /proc/device-tree/model 2>/dev/null || true)"
  fi

  case "$device_model" in
    *"Jetson Orin Nano"*)
      info "PVA unavailable (expected on Jetson Orin Nano); CUDA NvDCF selected"
      ;;
  esac
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
}

check_desktop_suspend_policy() {
  if ! command -v gsettings >/dev/null 2>&1; then
    printf '[FAIL] Desktop suspend policy unavailable (gsettings is not installed).\n'
    printf '       Scientific acquisition requires explicit proof that automatic AC suspend is disabled.\n'
    fail=1
    return 0
  fi
  local timeout action
  timeout="$(gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-ac-timeout 2>/dev/null || true)"
  action="$(gsettings get org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 2>/dev/null || true)"
  timeout="${timeout//[^0-9]/}"
  if [ "$timeout" = "0" ]; then
    pass "Automatic desktop suspend on AC power is disabled"
  elif [ -n "$timeout" ] && [[ "$action" == *suspend* ]]; then
    printf '[FAIL] Automatic desktop suspend is enabled after %s second(s) on AC power.\n' "$timeout"
    printf '       Disable it before scientific acquisition; Jetson Linux 39.2.1 documents an Orin Nano SC7 resume watchdog risk.\n'
    fail=1
  else
    printf '[FAIL] Desktop suspend policy could not be determined.\n'
    printf '       Set sleep-inactive-ac-timeout to 0 before scientific acquisition.\n'
    fail=1
  fi
}

check_serial_access() {
  if [ "$SERIAL_ENABLED" -ne 1 ]; then
    return 0
  fi
  if ! getent group dialout >/dev/null; then
    printf '[FAIL] Serial access requires the dialout group, but it does not exist.\n'
    fail=1
    return 0
  fi
  if ! id -nG | tr ' ' '\n' | grep -Fxq dialout; then
    printf '[FAIL] This login session does not have effective dialout access.\n'
    if id -nG "$(id -un)" | tr ' ' '\n' | grep -Fxq dialout; then
      printf '       Group membership is configured, but this session is stale. Reboot, then retry.\n'
    else
      printf '       Run: sudo usermod -aG dialout $USER\n'
      printf '       Then reboot before starting SqueakView.\n'
    fi
    fail=1
    return 0
  fi
  pass "Current login has effective dialout access"
  if [ ! -e "$SERIAL_PORT" ]; then
    printf '[FAIL] Configured serial controller port is not present: %s\n' "$SERIAL_PORT"
    printf '       Connect the controller or select its actual /dev/ttyACM* device.\n'
    fail=1
  elif [ ! -r "$SERIAL_PORT" ] || [ ! -w "$SERIAL_PORT" ]; then
    printf '[FAIL] Serial controller port is not readable and writable: %s\n' "$SERIAL_PORT"
    fail=1
  else
    pass "Serial controller port is readable and writable: $SERIAL_PORT"
  fi
}

check_ffmpeg_decode_path() {
  if ! command -v ffmpeg >/dev/null 2>&1; then
    printf '[FAIL] FFmpeg is not installed.\n'
    printf '       Install it with: sudo apt install ffmpeg\n'
    fail=1
    return 0
  fi
  if ! command -v ffprobe >/dev/null 2>&1; then
    printf '[FAIL] ffprobe is not installed.\n'
    printf '       Install it with: sudo apt install ffmpeg\n'
    fail=1
    return 0
  fi
  pass "ffprobe is installed ($(command -v ffprobe))"

  local probe_dir probe_video probe_detail probe_status
  probe_dir="$(mktemp -d /tmp/squeakview-ffmpeg-preflight.XXXXXX)" || {
    printf '[FAIL] Could not create the FFmpeg preflight directory.\n'
    fail=1
    return 0
  }
  probe_video="$probe_dir/decoder-self-test.mp4"
  if ! timeout 20 gst-launch-1.0 -q \
      videotestsrc num-buffers=30 \
      ! video/x-raw,format=I420,width=64,height=48,framerate=30/1 \
      ! x264enc tune=zerolatency key-int-max=15 \
      ! h264parse \
      ! mp4mux \
      ! filesink location="$probe_video" >/dev/null 2>&1; then
    printf '[FAIL] Could not create the H.264 decoder self-test fixture.\n'
    fail=1
    rm -f -- "$probe_video"
    rmdir -- "$probe_dir" 2>/dev/null || true
    return 0
  fi

  probe_detail="$(
    PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    SQUEAKVIEW_VIDEO_VALIDATION_TIMEOUT_S=120 \
      "$PYTHON_BIN" -m squeakview.apps.inference.video_probe \
      "$probe_video" --expect-frames 30 2>&1
  )"
  probe_status=$?
  rm -f -- "$probe_video"
  rmdir -- "$probe_dir" 2>/dev/null || true
  if [ "$probe_status" -eq 0 ]; then
    pass "Jetson H.264 full-decode validation path works ($probe_detail)"
  else
    printf '[FAIL] Jetson H.264 full-decode validation path failed: %s\n' "$probe_detail"
    printf '       SqueakView requires NVIDIA nvv4l2decoder; FFmpeg h264_nvv4l2dec remains the fallback.\n'
    fail=1
  fi
}

check_new_streammux_path() {
  local mux_detail mux_probe_detail
  mux_detail="$(USE_NEW_NVSTREAMMUX=yes gst-inspect-1.0 nvstreammux 2>/dev/null)"
  case "$mux_detail" in
    *"config-file-path"*) ;;
    *)
      printf '[FAIL] DeepStream new nvstreammux could not be selected.\n'
      printf '       SqueakView requires DeepStream 9.1 Stream multiplexer 2.\n'
      fail=1
      return 0
      ;;
  esac
  case "$mux_detail" in
    *"live-source"*)
      printf '[FAIL] Legacy nvstreammux was selected instead of the new implementation.\n'
      fail=1
      return 0
      ;;
    *) ;;
  esac
  if ! mux_probe_detail="$(timeout 45 env USE_NEW_NVSTREAMMUX=yes gst-launch-1.0 -q \
      nvstreammux name=mux batch-size=1 batched-push-timeout=33333 \
        sync-inputs=false max-latency=0 \
      ! fakesink sync=false async=false \
      videotestsrc num-buffers=30 \
      ! video/x-raw,format=GRAY8,width=64,height=48,framerate=30/1 \
      ! nvvideoconvert compute-hw=VIC copy-hw=VIC \
        nvbuf-memory-type=nvbuf-mem-surface-array \
      ! 'video/x-raw(memory:NVMM),format=NV12,width=64,height=48' \
      ! mux.sink_0 2>&1)"; then
    printf '[FAIL] DeepStream new nvstreammux VIC/NVMM self-test failed: %s\n' \
      "$mux_probe_detail"
    printf '       Verify the DeepStream 9.1 and Jetson multimedia packages, then reboot.\n'
    fail=1
    return 0
  fi
  pass "DeepStream new nvstreammux VIC/NVMM path works"
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
printf 'Inference enabled: %s\n' "$INFERENCE_ENABLED"
printf 'Serial controller enabled: %s\n' "$SERIAL_ENABLED"
if [ "$SERIAL_ENABLED" -eq 1 ]; then
  printf 'Serial controller port: %s\n' "$SERIAL_PORT"
fi
printf 'DeepStream config: %s\n' "${CFG:-N/A}"
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
required_elements="nvstreammux nvstreamdemux nvurisrcbin nvunixfdsink nvunixfdsrc nvinfer nvtracker nvvideoconvert nvegltransform nveglglessink videotestsrc videoconvert x264enc h264parse mp4mux filesink"
for element in $required_elements; do
  if gst-inspect-1.0 "$element" >/dev/null 2>&1; then
    pass "GStreamer element '$element' is available"
  else
    printf '[FAIL] GStreamer element missing: %s\n' "$element"
    fail=1
  fi
done
check_new_streammux_path
check_ffmpeg_decode_path

if [ "$CAPTURE_BACKEND" = "flir_direct" ]; then
  if gst-inspect-1.0 flirspinsrc >/dev/null 2>&1; then
    pass "GStreamer element 'flirspinsrc' is available"
    if gst-inspect-1.0 flirspinsrc 2>/dev/null | grep -q 'capture-log-path'; then
      pass "GStreamer element 'flirspinsrc' supports the source capture ledger"
    else
      printf '[FAIL] flirspinsrc is stale: capture-log-path is unavailable\n'
      fail=1
    fi
  else
    printf '[FAIL] GStreamer element missing: flirspinsrc\n'
    printf '       Build it with: cmake -S native/flir_gst_source -B native/flir_gst_source/build && cmake --build native/flir_gst_source/build -j\n'
    fail=1
  fi
fi

if [ "$INFERENCE_ENABLED" -eq 1 ]; then
  TRACKER_LIB="$DEEPSTREAM_SDK/lib/libnvds_nvmultiobjecttracker.so"
  if [ ! -f "$TRACKER_LIB" ]; then
    printf '[FAIL] DeepStream tracker library missing: %s\n' "$TRACKER_LIB"
    fail=1
  elif ldd "$TRACKER_LIB" 2>/dev/null | grep -q 'not found'; then
    printf '[FAIL] DeepStream tracker has unresolved runtime dependencies:\n'
    ldd "$TRACKER_LIB" 2>/dev/null | grep 'not found'
    printf '       Install the DeepStream 9.1 prerequisites (including libmosquitto1).\n'
    fail=1
  else
    pass "DeepStream tracker runtime dependencies resolved"
  fi
  report_tracker_accelerator
  if "$PYTHON_BIN" -m squeakview.model_package --config "$CFG" --require-engine-identity; then
    pass "Selected model package validated"
  else
    fail=1
  fi
fi

check_tegrastats_memory
check_desktop_suspend_policy
check_serial_access

if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstVideo", "1.0")
from gi.repository import Gst, GstVideo
import pyservicemaker
from PySide6 import QtWidgets
PY
then
  pass "Python imports for Gst, PyServiceMaker, and PySide6 resolved"
else
  printf '[FAIL] Python imports failed for Gst, PyServiceMaker, or PySide6 in this environment\n'
  fail=1
fi

if [ "$fail" -eq 0 ]; then
  printf '\nAll preflight checks passed.\n'
else
  printf '\nPreflight checks failed.\n'
fi

exit "$fail"
