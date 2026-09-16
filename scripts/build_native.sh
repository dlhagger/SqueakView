#!/usr/bin/env bash
set -euo pipefail

# Build and validate application-owned native components without sudo. System
# packages and device permissions are handled separately by setup_jetson.sh.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
APP_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
BUILD_JOBS="${SQUEAKVIEW_BUILD_JOBS:-$(nproc)}"

required_paths=(
  /opt/spinnaker/include
  /opt/spinnaker/lib
  /opt/nvidia/deepstream/deepstream/sources/includes
  /usr/local/cuda/bin/nvcc
)
for path in "${required_paths[@]}"; do
  if [ ! -e "$path" ]; then
    printf '[FAIL] Required vendor SDK path is missing: %s\n' "$path" >&2
    exit 1
  fi
done

CUDA_VER="${CUDA_VER:-$(/usr/local/cuda/bin/nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)}"
if [ -z "$CUDA_VER" ] || [ ! -x "/usr/local/cuda-$CUDA_VER/bin/nvcc" ]; then
  printf '[FAIL] Could not resolve a versioned CUDA toolkit from nvcc (detected: %s).\n' "${CUDA_VER:-none}" >&2
  exit 1
fi

printf 'Building FLIR GStreamer source with %s parallel job(s)...\n' "$BUILD_JOBS"
cmake \
  -S "$APP_ROOT/native/flir_gst_source" \
  -B "$APP_ROOT/native/flir_gst_source/build"
cmake --build "$APP_ROOT/native/flir_gst_source/build" --parallel "$BUILD_JOBS"

printf 'Building DeepStream YOLO parser against CUDA %s...\n' "$CUDA_VER"
make \
  -C "$APP_ROOT/native/nvdsinfer_custom_impl_yolo" \
  "CUDA_VER=$CUDA_VER" \
  -j"$BUILD_JOBS"

FLIR_PLUGIN="$APP_ROOT/native/flir_gst_source/build/gstflirspinsrc.so"
YOLO_PARSER="$APP_ROOT/native/nvdsinfer_custom_impl_yolo/libnvdsinfer_custom_impl_Yolo.so"
for output in "$FLIR_PLUGIN" "$YOLO_PARSER"; do
  if [ ! -s "$output" ]; then
    printf '[FAIL] Required native build output is missing or empty: %s\n' "$output" >&2
    exit 1
  fi
  if ldd "$output" 2>/dev/null | grep -q 'not found'; then
    printf '[FAIL] Native build output has unresolved runtime dependencies: %s\n' "$output" >&2
    ldd "$output" 2>/dev/null | grep 'not found' >&2 || true
    exit 1
  fi
done

if ! nm -D "$YOLO_PARSER" 2>/dev/null | grep -Eq '[[:space:]]NvDsInferParseYolo26Pose$'; then
  printf '[FAIL] DeepStream parser is missing required symbol NvDsInferParseYolo26Pose: %s\n' "$YOLO_PARSER" >&2
  exit 1
fi

FLIR_INSPECT="$(
  GST_PLUGIN_PATH="$APP_ROOT/native/flir_gst_source/build:/opt/nvidia/deepstream/deepstream/lib/gst-plugins${GST_PLUGIN_PATH:+:$GST_PLUGIN_PATH}" \
    gst-inspect-1.0 flirspinsrc
)"
for property in capture-log-path frame-manifest-path camera-telemetry-path error-log-path camera-runtime-path; do
  case "$FLIR_INSPECT" in
    *"$property"*) ;;
    *)
      printf '[FAIL] Built flirspinsrc is stale: required property %s is unavailable.\n' "$property" >&2
      exit 1
      ;;
  esac
done
printf '[PASS] Native FLIR and DeepStream components built successfully.\n'
