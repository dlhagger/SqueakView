#!/usr/bin/env bash
set -euo pipefail

# Build and validate application-owned native components without sudo. System
# packages and device permissions are handled separately by setup_jetson.sh.

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
APP_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
BUILD_JOBS="${SQUEAKVIEW_BUILD_JOBS:-$(nproc)}"

DEEPSTREAM_SDK="$(readlink -f -- "${SQUEAKVIEW_DEEPSTREAM_SDK:-/opt/nvidia/deepstream/deepstream}")"
PARSER_SELECTOR="$APP_ROOT/squeakview/common/parser_selection.py"
DEEPSTREAM_VERSION="$(python3 "$PARSER_SELECTOR" --sdk "$DEEPSTREAM_SDK" --version)"
DEEPSTREAM_INCLUDES="$(python3 "$PARSER_SELECTOR" --sdk "$DEEPSTREAM_SDK" --includes)"
PARSER_VARIANT="$(python3 "$PARSER_SELECTOR" --sdk "$DEEPSTREAM_SDK" --native "$APP_ROOT/native" --variant)"
YOLO_PARSER="$(python3 "$PARSER_SELECTOR" --sdk "$DEEPSTREAM_SDK" --native "$APP_ROOT/native")"
printf '[INFO] Detected DeepStream %s: %s (parser: %s)\n' "$DEEPSTREAM_VERSION" "$DEEPSTREAM_SDK" "$PARSER_VARIANT"
printf '[INFO] DeepStream headers: %s\n' "$DEEPSTREAM_INCLUDES"

required_paths=(
  /opt/spinnaker/include
  /opt/spinnaker/lib
  "$DEEPSTREAM_INCLUDES"
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
  -B "$APP_ROOT/native/flir_gst_source/build" \
  -DDEEPSTREAM_SDK="$DEEPSTREAM_SDK" \
  -DDEEPSTREAM_INCLUDES="$DEEPSTREAM_INCLUDES"
cmake --build "$APP_ROOT/native/flir_gst_source/build" --parallel "$BUILD_JOBS"

# Share the exact SDK-version policy with the model-package builder.
if [ "$PARSER_VARIANT" = yolo26 ]; then
  printf 'Building parser-only YOLO26 library for DeepStream 9.2...\n'
  cmake -S "$APP_ROOT/native/yolo26_pose_parser" -B "$APP_ROOT/native/yolo26_pose_parser/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=ON \
    -DBASELINE_PARSER= \
    -DDEEPSTREAM_SDK="$DEEPSTREAM_SDK" \
    -DDEEPSTREAM_INCLUDES="$DEEPSTREAM_INCLUDES"
  cmake --build "$APP_ROOT/native/yolo26_pose_parser/build" --parallel "$BUILD_JOBS"
  ctest --test-dir "$APP_ROOT/native/yolo26_pose_parser/build" --output-on-failure --no-tests=error
else
  printf 'Building legacy DeepStream 9.1 YOLO parser against CUDA %s...\n' "$CUDA_VER"
  make -C "$APP_ROOT/native/nvdsinfer_custom_impl_yolo" \
    "CUDA_VER=$CUDA_VER" "DEEPSTREAM_SDK=$DEEPSTREAM_SDK" \
    "DEEPSTREAM_INCLUDES=$DEEPSTREAM_INCLUDES" -j"$BUILD_JOBS"
fi

FLIR_PLUGIN="$APP_ROOT/native/flir_gst_source/build/gstflirspinsrc.so"
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

# Read all nm output: grep -q can close the pipe early and make nm fail with
# SIGPIPE under pipefail even when the required symbol is present.
if ! nm -D "$YOLO_PARSER" 2>/dev/null | grep -E '[[:space:]]NvDsInferParseYolo26Pose$' >/dev/null; then
  printf '[FAIL] DeepStream parser is missing required symbol NvDsInferParseYolo26Pose: %s\n' "$YOLO_PARSER" >&2
  exit 1
fi

FLIR_INSPECT="$(
  GST_PLUGIN_PATH="$APP_ROOT/native/flir_gst_source/build:$DEEPSTREAM_SDK/lib/gst-plugins${GST_PLUGIN_PATH:+:$GST_PLUGIN_PATH}" \
    gst-inspect-1.0 flirspinsrc
)"
for property in capture-health-path frame-manifest-path camera-telemetry-path error-log-path camera-runtime-path native-timing-validation graceful-eos live-mailbox-path; do
  case "$FLIR_INSPECT" in
    *"$property"*) ;;
    *)
      printf '[FAIL] Built flirspinsrc is stale: required property %s is unavailable.\n' "$property" >&2
      exit 1
      ;;
  esac
done
printf '[PASS] Native FLIR and DeepStream components built successfully.\n'
