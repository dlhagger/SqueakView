#!/usr/bin/env bash
set -euo pipefail

# Install SqueakView's OS-level dependencies, build its native components, and
# grant the desktop user access to USB serial controllers. Run this as the
# intended desktop user; the script invokes sudo only for privileged steps.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_JOBS="${SQUEAKVIEW_BUILD_JOBS:-$(nproc)}"

if [ "$(id -u)" -eq 0 ]; then
  TARGET_USER="${SQUEAKVIEW_DESKTOP_USER:-${SUDO_USER:-}}"
  if [ -z "$TARGET_USER" ] || [ "$TARGET_USER" = "root" ]; then
    printf '[FAIL] Cannot determine the desktop user from a root shell.\n' >&2
    printf '       Run this script as that user, or set SQUEAKVIEW_DESKTOP_USER.\n' >&2
    exit 2
  fi
  AS_ROOT=()
else
  TARGET_USER="${SQUEAKVIEW_DESKTOP_USER:-$(id -un)}"
  AS_ROOT=(sudo)
fi

if ! id "$TARGET_USER" >/dev/null 2>&1; then
  printf '[FAIL] User does not exist: %s\n' "$TARGET_USER" >&2
  exit 2
fi

printf 'Configuring this Jetson for SqueakView (desktop user: %s)\n' "$TARGET_USER"
"${AS_ROOT[@]}" apt-get install -y \
  ffmpeg \
  build-essential \
  cmake \
  pkg-config \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev

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
  -S "$ROOT/native/flir_gst_source" \
  -B "$ROOT/native/flir_gst_source/build"
cmake --build "$ROOT/native/flir_gst_source/build" --parallel "$BUILD_JOBS"

printf 'Building DeepStream YOLO parser against CUDA %s...\n' "$CUDA_VER"
make \
  -C "$ROOT/native/nvdsinfer_custom_impl_yolo" \
  "CUDA_VER=$CUDA_VER" \
  -j"$BUILD_JOBS"

GST_PLUGIN_PATH="$ROOT/native/flir_gst_source/build:/opt/nvidia/deepstream/deepstream/lib/gst-plugins${GST_PLUGIN_PATH:+:$GST_PLUGIN_PATH}" \
  gst-inspect-1.0 flirspinsrc >/dev/null
printf '[PASS] Native FLIR and DeepStream components built successfully.\n'

if ! getent group dialout >/dev/null; then
  printf '[FAIL] The required dialout group does not exist on this system.\n' >&2
  exit 1
fi

if id -nG "$TARGET_USER" | tr ' ' '\n' | grep -Fxq dialout; then
  printf '[PASS] %s already has serial-port access through dialout.\n' "$TARGET_USER"
else
  "${AS_ROOT[@]}" usermod -aG dialout "$TARGET_USER"
  printf '[PASS] Added %s to dialout.\n' "$TARGET_USER"
  printf '[ACTION REQUIRED] Sign out and back in (or reboot) before using a serial controller.\n'
fi

printf '[PASS] FFmpeg/ffprobe is installed.\n'
printf '[SETUP COMPLETE] Reboot this Jetson before starting SqueakView.\n'
