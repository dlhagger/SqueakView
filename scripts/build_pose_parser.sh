#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="${SQUEAKVIEW_POSE_BASELINE:-}"
if [ ! -f "$BASELINE" ]; then
  printf '[FAIL] Baseline parser missing: %s\n' "$BASELINE" >&2
  printf 'Set SQUEAKVIEW_POSE_BASELINE to an accepted model-package parser.\n' >&2
  exit 1
fi
BUILD_DIR="$APP_ROOT/native/yolo26_pose_parser/build-comparison"
cmake -S "$APP_ROOT/native/yolo26_pose_parser" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DDEEPSTREAM_SDK="${SQUEAKVIEW_DEEPSTREAM_SDK:-/opt/nvidia/deepstream/deepstream}" \
  -DBASELINE_PARSER="$(readlink -f "$BASELINE")"
cmake --build "$BUILD_DIR" --parallel "${SQUEAKVIEW_BUILD_JOBS:-2}"
ctest --test-dir "$BUILD_DIR" --output-on-failure --no-tests=error
printf '[PASS] Parser-only comparison build passed; existing model packages are unchanged.\n'
printf '[INFO] Candidate: %s/libnvdsinfer_squeakview_pose.so\n' "$BUILD_DIR"
