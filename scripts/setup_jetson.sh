#!/usr/bin/env bash
set -euo pipefail

# Provision OS-level dependencies and serial permissions, then invoke the
# separate unprivileged native builder. Run this as the intended desktop user;
# the script invokes sudo only for privileged steps.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

TARGET_USER_HOME="$(getent passwd "$TARGET_USER" | cut -d: -f6)"
if [ -z "$TARGET_USER_HOME" ] || [ ! -d "$TARGET_USER_HOME" ]; then
  printf '[FAIL] Could not resolve the home directory for %s.\n' "$TARGET_USER" >&2
  exit 2
fi

PROJECTS_PARENT_RAW="${SQUEAKVIEW_PROJECTS_DIR:-$TARGET_USER_HOME/Documents/SqueakView Projects}"
case "$PROJECTS_PARENT_RAW" in
  /*) ;;
  *)
    printf '[FAIL] SQUEAKVIEW_PROJECTS_DIR must be an absolute path: %s\n' "$PROJECTS_PARENT_RAW" >&2
    exit 2
    ;;
esac
if [ -L "$PROJECTS_PARENT_RAW" ] && [ ! -e "$PROJECTS_PARENT_RAW" ]; then
  printf '[FAIL] Project parent is a dangling symbolic link: %s\n' "$PROJECTS_PARENT_RAW" >&2
  exit 2
fi
PROJECTS_PARENT="$(readlink -m -- "$PROJECTS_PARENT_RAW")"
case "$PROJECTS_PARENT" in
  "$ROOT"|"$ROOT"/*)
    printf '[FAIL] Project parent must be outside the application checkout: %s\n' "$PROJECTS_PARENT" >&2
    exit 2
    ;;
esac
case "$ROOT" in
  "$PROJECTS_PARENT"|"$PROJECTS_PARENT"/*)
    printf '[FAIL] Project parent may not contain the application checkout: %s\n' "$PROJECTS_PARENT" >&2
    exit 2
    ;;
esac
if [ -L "$PROJECTS_PARENT" ] || { [ -e "$PROJECTS_PARENT" ] && [ ! -d "$PROJECTS_PARENT" ]; }; then
  printf '[FAIL] Project parent exists but is not a real directory: %s\n' "$PROJECTS_PARENT" >&2
  exit 2
elif [ -d "$PROJECTS_PARENT" ]; then
  printf '[PASS] Existing project parent left unchanged: %s\n' "$PROJECTS_PARENT"
elif [ "$(id -u)" -eq 0 ]; then
  TARGET_GROUP="$(id -gn "$TARGET_USER")"
  install -d -m 0700 -o "$TARGET_USER" -g "$TARGET_GROUP" "$PROJECTS_PARENT"
  printf '[PASS] Created project parent: %s\n' "$PROJECTS_PARENT"
else
  mkdir -p -- "$PROJECTS_PARENT"
  chmod 0700 "$PROJECTS_PARENT"
  printf '[PASS] Created project parent: %s\n' "$PROJECTS_PARENT"
fi

printf 'Configuring this Jetson for SqueakView (desktop user: %s)\n' "$TARGET_USER"
"${AS_ROOT[@]}" apt-get install -y \
  ffmpeg \
  build-essential \
  cmake \
  pkg-config \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev

if [ "$(id -u)" -eq 0 ]; then
  if ! command -v runuser >/dev/null 2>&1; then
    printf '[FAIL] runuser is required to keep native build outputs owned by %s.\n' "$TARGET_USER" >&2
    exit 1
  fi
  BUILD_ENV=()
  for name in CUDA_VER SQUEAKVIEW_BUILD_JOBS SQUEAKVIEW_DEEPSTREAM_SDK GST_PLUGIN_PATH; do
    if [ -n "${!name:-}" ]; then
      BUILD_ENV+=("$name=${!name}")
    fi
  done
  runuser -u "$TARGET_USER" -- env "${BUILD_ENV[@]}" \
    bash "$ROOT/scripts/build_native.sh"
else
  bash "$ROOT/scripts/build_native.sh"
fi

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
