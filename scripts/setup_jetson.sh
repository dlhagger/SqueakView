#!/usr/bin/env bash
set -euo pipefail

# Provision OS dependencies, power settings, firmware workaround, and the UV
# environment, then invoke the separate unprivileged native builder. Run as the
# intended desktop user; the script invokes sudo only for privileged steps.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

step() {
  printf '\n[%s/7] %s\n' "$1" "$2"
  printf '%s\n' '------------------------------------------------------------'
}

printf '\n%s\n' 'SqueakView | Jetson setup'
printf '%s\n' '============================================================'
step 1 'Check desktop user and prepare project storage'

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

# Keep Python environments, caches, and native builds owned by the desktop user.
if [ "$(id -u)" -ne 0 ] && [ "$TARGET_USER" != "$(id -un)" ]; then
  printf '[FAIL] Run as the intended desktop user, or use sudo with SQUEAKVIEW_DESKTOP_USER.\n' >&2
  exit 2
fi
if [ "$(id -u)" -eq 0 ] && ! command -v runuser >/dev/null 2>&1; then
  printf '[FAIL] runuser is required for unprivileged setup steps.\n' >&2
  exit 2
fi
run_as_desktop() {
  if [ "$(id -u)" -eq 0 ]; then
    runuser -u "$TARGET_USER" -- env HOME="$TARGET_USER_HOME" \
      PATH="$TARGET_USER_HOME/.local/bin:$PATH" "$@"
  else
    env PATH="$TARGET_USER_HOME/.local/bin:$PATH" "$@"
  fi
}
if ! run_as_desktop sh -c 'command -v uv >/dev/null'; then
  printf '[FAIL] Install uv for %s before running setup (https://docs.astral.sh/uv/getting-started/installation/).\n' "$TARGET_USER" >&2
  exit 2
fi

printf '[INFO] Desktop user: %s\n' "$TARGET_USER"
printf '[INFO] Application:  %s\n' "$ROOT"

# Validate the installed SDK before changing system packages or desktop policy.
# Keep that exact SDK root for the later unprivileged build.
DEEPSTREAM_SDK="$(readlink -f -- "${SQUEAKVIEW_DEEPSTREAM_SDK:-/opt/nvidia/deepstream/deepstream}")"
PARSER_SELECTOR="$ROOT/squeakview/common/parser_selection.py"
DEEPSTREAM_VERSION="$(python3 "$PARSER_SELECTOR" --sdk "$DEEPSTREAM_SDK" --version)"
DEEPSTREAM_INCLUDES="$(python3 "$PARSER_SELECTOR" --sdk "$DEEPSTREAM_SDK" --includes)"
PARSER_VARIANT="$(python3 "$PARSER_SELECTOR" --sdk "$DEEPSTREAM_SDK" --variant)"
export SQUEAKVIEW_DEEPSTREAM_SDK="$DEEPSTREAM_SDK"
printf '[INFO] Detected DeepStream %s: %s (parser: %s)\n' "$DEEPSTREAM_VERSION" "$DEEPSTREAM_SDK" "$PARSER_VARIANT"
printf '[INFO] DeepStream headers: %s\n' "$DEEPSTREAM_INCLUDES"

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

step 2 'Install system packages'
printf '[INFO] Media:     VLC, FFmpeg, and ffprobe\n'
printf '[INFO] Build:     C/C++ toolchain, CMake, and pkg-config\n'
printf '[INFO] GStreamer: tools, plugins, and development libraries\n'
printf '[INFO] Package installation may request your sudo password.\n'
"${AS_ROOT[@]}" apt-get update
"${AS_ROOT[@]}" apt-get install -y \
  ffmpeg \
  vlc \
  build-essential \
  cmake \
  pkg-config \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-libav \
  libgstrtspserver-1.0-0 \
  libssl-dev \
  libcurl4-openssl-dev \
  libjansson4 \
  libyaml-cpp-dev \
  libmosquitto1 \
  libmosquitto-dev \
  libusb-1.0-0 \
  qtbase5-dev \
  qtchooser \
  qt5-qmake \
  qtbase5-dev-tools \
  python3.12 \
  python3.12-venv \
  python3-gi \
  gir1.2-gstreamer-1.0 \
  gir1.2-gst-plugins-base-1.0 \
  dbus-daemon \
  libglib2.0-bin \
  gnome-settings-daemon

printf '[PASS] VLC is installed.\n'
printf '[PASS] FFmpeg/ffprobe is installed.\n'
printf '[PASS] Build tools and GStreamer dependencies are installed.\n'

# Configure serial permissions before compilation so a failed native build
# cannot skip this device-provisioning step.
step 3 'Configure serial-controller access'
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

# PlatformIO's upstream rules include RP2040 bootloader USB access. Keep a
# reviewed copy in the checkout so provisioning does not depend on a download.
"${AS_ROOT[@]}" install -d -m 0755 /etc/udev/rules.d
"${AS_ROOT[@]}" install -m 0644 "$ROOT/scripts/udev/99-platformio-udev.rules" /etc/udev/rules.d/99-platformio-udev.rules
"${AS_ROOT[@]}" udevadm control --reload-rules
# Apply on the next connection; avoid triggering devices during an acquisition.
printf '[PASS] PlatformIO USB/bootloader udev rules installed and reloaded.\n'
printf '[ACTION REQUIRED] Unplug and reconnect the controller before flashing to apply the rules.\n'

# Install a fixed root-owned helper; normal acquisition never needs root.
USB_HELPER=/usr/local/libexec/squeakview-usb-recovery
USB_SUDOERS=/etc/sudoers.d/squeakview-usb-recovery
USB_TMPFILES=/etc/tmpfiles.d/squeakview-usb.conf
"${AS_ROOT[@]}" install -d -m 0755 /usr/local/libexec /etc/sudoers.d /etc/tmpfiles.d
"${AS_ROOT[@]}" install -m 0755 "$ROOT/scripts/jetson_usb_recovery.py" "$USB_HELPER"
"${AS_ROOT[@]}" install -m 0644 "$ROOT/scripts/systemd/squeakview-usb.conf" "$USB_TMPFILES"
"${AS_ROOT[@]}" systemd-tmpfiles --create "$USB_TMPFILES"
TARGET_UID="$(id -u "$TARGET_USER")"
USB_SUDOERS_TEMP="$(mktemp)"
printf '#%s ALL=(root) NOPASSWD: NOSETENV: %s ""\n' "$TARGET_UID" "$USB_HELPER" > "$USB_SUDOERS_TEMP"
if "${AS_ROOT[@]}" visudo -cf "$USB_SUDOERS_TEMP"; then
  "${AS_ROOT[@]}" install -m 0440 "$USB_SUDOERS_TEMP" "$USB_SUDOERS"
else
  rm -f -- "$USB_SUDOERS_TEMP"
  printf '[FAIL] USB recovery sudo policy did not validate.\n' >&2
  exit 1
fi
rm -f -- "$USB_SUDOERS_TEMP"
printf '[PASS] Idle-only automatic controller USB recovery installed.\n'

step 4 'Configure GPU firmware search path'
# Community workaround for the R39.2.x nvgpu firmware ELOOP boot failure:
# https://forums.developer.nvidia.com/t/384170/7
# Do not silently install a release-specific workaround on unrelated JetPack versions.
if grep -q '^# R39 (release), REVISION: 2\.' /etc/nv_tegra_release 2>/dev/null; then
  FIRMWARE_UNIT="$ROOT/scripts/systemd/firmware-path.service"
  FIRMWARE_DEST=/etc/systemd/system/firmware-path.service
  if [ ! -d /usr/lib/firmware ]; then
    printf '[FAIL] Firmware directory /usr/lib/firmware is missing.\n' >&2
    exit 1
  fi
  # Preserve an existing unit (including the one installed manually). Require
  # review if its actual directives differ; descriptions/comments may differ.
  if [ -e "$FIRMWARE_DEST" ] || [ -L "$FIRMWARE_DEST" ]; then
    if ! diff -u <(sed '/^[[:space:]]*#/d; /^[[:space:]]*Description=/d; /^[[:space:]]*$/d' "$FIRMWARE_UNIT") \
        <(sed '/^[[:space:]]*#/d; /^[[:space:]]*Description=/d; /^[[:space:]]*$/d' "$FIRMWARE_DEST"); then
      printf '[FAIL] Existing firmware-path.service differs; review it before continuing.\n' >&2
      exit 1
    fi
  else
    "${AS_ROOT[@]}" install -m 0644 "$FIRMWARE_UNIT" "$FIRMWARE_DEST"
  fi
  "${AS_ROOT[@]}" systemctl daemon-reload
  "${AS_ROOT[@]}" systemctl enable firmware-path.service
  printf '[PASS] Firmware workaround enabled for the next boot.\n'
else
  printf '[INFO] Firmware workaround applies to L4T R39.2.x; skipped on this release.\n'
fi

step 5 'Disable automatic sleep and hibernation'
"${AS_ROOT[@]}" systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target suspend-then-hibernate.target
# Use the desktop user's dconf database, even from sudo or an SSH terminal.
run_as_desktop dbus-run-session -- bash -eu -c '
  for supply in ac battery; do
    gsettings set org.gnome.settings-daemon.plugins.power "sleep-inactive-$supply-type" nothing
    gsettings set org.gnome.settings-daemon.plugins.power "sleep-inactive-$supply-timeout" 0
  done
  gsettings set org.gnome.desktop.session idle-delay 0
  gsettings set org.gnome.desktop.screensaver lock-enabled false
'
printf '[PASS] Automatic suspend, idle screen blanking, and screen locking disabled; system sleep/hibernate targets masked.\n'

step 6 'Create or reuse the Python 3.12 UV environment'
if [ -e "$ROOT/.venv" ] || [ -L "$ROOT/.venv" ]; then
  if [ ! -x "$ROOT/.venv/bin/python" ] \
      || ! run_as_desktop "$ROOT/.venv/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12)' \
      || ! grep -Eiq '^include-system-site-packages[[:space:]]*=[[:space:]]*true[[:space:]]*$' "$ROOT/.venv/pyvenv.cfg"; then
    printf '[FAIL] Existing .venv must use Python 3.12 with system-site-packages.\n' >&2
    printf '       Move it aside and rerun setup to create a compatible environment.\n' >&2
    exit 1
  fi
  printf '[INFO] Reusing existing Python environment.\n'
else
  run_as_desktop uv venv --python /usr/bin/python3.12 --system-site-packages "$ROOT/.venv"
fi
run_as_desktop env UV_PROJECT_ENVIRONMENT="$ROOT/.venv" uv sync --locked --project "$ROOT"
printf '[PASS] Python environment synchronized from uv.lock.\n'

step 7 'Build and verify native camera and pose components'
printf '[INFO] Building as desktop user: %s\n' "$TARGET_USER"
BUILD_ENV=()
for name in CUDA_VER SQUEAKVIEW_BUILD_JOBS SQUEAKVIEW_DEEPSTREAM_SDK GST_PLUGIN_PATH; do
  if [ -n "${!name:-}" ]; then
    BUILD_ENV+=("$name=${!name}")
  fi
done
run_as_desktop env "${BUILD_ENV[@]}" bash "$ROOT/scripts/build_native.sh"

printf '[PASS] Native camera plugin and pose parser are built and verified.\n'
printf '\n%s\n' '============================================================'
printf '[SETUP COMPLETE] All seven setup steps succeeded.\n'
printf '[INFO] Project storage: %s\n' "$PROJECTS_PARENT"
printf '[SETUP COMPLETE] Reboot this Jetson before starting SqueakView.\n'
