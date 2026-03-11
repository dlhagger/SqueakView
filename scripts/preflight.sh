#!/usr/bin/env bash

set -u

FAIL=0
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif [[ -x ".venv/bin/python3" ]]; then
  PYTHON_BIN=".venv/bin/python3"
else
  PYTHON_BIN="python3"
fi

pass() {
  echo "[PASS] $1"
}

fail() {
  echo "[FAIL] $1"
  FAIL=1
}

check_plugin() {
  local name="$1"
  if gst-inspect-1.0 "$name" >/dev/null 2>&1; then
    pass "GStreamer element '$name' is available."
  else
    fail "GStreamer element '$name' is missing."
  fi
}

echo "=== SqueakView Preflight ==="
echo "Python: $PYTHON_BIN"
echo "Capture backend: ${CAPTURE_BACKEND:-flir}"
echo

if command -v gst-inspect-1.0 >/dev/null 2>&1; then
  pass "gst-inspect-1.0 found."
else
  fail "gst-inspect-1.0 not found in PATH."
fi

check_plugin "shmsrc"
check_plugin "shmsink"
check_plugin "nvinfer"
check_plugin "nvstreammux"

if [[ "${CAPTURE_BACKEND:-flir}" == "zed" ]]; then
  check_plugin "zedsrc"
  check_plugin "zeddemux"
fi

PYSPIN_SO="$("$PYTHON_BIN" -c "import importlib.util; s=importlib.util.find_spec('_PySpin'); print(s.origin if s and s.origin else '')" 2>/dev/null || true)"
if [[ -z "$PYSPIN_SO" ]]; then
  fail "_PySpin extension was not found in the active Python environment."
else
  pass "_PySpin extension found: $PYSPIN_SO"
  if ! command -v ldd >/dev/null 2>&1; then
    fail "ldd not found; cannot validate native library dependencies for _PySpin."
  else
    LDD_OUT="$(ldd "$PYSPIN_SO" 2>/dev/null || true)"
    if echo "$LDD_OUT" | grep -q "not found"; then
      fail "_PySpin has unresolved shared libraries."
      echo "$LDD_OUT" | grep "not found" | sed 's/^/  - /'
    else
      pass "_PySpin shared-library dependencies resolved."
    fi
  fi
fi

if [[ "$FAIL" -ne 0 ]]; then
  echo
  echo "=== Suggested fixes ==="
  echo "1. Install GStreamer shared-memory plugins: gstreamer1.0-plugins-bad"
  echo "2. Ensure Spinnaker SDK libs are installed and discoverable by the linker."
  echo "3. If using ZED backend, install zed-gstreamer plugins (zedsrc/zeddemux)."
  echo "4. If plugins are installed in a non-standard location, export GST_PLUGIN_PATH."
  echo "5. Re-run this script after fixes."
  exit 1
fi

echo
echo "All preflight checks passed."
exit 0
