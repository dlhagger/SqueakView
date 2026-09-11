#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point. The canonical operator command is now:
#   bash squeakview.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/squeakview.sh" "$@"
