"""UV/console entry point for the existing supervised desktop launcher."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the SqueakView desktop application.")
    parser.add_argument("project", nargs="?", help="Project directory; opens the project chooser if omitted.")
    args = parser.parse_args()
    script = Path(__file__).resolve().parents[1] / "squeakview.sh"
    if not script.is_file():
        parser.error("Launch from a SqueakView application checkout containing squeakview.sh.")
    command = ["bash", str(script)]
    if args.project is not None:
        command.append(args.project)
    # Pin the supervisor and all its workers to the interpreter selected by UV.
    os.execvpe("bash", command, dict(os.environ, PYTHON_BIN=sys.executable))
    return 0
