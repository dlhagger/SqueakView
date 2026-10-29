#!/usr/bin/env python3
"""Run dependency health, nonempty Python discovery, and both camera-free native suites."""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import threading
import traceback
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_python_suite(suite, *, stream=sys.stderr):
    if suite.countTestCases() == 0:
        print('[FAIL] Test discovery collected zero tests.', file=stream)
        return False
    failures = []
    previous = threading.excepthook
    def thread_failure(args):
        failures.append(args)
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=stream)
    threading.excepthook = thread_failure
    try:
        result = unittest.TextTestRunner(stream=stream, verbosity=1).run(suite)
    finally:
        threading.excepthook = previous
    if failures:
        print(f'[FAIL] {len(failures)} uncaught worker exception(s).', file=stream)
    return result.wasSuccessful() and not failures


def native_suites(build_dir):
    for source, name, target in (
        ('native/flir_gst_source', 'flir', 'flir_source_eos_test'),
        ('native/nvdsinfer_custom_impl_yolo/tests', 'pose', 'test_pose_parser'),
    ):
        build = Path(build_dir) / name
        for command in (
            ['cmake', '-S', str(ROOT / source), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_TESTING=ON'],
            ['cmake', '--build', str(build), '--parallel', '2', '--target', target],
            ['ctest', '--test-dir', str(build), '--output-on-failure', '--no-tests=error', '--timeout', '30'],
        ):
            subprocess.run(command, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python-only', action='store_true', help='Explicitly omit native compilation/tests; not a complete SDK software check.')
    parser.add_argument('--pattern', default='test_*.py')
    parser.add_argument('--build-dir', type=Path, help='Keep native build/log files here; defaults to a temporary directory.')
    args = parser.parse_args(argv)
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from squeakview.common.dependency_health import check_dependencies
    healthy, message = check_dependencies(allow_reviewed_vendor_metadata=True)
    print(message, flush=True)
    if not healthy:
        return 1
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern=args.pattern, top_level_dir=str(ROOT))
    if not run_python_suite(suite):
        return 1
    if args.python_only:
        print('[NOT RUN] Native parser/EOS tests (--python-only).')
    else:
        try:
            if args.build_dir:
                native_suites(args.build_dir)
            else:
                with tempfile.TemporaryDirectory(prefix='squeakview-tests-') as directory:
                    native_suites(directory)
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f'[FAIL] Native software checks: {exc}', file=sys.stderr)
            return 1
    print('[PASS] Requested software checks passed; hardware/GPU qualification was not run.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
