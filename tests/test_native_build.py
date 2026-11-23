"""Exercise SDK dispatch without privileged setup, GPU execution, or vendor builds."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeBuildTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.app = self.root / 'app'
        for relative in ('scripts', 'squeakview/common', 'native/flir_gst_source',
                         'native/yolo26_pose_parser'):
            (self.app / relative).mkdir(parents=True)
        shutil.copy(ROOT / 'squeakview/common/parser_selection.py', self.app / 'squeakview/common')
        vendor = self.root / 'vendor'
        for relative in ('spinnaker/include', 'spinnaker/lib', 'cuda/bin', 'cuda-13.2/bin'):
            (vendor / relative).mkdir(parents=True)
        for relative in ('cuda/bin/nvcc', 'cuda-13.2/bin/nvcc'):
            path = vendor / relative
            path.write_text('#!/bin/sh\necho "Cuda compilation tools, release 13.2, V13.2"\n')
            path.chmod(0o755)
        script = (ROOT / 'scripts/build_native.sh').read_text()
        script = script.replace('/opt/spinnaker', str(vendor / 'spinnaker'))
        script = script.replace('/usr/local/cuda', str(vendor / 'cuda'))
        (self.app / 'scripts/build_native.sh').write_text(script)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        dispatcher = self.bin / 'dispatch'
        dispatcher.write_text('''#!/usr/bin/python3
import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['NATIVE_BUILD_LOG'], 'a') as log:
    log.write(json.dumps([name, *args]) + '\\n')
if name == 'cmake' and '-B' in args:
    build = pathlib.Path(args[args.index('-B') + 1])
    build.mkdir(parents=True, exist_ok=True)
    filename = 'gstflirspinsrc.so' if build.parent.name == 'flir_gst_source' else 'libnvdsinfer_squeakview_pose.so'
    (build / filename).write_text('test output')
elif name == 'nm':
    print('00000000 T NvDsInferParseYolo26Pose')
elif name == 'gst-inspect-1.0':
    print('capture-health-path frame-manifest-path camera-telemetry-path error-log-path camera-runtime-path native-timing-validation graceful-eos live-mailbox-path pipeline-null')
''')
        dispatcher.chmod(0o755)
        for command in ('cmake', 'ctest', 'make', 'ldd', 'nm', 'gst-inspect-1.0'):
            (self.bin / command).symlink_to(dispatcher)
        self.log = self.root / 'commands.jsonl'
        self.env = dict(os.environ, PATH=f'{self.bin}:{os.environ["PATH"]}',
                        NATIVE_BUILD_LOG=str(self.log), SQUEAKVIEW_BUILD_JOBS='2')

    def sdk(self, version, layout):
        sdk = self.root / version
        (sdk / layout).mkdir(parents=True)
        (sdk / layout / 'nvdsinfer_custom_impl.h').touch()
        (sdk / 'version').write_text(f'Version: {version}\n')
        return sdk

    def run_build(self, sdk):
        return subprocess.run(['bash', str(self.app / 'scripts/build_native.sh')],
                              env=dict(self.env, SQUEAKVIEW_DEEPSTREAM_SDK=str(sdk)),
                              text=True, capture_output=True)

    def test_modern_build_does_not_need_legacy_sources_or_comparison_baseline(self):
        sdk = self.sdk('9.2.0', 'includes')
        result = self.run_build(sdk)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        commands = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertIn('Detected DeepStream 9.2.0', result.stdout)
        configure = next(command for command in commands
                         if command[0] == 'cmake' and '-S' in command
                         and command[command.index('-S') + 1].endswith('yolo26_pose_parser'))
        self.assertIn('-DBUILD_TESTING=ON', configure)
        self.assertIn('-DBASELINE_PARSER=', configure)
        self.assertTrue(any(command[0] == 'ctest' for command in commands))
        self.assertFalse(any(command[0] == 'make' for command in commands))

    def test_unsupported_sdk_fails_before_any_build(self):
        for version in ('9.1.0', '9.3.0'):
            with self.subTest(version=version):
                result = self.run_build(self.sdk(version, 'includes'))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('requires DeepStream 9.2', result.stderr)
                self.assertFalse(self.log.exists())


@unittest.skipUnless(shutil.which('cmake'), 'requires CMake')
class NativeBuildCacheTests(unittest.TestCase):
    def test_cmake_headers_follow_sdk_after_cached_override_or_layout_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / 'source'
            project.mkdir()
            (project / 'CMakeLists.txt').write_text(
                f'cmake_minimum_required(VERSION 3.16)\nproject(layout NONE)\n'
                f'include("{ROOT / "native/cmake/DeepStream.cmake"}")\n'
                'file(WRITE "${CMAKE_BINARY_DIR}/headers.txt" "${DEEPSTREAM_INCLUDES}")\n')
            for version, layout in (('sdk-a', 'includes'), ('sdk-b', 'includes')):
                sdk = root / version
                (sdk / layout).mkdir(parents=True)
                (sdk / 'version').write_text('Version: 9.2.0\n')
                (sdk / layout / 'nvdsinfer_custom_impl.h').touch()
                result = subprocess.run(['cmake', '-S', str(project), '-B', str(root / 'build'),
                                         f'-DDEEPSTREAM_SDK={sdk}', '-DDEEPSTREAM_INCLUDES=/obsolete/headers'],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual((root / 'build/headers.txt').read_text(), str(sdk / layout))

    def test_direct_cmake_rejects_91_even_with_modern_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            sdk = Path(directory) / 'sdk'
            (sdk / 'includes').mkdir(parents=True)
            (sdk / 'includes/nvdsinfer_custom_impl.h').touch()
            (sdk / 'version').write_text('Version: 9.1.0\n')
            result = subprocess.run(['cmake', '-S', str(ROOT / 'native/yolo26_pose_parser'),
                                     '-B', str(Path(directory) / 'build'), f'-DDEEPSTREAM_SDK={sdk}'],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('requires DeepStream 9.2', result.stderr)
