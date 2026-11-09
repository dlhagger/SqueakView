"""Exercise provisioning flow with fake system commands and an isolated checkout."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


class JetsonSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.app = self.root / 'checkout'
        scripts = self.app / 'scripts'
        (scripts / 'systemd').mkdir(parents=True)
        shutil.copytree(REPO / 'scripts/udev', scripts / 'udev')
        common = self.app / 'squeakview/common'
        common.mkdir(parents=True)
        shutil.copy(REPO / 'squeakview/common/parser_selection.py', common)
        self.sdk = self.root / 'sdk'
        (self.sdk / 'includes').mkdir(parents=True)
        (self.sdk / 'includes/nvdsinfer_custom_impl.h').touch()
        (self.sdk / 'version').write_text('Version: 9.2.0\n')
        shutil.copy(REPO / 'scripts/systemd/firmware-path.service', scripts / 'systemd')
        shutil.copy(REPO / 'scripts/systemd/squeakview-usb.conf', scripts / 'systemd')
        shutil.copy(REPO / 'scripts/jetson_usb_recovery.py', scripts)
        self.release = self.root / 'nv_tegra_release'
        self.release.write_text('# R39 (release), REVISION: 2.1, BOARD: generic\n')
        self.unit = self.root / 'firmware-path.service'
        (self.root / 'firmware').mkdir()
        # Redirect only system file paths; privileged commands are mocked below.
        setup = (REPO / 'scripts/setup_jetson.sh').read_text()
        setup = setup.replace('/etc/nv_tegra_release', str(self.release))
        setup = setup.replace('/etc/systemd/system/firmware-path.service', str(self.unit))
        setup = setup.replace('/usr/lib/firmware', str(self.root / 'firmware'))
        for directory in ('/usr/local/libexec', '/etc/sudoers.d', '/etc/tmpfiles.d', '/etc/udev/rules.d'):
            setup = setup.replace(directory, str(self.root / directory.lstrip('/')))
        (scripts / 'setup_jetson.sh').write_text(setup)
        (scripts / 'build_native.sh').write_text('printf "native-build\\nnative-sdk=%s\\n" "$SQUEAKVIEW_DEEPSTREAM_SDK" >> "$SETUP_LOG"\n')
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        dispatcher = self.bin / 'dispatch'
        dispatcher.write_text('''#!/bin/bash
set -eu
name=${0##*/}
printf '%s %s\\n' "$name" "$*" >> "$SETUP_LOG"
case "$name" in
  id)
    case "${1:-}" in
      -u) echo "${MOCK_UID:-1000}";;
      -un) echo tester;;
      -nG) echo 'users dialout';;
    esac;;
  getent)
    if [ "$1" = passwd ]; then echo "tester:x:1000:1000::${MOCK_HOME}:/bin/bash"; fi;;
  sudo) exec "$@";;
  runuser) shift 3; exec "$@";;
  dbus-run-session) shift; exec "$@";;
  uv)
    if [ "$1" = venv ]; then
      dest=${!#}
      mkdir -p "$dest/bin"
      printf '#!/bin/sh\\nexit 0\\n' > "$dest/bin/python"
      chmod +x "$dest/bin/python"
      echo 'include-system-site-packages = true' > "$dest/pyvenv.cfg"
    fi;;
  apt-get|systemctl|gsettings|visudo|systemd-tmpfiles|udevadm) :;;
  *) exit 99;;
esac
''')
        dispatcher.chmod(0o755)
        for name in ('id', 'getent', 'sudo', 'runuser', 'dbus-run-session',
                     'uv', 'apt-get', 'systemctl', 'gsettings', 'visudo', 'systemd-tmpfiles', 'udevadm'):
            (self.bin / name).symlink_to(dispatcher)
        self.log = self.root / 'commands.log'
        self.env = dict(os.environ, PATH=f'{self.bin}:/usr/bin:/bin',
                        SETUP_LOG=str(self.log), MOCK_HOME=str(self.root),
                        SQUEAKVIEW_PROJECTS_DIR=str(self.root / 'projects'),
                        SQUEAKVIEW_DEEPSTREAM_SDK=str(self.sdk),
                        SQUEAKVIEW_DESKTOP_USER='tester')

    def run_setup(self):
        return subprocess.run(['bash', str(self.app / 'scripts/setup_jetson.sh')],
                              env=self.env, cwd=self.root, capture_output=True, text=True)

    def test_fresh_setup_and_rerun_preserve_environment_and_unit(self):
        for _ in range(2):
            result = self.run_setup()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        commands = self.log.read_text()
        self.assertEqual(commands.count('uv venv '), 1)
        self.assertEqual(commands.count('uv sync --locked --project '), 2)
        self.assertEqual(commands.count('native-build\n'), 2)
        self.assertIn('libgstrtspserver-1.0-0', commands)
        self.assertIn('libmosquitto1 libmosquitto-dev', commands)
        self.assertIn('systemctl enable firmware-path.service', commands)
        self.assertIn('systemctl mask sleep.target suspend.target hibernate.target', commands)
        self.assertIn('sleep-inactive-ac-type nothing', commands)
        self.assertIn('sleep-inactive-battery-timeout 0', commands)
        self.assertIn('gsettings set org.gnome.desktop.session idle-delay 0', commands)
        self.assertIn('gsettings set org.gnome.desktop.screensaver lock-enabled false', commands)
        self.assertTrue(self.unit.is_file())
        helper = self.root / 'usr/local/libexec/squeakview-usb-recovery'
        self.assertEqual(helper.read_bytes(), (REPO / 'scripts/jetson_usb_recovery.py').read_bytes())
        policy = self.root / 'etc/sudoers.d/squeakview-usb-recovery'
        self.assertEqual(policy.read_text(), f'#1000 ALL=(root) NOPASSWD: NOSETENV: {helper} ""\n')
        self.assertEqual(policy.stat().st_mode & 0o777, 0o440)
        self.assertIn('systemd-tmpfiles --create', commands)
        self.assertIn('visudo -cf', commands)
        rules = self.root / 'etc/udev/rules.d/99-platformio-udev.rules'
        self.assertEqual(rules.read_bytes(), (REPO / 'scripts/udev/99-platformio-udev.rules').read_bytes())
        self.assertEqual(rules.stat().st_mode & 0o777, 0o644)
        self.assertEqual(commands.splitlines().count('udevadm control --reload-rules'), 2)
        self.assertNotIn('udevadm trigger', commands)
        self.assertNotIn('systemctl reboot', commands)

    def test_conflicting_existing_service_is_preserved(self):
        self.unit.write_text('[Service]\nExecStart=/bin/false\n')
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Existing firmware-path.service differs', result.stderr)
        self.assertEqual(self.unit.read_text(), '[Service]\nExecStart=/bin/false\n')
        self.assertNotIn('native-build', self.log.read_text())

    def test_both_sdk_versions_are_detected_and_forwarded_to_native_build(self):
        for version, variant, layout in (('9.1.0', 'legacy', 'sources/includes'),
                                         ('9.2.0', 'yolo26', 'includes')):
            sdk = self.root / version
            (sdk / layout).mkdir(parents=True)
            (sdk / layout / 'nvdsinfer_custom_impl.h').touch()
            (sdk / 'version').write_text(f'Version: {version}\n')
            self.env['SQUEAKVIEW_DEEPSTREAM_SDK'] = str(sdk)
            result = self.run_setup()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(f'Detected DeepStream {version}: {sdk} (parser: {variant})', result.stdout)
            self.assertIn(f'native-sdk={sdk}', self.log.read_text())

    def test_unsupported_or_incomplete_sdk_stops_before_system_changes(self):
        for version in ('9.3.0', '9.2.0'):
            (self.sdk / 'version').write_text(f'Version: {version}\n')
            (self.sdk / 'includes/nvdsinfer_custom_impl.h').unlink(missing_ok=True)
            result = self.run_setup()
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('apt-get', self.log.read_text())
            self.assertNotIn('native-build', self.log.read_text())

    def test_incompatible_environment_is_preserved(self):
        venv = self.app / '.venv'
        venv.mkdir()
        (venv / 'keep-me').write_text('user data')
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Existing .venv must use Python 3.12', result.stderr)
        self.assertEqual((venv / 'keep-me').read_text(), 'user data')
        self.assertNotIn('uv sync', self.log.read_text())

    def test_other_l4t_release_skips_firmware_workaround(self):
        self.release.write_text('# R40 (release), REVISION: 1.0\n')
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.unit.exists())
        self.assertNotIn('systemctl enable firmware-path', self.log.read_text())

    def test_sudo_invocation_drops_privileges_for_python_and_native_builds(self):
        self.env['MOCK_UID'] = '0'
        (self.root / 'projects').mkdir()
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        commands = self.log.read_text()
        self.assertIn('runuser -u tester -- env HOME=', commands)
        self.assertIn('uv sync --locked', commands)
        self.assertIn('native-build', commands)


if __name__ == '__main__':
    unittest.main()
