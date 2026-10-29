"""A vendor metadata exception must never hide another dependency problem."""
import subprocess
import unittest
from unittest import mock

from squeakview.common import dependency_health as health

KNOWN = 'Checked 82 packages in 4ms\nFound 1 incompatibility\nThe package `nvidia-cusparselt-cu13` was built for a different platform\n'


class DependencyHealthTests(unittest.TestCase):
    def check(self, output=KNOWN, *, allow=False, verified=True, code=1):
        with mock.patch.object(health.shutil, 'which', return_value='/uv'), \
             mock.patch.object(health.subprocess, 'run', return_value=subprocess.CompletedProcess([], code, output, '')), \
             mock.patch.object(health, 'reviewed_cusparselt_metadata', return_value=verified):
            return health.check_dependencies(allow_reviewed_vendor_metadata=allow)

    def test_exception_requires_explicit_opt_in_and_verified_payload(self):
        self.assertFalse(self.check()[0])
        self.assertFalse(self.check(allow=True, verified=False)[0])
        ok, message = self.check(allow=True)
        self.assertTrue(ok)
        self.assertIn('[WARN]', message)

    def test_additional_diagnostic_or_different_error_is_not_waived(self):
        for output in ('another failure\n' + KNOWN, KNOWN + 'another failure', KNOWN.replace('1 incompatibility', '2 incompatibilities'), KNOWN.replace('cusparselt', 'cudnn')):
            self.assertFalse(self.check(output, allow=True)[0])
        self.assertFalse(self.check(allow=True, code=2)[0])

    def test_wrong_platform_version_pin_or_content_is_not_waived(self):
        from types import SimpleNamespace
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(health.platform, 'system', return_value='Linux'), \
             mock.patch.object(health.platform, 'machine', return_value='aarch64'), \
             mock.patch.object(health.metadata, 'distribution') as distribution, \
             mock.patch.object(health.metadata, 'requires', return_value=['nvidia-cusparselt-cu13==0.8.1']):
            distribution.return_value = SimpleNamespace(version='0.8.2', locate_file=lambda _: Path(directory))
            self.assertFalse(health.reviewed_cusparselt_metadata())
            distribution.return_value.version = '0.8.1'
            for relative in health._VENDOR_FILES:
                path = Path(directory) / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'changed payload')
            self.assertFalse(health.reviewed_cusparselt_metadata())
            with mock.patch.object(health.metadata, 'requires', return_value=['nvidia-cusparselt-cu13==0.8.2']):
                self.assertFalse(health.reviewed_cusparselt_metadata())
            with mock.patch.object(health.platform, 'machine', return_value='x86_64'):
                self.assertFalse(health.reviewed_cusparselt_metadata())

    def test_checker_timeout_is_an_actionable_failure(self):
        with mock.patch.object(health.shutil, 'which', return_value='/uv'), \
             mock.patch.object(health.subprocess, 'run', side_effect=subprocess.TimeoutExpired('uv', 30)):
            ok, message = health.check_dependencies()
        self.assertFalse(ok)
        self.assertIn('Dependency check failed', message)
