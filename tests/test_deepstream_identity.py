"""Qualification must validate recorded evidence for either supported SDK."""
import unittest
from copy import deepcopy

from squeakview.common.deepstream_identity import deepstream_package_errors
from squeakview.common.diagnostics.qualification import _platform_provenance_complete
from squeakview.common.diagnostics.evidence_identity import runtime_identity_errors


class DeepStreamIdentityTests(unittest.TestCase):
    def platform(self, version):
        return dict(device_model='Orin', machine='aarch64', kernel='tegra', python='3.12.3',
                    python_executable='/usr/bin/python3', jetson_linux_release='R39',
                    deepstream_build=f'Version: {version}.0\nDATE: example', nvpmodel='MAXN',
                    packages={**{name: 'installed' for name in ('nvidia-l4t-core', 'cuda-toolkit-13-2',
                              'libcudnn9-cuda-13', 'libnvinfer10', 'libgstreamer1.0-0', 'libspinnaker', 'ffmpeg')},
                              f'deepstream-{version}': f'{version}.0-1'})

    def test_each_sdk_requires_its_own_package_in_both_qualification_paths(self):
        for version, other in (('9.1', '9.2'), ('9.2', '9.1')):
            platform = self.platform(version)
            self.assertEqual(deepstream_package_errors(platform), [])
            self.assertTrue(_platform_provenance_complete(platform))
            errors = runtime_identity_errors({'platform': platform, 'native_plugins': {}})
            self.assertFalse(any('packages' in error or 'deepstream_build' in error for error in errors))
            platform['packages'].pop(f'deepstream-{version}')
            platform['packages'][f'deepstream-{other}'] = f'{other}.0-1'
            self.assertFalse(_platform_provenance_complete(platform))
            self.assertIn(f'runtime_identity.packages missing: deepstream-{version}',
                          runtime_identity_errors({'platform': platform, 'native_plugins': {}}))

    def test_mismatch_unknown_sdk_and_missing_identity_fail_closed(self):
        platform = self.platform('9.2')
        platform['packages']['deepstream-9.2'] = '9.1.0-1'
        self.assertTrue(deepstream_package_errors(platform))
        platform['packages']['deepstream-9.2'] = '9.2.3-1'
        self.assertTrue(deepstream_package_errors(platform))
        for build in ('Version: 9.3.0', '', None):
            platform['deepstream_build'] = build
            self.assertTrue(deepstream_package_errors(platform))

    def test_archived_91_format_and_coinstalled_packages_are_supported_without_mutation(self):
        platform = self.platform('9.1')
        platform['deepstream_build'] = 'DeepStreamSDK 9.1'
        platform['packages']['deepstream-9.2'] = '9.2.0-1'
        original = deepcopy(platform)
        self.assertEqual(deepstream_package_errors(platform), [])
        self.assertEqual(platform, original)
