"""SDK-version selection must agree for provisioning and model packaging."""
from pathlib import Path
import tempfile
import unittest

from squeakview.common.parser_selection import parser_variant, parser_library_for_sdk, sdk_headers, sdk_version


class ParserSelectionTests(unittest.TestCase):
    def test_headers_and_version_follow_installed_sdk_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for version, layout in (('9.1.0', 'sources/includes'), ('9.2.0', 'includes')):
                sdk = root / version
                (sdk / layout).mkdir(parents=True)
                (sdk / 'version').write_text(f'Version: {version}\n')
                with self.assertRaisesRegex(ValueError, 'headers missing'):
                    sdk_headers(sdk)
                (sdk / layout / 'nvdsinfer_custom_impl.h').touch()
                self.assertEqual(sdk_headers(sdk), sdk / layout)
                self.assertEqual(sdk_version(sdk), version)

    def test_supported_versions_select_distinct_libraries(self):
        with tempfile.TemporaryDirectory() as directory:
            sdk = Path(directory)
            for version, variant, suffix in (
                ('9.1.0', 'legacy', 'nvdsinfer_custom_impl_yolo/libnvdsinfer_custom_impl_Yolo.so'),
                ('9.1.4', 'legacy', 'nvdsinfer_custom_impl_yolo/libnvdsinfer_custom_impl_Yolo.so'),
                ('9.2.0', 'yolo26', 'yolo26_pose_parser/build/libnvdsinfer_squeakview_pose.so'),
                ('9.2.3', 'yolo26', 'yolo26_pose_parser/build/libnvdsinfer_squeakview_pose.so'),
            ):
                with self.subTest(version=version):
                    (sdk / 'version').write_text(f'Version: {version}\nDATE: example\n')
                    self.assertEqual(parser_variant(sdk), variant)
                    self.assertEqual(parser_library_for_sdk(Path('/native'), sdk), Path('/native') / suffix)

    def test_unknown_or_missing_sdk_never_silently_selects_a_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            sdk = Path(directory)
            with self.assertRaises(ValueError):
                parser_variant(sdk)
            for text in ('Version: 9.3.0\n', 'Version: 10.0.0\n', 'invalid\n'):
                (sdk / 'version').write_text(text)
                with self.assertRaises(ValueError):
                    parser_variant(sdk)
