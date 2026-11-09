"""Select the validated parser build for an installed DeepStream SDK.

No GPU/vendor Python imports: also used by the native build shell script.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re


def sdk_version(sdk_root: Path) -> str:
    version_file = Path(sdk_root) / 'version'
    try:
        text = version_file.read_text(encoding='utf-8')
    except OSError as exc:
        raise ValueError(f'Cannot read DeepStream SDK version: {version_file}') from exc
    match = re.search(r'^Version:\s*(\d+)\.(\d+)\.(\d+)\b', text, re.MULTILINE)
    if match is None:
        raise ValueError(f'Unrecognized DeepStream version in {version_file}')
    version = tuple(map(int, match.groups()[:2]))
    if version not in {(9, 1), (9, 2)}:
        raise ValueError(f'Parser selection requires DeepStream 9.1 or 9.2; found {match.group(0)}')
    return '.'.join(match.groups())


def sdk_headers(sdk_root: Path) -> Path:
    sdk_version(sdk_root)
    for relative in ('includes', 'sources/includes'):
        candidate = Path(sdk_root) / relative
        if (candidate / 'nvdsinfer_custom_impl.h').is_file():
            return candidate
    raise ValueError(f'DeepStream headers missing under {sdk_root} (checked includes/ and sources/includes/)')


def parser_variant(sdk_root: Path) -> str:
    version = tuple(map(int, sdk_version(sdk_root).split('.')[:2]))
    if version == (9, 2):
        return 'yolo26'
    if version == (9, 1):
        return 'legacy'


def parser_library_for_sdk(native_root: Path, sdk_root: Path) -> Path:
    if parser_variant(sdk_root) == 'yolo26':
        return Path(native_root) / 'yolo26_pose_parser/build/libnvdsinfer_squeakview_pose.so'
    return Path(native_root) / 'nvdsinfer_custom_impl_yolo/libnvdsinfer_custom_impl_Yolo.so'


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--sdk', type=Path, required=True)
    cli.add_argument('--native', type=Path)
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument('--variant', action='store_true')
    mode.add_argument('--version', action='store_true')
    mode.add_argument('--includes', action='store_true')
    args = cli.parse_args()
    try:
        if args.version:
            print(sdk_version(args.sdk))
        elif args.includes:
            print(sdk_headers(args.sdk))
        elif args.variant:
            print(parser_variant(args.sdk))
        elif args.native is not None:
            print(parser_library_for_sdk(args.native, args.sdk))
        else:
            cli.error('--native is required when selecting a library')
    except ValueError as exc:
        cli.exit(2, f'[FAIL] {exc}\n')
