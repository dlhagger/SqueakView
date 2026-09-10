"""Strict bounded reads for operator-supplied configuration documents."""

from __future__ import annotations

import os
import json
import hashlib
import math
import stat
from pathlib import Path
from typing import Any


def read_stable_regular_file_with_identity(
    path: Path, *, max_bytes: int, label: str
) -> tuple[bytes, dict[str, object]]:
    """Read one stable regular file and identify the exact bytes returned."""

    unresolved = Path(path)
    try:
        source = unresolved.resolve()
        with source.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"{label} must be a regular file")
            if before.st_size > max_bytes:
                raise ValueError(f"{label} exceeds {max_bytes} byte limit")
            data = handle.read(before.st_size)
            grew = bool(handle.read(1))
            after = os.fstat(handle.fileno())
        current = source.stat()
    except OSError as exc:
        raise ValueError(f"{label} could not be read: {exc}") from exc
    if (
        len(data) != before.st_size
        or grew
        or after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_ctime_ns != before.st_ctime_ns
        or current.st_dev != before.st_dev
        or current.st_ino != before.st_ino
        or current.st_size != before.st_size
        or current.st_mtime_ns != before.st_mtime_ns
        or current.st_ctime_ns != before.st_ctime_ns
    ):
        raise ValueError(f"{label} changed while being read")
    return data, {
        "path": str(source),
        "available": True,
        "size_bytes": len(data),
        "mtime_ns": before.st_mtime_ns,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def read_stable_regular_file(path: Path, *, max_bytes: int, label: str) -> bytes:
    """Read one regular file once, rejecting size or identity changes."""

    data, _identity = read_stable_regular_file_with_identity(
        path, max_bytes=max_bytes, label=label
    )
    return data


def decode_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    """Decode strict UTF-8 JSON and reject duplicate keys at every depth."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        def invalid_constant(value: str) -> None:
            raise ValueError(f"{label} contains non-standard number {value}")

        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_object,
            parse_constant=invalid_constant,
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be strict UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON: {exc}") from exc
    except RecursionError as exc:
        raise ValueError(f"{label} exceeds the supported JSON nesting depth") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")

    pending: list[Any] = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{label} contains a non-finite number")
    return value


def read_json_object(path: Path, *, max_bytes: int, label: str) -> dict[str, Any]:
    """Read one stable bounded regular file as a strict JSON object."""

    return decode_json_object(
        read_stable_regular_file(path, max_bytes=max_bytes, label=label),
        label=label,
    )


def read_json_object_with_identity(
    path: Path, *, max_bytes: int, label: str
) -> tuple[dict[str, Any], dict[str, object]]:
    """Decode a JSON object and identify the exact stable bytes decoded."""

    raw, identity = read_stable_regular_file_with_identity(
        path, max_bytes=max_bytes, label=label
    )
    return decode_json_object(raw, label=label), identity


def decode_yaml_mapping(raw: bytes, *, label: str) -> dict[str, Any]:
    """Decode strict UTF-8 safe YAML and reject duplicate mapping keys."""

    import yaml

    class UniqueKeySafeLoader(yaml.SafeLoader):
        pass

    def unique_mapping(loader, node, deep=False):
        result: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeySafeLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping
    )
    try:
        value = yaml.load(
            raw.decode("utf-8", errors="strict"), Loader=UniqueKeySafeLoader
        )
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be strict UTF-8") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"{label} must be valid YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a mapping")
    return value


def read_yaml_mapping(path: Path, *, max_bytes: int, label: str) -> dict[str, Any]:
    """Read one stable bounded regular file as a safe YAML mapping."""

    return decode_yaml_mapping(
        read_stable_regular_file(path, max_bytes=max_bytes, label=label),
        label=label,
    )


__all__ = [
    "decode_json_object",
    "decode_yaml_mapping",
    "read_json_object",
    "read_json_object_with_identity",
    "read_stable_regular_file",
    "read_stable_regular_file_with_identity",
    "read_yaml_mapping",
]
