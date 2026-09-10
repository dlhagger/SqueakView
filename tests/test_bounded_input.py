from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from squeakview.common.bounded_input import (
    read_json_object,
    read_json_object_with_identity,
    read_stable_regular_file,
    read_stable_regular_file_with_identity,
    read_yaml_mapping,
)


class BoundedInputTests(unittest.TestCase):
    def test_reads_a_regular_file_within_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.yaml"
            path.write_bytes(b"schema_version: '1.0'\n")

            self.assertEqual(
                read_stable_regular_file(path, max_bytes=1024, label="input"),
                path.read_bytes(),
            )

    def test_stable_read_identity_hashes_the_exact_bytes_returned(self) -> None:
        import hashlib

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metadata.json"
            encoded = b'{"value":1}\n'
            path.write_bytes(encoded)

            raw, identity = read_stable_regular_file_with_identity(
                path, max_bytes=1024, label="metadata"
            )
            payload, json_identity = read_json_object_with_identity(
                path, max_bytes=1024, label="metadata"
            )

            self.assertEqual(raw, encoded)
            self.assertEqual(payload, {"value": 1})
            self.assertEqual(identity["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(json_identity["sha256"], identity["sha256"])

    def test_rejects_non_regular_and_oversized_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "regular file"):
            read_stable_regular_file(Path("/dev/null"), max_bytes=1024, label="input")

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "large.yaml"
            path.write_bytes(b"x" * 5)
            with self.assertRaisesRegex(ValueError, "exceeds"):
                read_stable_regular_file(path, max_bytes=4, label="input")

    def test_json_object_rejects_duplicate_nested_keys_and_invalid_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metadata.json"
            path.write_text('{"outer":{"value":1,"value":2}}')
            with self.assertRaisesRegex(ValueError, "duplicate key 'value'"):
                read_json_object(path, max_bytes=1024, label="metadata")

            path.write_bytes(b'{"value":"\xff"}')
            with self.assertRaisesRegex(ValueError, "strict UTF-8"):
                read_json_object(path, max_bytes=1024, label="metadata")

    def test_json_object_rejects_non_object_and_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metadata.json"
            path.write_text("[]")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                read_json_object(path, max_bytes=1024, label="metadata")
            path.write_text('{"padding":"xxxxxxxx"}')
            with self.assertRaisesRegex(ValueError, "exceeds"):
                read_json_object(path, max_bytes=4, label="metadata")

    def test_json_object_rejects_nonstandard_and_overflowed_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metadata.json"
            for payload in ('{"value":NaN}', '{"value":1e999}'):
                path.write_text(payload)
                with self.subTest(payload=payload):
                    with self.assertRaisesRegex(ValueError, "number"):
                        read_json_object(path, max_bytes=1024, label="metadata")

    def test_yaml_mapping_rejects_duplicate_nested_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "task.yaml"
            path.write_text("outer:\n  value: 1\n  value: 2\n")
            with self.assertRaisesRegex(ValueError, "duplicate key 'value'"):
                read_yaml_mapping(path, max_bytes=1024, label="task config")


if __name__ == "__main__":
    unittest.main()
