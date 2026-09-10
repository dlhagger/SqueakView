from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from squeakview import model_package
from squeakview.apps.operator.gui import model_catalog


class ModelCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "models"
        self.root.mkdir()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def config(self, package: str, name: str = "model.txt") -> Path:
        path = self.root / package / "configs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("config\n")
        return path

    @staticmethod
    def validator(path: Path):
        if "broken" in path.parts:
            raise model_package.ModelPackageError("engine hash mismatch")
        schema = 2 if "legacy" in path.parts else 3
        return SimpleNamespace(
            name=path.parent.parent.name,
            model_manifest_schema=schema,
            engine_build_identity=({"gpu": "orin"} if schema == 3 else None),
        )

    def test_enumerates_valid_and_flags_invalid_packages(self) -> None:
        valid = self.config("mousehouse")
        legacy = self.config("legacy")
        broken = self.config("broken")

        choices = model_catalog.enumerate_model_configs(
            self.root, validator=self.validator
        )

        self.assertEqual([item.config for item in choices], [valid, broken, legacy])
        self.assertTrue(choices[0].eligible)
        self.assertFalse(choices[1].eligible)
        self.assertIn("hash mismatch", choices[1].detail)
        self.assertFalse(choices[2].eligible)
        self.assertIn("migration-only schema 2", choices[2].detail)

    def test_hidden_build_packages_and_nested_configs_are_excluded(self) -> None:
        self.config(".mousehouse.build-123/mousehouse")
        nested = self.root / "outer" / "nested" / "configs" / "nested.txt"
        nested.parent.mkdir(parents=True)
        nested.write_text("config\n")

        choices = model_catalog.enumerate_model_configs(
            self.root, validator=self.validator
        )

        self.assertEqual(choices, ())

    def test_production_validation_rejects_schema_two(self) -> None:
        config = self.config("legacy")

        with self.assertRaisesRegex(
            model_package.ModelPackageError, "migration-only schema 2"
        ):
            model_catalog.validate_production_model(
                config, validator=self.validator
            )


if __name__ == "__main__":
    unittest.main()
