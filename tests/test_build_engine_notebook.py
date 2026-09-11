from __future__ import annotations

import json
import unittest
from pathlib import Path


class BuildEngineNotebookTests(unittest.TestCase):
    def test_notebook_writes_schema_three_manifest_and_schema_two_pose_sidecar(self) -> None:
        path = Path(__file__).resolve().parents[1] / "build_engine" / "build_engine.ipynb"
        notebook = json.loads(path.read_text())
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )

        pose_start = source.index("pose_schema = {")
        pose_end = source.index("pose_path =", pose_start)
        manifest_start = source.index("manifest_path.write_text", pose_end)
        manifest_end = source.index("required =", manifest_start)
        self.assertIn('"schema_version": 2', source[pose_start:pose_end])
        self.assertIn('"schema_version": 3', source[manifest_start:manifest_end])
        self.assertIn('"build_environment": build_environment', source)
        self.assertIn('assert pose_schema["schema_version"] == 2', source)
        self.assertIn(
            'yaml.safe_load(manifest_path.read_text())["schema_version"] == 3',
            source,
        )
        self.assertIn("create_staging_package(MODELS_DIR, MODEL_NAME)", source)
        self.assertIn('os.environ.get("SQUEAKVIEW_BUILD_MODEL_NAME"', source)
        self.assertIn('SQUEAKVIEW_BUILD_OVERWRITE must be 0 or 1', source)
        self.assertLess(
            source.index('sys.path.insert(0, str(WORKSPACE))'),
            source.index('from squeakview.model_build import'),
        )
        self.assertIn("atexit.register(cleanup_staging_package, PACKAGE_DIR)", source)
        self.assertIn("validate_model_package(config_path)", source)
        self.assertIn("promote_model_package(", source)
        self.assertIn('"artifacts": artifact_identities', source)
        self.assertIn('"import_report": artifact_identity(report_path)', source)
        self.assertIn("engine_execution = run_trtexec_validation(engine_path)", source)
        self.assertIn('assert engine_execution["passed"]', source)
        self.assertIn('"engine_execution": True', source)
        self.assertIn('one2one_head = getattr(head, "one2one", None)', source)
        self.assertNotIn("assert bool(head.end2end)", source)
        self.assertIn("    nms=False,", source)
        self.assertNotIn("    end2end=True,", source)
        self.assertIn('engine_metadata.get("end2end") is True', source)
        self.assertIn('engine_metadata.get("args", {}).get("nms") is False', source)
        self.assertLess(
            source.index("report_path.write_text"),
            source.index("manifest_path.write_text"),
        )
        self.assertLess(
            source.index("validate_model_package(config_path)"),
            source.index("promote_model_package("),
        )


if __name__ == "__main__":
    unittest.main()
