from __future__ import annotations

import json
import unittest
from pathlib import Path


class BuildEngineNotebookTests(unittest.TestCase):
    def test_notebook_is_a_non_executable_migration_notice(self) -> None:
        path = Path(__file__).resolve().parents[1] / "build_engine" / "build_engine.ipynb"
        notebook = json.loads(path.read_text())
        code = [
            cell
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
        )
        self.assertEqual(code, [])
        self.assertIn("Project Setup", source)
        self.assertIn("isolated worker process", source)
        self.assertNotIn("SQUEAKVIEW_PROJECT", source)
        self.assertNotIn("assert", source)


if __name__ == "__main__":
    unittest.main()
