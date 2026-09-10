from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from squeakview.common.diagnostics import campaign_archive
from squeakview.common.recording_evidence import capture_recording_evidence


class CampaignArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.controls = {}
        for name in ("matrix", "assignments", "report", "limits"):
            path = self.root / f"{name}.json"
            path.write_text("{}")
            self.controls[name] = path
        self.run = self.root / "source-run"
        (self.run / "diagnostics").mkdir(parents=True)
        (self.run / "run_manifest.json").write_text('{"run_id":"run-1"}')
        (self.run / "run_status.json").write_text('{"state":"finalized"}')
        (self.run / "qualification_summary.json").write_text('{"result":"passed"}')
        (self.run / "raw.mp4").write_bytes(b"video")
        (self.run / "capture_cam0.jsonl").write_text('{"source_sequence_index":0}\n')
        (self.run / "record_admission.csv").write_text("stream_id,source_sequence_index\n0,0\n")
        (self.run / "diagnostics" / "system.csv").write_text("value\n1\n")
        self.case_id = "case-1"
        self.inventory = self.root / "campaign-inventory.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build(
        self, *, report_updates=None, evidence_errors=(),
        tamper_summary_after_report=False,
    ):
        matrix_hash = campaign_archive._content_identity(self.controls["matrix"])["sha256"]
        limits_identity = campaign_archive._content_identity(self.controls["limits"])
        recording_evidence = capture_recording_evidence(self.run, 1)
        (self.run / "qualification_summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "result": "passed",
                    "run_directory": str(self.run.resolve()),
                    "limits": {
                        "path": str(self.controls["limits"].resolve()),
                        "validated": True,
                    },
                    "source_evidence": {
                        "limits": limits_identity,
                        "recording_artifacts": recording_evidence,
                    },
                }
            )
        )
        summary_identity = campaign_archive.stable_file_identity(
            self.run / "qualification_summary.json"
        )
        report = {
            "schema_version": "1.0", "matrix_id": "matrix-1",
            "matrix_sha256": matrix_hash, "provenance_policy": {},
            "result": "passed", "case_count": 1, "passed_count": 1,
            "failed_count": 0, "incomplete_count": 0,
            "cases": [{
                "case_id": self.case_id,
                "run_directory": str(self.run),
                "qualification_result": "passed",
                "qualification_summary_identity": summary_identity,
                "result": "passed",
                "factor_mismatches": [],
                "provenance_mismatches": [],
            }],
        }
        report.update(report_updates or {})
        self.controls["report"].write_text(json.dumps(report))
        if tamper_summary_after_report:
            with (self.run / "qualification_summary.json").open("a") as handle:
                handle.write(" ")
        with (
            mock.patch.object(campaign_archive, "load_matrix", return_value={"matrix_id": "matrix-1"}),
            mock.patch.object(campaign_archive, "expand_cases", return_value=[{"case_id": self.case_id}]),
            mock.patch.object(campaign_archive, "load_assignments", return_value={self.case_id: self.run}),
            mock.patch.object(campaign_archive, "_read_yaml", return_value=({"schema_version": "1.0"}, None)),
            mock.patch.object(
                campaign_archive,
                "source_evidence_errors",
                return_value=list(evidence_errors),
            ),
        ):
            return campaign_archive.build_campaign_inventory(
                matrix_path=self.controls["matrix"],
                assignments_path=self.controls["assignments"],
                report_path=self.controls["report"],
                limits_path=self.controls["limits"],
                output_path=self.inventory,
            )

    def _copy_layout(self, inventory):
        archive = self.root / "archive"
        for record in inventory["controls"].values():
            target = archive / record["archive_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(record["source_path"], target)
        for run in inventory["runs"]:
            for artifact in run["artifacts"]:
                target = archive / run["archive_path"] / artifact["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(Path(run["source_run_directory"]) / artifact["path"], target)
        return archive

    def test_builds_atomic_inventory_and_verifies_exact_copied_tree(self) -> None:
        inventory = self._build()
        self.assertTrue(self.inventory.is_file())
        self.assertEqual(inventory["run_count"], 1)
        archive = self._copy_layout(inventory)

        result = campaign_archive.verify_campaign_archive(
            archive_root=archive, inventory_path=self.inventory
        )

        self.assertEqual(result["result"], "verified")

    def test_verify_rejects_tamper_and_extra_file(self) -> None:
        inventory = self._build()
        archive = self._copy_layout(inventory)
        target = archive / inventory["runs"][0]["archive_path"] / "run_status.json"
        target.write_text('{"state":"failed"}')
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            campaign_archive.verify_campaign_archive(
                archive_root=archive, inventory_path=self.inventory
            )

        shutil.copyfile(self.run / "run_status.json", target)
        (archive / "unexpected.txt").write_text("unexpected")
        with self.assertRaisesRegex(ValueError, "file set differs"):
            campaign_archive.verify_campaign_archive(
                archive_root=archive, inventory_path=self.inventory
            )

    def test_build_rejects_active_run_without_publishing(self) -> None:
        (self.run / "run_status.json").write_text('{"state":"recording"}')
        with self.assertRaisesRegex(ValueError, "not successfully finalized"):
            self._build()
        self.assertFalse(self.inventory.exists())

    def test_build_rejects_failed_report_or_case(self) -> None:
        for field, value, message in (
            ("result", "failed", "only a passed matrix report"),
            ("passed_count", 0, "declared counts are inconsistent"),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, message):
                    self._build(report_updates={field: value})
        failed_case = {
            "case_id": self.case_id,
            "run_directory": str(self.run),
            "qualification_result": "failed",
            "qualification_summary_identity": campaign_archive.stable_file_identity(
                self.run / "qualification_summary.json"
            ),
            "result": "failed",
            "factor_mismatches": ["fps"],
            "provenance_mismatches": [],
        }
        with self.assertRaisesRegex(ValueError, "case did not pass"):
            self._build(report_updates={"cases": [failed_case]})

    def test_build_rejects_failed_run_and_stale_qualification(self) -> None:
        (self.run / "run_status.json").write_text('{"state":"failed"}')
        with self.assertRaisesRegex(ValueError, "not successfully finalized"):
            self._build()

        (self.run / "run_status.json").write_text('{"state":"finalized"}')
        with self.assertRaisesRegex(ValueError, "source evidence is stale"):
            self._build(evidence_errors=["run manifest is stale"])

    def test_build_rejects_limits_identity_mismatch(self) -> None:
        original_identity = campaign_archive._content_identity

        def altered_identity(path: Path, **kwargs):
            identity = original_identity(path, **kwargs)
            if Path(path) == self.controls["limits"] and kwargs.get("max_bytes") is not None:
                identity = {**identity, "sha256": "f" * 64}
            return identity

        with mock.patch.object(
            campaign_archive, "_content_identity", side_effect=altered_identity
        ), self.assertRaisesRegex(ValueError, "limits identity is stale"):
            self._build()

    def test_build_rejects_summary_changed_after_matrix_report(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "matrix report qualification summary is stale"
        ):
            self._build(tamper_summary_after_report=True)

    def test_build_never_replaces_an_existing_inventory(self) -> None:
        self.inventory.write_text("keep me")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self._build()
        self.assertEqual(self.inventory.read_text(), "keep me")

    def test_build_rejects_symlinked_run_artifact(self) -> None:
        (self.run / "diagnostics" / "link").symlink_to(self.run / "run_status.json")
        with self.assertRaisesRegex(ValueError, "symlink"):
            self._build()

    def test_verify_rejects_traversal_in_inventory(self) -> None:
        inventory = self._build()
        inventory["controls"]["matrix"]["archive_path"] = "../matrix.yaml"
        self.inventory.write_text(json.dumps(inventory))
        archive = self.root / "archive"
        archive.mkdir()
        with self.assertRaisesRegex(ValueError, "escapes"):
            campaign_archive.verify_campaign_archive(
                archive_root=archive, inventory_path=self.inventory
            )

    def test_verify_rejects_duplicate_control_archive_path(self) -> None:
        inventory = self._build()
        inventory["controls"]["limits"]["archive_path"] = inventory["controls"][
            "matrix"
        ]["archive_path"]
        self.inventory.write_text(json.dumps(inventory))
        archive = self.root / "archive"
        archive.mkdir()
        with self.assertRaisesRegex(ValueError, "duplicate path"):
            campaign_archive.verify_campaign_archive(
                archive_root=archive, inventory_path=self.inventory
            )

    def test_verify_rejects_non_integer_declared_counts(self) -> None:
        inventory = self._build()
        inventory["run_count"] = True
        self.inventory.write_text(json.dumps(inventory))
        archive = self.root / "archive"
        archive.mkdir()
        with self.assertRaisesRegex(ValueError, "run_count is invalid"):
            campaign_archive.verify_campaign_archive(
                archive_root=archive, inventory_path=self.inventory
            )

    def test_verify_rejects_invalid_campaign_semantics(self) -> None:
        for mutation, message in (
            (lambda value: value.update(matrix_id=""), "matrix_id is invalid"),
            (lambda value: value["runs"].clear(), "run set is invalid"),
            (
                lambda value: value["runs"][0].update(terminal_state="failed"),
                "run metadata is invalid",
            ),
        ):
            with self.subTest(message=message):
                inventory = self._build()
                mutation(inventory)
                self.inventory.write_text(json.dumps(inventory))
                archive = self.root / f"archive-{message.replace(' ', '-')}"
                archive.mkdir()
                with self.assertRaisesRegex(ValueError, message):
                    campaign_archive.verify_campaign_archive(
                        archive_root=archive, inventory_path=self.inventory
                    )
                self.inventory.unlink()

    def test_verify_rejects_file_changed_after_its_hash(self) -> None:
        inventory = self._build()
        archive = self._copy_layout(inventory)
        first = archive / inventory["controls"]["matrix"]["archive_path"]
        original_identity = campaign_archive._content_identity
        calls = 0

        def mutate_after_first_hash(path: Path, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                first.write_bytes(first.read_bytes() + b"changed")
            return original_identity(path, **kwargs)

        with mock.patch.object(
            campaign_archive,
            "_content_identity",
            side_effect=mutate_after_first_hash,
        ), self.assertRaisesRegex(ValueError, "changed while verifying"):
            campaign_archive.verify_campaign_archive(
                archive_root=archive, inventory_path=self.inventory
            )


if __name__ == "__main__":
    unittest.main()
