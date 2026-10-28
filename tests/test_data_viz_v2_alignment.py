from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from squeakview.common import controller_protocol_v2 as protocol

from data_viz.v2_alignment import (
    associate_events_to_frames,
    find_latest_run,
    load_v2_run,
)


class LatestRunSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.projects = self.root / "projects"
        self.projects.mkdir()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _project_with_run(self, name: str, marker_time_ns: int) -> tuple[Path, Path]:
        project = self.projects / name
        run = project / "runs" / "experiment" / f"{name}-run"
        run.mkdir(parents=True)
        marker = project / "runs" / ".latest_run"
        marker.write_text(str(run.resolve()) + "\n", encoding="utf-8")
        os.utime(marker, ns=(marker_time_ns, marker_time_ns))
        return project, run

    def test_selects_newest_marker_across_projects(self) -> None:
        first_project, _ = self._project_with_run("first", 100)
        _, second_run = self._project_with_run("second", 200)
        catalog = self.root / "recent-projects.json"
        catalog.write_text(
            json.dumps({"schema_version": 1, "paths": [str(first_project)]}),
            encoding="utf-8",
        )

        selected = find_latest_run(
            recent_projects_path=catalog,
            projects_parent=self.projects,
        )

        self.assertEqual(selected, second_run.resolve())

    def test_explicit_project_limits_selection(self) -> None:
        first_project, first_run = self._project_with_run("first", 100)
        self._project_with_run("second", 200)

        selected = find_latest_run(project_root=first_project)

        self.assertEqual(selected, first_run.resolve())

    def test_rejects_marker_outside_project_runs_directory(self) -> None:
        project = self.projects / "first"
        runs = project / "runs"
        runs.mkdir(parents=True)
        outside = self.root / "outside-run"
        outside.mkdir()
        (runs / ".latest_run").write_text(
            str(outside.resolve()) + "\n", encoding="utf-8"
        )

        with self.assertRaisesRegex(FileNotFoundError, "No project run marker"):
            find_latest_run(project_root=project)


class V2AlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp_dir.name)
        diagnostics = self.run_dir / "diagnostics"
        diagnostics.mkdir()
        (self.run_dir / "run_status.json").write_text(
            json.dumps(
                {
                    "state": "finalized",
                    "overall_validation_passed": True,
                    "recording_validation_passed": True,
                }
            ),
            encoding="utf-8",
        )
        (self.run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "serial": {
                        "controller_protocol": "v2",
                        "alignment_required": False,
                    }
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            {
                "stream_id": [0, 0, 0, 0],
                "source_sequence_index": [0, 1, 2, 3],
                "raw_frame_index": [0, 1, 2, 3],
                "camera_frame_id": [101, 102, 103, 104],
                "camera_timestamp_ns": [10, 20, 30, 40],
                "gst_pts_ns": [0, 1_000_000, 2_000_000, 3_000_000],
            }
        ).to_csv(self.run_dir / "frames.csv", index=False)
        self.rows = [
            self._journal_row(
                10,
                3,
                1_000,
                "CAMERA_EPOCH,count=1,timestamp_us=1000,period_us=1000,"
                "pulse_us=10,health=0x3,queue=0/8,suppressed=0,reason=Start",
            ),
            self._journal_row(
                11,
                1,
                2_500,
                "POKE_START,123,2500,L,1,69420,69420,69420,Eligible,nan",
            ),
            self._journal_row(
                12,
                4,
                3_000,
                "CAMERA_CHECKPOINT,count=3,timestamp_us=3000,period_us=1000,"
                "pulse_us=10,health=0x3,queue=0/8,suppressed=0,reason=Periodic",
            ),
            self._journal_row(13, 2, 3_500, "ACK_CLEAR_JAM"),
            self._journal_row(
                14,
                5,
                4_000,
                "CAMERA_STOP,count=4,timestamp_us=4000,period_us=1000,"
                "pulse_us=10,health=0x1,queue=0/8,suppressed=0,reason=StopCommand",
            ),
        ]
        self._write_journal()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _journal_row(
        sequence: int, message_type: int, monotonic_us: int, payload: str
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "host_unix_ns": 1_000_000_000 + sequence,
            "host_monotonic_ns": 2_000_000_000 + sequence,
            "message_type": message_type,
            "flags": 1,
            "boot_id": 12_345,
            "session_id": 7,
            "sequence": sequence,
            "monotonic_us": monotonic_us,
            "payload_utf8": payload,
            "decoded_hex": protocol.cobs_decode(protocol.encode_frame(
                message_type=message_type, flags=1, boot_id=12_345, session_id=7,
                sequence=sequence, monotonic_us=monotonic_us, payload=payload.encode(),
            )[:-1]).hex(),
        }

    def _write_journal(self) -> None:
        diagnostics = self.run_dir / "diagnostics"
        (diagnostics / "controller_v2.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows),
            encoding="utf-8",
        )
        (diagnostics / "controller_v2_summary.json").write_text(
            json.dumps(
                {
                    "protocol": "mousehouse_v2",
                    "boot_id": 12_345,
                    "session_id": 7,
                    "integrity_latched": False,
                    "counts": {
                        "frames_stored": len(self.rows),
                        "crc_or_framing_errors": 0,
                        "conflicting_duplicates": 0,
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_primary_layout_reads_metadata_without_changing_alignment(self) -> None:
        from squeakview.common.run_layout import initialize_layout, artifact_path
        initialize_layout(self.run_dir)
        for name in ("run_status.json", "run_manifest.json", "diagnostics"):
            (self.run_dir / name).rename(artifact_path(self.run_dir, name))
        frames = pd.read_csv(self.run_dir / "frames.csv")
        frames["controller_time_us"] = [1000, 2000, 3000, 4000]
        frames["controller_time_method"] = ["exact_anchor", "piecewise_interpolated", "exact_anchor", "exact_anchor"]
        frames["controller_boot_id"] = 12_345
        frames["controller_session_id"] = 7
        frames.to_csv(self.run_dir / "frames.csv", index=False)
        run = load_v2_run(self.run_dir)
        self.assertEqual(run.frames["frame_controller_us"].tolist(), [1000, 2000, 3000, 4000])
        self.assertFalse(run.events.empty)

    def test_loads_sparse_v2_anchors_and_interpolates_frames(self) -> None:
        run = load_v2_run(self.run_dir)

        self.assertEqual(run.frames["controller_count"].tolist(), [1, 2, 3, 4])
        self.assertEqual(
            run.frames["frame_controller_us"].tolist(), [1000, 2000, 3000, 4000]
        )
        self.assertEqual(
            run.frames["controller_time_method"].tolist(),
            [
                "exact_anchor",
                "piecewise_interpolated",
                "exact_anchor",
                "exact_anchor",
            ],
        )
        self.assertEqual(run.anchors["record"].tolist(), [
            "CAMERA_EPOCH",
            "CAMERA_CHECKPOINT",
            "CAMERA_STOP",
        ])

    def test_associates_behavior_with_preceding_reconstructed_frame(self) -> None:
        run = load_v2_run(self.run_dir)

        associated = associate_events_to_frames(run.events, run.frames)
        poke = associated.loc[associated["eventType"].eq("POKE_START")].iloc[0]

        self.assertEqual(int(poke["controller_count"]), 2)
        self.assertAlmostEqual(float(poke["offset_from_frame_ms"]), 0.5)

    def test_rejects_controller_frame_count_mismatch(self) -> None:
        self.rows[-1] = self._journal_row(
            14,
            5,
            5_000,
            "CAMERA_STOP,count=5,timestamp_us=5000,period_us=1000,"
            "pulse_us=10,health=0x1,queue=0/8,suppressed=0,reason=StopCommand",
        )
        self._write_journal()

        with self.assertRaisesRegex(ValueError, "controller/frame count mismatch"):
            load_v2_run(self.run_dir)

    def test_rejects_durable_sequence_gap(self) -> None:
        self.rows[2]["sequence"] = 15
        self._write_journal()

        with self.assertRaisesRegex(ValueError, "durable sequence gap"):
            load_v2_run(self.run_dir)

    def test_rejects_reported_controller_boot_boundary(self) -> None:
        summary_path = self.run_dir / "diagnostics" / "controller_v2_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["counts"]["boot_boundaries"] = 1
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "boot boundary"):
            load_v2_run(self.run_dir)

    def _behavior_only(self, *, mapped=True):
        manifest_path = self.run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["serial"]["enabled"] = True
        manifest["capture"] = {"trigger_on": False, "num_cameras": 1}
        manifest_path.write_text(json.dumps(manifest))
        self.rows = [self._journal_row(10, 2, 1000, "ACK_START_BEHAVIOR"),
                     self._journal_row(11, 1, 2500, "POKE_START,123,2500,L,1,69420,69420,69420,Eligible,nan"),
                     self._journal_row(12, 2, 5000, "ACK_STOP")]
        self._write_journal()
        if mapped:
            (self.run_dir / "diagnostics" / "clock_validation.json").write_text(json.dumps({
                "result": "PASS", "before_samples": [{"round_trip_ns": 200,
                    "rp2040_receive_us": 1000, "rp2040_send_us": 1000,
                    "jetson_send_ns": 1000000000, "jetson_receive_ns": 1000000200}]}))
        from squeakview.common.run_outputs import ClockMapping
        frames = pd.read_csv(self.run_dir / "frames.csv")
        frames["host_unix_ns"] = [1000000100, 1001000100, 1002000100, 1003000100]
        mapping = ClockMapping(self.run_dir)
        pd.DataFrame([mapping.apply(row) for row in frames.to_dict("records")]).to_csv(self.run_dir / "frames.csv", index=False)

    def test_free_running_uses_labelled_estimates_without_ttl_counts(self):
        self._behavior_only()
        run = load_v2_run(self.run_dir)
        self.assertTrue(run.anchors.empty)
        self.assertTrue(run.frames["controller_count"].isna().all())
        self.assertEqual(run.frames["controller_time_method"].unique().tolist(), ["host_clock_estimate"])
        self.assertEqual(run.frames["frame_controller_us"].tolist(), [1000, 2000, 3000, 4000])
        associated = associate_events_to_frames(run.events, run.frames)
        self.assertEqual(int(associated.loc[associated.eventType.eq("POKE_START"), "raw_frame_index"].iloc[0]), 1)

    def test_unavailable_free_running_clock_stays_unavailable(self):
        self._behavior_only(mapped=False)
        run = load_v2_run(self.run_dir)
        self.assertTrue(run.frames.frame_controller_us.isna().all())
        with self.assertRaisesRegex(ValueError, "without a controller clock mapping"):
            associate_events_to_frames(run.events, run.frames)

    def test_changed_exported_clock_mapping_is_rejected(self):
        self._behavior_only()
        frames = pd.read_csv(self.run_dir / "frames.csv")
        frames.loc[1, "controller_time_us"] += 1
        frames.to_csv(self.run_dir / "frames.csv", index=False)
        with self.assertRaisesRegex(ValueError, "mapping disagrees"):
            load_v2_run(self.run_dir)

    def test_rejects_journal_projection_disagreeing_with_crc_bytes(self):
        self.rows[1]["payload_utf8"] = "changed"
        self._write_journal()
        with self.assertRaisesRegex(ValueError, "payload disagrees"):
            load_v2_run(self.run_dir)

    def test_one_frame_epoch_stop_and_multi_camera_timing(self):
        from squeakview.common.v2_timing import verify_v2_frame_timing
        self.rows = [self._journal_row(10, 3, 1000, "CAMERA_EPOCH,count=1,timestamp_us=1000,period_us=1000,pulse_us=10,health=0x3,queue=0/8,suppressed=0,reason=Start"),
                     self._journal_row(11, 5, 1100, "CAMERA_STOP,count=1,timestamp_us=1000,period_us=1000,pulse_us=10,health=0x1,queue=0/8,suppressed=0,reason=StopCommand")]
        self._write_journal()
        frames = pd.read_csv(self.run_dir / "frames.csv").iloc[:1]
        frames.to_csv(self.run_dir / "frames.csv", index=False)
        self.assertEqual(load_v2_run(self.run_dir).frames.frame_controller_us.tolist(), [1000])
        frames["stream_id"] = 1
        frames.to_csv(self.run_dir / "frames_cam1.csv", index=False)
        self.assertEqual(len(verify_v2_frame_timing(self.run_dir, triggered=True, camera_count=2)["cameras"]), 2)

    def test_qualification_dispatch_and_derived_reports(self):
        from squeakview.common.diagnostics.qualification import _integrity_gates
        from squeakview.common.analysis_timing import summarize_recording_timing
        for triggered in (True, False):
            if not triggered:
                self._behavior_only()
            manifest = json.loads((self.run_dir / "run_manifest.json").read_text())
            manifest["serial"]["enabled"] = True
            manifest.setdefault("capture", {})["trigger_on"] = triggered
            manifest["capture"]["num_cameras"] = 1
            (self.run_dir / "run_manifest.json").write_text(json.dumps(manifest))
            gates = _integrity_gates({}, manifest, self.run_dir)
            self.assertTrue(gates["alignment_validated_when_required"])
            self.assertTrue(gates["v2_controller_timing_validated_when_required"])
            with tempfile.TemporaryDirectory() as output:
                report = summarize_recording_timing(self.run_dir, Path(output), video_validation={"count": 4})
                self.assertTrue(report["validation"]["passed"])
            self.rows.pop()
            self._write_journal()
            gates = _integrity_gates({}, manifest, self.run_dir)
            self.assertFalse(gates["v2_controller_timing_validated_when_required"])

    def test_native_timing_report_requires_no_controller(self):
        from squeakview.common.analysis_timing import summarize_recording_timing
        (self.run_dir / "run_manifest.json").write_text(json.dumps({"serial": {"enabled": False}, "capture": {"trigger_on": False}}))
        with tempfile.TemporaryDirectory() as output:
            report = summarize_recording_timing(self.run_dir, Path(output), video_validation={"count": 4})
            self.assertEqual(report["timing"]["method"], "camera_native")


if __name__ == "__main__":
    unittest.main()
