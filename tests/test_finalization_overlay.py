from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from squeakview.apps.operator.gui.finalization_overlay import (
    FinalizationOverlay,
    present_finalization,
)


class FinalizationPresenterTests(unittest.TestCase):
    def test_every_persisted_stage_has_stable_operator_copy(self) -> None:
        expected_titles = {
            "stopping": "Stopping Capture…",
            "capture_draining": "Draining Capture…",
            "capture_drained": "Closing Capture…",
            "capture_closed": "Capture Saved — Validating…",
            "capture_reconciliation": "Validating Capture…",
            "recording_count_validation": "Checking Recording Counts…",
            "recording_count_validation_complete": "Recording Validated…",
            "inference_admission": "Validating Inference…",
            "recording_validation": "Validating Video…",
            "recording_validation_complete": "Video Validated…",
            "streaming_alignment": "Aligning Timing…",
            "complete": "Validation Complete…",
        }
        for stage, title in expected_titles.items():
            with self.subTest(stage=stage):
                if stage.startswith("capture_") and stage in {
                    "capture_draining",
                    "capture_drained",
                    "capture_closed",
                }:
                    status = {"state": stage}
                    progress = {"stage": "must_not_override_capture_state"}
                elif stage == "stopping":
                    status = {"state": stage}
                    progress = {}
                else:
                    status = {"state": "finalizing", "stage": stage}
                    progress = {}
                self.assertEqual(present_finalization(status, progress).title, title)

    def test_status_stage_precedes_progress_during_finalizing(self) -> None:
        presentation = present_finalization(
            {"state": "finalizing", "stage": "recording_validation"},
            {"stage": "streaming_alignment"},
        )
        self.assertEqual(presentation.stage, "recording_validation")

    def test_progress_stage_and_frame_count_are_used_for_other_states(self) -> None:
        presentation = present_finalization(
            {"state": "unexpected"},
            {"stage": "streaming_alignment", "frames_processed": 1234},
        )
        self.assertEqual(presentation.title, "Aligning Timing…")
        self.assertTrue(presentation.message.endswith("1,234 frames processed."))

    def test_unknown_stage_preserves_fallback_copy(self) -> None:
        presentation = present_finalization(
            {"state": "finalizing", "stage": "future_stage"},
            {},
        )
        self.assertEqual(presentation.title, "Finalizing Run…")
        self.assertEqual(presentation.message, "Post-run stage: future_stage")

    def test_video_decode_progress_reports_rate_eta_and_percentage(self) -> None:
        presentation = present_finalization(
            {"state": "finalizing", "stage": "recording_validation"},
            {
                "video_frames_decoded": 5_000,
                "video_frames_expected": 20_000,
                "video_validation_elapsed_s": 100.0,
                "video_validation_rate_fps": 50.0,
                "video_validation_eta_s": 300.0,
            },
        )

        self.assertEqual(presentation.progress_percent, 25)
        self.assertIn(
            "5,000 of 20,000 video samples (25.0%)", presentation.message
        )
        self.assertIn("50.0 frames/s", presentation.message)
        self.assertIn("About 5m 00s remaining", presentation.message)


class FinalizationOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self.parent = QtWidgets.QWidget()
        self.parent.resize(800, 600)
        self.overlay = FinalizationOverlay(self.parent)

    def tearDown(self) -> None:
        self.parent.close()
        self.parent.deleteLater()
        self.app.processEvents()

    def test_widget_preserves_object_names_indeterminate_bar_and_geometry(self) -> None:
        self.assertEqual(self.overlay.objectName(), "stopOverlay")
        self.assertEqual(self.overlay.title_label.objectName(), "stopOverlayTitle")
        self.assertEqual(self.overlay.message_label.objectName(), "stopOverlayMsg")
        self.assertEqual(
            (
                self.overlay.progress_bar.minimum(),
                self.overlay.progress_bar.maximum(),
            ),
            (0, 0),
        )
        self.assertEqual(self.overlay.geometry(), self.parent.rect())

    def test_show_hide_and_progress_update(self) -> None:
        self.overlay.show_overlay()
        self.assertFalse(self.overlay.isHidden())
        presentation = self.overlay.update_progress(
            {"state": "capture_closed"},
            {"frames_processed": 17},
        )
        self.assertEqual(self.overlay.title_label.text(), presentation.title)
        self.assertEqual(self.overlay.message_label.text(), presentation.message)

        self.overlay.hide_overlay()
        self.assertTrue(self.overlay.isHidden())

    def test_video_progress_switches_bar_to_determinate_mode(self) -> None:
        self.overlay.update_progress(
            {"state": "finalizing", "stage": "recording_validation"},
            {"video_frames_decoded": 9, "video_frames_expected": 10},
        )

        self.assertEqual(
            (self.overlay.progress_bar.minimum(), self.overlay.progress_bar.maximum()),
            (0, 100),
        )
        self.assertEqual(self.overlay.progress_bar.value(), 90)
        self.assertTrue(self.overlay.progress_bar.isTextVisible())
        self.assertEqual(self.overlay.progress_bar.format(), "%p% complete")

    def test_compact_overlay_keeps_copy_and_progress_separated(self) -> None:
        self.parent.resize(480, 300)
        self.parent.show()
        self.overlay.resize_to_parent()
        self.overlay.update_progress(
            {"state": "finalizing", "stage": "recording_validation"},
            {
                "video_frames_decoded": 10_326,
                "video_frames_expected": 20_000,
                "video_validation_rate_fps": 8_500.0,
                "video_validation_eta_s": 1.0,
            },
        )
        self.overlay.show_overlay()
        self.app.processEvents()

        self.assertLessEqual(self.overlay.content_panel.width(), 432)
        self.assertLess(
            self.overlay.title_label.geometry().bottom(),
            self.overlay.message_label.geometry().top(),
        )
        self.assertLess(
            self.overlay.message_label.geometry().bottom(),
            self.overlay.progress_bar.geometry().top(),
        )
        self.assertLessEqual(
            self.overlay.progress_bar.width(),
            self.overlay.content_panel.width(),
        )


if __name__ == "__main__":
    unittest.main()
