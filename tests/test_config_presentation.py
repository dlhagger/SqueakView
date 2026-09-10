from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from squeakview import model_package
from squeakview.apps.operator.gui.config_presentation import (
    ConfigPresentation,
    present_config,
)
from squeakview.apps.operator.gui.main_window import MainWindow


def _config(**updates: object) -> dict[str, object]:
    data: dict[str, object] = {
        "width": 1920,
        "height": 1080,
        "fps": 60,
        "pixel_format": "BayerRG8",
        "num_cameras": 2,
        "trigger_on": True,
        "inference_enabled": True,
        "bitrate": 12000,
        "serial_enabled": True,
        "serial_port": "/dev/ttyACM0",
        "serial_baud": 115200,
        "arduino_fps": 60,
        "experiment_name": "study_a",
        "mouse_id": "mouse_7",
    }
    data.update(updates)
    return data


class ConfigPresentationTest(unittest.TestCase):
    def test_presents_resolved_scientific_identity_and_capture_settings(self) -> None:
        model = mock.Mock(name="MouseDetector")
        model.name = "MouseDetector"
        with mock.patch(
            "squeakview.apps.operator.gui.config_presentation."
            "model_package.validate_model_package",
            return_value=model,
        ):
            result = present_config(
                _config(),
                ds_cfg=Path("/models/mouse/config.txt"),
                task_cfg=Path("/tasks/two_bottle.yaml"),
            )

        self.assertTrue(result.inference_enabled)
        self.assertEqual(result.session_text, "study_a / Subject mouse_7")
        self.assertIn("1920×1080 @ 60 FPS · BayerRG8 · 2 cam", result.summary_html)
        self.assertIn("MouseDetector", result.summary_html)
        self.assertIn("two_bottle.yaml", result.summary_html)
        self.assertIn("/dev/ttyACM0 @ 115200", result.summary_html)
        self.assertEqual(result.preview_info, "1920×1080 · 60 FPS · Trig · 12000 kbps")

    def test_inference_off_and_invalid_package_copy_match_operator_policy(self) -> None:
        off = present_config(_config(inference_enabled=False), ds_cfg=None, task_cfg=None)
        self.assertIn("Inference off", off.summary_html)
        self.assertIn("N/A", off.summary_html)

        with mock.patch(
            "squeakview.apps.operator.gui.config_presentation."
            "model_package.validate_model_package",
            side_effect=model_package.ModelPackageError("bad package"),
        ):
            invalid = present_config(
                _config(),
                ds_cfg=Path("/models/broken/config.txt"),
                task_cfg=None,
            )
        self.assertIn("Invalid: config.txt", invalid.summary_html)

    def test_user_controlled_identity_is_html_escaped(self) -> None:
        result = present_config(
            _config(experiment_name="<study>", mouse_id="mouse&7"),
            ds_cfg=None,
            task_cfg=None,
        )

        self.assertEqual(result.session_text, "<study> / Subject mouse&7")
        self.assertNotIn("<study>", result.summary_html)
        self.assertIn("&lt;study&gt; / Subject mouse&amp;7", result.summary_html)

    def test_main_window_adapter_applies_presentation_without_mutating_backend(self) -> None:
        data = _config()
        ds_cfg = Path("/models/mouse/config.txt")
        task_cfg = Path("/tasks/two_bottle.yaml")
        resolved = mock.Mock(data=data, ds_cfg=ds_cfg, task_cfg=task_cfg)
        presentation = ConfigPresentation(
            inference_enabled=True,
            session_text="Study / Subject mouse_7",
            summary_html="<table>summary</table>",
            preview_info="preview info",
        )
        host = mock.Mock()
        host._recording_active = False
        host._start_in_progress = False
        host._stop_in_progress = False
        host.backend.runtime_config = mock.sentinel.runtime_config

        with (
            mock.patch(
                "squeakview.apps.operator.gui.main_window.resolve_config_paths",
                return_value=resolved,
            ),
            mock.patch(
                "squeakview.apps.operator.gui.main_window.present_config",
                return_value=presentation,
            ) as present,
        ):
            MainWindow._apply_config(host, {"unresolved": True})

        present.assert_called_once_with(data, ds_cfg=ds_cfg, task_cfg=task_cfg)
        host.run_identity_label.setText.assert_called_once_with(presentation.session_text)
        host.summary_label.setText.assert_called_once_with(presentation.summary_html)
        host.preview.set_info.assert_called_once_with(presentation.preview_info)
        host._set_run_state.assert_called_once_with("ready")
        host.dashboard.apply_task_config.assert_called_once_with(task_cfg)
        self.assertEqual(host._config_data, data)
        self.assertIs(host.backend.runtime_config, mock.sentinel.runtime_config)


if __name__ == "__main__":
    unittest.main()
