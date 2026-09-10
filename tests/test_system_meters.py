from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from squeakview.apps.operator.gui.dashboard import (
    JetsonMeters as DashboardJetsonMeters,
)
from squeakview.apps.operator.gui.dashboard import MetersBar as DashboardMetersBar
from squeakview.apps.operator.gui.system_meters import JetsonMeters, MetersBar


class SystemMetersCompatibilityTests(unittest.TestCase):
    def test_dashboard_preserves_meter_re_exports(self) -> None:
        self.assertIs(DashboardJetsonMeters, JetsonMeters)
        self.assertIs(DashboardMetersBar, MetersBar)


class JetsonMetersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _meters(self) -> JetsonMeters:
        with mock.patch(
            "squeakview.apps.operator.gui.system_meters.shutil.which",
            return_value=None,
        ):
            return JetsonMeters(interval_ms=60_000)

    def test_preserves_legacy_idle_tegrastats_parsing(self) -> None:
        meters = self._meters()
        try:
            raw = "RAM 250/1000MB CPU [42%@1728,84%@1728] GR3D_FREQ 63%"
            self.assertEqual(meters._parse_ram(raw), 25.0)
            self.assertEqual(meters._parse_cpu(raw), 42.0)
            self.assertEqual(meters._parse_gpu(raw), 63.0)
            self.assertNotEqual(meters._parse_ram("invalid"), meters._parse_ram("invalid"))
        finally:
            meters.stop()

    def test_capture_telemetry_takes_precedence_over_idle_queue(self) -> None:
        meters = self._meters()
        updates: list[tuple[float, float, float, float, str]] = []
        meters.updated.connect(lambda *values: updates.append(values))
        meters._queue.put_nowait("RAM 1/2MB CPU [1%@1] GR3D_FREQ 1%")
        meters._run_system_csv = Path("/capture/diagnostics/system.csv")

        with (
            mock.patch(
                "squeakview.apps.operator.gui.system_meters.read_latest_system_telemetry",
                return_value={
                    "ram_pct": "71.5",
                    "gpu_util_pct": "62",
                    "cpu_util_mean_pct": "53.25",
                    "raw_line": "capture-owned",
                },
            ) as read_latest,
            mock.patch.object(meters, "_disk_pct", return_value=44.0),
        ):
            meters._drain()

        try:
            read_latest.assert_called_once_with(Path("/capture/diagnostics/system.csv"))
            self.assertEqual(updates, [(71.5, 62.0, 53.25, 44.0, "capture-owned")])
            self.assertEqual(meters._queue.qsize(), 1)
        finally:
            meters.stop()

    def test_follow_and_resume_switch_sampling_owners(self) -> None:
        meters = self._meters()
        run_csv = Path("/run/diagnostics/system.csv")
        with (
            mock.patch.object(meters, "_stop_tegrastats") as stop_idle,
            mock.patch.object(meters, "_start_tegrastats") as start_idle,
        ):
            meters.follow_run_telemetry(run_csv)
            self.assertEqual(meters._run_system_csv, run_csv)
            stop_idle.assert_called_once_with()

            meters.resume_idle_sampling()
            self.assertIsNone(meters._run_system_csv)
            start_idle.assert_called_once_with()
        meters.stop()

    def test_idle_tegrastats_process_is_parented_and_fully_reaped(self) -> None:
        process = mock.Mock()
        process.waitForFinished.side_effect = [False, True]
        with (
            mock.patch(
                "squeakview.apps.operator.gui.system_meters.shutil.which",
                return_value="/usr/bin/tegrastats",
            ),
            mock.patch.object(QtCore, "QProcess", return_value=process) as qprocess,
        ):
            meters = JetsonMeters(interval_ms=60_000)
            meters.stop()

        qprocess.assert_called_once_with(meters)
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertEqual(
            process.waitForFinished.call_args_list,
            [mock.call(500), mock.call(2000)],
        )
        process.close.assert_called_once_with()
        process.deleteLater.assert_called_once_with()
        self.assertIsNone(meters._proc)


class MetersBarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_values_are_rounded_and_missing_values_show_na(self) -> None:
        bar = MetersBar()
        try:
            bar.set_ram_pct(49.6)
            bar.set_gpu_pct(None)
            bar.set_cpu_pct(float("nan"))

            self.assertEqual(bar.ram_bar.value(), 50)
            self.assertEqual(bar.ram_bar.format(), "50%")
            self.assertEqual(bar.gpu_bar.value(), 0)
            self.assertEqual(bar.gpu_bar.format(), "N/A")
            self.assertEqual(bar.cpu_bar.format(), "N/A")
        finally:
            bar.close()

    def test_threshold_colors_remain_stable(self) -> None:
        self.assertIn("#777", MetersBar._style_for_pct(-1.0))
        self.assertIn("#2ecc71", MetersBar._style_for_pct(49.99))
        self.assertIn("#f1c40f", MetersBar._style_for_pct(50.0))
        self.assertIn("#e74c3c", MetersBar._style_for_pct(80.0))


if __name__ == "__main__":
    unittest.main()
