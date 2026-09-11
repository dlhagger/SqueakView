from __future__ import annotations

import sys
import subprocess
import unittest

from squeakview.apps.operator.gui import dashboard_presentation


class DashboardPresentationTests(unittest.TestCase):
    def test_module_is_qt_free(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import squeakview.apps.operator.gui.dashboard_presentation; "
                "assert 'PySide6' not in sys.modules",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_known_and_fallback_series_presentation_is_stable(self) -> None:
        self.assertEqual(dashboard_presentation.series_label("POKE_L"), "Left poke")
        self.assertEqual(dashboard_presentation.series_color("POKE_L", 4), "#37d67a")
        self.assertEqual(dashboard_presentation.series_label("CUSTOM_EVENT"), "Custom Event")
        self.assertEqual(dashboard_presentation.series_color("CUSTOM_EVENT", 7), "#06d6a0")

    def test_count_summary_preserves_plot_group_and_series_order(self) -> None:
        rendered = dashboard_presentation.counts_html(
            [("POKE_L", "POKE_R"), ("PELLET",)],
            {"POKE_L": 4, "POKE_R": 7, "PELLET": 2},
        )

        self.assertLess(rendered.index("LEFT POKE"), rendered.index("RIGHT POKE"))
        self.assertLess(rendered.index("RIGHT POKE"), rendered.index("PELLET"))
        self.assertIn("#37d67a; font-weight:800;'>4", rendered)
        self.assertIn("#6fa8ff; font-weight:800;'>7", rendered)
        self.assertEqual(rendered.count("color:#46506d"), 1)

    def test_count_summary_escapes_custom_task_series_labels(self) -> None:
        rendered = dashboard_presentation.counts_html(
            [("<unsafe>",)], {"<unsafe>": 1}
        )

        self.assertIn("&lt;UNSAFE&gt;", rendered)
        self.assertNotIn("<UNSAFE>", rendered)

    def test_trim_discards_whole_steps_and_keeps_left_edge_value(self) -> None:
        xs = [1.0, 2.0, 2.0, 4.0, 4.0, 8.0]
        ys = [0, 1, 1, 2, 2, 3]

        dashboard_presentation.trim_step_series(xs, ys, xstart=5.0)

        self.assertEqual(xs, [5.0, 8.0])
        self.assertEqual(ys, [2, 3])

    def test_curve_extension_does_not_grow_retained_history(self) -> None:
        xs = [10.0, 12.0]
        ys = [0, 1]

        plot_x, plot_y = dashboard_presentation.curve_points(xs, ys, now=20.0)

        self.assertEqual(plot_x, [10.0, 12.0, 20.0])
        self.assertEqual(plot_y, [0, 1, 1])
        self.assertEqual(xs, [10.0, 12.0])
        self.assertEqual(ys, [0, 1])

    def test_hard_cap_discards_complete_oldest_transitions(self) -> None:
        xs = [float(value) for value in range(12)]
        ys = list(range(12))

        dashboard_presentation.cap_step_series(xs, ys, max_points=7)

        self.assertEqual(xs, [6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        self.assertEqual(ys, [6, 7, 8, 9, 10, 11])

    def test_event_raster_history_is_trimmed_and_bounded(self) -> None:
        timestamps = [1.0, 2.0, 3.0, 4.0]

        dashboard_presentation.trim_event_times(timestamps, xstart=2.5)
        self.assertEqual(timestamps, [3.0, 4.0])

        timestamps.extend([5.0, 6.0])
        dashboard_presentation.cap_event_times(timestamps, max_points=3)
        self.assertEqual(timestamps, [4.0, 5.0, 6.0])

    def test_window_bounds_show_trailing_history_without_future_time(self) -> None:
        self.assertEqual(
            dashboard_presentation.window_bounds(1_000.0, 300.0),
            (700.0, 1_000.0),
        )

    def test_raster_window_starts_zoomed_and_expands_to_history_limit(self) -> None:
        self.assertEqual(
            dashboard_presentation.raster_window_bounds(100.0, 300.0, None),
            (71.5, 101.5),
        )
        self.assertEqual(
            dashboard_presentation.raster_window_bounds(400.0, 300.0, 0.0),
            (102.0, 402.0),
        )


if __name__ == "__main__":
    unittest.main()
