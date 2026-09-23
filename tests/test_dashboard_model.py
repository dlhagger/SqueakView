from __future__ import annotations

import unittest

from squeakview.apps.operator.gui import dashboard_model
from squeakview.common.dashboard import DashboardEvent


class DashboardConfigTests(unittest.TestCase):
    def test_compile_falls_back_atomically_for_malformed_dashboard(self) -> None:
        definition = dashboard_model.compile_task_config(
            {"events": [{"name": "BROKEN"}], "dashboard": "not-a-mapping"}
        )

        self.assertEqual(definition.series_order[:4], ("POKE_L", "DRINK_L", "POKE_R", "DRINK_R"))
        self.assertNotIn("BROKEN", definition.series_order)
        self.assertFalse(definition.settings_panel)

    def test_compile_normalizes_rules_and_adds_unplotted_series(self) -> None:
        definition = dashboard_model.compile_task_config(
            {
                "events": [
                    {
                        "name": " reward ",
                        "match": {"event_contains": " pellet ", "value_equals": 2},
                        "split_by_side": True,
                        "use_count_field": True,
                    }
                ],
                "dashboard": {
                    "settings_panel": True,
                    "plots": [{"id": "custom", "series": ["existing"]}],
                },
            }
        )

        self.assertEqual(definition.series_order, ("EXISTING", "REWARD_L", "REWARD_R"))
        self.assertTrue(definition.settings_panel)
        self.assertEqual(definition.rules[0]["match"]["event_contains"], "pellet")
        self.assertTrue(definition.rules[0]["use_count_field"])

    def test_default_config_is_not_shared_between_callers(self) -> None:
        first = dashboard_model.default_task_config()
        first["events"][0]["match"]["phase"] = "changed"

        second = dashboard_model.default_task_config()
        self.assertEqual(second["events"][0]["match"]["phase"], "start")

    def test_default_dashboard_combines_behavior_on_one_time_plot(self) -> None:
        definition = dashboard_model.compile_task_config(
            dashboard_model.default_task_config()
        )

        self.assertEqual(len(definition.plots), 1)
        self.assertEqual(definition.plots[0]["id"], "Behavior")
        self.assertEqual(definition.plots[0]["type"], "event_raster")
        self.assertEqual(
            definition.plots[0]["series"],
            [
                "POKE_L",
                "DRINK_L",
                "POKE_R",
                "DRINK_R",
                "PELLET",
                "WELL_CHECK",
            ],
        )


class DashboardEventSemanticsTests(unittest.TestCase):
    def test_feeder_jam_protocol_classification_is_exact(self) -> None:
        cases = {
            "FEED_JAM,100,200,nan,3,69420,69420,69420,Feeding,Pellet did not trigger sensor": dashboard_model.FeederJamEvent.JAMMED,
            "NACK,FEED,JAMMED": dashboard_model.FeederJamEvent.JAMMED,
            "ACK_CLEAR_JAM": dashboard_model.FeederJamEvent.CLEAR_ACK,
            "NACK,CLEAR_JAM,FEED_ACTIVE": dashboard_model.FeederJamEvent.CLEAR_FEED_ACTIVE,
            "NACK,CLEAR_JAM,NOT_JAMMED": dashboard_model.FeederJamEvent.CLEAR_NOT_JAMMED,
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                event = DashboardEvent.parse(raw)
                assert event is not None
                self.assertEqual(dashboard_model.feeder_jam_event(event), expected)

    def test_unrelated_and_plain_feed_stop_events_do_not_change_jam_state(self) -> None:
        for raw in (
            "POKE_START,10,20,L,1,30,40,50,Eligible,nan",
            "FEED_STOP,10,20,nan,1,30,40,50,Feeding,Complete",
            "NACK,FEED,OTHER",
        ):
            with self.subTest(raw=raw):
                event = DashboardEvent.parse(raw)
                assert event is not None
                self.assertIsNone(dashboard_model.feeder_jam_event(event))

    def test_typed_event_is_consumed_without_reparsing_raw_line(self) -> None:
        event = DashboardEvent.parse(
            "TASK_INFO,1000000,2000000,L,3,12500,2000500,70,ON,adaptive"
        )
        assert event is not None

        data = dashboard_model.event_data(event)

        self.assertIsNotNone(data)
        assert data is not None
        self.assertEqual(data["event_uc"], "TASK_INFO")
        self.assertEqual(data["unix_sec"], 1.0)
        self.assertEqual(data["reason"], "adaptive")

    def test_raw_event_compatibility_is_confined_to_model_edge(self) -> None:
        data = dashboard_model.event_data("PELLET_ARRIVAL")

        self.assertIsNotNone(data)
        assert data is not None
        self.assertEqual(data["event_uc"], "PELLET_ARRIVAL")

    def test_pellet_mode_tracks_both_wire_conventions(self) -> None:
        observed = dashboard_model.infer_pellet_mode(
            "auto", None, {"event_uc": "PELLET_ARRIVAL"}, "PELLET_ARRIVAL"
        )
        observed = dashboard_model.infer_pellet_mode(
            "auto", observed, {"event_uc": "PELLET_RETRIEVAL"}, "PELLET_RETRIEVAL"
        )

        self.assertEqual(observed, "both")
        self.assertEqual(dashboard_model.effective_pellet_mode("auto", observed), "both")

    def test_rule_matching_handles_retrieval_event_in_arrival_mode(self) -> None:
        rule = {"match": {"event_contains": "PELLET", "phase": "start"}}
        self.assertFalse(
            dashboard_model.rule_matches(
                {"event_uc": "PELLET_RETRIEVAL"},
                "PELLET_RETRIEVAL",
                rule,
                pellet_mode="arrival",
            )
        )
        self.assertTrue(
            dashboard_model.rule_matches(
                {"event_uc": "PELLET_RETRIEVAL"},
                "PELLET_RETRIEVAL",
                rule,
                pellet_mode="retrieval",
            )
        )

    def test_task_info_decodes_units_without_qt(self) -> None:
        updates = dashboard_model.task_settings_update(
            {
                "count": "3",
                "duration_us": "12500",
                "latency_us": "2000500",
                "value": "70",
                "reason": "adaptive",
            },
            "TASK_INFO",
        )

        self.assertEqual(
            updates,
            {
                "stage": "3",
                "hold_ms": "12",
                "go_ms": "2000",
                "go_pct": "70",
                "reason": "adaptive",
            },
        )

    def test_side_update_rejects_unknown_values(self) -> None:
        self.assertEqual(
            dashboard_model.task_settings_update({"side_uc": "CENTER"}, "SIDE_SET"),
            {},
        )

    def test_parse_int_rejects_nan_and_infinity(self) -> None:
        self.assertIsNone(dashboard_model.parse_int_field("nan"))
        self.assertIsNone(dashboard_model.parse_int_field(float("inf")))


if __name__ == "__main__":
    unittest.main()
