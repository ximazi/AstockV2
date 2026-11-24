from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mainwave_engine import observation as O
from mainwave_engine import theme_lifecycle as TL
from mainwave_engine.theme_registry import themes_as_of


class RegistryTests(unittest.TestCase):
    def test_registry_is_point_in_time(self) -> None:
        self.assertEqual(themes_as_of("2026-08-26"), [])
        self.assertEqual({x["id"] for x in themes_as_of("2026-08-27")},
                         {"storage", "optical_cpo", "ccl_fiberglass"})


class LifecycleTests(unittest.TestCase):
    @staticmethod
    def metrics() -> pd.DataFrame:
        return pd.DataFrame({
            "date": [f"2026-01-{i:02d}" for i in range(1, 9)],
            "batch_event": [False, True, False, False, False, False, False, False],
            "ret20": [0.0] * 8,
            "breadth_full_5d": [False] * 8,
            "drawdown_120": [0.0, 0.0, -0.30, -0.30, -0.30, -0.30, -0.30, -0.30],
            "breadth": [0.5] * 8,
            "pulse_members": [0] * 8,
            "no_pulse_days": list(range(1, 9)),
        })

    def test_batch_and_retreat_are_immediate_hard_events(self) -> None:
        theme = {"active_from": "2026-01-01", "historical_cycle_anchors": []}
        timeline = TL._stage_timeline(self.metrics(), theme)
        self.assertEqual(timeline.loc[1, "stage"], "IGNITED")
        self.assertEqual(timeline.loc[2, "stage"], "RETREAT")

    def test_crowded_does_not_block_next_day_retreat(self) -> None:
        metrics = self.metrics()
        metrics.loc[2, ["drawdown_120", "ret20"]] = [0.0, 0.50]
        metrics.loc[3, ["drawdown_120", "ret20"]] = [-0.30, 0.10]
        theme = {"active_from": "2026-01-01", "historical_cycle_anchors": []}
        timeline = TL._stage_timeline(metrics, theme)
        self.assertEqual(timeline.loc[2, "stage"], "CROWDED")
        self.assertEqual(timeline.loc[3, "stage"], "RETREAT")


class ObservationTests(unittest.TestCase):
    def test_placeholder_stages_do_not_override_production_state(self) -> None:
        evidence = {"data_fresh": True, "imp": True, "supply_ready": True,
                    "structure_ready": False, "health": True}
        self.assertEqual(O.derive_capital_stage(evidence, "FORMING"), "ABSORBING")
        self.assertEqual(O.derive_capital_stage(evidence, "ARMED"), "ARMED")

    def test_trajectory_upsert_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"date": "2026-08-28", "symbol": "sh_600000", "state": "FORMING"}
            daily = root / "daily.csv"
            log = root / "log.csv"
            O.write_trajectory([row], daily, log)
            O.write_trajectory([row], daily, log)
            self.assertEqual(len(pd.read_csv(log, encoding="utf-8-sig")), 1)


if __name__ == "__main__":
    unittest.main()
