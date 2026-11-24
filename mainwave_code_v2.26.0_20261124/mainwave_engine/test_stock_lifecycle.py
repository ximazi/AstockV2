from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mainwave_engine import stock_lifecycle as SL
from mainwave_engine.test_phase_events import PhaseEventTests


def observations(size=100):
    frame = pd.DataFrame({
        "date": pd.bdate_range("2025-01-01", periods=size),
        "return_index": 100.0, "structure_floor": 90.0, "structure_ceiling": 110.0,
        "broad_phase": "ACCUMULATION", "effort_result": "NEUTRAL",
        "SOS": False, "UTAD": False, "close_location": 0.6, "turnover": 2.0,
    })
    return frame


class StockLifecycleTests(unittest.TestCase):
    def test_start_then_absorption_preserves_cycle(self):
        frame = observations()
        frame.loc[70, "SOS"] = True
        trace = pd.DataFrame([{
            "symbol": "sh_600000", "date": str(frame.loc[71, "date"].date()),
            "state": "FORMING", "capital_stage": "ABSORBING", "data_fresh": True,
        }])
        result = SL.replay("sh_600000", frame, trace)
        self.assertEqual(result.loc[70, "cycle_stage"], "IGNITION")
        self.assertEqual(result.loc[71, "cycle_stage"], "PULLBACK")
        self.assertEqual(result.loc[70, "cycle_id"], result.loc[99, "cycle_id"])
        self.assertEqual(result.loc[71, "cycle_reason"], "POST_START_PAUSE_NOT_NEW_ACCUMULATION")

    def test_breakdown_does_not_silently_restart(self):
        frame = observations()
        frame.loc[60, "SOS"] = True
        frame.loc[62, "return_index"] = 85
        frame.loc[63, "SOS"] = True
        result = SL.replay("sh_600000", frame)
        self.assertEqual(result.loc[62, "cycle_stage"], "FAILED")
        self.assertEqual(result.loc[63, "cycle_stage"], "FAILED")
        self.assertEqual(result.loc[99, "cycle_stage"], "REBUILDING")
        self.assertEqual(result.loc[99, "cycle_number"], 1)
        trace = pd.DataFrame([{"symbol": "sh_600000", "date": str(frame.loc[90, "date"].date()),
                               "state": "TRIGGERED", "data_fresh": True}])
        rebuilt = SL.replay("sh_600000", frame, trace)
        self.assertEqual(rebuilt.loc[90, "cycle_number"], 2)
        self.assertNotEqual(rebuilt.loc[60, "cycle_id"], rebuilt.loc[90, "cycle_id"])
        self.assertEqual(rebuilt.loc[90, "production_trigger_at"], str(frame.loc[90, "date"].date()))
        self.assertEqual(rebuilt.loc[89, "production_trigger_at"], "")

    def test_raw_acceptance_restarts_without_waiting_for_rolling_low_to_rise(self):
        frame = observations()
        frame["close"] = frame["open"] = frame["return_index"]
        frame["high"], frame["low"], frame["atr20"] = 101., 99., 2.
        frame.loc[60, "SOS"] = True
        frame.loc[62, ["return_index", "close", "open", "high", "low"]] = [85, 85, 85, 86, 84]
        frame.loc[65:66, ["return_index", "close", "open", "high", "low"]] = [103, 103, 102, 104, 102]
        frame.loc[65:66, "structure_floor"] = 84
        result = SL.replay("sh_600000", frame)
        self.assertEqual(result.loc[65, "breakout_event"], "ATTEMPT")
        self.assertEqual(result.loc[65, "cycle_stage"], "REBUILDING")
        self.assertEqual(result.loc[66, "breakout_event"], "ACCEPTED")
        self.assertEqual(result.loc[66, "cycle_stage"], "IGNITION")
        self.assertNotEqual(result.loc[60, "cycle_id"], result.loc[66, "cycle_id"])
        pd.testing.assert_frame_equal(SL.replay("sh_600000", frame.iloc[:67]), result.iloc[:67])

    def test_risk_warning_requires_reclaim_not_one_quiet_day(self):
        frame = observations()
        frame.loc[60, "SOS"] = True
        frame.loc[62, "broad_phase"] = "DISTRIBUTION"
        frame.loc[64, "broad_phase"] = "MARKUP"
        result = SL.replay("sh_600000", frame)
        self.assertEqual(result.loc[64, "cycle_stage"], "DISTRIBUTION_RISK")
        frame.loc[65, ["return_index", "effort_result", "broad_phase"]] = [115, "DEMAND_EXPANSION", "MARKUP"]
        self.assertEqual(SL.replay("sh_600000", frame).loc[65, "cycle_stage"], "MARKUP")

    def test_prefix_stability_and_repeatability(self):
        frame = observations()
        frame.loc[65, "SOS"] = True
        frame.loc[90, "return_index"] = 80
        full = SL.replay("sh_600000", frame)
        pd.testing.assert_frame_equal(SL.replay("sh_600000", frame.iloc[:80]), full.iloc[:80])
        pd.testing.assert_frame_equal(full, SL.replay("sh_600000", frame))

    def test_relative_turnover_is_stock_specific(self):
        small, large = observations(), observations()
        large["turnover"] *= 10
        small.loc[99, "turnover"], large.loc[99, "turnover"] = 4, 40
        a, b = SL.turnover_frame(small), SL.turnover_frame(large)
        self.assertEqual(a.iloc[-1]["turnover_relative20"], 2)
        self.assertEqual(b.iloc[-1]["turnover_relative20"], 2)
        self.assertEqual(a.iloc[-1]["turnover_percentile120"], b.iloc[-1]["turnover_percentile120"])

    def test_missing_turnover_is_not_supply_exhaustion(self):
        for value in (np.nan, 0, -1):
            frame = observations()
            frame.loc[99, "turnover"] = value
            row = SL.replay("sh_600000", frame).iloc[-1]
            self.assertEqual(row["turnover_context"], "MISSING")
        row = SL.replay("sh_600000", observations().drop(columns="turnover")).iloc[-1]
        self.assertEqual(row["turnover_context"], "MISSING")

    def test_drying_after_launch_differs_from_quiet_unseen_stock(self):
        frame = observations()
        frame.loc[95:, "turnover"] = 0.6
        row = SL.replay("sh_600000", frame).iloc[-1]
        self.assertEqual(row["turnover_context"], "QUIET_UNCONFIRMED")
        frame.loc[90, "SOS"] = True
        self.assertEqual(SL.replay("sh_600000", frame).iloc[-1]["turnover_context"], "POST_START_DRYING")

    def test_production_hit_is_distinct_from_price_reconstruction(self):
        frame = observations()
        frame.loc[60, "SOS"] = True
        trace = pd.DataFrame([{
            "symbol": "sh_600000", "date": str(frame.loc[65, "date"].date()),
            "state": "TRIGGERED", "data_fresh": True,
        }])
        row = SL.replay("sh_600000", frame, trace).iloc[-1]
        self.assertEqual(row["cycle_start_source"], "PRICE_RECONSTRUCTION")
        self.assertEqual(row["production_trigger_at"], str(frame.loc[65, "date"].date()))

    def test_daily_tracks_removed_stock_and_ignores_future_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            data.mkdir()
            raw = PhaseEventTests.price_frame()
            raw["turnover"] = 2.0
            raw.to_csv(data / "sh_600000_daily.csv", index=False)
            day1, day2 = [str(raw.loc[i, "date"].date()) for i in (170, 171)]
            output1, output2 = [root / day.replace("-", "") for day in (day1, day2)]
            candidates = pd.DataFrame([{"symbol": "sh_600000"}])
            SL.write_daily(day1, candidates, output1, root, data)
            SL.write_daily(day2, pd.DataFrame(), output2, root, data)
            tracked = pd.read_csv(output2 / "stock_lifecycle.csv")
            self.assertEqual(tracked["symbol"].tolist(), ["sh_600000"])
            self.assertFalse(tracked.iloc[0]["in_current_pool"])
            before = (output1 / "stock_lifecycle.csv").read_bytes()
            SL.write_daily(day1, candidates, output1, root, data)
            self.assertEqual(before, (output1 / "stock_lifecycle.csv").read_bytes())
            SL.write_daily(day1, pd.DataFrame(), output1, root, data)
            same_day = pd.read_csv(output1 / "stock_lifecycle.csv")
            self.assertEqual(same_day["symbol"].tolist(), ["sh_600000"])
            self.assertFalse(same_day.iloc[0]["in_current_pool"])

    def test_asof_filters_future_and_reports_staleness(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = PhaseEventTests.price_frame()
            raw.to_csv(root / "sh_600000_daily.csv", index=False)
            day = str(raw.loc[150, "date"].date())
            snapshot, timeline = SL.analyze("sh_600000", day, root)
            self.assertEqual(timeline.iloc[-1]["date"], day)
            self.assertEqual(snapshot["cycle_data_status"], "FRESH")
            snapshot, _ = SL.analyze("sh_600000", "2026-01-01", root)
            self.assertEqual(snapshot["cycle_data_status"], "STALE")


if __name__ == "__main__":
    unittest.main()
