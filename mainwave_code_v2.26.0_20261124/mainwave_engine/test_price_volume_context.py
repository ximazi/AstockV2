import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from mainwave_engine import price_volume_context as PV, route_evidence as RE


def frame(n=100):
    dates = pd.bdate_range("2025-01-01", periods=n).strftime("%Y-%m-%d")
    return pd.DataFrame({"date": dates, "open": 100., "close": 100., "high": 100.2, "low": 99.8,
        "return_index": 100., "atr20": 1., "volume": 1e6, "turnover": 1., "amount": 1e8,
        "structure_floor": 90., "structure_ceiling": 100., "breakout_id": "test:breakout:A",
        "breakout_attempt_at": dates[20], "breakout_status": "ACCEPTED",
        "breakout_boundary_index": 100., "breakout_invalid_index": 90.})


def bar(f, i, close=100., activity=1., op=None, high=None, low=None):
    op = float(f.iloc[i-1].close) if op is None else op
    f.loc[i, ["close", "return_index", "open", "high", "low", "turnover", "volume", "amount"]] = [
        close, close, op, max(op, close) + .05 if high is None else high,
        min(op, close) - .05 if low is None else low, activity, activity * 1e6, activity * 1e8]


class PriceVolumeTests(unittest.TestCase):
    def test_sequence_is_causal_and_does_not_mutate_production_columns(self):
        f = frame(90)
        bar(f, 75, activity=1.5)
        bar(f, 77, activity=1.5)
        bar(f, 78, close=100.5, activity=.9)
        original = f.copy(deep=True)
        full = PV.annotate(f, "test")
        self.assertEqual(full.iloc[78].event, "CONTEST_TO_UP")
        self.assertTrue(full.iloc[78].alternative_demand_evidence)
        for end in (76, 78, 79, 81):
            assert_frame_equal(PV.annotate(f.iloc[:end], "test"), full.iloc[:end].reset_index(drop=True))
        assert_frame_equal(original, f)
        self.assertFalse(full.orders_enabled.any())

    def test_three_day_recovery_does_not_clear_any_production_risk(self):
        f = frame(85)
        bar(f, 75, activity=1.5)
        bar(f, 76, activity=1.5)
        bar(f, 77, close=100.5, activity=.9)
        bar(f, 78, close=100.5, activity=2., high=100.8, low=100.4)
        bar(f, 79, close=101., activity=.9)
        bar(f, 80, close=101.5, activity=.9)
        f["supply_warning"] = True
        r = PV.annotate(f, "test")
        self.assertEqual(r.iloc[78].event, "UP_TO_CONTEST")
        self.assertNotEqual(r.iloc[79].event, "UP_RECONFIRMED")
        self.assertEqual(r.iloc[80].event, "UP_RECONFIRMED")
        self.assertTrue(f.supply_warning.all())

    def test_low_volume_direction_weak_bounce_and_absorption_are_distinct(self):
        f = frame(83)
        bar(f, 75, close=100.5, activity=.9)
        bar(f, 76, close=100.6, activity=.9)
        bar(f, 77, close=100.1, activity=.9)
        bar(f, 78, close=100.1, activity=1.5, high=100.2, low=99.5)
        r = PV.annotate(f, "test")
        self.assertEqual(r.iloc[75].state, "EFFICIENT_UP")
        self.assertEqual(r.iloc[76].state, "UNRESOLVED")
        self.assertEqual(r.iloc[77].state, "EFFICIENT_DOWN")
        self.assertEqual(r.iloc[78].state, "CONTESTED")

    def test_locked_gap_illiquidity_missing_data_and_float_change(self):
        for kind in ("locked", "gap", "illiquid", "missing", "float", "zero_atr", "zero_amount"):
            with self.subTest(kind=kind):
                f = frame(80)
                bar(f, 79, close=100.5, activity=.9)
                if kind == "locked":
                    f.loc[79, ["open", "high", "low"]] = 100.5
                elif kind == "gap":
                    bar(f, 79, close=103., op=102., activity=.9)
                elif kind == "illiquid":
                    f["amount"] = 1e6
                elif kind == "missing":
                    f[["volume", "turnover"]] = np.nan
                elif kind == "float":
                    f.loc[79, "turnover"] = .1
                elif kind == "zero_atr":
                    f.loc[79, "atr20"] = 0
                else:
                    f.loc[79, "amount"] = 0
                result = PV.report(f, "test")
                self.assertIn(result["latest"]["state"], {"UNRESOLVED", "LOW_LIQUIDITY_MOVE"})
                self.assertFalse(result["latest"]["alternative_demand_evidence"])
                json.dumps(result, allow_nan=False)

    def test_volume_fallback_explicitly_marks_missing_turnover(self):
        f = frame(80)
        bar(f, 79, close=100.5, activity=.9)
        f["turnover"] = np.nan
        r = PV.report(f, "test")["latest"]
        self.assertEqual(r["activity_source"], "VOLUME")
        self.assertIn("TURNOVER_MISSING", r["missing"])
        self.assertIsNone(r["relative_turnover"])

    def test_new_breakout_cannot_reuse_old_contests(self):
        f = frame(81)
        bar(f, 75, activity=1.5)
        bar(f, 77, activity=1.5)
        f.loc[78:, "breakout_id"] = "test:breakout:B"
        bar(f, 78, close=100.5, activity=.9)
        bar(f, 79, close=101., activity=.9)
        r = PV.annotate(f, "test")
        self.assertEqual(r.iloc[78].invalid_reason, "NEW_BREAKOUT")
        self.assertFalse(r.iloc[79].alternative_demand_evidence)
        self.assertNotEqual(r.iloc[77].episode_id, r.iloc[79].episode_id)

    def test_fixed_ttl_and_structure_failure(self):
        f = frame(100)
        bar(f, 30, activity=1.5)
        bar(f, 60, activity=1.5)
        r = PV.annotate(f, "test")
        self.assertEqual(r.iloc[60].first_seen, f.iloc[30].date)
        self.assertEqual(r.iloc[61].invalid_reason, "TTL_EXPIRED")
        f.loc[40, "breakout_status"] = "FAILED"
        r = PV.annotate(f, "test")
        self.assertEqual(r.iloc[40].invalid_reason, "STRUCTURE_FAILED")
        self.assertFalse(r.iloc[40].episode_valid)

    def test_frozen_raw_attachment_and_live_calculation_match(self):
        f = frame(90)
        raw = f[["date", "open", "high", "low", "close", "volume", "amount", "turnover"]]
        raw.loc[80, ["open", "high", "low", "close"]] = [100., 101., 99.9, 100.9]
        case = {"summary": {"symbol": "sh_600001", "window_end": raw.iloc[-1].date},
                "case_hash": "unchanged", "raw_history": raw.to_dict("records")}
        original = copy.deepcopy(case)
        expected = PV.report(RE.build(raw, "sh_600001"), "sh_600001")
        actual = PV.case_attachment(case)
        self.assertEqual(actual["timeline"], expected["timeline"])
        self.assertEqual(original, case)
        self.assertEqual(actual["source_case_hash"], "unchanged")

    def test_incomplete_frozen_case_does_not_fetch_live_prices(self):
        case = {"summary": {"symbol": "sh_600001", "window_end": "2026-01-01"}, "case_hash": "old", "raw_history": []}
        with patch.object(pd, "read_csv", side_effect=AssertionError("No external reads")):
            self.assertEqual(PV.case_attachment(case)["timeline"], [])

    def test_shadow_states_cannot_change_buy_sell_or_market_contracts(self):
        from mainwave_engine.test_analysis_core import fixture
        from mainwave_engine import analysis_core as AC
        for market_state in ("GREEN", "ORANGE", "RED"):
            data, market = fixture()
            market["state"] = market_state
            before = AC.evaluate(data, market)
            for state in PV.LABELS:
                data["price_volume_context"] = {"mode": "SHADOW_ONLY", "latest": {
                    "state": state, "alternative_demand_evidence": True, "event": "UP_RECONFIRMED"}}
                self.assertEqual(AC.evaluate(data, market), before)


if __name__ == "__main__":
    unittest.main()
