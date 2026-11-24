import unittest

import numpy as np
import pandas as pd

from research.theme_breakout_scan import snapshot, theme_context


def fixture():
    n = 80
    return pd.DataFrame({
        "date": pd.bdate_range("2026-05-18", periods=n),
        "c": np.full(n, 100.0), "h": 100.2, "l": 99.8, "close": 100.0,
        "atr20": 1.0, "amount_ratio": 1.5, "close_location": 0.8,
        "ema10": 100.0, "ema20": 100.0, "ret5": 0.0, "ret20": 0.0,
        "turn": 2.0, "amount": 1e8, "broad_phase": "MARKUP", "effort_result": "NEUTRAL",
    })


class BreakoutTests(unittest.TestCase):
    def test_current_breakout(self):
        f = fixture()
        f.loc[79, ["c", "close", "h"]] = [102, 102, 102.1]
        result = snapshot(f)
        self.assertEqual(result["route"], "BREAKOUT_TODAY")
        self.assertTrue(result["above_high60"])
        self.assertAlmostEqual(result["breakout_level"], 100.2)

    def test_lost_breakout_not_counted_as_retained(self):
        f = fixture()
        f.loc[77, ["c", "close", "h"]] = [103, 103, 103.1]
        f.loc[79, ["c", "close"]] = [99, 99]
        result = snapshot(f)
        self.assertEqual(result["route"], "BREAKOUT_LOST")
        self.assertFalse(result["breakout_retained"])

    def test_recent_breakout_still_held(self):
        f = fixture()
        f.loc[77, ["c", "close", "h"]] = [103, 103, 103.1]
        f.loc[78:, ["c", "close", "h"]] = [102, 102, 102.1]
        result = snapshot(f)
        self.assertEqual(result["route"], "RECENT_BREAKOUT")
        self.assertTrue(result["breakout_retained"])
        self.assertNotIn("BREAK_LEVEL_LOST_AFTER_EVENT", result["risk_flags"])

    def test_near_breakout_is_not_confirmed(self):
        f = fixture()
        f.loc[78:, ["c", "close"]] = [100.1, 100.1]
        result = snapshot(f)
        self.assertEqual(result["route"], "NEAR_BREAKOUT")
        self.assertFalse(result["breakout_retained"])

    def test_ema_reclaim_below_range_high_is_separate(self):
        f = fixture()
        f.loc[70, "h"] = 105
        f.loc[79, ["c", "close", "h"]] = [101, 101, 101.1]
        result = snapshot(f)
        self.assertEqual(result["route"], "EMA_RECLAIM_ONLY")
        self.assertFalse(result["breakout_retained"])

    def test_price_without_volume_is_separate(self):
        f = fixture()
        f.loc[79, ["c", "close", "h", "amount_ratio"]] = [102, 102, 102.1, 0.5]
        self.assertEqual(snapshot(f)["route"], "PRICE_ONLY_BREAKOUT")

    def test_extended_signal_kept_but_flagged(self):
        f = fixture()
        f.loc[79, ["c", "close", "h"]] = [105, 105, 105.1]
        self.assertIn("EXTENDED_ABOVE_EMA20", snapshot(f)["risk_flags"])

    def test_break_reference_excludes_today_high(self):
        f = fixture()
        f.loc[79, ["c", "close", "h"]] = [102, 102, 108]
        self.assertAlmostEqual(snapshot(f)["breakout_level"], 100.2)

    def test_generic_membership_not_a_hot_theme(self):
        result = theme_context({"concepts": ["融资融券"]}, [], {},
                               {"融资融券": {"heat_score": 100}})
        self.assertEqual(result["hot_themes"], "")
        self.assertIn("NEEDS_BUSINESS_REVIEW", result["theme_evidence"])

    def test_weekly_ignition_survives_low_daily_heat(self):
        result = theme_context({"concepts": ["liquid_cooling"]}, [], {}, {},
                               {"liquid_cooling": "2026-08-31:IGNITED:HELD"})
        self.assertEqual(result["propagation_themes"], "liquid_cooling")
        self.assertEqual(result["hot_themes"], "")


if __name__ == "__main__":
    unittest.main()
