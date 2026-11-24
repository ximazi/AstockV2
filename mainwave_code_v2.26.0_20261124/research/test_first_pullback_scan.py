import unittest

import numpy as np
import pandas as pd

from research.first_pullback_scan import episodes, prepare, summarize


def fixture():
    n = 75
    close = np.full(n, 100.0)
    close[65:] = [102, 103.5, 103, 102.2, 102.5, 104, 104.2, 104.1, 104.3, 104.2]
    turn = np.full(n, 5.0)
    turn[65:71] = [20, 18, 8, 7, 6, 12]
    launch = np.zeros(n, dtype=bool)
    launch[65] = True
    frame = pd.DataFrame({
        "date": pd.bdate_range("2026-01-01", periods=n), "c": close,
        "h": close + 0.2, "l": close - 0.2, "close": close,
        "ema10": 102.0, "ema20": 101.5, "turn": turn,
        "launch": launch, "base_evidence_days": 8,
        "structure_floor": 98.0, "atr20": 1.0,
        "close_location": 0.8, "broad_phase": "MARKUP", "effort_result": "NEUTRAL",
        "ret5": 0.02, "ret20": 0.05,
    })
    return frame


class FirstPullbackTests(unittest.TestCase):
    def test_confirmation_requires_pivot_not_just_ema(self):
        f = fixture()
        self.assertEqual(episodes(f.iloc[:70])[-1]["status"], "FORMING")
        event = episodes(f.iloc[:71])[-1]
        self.assertEqual(event["status"], "CONFIRMED")
        self.assertEqual(event["confirmed_date"], str(f.iloc[70].date.date()))

    def test_supply_must_contract(self):
        f = fixture()
        f.loc[67:69, "turn"] = 30
        f.loc[70, "turn"] = 40
        self.assertNotEqual(episodes(f.iloc[:71])[-1]["status"], "CONFIRMED")

    def test_missing_turnover_does_not_confirm(self):
        f = fixture()
        f.loc[67, "turn"] = np.nan
        self.assertNotEqual(episodes(f.iloc[:71])[-1]["status"], "CONFIRMED")

    def test_price_recovery_without_volume_consumes_first_pullback(self):
        f = fixture()
        f.loc[70, "turn"] = 3
        f.loc[71, "turn"] = 30
        event = episodes(f)[-1]
        self.assertEqual(event["status"], "RECOVERED_WITHOUT_CONFIRMATION")
        self.assertEqual(event["confirmed_date"], "")

    def test_failed_first_is_not_relabelled(self):
        f = fixture()
        f.loc[68, ["c", "l"]] = [99, 98.8]
        f.loc[70, "launch"] = True
        events = episodes(f)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], "FAILED")

    def test_historical_confirmation_not_current(self):
        self.assertEqual(summarize(fixture(), episodes(fixture()))["scan_status"], "HISTORICAL_CONFIRMED")

    def test_post_confirmation_break_remains_flagged_after_recovery(self):
        f = fixture().iloc[:73].copy()
        f.loc[71, "c"] = 99
        result = summarize(f, episodes(f))
        self.assertIn("POST_CONFIRMATION_BROKEN", result["risk_flags"])

    def test_preparation_has_no_future_dependency(self):
        f = fixture()
        raw = pd.DataFrame({"date": f.date, "open": f.c - 0.1, "high": f.h,
                            "low": f.l, "close": f.c, "amount": 1e8,
                            "turnover": f.turn, "tradestatus": 1})
        as_of = str(f.date.iloc[70].date())
        expected = prepare(raw.iloc[:71], as_of)
        raw.loc[71:, "close"] *= 4
        pd.testing.assert_frame_equal(prepare(raw, as_of), expected)

    def test_confirmation_details_do_not_repaint(self):
        f = fixture()
        early = episodes(f.iloc[:71])[-1]
        later = episodes(f)[-1]
        self.assertEqual(early, later)


if __name__ == "__main__":
    unittest.main()
