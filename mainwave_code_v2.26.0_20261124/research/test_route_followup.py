import unittest
from unittest.mock import patch

import pandas as pd

from mainwave_engine import route_evidence as RE
from mainwave_engine import test_phase_events as fixtures
from research.route_followup import arrays, enrich, entry_signal, preparation_outcome, simulate


class FollowupTests(unittest.TestCase):
    def setUp(self):
        calendar = tuple(pd.bdate_range("2024-01-01", periods=110).strftime("%Y-%m-%d"))
        mock = patch("research.route_clock.sessions", return_value=calendar)
        mock.start()
        self.addCleanup(mock.stop)

    def fixture(self):
        f = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=110),
                          "open": 100., "high": 102., "preclose": 100., "isST": 0,
                          "index_open": 100., "index_high": 102., "index_low": 95.,
                          "index_close": 100., "atr20": 2., "low10": 94.,
                          "breakout_id": "one", "breakout_status": "ACCEPTED", "breakout_event": "",
                          "breakout_boundary_index": 99., "breakout_invalid_index": 98.,
                          "structure_floor": 94., "eligible": True,
                          "supply_warning": False, "ma20": 99., "amount_ratio": .8, "close_location": .8})
        return f

    def test_activation_does_not_tighten_before_advance(self):
        f = self.fixture()
        f["index_low"] = 99.
        f.loc[14, "index_low"] = 98.
        a = arrays(f)
        self.assertEqual(simulate(a, 10, "sh_600001", "legacy20")["reason"], "TRAIL_STOP")
        self.assertEqual(simulate(a, 10, "sh_600001", "activated20")["reason"], "HORIZON")

    def test_tplus1_and_stop_before_target(self):
        f = self.fixture()
        f.loc[12, "index_open"] = 92
        r = simulate(arrays(f), 10, "sh_600001", "stage36_20")
        self.assertEqual(r["reason"], "DEFERRED_STOP")
        self.assertAlmostEqual(r["net"], -8.3)
        f["index_low"] = 99.
        f.loc[12, ["index_low", "index_high", "index_open"]] = [90., 120., 100.]
        r = simulate(arrays(f), 10, "sh_600001", "stage36_20")
        self.assertEqual(r["partials"], 0)
        self.assertAlmostEqual(r["net"], -4.3)

    def test_partial_ladder_preserves_spike(self):
        f = self.fixture()
        f["index_low"] = 99.
        f.loc[12, "index_high"] = 113.
        f.loc[13, ["index_low", "index_open"]] = [90., 93.]
        a = arrays(f)
        self.assertGreater(simulate(a, 10, "sh_600001", "stage36_20")["net"],
                           simulate(a, 10, "sh_600001", "activated20")["net"])

    def test_limit_entry_and_unresolved_exit(self):
        f = self.fixture()
        f.loc[11, "open"] = 110
        self.assertEqual(simulate(arrays(f), 10, "sh_600001", "hold20")["status"], "ENTRY_LIMIT_UP")
        f.loc[11, "open"] = 100
        f.loc[30, "high"] = 90
        self.assertEqual(simulate(arrays(f), 10, "sh_600001", "hold20")["status"], "EXIT_UNRESOLVED")

    def test_structure_stop_and_invalid_gap_entry(self):
        f = self.fixture()
        f["index_low"] = 99.
        f.loc[12, "index_low"] = 97.
        r = simulate(arrays(f), 10, "sh_600001", "structure36_20")
        self.assertAlmostEqual(r["net"], -2.3)
        f.loc[11, "index_open"] = 97.
        self.assertEqual(simulate(arrays(f), 10, "sh_600001", "structure36_20")["status"], "ENTRY_BELOW_STRUCTURE")

    def test_wait_does_not_cross_failed_breakout_id(self):
        f = self.fixture()
        f.loc[12:, "breakout_id"] = "new"
        f.loc[13:, "breakout_status"] = "ADVANCING"
        self.assertEqual(entry_signal(arrays(f), 10, "advance"), (None, "INVALID_BEFORE_ENTRY"))

    def test_pullback_needs_prior_advance_and_prior_contraction(self):
        f = self.fixture()
        f["index_low"] = 99.
        f.loc[11, ["index_close", "index_high"]] = [104., 105.]
        f.loc[12, ["index_close", "index_high", "index_low"]] = [101., 102., 100.]
        f.loc[13, ["index_close", "index_high", "amount_ratio"]] = [103., 104., 1.2]
        self.assertEqual(entry_signal(arrays(f), 10, "pullback")[0], 13)
        self.assertIsNone(entry_signal(arrays(f.iloc[:13]), 10, "pullback")[0])

    def test_preparation_funnel_excludes_later_unrelated_breakout(self):
        f = self.fixture()
        f.loc[11, "breakout_event"] = "ATTEMPT"
        f.loc[12, ["breakout_status", "breakout_event"]] = ["REJECTED", "REJECTED"]
        f.loc[13:, "breakout_id"] = "new"
        f.loc[14, "breakout_event"] = "ACCEPTED"
        r = preparation_outcome(arrays(f), 10)
        self.assertTrue(r["attempt"])
        self.assertFalse(r["accepted"])

    def test_features_are_prefix_stable(self):
        raw = fixtures.PhaseEventTests.price_frame()
        raw["isST"], raw["turnover"] = 0, 2.
        full = enrich(RE.build(raw, "sh_600001"))
        pd.testing.assert_frame_equal(enrich(RE.build(raw.iloc[:160], "sh_600001")), full.iloc[:160])


if __name__ == "__main__":
    unittest.main()
