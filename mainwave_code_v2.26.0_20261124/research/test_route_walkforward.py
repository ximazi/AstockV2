import unittest

import pandas as pd

from mainwave_engine import route_evidence as RE
from mainwave_engine.test_phase_events import PhaseEventTests
from research.route_walkforward import paper_path, training_sample


class RouteResearchTests(unittest.TestCase):
    def test_training_excludes_future_outcomes_not_only_future_signals(self):
        frame = pd.DataFrame({"year": [2024, 2024, 2025],
                              "outcome20_at": ["2024-12-30", "2025-01-06", "2025-02-01"]})
        self.assertEqual(training_sample(frame).index.tolist(), [0])

    def test_evidence_is_prefix_stable(self):
        raw = PhaseEventTests.price_frame()
        raw["isST"], raw["turnover"] = 0, 2.
        full = RE.build(raw, "sh_600001")
        for end in (130, 160, 175):
            pd.testing.assert_frame_equal(RE.build(raw.iloc[:end], "sh_600001"), full.iloc[:end])

    def test_missing_turnover_does_not_prove_preparation(self):
        raw = PhaseEventTests.price_frame()
        raw["isST"] = 0
        self.assertFalse(RE.build(raw, "sh_600001").prepared.any())

    def paper_fixture(self):
        frame = pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=30),
                              "open": 100., "high": 101., "preclose": 100., "isST": 0,
                              "index_open": 100., "index_high": 101., "index_low": 99.,
                              "index_close": 100., "atr20": 2.})
        return frame

    def test_no_same_day_sale_after_entry_and_stop_first(self):
        frame = self.paper_fixture()
        frame.loc[1, "index_low"] = 90
        frame.loc[2, "index_open"] = 92
        trade = paper_path(frame, 0, "sh_600001", "staged")
        self.assertAlmostEqual(trade["net_pct"], -8.3)
        self.assertEqual(trade["exit_reason"], "DEFERRED_EXIT")

    def test_limit_up_entry_is_not_assumed_filled(self):
        frame = self.paper_fixture()
        frame.loc[1, "open"] = 110
        self.assertEqual(paper_path(frame, 0, "sh_600001", "hold")["status"], "ENTRY_LIMIT_UP")

    def test_partial_scale_out_captures_spike_before_reversal(self):
        frame = self.paper_fixture()
        frame.loc[2, "index_high"] = 113
        frame.loc[3, ["index_low", "index_open"]] = [95, 98]
        self.assertGreater(paper_path(frame, 0, "sh_600001", "staged")["net_pct"],
                           paper_path(frame, 0, "sh_600001", "trend")["net_pct"])


if __name__ == "__main__":
    unittest.main()
