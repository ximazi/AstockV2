import unittest
from unittest.mock import patch

import pandas as pd

from research import adaptive_exit_policy as P, route_followup as RF, scaled_entry_replay as S
from research import test_route_followup as fixtures


class AdaptiveExitTests(unittest.TestCase):
    def fixture(self):
        f = fixtures.FollowupTests().fixture()
        f["index_low"], f["low10"], f["low20"] = 99., 99., 94.
        f["exit_strong"], f["market_known"], f["upper_wicks5"] = True, True, 0.
        f["close"] = f.index_close
        f["low"] = f.index_low
        return f

    def simulate(self, f, policy="TREND"):
        a, trace = RF.arrays(f), []
        r = RF.simulate(a, 10, "sh_600001", "stage36_20", trace=trace,
                        calendar=tuple(a["day"]), exit_policy=policy)
        return r, trace

    def test_strong_trail_has_more_room_without_lowering_existing_stop(self):
        f = self.fixture()
        f.loc[12, "index_high"] = 105.
        f.loc[13, "index_low"] = 98.
        normal, _ = self.simulate(f, None)
        wide, logs = self.simulate(f)
        self.assertEqual(normal["reason"], "TRAIL_STOP")
        self.assertEqual(wide["reason"], "HORIZON")
        self.assertTrue(all(t["stop_next"] >= t["stop_before"] for t in logs))
        f.loc[12, "exit_strong"] = False
        r, logs = self.simulate(f)
        self.assertEqual(r["reason"], "TRAIL_STOP")

    def test_wider_initial_only_uses_signal_day_strength(self):
        f = self.fixture()
        f.loc[11, "index_low"] = 95.
        narrow, _ = self.simulate(f, "RISK")
        wide, _ = self.simulate(f, "WIDE_RISK")
        self.assertEqual(narrow["reason"], "DEFERRED_STOP")
        self.assertEqual(wide["reason"], "HORIZON")
        f.loc[10, "exit_strong"] = False
        r, _ = self.simulate(f, "WIDE_RISK")
        self.assertEqual(r["reason"], "DEFERRED_STOP")

    def test_close_warning_executes_next_open_partial_then_keeps_core(self):
        f = self.fixture()
        def warning(a, j, *args):
            return ("REDUCE_DISTRIBUTION_PROXY", {}) if j == 16 else ("", {})
        with patch.object(P, "warning", side_effect=warning):
            r, logs = self.simulate(f, "RISK")
        self.assertFalse(logs[5]["fills"])
        partial = next((t, s) for t in logs for s in t["fills"] if s["reason"] == "RISK_REDUCTION")
        self.assertEqual(partial[0]["date"], str(f.date.iloc[17].date()))
        self.assertEqual(partial[1]["weight"], .5)
        a = RF.arrays(f)
        half = S.replay(a, 10, "sh_600001", r, logs, tuple(a["day"]), initial=.5)
        self.assertAlmostEqual(half["net"], .5 * r["net"])
        self.assertEqual(half["exit_date"], r["exit_date"])

    def test_next_open_exit_preempts_later_intraday_target(self):
        f = self.fixture()
        f.loc[17, "index_high"] = 120.
        with patch.object(P, "warning", side_effect=lambda a, j, *args: ("EXIT_STRUCTURE", {}) if j == 16 else ("", {})):
            r, logs = self.simulate(f, "RISK")
        self.assertEqual(r["partials"], 0)
        self.assertEqual(logs[-1]["fills"][0]["execution"], "NEXT_OPEN")

    def test_partial_before_intraday_stop_matches_lot_accounting(self):
        f = self.fixture()
        f.loc[17, "index_low"] = 95.
        with patch.object(P, "warning", side_effect=lambda a, j, *args: ("REDUCE_DISTRIBUTION_PROXY", {}) if j == 16 else ("", {})):
            r, logs = self.simulate(f, "RISK")
        self.assertEqual([s["reason"] for s in logs[-1]["fills"]], ["RISK_REDUCTION", "INITIAL_STOP"])
        a = RF.arrays(f)
        half = S.replay(a, 10, "sh_600001", r, logs, tuple(a["day"]), initial=.5)
        self.assertAlmostEqual(half["net"], .5 * r["net"])

    def test_warning_needs_maturity_not_single_early_wick(self):
        f = self.fixture()
        f.loc[12, ["index_high", "index_close", "close_location", "amount_ratio"]] = [108., 101., .2, 2.]
        a = RF.arrays(f)
        self.assertEqual(P.warning(a, 12, 11, 100., 2., 108.)[0], "")
        f.loc[18, ["index_high", "index_close", "close_location", "amount_ratio"]] = [108., 101., .2, 2.]
        self.assertEqual(P.warning(RF.arrays(f), 18, 11, 100., 2., 108.)[0], "REDUCE_DISTRIBUTION_PROXY")

    def test_soft_close_does_not_sell_intraday_touch(self):
        f = self.fixture()
        f["exit_strong"] = False
        f.loc[12, "index_high"] = 105.
        f.loc[13, "index_low"] = 98.
        with patch.object(P, "warning", return_value=("", {})):
            r, logs = self.simulate(f, "CLOSE_RISK")
        self.assertEqual(r["reason"], "HORIZON")
        self.assertTrue(all(t["stop_next"] == 96. for t in logs))
        self.assertGreater(logs[-2]["soft_stop_next"], 96.)

    def test_soft_breach_uses_previous_strength_and_two_closes(self):
        f = self.fixture()
        f["exit_strong"] = False
        f.loc[12, "index_high"] = 105.
        f.loc[13, "exit_strong"] = True
        f.loc[14:15, ["index_low", "index_close"]] = [97., 98.]
        with patch.object(P, "warning", return_value=("", {})):
            r, logs = self.simulate(f, "CLOSE_RISK")
        self.assertEqual(r["exit_date"], str(f.date.iloc[16].date()))
        self.assertEqual(r["reason"], "SOFT_CLOSE_EXIT")
        self.assertEqual(logs[3]["soft_required"], 2)

    def test_soft_policy_never_ignores_hard_stop(self):
        f = self.fixture()
        f.loc[12, "index_low"] = 90.
        r, _ = self.simulate(f, "CLOSE_RISK")
        self.assertEqual(r["reason"], "INITIAL_STOP")

    def test_partial_warning_waits_through_locked_limit(self):
        f = self.fixture()
        f.loc[17, "high"] = 90.
        with patch.object(P, "warning", side_effect=lambda a, j, *args: ("REDUCE_DISTRIBUTION_PROXY", {}) if j == 16 else ("", {})):
            _, logs = self.simulate(f, "RISK")
        reductions = [t["date"] for t in logs for s in t["fills"] if s["reason"] == "RISK_REDUCTION"]
        self.assertEqual(reductions, [str(f.date.iloc[18].date())])

    def test_market_features_are_causal_and_missing_is_not_strong(self):
        raw = pd.DataFrame({"trade_date": pd.bdate_range("2024-01-01", periods=110), "close": range(100, 210)})
        full = P.market_frame(raw, "2025-01-01")
        prefix = P.market_frame(raw.iloc[:85], "2025-01-01")
        pd.testing.assert_frame_equal(full.iloc[:85], prefix)
        f = self.fixture()
        enriched = P.enrich(f, full.iloc[:80])
        self.assertFalse(enriched.exit_strong.iloc[-1])
        self.assertFalse(enriched.market_known.iloc[-1])

    def test_extension_uses_day20_close_not_entry_regime(self):
        f = self.fixture()
        f.loc[12:, ["index_high", "index_close", "index_low"]] = [105., 104.5, 100.]
        f.loc[10, "exit_strong"] = False
        a = RF.arrays(f)
        trace = []
        with patch.object(P, "warning", return_value=("", {})):
            r = RF.simulate(a, 10, "sh_600001", "stage36_60", trace=trace, calendar=tuple(a["day"]), exit_policy="EXTEND_CLOSE")
        review = next(t["extension_review"] for t in trace if "extension_review" in t)
        self.assertTrue(review["extend"])
        self.assertEqual(r["held"], 60)
        f.loc[30, "exit_strong"] = False
        a = RF.arrays(f)
        with patch.object(P, "warning", return_value=("", {})):
            r = RF.simulate(a, 10, "sh_600001", "stage36_60", calendar=tuple(a["day"]), exit_policy="EXTEND_CLOSE")
        self.assertEqual(r["held"], 21)
        self.assertEqual(r["reason"], "TIME_REVIEW_EXIT")


if __name__ == "__main__":
    unittest.main()
