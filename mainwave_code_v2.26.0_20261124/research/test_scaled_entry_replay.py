import unittest

import pandas as pd

from research import route_followup as RF
from research.scaled_entry_replay import replay, summarize


class ScaledEntryTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=45), "open": 100.,
            "high": 101., "low": 99.5, "close": 100., "preclose": 100., "isST": 0,
            "index_open": 100., "index_high": 101., "index_low": 99.5, "index_close": 100.,
            "atr20": 2., "low10": 94., "breakout_invalid_index": 95.})

    def run_case(self, f, setting=("ATR", .5, 1.), path="LOW_FIRST", calendar=None, initial=.3):
        f = f.copy()
        for raw in ("open", "high", "low", "close"):
            f[raw] = f["index_" + raw]
        a = RF.arrays(f)
        calendar = tuple(a["day"]) if calendar is None else calendar
        trace = []
        baseline = RF.simulate(a, 10, "sh_600001", "stage36_20", trace=trace, calendar=calendar)
        return baseline, replay(a, 10, "sh_600001", baseline, trace, calendar, setting, path, initial)

    def test_no_pullback_keeps_70_percent_cash(self):
        old, new = self.run_case(self.frame(), ("ATR", 1., 1.5))
        self.assertEqual(new["status"], "CLOSED")
        self.assertEqual(new["invested"], .3)
        self.assertAlmostEqual(new["net"], .3 * old["net"])
        self.assertEqual(len(new["buys"]), 1)

    def test_one_fill_leaves_40_percent_cash(self):
        f = self.frame()
        f.loc[12, "index_low"] = 98.7
        _, r = self.run_case(f)
        self.assertAlmostEqual(r["invested"], .6)
        self.assertEqual([b["price_index"] for b in r["buys"]], [100., 99.])

    def test_two_fill_weights_and_cash_profit(self):
        f = self.frame()
        f.loc[12, "index_low"] = 97.5
        _, r = self.run_case(f)
        self.assertEqual([b["budget_fraction"] for b in r["buys"]], [.3, .3, .4])
        expected = (.3 + .3*100/99 + .4*100/98 - 1)*100 - .3
        self.assertAlmostEqual(r["net"], expected)

    def test_window_seventh_session_included_eighth_excluded(self):
        f = self.frame()
        f.loc[19, "index_low"] = 97.5
        _, r = self.run_case(f)
        self.assertEqual(len(r["buys"]), 1)
        f.loc[18, "index_low"] = 97.5
        _, r = self.run_case(f)
        self.assertEqual(len(r["buys"]), 3)

    def test_entry_day_dip_cannot_add(self):
        f = self.frame()
        f.loc[11, "index_low"] = 97.5
        _, r = self.run_case(f)
        self.assertEqual(len(r["buys"]), 1)

    def test_gap_down_above_stop_fills_at_better_open(self):
        f = self.frame()
        f.loc[12, ["index_open", "index_low"]] = [97.8, 97.7]
        _, r = self.run_case(f)
        self.assertEqual([b["price_index"] for b in r["buys"]], [100., 97.8, 97.8])

    def test_never_add_at_or_below_known_stop(self):
        f = self.frame()
        f.loc[12, "index_low"] = 95.
        _, r = self.run_case(f, ("ATR", 1., 2.))
        self.assertEqual(len(r["buys"]), 2)
        self.assertEqual(r["buys"][-1]["price_index"], 98.)
        f.loc[12, "index_open"] = 95.
        _, r = self.run_case(f)
        self.assertEqual(len(r["buys"]), 1)

    def test_new_lots_defer_stop_not_hindsight_cancel(self):
        f = self.frame()
        f.loc[12, ["index_open", "index_low"]] = [98.8, 95.]
        f.loc[13, ["index_open", "index_low"]] = [94., 93.]
        _, r = self.run_case(f)
        self.assertEqual(len(r["buys"]), 3)
        self.assertEqual(r["deferred"], 2)
        self.assertEqual(r["exit_date"], str(f.loc[13, "date"].date()))
        self.assertTrue(all(s["date"] > s["lot_entry_date"] for s in r["sells"]))

    def test_pending_new_lots_cannot_be_counted_closed_at_data_end(self):
        f = self.frame().iloc[:13].copy()
        f.loc[12, ["index_open", "index_low"]] = [98.8, 95.]
        old, r = self.run_case(f)
        self.assertEqual(old["status"], "CLOSED")
        self.assertEqual(r["status"], "CENSORED")
        self.assertNotIn("net", r)

    def test_pending_new_lots_wait_through_locked_limit_down(self):
        f = self.frame()
        f.loc[12, ["index_open", "index_low"]] = [98.8, 95.]
        f.loc[13, ["index_open", "index_high", "index_low", "index_close"]] = 90.
        f.loc[14, ["index_open", "index_high", "index_low", "index_close"]] = [89., 91., 88., 90.]
        _, r = self.run_case(f)
        self.assertEqual(r["status"], "CLOSED")
        self.assertEqual(r["exit_date"], str(f.loc[14, "date"].date()))
        self.assertFalse(any(s["date"] == str(f.loc[13, "date"].date()) for s in r["sells"]))

    def test_future_prices_do_not_change_earlier_fills(self):
        f = self.frame()
        f.loc[12, "index_low"] = 97.5
        f.loc[15, "index_high"] = 113.
        f.loc[20, ["index_open", "index_low"]] = [95., 94.]
        _, full = self.run_case(f)
        _, prefix = self.run_case(f.iloc[:15])
        cutoff = str(f.loc[14, "date"].date())
        for key in ("buys", "sells"):
            self.assertEqual(prefix[key], [fill for fill in full[key] if fill["date"] <= cutoff])

    def test_unknown_missing_session_cannot_be_used_for_adds(self):
        f = self.frame()
        calendar = tuple(f.date.dt.strftime("%Y-%m-%d"))
        f.loc[13, "index_low"] = 97.5
        f = f.drop(index=12).reset_index(drop=True)
        _, r = self.run_case(f, calendar=calendar)
        self.assertEqual(r["status"], "HOLDING_DATA_UNVERIFIED")
        self.assertEqual(len(r["buys"]), 1)
        self.assertNotIn("net", r)

    def test_high_first_partial_cancels_later_dip_buys(self):
        f = self.frame()
        f.loc[12, ["index_high", "index_low"]] = [113., 97.5]
        _, lo = self.run_case(f, path="LOW_FIRST")
        _, hi = self.run_case(f, path="HIGH_FIRST")
        self.assertEqual(len(lo["buys"]), 3)
        self.assertEqual(len(hi["buys"]), 1)
        self.assertEqual(lo["deferred"], 4)
        self.assertTrue(all(s["date"] > s["lot_entry_date"] for s in lo["sells"]))

    def test_allocation_replay_matches_original_full_size(self):
        f = self.frame()
        f.loc[12, "index_high"] = 113.
        f.loc[14, ["index_open", "index_low"]] = [95., 94.]
        for path in ("LOW_FIRST", "HIGH_FIRST"):
            old, r = self.run_case(f, setting=None, path=path, initial=1.)
            self.assertAlmostEqual(r["net"], old["net"])

    def test_suspension_consumes_exchange_calendar_window(self):
        f = self.frame()
        calendar = tuple(f.date.dt.strftime("%Y-%m-%d"))
        f.loc[19, "index_low"] = 97.5
        suspended = calendar[12]
        f = f.drop(index=12).reset_index(drop=True)
        f.attrs["suspended_days"] = [suspended]
        _, r = self.run_case(f, calendar=calendar)
        self.assertEqual(len(r["buys"]), 1)

    def test_summary_counts_reduced_loss_as_improvement_not_winner(self):
        f = pd.DataFrame({"paired": [True, True, True, False], "delta": [2., -1., 0., None],
            "baseline_net": [-4., 3., 0., None], "net": [-2., 2., 0., None],
            "delta_vs_only30": [-.8, 1.1, 0., None], "invested": [.3, .6, 1., .3],
            "buy_count": [1, 2, 3, 1], "deferred": [0, 0, 0, 0],
            "status": ["CLOSED", "CLOSED", "CLOSED", "CENSORED"]})
        s = summarize(f, 10000)
        self.assertEqual((s["total"],s["comparable"],s["improved"],s["worse"],s["unchanged"]),(4,3,1,1,1))
        self.assertEqual(s["net_improvement_yuan"], 100.)
        self.assertEqual(s["improvement_amount_yuan"], 200.)
        self.assertEqual(s["deterioration_amount_yuan"], 100.)
        self.assertEqual(s["reduced_losses"], 1)


if __name__ == "__main__":
    unittest.main()
