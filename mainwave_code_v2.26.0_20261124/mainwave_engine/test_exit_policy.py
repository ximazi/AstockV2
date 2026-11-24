import unittest
from unittest.mock import patch

import pandas as pd

from mainwave_engine import entry_policy as EP, exit_policy as XP
from research import route_followup as RF


class ExitPolicyTests(unittest.TestCase):
    def fixture(self):
        f = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=140), "accepted_bias2": False,
            "index_open": 100., "index_high": 101., "index_low": 99., "index_close": 100.,
            "open": 100., "high": 101., "close": 100., "preclose": 100., "isST": 0, "atr20": 2.,
            "breakout_invalid_index": 95., "breakout_id": "one", "breakout_status": "ACCEPTED",
            "eligible": True, "supply_warning": False, "close_location": .7, "amount": 100., "amount_ratio": .8,
            "exit_strong": False, "market_known": True, "upper_wicks5": 0., "ma20": 100., "low10": 99., "low20": 97.})
        f.loc[70, "accepted_bias2"] = True
        return f, list(f.date.dt.strftime("%Y-%m-%d"))

    def observe(self, f, calendar, end, anchor=None):
        view = f.iloc[:end].copy()
        anchor = EP.reference(view, calendar, "sh_600001") if anchor is None else anchor
        with patch.object(XP, "enrich", return_value=view):
            return XP.observe(view, calendar, "sh_600001", anchor, pd.DataFrame())

    def test_intraday_soft_touch_does_not_exit(self):
        f, calendar = self.fixture()
        f.loc[72, "index_high"] = 105.
        f.loc[73, "index_low"] = 98.
        r = self.observe(f, calendar, 74)
        self.assertEqual(r["status"], "HOLD_REFERENCE")
        self.assertEqual(r["hard_floor"], 96.)
        self.assertEqual(r["soft_floor"], 99.)
        self.assertFalse(r["cancel_adds"])

    def test_strong_confirmation_is_frozen_and_next_open_only(self):
        f, calendar = self.fixture()
        f.loc[72, "index_high"] = 105.
        f.loc[73, "exit_strong"] = True
        f.loc[74:75, ["index_low", "index_close"]] = [97., 98.]
        one = self.observe(f, calendar, 75)
        two = self.observe(f, calendar, 76)
        self.assertEqual(one["status"], "SOFT_CONFIRMING")
        self.assertEqual(two["status"], "EXIT_NEXT_OPEN")
        self.assertEqual(two["soft_required"], 2)
        self.assertFalse(two["events"])
        closed = self.observe(f, calendar, 77)
        self.assertEqual(closed["closed_at"], calendar[76])
        self.assertEqual(closed["reason"], "SOFT_CLOSE_EXIT")
        self.assertTrue(closed["cancel_adds"])

    def test_hard_floor_tplus1_and_locked_day(self):
        f, calendar = self.fixture()
        f.loc[71, "index_low"] = 95.
        f.loc[72, ["high", "index_high", "index_low"]] = [90., 90., 89.]
        self.assertEqual(self.observe(f, calendar, 72)["status"], "HARD_EXIT_PENDING")
        self.assertEqual(self.observe(f, calendar, 73)["status"], "HARD_EXIT_PENDING")
        r = self.observe(f, calendar, 74)
        self.assertEqual(r["reason"], "DEFERRED_STOP")
        self.assertEqual(r["closed_at"], calendar[73])
        self.assertEqual(r["hard_floor"], 96.)

    def test_mature_warning_reduces_once_and_waits_for_next_open(self):
        f, calendar = self.fixture()
        f.loc[72, "index_high"] = 107.
        for day in (77, 79):
            f.loc[day, ["index_open", "index_high", "index_low", "index_close", "amount_ratio", "close_location"]] = [106., 108., 105., 106., 2., .2]
        pending = self.observe(f, calendar, 78)
        self.assertEqual(pending["status"], "REDUCE_NEXT_OPEN")
        self.assertFalse(any(e["reason"] == "RISK_REDUCTION" for e in pending["events"]))
        r = self.observe(f, calendar, 81)
        reductions = [e for e in r["events"] if e["reason"] == "RISK_REDUCTION"]
        self.assertEqual(len(reductions), 1)
        self.assertEqual(reductions[0]["date"], calendar[78])
        self.assertEqual(reductions[0]["reference_fraction"], .25)

    def test_day20_current_strength_profit_and_alert_gate(self):
        f, calendar = self.fixture()
        f.loc[72, "index_high"] = 105.
        no = self.observe(f, calendar, 91)
        self.assertFalse(no["extension_review"]["extend"])
        self.assertEqual(no["pending_exit"], "TIME_REVIEW_EXIT")
        f.loc[90:, ["index_close", "index_high", "exit_strong"]] = [104., 105., True]
        yes = self.observe(f, calendar, 91)
        self.assertTrue(yes["extension_review"]["extend"])
        self.assertEqual(self.observe(f, calendar, 131)["reason"], "HORIZON")
        f.loc[90, "index_close"] = 102.
        self.assertFalse(self.observe(f, calendar, 91)["extension_review"]["extend"])

    def test_missing_market_and_future_prices_cannot_enable_strength(self):
        f, _ = self.fixture()
        market = pd.DataFrame({"trade_date": f.date, "close": range(100, 240)})
        x = XP.enrich(f, XP.market_frame(market, f.date.iloc[-1]))
        short = XP.enrich(f.iloc[:90], XP.market_frame(market, f.date.iloc[89]))
        pd.testing.assert_frame_equal(x.iloc[:90], short)
        market = XP.market_frame(market.iloc[:85], f.date.iloc[-1])
        missing = XP.enrich(f, market)
        self.assertFalse(missing.market_known.iloc[-1])
        self.assertFalse(missing.exit_strong.iloc[-1])

    def test_gaps_and_missing_entry_do_not_invent_a_position(self):
        f, calendar = self.fixture()
        bad = self.observe(f.drop(index=73).reset_index(drop=True), calendar, 76)
        self.assertEqual(bad["status"], "DATA_GAP")
        f["accepted_bias2"] = False
        r = self.observe(f, calendar, 90)
        p = XP.plan(r, True)
        self.assertEqual(p["status"], "NO_RECENT_ACCEPTANCE")
        self.assertNotIn("hard_floor", r)
        self.assertFalse(p["orders_enabled"])
        self.assertEqual(XP.plan(r, False)["status"], "DATA_WAIT")

    def test_observer_matches_selected_research_exit(self):
        for kind in ("hard", "close", "time", "reduce", "targets_complete"):
            f, calendar = self.fixture()
            f.loc[72, "index_high"] = 105.
            if kind == "hard":
                f.loc[73, "index_low"] = 95.
            elif kind == "close":
                f.loc[74, ["index_low", "index_close"]] = [97., 98.]
            elif kind in {"reduce", "targets_complete"}:
                f.loc[72, "index_high"] = 107.
                f.loc[77, ["index_open", "index_high", "index_close", "amount_ratio", "close_location"]] = [106., 108., 106., 2., .2]
                if kind == "targets_complete":
                    f.loc[80, "index_high"] = 113.
            anchor = EP.reference(f.iloc[:72], calendar, "sh_600001")
            reference = self.observe(f, calendar, 110, anchor)
            trace = []
            research = RF.simulate(RF.arrays(f), 70, "sh_600001", "stage36_60", trace=trace,
                                   calendar=tuple(calendar), exit_policy="EXTEND_CLOSE")
            self.assertEqual(reference["closed_at"], research["exit_date"], kind)
            self.assertEqual(reference["reason"], research["reason"], kind)
            self.assertEqual([(e["date"], e["reason"]) for e in reference["events"]],
                             [(t["date"], e["reason"]) for t in trace for e in t["fills"]], kind)

    def test_missing_market_cannot_mask_exit_but_stale_price_does(self):
        from mainwave_engine.test_analysis_core import fixture
        from mainwave_engine import analysis_core as AC
        data, market = fixture()
        data["sell_observation"] = {"status": "HARD_EXIT_PENDING", "events": [], "reference": {}}
        market["data_status"] = "STALE"
        result = AC.evaluate(data, market)
        self.assertFalse(result["buy_plan"]["initial_review"])
        self.assertEqual(result["sell_plan"]["status"], "HARD_EXIT_PENDING")
        data["meta"]["price_date"] = "2026-09-03"
        self.assertEqual(AC.evaluate(data, market)["sell_plan"]["status"], "DATA_WAIT")


if __name__ == "__main__":
    unittest.main()
