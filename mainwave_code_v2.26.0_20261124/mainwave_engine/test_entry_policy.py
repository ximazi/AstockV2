import unittest

import pandas as pd

from mainwave_engine import entry_policy as EP


class EntryPolicyTests(unittest.TestCase):
    def fixture(self):
        f = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=46), "accepted_bias2": False,
            "index_open": 100., "index_high": 101., "index_low": 99.5, "index_close": 100.,
            "open": 100., "close": 100., "preclose": 100., "isST": 0, "atr20": 2.,
            "breakout_invalid_index": 95., "breakout_id": "one", "breakout_status": "ACCEPTED",
            "eligible": True, "supply_warning": False, "close_location": .7, "amount": 100., "amount_ratio": 1.1})
        f.loc[30, "accepted_bias2"] = True
        f.loc[32, ["index_low", "index_close", "amount"]] = [96.5, 98., 70.]
        f.loc[33, ["index_low", "index_close", "index_high"]] = [98., 101.5, 102.]
        f["close"] = f.index_close
        return f, list(f.date.dt.strftime("%Y-%m-%d"))

    def test_initial_budget_and_remainder(self):
        p = EP.plan(None, "PRIORITY_REVIEW", [], True, True)
        self.assertEqual(p["initial_fraction"], .5)
        self.assertAlmostEqual(sum(p["add_fractions"]), .5)
        self.assertFalse(p["orders_enabled"])

    def test_reclaim_is_later_than_touch_and_next_open_only(self):
        f, calendar = self.fixture()
        r = EP.observe(f.iloc[:34], calendar, "sh_600001")
        self.assertEqual(r["status"], "CONDITIONAL_ADD_REVIEW")
        self.assertEqual(len(r["confirmations"]), 1)
        c = r["confirmations"][0]
        self.assertLess(c["touch_date"], c["confirm_date"])
        self.assertEqual(c["execution"], "NEXT_EXCHANGE_OPEN_ONLY")
        self.assertEqual(c["max_price"], 101.)
        self.assertFalse(EP.observe(f.iloc[:33], calendar, "sh_600001")["confirmations"])

    def test_contraction_must_precede_confirmation(self):
        f, calendar = self.fixture()
        f.loc[32, "amount"] = 100.
        f.loc[33, "amount"] = 60.
        self.assertFalse(EP.observe(f.iloc[:34], calendar, "sh_600001")["confirmations"])

    def test_invalid_stop_or_new_cycle_cancels(self):
        for field, value in (("index_low", 95.), ("breakout_id", "new"), ("index_high", 107.)):
            f, calendar = self.fixture()
            f.loc[32, field] = value
            self.assertIn(EP.observe(f.iloc[:34], calendar, "sh_600001")["status"],
                          {"STRUCTURE_INVALID", "SELL_REFERENCE_CANCELLED"})

    def test_no_eighth_day_add_or_unexplained_gap(self):
        f, calendar = self.fixture()
        self.assertEqual(EP.observe(f.iloc[:39], calendar, "sh_600001")["status"], "WINDOW_EXPIRED")
        self.assertEqual(EP.observe(f.iloc[:34].drop(index=32), calendar, "sh_600001")["status"], "DATA_GAP")

    def test_initial_reference_must_pass_price_guard(self):
        f, calendar = self.fixture()
        f.loc[31, ["index_open", "open"]] = 105.
        self.assertEqual(EP.observe(f.iloc[:34], calendar, "sh_600001")["status"], "INITIAL_REFERENCE_REJECTED")

    def test_confirmed_pullback_gets_its_own_entry_anchor(self):
        f, calendar = self.fixture()
        setup = {"setup_status": "CONFIRMED", "setup_confirmed_at": calendar[32], "setup_kind": "FIRST_PULLBACK",
                 "setup_breakout_id": "one", "setup_frozen_atr_price": 2., "setup_invalid_price": 95.}
        r = EP.observe(f.iloc[:34], calendar, "sh_600001", setup)
        self.assertEqual(r["signal_date"], calendar[32])
        self.assertEqual(r["reference_entry_date"], calendar[33])
        self.assertEqual(r["signal_kind"], "FIRST_PULLBACK")

    def test_risk_and_missing_context_cannot_be_overridden_by_add(self):
        obs = {"status": "CONDITIONAL_ADD_REVIEW", "confirmations": []}
        for risks, ready, fresh in (([{"label": "risk"}], True, True), ([], False, True), ([], True, False)):
            p = EP.plan(obs, "PRIORITY_REVIEW", risks, ready, fresh)
            self.assertFalse(p["initial_review"])
            self.assertFalse(p["add_review"])


if __name__ == "__main__":
    unittest.main()
