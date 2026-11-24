import unittest

import pandas as pd

from research import confirmed_entry_replay as C, route_followup as RF, scaled_entry_replay as S


class ConfirmedEntryTests(unittest.TestCase):
    def frame(self):
        f = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=45),
            "index_open": 100., "index_high": 101., "index_low": 99.5, "index_close": 100.,
            "preclose": 100., "isST": 0, "atr20": 2., "low10": 94., "breakout_invalid_index": 95.,
            "breakout_id": "chain1", "breakout_status": "ACCEPTED", "eligible": True,
            "supply_warning": False, "close_location": .7, "ma20": 99., "amount": 100., "amount_ratio": 1.1})
        f.loc[12, ["index_open", "index_high", "index_low", "index_close", "amount"]] = [99., 100., 97.5, 98.5, 70.]
        f.loc[13, ["index_open", "index_high", "index_low", "index_close"]] = [99., 102., 98., 101.]
        f.loc[14, ["index_open", "index_high", "index_low", "index_close"]] = [100.5, 102., 100., 101.]
        return f

    def replay(self, f, mode="ABSORB", initial=.3, calendar=None, cap_initial=True):
        f = f.copy()
        for k in ("open", "high", "low", "close"):
            f[k] = f["index_" + k]
        a = RF.arrays(f)
        calendar = tuple(a["day"]) if calendar is None else calendar
        trace = []
        base = RF.simulate(a, 10, "sh_600001", "stage36_20", trace=trace, calendar=calendar)
        setting = ("ATR", .5, 1.)
        orders, audit = C.schedule(a, 10, trace, calendar, setting, mode, cap_initial=cap_initial)
        result = S.replay(a, 10, "sh_600001", base, trace, calendar, setting, initial=initial,
                          add_weights=C.weights(initial), add_schedule=orders)
        return result, orders, audit

    def test_confirm_after_touch_fill_only_next_session(self):
        f = self.frame()
        result, orders, _ = self.replay(f)
        self.assertEqual(len(result["buys"]), 3)
        self.assertEqual(result["buys"][1]["price_index"], 100.5)
        self.assertEqual(result["buys"][1]["date"], str(f.date.iloc[14].date()))
        first = orders[str(f.date.iloc[14].date())][0]
        self.assertEqual(first["touch_date"], str(f.date.iloc[12].date()))
        self.assertEqual(first["confirm_date"], str(f.date.iloc[13].date()))

    def test_price_touch_without_recovery_never_adds(self):
        f = self.frame()
        f.loc[12:, "index_close"] = 98.5
        result, orders, _ = self.replay(f, "STABILIZE")
        self.assertEqual(len(result["buys"]), 1)
        self.assertFalse(orders)

    def test_recovery_and_absorption_are_distinct(self):
        f = self.frame()
        f["amount"] = 100.
        stabilize, _, _ = self.replay(f, "STABILIZE")
        absorb, _, _ = self.replay(f, "ABSORB")
        self.assertEqual(len(stabilize["buys"]), 3)
        self.assertEqual(len(absorb["buys"]), 1)

    def test_strong_reclaim_requires_break_of_previous_high(self):
        f = self.frame()
        f.loc[13, "index_close"] = 99.8
        f.loc[14:, "index_close"] = 99.8
        absorb, _, _ = self.replay(f, "ABSORB")
        strong, _, _ = self.replay(f, "RECLAIM")
        self.assertEqual(len(absorb["buys"]), 3)
        self.assertEqual(len(strong["buys"]), 1)

    def test_confirmation_cannot_be_touch_bar_itself(self):
        f = self.frame()
        f.loc[12, "index_close"] = 100.5
        f.loc[12, "index_low"] = 99.
        _, orders, _ = self.replay(f, "STABILIZE")
        self.assertNotIn(str(f.date.iloc[13].date()), orders)

    def test_seventh_day_confirmation_cannot_fill_eighth(self):
        f = self.frame()
        f.loc[13:17, "index_close"] = 98.5
        f.loc[18, "index_close"] = 100.
        result, orders, _ = self.replay(f)
        self.assertEqual(len(result["buys"]), 1)
        self.assertNotIn(str(f.date.iloc[19].date()), orders)

    def test_opening_gap_guard_is_not_a_future_day_low_veto(self):
        f = self.frame()
        f.loc[14, "index_open"] = 104.
        result, _, _ = self.replay(f)
        self.assertFalse(any(b["date"] == str(f.date.iloc[14].date()) for b in result["buys"]))
        self.assertIn("OPEN_TOO_HIGH", [r["reason"] for r in result["add_rejections"]])

    def test_accepted_add_then_intraday_stop_defers_new_lots(self):
        f = self.frame()
        f.loc[14, "index_low"] = 95.
        result, _, _ = self.replay(f)
        self.assertEqual(len(result["buys"]), 3)
        self.assertEqual(result["deferred"], 2)
        self.assertTrue(all(s["date"] > s["lot_entry_date"] for s in result["sells"]))

    def test_new_breakout_chain_cancels_old_add_plan(self):
        f = self.frame()
        f.loc[13:, "breakout_id"] = "chain2"
        result, orders, audit = self.replay(f)
        self.assertEqual(len(result["buys"]), 1)
        self.assertEqual(audit["terminal"], "STRUCTURE_INVALID")
        self.assertFalse(orders)

    def test_open_limit_up_rejects_even_if_later_unlocked(self):
        f = self.frame()
        f.loc[14, "preclose"] = 91.36
        result, _, _ = self.replay(f)
        self.assertIn("OPEN_LIMIT_UP", [r["reason"] for r in result["add_rejections"]])

    def test_each_weight_has_exact_budget_and_each_tranche_once(self):
        for initial in C.INITIALS:
            result, _, _ = self.replay(self.frame(), initial=initial)
            self.assertAlmostEqual(sum(b["budget_fraction"] for b in result["buys"]), 1.)
            self.assertEqual(result["buys"][0]["budget_fraction"], initial)
            self.assertEqual(len({b["tranche"] for b in result["buys"]}), len(result["buys"]))

    def test_prefix_invariant_confirmation_orders_and_fills(self):
        f = self.frame()
        full, orders, _ = self.replay(f)
        prefix, prefix_orders, _ = self.replay(f.iloc[:14], calendar=tuple(f.date.dt.strftime("%Y-%m-%d")))
        next_day = str(f.date.iloc[14].date())
        self.assertEqual(prefix_orders, {d: v for d, v in orders.items() if d <= next_day})
        self.assertEqual(prefix["buys"], [b for b in full["buys"] if b["date"] < next_day])

    def test_unknown_gap_does_not_execute_stale_confirmation(self):
        f = self.frame()
        calendar = tuple(f.date.dt.strftime("%Y-%m-%d"))
        f = f.drop(index=14).reset_index(drop=True)
        result, _, _ = self.replay(f, calendar=calendar)
        self.assertEqual(len(result["buys"]), 1)
        self.assertEqual(result["status"], "HOLDING_DATA_UNVERIFIED")

    def test_cap_ablation_only_removes_initial_price_ceiling(self):
        f = self.frame()
        f.loc[14, "index_open"] = 101.2
        capped, _, _ = self.replay(f)
        uncapped, _, _ = self.replay(f, cap_initial=False)
        self.assertFalse(any(b["date"] == str(f.date.iloc[14].date()) for b in capped["buys"]))
        self.assertEqual(uncapped["buys"][1]["price_index"], 101.2)
        f.loc[14, "index_open"] = 104.
        uncapped, _, _ = self.replay(f, cap_initial=False)
        self.assertFalse(any(b["date"] == str(f.date.iloc[14].date()) for b in uncapped["buys"]))

    def test_same_bar_contraction_cannot_confirm_absorption(self):
        f = self.frame()
        f.loc[12, "amount"] = 100.
        f.loc[13, "amount"] = 70.
        _, orders, _ = self.replay(f)
        self.assertNotIn(str(f.date.iloc[14].date()), orders)

    def test_sixth_session_confirmation_can_fill_seventh(self):
        f = self.frame()
        f.loc[13:16, "index_close"] = 98.5
        f.loc[17, "index_close"] = 100.
        result, _, _ = self.replay(f)
        self.assertEqual(result["buys"][1]["date"], str(f.date.iloc[18].date()))

    def test_supply_warning_blocks_confirmation(self):
        f = self.frame()
        f.loc[13:18, "supply_warning"] = True
        result, orders, _ = self.replay(f, "STABILIZE")
        self.assertEqual(len(result["buys"]), 1)
        self.assertFalse(orders)


if __name__ == "__main__":
    unittest.main()
