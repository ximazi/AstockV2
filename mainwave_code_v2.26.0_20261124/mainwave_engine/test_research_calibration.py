"""Audit v1.2 probes as regressions (Q-01..Q-05): the research bench must not sell on a non-trading row, must stop
MFE/MAE at the actual sale, must not let a later split rewrite a finished trade, must keep a raw tick invariant,
must apply the same warm-up as the daily run, and must check only last night's frozen plans."""
import unittest

import numpy as np
import pandas as pd

from . import strong_trend as ST
from . import strong_trend_semantics as SEM
from .test_strong_trend_semantics import frame, events

ZERO_FEE = lambda notional, sell: 0.0


class ExitTests(unittest.TestCase):
    def test_exit_waits_for_a_tradable_open(self):
        f = frame([10.] * 60 + [9.4, 8.5, 8.6, 8.7])
        f.loc[:59, "low"] = 9.7
        f.loc[60, "low"] = 9.3
        f.loc[61, ["open", "high", "low", "close", "volume", "tradestatus"]] = [8.5, 8.5, 8.5, 8.5, 0, 0]
        out = SEM.replay_exit_executable(ST.adjusted(f), 59, 9.5, "sh_600001")
        self.assertEqual((out["reason"], out["signal_index"], out["exit_index"]), ("HARD_RISK", 60, 62))
        self.assertEqual(out["blocked_opens"][0]["why"], "NOT_TRADED")
        self.assertEqual(ST.replay_exit(f, 59, 9.5)["exit_index"], 61, "CORE-V1 函数保持原样，仅研究台改用可卖退出")

    def test_limit_down_open_blocks_the_sale(self):
        f = frame([10.] * 60 + [9.4, 8.46, 8.5])
        f.loc[:59, "low"] = 9.7
        f.loc[60, "low"] = 9.3
        f.loc[61, ["open", "preclose"]] = [8.46, 9.4]  # 9.4 * 0.9 = 8.46: limit-down open
        out = SEM.replay_exit_executable(ST.adjusted(f), 59, 9.5, "sh_600001")
        self.assertEqual((out["exit_index"], out["blocked_opens"][0]["why"]), (62, "LIMIT_DOWN_OPEN"))

    def test_no_tradable_open_keeps_the_exit_pending(self):
        f = frame([10.] * 60 + [9.4, 8.5])
        f.loc[:59, "low"] = 9.7
        f.loc[60, "low"] = 9.3
        f.loc[61, ["volume", "tradestatus"]] = [0, 0]
        out = SEM.replay_exit_executable(ST.adjusted(f), 59, 9.5, "sh_600001")
        self.assertEqual((out["exit_index"], out["state"]), (None, "EXIT_PENDING"))


class TradeTests(unittest.TestCase):
    def test_mfe_mae_stop_at_the_sale(self):
        f = frame([10.] * 60 + [9.4, 9.3])
        f.loc[:59, "high"] = 10.1
        f.loc[:59, "low"] = 9.7
        f.loc[60, ["high", "low"]] = [9.5, 9.3]
        f.loc[61, ["open", "high", "low", "close"]] = [9.3, 10.3, 8.8, 9.3]
        e = {"index": 58, "invalidation": 8.0, "n0": 1.0, "boundary": 9.5}
        r = SEM.frozen_trade(ST.adjusted(f), e, "sh_600001", fee=ZERO_FEE)
        self.assertEqual((r["trade"], r["exit_date"], r["exit_price"]), ("CLOSED", f["date"].iloc[61], 9.3))
        self.assertAlmostEqual(r["mfe_r"], 0.05, places=3)   # 10.1 on the entry bar, not 10.3 after the sale
        self.assertAlmostEqual(r["mae_r"], -0.35, places=3)  # 9.3 exit open, not the 8.8 low after it

    def test_a_later_split_does_not_rewrite_a_finished_trade(self):
        f = frame([10.] * 60 + [10.5, 10.7, 10.1, 9.4, 9.2, 9.2])
        f.loc[61, "open"] = 10.55
        a = SEM.analyse(f, None, "sh_600001")
        e = next(x for x in a["events"] if x["index"] == 60)
        before = SEM.frozen_trade(a["f"], e, "sh_600001")
        later = pd.concat([f, frame([4.6], start="2025-04-10")], ignore_index=True)
        later.loc[len(f), "preclose"] = 4.6  # 2-for-1 after the trade has closed
        b = SEM.analyse(later, None, "sh_600001")
        e2 = next(x for x in b["events"] if x["index"] == 60)
        after = SEM.frozen_trade(b["f"], e2, "sh_600001")
        for k in ("qty", "entry", "exit_price", "net_pnl", "net_r", "exit_date"):
            self.assertEqual(before[k], after[k], k)
        self.assertEqual(before["qty"], 1100)

    def test_raw_tick_is_invariant_to_a_later_split(self):
        f = frame([10.] * 60 + [10.8, 11.2, 11.0, 10.9, 11.0, 11.5])
        a0 = SEM.analyse(f, None, "sh_600001")
        i = len(f) - 1
        later = pd.concat([f, frame([5.75], start="2025-04-03")], ignore_index=True)
        later.loc[len(f), "preclose"] = 5.75
        a1 = SEM.analyse(later, None, "sh_600001")
        self.assertAlmostEqual(a1["sig"]["ra_inv"][i], a0["sig"]["ra_inv"][i] * 0.5, places=9)
        self.assertAlmostEqual(SEM.raw_price(a1["f"], i, a1["sig"]["ra_inv"][i]), a0["sig"]["ra_inv"][i], places=9)


class EligibilityTests(unittest.TestCase):
    def test_warmup_matches_the_daily_run(self):
        f = frame([10.] * 25 + [10.5 * (1.025 ** k) for k in range(50)])
        ev, a = events(f)
        first = ev[0]
        el = SEM.eligibility_at(f, a["f"], first["index"], None, "sh_600001")
        self.assertFalse(el["research_eligible"])
        self.assertIn("WARMUP", el["reasons"])
        el60 = SEM.eligibility_at(f, a["f"], 60, None, "sh_600001")
        self.assertIn("BENCHMARK_MISSING", el60["reasons"])
        self.assertNotIn("WARMUP", el60["reasons"])

    def test_strength_rule_is_the_daily_rule(self):
        f = frame([10.] * 70 + [10.5])
        bench = pd.Series(np.full(len(f), 100.0), index=f["date"])
        a = SEM.analyse(f, bench, "sh_600001")
        self.assertTrue(SEM.eligibility_at(f, a["f"], 70, bench, "sh_600001")["plan_eligible"])
        falling = pd.Series(np.r_[np.full(60, 100.0), np.linspace(100, 130, 11)], index=f["date"])
        el = SEM.eligibility_at(f, a["f"], 70, falling, "sh_600001")
        self.assertEqual((el["research_eligible"], el["plan_eligible"], el["reasons"]), (True, False, ["STRENGTH_RULE_FAILED"]))


class FrozenPlanTests(unittest.TestCase):
    def test_only_the_frozen_plan_band_is_checked(self):
        raw = frame([10.] * 30 + [10.5, 10.6])
        plan = {"band": [10.4, 10.7], "setup_id": "s", "kind": "NEW", "execution_state": "CONDITIONAL_NEXT_SESSION"}
        day = raw["date"].iloc[-1]
        raw.loc[31, "open"] = 10.6
        self.assertEqual(SEM.plan_check(raw, plan, day, "sh_600001")["state"], "IN_BAND")
        raw.loc[31, "open"] = 10.9
        self.assertEqual(SEM.plan_check(raw, plan, day, "sh_600001")["state"], "ABOVE_BAND")
        raw.loc[31, "open"] = 11.55
        self.assertEqual(SEM.plan_check(raw, plan, day, "sh_600001")["state"], "LIMIT_FILL_UNKNOWN")
        raw.loc[31, ["volume", "tradestatus"]] = [0, 0]
        self.assertEqual(SEM.plan_check(raw, plan, day, "sh_600001")["state"], "NOT_TRADED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
