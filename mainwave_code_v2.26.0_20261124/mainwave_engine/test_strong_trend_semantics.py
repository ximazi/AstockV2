"""CORE-SEMANTICS acceptance (screening quality task book v1.1 §13): event identity, progress vs new setups,
retry/child/invalidation, ex-rights stability, next-open checks and review evidence."""
import unittest

import numpy as np
import pandas as pd

from . import strong_trend as ST
from . import strong_trend_semantics as SEM


def frame(closes, start="2025-01-01", preclose=None):
    closes = np.asarray(closes, float)
    days = pd.bdate_range(start, periods=len(closes)).strftime("%Y-%m-%d")
    pre = np.r_[closes[0], closes[:-1]] if preclose is None else np.asarray(preclose, float)
    return pd.DataFrame({"date": days, "open": closes, "high": closes * 1.01, "low": closes * 0.99, "close": closes,
                         "preclose": pre, "volume": 1000.0, "tradestatus": 1, "isST": 0})


def events(f, symbol="sh_600001"):
    a = SEM.analyse(f, None, symbol)
    return a["events"], a


class SignalParityTests(unittest.TestCase):
    def test_vectorised_signals_match_the_core_v1_detectors(self):
        rng = np.random.default_rng(7)
        for seed in range(3):
            closes = 10 * np.exp(np.cumsum(rng.normal(0.004, 0.025, 260)))
            f = ST.adjusted(frame(closes))
            atr = ST.wilder_atr(f).to_numpy()
            sig = SEM.bar_signals(f, atr)
            for i in range(len(f)):
                bo, ra = ST.breakout_at(f, i, atr), ST.reacceleration_at(f, i, atr)
                self.assertEqual(bool(sig["bo"][i]), bo is not None, (seed, i))
                self.assertEqual(bool(sig["ra"][i]), ra is not None, (seed, i))
                if ra:
                    # CORE-V1 rounds to 4 decimals; the semantics layer keeps full precision so ex-rights scaling stays exact.
                    self.assertAlmostEqual(sig["ra_inv"][i], ra["invalidation"], places=3)


class EventTests(unittest.TestCase):
    def test_five_consecutive_new_highs_are_one_setup(self):
        ev, _ = events(frame([10.0] * 60 + [10.5, 10.75, 11.0, 11.3, 11.6]))
        self.assertEqual([e["kind"] for e in ev], ["NEW", "PROGRESS", "PROGRESS", "PROGRESS", "PROGRESS"])
        self.assertEqual(len({e["setup_id"] for e in ev}), 1)

    def test_a_real_new_consolidation_is_a_child_not_blocked_by_a_cooldown(self):
        ev, _ = events(frame([10.0] * 60 + [10.5, 10.75, 11.0, 11.3, 11.6, 11.5, 11.4, 11.45, 12.0]))
        self.assertEqual(ev[-1]["kind"], "CHILD")
        self.assertEqual(ev[-1]["parent_id"], ev[0]["setup_id"])
        self.assertNotEqual(ev[-1]["setup_id"], ev[0]["setup_id"])

    def test_fall_back_then_break_is_a_retry_of_the_same_setup(self):
        ev, _ = events(frame([10.0] * 60 + [10.5, 10.05, 10.8]))
        self.assertEqual([e["kind"] for e in ev], ["NEW", "RETRY"])
        self.assertEqual(ev[0]["setup_id"], ev[1]["setup_id"])

    def test_invalidation_ends_the_setup(self):
        ev, a = events(frame([10.0] * 60 + [10.5, 9.0] + [9.0] * 25 + [9.5]))
        self.assertEqual([e["kind"] for e in ev], ["NEW", "NEW"])
        self.assertNotEqual(ev[0]["setup_id"], ev[1]["setup_id"])

    def test_identity_survives_new_days_and_an_ex_rights_day(self):
        closes = [10.0] * 60 + [10.5, 10.75, 11.0] + [11.0] * 12
        base, _ = events(frame(closes))
        later_closes = closes + [9.9] * 5  # a 10% cash dividend on the next bar: preclose drops, price level drops
        pre = np.r_[later_closes[0], later_closes[:-1]]
        pre[len(closes)] = 9.9
        later, _ = events(frame(later_closes, preclose=pre))
        self.assertEqual(base[0]["setup_id"], later[0]["setup_id"])
        self.assertEqual([e["kind"] for e in later], [e["kind"] for e in base], "除权日不产生新结构")

    def test_setup_id_carries_no_price(self):
        self.assertEqual(SEM.setup_id("sh_600001", "2026-09-01", "BREAKOUT"), SEM.setup_id("sh_600001", "2026-09-01", "BREAKOUT"))
        self.assertNotEqual(SEM.setup_id("sh_600001", "2026-09-01", "BREAKOUT"), SEM.setup_id("sh_600001", "2026-09-02", "BREAKOUT"))


class ExecutionTests(unittest.TestCase):
    def check(self, next_open, next_pre=None):
        closes = [10.0] * 60 + [10.5, 10.5]
        raw = frame(closes)
        raw.loc[61, "open"] = next_open
        if next_pre:
            raw.loc[61, "preclose"] = next_pre
        ev, a = events(raw)
        return SEM.open_check(raw, a["f"], ev[0], "sh_600001")

    def test_next_open_states(self):
        self.assertEqual(self.check(10.3)["state"], "IN_BAND")
        self.assertEqual(self.check(10.9)["state"], "ABOVE_BAND")
        self.assertEqual(self.check(10.0)["state"], "BELOW_BAND")
        self.assertEqual(self.check(11.55)["state"], "LIMIT_FILL_UNKNOWN", "涨停开盘只标成交未知")

    def test_review_items_are_pass_fail_or_unknown_with_evidence(self):
        row = {"kind": "PROGRESS", "setup_id": "x", "setup_first_day": "d", "event_age": 3, "strength_eligible": True,
               "rs5": .1, "rs20": .2, "plan_legal": True, "band": [1, 2], "invalidation": .5, "execution_state": "NOT_NEW_ENTRY"}
        items = SEM.review_items(row)
        self.assertTrue(all(i["result"] in ("PASS", "FAIL", "UNKNOWN") and i["evidence"] for i in items))
        self.assertEqual(items[0]["result"], "FAIL")
        self.assertEqual(items[-1]["result"], "UNKNOWN", "账户政策待配置，不能标成执行完整")

    def test_domains(self):
        self.assertEqual(SEM.domain_of("sh_600001", "甲"), "EXECUTABLE")
        self.assertEqual(SEM.domain_of("sz_300001", "乙"), "RESEARCH")
        self.assertEqual(SEM.domain_of("sz_000001", "*ST丙"), "RESEARCH")


if __name__ == "__main__":
    unittest.main(verbosity=2)
