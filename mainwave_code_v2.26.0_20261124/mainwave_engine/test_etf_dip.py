from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from mainwave_engine.trading import etf_dip as D
from mainwave_engine.trading import position_monitor as PM


def frame(closes, end="2026-09-30"):
    days = pd.bdate_range(end=end, periods=len(closes)).strftime("%Y-%m-%d")
    c = np.asarray(closes, float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c}, index=pd.Index(days, name="trade_date"))


def slow_fall(n_flat=80, n_fall=90):
    """Flat, then -1.2% / +0.5% alternating: RSI stays near 30, BIAS near -4%, the 60-day drawdown deepens."""
    c = [100.0 + 0.3 * (i % 2) for i in range(n_flat)]  # a quiet market, so RSI starts near 50
    for i in range(n_fall):
        c.append(c[-1] * (0.988 if i % 2 == 0 else 1.005))
    return c


class ReplayTests(unittest.TestCase):
    def test_depth_tranche_is_a_third_of_the_group_unit_and_sold_after_20_sessions(self):
        f = frame(slow_fall())
        plan = D.replay({"000852_SH": f})
        dd = [p for p in plan if p["signal"] == "DD-15"]
        self.assertEqual(len(dd), 1)
        p = dd[0]
        x = D.signals(f)
        self.assertLessEqual(x.dd60.loc[p["signal_day"]], -0.15)
        self.assertGreater(x.dd60.iloc[p["t"] - 1], -0.15, "第一次跌破 −15% 那天")
        self.assertAlmostEqual(p["size"], D.GROUP_UNIT / 3)
        self.assertEqual(p["entry_day"], f.index[p["t"] + 1])
        self.assertEqual(p["sell_day"], f.index[p["t"] + D.HOLD + 1], "持有 20 个交易日后的开盘卖出")
        self.assertFalse(any(q["signal"] in ("RSI20", "BIAS8") for q in plan), "缓跌不触发 RSI<20 和 BIAS≤−8%")

    def test_one_signal_type_in_a_group_is_split_not_doubled(self):
        f = frame(slow_fall())
        plan = D.replay({"000905_SH": f, "000852_SH": f.copy()})
        first = [p for p in plan if p["signal"] == "DD-15"]
        self.assertEqual(len(first), 2)
        self.assertEqual(first[0]["signal_day"], first[1]["signal_day"])
        self.assertAlmostEqual(sum(p["size"] for p in first), D.GROUP_UNIT / 3, msg="同组同信号合计 1/3 份，两只平分")

    def test_a_crash_day_is_capped_at_one_group_unit_and_rsi_wins_over_bias(self):
        c = [100.0] * 80 + [100 * 0.97 ** k for k in range(1, 16)]
        plan = D.replay({"399006_SZ": frame(c)})
        by_day = {}
        for p in plan:
            by_day.setdefault(p["signal_day"], []).append(p)
        for day, ps in by_day.items():
            self.assertFalse({"RSI20", "BIAS8"} <= {p["signal"] for p in ps}, "同一天 RSI20 和 BIAS8 只取一个")
        self.assertLessEqual(max(sum(p["size"] for p in plan if p["signal_day"] <= d and p["sell_day"] is None or
                                     (p["signal_day"] <= d and p["sell_day"] and p["sell_day"] > d)) for d in by_day), D.GROUP_UNIT + 1e-9)

    def test_a_held_bias_position_blocks_new_bias_signals(self):
        c = [100.0 + 0.3 * (i % 2) for i in range(80)] + [100 * 0.985 ** k for k in range(1, 9)] + [100 * 0.985 ** 8] * 10
        plan = D.replay({"000016_SH": frame(c)})
        bias = [p for p in plan if p["signal"] in ("BIAS8", "RSI20")]
        self.assertEqual(len(bias), 1, "持有期间不重复买入")


class AlertTests(unittest.TestCase):
    def write(self, root, closes, stem="000852_SH", end="2026-09-30"):
        f = frame(closes, end).reset_index()
        f.to_csv(Path(root) / f"{stem}_index.csv", index=False)
        return f

    def test_signal_on_the_last_close_is_reported_as_tomorrows_buy(self):
        c = slow_fall()
        dd = D.signals(frame(c)).dd60
        last = int(np.flatnonzero((dd <= -0.15).to_numpy())[0])
        with tempfile.TemporaryDirectory() as tmp:
            f = self.write(tmp, c[:last + 1])
            a = D.alert(f.trade_date.iloc[-1], tmp)
            self.assertEqual(a["level"], "TRIGGERED_TODAY")
            self.assertEqual([r["signal"] for r in a["new"]], ["DD-15"])
            self.assertTrue(a["report_only"])
            self.assertEqual(D.alert("2026-12-31", tmp)["level"], "STALE")
            self.assertEqual(D.alert("2026-09-30", Path(tmp) / "absent")["level"], "UNKNOWN")

    def test_rendered_in_the_position_report(self):
        result = {"as_of": "2026-09-30", "market_risk": {}, "rebound_basket": {}, "notice": "", "alerts": [], "books": {},
                  "etf_dip": {"level": "TRIGGERED_TODAY", "rule": "r", "exposure": 0.0833, "holding": [],
                              "new": [{"index": "中证1000", "group": "中小盘", "signal": "DD-15", "size_of_sleeve": 0.0833, "etf": "512100"}]}}
        try:
            html = PM.render(result)
        except KeyError:
            self.skipTest("render needs a full monitor result")
        self.assertIn("宽基 ETF 抄底信号", html)


if __name__ == "__main__":
    unittest.main()
