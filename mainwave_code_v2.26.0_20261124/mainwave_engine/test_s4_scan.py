import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from mainwave_engine import s4_scan as S4


def s4_frame(last_pct=0.03):
    """An uptrend, three down days onto EMA20, five quiet days, then a relaunch day."""
    closes = list(100 * np.cumprod(np.full(120, 1.008)))
    for _ in range(3):
        closes.append(closes[-1] * 0.975)
    closes += [closes[-1]] * 5
    closes.append(closes[-1] * (1 + last_pct))
    c = np.array(closes)
    return pd.DataFrame({"open": c / 1.01, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1000.0})


class SignalTests(unittest.TestCase):
    def test_pullback_then_relaunch_is_s4(self):
        x = S4.signals(s4_frame())
        self.assertTrue(x["s4"].iloc[-1], x.tail(3).to_string())
        self.assertFalse(x["R3"].iloc[-1])

    def test_a_weak_day_is_not_a_relaunch(self):
        self.assertFalse(S4.signals(s4_frame(last_pct=0.01))["s4"].iloc[-1])

    def test_a_frame_without_any_signal_still_reports(self):
        f = s4_frame().iloc[:100]
        x = S4.signals(f)
        self.assertFalse(x["s4"].any())
        self.assertTrue(np.isfinite(x["pct"].iloc[-1]), "前一日收盘不能被事件变量覆盖")


class PickTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        folder = Path(self.tmp.name) / "20260924"
        folder.mkdir()
        (folder / "s4_opportunities.json").write_text(json.dumps({
            "as_of": "2026-09-24", "buy_advised": True, "market_risk": {"as_of": "2026-09-24", "level": "NORMAL"},
            "index_trend": {"as_of": "2026-09-24", "status": "FRESH", "regime": "DOWN", "gate": "ALLOW",
                            "slope": -0.017, "text": "沪深300 EMA100 中期趋势向下, 允许 S4 新仓"},
            "recommended": [{"symbol": "sz_002093", "event": {"event_id": "sz_002093:2026-09-17", "touch_day": "2026-09-17",
                                                             "pullback_low": 7.05, "signal_open": 7.60}}],
            "excluded": [{"symbol": "sh_605177", "excluded": ["R1 顶背离"]}]}), encoding="utf-8")

    def pick(self, symbol):
        with patch.object(S4.VP, "add", return_value={"created": True}) as add, \
                patch.object(S4.SA, "normalize_symbol", side_effect=lambda s: s):
            return S4.pick("2026-09-24", symbol, out_root=self.tmp.name), add

    def test_recommended_symbol_enters_with_the_lad_exit(self):
        _, add = self.pick("sz_002093")
        add.assert_called_once_with("sz_002093", "2026-09-24", S4.VP.DB_PATH, strategy="S4_LAD",
                                    meta={"s4_event_id": "sz_002093:2026-09-17", "s4_touch_day": "2026-09-17",
                                          "s4_pullback_low": 7.05, "s4_signal_open": 7.60})

    def test_excluded_or_unknown_symbol_is_refused_with_the_reason(self):
        with self.assertRaisesRegex(ValueError, "顶背离"):
            self.pick("sh_605177")
        with self.assertRaisesRegex(ValueError, "不在当日"):
            self.pick("sh_600000")

    def test_missing_scan_is_refused(self):
        with self.assertRaisesRegex(ValueError, "还没有"):
            S4.pick("2026-09-25", "sz_002093", out_root=self.tmp.name)

    def test_market_gate_rejects_red_unknown_stale_and_missing(self):
        path = Path(self.tmp.name) / "20260924/s4_opportunities.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        for market in ({"as_of": "2026-09-24", "level": "TREND_EXIT"},
                       {"as_of": "2026-09-24", "level": "UNKNOWN"},
                       {"as_of": "2026-09-23", "level": "NORMAL"}, {}):
            path.write_text(json.dumps({**original, "market_risk": market}), encoding="utf-8")
            with patch.object(S4.VP, "add") as add, self.assertRaisesRegex(ValueError, "入场复核未通过"):
                S4.pick("2026-09-24", "sz_002093", out_root=self.tmp.name)
            add.assert_not_called()

    def test_index_trend_gate_rejects_blocked_unknown_and_missing(self):
        path = Path(self.tmp.name) / "20260924/s4_opportunities.json"
        original = json.loads(path.read_text(encoding="utf-8"))
        for trend in ({"as_of": "2026-09-24", "status": "FRESH", "regime": "UP", "gate": "BLOCK",
                       "slope": 0.021, "text": "中期趋势向上, 不开新仓"},
                      {"as_of": "2026-09-24", "status": "STALE", "gate": "UNKNOWN", "text": "数据过期"},
                      {}):
            path.write_text(json.dumps({**original, "index_trend": trend}), encoding="utf-8")
            with patch.object(S4.VP, "add") as add, self.assertRaisesRegex(ValueError, "中期趋势门"):
                S4.pick("2026-09-24", "sz_002093", out_root=self.tmp.name)
            add.assert_not_called()

    def test_theme_caution_rejects_before_ledger_write(self):
        with patch.object(S4.TR, "evaluate_many", return_value={"theme_risk_action": "REQUIRE_EXTERNAL_CONFIRMATION", "theme_risk_reason": "题材风险"}), \
                patch.object(S4.VP, "add") as add, self.assertRaisesRegex(ValueError, "题材风险"):
            S4.pick("2026-09-24", "sz_002093", out_root=self.tmp.name)
        add.assert_not_called()


class ApiTests(PickTests):
    def test_list_endpoint_and_s4_add_only_for_listed_names(self):
        import mainwave_visual_app as web
        with patch.object(web, "OUT_ROOT", Path(self.tmp.name)), patch.object(S4.SA, "OUT_ROOT", Path(self.tmp.name)),                 patch.dict(web.app.config, VIRTUAL_PORTFOLIO_DB=Path(self.tmp.name) / "vp.sqlite3"), web.app.test_client() as client:
            self.assertEqual([r["symbol"] for r in client.get("/api/s4?date=2026-09-24").json["recommended"]], ["sz_002093"])
            self.assertTrue(client.get("/api/s4?date=2026-09-25").json["missing"])
            refused = client.post("/api/virtual-positions", json={"symbol": "605177", "date": "2026-09-24", "strategy": "S4_LAD"})
            self.assertEqual(refused.status_code, 400)
            self.assertIn("顶背离", refused.json["error"])
            with patch.object(S4.VP, "add", return_value={"created": True}) as add:
                ok = client.post("/api/virtual-positions", json={"symbol": "002093", "date": "2026-09-24", "strategy": "S4_LAD"})
            self.assertEqual(ok.status_code, 200)
            self.assertEqual(add.call_args.kwargs["strategy"], "S4_LAD")


class ReferenceRankTests(unittest.TestCase):
    def test_close_location(self):
        self.assertEqual(S4.clv({"high": 11.0, "low": 9.0, "close": 10.5}), 0.75)
        self.assertIsNone(S4.clv({"high": 10.0, "low": 10.0, "close": 10.0}))

    def test_fundamentals_use_the_latest_report_announced_before_the_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fin.csv"
            pd.DataFrame({"ts_code": ["600001.SH", "600001.SH", "600001.SH", "000002.SZ"],
                          "ann_date": [20260425.0, 20260425.0, 20260924.0, 20260801.0],
                          "end_date": [20251231, 20260331, 20260630, 20260630],
                          "q_sales_yoy": [5.0, 12.0, 30.0, None]}).to_csv(path, index=False)
            sales, latest = S4.fundamentals("2026-09-24", ["sh_600001", "sz_000002"], path)
            self.assertEqual(sales["sh_600001"], 12.0, "当天公告的不用；同日公告取最新一期")
            self.assertIsNone(sales["sz_000002"])
            self.assertEqual(latest, "20260801")
            self.assertEqual(S4.fundamentals("2026-09-24", ["sh_600001"], Path(tmp) / "absent.csv"), ({}, ""))

    def test_reference_rank_favours_sales_growth_and_a_lower_close_and_keeps_the_official_order(self):
        kept = [{"symbol": "a", "rank": 1, "clv": 1.0}, {"symbol": "b", "rank": 2, "clv": 0.4}, {"symbol": "c", "rank": 3, "clv": 0.7}]
        S4.reference(kept, {"a": -5.0, "b": 25.0})
        self.assertEqual([r["rank"] for r in kept], [1, 2, 3])
        self.assertEqual({r["symbol"]: r["ref_rank"] for r in kept}, {"b": 1, "c": 2, "a": 3})
        self.assertEqual(kept[2]["ref_score"], round((2 / 3 + 0.5) / 2, 3), "缺营收数据记中位")

    def test_forward_log_replaces_a_rerun_day(self):
        out = {"as_of": "2026-10-09", "version": "v", "buy_advised": False, "fundamentals_latest_ann": "20261008",
               "recommended": [{"symbol": "a", "name": "A", "rank": 1, "ref_rank": 1, "ref_score": 0.9, "clv": 0.2, "q_sales_yoy": 10.0}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.csv"
            S4.log_reference(out, path)
            S4.log_reference({**out, "as_of": "2026-10-10"}, path)
            S4.log_reference(out, path)
            log = pd.read_csv(path, dtype={"date": str})
            self.assertEqual(sorted(log.date), ["2026-10-09", "2026-10-10"])


if __name__ == "__main__":
    unittest.main()
