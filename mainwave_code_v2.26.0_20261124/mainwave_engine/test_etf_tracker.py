from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from mainwave_engine import etf_tracker as ET


def fixture(n=450):
    dates = pd.bdate_range("2025-01-01", periods=n)
    x = np.arange(n)
    close = 1 + x * .001 + np.sin(x / 8) * .025
    daily = pd.DataFrame({"trade_date": dates.strftime("%Y%m%d"), "open": close * .997,
                          "high": close * 1.012, "low": close * .986, "close": close,
                          "amount": np.full(n, 60000.), "ts_code": "588170.SH"})
    factors = pd.DataFrame({"trade_date": daily.trade_date, "adj_factor": np.ones(n)})
    return daily, factors


class ETFTrackerTests(unittest.TestCase):
    def setUp(self):
        self.item = ET.registry()["items"][0]
        self.daily, self.adj = fixture()
        self.end = ET.normalize_date(self.daily.trade_date.iloc[-1])

    def test_fixed_universe_and_snapshot(self):
        items = ET.registry()["items"]
        self.assertEqual(len(items), 10)
        self.assertEqual(len({r["theme"] for r in items}), 10)
        self.assertEqual(ET.find_item("159326")["code"], "159326.SZ")
        self.assertEqual(ET.find_item("513520.sh")["scope"], "海外/QDII")
        for code in ("600354", "../../secret", "513350.SZ"):
            with self.assertRaises(ValueError):
                ET.find_item(code)

    def test_split_does_not_create_crash_or_change_stage(self):
        original = ET.analyze_etf(self.item, self.daily, self.adj, pd.DataFrame(), self.end)
        split, factors = self.daily.copy(), self.adj.copy()
        split.loc[split.index[-4:], ["open", "high", "low", "close"]] /= 3
        factors.loc[factors.index[-4:], "adj_factor"] = 3
        result = ET.analyze_etf(self.item, split, factors, pd.DataFrame(), self.end)
        for key in ("ytd_pct", "r5_pct", "r20_pct", "bias20_pct", "bias_atr", "stabilizing"):
            self.assertEqual(result["metrics"][key], original["metrics"][key], key)
        self.assertEqual(result["lifecycle"], original["lifecycle"])
        self.assertAlmostEqual(result["metrics"]["ema20"] * 3, original["metrics"]["ema20"], places=3)
        self.assertLess(abs(result["chart"]["bars"][-1]["close"] / result["chart"]["bars"][-2]["close"] - 1), .03)
        self.assertEqual(result["metrics"]["amount20_yi"], .6)
        json.dumps(result, allow_nan=False)

    def test_dividend_factor_is_in_ytd(self):
        frame = ET.adjusted_frame(self.daily, self.adj, self.end)
        base = frame.loc[frame.trade_date == "20251231"].iloc[0]
        adj = self.adj.copy()
        adj.loc[adj.trade_date > "20251231", "adj_factor"] = 1.1
        result = ET.analyze_etf(self.item, self.daily, adj, pd.DataFrame(), self.end)
        expected = round((self.daily.close.iloc[-1] * 1.1 / base.close - 1) * 100, 2)
        self.assertEqual(result["metrics"]["ytd_pct"], expected)

    def test_asof_truncation_and_continuous_prefix(self):
        day = ET.normalize_date(self.daily.trade_date.iloc[-20])
        a = ET.analyze_etf(self.item, self.daily, self.adj, pd.DataFrame(), day)
        b = ET.analyze_etf(self.item, self.daily.iloc[:-19], self.adj.iloc[:-19], pd.DataFrame(), day)
        self.assertEqual(a, b)
        c = ET.analyze_etf(self.item, self.daily, self.adj, pd.DataFrame(), self.end)
        old = {r["date"]: r for r in a["lifecycle"]["history"]}
        for row in c["lifecycle"]["history"]:
            if row["date"] in old:
                self.assertEqual(row, old[row["date"]])

    def test_missing_factors_reject_instead_of_raw_fallback(self):
        with self.assertRaises(ValueError):
            ET.adjusted_frame(self.daily, self.adj.iloc[:-1], self.end)
        for column, bad in (("close", 0), ("low", 100), ("amount", -1)):
            daily = self.daily.copy()
            daily.loc[daily.index[-1], column] = bad
            with self.assertRaises(ValueError):
                ET.adjusted_frame(daily, self.adj, self.end)

    def test_nav_requires_known_announcement_and_matching_close(self):
        nav = pd.DataFrame({"nav_date": ["20260828", "20260831", "20260901"],
                            "ann_date": ["20260831", "20260901", "20260902"], "unit_nav": [1., 2., 3.]})
        daily = pd.DataFrame({"trade_date": ["20260828", "20260831"], "close": [1.01, 2.2]})
        result = ET.nav_observation(nav, daily, "2026-08-31")
        self.assertEqual(result["date"], "2026-08-28")
        self.assertAlmostEqual(result["premium_pct"], 1)
        self.assertFalse(result["current"])
        self.assertIsNone(ET.nav_observation(nav, daily, "2026-08-27")["premium_pct"])

    def test_orthogonal_trend_value_and_reversion(self):
        base = {"strong_trend": True, "bias20_pct": 1, "bias_atr": .5, "bias_percentile": 60,
                "amount20_yi": .5, "reclaimed": True, "stabilizing": True}
        decide = lambda changes={}, stage="MARKUP", fresh=True: ET.technical_decision({**base, **changes}, stage, fresh)
        self.assertEqual(decide()["tier"], "优先研究")
        hot = decide({"bias20_pct": 8, "bias_atr": 3, "bias_percentile": 95})
        self.assertTrue(hot["strong_trend"] and hot["overheated"])
        self.assertEqual(hot["tier"], "过热不追")
        cold = decide({"bias20_pct": -12, "bias_atr": -3, "bias_percentile": 2, "strong_trend": False, "stabilizing": False})
        self.assertTrue(cold["oversold"])
        self.assertEqual(cold["tier"], "超调待确认")
        self.assertNotEqual(decide(stage="DISTRIBUTION_RISK")["tier"], "优先研究")
        self.assertFalse(decide({"amount20_yi": .499})["eligible"])
        self.assertFalse(decide(fresh=False)["eligible"])

    def test_stale_data_cannot_remain_priority(self):
        data = ET.analyze_etf(self.item, self.daily.iloc[:-1], self.adj, pd.DataFrame(), self.end)
        self.assertFalse(data["meta"]["fresh"])
        self.assertFalse(data["decision"]["eligible"])

    def test_calendar_and_cache_idempotence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ET.merge_csv(root / "calendar.csv", pd.DataFrame({"cal_date": ["20260904", "20260905", "20260906"], "is_open": [1, 0, 0]}), ["cal_date"])
            self.assertEqual(ET.resolve_session("2026-09-06", root), "2026-09-04")
            with self.assertRaises(ValueError):
                ET.resolve_session("2026-09-07", root)
            pro = Mock()
            pro.fund_daily.return_value = self.daily
            pro.fund_adj.return_value = self.adj
            pro.fund_nav.return_value = pd.DataFrame({"nav_date": ["20260831"], "ann_date": ["20260901"], "unit_nav": [1]})
            self.assertEqual(ET.refresh_one(self.item, self.end, pro, root), [])
            first = (root / "588170.SH.daily.csv").read_bytes()
            ET.refresh_one(self.item, self.end, pro, root)
            self.assertEqual(pro.fund_daily.call_count, 1)
            self.assertEqual(pro.fund_adj.call_count, 1)
            self.assertEqual(pro.fund_nav.call_count, 1)
            self.assertEqual(first, (root / "588170.SH.daily.csv").read_bytes())
            ET.refresh_one(self.item, self.end, pro, root, force=True)
            self.assertEqual(pro.fund_daily.call_count, 2)
            self.assertEqual(first, (root / "588170.SH.daily.csv").read_bytes())

    def test_failed_refresh_preserves_cache_and_redacts_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "588170.SH.daily.csv"
            ET.merge_csv(path, self.daily, ["trade_date"])
            before = path.read_bytes()
            pro = Mock()
            for name in ("fund_daily", "fund_adj", "fund_nav"):
                getattr(pro, name).side_effect = RuntimeError("SECRET_TOKEN")
            errors = ET.refresh_one(self.item, self.end, pro, root)
            self.assertEqual(len(errors), 3)
            self.assertNotIn("SECRET", str(errors))
            self.assertEqual(before, path.read_bytes())

    def test_report_dates_routes_and_no_get_fetch(self):
        import mainwave_visual_app as visual
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ET.write_json(root / "20260831" / "summary.json", {"as_of": "2026-08-31", "records": []})
            ET.write_json(root / "20260904" / "summary.json", {"as_of": "2026-09-04", "records": []})
            ET.write_json(root / "20260831" / "588170.SH.json", {"meta": {"report_date": "2026-08-31"}})
            self.assertEqual(ET.watch_list("2026-09-01", root)["as_of"], "2026-08-31")
            with self.assertRaises(FileNotFoundError):
                ET.report_path("2026-08-30", root)
            with patch.object(ET, "provider", side_effect=AssertionError("GET must be offline")), \
                 patch.object(ET, "report_path", return_value=root / "20260831" / "summary.json"):
                client = visual.app.test_client()
                self.assertEqual(client.get("/api/etfs?date=2026-08-31").status_code, 200)
                self.assertEqual(client.get("/api/etfs/588170?date=2026-08-31").status_code, 200)
                self.assertEqual(client.get("/api/etfs/600354").status_code, 400)
                self.assertEqual(client.post("/api/etfs").status_code, 405)
            page = visual.app.test_client().get("/").get_data(as_text=True)
            self.assertIn('id="etfSource"', page)
            self.assertIn("mainwave_etf.js", page)


if __name__ == "__main__":
    unittest.main()
