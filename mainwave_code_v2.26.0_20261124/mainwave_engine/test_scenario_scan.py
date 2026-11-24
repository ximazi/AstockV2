import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from mainwave_engine import scenario_scan as SC


def write_bars(root, sym, day, low=9.5, st=0):
    days = pd.bdate_range(end=day, periods=5).strftime("%Y-%m-%d")
    pd.DataFrame({"date": days, "open": 10.0, "high": 10.5, "low": low, "close": 10.2, "preclose": 10.0, "volume": 1000,
                  "amount": 1e7, "tradestatus": 1, "isST": st}).to_csv(Path(root) / f"{sym}_daily.csv", index=False)


class CoreEventTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        folder = self.root / "out" / "20260924"
        folder.mkdir(parents=True)
        board = [{"symbol": "sh_600001", "name": "甲", "entry": "BREAKOUT", "close": 10.2, "pct_chg": 2.0, "problems": [], "band": [10.0, 10.45]},
                 {"symbol": "sh_600002", "name": "乙", "entry": "REACCELERATION", "close": 10.2, "pct_chg": 3.0, "problems": []},
                 {"symbol": "sz_300001", "name": "创", "entry": "BREAKOUT", "close": 10.2, "pct_chg": 5.0, "problems": []},
                 {"symbol": "sh_600003", "name": "丙", "entry": "BREAKOUT", "close": 10.2, "pct_chg": 1.0, "problems": []},
                 {"symbol": "sh_600004", "name": "丁", "entry": "REACCELERATION", "close": 10.2, "pct_chg": 0.7, "problems": []}]
        (folder / "strong_trend_shadow.json").write_text(json.dumps({"boards": {"attention": board}}), encoding="utf-8")
        write_bars(self.root, "sh_600001", "2026-09-24")
        write_bars(self.root, "sh_600002", "2026-09-24")
        write_bars(self.root, "sh_600003", "2026-09-24", st=1)
        write_bars(self.root, "sh_600004", "2026-09-24")

    def events(self, kind):
        with patch.object(SC.SA, "OUT_ROOT", self.root / "out"):
            return SC.core_events(kind, "2026-09-24", self.root, {}, {})

    def test_s1_is_main_board_breakouts_with_the_p3_denial_at_the_signal_low(self):
        rows = self.events("s1")
        self.assertEqual([r["symbol"] for r in rows], ["sh_600001"], "只取主板、非 ST（当日记录）的突破")
        self.assertIn("9.50", rows[0]["plan"]["denial"])
        self.assertEqual(rows[0]["plan"]["stop"], 9.18)
        self.assertIn("10.45", rows[0]["plan"]["price_cap"], "显示核心计划自带的买价上限")

    def test_s2_is_reacceleration_with_the_p3_denial(self):
        rows = self.events("s2")
        self.assertEqual([r["symbol"] for r in rows], ["sh_600002"], "当天只涨 0.7% 的防御股不算再转强")
        self.assertIn("9.50", rows[0]["plan"]["denial"])
        self.assertEqual(rows[0]["plan"]["price_cap"], "", "影子没给区间时不编造上限")

    def test_missing_shadow_is_reported_not_empty(self):
        with patch.object(SC.SA, "OUT_ROOT", self.root / "absent"):
            self.assertIsNone(SC.core_events("s1", "2026-09-24", self.root, {}, {}))


class BottomReversalTests(unittest.TestCase):
    def test_a_flat_series_is_not_a_bottom_reversal(self):
        n = 300
        f = pd.DataFrame({"open": 10.0, "high": 10.1, "low": 9.9, "close": np.full(n, 10.0), "volume": 1000.0})
        self.assertIsNone(SC.bottom_reversal(f))


class ExportTests(unittest.TestCase):
    def test_export_merges_s1_to_s5_and_lists_missing_scans(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            plan = {"stop": 9.0, "tp1": 11.0, "tp2": 12.0, "denial": "x"}
            (folder / "s1_observations.json").write_text(json.dumps({"observations": [
                {"symbol": "sh_600001", "name": "甲", "plan": plan, "close_at_limit": True}, {"symbol": "sh_600005", "name": "戊", "plan": plan}]}), encoding="utf-8")
            (folder / "s4_opportunities.json").write_text(json.dumps({
                "recommended": [{"symbol": "sz_002003", "name": "伟星股份", "plan": plan, "lead": 0.47}],
                "excluded": [{"symbol": "sh_605177", "name": "东亚药业", "plan": plan, "excluded": ["R1 顶背离"]}]}), encoding="utf-8")
            path, n, missing = SC.export("2026-09-24", folder)
            self.assertEqual(n, 4)
            self.assertEqual(missing, ["S2 调整后再转强", "S3 平台突破", "S5 底部反转"])
            with open(path, encoding="utf-8-sig") as fh:
                rows = list(csv.reader(fh))
            self.assertEqual(rows[0], SC.EXPORT_COLUMNS)
            self.assertEqual([(r[3], r[2]) for r in rows[1:5]], [("600005", "研究观察"), ("600001", "研究观察（涨停，次日可能买不到）"),
                                                                ("002003", "推荐"), ("605177", "排除：R1 顶背离")], "涨停买不到的排在后面单列")
            self.assertIn("未生成", rows[-1][2])

    def test_export_endpoint_downloads_the_csv(self):
        import mainwave_visual_app as web
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(web, "OUT_ROOT", Path(tmp)), web.app.test_client() as client:
                r = client.get("/api/scenario-export?date=2026-09-24")
                self.assertEqual(r.status_code, 200)
                self.assertIn("attachment", r.headers["Content-Disposition"])
                self.assertIn("scenario_opportunities_20260924.csv", r.headers["Content-Disposition"])
                r.close()
                self.assertEqual(client.get("/api/scenario-export?date=bad").status_code, 400)


if __name__ == "__main__":
    unittest.main()
