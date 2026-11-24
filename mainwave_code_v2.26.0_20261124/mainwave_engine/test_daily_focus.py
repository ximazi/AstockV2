import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from . import daily_focus as F


DAY = "2026-11-19"
RED = {"state": "RED", "trend_review_allowed": False, "data_status": "FRESH"}


def r1(symbol, rank, bias=1.6, **extra):
    return {"symbol": symbol, "price_date": DAY, "participation_rank": rank,
            "route_ids": "R1", "source_record": {"route_status": "WAIT_ACCEPTANCE", "route_bias_atr": bias},
            "features": {"close": 33.18}, **extra}


def first_board(symbol, **extra):
    return {"symbol": symbol, "name": symbol, "priority": "PRIOR_P5_FIRST_BOARD", "one_price": False,
            "evidence": [{"date": "2026-11-17"}], "first_board_low": 18.41, "first_board_close": 20.82, **extra}


class DailyFocusTests(unittest.TestCase):
    def test_p5_hotspot_position_and_wind_are_not_one_score_or_permission(self):
        one_price = {**first_board("sh_600001"), "one_price": True}
        facts = {"p5_limit_up": {"focus_count": 2, "rows": [one_price, first_board("sh_600002")]},
                 "ladder": {"max_height": 3}, "themes": {"top_by_limit_share": [{"theme": "材料", "members": 77, "limit_up": 5}]}}
        leader = {"symbol": "sh_600003", "height": 3, "close": 25.19, "themes": ["材料"]}
        hot = {"market_leaders": [leader["symbol"]], "rows": [leader]}
        records = [r1("sh_600004", 1, bias=4, paper_plan_eligible=True), r1("sh_600005", 2), r1("sh_600006", 3, theme="材料")]
        result = F.build(DAY, {"market": RED, "records": records}, facts, hot, {}, {})
        self.assertEqual([r["symbol"] for r in result["rows"]], ["sh_600002", "sh_600006", "sh_600003"])
        self.assertEqual([r["role"] for r in result["rows"]], ["P5首板关注", "R1结构关注", "热点风向标"])
        self.assertEqual(result["formal_review_count"], 0)
        self.assertIn("下一步", F.render(result))
        self.assertFalse(result["orders_enabled"])

    def test_formal_ready_is_prioritized_and_symbol_not_duplicated(self):
        market = {**RED, "state": "GREEN", "trend_review_allowed": True}
        records = [r1("sh_600001", 1, paper_plan_eligible=True), r1("sh_600002", 2, paper_plan_eligible=True)]
        facts = {"p5_limit_up": {"rows": [first_board("sh_600001")]}}
        result = F.build(DAY, {"market": market, "records": records}, facts, {}, {}, {})
        self.assertEqual(result["formal_review_count"], 2)
        self.assertEqual([r["role"] for r in result["rows"]], ["正式计划复核", "正式计划复核"])

    def test_second_structure_review_can_cover_another_hot_theme(self):
        facts = {"themes": {"top_by_limit_share": [{"theme": "材料"}, {"theme": "封装"}]}}
        records = [r1("sh_600001", 1, theme="材料"), r1("sh_600002", 2, theme="材料"),
                   r1("sh_600003", 3, theme="封装")]
        result = F.build(DAY, {"market": RED, "records": records}, facts, {}, {}, {})
        self.assertEqual([r["symbol"] for r in result["rows"]], ["sh_600001", "sh_600003"])

    def test_s4_fallback_and_no_forced_fill(self):
        s4 = {"recommended": [{"symbol": "sz_001328", "name": "登康口腔", "close": 36.3}], "buy_advised": False}
        result = F.build(DAY, {"market": RED, "records": [r1("sh_600001", 1, state="RISK_WATCH")]}, {}, {}, {}, s4)
        self.assertEqual([r["role"] for r in result["rows"]], ["S4形态关注"])
        self.assertIn("不允许新开仓", F.render(result))
        self.assertIn("继续等待", F.render(F.build(DAY, {}, {}, {}, {}, {})))

    def test_failed_step_and_wrong_report_date_disqualify_leftovers(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)

            def write(name, value):
                path = folder / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding="utf-8")

            write("mainwave_run_status.json", {"as_of": DAY, "status": "SUCCESS", "steps": [
                {"name": "fupan", "exit_code": 1}, {"name": "eod_review", "exit_code": 0}]})
            for step in ("fupan", "eod_review"):
                write(f"{step}/status.json", {"as_of": DAY, "status": "PARTIAL", "version": "v001"})
            write("eod_review/v001/facts.json", {"as_of": "2026-11-18", "p5_limit_up": {"rows": [first_board("sh_600001")]}})
            write("fupan/v001/hotspot_core.json", {"as_of": DAY, "rows": [{"symbol": "sh_600002"}]})
            write("s4_opportunities.json", {"as_of": DAY, "recommended": [{"symbol": "sz_001328"}]})
            result = F.load(DAY, folder, allow_s4=False)
            self.assertEqual(result["rows"], [])
            self.assertEqual(result["missing"], ["交易闭环", "盘后事实", "热点榜", "S4"])
            write("mainwave_run_status.json", {"as_of": DAY, "status": "FAILED"})
            with self.assertRaises(ValueError):
                F.load(DAY, folder)
