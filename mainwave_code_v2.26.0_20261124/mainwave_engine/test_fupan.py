"""Daily review V2 (fupan): contract, packet, render, builder pieces, analysis merge and the read-only web door."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from . import eod_review as R
from .fupan import ASSETS, ai_packet, analyzer as AZ, build as B
from .fupan.contract import validate
from .fupan.render import render, render_html


def example():
    return json.loads((ASSETS / "report.example.json").read_text(encoding="utf-8"))


def blank():
    return json.loads((ASSETS / "report.blank.json").read_text(encoding="utf-8"))


class ContractTests(unittest.TestCase):
    """The template package's own contract tests, run against the ported validator."""

    def test_example_and_blank_valid(self):
        self.assertEqual(validate(example()), [])
        self.assertEqual(validate(blank()), [])

    def test_no_order_authority(self):
        d = example()
        d["meta"]["order_enabled"] = True
        self.assertTrue(validate(d))

    def test_missing_pinned(self):
        d = example()
        d["stocks"] = [x for x in d["stocks"] if x["symbol"] != "sz_000863"]
        self.assertTrue(any("必保留" in x for x in validate(d)))

    def test_no_false_ready_in_historical_demo(self):
        d = example()
        d["stocks"][0]["status"] = "PLAN_READY"
        self.assertTrue(any("历史演示" in x or "缺plan" in x for x in validate(d)))

    def test_no_future_history(self):
        d = example()
        d["history"][0]["date"] = "2026-09-22"
        self.assertTrue(any("复盘日之后" in x for x in validate(d)))

    def test_unknown_evidence(self):
        d = example()
        d["claims"][0]["evidence_ids"].append("INVENTED")
        self.assertTrue(any("INVENTED" in x for x in validate(d)))

    def test_summary_must_reference_shared_claim(self):
        d = example()
        d["summary"]["claim_ids"].append("FAKE")
        self.assertTrue(validate(d))
        d = example()
        d["summary"]["headline"] = "编造的独立标题"
        self.assertTrue(validate(d))

    def test_keep_seven_steps_and_table_shape(self):
        d = example()
        d["sections"].reverse()
        self.assertTrue(any("七步" in x for x in validate(d)))
        d = example()
        d["sections"][0]["table"]["rows"][0].append("extra")
        self.assertTrue(any("列数" in x for x in validate(d)))

    def test_comparison_symmetry(self):
        d = example()
        d["comparisons"][0]["matched"] = True
        self.assertTrue(any("齐备" in x for x in validate(d)))

    def test_valid_synthetic_plan_and_expired_case(self):
        d = blank()
        s = copy.deepcopy(example()["stocks"][0])
        s.update(id="TEST", symbol="TEST", name="契约测试标的", missing=[], model_version="approved-test", status="PLAN_READY", support_ids=["TEST_SRC"])
        d["stocks"] = [s]
        d["meta"].update(mode="FROZEN_EOD", trade_date="2026-09-21", target_date="2026-09-22", market_as_of="2026-09-21T15:00:00+08:00",
                         knowledge_as_of="2026-09-21T23:00:00+08:00", source_status="VERIFIED", required_symbols=["TEST"])
        d["sources"].append({"id": "TEST_SRC", "title": "合成测试，不是行情", "origin": "USER_DATA", "source_path": "TEST", "locator": "TEST",
                             "snapshot_sha256": "0" * 64, "available_at": "2026-09-21T22:00:00+08:00", "content": "测试"})
        s["plan"] = {"id": "TEST_PLAN", "strategy_version": "approved-test", "approval_status": "APPROVED", "signal_condition": "测试条件",
                     "trigger_state": "PENDING", "entry_min": 10, "entry_max": 10.1, "invalidation_price": 9.8, "exit_rule": "测试退出",
                     "max_loss_cash": 100, "budget_status": "PASS", "valid_from": "2026-09-22T09:30:00+08:00", "expires_at": "2026-09-22T14:30:00+08:00",
                     "quote_as_of": None, "quote_fresh": False, "source_ids": ["TEST_SRC"], "hard_blocks": []}
        self.assertEqual(validate(d), [])
        s["plan"]["expires_at"] = "2026-09-20T14:30:00+08:00"
        self.assertTrue(any("时效" in x for x in validate(d)))


class PacketAndRenderTests(unittest.TestCase):
    def test_packet_covers_required_and_drops_claims(self):
        d = example()
        p = ai_packet.prepare(d)
        self.assertTrue(set(d["meta"]["required_symbols"]).issubset({s["symbol"] for s in p["stocks"]}))
        self.assertNotIn("claims", p)
        self.assertTrue(all("content" not in s for s in p["source_catalog"]))
        self.assertTrue(all(s["plan"] is None for s in p["stocks"]))
        self.assertFalse(p["scope_is_full_market_stock_screen"])
        st = ai_packet.stats(d, p)
        self.assertTrue(st["covered_required"])
        self.assertIn("review.ai.input.v2", ai_packet.prompt_text(p))

    def test_script_escape_and_single_file(self):
        d = blank()
        d["claims"][0]["conclusion"] = "</script><script>window.__evil=1</script>"
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "out.html"
            render(d, out)
            text = out.read_text(encoding="utf-8")
        self.assertNotIn("</script><script>window.__evil", text)
        self.assertIn("\\u003c/script", text)
        for marker in ("__REPORT_SCHEMA__", "__REPORT_DATA__", "/*__REPORT_CSS__*/", "/*__REPORT_JS__*/"):
            self.assertNotIn(marker, text)

    def test_render_refuses_invalid(self):
        d = example()
        d["meta"]["order_enabled"] = True
        with self.assertRaises(ValueError):
            render_html(d)


class MergeTests(unittest.TestCase):
    ANALYSIS = {"section_assessments": [{"section_id": "01", "conclusion": "AI 结论", "interpretation": "AI 解释", "support_ids": ["M:20260921", "INVENTED"],
                                         "contrary_ids": ["H:5D"], "unknowns": ["盘中先后未知"]}],
                "stock_assessments": [{"symbol": "sz_301520", "opportunity": "AI 机会描述", "strongest_counterevidence": "AI 反证",
                                       "mode_assessments": [{"mode": "STRONG_CONFIRM", "result": "INSUFFICIENT", "reason": "缺分钟", "support_ids": ["S:sz_301520"]}]}],
                "pairwise_comparisons": [{"symbols": ["sz_301520", "sh_603259"], "matched": True, "missing": ["同期限"], "conclusion": "不下结论", "support_ids": []}],
                "bounded_data_requests": [{"symbols": ["sz_301520"], "priority": "P0", "claim_to_test": "封板顺序", "fields": ["分钟"], "max_rows": 60}],
                "unresolved_conflicts": ["容量与短线优先级冲突"]}

    def test_merge_is_interpretation_only(self):
        d = example()
        before = {s["symbol"]: (s["status"], s["plan"], s["metrics"]) for s in d["stocks"]}
        d["issues"].append({"id": "Q_NO_AI", "severity": "WARN", "title": "AI 解读未合并", "detail": "", "impact": "", "evidence_ids": []})
        result = ai_packet.merge_analysis(d, self.ANALYSIS)
        self.assertEqual(result["dropped_unknown_source_ids"], ["INVENTED"])
        c01 = next(c for c in d["claims"] if c["id"] == "C01")
        self.assertEqual((c01["type"], c01["conclusion"], c01["interpretation"]), ("INTERPRETATION", "AI：AI 结论", "AI：AI 解释"))
        self.assertIn("盘中先后未知", c01["contrary"])
        self.assertEqual({s["symbol"]: (s["status"], s["plan"], s["metrics"]) for s in d["stocks"]}, before)
        self.assertFalse(d["comparisons"][-1]["matched"])  # missing present → cannot claim matched
        self.assertEqual(d["requests"][-1]["priority"], "P0")
        self.assertTrue(any(q["id"] == "AI_Q01" for q in d["issues"]))
        self.assertFalse(any(q["id"] == "Q_NO_AI" for q in d["issues"]))
        self.assertEqual(validate(d), [])

    def test_persist_versions_and_apply_analysis(self):
        d = example()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            p1 = B.persist("2026-09-21", copy.deepcopy(d), {"input_hash": "h1", "inputs": {"x": 1}, "merged": None, "live": False}, root)
            p2 = B.persist("2026-09-21", copy.deepcopy(d), {"input_hash": "h1", "inputs": {"x": 1}, "merged": None, "live": False}, root)
            self.assertEqual((p1["version"], p2["version"], p2["reused"]), ("v001", "v001", True))
            for name in ("report.json", "report.html", "ai_packet.json", "ai_prompt.md", "packet_stats.json", "inputs.json"):
                self.assertTrue((Path(p1["path"]) / name).exists(), name)
            p3 = B.apply_analysis("2026-09-21", self.ANALYSIS, root)
            self.assertEqual((p3["version"], p3["analysis_merged"]), ("v002", True))
            merged = json.loads((Path(p3["path"]) / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(merged["meta"]["report_id"], "fupan-2026-09-21-v002")
            self.assertEqual(next(c for c in merged["claims"] if c["id"] == "C01")["type"], "INTERPRETATION")
            self.assertEqual(B.latest_report(root, "2026-09-21")["meta"]["report_id"], "fupan-2026-09-21-v002")


def frame(days, closes, limit_up_days=(), amount=1e8):
    rows = []
    prev = closes[0]
    for d, c in zip(days, closes):
        up = round(prev * 1.1, 2)
        close = up if d in limit_up_days else c
        rows.append({"date": d, "open": prev * 1.01, "high": max(close, prev * 1.02), "low": min(close, prev) * 0.99, "close": close, "preclose": prev,
                     "volume": 1000, "amount": amount, "turnover": 1.0, "tradestatus": 1, "isST": 0, "adjustflag": 3})
        prev = close
    return R.annotate(pd.DataFrame(rows), "MAIN")


class BuilderPieceTests(unittest.TestCase):
    DAYS = [f"2026-09-{d:02d}" for d in (1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 21)]

    def test_stock_metrics_from_bars(self):
        f = frame(self.DAYS, [10 + i * 0.1 for i in range(len(self.DAYS))])
        m = B.stock_metrics({"frame": f})
        last = f.iloc[-1]
        self.assertAlmostEqual(m["change_pct"], round((last.close / last.preclose - 1) * 100, 2))
        self.assertEqual(m["amount_yi"], 1.0)
        self.assertIsNone(m["return20_pct"])  # fewer than 21 bars: unknown, not zero
        self.assertEqual(B.stock_metrics(None)["change_pct"], None)

    def test_pool_history_uses_previous_day_limit_ups(self):
        rows = {"a": {"frame": frame(self.DAYS, [10.0] * len(self.DAYS), limit_up_days=("2026-09-17",))},
                "b": {"frame": frame(self.DAYS, [20.0] * len(self.DAYS))}}
        hist = B.pool_history(rows, self.DAYS, "2026-09-21", window=5)
        by = {h["date"]: h for h in hist}
        self.assertEqual([h["date"] for h in hist], self.DAYS[-5:])
        self.assertEqual(by["2026-09-17"]["limit_up"], 1)
        self.assertNotEqual(by["2026-09-18"]["pool_close_median"], 0.0)  # 'a' was in yesterday's pool on 09-18
        self.assertEqual(by["2026-09-16"]["pool_close_median"], 0.0)

    def test_next_session_and_plan_shape(self):
        self.assertEqual(B.next_session(self.DAYS, "2026-09-18"), "2026-09-21")
        self.assertEqual(B.next_session(self.DAYS, "2026-09-21"), "2026-09-22")
        self.assertEqual(B.next_session(self.DAYS, "2026-09-25"), "2026-09-28")  # weekend skipped
        payload = {"id": "P1", "ranking_version": "rank_v1", "exit_version": "exit_v1", "lower_cents": 1131, "limit_cents": 1146, "stop_cents": 1023,
                   "quantity": 8000, "model": "NEXT_OPEN_FIXED_CAP_PAPER_ONLY", "exit_rules": {"soft": "软保护", "structure": "结构退出"}}
        plan = B.plan_from_paper(payload, {"risk_cents": 993299, "status": "OPEN"}, 0, "2026-09-21", "2026-09-22", ["S:x"], [])
        self.assertEqual((plan["entry_min"], plan["entry_max"], plan["invalidation_price"], plan["max_loss_cash"]), (11.31, 11.46, 10.23, 9932.99))
        self.assertEqual((plan["approval_status"], plan["budget_status"], plan["trigger_state"]), ("APPROVED", "PASS", "PENDING"))
        self.assertIn("AUTO100W", plan["signal_condition"])
        self.assertEqual(plan["expires_at"][:10], "2026-09-22")
        # the window is the plan's own effective session: a later report day does not move it
        same = B.plan_from_paper(payload, {"risk_cents": 993299, "status": "OPEN"}, 0, "2026-09-23", "2026-09-22", ["S:x"], [])
        self.assertEqual((same["valid_from"], same["expires_at"]), (plan["valid_from"], plan["expires_at"]))
        filled = B.plan_from_paper(payload, {"risk_cents": 993299, "status": "FILLED"}, 8000, "2026-09-22", "2026-09-22", ["S:x"], [])
        self.assertEqual(filled["trigger_state"], "CONFIRMED")
        dead = B.plan_from_paper(payload, {"risk_cents": 993299, "status": "CANCELLED"}, 0, "2026-09-23", "2026-09-22", ["S:x"], [])
        self.assertEqual(dead["trigger_state"], "FAILED")
        draft = B.plan_from_paper(payload, None, 0, "2026-09-21", "2026-09-22", ["S:x"], ["BLOCK"])
        self.assertEqual((draft["approval_status"], draft["budget_status"], draft["hard_blocks"]), ("DRAFT", "UNKNOWN", ["BLOCK"]))

    def test_stock_card_status_rules(self):
        f = frame(self.DAYS, [10.0] * len(self.DAYS))
        src = B.Sources()
        tracking = {"pinned": [{"symbol": "sz_1", "name": "指定股", "reason": "跟踪", "mode": "STRONG_CONFIRM", "plan_id": None}]}
        plan = {"payload": {"id": "P1", "ranking_version": "rank_v1", "exit_version": "e", "lower_cents": 1131, "limit_cents": 1146, "stop_cents": 1023, "quantity": 100, "model": "M", "exit_rules": {}, "entry_day": "2026-09-21"},
                "order": {"risk_cents": 100, "status": "OPEN"}, "fills": 0, "open_position": 0, "entry_day": "2026-09-21", "account_id": "AUTO100W"}
        live = B.stock_card("sz_1", "指定股", {"frame": f}, {"state": "PLAN_REVIEW", "mode": "TREND_PULLBACK", "blockers": [], "participation_rank": 1, "next_condition": "收复"},
                            {"status": "ORDER_APPROVED"}, plan, None, None, tracking, "2026-09-21", "2026-09-22", True, src, [])
        self.assertEqual((live["status"], live["mode"], live["pinned"], live["plan"]["id"]), ("PLAN_READY", "TREND_PULLBACK", True, "P1"))
        rebuilt = B.stock_card("sz_1", "指定股", {"frame": f}, {"state": "PLAN_REVIEW", "mode": "TREND_PULLBACK", "blockers": [], "participation_rank": 1},
                               {"status": "ORDER_APPROVED"}, plan, None, None, tracking, "2026-09-21", "2026-09-22", False, src, [])
        self.assertEqual(rebuilt["status"], "WAIT_TRIGGER")  # a reconstruction never claims a ready plan
        # the same plan seen from a later report: its own window (09-22) has passed without a fill -> EXPIRED, never re-dated
        later = B.stock_card("sz_1", "指定股", {"frame": f}, {"state": "PLAN_REVIEW", "mode": "TREND_PULLBACK", "blockers": [], "participation_rank": 1},
                             {"status": "ORDER_APPROVED"}, plan, None, None, tracking, "2026-09-22", "2026-09-23", True, src, [])
        self.assertEqual(later["status"], "EXPIRED")
        self.assertEqual(later["plan"]["expires_at"][:10], "2026-09-22")
        held = B.stock_card("sz_1", "指定股", {"frame": f}, None, None, dict(plan, open_position=100, order={"risk_cents": 100, "status": "FILLED"}, fills=100),
                            None, None, tracking, "2026-09-22", "2026-09-23", True, src, [])
        self.assertEqual(held["status"], "WATCH")
        self.assertIn("已成交持仓", held["contrary"])
        gap = B.stock_card("sz_1", "指定股", None, None, None, None, None, None, tracking, "2026-09-21", "2026-09-22", True, src, [])
        self.assertEqual((gap["status"], gap["plan"]), ("DATA_GAP", None))
        self.assertIn("当日可核验行情", gap["missing"])
        watch = B.stock_card("sz_1", "指定股", {"frame": f}, None, None, None, None, None, tracking, "2026-09-21", "2026-09-22", True, src, [])
        self.assertEqual((watch["status"], watch["mode"]), ("WATCH", "STRONG_CONFIRM"))
        self.assertTrue(any("原计划" in m for m in watch["missing"]))
        self.assertEqual(len(src.items), 1)  # one bar source per symbol, deduplicated

    def test_build_refuses_without_facts(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError) as ctx:
                B.build("2026-09-21", Path(temp), Path(temp), Path(temp) / "none.json", rows={})
            self.assertTrue(str(ctx.exception).startswith("DATA_WAIT:"))


class WebTests(unittest.TestCase):
    def test_read_only_door(self):
        from flask import Flask
        from .trading import web
        app = Flask(__name__)
        app.register_blueprint(web.bp)
        with tempfile.TemporaryDirectory() as temp, patch.object(web.store, "ROOT", Path(temp)):
            client = app.test_client()
            self.assertEqual(client.get("/api/trading/fupan?as_of=2026-09-21").get_json()["status"], "NO_REPORT")
            self.assertEqual(client.get("/api/trading/fupan/report.html?as_of=2026-09-21").status_code, 404)
            B.persist("2026-09-21", example(), {"input_hash": "h", "inputs": {}, "merged": None, "live": False}, Path(temp) / "results/mainwave_engine")
            r = client.get("/api/trading/fupan?as_of=2026-09-21").get_json()
            self.assertEqual((r["status"], r["pointer"]["version"], r["meta"]["order_enabled"]), ("READY", "v001", False))
            self.assertIn("sz_301520", [s["symbol"] for s in r["stocks"]])
            html = client.get("/api/trading/fupan/report.html?as_of=2026-09-21")
            self.assertEqual(html.status_code, 200)
            self.assertIn("review.report.v2", html.get_data(as_text=True))
            merged = client.post("/api/trading/fupan/analysis", json={"as_of": "2026-09-21", "analysis": MergeTests.ANALYSIS}).get_json()
            self.assertEqual((merged["status"], merged["pointer"]["version"]), ("MERGED", "v002"))
            self.assertEqual(client.post("/api/trading/fupan/analysis", json={"as_of": "2026-09-21", "analysis": "text"}).status_code, 400)


@unittest.skipUnless((Path("results/mainwave_engine/20260921/fupan/latest.json")).exists(), "需要真实 09-21 复盘产物")
class RealArtefactTests(unittest.TestCase):
    def test_persisted_report_meets_contract(self):
        pointer = json.loads(Path("results/mainwave_engine/20260921/fupan/latest.json").read_text(encoding="utf-8"))
        report = json.loads((Path(pointer["path"]) / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(validate(report), [])
        self.assertFalse(report["meta"]["order_enabled"])
        self.assertTrue(set(report["meta"]["required_symbols"]) <= {s["symbol"] for s in report["stocks"]})


if __name__ == "__main__":
    unittest.main(verbosity=2)


class OpportunityTests(unittest.TestCase):
    DAYS = BuilderPieceTests.DAYS

    def facts(self, heights=None, gap=(), breakdown=()):
        return {"ladder": {"heights": heights or {}},
                "themes": {"top_by_limit_up": [{"theme": "国企改革", "limit_up": 34}], "top_by_limit_share": [{"theme": "租售同权", "limit_up": 4}]},
                "loss_effects": {"events": {"gap_up_reversal": {"symbols": list(gap)}, "breakdown_20d": {"symbols": list(breakdown)}}}}

    def test_score_names_every_point_and_stays_bounded(self):
        f = frame(self.DAYS, [10.0] * len(self.DAYS), limit_up_days=("2026-09-21",), amount=5e8)
        facts = self.facts(heights={"a": 3})
        leads = B.theme_leads(facts)
        o = B.opportunity("a", {"frame": f}, facts, {"a": {"concepts": ["国企改革", "融资融券"]}}, leads)
        labels = [lab for lab, _ in o["parts"]]
        self.assertTrue(any(lab.startswith("今日 +10") for lab in labels))
        self.assertIn("3 板", labels)
        self.assertTrue(any(lab.startswith("题材 国企改革") for lab in labels))
        self.assertEqual(o["score"], min(100, sum(p for _, p in o["parts"])))
        self.assertEqual(o["text"].count("；"), len(o["parts"]) - 1)
        penalised = B.opportunity("a", {"frame": f}, self.facts(heights={"a": 3}, gap=("a",), breakdown=("a",)), {}, leads)
        self.assertEqual(penalised["score"], max(0, o["score"] - 30 - 10))  # −15 −15, and the theme point is gone
        self.assertIsNone(B.opportunity("z", None, facts, {}, leads)["score"])

    def test_verdict_rule(self):
        def card(name, score, status="WATCH", plan=None, missing=()):
            return {"name": name, "_score": score, "status": status, "plan": plan, "mode": "TREND_PULLBACK", "missing": list(missing), "contrary": ""}
        plan = {"entry_min": 11.31, "entry_max": 11.46, "invalidation_price": 10.23}
        code, text = B.verdict([card("甲", 80, "PLAN_READY", plan), card("乙", 70, missing=["分钟数据"])])
        self.assertEqual(code, "✅")
        self.assertIn("✅ 有买点 1 个（已批准待触发（今日生效））：甲", text)
        self.assertIn("🟡 强势主线无买点（有观察价值，待确认）：乙（70 分），差：分钟数据", text)
        code, text = B.verdict([card("甲", 80, "WAIT_TRIGGER", plan), card("丙", 30)])
        self.assertEqual(code, "🟡")  # an approved plan awaiting its window is worth attention, never a buy point
        self.assertIn("已批准待触发（重建或未到生效日），不是当前买入许可", text)
        self.assertNotIn("✅", text)
        code, text = B.verdict([card("甲", 80, "EXPIRED", plan), card("丙", 30)])
        self.assertEqual(code, "❌")
        self.assertIn("⚪ 过期未成交（延期须新批准）：甲", text)
        code, text = B.verdict([card("乙", 70, missing=["分钟数据"]), card("丙", 30)])
        self.assertEqual(code, "🟡")
        self.assertTrue(text.startswith("🟡"))
        code, text = B.verdict([card("丙", 30), card("丁", None, "DATA_GAP")])
        self.assertEqual(code, "❌")
        self.assertIn("最高 30 分（丙）", text)
        self.assertEqual(B.verdict([])[0], "❌")

    def test_s4_verdict_is_capped_by_the_market(self):
        rec = [{"name": "宁科生物", "symbol": "sh_600165"}]
        ok = {"recommended": rec, "buy_advised": True, "market_risk": {"level": "NORMAL"}, "market_warning": {"level": "NORMAL", "indices": []}}
        self.assertEqual(B.s4_verdict(ok, "❌")[0], "🟡")
        warn = {**ok, "market_warning": {"level": "WARNING", "indices": [{"index": "中证500", "level": "WARNING"}]}}
        code, head, _ = B.s4_verdict(warn, "❌")
        self.assertEqual(code, "🟡")
        self.assertIn("短期回调预警（中证500）", head)
        critical = {**ok, "buy_advised": False, "market_risk": {"level": "CRITICAL"}}
        self.assertEqual(B.s4_verdict(critical, "✅")[0], "🟡")
        self.assertIn("大盘趋势临界", B.s4_verdict({**critical, "recommended": []}, "✅")[1])
        self.assertEqual(B.s4_verdict({**critical, "recommended": []}, "✅")[0], "🟡", "没有 S4 推荐时，观察榜的 ✅ 也降为 🟡")
        self.assertEqual(B.s4_verdict({**critical, "recommended": []}, "❌")[0], "❌")
        exit_ = {**ok, "buy_advised": False, "market_risk": {"level": "TREND_EXIT"}}
        self.assertEqual(B.s4_verdict(exit_, "✅")[0], "🔴")
        self.assertEqual(B.s4_verdict({}, "✅"), ("✅", "", ""))

    def test_readings_are_rule_based_not_placeholders(self):
        def metric(reading, today=1.0):
            return {"label": "x", "status": "OK", "reading": reading, "today": today, "median": 0.5, "percentile": 0.9}
        facts = {"sentiment": {"metrics": {k: metric("热") for k in B.SENT_KEYS}},
                 "yesterday_pool": {"n_valid": 77, "n_pool": 78}, "boards": {"counts": {"U": 1, "Z": 1, "D": 1}},
                 "ladder": {"by_height": {"1": 80, "5": 1}, "heights": {"sz_001216": 5, "sh_600001": 1}},
                 "themes": {"top_by_limit_up": [{"theme": "国企改革", "limit_up": 34, "breadth": {"value": 0.8, "numerator": 8, "denominator": 10}}],
                            "top_by_limit_share": [{"theme": "租售同权", "limit_up": 4, "members": 24}]},
                 "loss_effects": {"events": {"yesterday_pool_negative": {"count": 24}, "gap_up_reversal": {"count": 17}, "breakdown_20d": {"count": 27}}}}
        rd = B.readings(facts)
        self.assertEqual((rd["env"], rd["hot"], rd["cold"]), ("偏热", 8, 0))
        self.assertIn("环境支持按强势确认与趋势回踩", rd["env_conclusion"])
        self.assertEqual(rd["ladder_read"], "梯队有延续且在拔高")
        self.assertEqual(rd["leaders"], ["sz_001216（5 板）"])
        self.assertTrue(rd["risk_read"].startswith("亏钱效应未扩散"))
        facts["sentiment"]["metrics"] = {k: metric("冷") for k in B.SENT_KEYS}
        cold = B.readings(facts)
        self.assertEqual(cold["env"], "偏冷")
        self.assertTrue(cold["risk_read"].startswith("亏钱效应扩散"))
        self.assertEqual(cold["ladder_read"], "梯队断层，接力成本高")


@unittest.skipUnless((Path("results/mainwave_engine/20260921/fupan/latest.json")).exists(), "需要真实 09-21 复盘产物")
class RealOpportunityTests(unittest.TestCase):
    def test_opportunity_layer_is_filled(self):
        pointer = json.loads(Path("results/mainwave_engine/20260921/fupan/latest.json").read_text(encoding="utf-8"))
        report = json.loads((Path(pointer["path"]) / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["summary"]["claim_ids"][0], "C06")
        headline = next(c for c in report["claims"] if c["id"] == "C06")["title"]
        self.assertTrue(headline[0] in "✅🟡❌", headline)
        self.assertFalse(any("本地程序不生成解释" in c["interpretation"] for c in report["claims"]))
        labels = [s["opportunity_label"] for s in report["stocks"]]
        self.assertTrue(labels[0].startswith("机会 #1"), labels[:3])
        self.assertTrue(set(report["meta"]["required_symbols"]) <= {s["symbol"] for s in report["stocks"]})


class AnalyzerTests(unittest.TestCase):
    """The model is a transport behind a fake here: parse, contract check, cache, merge and failure paths."""

    @staticmethod
    def answer(packet, extra=None):
        symbols = [s["symbol"] for s in packet["stocks"]]
        src = packet["source_catalog"][0]["id"]
        data = {"as_of": packet["time_boundary"]["trade_date"], "objective_confirmation": "短线机会，原策略合约",
                "coverage": {"required_symbols": packet["required_symbols"], "covered_symbols": symbols, "missing_symbols": []},
                "section_assessments": [{"section_id": sid, "conclusion": f"S{sid} 结论", "support_ids": [src], "comparison": "同口径",
                                         "interpretation": f"S{sid} 解释", "contrary_ids": [], "unknowns": [], "next_check": "明日核对"} for sid in AZ.SECTION_IDS],
                "stock_assessments": [{"symbol": sym, "opportunity": f"{sym} 机会", "strongest_counterevidence": "反证",
                                       "mode_assessments": [{"mode": "STRONG_CONFIRM", "result": "UNKNOWN", "reason": "缺分钟", "support_ids": [src]}],
                                       "existing_plan_id": None, "next_check": "任务", "data_requests": []} for sym in symbols],
                "pairwise_comparisons": [], "complete_plan_ids": [], "bounded_data_requests": [], "unresolved_conflicts": []}
        data.update(extra or {})
        return data

    def seed(self, root):
        return B.persist("2026-09-21", example(), {"input_hash": "h", "inputs": {"x": 1}, "merged": None, "live": False}, root)

    def test_parse_picks_the_contract_object_among_several(self):
        text = '先给个摘要 {"note": "short"} 然后正文 {"section_assessments": [{"section_id": "01"}], "x": {"y": 1}} 完'
        self.assertEqual(AZ.parse(text)["x"], {"y": 1})
        self.assertEqual(AZ.parse('{"a": 1} {"a": 2, "b": 3}'), {"a": 2, "b": 3})

    def test_batches_merge_only_each_batch_own_stocks(self):
        packet = {"stocks": [{"symbol": f"s{i}"} for i in range(19)], "required_symbols": ["s0", "s18"]}
        groups = AZ.batches(packet)
        self.assertEqual([len(g) for g in groups], [8, 8, 3])
        everyone = [{"symbol": f"s{i}", "mode_assessments": []} for i in range(19)]
        parts = [{"as_of": "d", "section_assessments": [{"section_id": "01"}], "stock_assessments": everyone,
                  "bounded_data_requests": [{"request_id": "REQ1"}]},
                 {"section_assessments": [], "stock_assessments": everyone, "bounded_data_requests": [{"request_id": "REQ1", "x": 2}]},
                 {"section_assessments": [], "stock_assessments": everyone[16:]}]
        merged = AZ.merge_batches(parts, groups, packet)
        self.assertEqual([a["symbol"] for a in merged["stock_assessments"]], [f"s{i}" for i in range(19)])
        self.assertEqual(merged["section_assessments"], [{"section_id": "01"}])
        self.assertEqual([r["request_id"] for r in merged["bounded_data_requests"]], ["REQ1", "REQ1_B2"])
        self.assertEqual(merged["coverage"]["missing_symbols"], [])
        prompt = AZ.user_prompt({"time_boundary": {"trade_date": "d"}}, ["s8", "s9"], first=False)
        self.assertIn("只写这 2 只：s8、s9", prompt)
        self.assertIn("section_assessments 输出空数组", prompt)

    def test_parse_tolerates_fences_and_chatter(self):
        self.assertEqual(AZ.parse('好的，结果如下：\n```json\n{"a": 1}\n```\n以上。'), {"a": 1})
        self.assertEqual(AZ.parse('{"a": {"b": 2}}'), {"a": {"b": 2}})
        with self.assertRaises(ValueError):
            AZ.parse("没有对象")

    def test_check_contract(self):
        packet = ai_packet.prepare(example())
        self.assertEqual(AZ.check(self.answer(packet), packet), [])
        bad = self.answer(packet)
        bad["section_assessments"] = bad["section_assessments"][:3]
        bad["stock_assessments"].append({"symbol": "sz_999999", "mode_assessments": [{"mode": "OVERNIGHT", "result": "MAYBE"}]})
        problems = AZ.check(bad, packet)
        self.assertTrue(any("未覆盖" in p for p in problems))
        self.assertTrue(any("sz_999999" in p for p in problems))
        self.assertTrue(any("MAYBE" in p for p in problems))
        self.assertEqual(AZ.blocking(problems), problems)  # every contract problem blocks
        self.assertTrue(AZ.blocking(AZ.check({"foo": 1}, packet)))
        self.assertTrue(AZ.blocking(AZ.check("text", packet)))
        # the audit's counterexample: one section, required stock covered, no as_of -> refused
        partial = {"section_assessments": [{"section_id": "01"}], "stock_assessments": [{"symbol": s} for s in packet["required_symbols"]]}
        pr = AZ.check(partial, packet)
        self.assertTrue(any("未覆盖" in x for x in pr) and any("缺 as_of" in x for x in pr))
        self.assertTrue(AZ.blocking(pr))
        dup = self.answer(packet)
        dup["section_assessments"].append(dict(dup["section_assessments"][0]))
        self.assertTrue(any("重复" in x for x in AZ.check(dup, packet)))
        wrong_day = self.answer(packet, {"as_of": "2026-09-22T21:55:56+08:00"})
        self.assertTrue(any("as_of 与输入交易日不一致" in x for x in AZ.check(wrong_day, packet)))
        alien = self.answer(packet)
        alien["section_assessments"][0]["support_ids"] = ["NOT_IN_CATALOG"]
        self.assertTrue(any("事实包之外的证据ID" in x for x in AZ.check(alien, packet)))

    def test_run_merges_caches_and_reuses(self):
        calls = []
        def client(system, user, cfg):
            calls.append(len(user))
            packet = json.loads(user.split("```json\n", 1)[1].split("\n```", 1)[0])
            return "```json\n" + json.dumps(self.answer(packet), ensure_ascii=False) + "\n```", {"input_tokens": 10, "output_tokens": 5}
        cfg = {"provider": "fake", "model": "fake-1", "timeout": 5, "proxy": "", "base_url": ""}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.assertEqual(AZ.run("2026-09-21", root, cfg, client)["status"], "NO_REPORT")
            self.seed(root)
            first = AZ.run("2026-09-21", root, cfg, client)
            self.assertEqual((first["status"], first["pointer"]["version"], first["cached"], first["usage"]["input_tokens"]), ("MERGED", "v002", False, 10))
            merged = json.loads((Path(first["pointer"]["path"]) / "report.json").read_text(encoding="utf-8"))
            self.assertTrue(next(c for c in merged["claims"] if c["id"] == "C01")["conclusion"].startswith("AI："))
            self.assertIn("AI 目标确认", next(c for c in merged["claims"] if c["id"] == "C06")["comparison"])
            self.assertTrue((Path(first["pointer"]["path"]) / "analysis_meta.json").exists())
            self.assertEqual(validate(merged), [])
            again = AZ.run("2026-09-21", root, cfg, client)
            self.assertEqual((again["status"], len(calls)), ("ALREADY_MERGED", 1))
            # A rebuild with unchanged facts keeps the analysed version instead of starting a bare v003.
            kept = B.persist("2026-09-21", example(), {"input_hash": "h", "inputs": {"x": 1}, "merged": None, "live": False}, root)
            self.assertEqual((kept["version"], kept["reused"]), ("v002", True))

    def test_run_failure_and_rejection_keep_the_rule_version(self):
        cfg = {"provider": "fake", "model": "fake-1", "timeout": 5, "proxy": "", "base_url": ""}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.seed(root)
            def broken(system, user, cfg):
                raise RuntimeError("网络断了")
            failed = AZ.run("2026-09-21", root, cfg, broken)
            self.assertEqual((failed["status"], failed["error"]), ("FAILED", "网络断了"))
            rejected = AZ.run("2026-09-21", root, cfg, lambda s, u, c: ('{"hello": "world"}', {}))
            self.assertEqual(rejected["status"], "REJECTED")
            self.assertTrue(list((root / "20260921/fupan/analysis").glob("*.rejected.txt")))
            self.assertEqual(B.latest_report(root, "2026-09-21")["meta"]["report_id"], "fupan-2026-09-21-v001")
            self.assertEqual(AZ.run("2026-09-21", root, {"provider": "off", "model": "", "timeout": 5, "proxy": "", "base_url": ""})["status"], "SKIPPED_NO_PROVIDER")

    def test_config_detection(self):
        self.assertEqual(AZ.config({"FUPAN_ANALYZER": "off"})["provider"], "off")
        self.assertEqual(AZ.config({"ANTHROPIC_API_KEY": "k"})["provider"], "codex-cli")
        self.assertEqual(AZ.config({"OPENAI_API_KEY": "k", "FUPAN_MODEL": "m"})["model"], "m")
        self.assertEqual(AZ.config({"FUPAN_ANALYZER": "nonsense"})["provider"], "off")

    def test_provider_change_does_not_reuse_merged_legacy_analysis(self):
        cfg = {"provider": "legacy", "model": "m"}
        packet = ai_packet.prepare(example())
        client = lambda s, u, c: (json.dumps(self.answer(packet)), {})
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.seed(root)
            old = AZ.run("2026-09-21", root, cfg, client)
            new = AZ.run("2026-09-21", root, {**cfg, "provider": "codex-cli"}, client)
            self.assertEqual((old["status"], new["status"]), ("MERGED", "MERGED"))
            self.assertNotEqual(old["cache_key"], new["cache_key"])
            self.assertEqual(new["pointer"]["version"], "v003")

    def test_force_merge_starts_from_rule_version_and_packet_carries_local_priority(self):
        cfg = {"provider": "fake", "model": "fake-1", "timeout": 5, "proxy": "", "base_url": ""}
        calls = []
        def client(system, user, cfg):
            packet = json.loads(user.split("```json\n", 1)[1].split("\n```", 1)[0])
            self.assertTrue(all("local_priority" in s and "local_role" in s for s in packet["stocks"]))
            self.assertIn("06", [t["section"] for t in packet["local_tables"]])
            calls.append(1)
            return json.dumps(self.answer(packet, {"unresolved_conflicts": ["冲突甲" if len(calls) == 1 else "冲突乙"]}), ensure_ascii=False), {}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.seed(root)
            first = AZ.run("2026-09-21", root, cfg, client)
            second = AZ.run("2026-09-21", root, cfg, client, force=True)
            self.assertEqual((first["pointer"]["version"], second["pointer"]["version"]), ("v002", "v003"))
            merged = json.loads((Path(second["pointer"]["path"]) / "report.json").read_text(encoding="utf-8"))
            ai_issues = [(q["id"], q["detail"]) for q in merged["issues"] if q["id"].startswith("AI_")]
            self.assertEqual(ai_issues, [("AI_Q01", "冲突乙")])  # the second answer replaces the first, never stacks on it
            self.assertEqual(merged["claims"][0]["conclusion"].count("AI："), 1)
            self.assertEqual(second["pointer"]["rule_path"], first["pointer"]["rule_path"])
