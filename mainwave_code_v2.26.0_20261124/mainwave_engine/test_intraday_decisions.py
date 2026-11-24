from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from mainwave_engine import intraday_review as R, intraday_quotes as Q, intraday_decisions as ID


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.at = Q.stamp("2026-09-18 14:35")
        self.symbols = [f"sh_{600000 + i}" for i in range(12)]
        self.rows = [{"ts_code": Q.ts_code(s), "trade_time": str(self.at), "pre_close": 10., "open": 10.,
                      "high": 10.5, "low": 9.8, "close": 10.4 if i < 6 else 9.9, "vol": 1000000., "amount": 10100000.}
                     for i, s in enumerate(self.symbols)]
        self.bundle = {"raw": [{"api": "rt_k", "records": self.rows},
                               {"api": "rt_idx_k", "records": [{**self.rows[0], "ts_code": c, "close": 10.01} for c in Q.INDEX_CODES]}], "issues": []}
        self.membership = {"mapping": {s: {"name": f"Stock{i}", "concepts": ["液冷", "液冷", "融资融券"] if i < 6 else ["对照题材"]} for i, s in enumerate(self.symbols)},
                           "meta": {"status": "RICH_DYNAMIC", "as_of": "2026-09-17"}}
        self.baseline = {"market": {"data_status": "FRESH", "trend_review_allowed": True}, "records": [
            {"symbol": s, "structural_advice": "PRIORITY_REVIEW", "route_ids": "R2", "value_tier": "PRIORITY_REVIEW",
             "review_lower": 10., "review_upper": 10.5, "invalid": 9.5, "value_rank": 12-i} for i, s in enumerate(self.symbols)]}
        self.histories = {s: {"status": "READY", "as_of": "2026-09-17", "close": 10., "ema20": 10., "ema60": 10., "high20": 10.2, "volume5": 2000000.} for s in self.symbols}

    def report(self, scope="market"):
        return R.evaluate(self.bundle, self.symbols, "2026-09-17", self.baseline, [], {}, self.histories, {}, self.at, self.membership, scope)

    def complete_minutes(self):
        times = pd.date_range("2026-09-18 09:31", "2026-09-18 11:30", freq="min").union(pd.date_range("2026-09-18 13:01", "2026-09-18 14:35", freq="min"))
        for s in self.symbols[:6]:
            frame = pd.DataFrame({"ts_code": Q.ts_code(s), "time": times.astype(str), "open": 10., "high": 10.5, "low": 9.8, "close": 10.4, "vol": 100., "amount": 1010.})
            self.bundle["raw"].append({"api": "rt_min_daily", "records": frame.to_dict("records")})

    def test_theme_deduplicates_members_excludes_generic_and_reports_coverage(self):
        r = self.report()
        themes = {t["name"]: t for t in r["themes"]["records"]}
        self.assertNotIn("融资融券", themes)
        self.assertEqual(themes["液冷"]["fresh_members"], 6)
        self.assertEqual(themes["液冷"]["state"], "HOT")
        self.assertEqual(themes["对照题材"]["state"], "WEAK")
        self.assertEqual(r["candidates"]["paper_review_symbols"], [])

    def test_missing_or_old_membership_and_pool_never_claim_hot_theme(self):
        for scope, state in (("pool", "RICH"), ("market", "MISSING"), ("market", "STALE_RICH")):
            self.membership["meta"]["status"] = state
            r = self.report(scope)
            self.assertEqual(r["themes"]["status"], "DATA_WAIT")
            self.assertFalse(r["candidates"]["paper_review_symbols"])
        self.assertFalse(self.report("pool")["market"]["new_review_allowed"])

    def test_low_theme_coverage_cannot_be_hot(self):
        self.bundle["raw"][0]["records"] = self.rows[:3] + self.rows[6:]
        t = next(t for t in self.report()["themes"]["records"] if t["name"] == "液冷")
        self.assertEqual(t["coverage"], .5)
        self.assertEqual(t["state"], "DATA_WAIT")

    def test_single_heavy_leader_is_divergence_not_broad_resonance(self):
        self.rows[0].update(vol=100000000., amount=1010000000.)
        t = next(t for t in self.report()["themes"]["records"] if t["name"] == "液冷")
        self.assertEqual(t["state"], "SPLIT")

    def test_paper_candidates_need_confirmation_and_do_not_force_three(self):
        self.complete_minutes()
        report = self.report()
        self.assertEqual(len(report["candidates"]["paper_review_symbols"]), 2)
        self.assertLessEqual(len(report["candidates"]["focus_symbols"]), 5)
        self.assertFalse(report["orders_enabled"])
        for row in report["records"][:2]:
            self.assertTrue(row["paper_review_eligible"])
            self.assertFalse(row["blockers"])
            self.assertTrue(row["reasons"])

    def test_hot_theme_cannot_rescue_chase_invalid_or_missing_route(self):
        self.complete_minutes()
        self.baseline["records"][0]["review_upper"] = 10.2
        self.baseline["records"][1]["invalid"] = 10.45
        self.baseline["records"][2]["route_ids"] = ""
        rows = {r["symbol"]: r for r in self.report()["records"]}
        for symbol in self.symbols[:3]:
            self.assertFalse(rows[symbol]["paper_review_eligible"])
            self.assertTrue(rows[symbol]["blockers"])

    def test_bear_keeps_discoveries_but_no_paper_participation(self):
        self.complete_minutes()
        self.baseline["market"]["trend_review_allowed"] = False
        r = self.report()
        self.assertTrue(r["candidates"]["focus_symbols"])
        self.assertFalse(r["candidates"]["paper_review_symbols"])

    def test_severe_indices_survive_incomplete_market_data(self):
        self.bundle["raw"][0]["records"] = []
        self.bundle["raw"][1]["records"] = [{**r, "close": 9.8} for r in self.bundle["raw"][1]["records"][:2]]
        self.assertEqual(self.report()["market"]["state"], "RISK")
        self.assertEqual(self.report()["summary"]["status"], "PARTIAL")

    def test_pool_decline_does_not_claim_market_risk(self):
        self.bundle["raw"][0]["records"] = [{**r, "close": 9.9} for r in self.rows]
        self.assertEqual(self.report("pool")["market"]["state"], "UNKNOWN")

    def test_holding_hard_alert_never_cleared_and_context_can_tighten_hold(self):
        row = self.report()["records"][0]
        holding = {"symbol": row["symbol"], "status": "SELL_REVIEW", "pending_action": "SELL_REVIEW", "reason": "hard floor"}
        ID.holding_context(holding, row, {"state": "REVIEW_ONLY"})
        self.assertEqual(holding["status"], "SELL_REVIEW")
        row["quote"]["close"] = 10.
        holding.update(status="HOLD_OBSERVATION", pending_action="HOLD_OBSERVATION")
        ID.holding_context(holding, row, {"state": "RISK"})
        self.assertEqual(holding["status"], "REDUCE_REVIEW")
        holding.update(status="T1_LOCKED", pending_action="HOLD_OBSERVATION")
        ID.holding_context(holding, row, {"state": "RISK"})
        self.assertEqual(holding["status"], "T1_LOCKED")
        self.assertEqual(holding["pending_action"], "REDUCE_REVIEW")
        ID.holding_context({"status": "DATA_WAIT"}, {"quote": None}, {"state": "UNKNOWN"})

    def test_expiry_and_auction_clear_action_lists(self):
        self.complete_minutes()
        report = self.report()
        auction = deepcopy(report)
        R.check_expiry(auction, "2026-09-18 14:57")
        self.assertFalse(auction["candidates"]["paper_review_symbols"])
        R.check_expiry(report, self.at + pd.Timedelta(minutes=3))
        self.assertFalse(report["candidates"]["focus_symbols"])
        self.assertEqual(report["themes"]["status"], "DATA_WAIT")
        self.assertFalse(any(r["paper_review_eligible"] for r in report["records"]))

    def test_index_expiry_is_not_hidden_by_newer_stock_quote(self):
        for row in self.bundle["raw"][1]["records"]:
            row["trade_time"] = "2026-09-18 14:33:30"
        report = self.report()
        R.check_expiry(report, self.at + pd.Timedelta(seconds=31))
        self.assertEqual(report["summary"]["status"], "EXPIRED")
        self.assertFalse(report["candidates"]["paper_review_symbols"])

    def test_minute_selection_uses_current_snapshot_not_fixed_previous_list(self):
        seen = []
        class Client:
            audit = []
            def query(client, api, **kwargs):
                if api == "rt_k":
                    return pd.DataFrame(self.rows)
                if api == "rt_idx_k":
                    return pd.DataFrame(self.bundle["raw"][1]["records"])
                seen.append(kwargs["ts_code"])
                return pd.DataFrame()
        def select(raw, at):
            self.assertEqual([b["api"] for b in raw], ["rt_k", "rt_idx_k"])
            return [self.symbols[4], self.symbols[4]]
        bundle = Q.collect(Client(), self.symbols, [self.symbols[0]], self.at, minute_selector=select)
        self.assertEqual(seen, [Q.ts_code(self.symbols[4])])
        self.assertEqual(bundle["minute_symbols"], [self.symbols[4]])

    def test_report_three_sections_and_dashboard_are_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = self.report()
            R.check_expiry(report, self.at)
            path = R.publish(self.bundle, report, root)
            content = (path / "review.md").read_text(encoding="utf-8")
            for title in ("1. 行情总体与风险", "2. 已有持仓决策", "3. 潜力标的优先级", "本轮总结"):
                self.assertIn(title, content)
            before = (path / "review.json").read_bytes()
            view = R.dashboard("2026-09-18", root, self.at + pd.Timedelta(minutes=3))
            self.assertEqual(view["mode"], "STALE")
            self.assertFalse(view["report"]["candidates"]["paper_review_symbols"])
            self.assertEqual(before, (path / "review.json").read_bytes())
            with self.assertRaises(ValueError):
                R.dashboard("../../other", root)

    def test_visual_api_renders_live_missing_and_stale_without_writes(self):
        from mainwave_visual_app import app
        with tempfile.TemporaryDirectory() as tmp, patch.dict(app.config, INTRADAY_ROOT=Path(tmp)):
            client = app.test_client()
            self.assertEqual(client.get('/intraday?date=2026-09-18').status_code, 200)
            self.assertEqual(client.get('/api/intraday?date=../../bad').status_code, 400)
            self.complete_minutes()
            report = self.report()
            R.check_expiry(report, self.at)
            output = R.publish(self.bundle, report, Path(tmp))
            before = (output / 'review.json').read_bytes()
            with patch.object(Q, 'now_china', return_value=self.at.to_pydatetime()):
                live = client.get('/intraday?date=2026-09-18')
                self.assertEqual(live.status_code, 200)
                self.assertIn('模拟参与复核 2 只', live.get_data(as_text=True))
                self.assertEqual(live.headers['Cache-Control'], 'no-store')
            with patch.object(Q, 'now_china', return_value=(self.at + pd.Timedelta(minutes=3)).to_pydatetime()):
                stale = client.get('/api/intraday?date=2026-09-18').get_json()
                self.assertEqual(stale['mode'], 'STALE')
                self.assertEqual(stale['report']['candidates']['paper_review_symbols'], [])
            self.assertEqual(before, (output / 'review.json').read_bytes())


if __name__ == "__main__":
    unittest.main()
