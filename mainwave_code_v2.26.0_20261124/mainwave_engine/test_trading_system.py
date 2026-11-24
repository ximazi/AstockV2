"""Closure invariants: frozen evidence, permission boundaries and fill-led accounting."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from flask import Flask

from .trading import daily, knowledge, paper, ranking, store, web


DAY = "2026-09-18"
AT = "2026-09-19T18:00:00+08:00"
CONFIG = dict(capital=100000, fee_bps=3, min_fee=5, sell_tax_bps=5,
              single_risk_pct=1, total_risk_pct=3, symbol_cap_pct=20,
              theme_cap_pct=40, gross_cap_pct=80, max_fill_lots=3)


def inputs(count=4):
    records, cores, metrics = [], {}, {}
    for i in range(count):
        symbol = f"sh_{600010+i}"
        records.append(dict(symbol=symbol, name="Fixture", belonging_theme="test", cycle_id="test-cycle",
                            route_ids="R1|R2", price_date=DAY, initial_review=True, value_rank=i+1))
        cores[symbol] = dict(state_vector={"setup_kind": "FIRST_PULLBACK", "risk_overlays": []},
                             review_window=dict(active=True, lower=10., upper=10.5, invalid=9., frozen_atr=.5))
        metrics[symbol] = dict(status="READY", return5=i, return20=i*2, drawdown20=-i,
                               efficiency10=i/10, close=10.)
    return (dict(as_of=DAY, summary={"status": "SUCCESS"},
                 market={"trend_review_allowed": True, "data_status": "FRESH"}, records=records),
            dict(as_of=DAY, records=cores), metrics)


def payload(count=4):
    p = ranking.build(*inputs(count), {"return5": 0., "return20": 0.})
    p["knowledge_registry_id"] = "fixture"
    return p


def bar(day="2026-09-21", op=10.2, pre=10.1):
    return dict(date=day, open=op, preclose=pre, raw_verified=True, ordinary_mainboard=True,
                corporate_action=False, previous_market_day=DAY)


class ClosureTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / "trading.sqlite3"
        with patch.object(store, "now", return_value=AT):
            self.snap = store.persist(payload(), self.db)

    def tearDown(self):
        self.temp.cleanup()

    def account(self):
        return paper.initialize(CONFIG, self.db, at=AT)

    def approve(self, symbol="sh_600010"):
        return paper.submit(self.snap["snapshot_id"], symbol, self.db, at=AT)

    def test_persist_idempotent_and_old_snapshot_does_not_rewind(self):
        self.assertEqual(store.persist(payload(), self.db)["snapshot_id"], self.snap["snapshot_id"])
        old = payload()
        old["as_of"] = "2026-09-17"
        store.persist(old, self.db)
        with store.connect(self.db) as conn:
            self.assertEqual(store.latest(conn)["as_of"], DAY)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0], 2)
            self.assertEqual({r[0] for r in conn.execute("SELECT last_seen FROM opportunities")}, {DAY})
            event = json.loads(conn.execute("SELECT payload FROM opportunity_events WHERE as_of='2026-09-17' LIMIT 1").fetchone()[0])
            self.assertIsNone(event["previous_state"])

    def test_manual_reviews_append_without_changing_permission(self):
        oid = self.snap["records"][0]["opportunity_id"]
        for note in ("first", "revised"):
            store.assess(self.snap["snapshot_id"], oid, {"verdict": "REVIEWED", "note": note}, self.db)
        self.assertEqual(len(store.history(oid, self.db)["assessments"]), 2)
        with store.connect(self.db) as conn:
            self.assertEqual(store.latest(conn), self.snap)

    def test_explicit_account_and_no_reset(self):
        with self.assertRaises(paper.Conflict):
            self.approve()
        with self.assertRaises(ValueError):
            paper.initialize({"capital": 10000}, self.db)
        self.account()
        with self.assertRaises(paper.Conflict):
            self.account()

    def test_duplicate_plan_no_double_reservation(self):
        self.account()
        first = self.approve()
        before = paper.view(self.db)
        second = self.approve()
        self.assertFalse(second["created"])
        self.assertEqual(first["plan_id"], second["plan_id"])
        self.assertEqual(before["account"], paper.view(self.db)["account"])

    def test_budget_is_atomic_across_orders(self):
        self.account()
        for s in ("sh_600010", "sh_600011", "sh_600012", "sh_600013"):
            self.approve(s)
        v = paper.view(self.db)
        self.assertLessEqual(sum(o["risk_cents"] for o in v["orders"]), CONFIG["capital"]*100*.03)
        self.assertGreaterEqual(v["account"]["available_cents"], 0)
        self.assertTrue(v["reconciled"])

    def test_partial_fill_idempotency_and_tplus1(self):
        self.account()
        o = self.approve()
        with paper.database(self.db) as conn:
            self.assertTrue(paper.apply_fill(conn, o["order_id"], "fill-1", "2026-09-21", 100, 10.2))
            self.assertFalse(paper.apply_fill(conn, o["order_id"], "fill-1", "2026-09-21", 100, 10.2))
        with self.assertRaises(paper.Conflict), paper.database(self.db) as conn:
            paper.apply_fill(conn, o["order_id"], "fill-1", "2026-09-21", 200, 10.2)
        with paper.database(self.db) as conn:
            paper.queue_exit(conn, o["plan_id"], "2026-09-21", "INITIAL_STOP")
            sell = conn.execute("SELECT id FROM paper_orders WHERE side='SELL'").fetchone()[0]
        with self.assertRaises(paper.Conflict), paper.database(self.db) as conn:
            paper.apply_fill(conn, sell, "sell-same-day", "2026-09-21", 100, 9.)
        v = paper.view(self.db)
        self.assertEqual(len(v["fills"]), 1)
        self.assertEqual(v["positions"][0]["remaining"], 100)
        self.assertTrue(v["reconciled"])

    def test_ioc_fill_cap_and_replay(self):
        self.account()
        self.approve()
        with paper.database(self.db) as conn:
            paper.process_open(conn, "2026-09-21", {"sh_600010": bar()})
            paper.process_open(conn, "2026-09-21", {"sh_600010": bar()})
        v = paper.view(self.db)
        self.assertEqual(len(v["fills"]), 1)
        self.assertEqual(v["fills"][0]["quantity"], 300)
        self.assertEqual(v["account"]["reserved_cents"], 0)
        self.assertEqual(v["orders"][0]["status"], "CANCELLED")
        self.assertTrue(v["reconciled"])

    def test_no_backdated_fill(self):
        self.account()
        o = self.approve()
        with self.assertRaises(ValueError), paper.database(self.db) as conn:
            paper.apply_fill(conn, o["order_id"], "bad-date", DAY, 100, 10.2)
        self.assertEqual(len(paper.view(self.db)["fills"]), 0)

    def test_missed_next_session_or_gap_open_cancels(self):
        self.account()
        self.approve()
        b = bar(op=10.8)
        with paper.database(self.db) as conn:
            paper.process_open(conn, "2026-09-21", {"sh_600010": b})
        self.assertFalse(paper.view(self.db)["fills"])
        self.assertEqual(paper.view(self.db)["orders"][0]["status"], "CANCELLED")

    def test_limit_exit_blocked_retains_position_and_later_fills(self):
        self.account()
        o = self.approve()
        with paper.database(self.db) as conn:
            paper.process_open(conn, "2026-09-21", {"sh_600010": bar()})
            paper.queue_exit(conn, o["plan_id"], "2026-09-21", "INITIAL_STOP")
            paper.process_open(conn, "2026-09-22", {"sh_600010": bar("2026-09-22", 9., 10.)})
        v = paper.view(self.db)
        self.assertEqual(v["positions"][0]["remaining"], 300)
        self.assertEqual(v["orders"][0]["status"], "EXIT_BLOCKED")
        with paper.database(self.db) as conn:
            paper.process_open(conn, "2026-09-23", {"sh_600010": bar("2026-09-23", 9.1, 9.)})
        v = paper.view(self.db)
        self.assertEqual(v["positions"][0]["remaining"], 0)
        self.assertTrue(v["reconciled"])

    def test_unknown_order_keeps_reserve(self):
        self.account()
        o = self.approve()
        with paper.database(self.db) as conn:
            conn.execute("UPDATE paper_orders SET status='UNKNOWN'")
            paper.process_open(conn, "2026-09-21", {"sh_600010": bar()})
        with self.assertRaises(paper.Conflict):
            paper.cancel(o["order_id"], self.db)
        with self.assertRaises(paper.Conflict):
            self.approve("sh_600011")
        self.assertFalse(paper.view(self.db)["fills"])
        self.assertGreater(paper.view(self.db)["account"]["reserved_cents"], 0)

    def test_changed_exit_code_blocks_unfilled_entry(self):
        self.account()
        self.approve()
        with patch.object(paper, "source_fingerprint", return_value={}):
            with paper.database(self.db) as conn:
                paper.process_open(conn, "2026-09-21", {"sh_600010": bar()})
        self.assertFalse(paper.view(self.db)["fills"])

    def test_stale_snapshot_rejected(self):
        self.account()
        p = payload()
        p["revision"] = 2
        store.persist(p, self.db)
        with self.assertRaises(paper.Conflict):
            self.approve()

    def test_market_gate_independent_from_strength(self):
        u, e, m = inputs()
        u["market"]["trend_review_allowed"] = False
        result = ranking.build(u, e, m, {"return5": 0, "return20": 0})
        self.assertEqual(result["summary"]["rankable"], 4)
        self.assertEqual(result["summary"]["paper_plan_eligible"], 0)
        self.assertEqual({r["state"] for r in result["records"]}, {"MARKET_WAIT"})

    def test_missing_benchmark_fails_closed(self):
        p = ranking.build(*inputs(), {})
        self.assertEqual(p["summary"]["rankable"], 0)
        self.assertEqual({r["state"] for r in p["records"]}, {"DATA_WAIT"})
        self.assertEqual(p["summary"]["paper_plan_eligible"], 0)

    def test_peer_strength_excludes_self(self):
        p = payload()
        first = next(r for r in p["records"] if r["symbol"] == "sh_600010")
        self.assertEqual(first["features"]["peer_excess5"], -2)
        self.assertFalse(first["leadership_confirmed"])

    def test_future_bars_do_not_change_features(self):
        dates = pd.bdate_range(end=DAY, periods=80).strftime("%Y-%m-%d")
        f = pd.DataFrame(dict(date=dates, pct_chg=1., close=10., high=11., low=9., volume=100., turnover=2.))
        base = ranking.features(f, DAY)
        f.loc[len(f)] = ["2026-09-21", 1000., 100., 110., 90., 10000., 30.]
        self.assertEqual(base, ranking.features(f, DAY))

    def test_evaluation_keeps_missing_top_pick_and_immature_null(self):
        # Only fourth-ranked name has prices: it must not replace the frozen top three.
        fourth = next(r for r in self.snap["records"] if r["strength_rank"] == 4)
        pd.DataFrame(dict(date=[DAY, "2026-09-21"], pct_chg=[0., 1.])).to_csv(self.root / f"{fourth['symbol']}_daily.csv", index=False)
        result = daily.evaluate_rankings("2026-09-21", self.db, self.root)
        top = next(r for r in result["comparisons"] if r["group"] == "strength_top3" and r["horizon"] == 5)
        self.assertEqual(top["selected_n"], 3)
        self.assertEqual(top["mature_n"], 0)
        self.assertIsNone(top["mean_return_pct"])
        self.assertGreater(result["summary"]["data_wait_rows"], 0)
        self.assertTrue(all(r["return_pct"] is None for r in result["records"]))

    def test_web_origin_validation_and_exports(self):
        app = Flask(__name__, template_folder=str(store.ROOT / "templates"))
        app.config["TRADING_DB"] = self.db
        app.register_blueprint(web.bp)
        client = app.test_client()
        self.assertEqual(client.get("/trading").status_code, 200)
        self.assertFalse(client.get("/api/trading").json["live_orders_enabled"])
        self.assertEqual(client.post("/api/trading/paper/account", json=CONFIG, headers={"Origin": "http://bad.invalid"}).status_code, 403)
        self.assertEqual(client.post("/api/trading/paper/plans", json={"snapshot_id": {}, "symbol": "sh_600010"}).status_code, 400)
        oid = self.snap["records"][0]["opportunity_id"]
        r = client.post(f"/api/trading/opportunities/{oid}/review", json={"snapshot_id": {}, "verdict": "WAIT", "note": "test"})
        self.assertEqual(r.status_code, 400)
        self.assertIn("attachment", client.get("/api/trading/export").headers["Content-Disposition"])

    def test_registry_actual_sources_and_pending_not_activated(self):
        result = knowledge.build()
        self.assertEqual(result["summary"]["knowledge_items"], 155)
        self.assertEqual(result["summary"]["registered_sources"], 10)
        self.assertGreaterEqual(result["summary"]["pending_sources"], 3)
        self.assertFalse(any(i["live_authorized"] for i in result["items"]))

    def write_prices(self, end="2026-09-22"):
        days = pd.bdate_range(end=end, periods=160).strftime("%Y-%m-%d")
        f = pd.DataFrame(dict(date=days, open=10.2, high=10.4, low=10., close=10.2,
                              preclose=10.2, pct_chg=0., volume=1000000., amount=102000000.,
                              turnover=2., isST=0, tradestatus=1))
        f.to_csv(self.root / "sh_600010_daily.csv", index=False)
        pd.DataFrame(dict(trade_date=days, close=100.)).to_csv(self.root / "000300_SH_index.csv", index=False)
        return f

    def test_eod_monitor_repeat_and_rule_pin(self):
        self.account()
        self.approve()
        self.write_prices()
        first = daily.monitor_paper("2026-09-21", self.db, self.root)
        self.assertEqual(first["status"], "SUCCESS", first)
        second = daily.monitor_paper("2026-09-21", self.db, self.root)
        self.assertEqual(second["status"], "SUCCESS", second)
        v = paper.view(self.db)
        self.assertEqual(len(v["fills"]), 1)
        self.assertEqual(v["positions"][0]["valuation"]["held_sessions"], 1)
        self.assertTrue(v["reconciled"])
        with patch.object(paper, "source_fingerprint", return_value={}):
            result = daily.monitor_paper("2026-09-22", self.db, self.root)
        self.assertEqual(result["status"], "DATA_WAIT")
        # Changed exit code: the position is flagged for review, yet its deterministic floors were still run.
        valuation = paper.view(self.db)["positions"][0]["valuation"]
        self.assertEqual(valuation["status"], "RULES_REVIEW_HOLD")
        self.assertTrue(valuation["deterministic_only"])
        self.assertIn("hard_floor", valuation)

    def test_corporate_action_requires_reconciliation(self):
        self.account()
        self.approve()
        f = self.write_prices()
        daily.monitor_paper("2026-09-21", self.db, self.root)
        f.loc[f.date.eq("2026-09-22"), ["open", "high", "low", "close", "preclose"]] /= 2
        f.to_csv(self.root / "sh_600010_daily.csv", index=False)
        result = daily.monitor_paper("2026-09-22", self.db, self.root)
        self.assertEqual(result["status"], "DATA_WAIT")
        v = paper.view(self.db)
        self.assertEqual(v["positions"][0]["remaining"], 300)
        self.assertIsNone(v["account"]["equity_cents"])

    def test_missing_session_is_not_silently_a_suspension(self):
        self.account()
        self.approve()
        f = self.write_prices(end="2026-09-23")
        daily.monitor_paper("2026-09-21", self.db, self.root)
        f = f[~f.date.eq("2026-09-22")]
        f.to_csv(self.root / "sh_600010_daily.csv", index=False)
        result = daily.monitor_paper("2026-09-23", self.db, self.root)
        self.assertEqual(result["status"], "DATA_WAIT")
        self.assertTrue(any("POSITION_HISTORY_GAP" in s for s in result["issues"]))


if __name__ == "__main__":
    unittest.main()
