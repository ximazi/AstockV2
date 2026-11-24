"""Reproductions of the 2026-09-20 code audit (F01-F10); each test failed on the code the audit reviewed."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from .trading import (adds, assessment, brief, corporate_actions, daily, experiments, explain, modes, paper,
                      performance, preflight, ranking, rules, runner, store, validation)
from .test_trading_system import AT, CONFIG, DAY, bar, inputs, payload

NEXT = "2026-09-21"
# Cash-bound account: risk and exposure caps wide enough that only cash limits an add.
CASH_BOUND = {**CONFIG, "single_risk_pct": 20, "total_risk_pct": 20,
              "symbol_cap_pct": 100, "theme_cap_pct": 100, "gross_cap_pct": 100}


class Ledger(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / "t.sqlite3"
        with patch.object(store, "now", return_value=AT):
            self.snap = store.persist(payload(), self.db)

    def tearDown(self):
        self.temp.cleanup()

    def open_filled_plan(self, config=CONFIG):
        with patch.object(store, "now", return_value=AT):
            paper.initialize(config, self.db, at=AT)
            plan_id = paper.submit(self.snap["snapshot_id"], "sh_600010", self.db, at=AT)["plan_id"]
        with paper.database(self.db) as conn:
            paper.process_open(conn, NEXT, {"sh_600010": bar()})
        return plan_id

    def plan(self, plan_id):
        with paper.database(self.db) as conn:
            return json.loads(conn.execute("SELECT payload FROM paper_plans WHERE id=?", (plan_id,)).fetchone()[0])

    def write_prices(self, end="2026-09-23"):
        days = pd.bdate_range(end=end, periods=160).strftime("%Y-%m-%d")
        f = pd.DataFrame(dict(date=days, open=10.2, high=10.4, low=10., close=10.2, preclose=10.2, pct_chg=0.,
                              volume=1000000., amount=102000000., turnover=2., isST=0, tradestatus=1))
        f.to_csv(self.root / "sh_600010_daily.csv", index=False)
        pd.DataFrame(dict(trade_date=days, close=100.)).to_csv(self.root / "000300_SH_index.csv", index=False)
        return f

    def valuation(self, plan_id, day):
        with paper.database(self.db) as conn:
            row = conn.execute("SELECT payload FROM paper_valuations WHERE plan_id=? AND as_of=? ORDER BY rowid DESC LIMIT 1",
                               (plan_id, day)).fetchone()
        return json.loads(row[0]) if row else None


class F01DispatchInsideLedgerTransaction(Ledger):
    def old_plan(self):
        rules.register(self.db)
        frozen = rules.fingerprint()
        frozen["exit"] = {"version": "exit_old_v0", "sha256": "a" * 64, "source": "x"}
        return {"id": "p", "rule_fingerprint": frozen}

    def test_lookup_reuses_the_open_connection_instead_of_self_locking(self):
        plan = self.old_plan()
        real = sqlite3.connect
        with patch.object(store.sqlite3, "connect", side_effect=lambda p, timeout=30: real(p, timeout=.05)):
            with paper.database(self.db) as conn:
                # The audit's reproduction: a second writer inside the ledger transaction locks the database.
                with self.assertRaises(sqlite3.OperationalError):
                    rules.dispatch(plan, self.db)
                result = rules.dispatch(plan, self.db, conn=conn)
        self.assertEqual(result["status"], "PENDING_REVIEW")
        self.assertFalse(result["new_adds_allowed"])

    def test_registered_parameters_are_not_called_restored_execution(self):
        rules.register(self.db)
        current = rules.build()
        frozen = rules.fingerprint()
        with store.connect(self.db) as conn:
            rules._table(conn)
            item = {**current["parts"]["entry"], "version": "entry_old_v0", "sha256": "c" * 64}
            conn.execute("INSERT INTO rule_versions VALUES (?,?,?,?,?)",
                         ("entry:entry_old_v0:" + "c" * 12, "entry", "entry_old_v0", AT, store.encoded(item)))
        frozen["entry"] = {"version": "entry_old_v0", "sha256": "c" * 64, "source": "x"}
        result = rules.dispatch({"id": "p", "rule_fingerprint": frozen}, self.db)
        self.assertEqual(result["status"], "PARAMETERS_ON_RECORD_CODE_CHANGED")
        self.assertFalse(result["executable_as_frozen"])
        # Adds stop on any drift, registered parameters or not.
        stopped = adds.evaluate(None, {"id": "p", "symbol": "sh_600010"}, None, [], DAY, [], {}, {},
                                dispatch_status=result["status"])
        self.assertEqual(stopped["stop_reason"], "RULES_PENDING_REVIEW")

    def test_changed_exit_code_still_runs_the_deterministic_stop(self):
        plan_id = self.open_filled_plan()
        f = self.write_prices()
        first = daily.monitor_paper(NEXT, self.db, self.root)
        self.assertEqual(first["status"], "SUCCESS", first)
        plan = self.plan(plan_id)
        floor = plan["atr_cents"] and (1020 - 2 * plan["atr_cents"]) / 100
        f.loc[f.date.eq("2026-09-22"), "low"] = floor - .05
        f.to_csv(self.root / "sh_600010_daily.csv", index=False)
        with patch.object(paper, "source_fingerprint", return_value={}):
            result = daily.monitor_paper("2026-09-22", self.db, self.root)
        self.assertEqual(result["status"], "DATA_WAIT")
        self.assertTrue(any("FROZEN_RULE_MISMATCH_REQUIRES_REVIEW" in s for s in result["issues"]))
        v = self.valuation(plan_id, "2026-09-22")
        self.assertEqual(v["status"], "EXIT_REQUIRED")
        self.assertEqual(v["reason"], "INITIAL_STOP")
        self.assertTrue(v["deterministic_only"])
        with paper.database(self.db) as conn:
            sell = conn.execute("SELECT status FROM paper_orders WHERE plan_id=? AND side='SELL'", (plan_id,)).fetchone()
        self.assertEqual(sell["status"], "OPEN", "硬底线击穿的退出任务不因规则版本变化而消失")


class F02AddBudgetIsReadAfterEveryPlacement(Ledger):
    def test_two_tranches_cannot_reserve_more_than_the_cash(self):
        plan_id = self.open_filled_plan(CASH_BOUND)
        plan = self.plan(plan_id)
        weights = adds.SCHEME["weights"]
        evaluation = {"anchor": {"entry_day": NEXT}, "budget": {}, "tranches": [
            {"tranche": k + 2, "weight": weights[k], "max_price_cents": plan["limit_cents"],
             "min_price_cents": plan["stop_cents"], "confirm_date": "2026-09-23"} for k in range(2)]}
        with paper.database(self.db) as conn:
            conn.execute("UPDATE paper_account SET cash_cents=200000 WHERE id=1")  # 2000 yuan
            account, cfg = paper._account(conn)
            placed = adds.queue(conn, plan, evaluation, account, cfg, at="2026-09-23T18:00:00+08:00")
            reserved = sum(r[0] for r in conn.execute(
                "SELECT reserve_cents FROM paper_orders WHERE status IN ('OPEN','PARTIAL','UNKNOWN')"))
            cash = conn.execute("SELECT cash_cents FROM paper_account").fetchone()[0]
        self.assertEqual(len(placed), 1, "第二档看到第一档已预留后应被拒绝，而不是再登记一次同样的钱")
        self.assertLessEqual(reserved, cash)

    def test_budget_honours_other_plans_reservations_and_exposure_caps(self):
        plan_id = self.open_filled_plan(CASH_BOUND)
        plan = self.plan(plan_id)
        with paper.database(self.db) as conn:
            _, cfg = paper._account(conn)
            free = adds.budget(plan, None, cfg, None, conn)
            conn.execute("INSERT INTO paper_orders VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (plan_id + ":other", plan_id, "BUY", 100, 0, "OPEN", 5_000_000, 0, AT, "{}"))
            after = adds.budget(plan, None, cfg, None, conn)
        self.assertEqual(free["cash_cents"] - after["cash_cents"], 5_000_000)
        for key in ("symbol_headroom_cents", "theme_headroom_cents", "gross_headroom_cents", "single_risk_headroom_cents"):
            self.assertIn(key, after)


class F03HardFloorIsFrozenOnTheFirstFill(Ledger):
    def test_a_cheaper_add_lowers_the_average_but_not_the_monitored_floor(self):
        plan_id = self.open_filled_plan()
        self.write_prices()
        daily.monitor_paper(NEXT, self.db, self.root)
        before = self.valuation(plan_id, NEXT)
        with paper.database(self.db) as conn:
            oid = plan_id + ":add2"
            conn.execute("INSERT INTO paper_orders VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (oid, plan_id, "BUY", 500, 0, "OPEN", 500 * 930 + 500, 0, NEXT + "T18:00:00+08:00",
                          store.encoded({"reason": "CONDITIONAL_ADD", "tranche": 2, "min_price_cents": 900,
                                         "max_price_cents": 1000, "reference_day": NEXT, "model": paper.MODEL})))
            paper.apply_fill(conn, oid, "fill-add2", "2026-09-22", 500, 9.30)
        daily.monitor_paper("2026-09-22", self.db, self.root)
        after = self.valuation(plan_id, "2026-09-22")
        self.assertEqual(after["cost_basis_cents"], 964, "均价确实被拉低到 9.64 元")
        self.assertAlmostEqual(after["hard_floor"], before["hard_floor"], places=6)
        self.assertGreater(after["hard_floor"], 9.0, "审核复现的 8.6375 不再出现")


class F04CorporateActionsAreDatedEvents(Ledger):
    def test_share_event_leaves_history_intact_and_releases_the_block(self):
        plan_id = self.open_filled_plan()
        f = self.write_prices()
        daily.monitor_paper(NEXT, self.db, self.root)
        f.loc[f.date.ge("2026-09-22"), ["open", "high", "low", "close", "preclose"]] /= 2
        f.to_csv(self.root / "sh_600010_daily.csv", index=False)
        blocked = daily.monitor_paper("2026-09-22", self.db, self.root)
        self.assertEqual(blocked["status"], "DATA_WAIT")
        pending = corporate_actions.pending(self.db)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["kind"], "SHARE_EVENT")
        plan = self.plan(plan_id)
        with paper.database(self.db) as conn:
            nav_before, _ = performance.nav_series(conn, 10_000_000, [plan])
        corporate_actions.confirm(pending[0]["id"], "SHARE_EVENT", self.db, at=AT)
        with paper.database(self.db) as conn:
            nav_after, _ = performance.nav_series(conn, 10_000_000, [plan])
        day_before = {r["day"]: r for r in nav_before}[NEXT]
        day_after = {r["day"]: r for r in nav_after}[NEXT]
        self.assertEqual(day_before["equity_cents"], day_after["equity_cents"], "事件前一日净值不得被改写")
        released = daily.monitor_paper("2026-09-22", self.db, self.root)
        self.assertEqual(released["status"], "SUCCESS", released)
        v = self.valuation(plan_id, "2026-09-22")
        self.assertEqual(v["status"], "HOLD")
        self.assertEqual(v["remaining"], 600)
        self.assertEqual(v["market_value_cents"], 300 * 1020, "拆股只换坐标，不改市值")
        self.assertEqual(v["cost_basis_cents"], 510)

    def test_cash_event_enters_reconciliation_and_nav_on_its_day(self):
        plan_id = self.open_filled_plan()
        with paper.database(self.db) as conn:
            recorded = corporate_actions.record(conn, plan_id, {"day": "2026-09-23", "prior_close": 10.2, "preclose": 9.9,
                                                                 "factor": 1.0303, "kind": "CASH_EVENT"})
        corporate_actions.confirm(recorded["id"], "CASH_EVENT", self.db, at=AT, cash_per_share_cents=30)
        result = runner.reconcile("2026-09-23", self.db)
        self.assertEqual(result["status"], "CONSISTENT", result)
        self.assertEqual(result["corporate_cash_cents"], 300 * 30)
        plan = self.plan(plan_id)
        with paper.database(self.db) as conn:
            for day in (NEXT, "2026-09-22", "2026-09-23"):
                conn.execute("INSERT OR IGNORE INTO paper_valuations VALUES (?,?,?,?)",
                             (day, day, plan_id, store.encoded({"day": day, "status": "HOLD", "mark_cents": 1020})))
            nav, _ = performance.nav_series(conn, 10_000_000, [plan])
        by_day = {r["day"]: r for r in nav}
        self.assertEqual(by_day["2026-09-23"]["cash_cents"] - by_day["2026-09-22"]["cash_cents"], 9000)


class F05PublishCheckNeverPassesOnSkips(Ledger):
    def test_all_skipped_is_incomplete_not_pass(self):
        check = experiments.publish_check(self.db, at="2026-09-19T18:30:00+08:00", root=self.root)
        self.assertTrue(all(o["status"] == "SKIPPED" for o in check["scenarios"]))
        self.assertEqual(check["replay"]["status"], "SKIPPED")
        self.assertEqual(check["status"], "INCOMPLETE")
        self.assertTrue(check["blocking"])

    def test_replay_compares_rows_not_counts(self):
        with patch.object(store, "now", return_value=AT):
            older = payload()
            older["as_of"] = "2026-09-17"
            store.persist(older, self.db)
        (self.root / "data").mkdir()
        days = ["2026-09-10", "2026-09-11", "2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", DAY]
        pd.DataFrame({"trade_date": days}).to_csv(self.root / "data/000300_SH_index.csv", index=False)
        stored = validation.evaluate(DAY, self.db, self.root / "data")
        self.assertTrue(stored["records"])
        stored["records"][0]["status"] = "MATURE"  # same row count, different content
        folder = self.root / "results/mainwave_engine" / DAY.replace("-", "")
        folder.mkdir(parents=True)
        (folder / "ranking_validation.json").write_text(store.encoded(stored), encoding="utf-8")
        check = experiments.publish_check(self.db, at="2026-09-19T18:30:00+08:00", root=self.root)
        self.assertEqual(check["replay"]["status"], "FAIL")
        self.assertEqual(check["status"], "FAIL")


class F06MissedAuditMaturesPerSignalDay(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / "t.sqlite3"
        (self.root / "data").mkdir()
        self.days = pd.bdate_range("2026-07-01", DAY).strftime("%Y-%m-%d").tolist()
        pd.DataFrame({"trade_date": self.days}).to_csv(self.root / "data/000300_SH_index.csv", index=False)
        for i in range(4):
            pct = [0.] * len(self.days)
            if i == 0:
                pct[3] = 30.  # sh_600010 gains 30% shortly after the July signal day
            pd.DataFrame({"date": self.days, "pct_chg": pct, "close": (1 + pd.Series(pct) / 100).cumprod() * 10}).to_csv(
                self.root / f"data/sh_{600010 + i}_daily.csv", index=False)
        with patch.object(store, "now", return_value=AT):
            july = payload()
            july["as_of"] = "2026-07-01"
            store.persist(july, self.db)
            store.persist(payload(), self.db)

    def tearDown(self):
        self.temp.cleanup()

    def test_older_signal_day_matures_while_the_newest_is_still_open(self):
        result = validation.missed_audit(DAY, self.db, self.root / "data")
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["matured_signal_days"], ["2026-07-01"])
        self.assertEqual([x["signal_day"] for x in result["immature_signal_days"]], [DAY])
        self.assertEqual(result["signal_day"], "2026-07-01")
        self.assertEqual(result["evaluation_as_of"], DAY)
        self.assertTrue(any(c["symbol"] == "sh_600010" and c["in_pool"] for c in result["cases"]))

    def test_matured_results_are_kept_not_rescanned(self):
        validation.missed_audit(DAY, self.db, self.root / "data")
        with patch.object(validation, "_audit_day", side_effect=AssertionError("已成熟的信号日不应重扫")):
            again = validation.missed_audit(DAY, self.db, self.root / "data")
        self.assertEqual(again["matured_signal_days"], ["2026-07-01"])


class F07ExplanationReadsTheScoredPercentiles(unittest.TestCase):
    def identical(self):
        universe, evidence, metrics = inputs(4)
        for symbol in metrics:
            metrics[symbol] = dict(status="READY", return5=1., return20=2., drawdown20=-1., efficiency10=.1, close=10.)
        p = ranking.build(universe, evidence, metrics, {"return5": 0., "return20": 0.})
        p["knowledge_registry_id"] = "fixture"
        return p

    def test_ties_explain_the_way_they_scored(self):
        p = self.identical()
        for record in p["records"]:
            self.assertEqual(record["strength_score"], 62.5)
            e = explain.explain(record, p)
            self.assertTrue(e["strength"]["consistency"]["consistent"], e["strength"]["consistency"])
            self.assertEqual(e["strength"]["consistency"]["points_sum"], 62.5)
            self.assertEqual({c["basis"] for c in e["strength"]["contributions"]}, {"SCORED_PERCENTILE"})
        # A freshly built payload carries numpy scalars; the board must still serialise on its first publication.
        json.loads(store.encoded(explain.board(p)))

    def test_older_snapshots_recompute_with_the_same_tie_rule(self):
        p = self.identical()
        for record in p["records"]:
            record.pop("score_percentiles")
            e = explain.explain(record, p)
            self.assertTrue(e["strength"]["consistency"]["consistent"])
            self.assertEqual({c["basis"] for c in e["strength"]["contributions"]}, {"RECOMPUTED_AVERAGE_TIE"})
        self.assertEqual(explain.board(p)["summary"]["explained_inconsistent"], 0)


class F08ReconstructionIsCountedApart(unittest.TestCase):
    def test_summary_splits_rows_by_basis(self):
        live, rebuilt = brief.LIVE_BASIS, "DATED_SOURCE_RECONSTRUCTION_NOT_PROOF_OF_LIVE_DISCOVERY"
        rows = [{"status": "MATURE", "signal_day": "2026-09-01", "opportunity_id": f"o{i}", "symbol": f"s{i}",
                 "cycle_id": "c", "basis": rebuilt} for i in range(5)]
        rows += [{"status": "IMMATURE", "signal_day": DAY, "opportunity_id": "o9", "symbol": "s9", "cycle_id": "c",
                  "basis": live}]
        counts = validation._counts(rows)
        self.assertEqual(counts["mature_rows"], 5)
        self.assertEqual(counts["by_basis"][live]["mature_rows"], 0)
        self.assertEqual(counts["by_basis"][rebuilt]["rows"], 5)


class F09CachedVerdictsExpire(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "t.sqlite3"
        with patch.object(store, "now", return_value=AT):
            self.snap = store.persist(payload(), self.db)
        record = self.snap["records"][0]
        self.pack = assessment.bundle(record, self.snap, {"cards": []}, modes.evaluations(record))

    def tearDown(self):
        self.temp.cleanup()

    def support(self, pack, timeout=None):
        return {"verdict": "SUPPORT", "supporting_evidence": [{"evidence_id": self.pack["evidence"][0]["id"]}],
                "counter_evidence": [], "unknowns": [], "alternative_explanations": [],
                "mode_fit": "TREND_PULLBACK", "next_review_event": "次日开盘复核",
                "expires_at": "2026-09-19T19:00:00+08:00"}

    def test_cache_hit_after_expiry_is_not_a_valid_support(self):
        first = assessment.assess(self.pack, client=self.support, db=self.db, at="2026-09-19T18:00:00+08:00", model="m")
        self.assertEqual(first["assessment"]["verdict"], "SUPPORT")
        self.assertTrue(assessment.gate(first, required=True)["new_entry_allowed"])
        later = assessment.assess(self.pack, client=self.support, db=self.db, at="2026-09-20T18:00:00+08:00", model="m")
        self.assertTrue(later["cached"])
        self.assertEqual(later["status"], "EXPIRED")
        self.assertEqual(later["assessment"]["verdict"], "UNKNOWN")
        self.assertFalse(assessment.gate(later, required=True)["new_entry_allowed"])

    def test_wait_does_not_open_a_required_gate(self):
        waiting = {"status": "OK", "assessment": {"verdict": "WAIT"}}
        self.assertFalse(assessment.gate(waiting, required=True)["new_entry_allowed"])
        self.assertTrue(assessment.gate(waiting, required=False)["new_entry_allowed"])


class F10FailedClosureReleasesItsLock(unittest.TestCase):
    def test_blocked_preflight_leaves_no_lock_behind(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = root / "t.sqlite3"
            blocked = {"status": "BLOCKED", "blocking": ["analysis_universe.json"], "degraded": []}
            with patch.object(preflight, "check", return_value=blocked), \
                    patch.object(daily, "write", lambda *a, **k: None):
                with self.assertRaises(ValueError):
                    daily.run(DAY, db, root)
            with patch.object(runner, "owner", return_value="other-host:1"):
                lock = runner.acquire(f"after_close:{DAY}", db)
            self.assertTrue(lock["acquired"])
            self.assertNotIn("reclaimed", lock, "锁应已被释放，而不是等超时后被接管")
            with store.connect(db) as conn:
                failed = conn.execute("SELECT status FROM run_checkpoints WHERE day=? AND step='closure'", (DAY,)).fetchone()
            self.assertEqual(failed["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()
