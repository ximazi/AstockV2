"""Batch D invariants: reversible schema ops, experiment discipline, security scope, corporate actions, ledger performance."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from .trading import (corporate_actions, experiments, migrations, paper, performance,
                      security_master, store)
from .test_trading_system import AT, CONFIG, DAY, bar, payload

NEXT = "2026-09-21"
CALENDAR = ["2026-09-15", "2026-09-16", "2026-09-17", DAY, NEXT, "2026-09-22", "2026-09-23"]


class Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / "t.sqlite3"
        with patch.object(store, "now", return_value=AT):
            self.snap = store.persist(payload(), self.db)
            paper.initialize(CONFIG, self.db, at=AT)

    def tearDown(self):
        self.temp.cleanup()

    def freeze_validation(self):
        """A frozen validation artefact for the latest snapshot day, so the replay has something to match."""
        from .trading import validation
        (self.root / "data").mkdir(exist_ok=True)
        pd.DataFrame({"trade_date": CALENDAR}).to_csv(self.root / "data/000300_SH_index.csv", index=False)
        folder = self.root / "results/mainwave_engine" / DAY.replace("-", "")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "ranking_validation.json").write_text(
            store.encoded(validation.evaluate(DAY, self.db, self.root / "data")), encoding="utf-8")

    def open_plan(self, symbol="sh_600010", fill=True):
        with patch.object(store, "now", return_value=AT):
            plan_id = paper.submit(self.snap["snapshot_id"], symbol, self.db, at=AT)["plan_id"]
        if fill:
            with paper.database(self.db) as conn:
                paper.process_open(conn, NEXT, {symbol: bar()})
        return plan_id


class MigrationTest(Base):
    def test_upgrade_is_idempotent_and_versioned(self):
        first = migrations.upgrade(self.db)
        second = migrations.upgrade(self.db)
        self.assertEqual(first["to"], migrations.CURRENT)
        self.assertEqual(second["applied"], [])

    def test_downgrade_exports_before_dropping_and_keeps_the_index(self):
        migrations.upgrade(self.db)
        with store.connect(self.db) as conn:
            experiments._table(conn)
            conn.execute("INSERT INTO experiments VALUES ('x', ?, 'REGISTERED', '{}')", (AT,))
        result = migrations.downgrade(self.db, migrations.CURRENT - 1, self.root / "arch")
        self.assertEqual(result["reversed"], [migrations.CURRENT])
        self.assertEqual(result["exported"][0]["rows"], 1)
        self.assertTrue(Path(result["exported"][0]["path"]).exists())
        status = migrations.status(self.db)
        self.assertEqual(status["schema_version"], migrations.CURRENT - 1)
        self.assertTrue(status["archives"], "回退导出必须登记在归档索引里")

    def test_backup_verifies_and_a_tampered_copy_fails(self):
        self.open_plan()
        result = migrations.backup(self.db, self.root / "bk" / "t.sqlite3")
        self.assertEqual(migrations.verify_restore(result["backup"])["status"], "CONSISTENT")
        Path(result["backup"]).write_bytes(Path(result["backup"]).read_bytes() + b"\x00")
        self.assertEqual(migrations.verify_restore(result["backup"])["status"], "MISMATCH")

    def test_archive_keeps_live_rows(self):
        with store.connect(self.db) as conn:
            before = conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
        result = migrations.archive(self.db, "2026-12-31", self.root / "arch")
        self.assertEqual(result["status"], "ARCHIVED")
        self.assertFalse(result["pruned"])
        with store.connect(self.db) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0], before)


class ExperimentTest(Base):
    BODY = {"hypothesis": "换手加权提升提前量", "control": "强势基线", "metric": "lead_days_coverage",
            "sample_split": {"discover_start": "2022-07-01", "discover_end": "2024-12-31",
                             "freeze_date": "2025-01-01", "oos_start": "2026-01-01"}, "max_attempts": 2,
            "oos_acceptance": {"expectancy_net_min": 0.0, "n_effective_min": 20}}
    OOS = {"n_effective": 30, "expectancy_net": 0.4, "source": "AUTO_BOOK", "oos_start": "2026-01-05"}

    def test_registration_requires_split_and_refuses_return_metrics(self):
        with self.assertRaises(ValueError):
            experiments.register({**self.BODY, "metric": "win_rate"}, self.db)
        bad = {**self.BODY, "sample_split": {**self.BODY["sample_split"], "freeze_date": "2024-06-01"}}
        with self.assertRaises(ValueError):
            experiments.register(bad, self.db)
        self.assertTrue(experiments.register(self.BODY, self.db, at=AT)["created"])

    def test_attempt_budget_is_enforced(self):
        ident = experiments.register(self.BODY, self.db, at=AT)["id"]
        experiments.attempt(ident, {"n": 1}, self.db, at=AT)
        experiments.attempt(ident, {"n": 2}, self.db, at=AT)
        with self.assertRaises(ValueError):
            experiments.attempt(ident, {"n": 3}, self.db, at=AT)

    def test_publish_needs_a_passed_scenario_check(self):
        ident = experiments.register(self.BODY, self.db, at=AT)["id"]
        experiments.conclude(ident, "INCONCLUSIVE", {}, self.db, at=AT)
        with self.assertRaises(ValueError):
            experiments.decide(ident, "PUBLISHED", "试", self.db, at=AT)
        self.freeze_validation()
        check = experiments.publish_check(self.db, at="2026-09-19T18:30:00+08:00", root=self.root)
        self.assertEqual(check["status"], "PASS", check)
        self.assertTrue(all(o["status"] == "PASS" for o in check["scenarios"]))
        self.assertEqual(check["replay"]["status"], "PASS")
        experiments.attach_check(ident, check, self.db)
        # Software gate alone never publishes; the frozen out-of-sample economics must pass too.
        with self.assertRaises(ValueError):
            experiments.decide(ident, "PUBLISHED", "ok", self.db, at=AT)
        experiments.economic_check(ident, self.OOS, self.db, at=AT)
        decided = experiments.decide(ident, "PUBLISHED", "ok", self.db, at=AT)["decision"]
        self.assertEqual(decided["decision"], "PUBLISHED")
        self.assertEqual(decided["gates"], {"software": "PASS", "strategy": "PASS"})

    def test_scenarios_run_on_a_copy_and_never_touch_the_ledger(self):
        with paper.database(self.db) as conn:
            before = conn.execute("SELECT COUNT(*) FROM paper_plans").fetchone()[0]
        experiments.publish_check(self.db, at="2026-09-19T18:30:00+08:00", root=self.root)
        with paper.database(self.db) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM paper_plans").fetchone()[0], before)

    def test_rollback_leaves_fills_unchanged(self):
        self.open_plan()
        ident = experiments.register(self.BODY, self.db, at=AT)["id"]
        experiments.conclude(ident, "SUPPORTED", {}, self.db, at=AT)
        check = experiments.publish_check(self.db, at="2026-09-21T18:30:00+08:00", root=self.root)
        experiments.attach_check(ident, {**check, "status": "PASS"}, self.db)
        experiments.economic_check(ident, self.OOS, self.db, at=AT)
        experiments.decide(ident, "PUBLISHED", "ok", self.db, at=AT, effective_from="2026-09-22")
        result = experiments.rollback(ident, "样本外失效", self.db, at=AT, effective_from="2026-09-23")
        self.assertTrue(result["fills_unchanged"])
        self.assertEqual(result["decision"]["decision"], "ROLLED_BACK")

    def test_cadence_reports_next_and_overdue(self):
        result = experiments.cadence("2026-09-26", done=("2026-09-22",))
        self.assertEqual(result["overdue"], ["2026-09-25"])
        self.assertEqual(result["next"], "2026-09-28")


class SecurityMasterTest(unittest.TestCase):
    def test_main_board_ordinary_is_in_scope_and_others_are_not(self):
        main = security_master.profile("sh_600010", NEXT, CALENDAR, bars=set(CALENDAR))
        star = security_master.profile("sh_688001", NEXT, CALENDAR, bars=set(CALENDAR))
        self.assertTrue(main["in_paper_scope"])
        self.assertFalse(star["in_paper_scope"])
        self.assertEqual(star["limit_pct"], 20.0)

    def test_missing_day_is_suspected_not_proven_halt(self):
        row = security_master.profile("sz_000001", NEXT, CALENDAR, bars=set(CALENDAR) - {NEXT})
        self.assertEqual(row["halt"], "SUSPECTED_HALT_OR_MISSING")
        self.assertFalse(row["in_paper_scope"])
        self.assertEqual(row["master_basis"], security_master.MASTER_BASIS)


class CorporateActionTest(Base):
    def frame(self):
        return pd.DataFrame({"date": [NEXT, "2026-09-22", "2026-09-23"],
                             "close": [10.2, 10.3, 5.2], "preclose": [10.1, 10.2, 5.15]})

    def test_detection_infers_a_share_event_but_waits_for_confirmation(self):
        events = corporate_actions.detect(self.frame())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "SHARE_EVENT")
        self.assertEqual(events[0]["basis"], "PRICE_INFERRED_REQUIRES_MANUAL_CONFIRMATION")

    def test_confirmed_share_event_preserves_cost_basis(self):
        plan_id = self.open_plan()
        with paper.database(self.db) as conn:
            rec = corporate_actions.record(conn, plan_id, corporate_actions.detect(self.frame())[0])
            cost = conn.execute("SELECT remaining*price_cents FROM paper_lots WHERE plan_id=?", (plan_id,)).fetchone()[0]
        result = corporate_actions.confirm(rec["id"], "SHARE_EVENT", self.db, at=AT)
        self.assertEqual(result["status"], "APPLIED")
        with paper.database(self.db) as conn:
            after = conn.execute("SELECT remaining*price_cents FROM paper_lots WHERE plan_id=?", (plan_id,)).fetchone()[0]
        self.assertEqual(after, cost)
        self.assertEqual(corporate_actions.confirm(rec["id"], "SHARE_EVENT", self.db, at=AT)["status"], "APPLIED")

    def test_cash_event_credits_cash_and_dismissal_changes_nothing(self):
        plan_id = self.open_plan()
        cash_event = {"day": "2026-09-23", "prior_close": 10.3, "preclose": 10.2, "factor": 1.0098, "kind": "CASH_EVENT",
                      "basis": "PRICE_INFERRED_REQUIRES_MANUAL_CONFIRMATION"}
        with paper.database(self.db) as conn:
            rec = corporate_actions.record(conn, plan_id, cash_event)
            before = conn.execute("SELECT cash_cents FROM paper_account").fetchone()[0]
        corporate_actions.confirm(rec["id"], "CASH_EVENT", self.db, at=AT)
        with paper.database(self.db) as conn:
            after = conn.execute("SELECT cash_cents FROM paper_account").fetchone()[0]
            rec2 = corporate_actions.record(conn, plan_id, {**cash_event, "day": "2026-09-24", "factor": 1.02})
        self.assertGreater(after, before)
        self.assertEqual(corporate_actions.confirm(rec2["id"], "NOT_AN_EVENT", self.db, at=AT)["status"], "DISMISSED")

    def test_relation_maps_raw_fill_to_adjusted(self):
        result = corporate_actions.relation(1020, [2.0])
        self.assertEqual(result["adjusted_equivalent_cents"], 510)


class PerformanceTest(Base):
    def mark(self, plan_id, day, cents):
        with paper.database(self.db) as conn:
            conn.execute("INSERT INTO paper_valuations VALUES (?,?,?,?)",
                         (f"v-{day}", day, plan_id, store.encoded({"day": day, "status": "HOLD", "mark_cents": cents})))

    def test_unconfigured_account_has_no_net_performance(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / "e.sqlite3"
            with patch.object(store, "now", return_value=AT):
                store.persist(payload(), db)
            self.assertEqual(performance.report(NEXT, db)["status"], "UNCONFIGURED")

    def test_unmarked_day_is_data_wait_not_zero(self):
        plan_id = self.open_plan()
        report = performance.report("2026-09-22", self.db)
        self.assertIsNone(report["account"]["equity_cents"])
        self.assertFalse(report["account"]["marked"])
        self.assertEqual(report["nav"][0]["status"], "DATA_WAIT")

    def test_marked_ledger_reports_cost_unrealized_and_t_plus_1(self):
        plan_id = self.open_plan()
        self.mark(plan_id, "2026-09-22", 1100)
        report = performance.report("2026-09-22", self.db)
        plan = report["plans"][0]
        self.assertEqual(plan["sellable_today"], plan["remaining"])
        self.assertGreater(plan["unrealized_cents"], 0)
        self.assertEqual(report["account"]["equity_cents"], report["account"]["cash_cents"] + plan["remaining"] * 1100)
        self.assertEqual(report["nav"][-1]["status"], "MARKED")

    def test_realized_uses_fifo_and_exposure_survives_a_blocked_exit(self):
        plan_id = self.open_plan()
        with paper.database(self.db) as conn:
            paper.queue_exit(conn, plan_id, "2026-09-22", "TARGET_3ATR", .5)
            paper.process_open(conn, "2026-09-23", {"sh_600010": dict(date="2026-09-23", open=11.0, preclose=10.9,
                                                                       raw_verified=True, ordinary_mainboard=True,
                                                                       corporate_action=False, previous_market_day="2026-09-22")})
            conn.execute("UPDATE paper_orders SET status='EXIT_BLOCKED' WHERE plan_id=? AND side='SELL'", (plan_id,))
        self.mark(plan_id, "2026-09-23", 1100)
        report = performance.report("2026-09-23", self.db)
        plan = report["plans"][0]
        self.assertGreater(plan["realized_cents"], 0)
        self.assertTrue(plan["exposure_continues"])
        self.assertIn("sh_600010", report["exposure"]["continuing"])

    def test_price_observation_and_paper_net_are_reported_apart(self):
        self.open_plan()
        report = performance.report(NEXT, self.db)
        self.assertIn("paper_net", report["separation"])
        self.assertIn("price_observation", report["separation"])


if __name__ == "__main__":
    unittest.main()
