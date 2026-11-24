"""Batch B invariants: rule dispatch, mode contracts, fill-anchored adds and run recovery."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from .trading import adds, modes, paper, rules, runner, store
from .test_trading_system import AT, CONFIG, DAY, bar, payload

NEXT = "2026-09-21"
CALENDAR = ["2026-09-16", "2026-09-17", DAY, NEXT, "2026-09-22", "2026-09-23", "2026-09-24",
            "2026-09-25", "2026-09-28", "2026-09-29", "2026-09-30"]


class DispatchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "t.sqlite3"
        with patch.object(store, "now", return_value=AT):
            store.persist(payload(), self.db)

    def tearDown(self):
        self.temp.cleanup()

    def test_unchanged_rules_run_as_current(self):
        rules.register(self.db)
        plan = {"id": "p", "rule_fingerprint": rules.fingerprint()}
        self.assertEqual(rules.dispatch(plan, self.db)["status"], "CURRENT")

    def test_registered_old_parameters_are_restored_for_an_old_plan(self):
        rules.register(self.db)
        current = rules.build()
        frozen = rules.fingerprint()
        with store.connect(self.db) as conn:
            rules._table(conn)
            item = {**current["parts"]["exit"], "version": "exit_old_v0", "sha256": "a" * 64}
            conn.execute("INSERT INTO rule_versions VALUES (?,?,?,?,?)",
                         ("exit:exit_old_v0:" + "a" * 12, "exit", "exit_old_v0", AT, store.encoded(item)))
        frozen["exit"] = {"version": "exit_old_v0", "sha256": "a" * 64, "source": "x"}
        result = rules.dispatch({"id": "p", "rule_fingerprint": frozen}, self.db)
        # Parameters come back from the registry, but that is not execution under the frozen code.
        self.assertEqual(result["status"], "PARAMETERS_ON_RECORD_CODE_CHANGED")
        self.assertEqual(result["parts"]["exit"]["dispatch"], "FROZEN_PARAMETERS_FROM_REGISTRY")
        self.assertEqual(result["parts"]["exit"]["limit"], rules.RESTORABLE)
        self.assertFalse(result["executable_as_frozen"])
        self.assertFalse(result["new_adds_allowed"])

    def test_unrestorable_version_waits_and_keeps_risk_tasks(self):
        rules.register(self.db)
        frozen = rules.fingerprint()
        frozen["exit"] = {"version": "missing_v9", "sha256": "b" * 64, "source": "x"}
        result = rules.dispatch({"id": "p", "rule_fingerprint": frozen}, self.db)
        self.assertEqual(result["status"], "PENDING_REVIEW")
        self.assertIn("exit", result["unrestorable"])
        self.assertTrue(result["risk_tasks_retained"])

    def test_software_version_alone_is_not_a_strategy_freeze(self):
        built = rules.build()
        self.assertIn("software_version", built)
        self.assertNotEqual(built["software_version"], built["parts"]["exit"]["version"])


class ModeTest(unittest.TestCase):
    def record(self, kind="FIRST_PULLBACK", eligible=True):
        return {"symbol": "sh_600010", "route_ids": "R1|R2", "setup_kind": kind,
                "paper_plan_eligible": eligible, "state": "PLAN_REVIEW" if eligible else "WAIT_SETUP",
                "evidence": {"state_vector": {"setup_kind": kind}},
                "source_record": {"discovery_path_label": "LURK"}}

    def test_five_fields_stay_separate(self):
        fields = modes.classify(self.record())
        for key in ("discovery_source", "route_ids", "trade_mode", "local_setup", "participation_permission"):
            self.assertIn(key, fields)
        self.assertEqual(fields["trade_mode"], modes.TREND_PULLBACK)
        self.assertEqual(fields["sub_scenario"], "FIRST_PULLBACK")

    def test_unknown_setup_does_not_inherit_permission(self):
        fields = modes.classify(self.record(kind="UNKNOWN"))
        self.assertEqual(fields["trade_mode"], "UNASSIGNED")
        self.assertFalse(fields["executable"])

    def test_mapped_but_disabled_scenario_is_not_executable(self):
        fields = modes.classify(self.record(kind="EMA10_RECLAIM"))
        self.assertEqual(fields["sub_scenario"], "EMA10_RECLAIM")
        self.assertFalse(fields["executable"])

    def test_one_opportunity_keeps_several_mode_verdicts(self):
        result = modes.evaluations(self.record())
        states = {row["sub_scenario"] or row["mode"]: row["state"] for row in result["evaluations"]}
        self.assertEqual(states["FIRST_PULLBACK"], "PLAN_REVIEW")
        self.assertEqual(states["BREAKOUT_TEST"], "NOT_THIS_SETUP")
        self.assertTrue(states[modes.OVERNIGHT].startswith("OBSERVATION_ONLY"))
        self.assertEqual(sum(row["executable"] for row in result["evaluations"]), 1)

    def test_active_exit_outranks_a_new_mode_signal(self):
        self.assertEqual(modes.conflict({"active_exit": True, "new_mode_signal": True})["winner"], "ACTIVE_EXIT")
        self.assertEqual(modes.conflict({"new_mode_signal": True})["winner"], "NEW_MODE_SIGNAL")


class AddTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / "t.sqlite3"
        with patch.object(store, "now", return_value=AT):
            self.snap = store.persist(payload(), self.db)
            paper.initialize(CONFIG, self.db, at=AT)
            self.plan_id = paper.submit(self.snap["snapshot_id"], "sh_600010", self.db, at=AT)["plan_id"]

    def tearDown(self):
        self.temp.cleanup()

    def plan(self, conn):
        return json.loads(conn.execute("SELECT payload FROM paper_plans WHERE id=?", (self.plan_id,)).fetchone()[0])

    def frame(self, rows):
        frame = pd.DataFrame(rows, columns=None if rows else ["date", "close"])
        frame["date"] = pd.to_datetime(frame["date"])
        for column in ("index_close", "index_low", "index_high", "index_open"):
            if column not in frame:
                frame[column] = frame["close"]
        return frame

    def test_no_initial_fill_means_no_add_process(self):
        with paper.database(self.db) as conn:
            account, cfg = paper._account(conn)
            result = adds.evaluate(conn, self.plan(conn), self.frame([]), CALENDAR, NEXT, [], account, cfg)
        self.assertEqual(result["status"], "STOPPED")
        self.assertEqual(result["stop_reason"], "NO_INITIAL_FILL")

    def fill_initial(self, quantity=None):
        with paper.database(self.db) as conn:
            paper.process_open(conn, NEXT, {"sh_600010": bar()})
            if quantity:
                conn.execute("UPDATE paper_lots SET quantity=?,remaining=? WHERE plan_id=?",
                             (quantity, quantity, self.plan_id))
        # Cached outside any transaction; a nested connection would deadlock the writer.
        with paper.database(self.db) as conn:
            row = conn.execute("SELECT price_cents FROM paper_lots WHERE plan_id=?", (self.plan_id,)).fetchone()
        self.fill_price = row["price_cents"] / 100 if row else None

    def test_add_clock_starts_at_the_actual_fill(self):
        self.fill_initial()
        with paper.database(self.db) as conn:
            base = adds.anchor(conn, self.plan_id)
        self.assertEqual(base["status"], "ANCHORED")
        self.assertEqual(base["entry_day"], NEXT)
        self.assertGreater(base["filled"], 0)

    def test_partial_initial_is_reported_and_uses_actual_quantity(self):
        self.fill_initial()
        with paper.database(self.db) as conn:
            order = conn.execute("SELECT quantity,filled FROM paper_orders WHERE id=?",
                                 (self.plan_id + ":initial",)).fetchone()
            base = adds.anchor(conn, self.plan_id)
        if order["filled"] < order["quantity"]:
            self.assertTrue(base["partial"])
        self.assertEqual(base["filled"], order["filled"])

    def series(self, fall=.05, recover=True):
        """Flat history for MA20, the fill day, a contracted pullback, then a strong reclaim."""
        price = self.fill_price
        history = pd.bdate_range(end="2026-09-18", periods=22).strftime("%Y-%m-%d").tolist()
        rows = [{"date": day, "close": price, "open": price, "high": price, "low": price,
                 "amount": 100., "volume": 100., "preclose": price, "isST": 0} for day in history]
        low = price * (1 - fall)
        rows += [
            {"date": NEXT, "close": price, "open": price, "high": price, "low": price,
             "amount": 100., "volume": 100., "preclose": price, "isST": 0},
            {"date": "2026-09-22", "close": low * 1.001, "open": low * 1.01, "high": low * 1.02,
             "low": low, "amount": 50., "volume": 50., "preclose": price, "isST": 0},
            {"date": "2026-09-23", "close": price * (1.002 if recover else .97),
             "open": low, "high": price * 1.01, "low": low,
             "amount": 120., "volume": 120., "preclose": low, "isST": 0},
        ]
        frame = pd.DataFrame(rows)
        frame["date"] = pd.to_datetime(frame["date"])
        for column, source in (("index_close", "close"), ("index_open", "open"),
                               ("index_high", "high"), ("index_low", "low")):
            frame[column] = frame[source]
        frame["amount_ratio"] = frame.amount / frame.amount.rolling(5, min_periods=1).median().shift().fillna(100.)
        frame["close_location"] = ((frame.close - frame.low)
                                   / (frame.high - frame.low).replace(0, 1))
        frame["eligible"] = True
        frame["supply_warning"] = False
        return frame

    def evaluate(self, frame, day="2026-09-23"):
        with paper.database(self.db) as conn:
            account, cfg = paper._account(conn)
            plan = self.plan(conn)
            return adds.evaluate(conn, plan, frame, CALENDAR, day, paper._exposure(conn), account, cfg), plan

    def test_pullback_without_reclaim_keeps_the_initial_only(self):
        self.fill_initial()
        result, _ = self.evaluate(self.series(recover=False))
        self.assertEqual(result["status"], "NO_ACTION")
        self.assertEqual(result["tranches"], [])

    def test_window_expiry_stops_remaining_tranches(self):
        self.fill_initial()
        with paper.database(self.db) as conn:
            account, cfg = paper._account(conn)
            result = adds.evaluate(conn, self.plan(conn), self.series(), CALENDAR, "2026-09-30",
                                   paper._exposure(conn), account, cfg)
        self.assertEqual(result["status"], "STOPPED")
        self.assertEqual(result["stop_reason"], "WINDOW_EXPIRED")

    def test_pending_exit_outranks_any_add(self):
        self.fill_initial()
        with paper.database(self.db) as conn:
            paper.queue_exit(conn, self.plan_id, "2026-09-22", "INITIAL_STOP")
            account, cfg = paper._account(conn)
            result = adds.evaluate(conn, self.plan(conn), self.series(), CALENDAR, "2026-09-23",
                                   paper._exposure(conn), account, cfg)
        self.assertEqual(result["stop_reason"], "EXIT_PENDING")

    def test_rules_pending_review_blocks_adds(self):
        self.fill_initial()
        with paper.database(self.db) as conn:
            account, cfg = paper._account(conn)
            result = adds.evaluate(conn, self.plan(conn), self.series(), CALENDAR, "2026-09-23",
                                   paper._exposure(conn), account, cfg, dispatch_status="PENDING_REVIEW")
        self.assertEqual(result["stop_reason"], "RULES_PENDING_REVIEW")

    def test_structure_failure_stops_adds(self):
        self.fill_initial()
        frame = self.series()
        with paper.database(self.db) as conn:
            plan = self.plan(conn)
            frame.loc[frame.index[-1], ["close", "index_close"]] = plan["structure_cents"] / 100 - 1
            account, cfg = paper._account(conn)
            result = adds.evaluate(conn, plan, frame, CALENDAR, "2026-09-23",
                                   paper._exposure(conn), account, cfg)
        self.assertEqual(result["stop_reason"], "STRUCTURE_FAILED")


    def test_full_forward_simulation_initial_to_add_to_staged_exit(self):
        """50% initial, partial fill, recorded cancellation, fill-anchored add, then staged exit."""
        with paper.database(self.db) as conn:
            plan = self.plan(conn)
            ordered = conn.execute("SELECT quantity FROM paper_orders WHERE id=?",
                                   (self.plan_id + ":initial",)).fetchone()["quantity"]
        # The approved initial is exactly half of the frozen plan budget.
        self.assertEqual(plan["planned_budget_cents"], ordered * plan["limit_cents"] * 2)

        self.fill_initial()
        with paper.database(self.db) as conn:
            initial = conn.execute("SELECT filled,status FROM paper_orders WHERE id=?",
                                   (self.plan_id + ":initial",)).fetchone()
            lots = [dict(r) for r in conn.execute(
                "SELECT entry_day,quantity,price_cents FROM paper_lots WHERE plan_id=?", (self.plan_id,))]
        self.assertEqual(len(lots), 1)
        self.assertGreater(initial["filled"], 0)
        if initial["filled"] < ordered:
            # An unfilled remainder must be recorded, not silently dropped.
            self.assertEqual(initial["status"], "CANCELLED")

        result, plan = self.evaluate(self.series())
        self.assertEqual(result["status"], "ADD_CONFIRMED")
        tranche = result["tranches"][0]
        self.assertEqual(tranche["tranche"], 2)
        self.assertEqual(result["anchor"]["entry_day"], NEXT)
        self.assertEqual(tranche["execution"], "NEXT_EXCHANGE_OPEN_ONLY")

        with paper.database(self.db) as conn:
            account, cfg = paper._account(conn)
            placed = adds.queue(conn, plan, result, account, cfg, at="2026-09-23T18:00:00+08:00")
            cash_before = conn.execute("SELECT cash_cents FROM paper_account").fetchone()[0]
        self.assertTrue(placed, "确认后应登记补仓委托")
        self.assertEqual(placed[0]["tranche"], 2)

        price = tranche["max_price_cents"] / 100
        add_bar = dict(date="2026-09-24", open=price, preclose=price, raw_verified=True,
                       ordinary_mainboard=True, corporate_action=False, previous_market_day="2026-09-23")
        with paper.database(self.db) as conn:
            paper.process_open(conn, "2026-09-24", {"sh_600010": add_bar})
            after = [dict(r) for r in conn.execute(
                "SELECT entry_day,quantity,price_cents,remaining FROM paper_lots WHERE plan_id=? ORDER BY entry_day",
                (self.plan_id,))]
            cash_after = conn.execute("SELECT cash_cents FROM paper_account").fetchone()[0]
            frozen = self.plan(conn)["stop_cents"]
        self.assertEqual(len(after), 2, "补仓成交应形成第二批")
        self.assertEqual(after[1]["entry_day"], "2026-09-24")
        spent = after[1]["quantity"] * after[1]["price_cents"]
        self.assertLessEqual(cash_after, cash_before - spent)
        # A lower average cost must never move the frozen hard floor down.
        self.assertEqual(frozen, plan["stop_cents"])

        with paper.database(self.db) as conn:
            paper.queue_exit(conn, self.plan_id, "2026-09-24", "TARGET_3ATR", .5)
            sell = conn.execute("SELECT quantity,status FROM paper_orders WHERE plan_id=? AND side='SELL'",
                                (self.plan_id,)).fetchone()
            held = sum(r[0] for r in conn.execute(
                "SELECT remaining FROM paper_lots WHERE plan_id=?", (self.plan_id,)))
        self.assertEqual(sell["status"], "OPEN")
        self.assertLess(sell["quantity"], held, "分阶段兑现只卖一部分")

    def test_second_add_of_the_same_tranche_is_refused(self):
        self.fill_initial()
        result, plan = self.evaluate(self.series())
        with paper.database(self.db) as conn:
            account, cfg = paper._account(conn)
            first = adds.queue(conn, plan, result, account, cfg, at="2026-09-23T18:00:00+08:00")
            again = adds.queue(conn, plan, result, account, cfg, at="2026-09-23T18:00:00+08:00")
        self.assertTrue(first)
        self.assertEqual(again, [], "每档最多一次")

    def test_add_never_widens_the_frozen_hard_floor(self):
        self.fill_initial()
        result, plan = self.evaluate(self.series())
        for tranche in result.get("tranches", []):
            self.assertGreaterEqual(tranche["min_price_cents"], plan["stop_cents"])


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / "t.sqlite3"
        with patch.object(store, "now", return_value=AT):
            store.persist(payload(), self.db)

    def tearDown(self):
        self.temp.cleanup()

    def test_repeat_schedule_does_not_start_a_second_run(self):
        first = runner.acquire("after_close", self.db, at=AT)
        self.assertTrue(first["acquired"])
        with patch.object(runner, "owner", return_value="other:1"):
            second = runner.acquire("after_close", self.db, at=AT)
        self.assertFalse(second["acquired"])
        self.assertEqual(second["reason"], "ANOTHER_RUN_IN_PROGRESS")

    def test_dead_run_lock_is_reclaimed_after_timeout(self):
        with patch.object(runner, "owner", return_value="dead:1"):
            runner.acquire("after_close", self.db, at="2026-09-21T09:00:00+08:00")
        result = runner.acquire("after_close", self.db, at="2026-09-21T10:00:00+08:00")
        self.assertTrue(result["acquired"])
        self.assertEqual(result["previous"], "dead:1")

    def test_checkpoint_survives_restart_and_is_not_redone(self):
        runner.checkpoint(DAY, "AFTER_CLOSE", "ranking", "DONE", {"rows": 3}, self.db, at=AT)
        self.assertTrue(runner.completed(DAY, "AFTER_CLOSE", "ranking", self.db))
        self.assertFalse(runner.completed(DAY, "AFTER_CLOSE", "paper", self.db))

    def test_partial_staleness_names_the_affected_capability(self):
        result = runner.freshness(DAY, self.root, self.root / "missing")
        self.assertEqual(result["status"], "PARTIAL_STALE")
        self.assertTrue(all(v["impact"] for v in result["sources"].values()))

    def test_next_task_is_risk_first(self):
        with patch.object(store, "now", return_value=AT):
            paper.initialize(CONFIG, self.db, at=AT)
        with store.connect(self.db) as conn:
            identity = store.latest(conn)["snapshot_id"]
        with patch.object(store, "now", return_value=AT):
            paper.submit(identity, "sh_600010", self.db, at=AT)
        result = runner.pending(DAY, self.db)
        self.assertEqual(result["order"][0], "PENDING_EXITS")
        self.assertTrue(result["next"])

    def test_reconciliation_matches_cash_to_fills(self):
        with patch.object(store, "now", return_value=AT):
            paper.initialize(CONFIG, self.db, at=AT)
        result = runner.reconcile(DAY, self.db)
        self.assertEqual(result["status"], "CONSISTENT")
        self.assertEqual(result["difference_cents"], 0)


if __name__ == "__main__":
    unittest.main()
