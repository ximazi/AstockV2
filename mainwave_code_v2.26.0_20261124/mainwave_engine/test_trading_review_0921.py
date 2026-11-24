"""Review 2026-09-21: symbol risk headroom, three books apart, two release gates, freeze, R3 apart, clusters."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from .trading import adds, books, brief, experiments, paper, ranking, rules, store, validation
from .test_trading_system import AT, CONFIG, DAY, bar, inputs, payload

NEXT = "2026-09-21"
BODY = {"hypothesis": "h", "control": "c", "metric": "lead_days_coverage", "max_attempts": 2,
        "sample_split": {"discover_start": "2022-07-01", "discover_end": "2024-12-31",
                         "freeze_date": "2025-01-01", "oos_start": "2026-01-01"},
        "oos_acceptance": {"expectancy_net_min": 0.0, "n_effective_min": 20, "max_drawdown_max": 0.2}}
OOS = {"n_effective": 30, "expectancy_net": 0.3, "max_drawdown": 0.1, "source": "AUTO_BOOK", "oos_start": "2026-01-05"}


class Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / "t.sqlite3"
        self.payload = payload()
        with patch.object(store, "now", return_value=AT):
            self.snap = store.persist(self.payload, self.db)

    def tearDown(self):
        self.temp.cleanup()


class SymbolRiskHeadroom(Base):
    def test_single_symbol_cap_subtracts_what_the_symbol_already_risks(self):
        with patch.object(store, "now", return_value=AT):
            paper.initialize(CONFIG, self.db, at=AT)
            plan_id = paper.submit(self.snap["snapshot_id"], "sh_600010", self.db, at=AT)["plan_id"]
        with paper.database(self.db) as conn:
            paper.process_open(conn, NEXT, {"sh_600010": bar()})
            plan = json.loads(conn.execute("SELECT payload FROM paper_plans WHERE id=?", (plan_id,)).fetchone()[0])
            _, cfg = paper._account(conn)
            money = adds.budget(plan, None, cfg, None, conn)
            held_risk = sum(x["risk"] for x in paper._exposure(conn) if x["symbol"] == "sh_600010")
        self.assertGreater(held_risk, 0)
        self.assertEqual(money["symbol_risk_cents"], held_risk)
        self.assertEqual(money["single_risk_headroom_cents"], max(0, money["single_risk_cap_cents"] - held_risk))
        self.assertLess(money["single_risk_headroom_cents"], money["single_risk_cap_cents"])

    def test_an_add_cannot_push_the_symbol_past_the_single_risk_cap(self):
        with patch.object(store, "now", return_value=AT):
            paper.initialize(CONFIG, self.db, at=AT)
            plan_id = paper.submit(self.snap["snapshot_id"], "sh_600010", self.db, at=AT)["plan_id"]
        with paper.database(self.db) as conn:
            paper.process_open(conn, NEXT, {"sh_600010": bar()})
            plan = json.loads(conn.execute("SELECT payload FROM paper_plans WHERE id=?", (plan_id,)).fetchone()[0])
            account, cfg = paper._account(conn)
            evaluation = {"anchor": {"entry_day": NEXT}, "budget": {}, "tranches": [
                {"tranche": 2, "weight": .9, "max_price_cents": plan["limit_cents"],
                 "min_price_cents": plan["stop_cents"], "confirm_date": "2026-09-23"}]}
            adds.queue(conn, plan, evaluation, account, cfg, at="2026-09-23T18:00:00+08:00")
            cap = int(min(account["initial_cents"], account["cash_cents"] + sum(x.get("market_value", 0) for x in paper._exposure(conn)))
                      * cfg["single_risk_pct"] / 100)
            symbol_risk = sum(x["risk"] for x in paper._exposure(conn) if x["symbol"] == "sh_600010")
        self.assertLessEqual(symbol_risk, cap + 1, "首仓加补仓的单标的风险不得超过 single_risk_pct")


class ThreeBooksApart(Base):
    def test_trade_set_is_the_literal_top3_with_no_backfill(self):
        records = self.payload["records"]
        for r in records:
            r["paper_plan_eligible"] = True
        records[0]["participation_rank"], records[0]["route_ids"], records[0]["alpha_scope"] = 1, "R3", "R3_RESEARCH"
        for n, r in enumerate(records[1:], 2):
            r["participation_rank"] = n
        chosen = books.trade_set(self.payload)
        # Rank 1 is R3-only and drops out; rank 4 does not move up to take its place.
        self.assertEqual([c["symbol"] for c in chosen], [r["symbol"] for r in records[1:3]])
        self.assertTrue(all(c["alpha_scope"] == "PRICE_ALPHA" for c in chosen))
        records[2]["participation_rank"] = None
        self.assertEqual([c["symbol"] for c in books.trade_set(self.payload)], [records[1]["symbol"]])

    def test_signal_book_records_an_empty_set_as_a_valid_answer(self):
        for r in self.snap["records"]:
            r["paper_plan_eligible"] = False
        entry = books.record_signal(DAY, self.snap, self.db)
        self.assertEqual(entry["trade_set"], [])
        self.assertEqual(books.signal_history(self.db)[0]["day"], DAY)

    def test_auto_book_lives_in_its_own_ledger_and_only_orders_on_a_live_day(self):
        (self.root / "data").mkdir()
        with patch.object(store, "now", return_value=AT):
            rebuilt = books.run_auto(DAY, self.payload, self.snap, self.root, self.root / "data", live=False)
            self.assertEqual({p["status"] for p in rebuilt["placed"]}, {"NOT_LIVE_DAY_NO_ORDER"})
            self.assertEqual(rebuilt["plans"], 0)
            live = books.run_auto(DAY, self.payload, self.snap, self.root, self.root / "data", live=True)
        self.assertTrue(live["placed"])
        self.assertEqual(live["placed"][0]["status"], "ORDER_APPROVED", live["placed"])
        self.assertGreater(live["plans"], 0)
        auto_db = self.root / books.AUTO_DB
        self.assertTrue(auto_db.exists())
        self.assertNotEqual(auto_db, self.db)
        with paper.database(auto_db) as conn:
            plans = [json.loads(r[0]) for r in conn.execute("SELECT payload FROM paper_plans")]
        self.assertTrue(all(p["book"] == "AUTO" for p in plans))
        self.assertIn("participation_rank", plans[0])
        with paper.database(self.db) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM paper_plans").fetchone()[0], 0,
                             "自动账的计划不得出现在人工账里")

    def test_manual_submit_is_labelled_and_auto_refuses_r3_research(self):
        with patch.object(store, "now", return_value=AT):
            paper.initialize(CONFIG, self.db, at=AT)
            plan_id = paper.submit(self.snap["snapshot_id"], "sh_600010", self.db, at=AT)["plan_id"]
        with paper.database(self.db) as conn:
            plan = json.loads(conn.execute("SELECT payload FROM paper_plans WHERE id=?", (plan_id,)).fetchone()[0])
        self.assertEqual(plan["book"], "MANUAL")
        self.assertEqual(plan["alpha_scope"], "PRICE_ALPHA")
        research = payload()
        research["records"][1]["route_ids"], research["records"][1]["alpha_scope"] = "R3", "R3_RESEARCH"
        with patch.object(store, "now", return_value=AT):
            snap = store.persist(research, self.db)
        with self.assertRaises(paper.Conflict):
            paper.submit(snap["snapshot_id"], research["records"][1]["symbol"], self.db, at=AT, book="AUTO")


class TwoReleaseGates(Base):
    def test_registration_freezes_economic_criteria_and_still_refuses_return_tuning(self):
        with self.assertRaises(ValueError):
            experiments.register({k: v for k, v in BODY.items() if k != "oos_acceptance"}, self.db, at=AT)
        with self.assertRaises(ValueError):
            experiments.register({**BODY, "metric": "win_rate"}, self.db, at=AT)
        registered = experiments.register(BODY, self.db, at=AT)
        self.assertEqual(registered["oos_acceptance"], BODY["oos_acceptance"])

    def test_frozen_parts_accept_only_shadow_experiments(self):
        with self.assertRaises(ValueError):
            experiments.register({**BODY, "target_part": "exit"}, self.db, at=AT)
        shadow = experiments.register({**BODY, "target_part": "exit", "shadow_only": True}, self.db, at=AT)
        self.assertTrue(shadow["shadow_only"])
        self.assertIn("exit", rules.build()["freeze"]["parts"])

    def test_publish_needs_software_and_strategy_gates(self):
        ident = experiments.register(BODY, self.db, at=AT)["id"]
        experiments.conclude(ident, "SUPPORTED", {}, self.db, at=AT)
        experiments.attach_check(ident, {"status": "PASS", "at": AT, "scenarios": [], "replay": {"status": "PASS"}}, self.db)
        with self.assertRaises(ValueError):
            experiments.decide(ident, "PUBLISHED", "软件全过", self.db, at=AT)
        thin = experiments.economic_check(ident, {**OOS, "n_effective": 5}, self.db, at=AT)
        self.assertEqual(thin["economic_check"]["status"], "INSUFFICIENT_SAMPLE")
        with self.assertRaises(ValueError):
            experiments.decide(ident, "PUBLISHED", "样本不足", self.db, at=AT)
        losing = experiments.economic_check(ident, {**OOS, "expectancy_net": -0.1}, self.db, at=AT)
        self.assertEqual(losing["economic_check"]["status"], "FAIL")
        with self.assertRaises(ValueError):
            experiments.decide(ident, "PUBLISHED", "不赚钱", self.db, at=AT)
        with self.assertRaises(ValueError):
            experiments.economic_check(ident, {**OOS, "oos_start": "2025-06-01"}, self.db, at=AT)
        passed = experiments.economic_check(ident, OOS, self.db, at=AT)
        self.assertEqual(passed["economic_check"]["status"], "PASS")
        decided = experiments.decide(ident, "PUBLISHED", "两关都过", self.db, at=AT)
        self.assertEqual(decided["decision"]["gates"], {"software": "PASS", "strategy": "PASS"})


class R3ApartAndClusters(unittest.TestCase):
    def rows(self):
        base = {"status": "MATURE", "horizon": 5, "signal_day": DAY, "version": "v", "basis": brief.LIVE_BASIS,
                "return_pct": 1.0, "cycle_id": "c"}
        rows = [{**base, "symbol": f"s{i}", "opportunity_id": f"o{i}", "strength_rank": i, "participation_rank": i,
                 "legacy_rank": i, "momentum_rank": i, "alpha_scope": "PRICE_ALPHA",
                 "strata": {"route": "R1|R2" if i % 2 else "R1", "theme": "机器人"}} for i in range(1, 6)]
        rows.append({**base, "symbol": "s9", "opportunity_id": "o9", "strength_rank": 1, "participation_rank": 1,
                     "legacy_rank": 1, "momentum_rank": 1, "alpha_scope": "R3_RESEARCH",
                     "strata": {"route": "R3", "theme": "机器人"}})
        return rows

    def test_r3_only_rows_never_enter_strategy_cohorts(self):
        groups = {("snap", 5): self.rows()}
        comparisons = []
        with patch.object(validation, "_calendar", return_value=[DAY]):
            for (snapshot_id, horizon), everyone in groups.items():
                group = [r for r in everyone if r["alpha_scope"] != "R3_RESEARCH"]
                comparisons.append(validation._cohort("strength_top3", [r for r in group if r["strength_rank"] <= 3], everyone[0], 5))
                comparisons.append(validation._cohort("r3_research", [r for r in everyone if r["alpha_scope"] == "R3_RESEARCH"],
                                                      everyone[0], 5, research=True))
        top3, research = comparisons
        self.assertNotIn("s9", [r for r in ()])
        self.assertEqual(top3["selected_n"], 3)
        self.assertEqual(research["selected_n"], 1)
        self.assertEqual(research["alpha_scope"], "R3_RESEARCH")
        self.assertIn("不进入策略收益统计", research["note"])

    def test_evaluate_emits_route_and_research_cohorts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").mkdir()
            pd.DataFrame({"trade_date": ["2026-09-17", DAY]}).to_csv(root / "data/000300_SH_index.csv", index=False)
            db = root / "t.sqlite3"
            older = payload()
            older["as_of"] = "2026-09-17"
            older["records"][0]["route_ids"], older["records"][0]["alpha_scope"] = "R3", "R3_RESEARCH"
            with patch.object(store, "now", return_value=AT):
                store.persist(older, db)
            review = validation.evaluate(DAY, db, root / "data")
        groups = {c["group"] for c in review["comparisons"]}
        self.assertTrue({"route_R1", "route_R2", "r3_research"} <= groups)
        everyone = [c for c in review["comparisons"] if c["group"] == "all_candidates"][0]
        research = [c for c in review["comparisons"] if c["group"] == "r3_research"][0]
        self.assertEqual(everyone["selected_n"] + research["selected_n"], len(older["records"]))
        self.assertEqual(review["summary"]["research_rows"], len(review["horizons"]))

    def test_five_symbols_from_one_theme_day_are_one_cluster(self):
        counts = validation._counts(self.rows())
        self.assertEqual(counts["independent_symbols"], 6)
        self.assertEqual(counts["clusters"], {"signal_days": 1, "theme_days": 1})
        self.assertIn("簇", counts["n_effective_note"])


class BriefShowsBooks(unittest.TestCase):
    def test_books_section_is_stamped_and_rendered(self):
        from .test_trading_brief import artefacts
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "results/mainwave_engine" / DAY.replace("-", "")
            folder.mkdir(parents=True)
            files = artefacts()
            files["books_daily.json"]["signal"]["trade_set"] = [{"symbol": "sh_600010", "participation_rank": 1}]
            files["books_daily.json"]["auto"].update(live=True, plans=1, fills=0,
                                                    placed=[{"symbol": "sh_600010", "status": "ORDER_APPROVED"}])
            for name, body in files.items():
                (folder / name).write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
            result = brief.build(DAY, root)
        section = result["sections"]["books"]
        self.assertEqual(section["stamp"]["historical"], "NOT_DONE")
        self.assertEqual(section["stamp"]["forward"], "RECORDING")
        self.assertFalse(result["screening_ability_proven"])
        text = brief.render(result)
        self.assertIn("五、三本账", text)
        self.assertIn("sh_600010#1", text)
        self.assertIn("sh_600010:ORDER_APPROVED", text)


if __name__ == "__main__":
    unittest.main()
