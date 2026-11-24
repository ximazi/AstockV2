"""Re-review 2026-09-21 boundaries (F01-F05): frozen same-day signal, untouched ledger on non-live days,
literal Top3, validated economic evidence with history, R3 out of every stratum."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from .trading import books, experiments, paper, store, validation
from .test_trading_system import AT, DAY, payload
from .test_trading_review_0921 import BODY, OOS

NEXT = "2026-09-21"
LATER = "2026-09-22T18:00:00+08:00"


def only_eligible(symbol):
    body = payload()
    for r in body["records"]:
        r["paper_plan_eligible"] = r["symbol"] == symbol
    return body


class Ledger(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / "t.sqlite3"
        self.data = self.root / "data"
        self.data.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def persist(self, body):
        with patch.object(store, "now", return_value=AT):
            return store.persist(body, self.db)

    def write_prices(self, symbols, end="2026-09-23"):
        days = pd.bdate_range(end=end, periods=160).strftime("%Y-%m-%d")
        for symbol in symbols:
            pd.DataFrame(dict(date=days, open=10.2, high=10.4, low=10., close=10.2, preclose=10.2, pct_chg=0.,
                              volume=1000000., amount=102000000., turnover=2., isST=0, tradestatus=1)
                         ).to_csv(self.data / f"{symbol}_daily.csv", index=False)
        pd.DataFrame(dict(trade_date=days, close=100.)).to_csv(self.data / "000300_SH_index.csv", index=False)

    def auto_counts(self):
        with paper.database(self.root / books.AUTO_DB) as conn:
            return {"plans": conn.execute("SELECT COUNT(*) FROM paper_plans").fetchone()[0],
                    "orders": conn.execute("SELECT COUNT(*) FROM paper_orders").fetchone()[0],
                    "sells": conn.execute("SELECT COUNT(*) FROM paper_orders WHERE side='SELL'").fetchone()[0],
                    "fills": conn.execute("SELECT COUNT(*) FROM paper_fills").fetchone()[0],
                    "valuations": conn.execute("SELECT COUNT(*) FROM paper_valuations").fetchone()[0],
                    "cash": conn.execute("SELECT cash_cents FROM paper_account").fetchone()[0],
                    "snapshots": conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]}


class F01SameDayRerunKeepsOneSource(Ledger):
    def test_signal_book_returns_the_frozen_entry_and_keeps_the_rerun_as_a_revision(self):
        first_body, second_body = only_eligible("sh_600013"), only_eligible("sh_600012")
        first, second = self.persist(first_body), self.persist(second_body)
        self.assertNotEqual(first["snapshot_id"], second["snapshot_id"])
        frozen = books.record_signal(DAY, first, self.db)
        self.assertIsNone(frozen["rerun"])
        again = books.record_signal(DAY, second, self.db)
        self.assertEqual(again["snapshot_id"], first["snapshot_id"])
        self.assertEqual([r["symbol"] for r in again["trade_set"]], ["sh_600013"])
        self.assertEqual(again["rerun"]["status"], "DIFFERENT_SNAPSHOT_KEPT_AS_REVISION")
        self.assertEqual(again["rerun"]["trade_set"], ["sh_600012"])
        with store.connect(self.db) as conn:
            stored = json.loads(conn.execute("SELECT payload FROM signal_book WHERE day=?", (DAY,)).fetchone()[0])
            revisions = conn.execute("SELECT COUNT(*) FROM signal_book_revisions WHERE day=?", (DAY,)).fetchone()[0]
        self.assertEqual(stored["snapshot_id"], first["snapshot_id"])
        self.assertEqual(revisions, 1)

    def test_auto_book_executes_the_frozen_signal_not_the_rerun_snapshot(self):
        first_body, second_body = only_eligible("sh_600013"), only_eligible("sh_600012")
        first, second = self.persist(first_body), self.persist(second_body)
        frozen = books.record_signal(DAY, first, self.db)
        books.record_signal(DAY, second, self.db)
        with patch.object(store, "now", return_value=AT):
            result = books.run_auto(DAY, second_body, second, self.root, self.data, live=True,
                                    signal=frozen, main_db=self.db)
        self.assertEqual(result["signal_source"], "FROZEN_SIGNAL_BOOK")
        self.assertEqual(result["snapshot_id"], first["snapshot_id"])
        self.assertEqual([(p["symbol"], p["status"]) for p in result["placed"]], [("sh_600013", "ORDER_APPROVED")])
        with paper.database(self.root / books.AUTO_DB) as conn:
            ids = {r[0] for r in conn.execute("SELECT id FROM snapshots")}
            plans = [json.loads(r[0]) for r in conn.execute("SELECT payload FROM paper_plans")]
        self.assertEqual(ids, {first["snapshot_id"]}, "自动账只认首发快照，重跑快照不得进入")
        self.assertEqual([p["symbol"] for p in plans], ["sh_600013"])
        # Running the same day again adds nothing: same frozen set, existing plan, no second batch.
        with patch.object(store, "now", return_value=AT):
            rerun = books.run_auto(DAY, second_body, second, self.root, self.data, live=True,
                                   signal=books.record_signal(DAY, second, self.db), main_db=self.db)
        self.assertEqual([p["status"] for p in rerun["placed"]], ["NOT_CREATED"])
        self.assertEqual(rerun["plans"], 1)


class F02NonLiveRunNeverTouchesTheLedger(Ledger):
    def seed_open_order(self):
        body = payload()
        snap = self.persist(body)
        self.write_prices([r["symbol"] for r in body["records"]])
        with patch.object(store, "now", return_value=AT):
            live = books.run_auto(DAY, body, snap, self.root, self.data, live=True)
        self.assertTrue(any(p["status"] == "ORDER_APPROVED" for p in live["placed"]), live["placed"])
        return body, snap

    def test_historical_rerun_matches_values_and_places_nothing(self):
        body, snap = self.seed_open_order()
        before = self.auto_counts()
        self.assertGreater(before["orders"], 0)
        self.assertEqual(before["fills"], 0)
        with patch.object(store, "now", return_value=LATER):
            result = books.run_auto(NEXT, body, snap, self.root, self.data, live=False)
        after = self.auto_counts()
        self.assertFalse(result["ledger_touched"])
        self.assertEqual(result["monitor"]["status"], "SKIPPED_NOT_LIVE")
        self.assertEqual({p["status"] for p in result["placed"]}, {"NOT_LIVE_DAY_NO_ORDER"})
        self.assertEqual(after, before, "非当日运行不得新增成交、退出、估值或改动现金")
        # The same day run live is what would have filled: the fixture is not vacuous.
        with patch.object(store, "now", return_value=NEXT + "T18:00:00+08:00"):
            live = books.run_auto(NEXT, body, snap, self.root, self.data, live=True)
        self.assertTrue(live["ledger_touched"])
        self.assertGreater(self.auto_counts()["fills"], 0)
        self.assertTrue(live["executions"])
        self.assertIn("unfilled_reason", live["executions"][0])


class F03TradeSetIsTheLiteralDefinition(Ledger):
    def test_validation_participation_top3_and_trade_set_are_the_same_names(self):
        body = payload()
        for r in body["records"]:
            r["paper_plan_eligible"] = True
        ranked = sorted(body["records"], key=lambda r: r["participation_rank"])
        ranked[0]["route_ids"], ranked[0]["alpha_scope"] = "R3", "R3_RESEARCH"
        chosen = {c["symbol"] for c in books.trade_set(body)}
        cohort = {r["symbol"] for r in body["records"]
                  if r["alpha_scope"] != "R3_RESEARCH" and r["participation_rank"] <= 3}
        self.assertEqual(chosen, cohort)
        self.assertEqual(len(chosen), 2)


class F04EconomicEvidenceIsValidatedAndKept(Ledger):
    def setUp(self):
        super().setUp()
        self.persist(payload())
        self.ident = experiments.register(BODY, self.db, at=AT)["id"]

    def test_non_finite_or_malformed_results_are_refused(self):
        for bad in ({**OOS, "n_effective": math.nan}, {**OOS, "n_effective": math.inf}, {**OOS, "n_effective": True},
                    {**OOS, "n_effective": 0}, {**OOS, "n_effective": 2.5}, {**OOS, "n_effective": "30"},
                    {**OOS, "expectancy_net": math.nan}, {**OOS, "max_drawdown": None},
                    {**OOS, "source": " "}, {**OOS, "oos_start": "someday"},
                    {**OOS, "oos_start": "2026-03-01", "oos_end": "2026-02-01"}):
            with self.assertRaises(ValueError, msg=str(bad)):
                experiments.economic_check(self.ident, bad, self.db, at=AT)
        with store.connect(self.db) as conn:
            saved = experiments._load(conn, self.ident)
        self.assertIsNone(saved["economic_check"], "被拒的结果不得留下任何验收状态")

    def test_registration_refuses_non_finite_thresholds(self):
        for bad in ({"expectancy_net_min": math.nan, "n_effective_min": 20},
                    {"expectancy_net_min": 0, "n_effective_min": 0},
                    {"expectancy_net_min": 0, "n_effective_min": 2.5},
                    {"expectancy_net_min": 0, "n_effective_min": True}):
            with self.assertRaises(ValueError, msg=str(bad)):
                experiments.register({**BODY, "hypothesis": str(bad), "oos_acceptance": bad}, self.db, at=AT)

    def test_every_check_is_appended_and_bound_to_its_evidence(self):
        thin = experiments.economic_check(self.ident, {**OOS, "n_effective": 5}, self.db, at=AT)
        self.assertEqual(thin["economic_check"]["status"], "INSUFFICIENT_SAMPLE")
        passed = experiments.economic_check(self.ident, {**OOS, "oos_end": "2026-06-30"}, self.db, at=AT)
        history = passed["economic_checks"]
        self.assertEqual([c["status"] for c in history], ["INSUFFICIENT_SAMPLE", "PASS"])
        self.assertEqual([c["n"] for c in history], [1, 2])
        latest = passed["economic_check"]
        self.assertEqual(latest["evidence_sha256"], store.digest({**OOS, "oos_end": "2026-06-30"}))
        self.assertTrue(latest["rule_fingerprint"])
        self.assertEqual(latest["oos_window"], {"start": "2026-01-05", "end": "2026-06-30"})
        self.assertEqual(latest["source"], "AUTO_BOOK")


class F05ResearchRowsStayOutOfEveryStratum(Ledger):
    def test_r3_only_rows_have_their_own_strata(self):
        pd.DataFrame({"trade_date": ["2026-09-17", DAY]}).to_csv(self.data / "000300_SH_index.csv", index=False)
        older = payload()
        older["as_of"] = "2026-09-17"
        older["records"][0]["route_ids"], older["records"][0]["alpha_scope"] = "R3", "R3_RESEARCH"
        self.persist(older)
        review = validation.evaluate(DAY, self.db, self.data)
        price_routes = {c["group"]: c for c in review["strata"]["route"]}
        research_routes = {c["group"]: c for c in review["strata_research"]["route"]}
        self.assertNotIn("route=R3", price_routes)
        self.assertEqual(research_routes["route=R3"]["alpha_scope"], "R3_RESEARCH")
        self.assertIn("仅研究", research_routes["route=R3"]["note"])
        self.assertTrue(all(c["alpha_scope"] == "PRICE_ALPHA" for c in price_routes.values()))
        # The shared theme bucket counts only price rows in the price domain; the R3 row sits in its own copy.
        theme_price = review["strata"]["theme"][0]["selected_n"]
        theme_research = review["strata_research"]["theme"][0]["selected_n"]
        self.assertEqual(theme_price + theme_research, len(older["records"]))
        self.assertEqual(theme_research, 1)


class AutoBookEvidenceComesFromTheLedger(Ledger):
    def test_effective_config_is_read_back_and_compared_with_code(self):
        body = payload()
        snap = self.persist(body)
        with patch.object(store, "now", return_value=AT):
            result = books.run_auto(DAY, body, snap, self.root, self.data, live=False)
        self.assertTrue(result["account_matches_code"])
        self.assertEqual(result["effective_account"]["capital"], books.AUTO_ACCOUNT["capital"])
        self.assertTrue(result["config_fingerprint"])
        with paper.database(self.root / books.AUTO_DB) as conn:
            cfg = json.loads(conn.execute("SELECT config FROM paper_account").fetchone()[0])
            cfg["single_risk_pct"] = 5.0
            conn.execute("UPDATE paper_account SET config=? WHERE id=1", (store.encoded(cfg),))
        with patch.object(store, "now", return_value=AT):
            drifted = books.run_auto(DAY, body, snap, self.root, self.data, live=False)
        self.assertFalse(drifted["account_matches_code"], "报告打印的是数据库里实际生效的配置，不是代码常量")


if __name__ == "__main__":
    unittest.main()
