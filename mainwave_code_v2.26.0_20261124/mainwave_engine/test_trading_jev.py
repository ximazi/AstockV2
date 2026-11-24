"""Jev shadow layer: typed questions, computed-only state, honest failure, no influence on ranking or ledger."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from .trading import jev, store, validation
from .test_trading_system import AT, DAY, payload


class FakeClient:
    """Answers in the exact shape the live API returned on 2026-09-21 (choice / noul / score with legend)."""

    def __init__(self, fail_symbols=()):
        self.calls, self.fail = [], set(fail_symbols)

    def ask(self, state, questions, model=jev.MODEL):
        self.calls.append((state, sorted(questions)))
        if state.get("return20_pct") in self.fail:
            raise jev.JevError("HTTP 503: upstream")
        answers = {}
        for name, q in questions.items():
            if q["type"] == "choice":
                answers[name] = {"type": "choice", "choice": "LATE", "confidence": .25,
                                 "probabilities": {"EARLY": .05, "OPTIMAL": .30, "LATE": .55, "EXHAUSTED": .10}}
            elif q["type"] == "score":
                labels = q["criteria"]
                probs = {str(i): (0.7 if i == len(labels) - 1 else 0.3 / (len(labels) - 1)) for i in range(len(labels))}
                answers[name] = {"type": "score", "score": len(labels) - 1.3, "confidence": .5,
                                 "legend": {str(i): l for i, l in enumerate(labels)}, "probabilities": probs}
            else:
                answers[name] = {"type": "noul", "noul": .9 if name == "Q04_adverse_evidence_material" else .2}
        return {"model": "jev-1.13.0", "answers": answers, "usage": {"input_tokens": 500, "output_tokens": 40}, "latency_ms": 900}


class Questions(unittest.TestCase):
    def test_question_sets_are_typed_contracts(self):
        for qs in (jev.GLOBAL_QUESTIONS, jev.R1_QUESTIONS, jev.R2_QUESTIONS):
            self.assertEqual(jev.validate_questions(qs), [])
        self.assertEqual(sorted(jev.question_set("R1|R2")), sorted({**jev.GLOBAL_QUESTIONS, **jev.R1_QUESTIONS, **jev.R2_QUESTIONS}))
        self.assertEqual(sorted(jev.question_set("R3")), sorted(jev.GLOBAL_QUESTIONS))
        bad = {"x": {"type": "score", "instructions": "s", "criteria": {"LOW": "a"}}}
        self.assertTrue(any("有序列表" in p for p in jev.validate_questions(bad)))

    def test_state_holds_only_computed_evidence(self):
        snap = payload()
        snap["market"]["context"] = "BEAR"
        record = snap["records"][0]
        record["route_ids"] = "R1"
        state = jev.state_of(record, snap)
        self.assertNotIn("symbol", state)
        self.assertNotIn("name", state)
        for key in ("market_context", "routes", "engine_state", "drawdown20_pct", "return20_pct", "strength_percentiles", "_notes"):
            self.assertIn(key, state)
        self.assertEqual(jev.state_of(record, snap), state, "同一输入同一状态，哈希可复用")
        self.assertTrue(all(not isinstance(v, float) or round(v, 3) == v for v in state.values() if isinstance(v, float)))

    def test_normalize_maps_score_legend_and_noul(self):
        client = FakeClient()
        questions = jev.question_set("R1")
        raw = client.ask({"return20_pct": 1.0}, questions)
        out = jev.normalize(raw["answers"], questions)
        self.assertEqual(out["Q01_setup_quality"]["label"], "EXCEPTIONAL")
        self.assertEqual(out["Q01_setup_quality"]["probabilities"]["EXCEPTIONAL"], .7)
        self.assertEqual(out["Q02_entry_stage"]["label"], "LATE")
        self.assertEqual(out["Q04_adverse_evidence_material"]["label"], "YES")
        self.assertAlmostEqual(out["Q04_adverse_evidence_material"]["confidence"], .8)
        self.assertEqual(out["Q11_breakout_quality"]["label"], "STRONG")


class Shadow(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "t.sqlite3"
        with patch.object(store, "now", return_value=AT):
            self.snap = store.persist(payload(), self.db)

    def tearDown(self):
        self.temp.cleanup()

    def test_shadow_records_rows_without_touching_ranks_or_ledger(self):
        before = json.dumps([(r["symbol"], r["strength_rank"], r["participation_rank"], r["state"]) for r in self.snap["records"]])
        client = FakeClient()
        result = jev.run_shadow(self.snap, self.db, client=client, at=AT)
        self.assertEqual(result["status"], "RECORDED")
        self.assertEqual(len(result["rows"]), len(self.snap["records"]))
        self.assertEqual(json.dumps([(r["symbol"], r["strength_rank"], r["participation_rank"], r["state"]) for r in self.snap["records"]]), before)
        row = result["rows"][0]
        self.assertEqual(row["question_set"], jev.QUESTION_SET_VERSION)
        self.assertEqual(row["model"], "jev-1.13.0")
        self.assertTrue(row["state_hash"])
        self.assertTrue(row["flags"]["would_review"], "Q04=0.9 只产生影子标记")
        self.assertIn("不改排名", result["notice"])
        self.assertEqual(result["summary"]["cost_usd"], round(500 * len(result["rows"]) / 1e6 * jev.PRICE_PER_MILLION_INPUT_USD, 5))
        with store.connect(self.db) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM jev_assessments").fetchone()[0], len(result["rows"]))
            self.assertIsNone(conn.execute("SELECT name FROM sqlite_master WHERE name='paper_orders'").fetchone())

    def test_route_questions_follow_the_declared_routes_and_r3_stays_research(self):
        records = self.snap["records"]
        records[0]["route_ids"], records[1]["route_ids"] = "R1", "R2"
        records[2]["route_ids"], records[2]["alpha_scope"] = "R3", "R3_RESEARCH"
        client = FakeClient()
        result = jev.run_shadow(self.snap, self.db, client=client, at=AT)
        asked = {tuple(q) for _, q in client.calls}
        self.assertIn(tuple(sorted({**jev.GLOBAL_QUESTIONS, **jev.R1_QUESTIONS})), asked)
        self.assertIn(tuple(sorted({**jev.GLOBAL_QUESTIONS, **jev.R2_QUESTIONS})), asked)
        self.assertEqual(result["skipped"], {"R3_RESEARCH": 1})
        self.assertNotIn(records[2]["symbol"], [r["symbol"] for r in result["rows"]])

    def test_same_state_is_cached_and_not_billed_twice(self):
        client = FakeClient()
        first = jev.run_shadow(self.snap, self.db, client=client, at=AT)
        n_calls = len(client.calls)
        second = jev.run_shadow(self.snap, self.db, client=client, at=AT)
        self.assertEqual(len(client.calls), n_calls)
        self.assertEqual(second["summary"]["cached"], len(second["rows"]))
        self.assertEqual(second["summary"]["input_tokens"], 0)
        self.assertEqual(first["rows"][0]["id"], second["rows"][0]["id"])

    def test_unavailable_or_failing_client_never_invents_a_probability(self):
        with patch.dict(os.environ, {"JEV_API_KEY": ""}), patch("dotenv.load_dotenv", lambda *a, **k: None):
            off = jev.run_shadow(self.snap, self.db, client=None, at=AT)
        self.assertEqual(off["status"], "UNAVAILABLE")
        self.assertEqual(off["rows"], [])
        self.assertIn("JEV_API_KEY", off["reason"])
        failing = FakeClient(fail_symbols={jev.state_of(self.snap["records"][0], self.snap).get("return20_pct")})
        partial = jev.run_shadow(self.snap, self.db, client=failing, at=AT)
        self.assertEqual(partial["status"], "PARTIAL")
        errored = [r for r in partial["rows"] if r["status"] == "ERROR"]
        self.assertTrue(errored)
        self.assertEqual(errored[0]["answers"], {})
        self.assertIn("503", errored[0]["error"])
        with store.connect(self.db) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM jev_assessments").fetchone()[0], len(partial["rows"]) - len(errored))

    def test_render_carries_the_notice(self):
        text = jev.render(jev.run_shadow(self.snap, self.db, client=FakeClient(), at=AT))
        self.assertIn("不是上涨概率", text)
        self.assertIn("would_review", text)


class Evaluation(unittest.TestCase):
    def test_judgments_join_price_observations_and_report_layering_only_when_mature(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").mkdir()
            db = root / "t.sqlite3"
            days = pd.bdate_range("2026-08-01", DAY).strftime("%Y-%m-%d").tolist()
            pd.DataFrame({"trade_date": days}).to_csv(root / "data/000300_SH_index.csv", index=False)
            older = payload()
            older["as_of"] = "2026-08-05"
            for r in older["records"]:
                pd.DataFrame({"date": days, "pct_chg": [0.5] * len(days), "close": [10 + i * .05 for i in range(len(days))]}).to_csv(
                    root / f"data/{r['symbol']}_daily.csv", index=False)
            with patch.object(store, "now", return_value=AT):
                snap = store.persist(older, db)
            jev.run_shadow(snap, db, client=FakeClient(), at=AT)
            review = jev.evaluate(DAY, db, root / "data")
        self.assertGreater(review["judged_rows"], 0)
        self.assertEqual(review["judged_opportunities"], len(older["records"]))
        self.assertEqual(review["mature_rows"], review["judged_rows"], "08-05 信号到 09-18 已成熟")
        q1 = review["by_question"]["Q01_setup_quality"]["5"]
        self.assertEqual(q1["EXCEPTIONAL"]["selected_n"], len(older["records"]))
        self.assertEqual(q1["_layering"], "INSUFFICIENT_SAMPLE", "4 只不够 20 的分桶门槛，不得宣称分层")
        self.assertIn("不是上涨概率", review["notice"])
        self.assertIn("theme_days", review["clusters"])


class LiveContract(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("JEV_LIVE_TEST") == "1", "set JEV_LIVE_TEST=1 to hit the real API once")
    def test_live_response_shape(self):
        client = jev.Client()
        raw = client.ask({"market_context": "BEAR", "retained": True}, jev.question_set("R1"))
        out = jev.normalize(raw["answers"], jev.question_set("R1"))
        self.assertEqual(set(out), set(jev.question_set("R1")))
        self.assertTrue(all(0 <= p <= 1 for a in out.values() for p in a["probabilities"].values()))


if __name__ == "__main__":
    unittest.main()
