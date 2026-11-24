"""Batch A invariants: self-check tiering, continuity, route cards, rank explanation, validation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from flask import Flask

from .trading import (continuity, diagnostics, explain, preflight, ranking,
                      store, validation, web)
from .test_trading_system import AT, inputs, payload

DAY = "2026-09-18"
CALENDAR = ["2026-09-15", "2026-09-16", "2026-09-17", DAY, "2026-09-21", "2026-09-22",
            "2026-09-23", "2026-09-24", "2026-09-25"]


def bars(dates, closes=None):
    closes = closes or [10.0 + i for i in range(len(dates))]
    return pd.DataFrame({"date": dates, "close": closes,
                         "pct_chg": [0.0] + [1.0] * (len(dates) - 1)})


class PreflightTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "data").mkdir()
        (self.root / "results/mainwave_engine" / DAY.replace("-", "")).mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def write_inputs(self, symbols=("sh_600010",), price_dates=None, gapped=()):
        folder = self.root / "results/mainwave_engine" / DAY.replace("-", "")
        records = [{"symbol": s, "belonging_theme": "test"} for s in symbols]
        for name in ("analysis_universe.json", "analysis_evidence.json"):
            (folder / name).write_text(json.dumps({"as_of": DAY, "records": records}), encoding="utf-8")
        pd.DataFrame({"trade_date": CALENDAR}).to_csv(self.root / "data/000300_SH_index.csv", index=False)
        for symbol in symbols:
            dates = [d for d in (price_dates or CALENDAR[:4]) if not (symbol in gapped and d == CALENDAR[1])]
            frame = pd.DataFrame({"date": dates, "adjustflag": 3, "volume": 1000,
                                  "close": 10.0, "amount": 10000.0, "turnover": 1.0})
            frame.to_csv(self.root / f"data/{symbol}_daily.csv", index=False)

    def test_missing_required_input_blocks_and_names_the_gap(self):
        result = preflight.check(DAY, self.root, self.root / "db.sqlite3", self.root / "data")
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("calendar", result["blocking"])
        self.assertIn("universe", result["blocking"])

    def test_calendar_preflight_fails_for_stale_file_and_passes_after_refresh(self):
        path = self.root / "data/000300_SH_index.csv"
        path.write_text("trade_date\n20260917\ninvalid\n", encoding="utf-8")
        with patch.object(store, "ROOT", self.root), patch("sys.argv", ["preflight", "--date", DAY]), patch("builtins.print") as log:
            self.assertEqual(preflight.main(), 1)
            self.assertIn("2026-09-17", log.call_args_list[0].args[0])
            self.assertIn("--end-date 20260918", log.call_args_list[1].args[0])
            path.write_text("trade_date\n20260917\n2026-09-18\n", encoding="utf-8")
            self.assertEqual(preflight.main(), 0)

    def test_malformed_index_file_is_a_calendar_failure(self):
        path = self.root / "data/000300_SH_index.csv"
        path.write_text("close\n5000\n", encoding="utf-8")
        _, calendar = preflight._calendar(path.parent, DAY)
        self.assertEqual(calendar["status"], "FAIL")
        self.assertIn(path.name, calendar["detail"])

    def test_per_symbol_gap_degrades_without_blocking_the_run(self):
        self.write_inputs(symbols=("sh_600010", "sh_600011"), gapped=("sh_600011",))
        result = preflight.check(DAY, self.root, self.root / "db.sqlite3", self.root / "data")
        self.assertEqual(result["status"], "DEGRADED")
        self.assertEqual(result["blocking"], [])
        self.assertIn("bars_coverage", result["degraded"])
        self.assertEqual(result["data_wait"]["counts"]["gapped"], 1)
        self.assertEqual(result["data_wait"]["counts"]["ready"], 1)

    def test_no_usable_candidate_blocks_the_closure(self):
        self.write_inputs(symbols=("sh_600010",), gapped=("sh_600010",))
        result = preflight.check(DAY, self.root, self.root / "db.sqlite3", self.root / "data")
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIn("bars", result["blocking"])

    def test_contract_reports_units_and_adjust_basis(self):
        self.write_inputs()
        result = preflight.check(DAY, self.root, self.root / "db.sqlite3", self.root / "data")
        contract = next(c for c in result["checks"] if c["id"] == "contract")
        self.assertEqual(contract["status"], "PASS")
        self.assertEqual(result["contract"]["price_basis"], "RAW_UNADJUSTED")

    def test_package_round_trip_rejects_a_tampered_source(self):
        (self.root / "talk/transcripts").mkdir(parents=True)
        (self.root / "02_知识点覆盖与待验证.md").write_text("register", encoding="utf-8")
        source = self.root / "talk/transcripts/a.transcript.reviewed.md"
        source.write_text("original", encoding="utf-8")
        package = self.root / "pkg"
        preflight.export_package(self.root, package)
        target = self.root / "restored"
        (target / "talk/transcripts").mkdir(parents=True)
        self.assertEqual(preflight.import_package(package, target)["imported"], 1)
        (package / "talk/transcripts/a.transcript.reviewed.md").write_text("tampered", encoding="utf-8")
        with self.assertRaises(ValueError):
            preflight.import_package(package, target)


class ContinuityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "t.sqlite3"

    def tearDown(self):
        self.temp.cleanup()

    def snapshot(self, day, count=4, cycle="test-cycle"):
        universe, evidence, metrics = inputs(count)
        universe["as_of"] = evidence["as_of"] = day
        for row in universe["records"]:
            row["price_date"] = day
            row["cycle_id"] = cycle
        built = ranking.build(universe, evidence, metrics, {"return5": 0., "return20": 0.})
        built["knowledge_registry_id"] = "fixture"
        with patch.object(store, "now", return_value=AT):
            return store.persist(built, self.db)

    def test_first_day_discovers_and_rerun_is_idempotent(self):
        snap = self.snapshot(DAY)
        first = continuity.advance(snap, self.db, CALENDAR)
        again = continuity.advance(snap, self.db, CALENDAR)
        self.assertEqual(first["summary"]["by_event"]["DISCOVERED"], 4)
        self.assertEqual(first["summary"], again["summary"])

    def test_absent_candidate_is_kept_as_unseen_then_expires(self):
        continuity.advance(self.snapshot("2026-09-15", count=4), self.db, CALENDAR)
        later = continuity.advance(self.snapshot("2026-09-16", count=2), self.db, CALENDAR)
        unseen = [e for e in later["events"] if e["event"] == "TEMPORARILY_UNSEEN"]
        self.assertEqual(len(unseen), 2)
        self.assertEqual(unseen[0]["absence"], "NOT_SCANNED")
        # The counter must run from the last day actually seen, not reset on each absence.
        reasons, expired = [], []
        for day in CALENDAR[2:7]:
            result = continuity.advance(self.snapshot(day, count=2), self.db, CALENDAR)
            for event in result["events"]:
                if event["event"] == "TEMPORARILY_UNSEEN":
                    reasons.append(event["reason"])
                if event["event"] == "EXPIRED":
                    expired.append(event)
        self.assertTrue(expired, "未见记录必须在到达期限后过期")
        self.assertIn("第 4 个交易日", " ".join(reasons))
        self.assertIn(f"连续 {continuity.UNSEEN_LIMIT} 个交易日", expired[0]["reason"])

    def test_new_cycle_starts_a_new_opportunity_linked_to_the_old_round(self):
        continuity.advance(self.snapshot("2026-09-16", cycle="cycle-a"), self.db, CALENDAR)
        rotated = continuity.advance(self.snapshot("2026-09-17", cycle="cycle-b"), self.db, CALENDAR)
        events = {e["event"] for e in rotated["events"]}
        self.assertIn("REDISCOVERED", events)
        self.assertIn("CYCLE_ROTATED", events)
        fresh = next(e for e in rotated["events"] if e["event"] == "REDISCOVERED")
        self.assertTrue(fresh["previous_opportunity_id"])
        self.assertNotEqual(fresh["opportunity_id"], fresh["previous_opportunity_id"])
        line = continuity.timeline(fresh["opportunity_id"], self.db)
        self.assertEqual(line["previous_round"]["opportunity_id"], fresh["previous_opportunity_id"])

    def test_new_cycle_does_not_inherit_the_previous_rank(self):
        first = self.snapshot("2026-09-16", cycle="cycle-a")
        universe, evidence, metrics = inputs(4)
        universe["as_of"] = evidence["as_of"] = "2026-09-17"
        for row in universe["records"]:
            row["price_date"] = "2026-09-17"
            row["cycle_id"] = "cycle-b"
        rebuilt = ranking.build(universe, evidence, metrics, {"return5": 0., "return20": 0.}, first)
        self.assertTrue(all(r["previous_rank"] is None for r in rebuilt["records"]))


class DiagnosticsTest(unittest.TestCase):
    def record(self, route="R1", **evidence):
        base = {"eligible": True, "route_ids": route, "retained": True, "demand": True,
                "supply": False, "bias_atr": 1.0, "strict_chain_verified": True,
                "prepared_at": "2026-09-16", "probe_at": "2026-09-15", "attempt_at": "2026-09-17",
                "accepted_at": DAY, "preparation_valid": True}
        base.update(evidence)
        return {"symbol": "sh_600010", "name": "Fixture", "opportunity_id": "oid", "theme": "test",
                "route_ids": route, "setup_kind": "FIRST_PULLBACK", "next_condition": "等待",
                "evidence": {"route_evidence": base, "state_vector": {},
                             "checks": [{"id": k, "status": "PASS"} for k in
                                        ("breakout", "confirmation", "boundary", "route_context",
                                         "route_demand", "route_retention", "route_supply")],
                             "review_window": {"active": True}},
                "source_record": {"breakout_status": "ADVANCING"}}

    def test_missing_evidence_is_unknown_not_a_failure(self):
        card = diagnostics.card(self.record(retained=None))
        step = next(s for s in card["cards"][0]["steps"] if s["id"] == "retention")
        self.assertEqual(step["status"], "UNKNOWN")
        self.assertEqual(card["cards"][0]["verdict"], "INSUFFICIENT_DATA")

    def test_waiting_check_does_not_read_as_failed(self):
        row = self.record()
        row["evidence"]["checks"] = [{"id": "breakout", "status": "WAIT"}]
        card = diagnostics.card(row)
        entry = card["cards"][0]
        self.assertEqual(entry["verdict"], "WAITING")
        self.assertEqual(entry["stopped_status"], "WAIT")

    def test_first_blocking_step_is_reported_for_a_failure(self):
        card = diagnostics.card(self.record(supply=True))
        entry = card["cards"][0]
        self.assertEqual(entry["verdict"], "FAILED")
        self.assertEqual(entry["first_blocking_step"], "supply_warning")

    def test_overlapping_routes_keep_separate_answers(self):
        card = diagnostics.card(self.record(route="R2|R1"))
        self.assertEqual({c["route"] for c in card["cards"]}, {"R1", "R2"})

    def test_report_counts_denominators_per_declared_route(self):
        report = diagnostics.report({"as_of": DAY, "records": [self.record(), self.record(route="R3")]})
        by_route = {r["route"]: r["total"] for r in report["routes"]}
        self.assertEqual(by_route, {"R1": 1, "R3": 1})
        self.assertEqual(report["summary"]["without_route"], 0)

    def test_zero_ready_targets_is_still_a_valid_report(self):
        report = diagnostics.report({"as_of": DAY, "records": [self.record(supply=True)]})
        self.assertEqual(report["routes"][0]["verdicts"], {"FAILED": 1})


class ExplainTest(unittest.TestCase):
    def test_rank_is_reconstructable_from_components_and_domain(self):
        snap = payload()
        best = next(r for r in snap["records"] if r["strength_rank"] == 1)
        result = explain.explain(best, snap)
        self.assertEqual({c["component"] for c in result["strength"]["contributions"]},
                         {"relative_momentum", "drawdown20", "efficiency10"})
        self.assertTrue(result["comparison_domain"]["self_excluded"])
        self.assertIsNotNone(result["comparison_domain"]["board_count"])

    def test_strong_but_blocked_is_named(self):
        universe, evidence, metrics = inputs(4)
        universe["market"]["trend_review_allowed"] = False
        snap = ranking.build(universe, evidence, metrics, {"return5": 0., "return20": 0.})
        best = next(r for r in snap["records"] if r["strength_rank"] == 1)
        result = explain.explain(best, snap)
        self.assertEqual(result["verdict"], "STRONG_BUT_NOT_ACTIONABLE")
        self.assertIn("MARKET_WAIT", [b["code"] for b in result["participation"]["blockers"]])

    def test_score_up_but_rank_down_is_attributed_to_the_cohort(self):
        previous = payload()
        current = payload()
        target = next(r for r in current["records"] if r["symbol"] == "sh_600010")
        before = next(r for r in previous["records"] if r["symbol"] == "sh_600010")
        before["strength_score"] = (target["strength_score"] or 0) - 5
        target["previous_rank"] = (target["strength_rank"] or 2) - 1
        target["rank_change"] = -1
        result = explain.explain(target, current, previous)
        self.assertEqual(result["change"]["attribution"], "COHORT_STRENGTHENED")

    def test_top_lists_deduplicate_symbols_and_keep_theme_links(self):
        snap = payload()
        result = explain.board(snap, limit=3)
        symbols = [r["symbol"] for r in result["strength_top"]]
        self.assertEqual(len(symbols), len(set(symbols)))
        self.assertTrue(all(r["themes"] for r in result["strength_top"]))

    def test_observed_features_are_disclosed_as_outside_the_score(self):
        snap = payload()
        result = explain.explain(snap["records"][0], snap)
        self.assertTrue(all(not v["in_score"] for v in result["strength"]["observed_not_scored"].values()))
        self.assertIn("pearson", result and explain.collinearity(snap)["turnover_vs_relative_turnover"])


class ValidationTest(unittest.TestCase):
    def test_window_uses_market_days_and_a_halt_breaks_maturity(self):
        full = bars(CALENDAR[:5])
        self.assertEqual(validation.window(full, CALENDAR, CALENDAR[0], 3)["status"], "MATURE")
        halted = full[full.date != CALENDAR[2]]
        gap = validation.window(halted, CALENDAR, CALENDAR[0], 3)
        self.assertEqual(gap["status"], "DATA_GAP")
        self.assertEqual(gap["missing_days"], [CALENDAR[2]])

    def test_pending_market_days_are_immature_not_zero(self):
        result = validation.window(bars(CALENDAR), CALENDAR, CALENDAR[-2], 5)
        self.assertEqual(result["status"], "IMMATURE")
        self.assertIsNone(result.get("return_pct"))

    def test_missing_calendar_waits_instead_of_inventing_a_window(self):
        self.assertEqual(validation.window(bars(CALENDAR), [], CALENDAR[0], 3)["status"], "DATA_WAIT")

    def test_independent_counting_separates_rows_from_opportunities(self):
        rows = [{"status": "MATURE", "signal_day": "2026-09-16", "symbol": "sh_600010",
                 "opportunity_id": "a", "cycle_id": "c1", "return_pct": 1.0},
                {"status": "MATURE", "signal_day": "2026-09-17", "symbol": "sh_600010",
                 "opportunity_id": "a", "cycle_id": "c1", "return_pct": 2.0}]
        counts = validation._counts(rows)
        self.assertEqual(counts["rows"], 2)
        self.assertEqual(counts["independent_opportunities"], 1)
        self.assertEqual(counts["independent_days"], 2)

    def test_insufficient_stratum_keeps_its_status(self):
        rows = [{"status": "MATURE", "signal_day": DAY, "version": "v", "symbol": "sh_600010",
                 "opportunity_id": "a", "cycle_id": "c1", "return_pct": 1.0}]
        result = validation._cohort("route=R1", rows, rows[0], 5, stratum=True)
        self.assertEqual(result["status"], "INSUFFICIENT_SAMPLE_RETAINED")
        self.assertFalse(result["sufficient"])

    def test_ablation_refuses_to_substitute_another_day(self):
        with tempfile.TemporaryDirectory() as temp:
            result = validation.ablate(DAY, Path(temp))
        self.assertEqual(result["status"], "PROTOCOL_DAY_MISSING")
        self.assertFalse(result.get("baseline_top"))

    def test_missed_audit_waits_until_the_posterior_window_matures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").mkdir()
            pd.DataFrame({"trade_date": CALENDAR}).to_csv(root / "data/000300_SH_index.csv", index=False)
            db = root / "t.sqlite3"
            with patch.object(store, "now", return_value=AT):
                store.persist(payload(), db)
            result = validation.missed_audit(DAY, db, root / "data")
        self.assertEqual(result["status"], "WINDOW_IMMATURE")
        self.assertEqual(result["cases"], [])


class BatchAWebTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "t.sqlite3"
        with patch.object(store, "now", return_value=AT):
            self.snap = store.persist(payload(), self.db)
        self.app = Flask(__name__, template_folder=str(store.ROOT / "templates"))
        self.app.config["TRADING_DB"] = self.db
        self.app.register_blueprint(web.bp)

    def tearDown(self):
        self.temp.cleanup()

    def test_new_panels_answer_without_a_frozen_artefact(self):
        client = self.app.test_client()
        cards = client.get("/api/trading/diagnostics").get_json()
        self.assertIn("routes", cards)
        self.assertTrue(client.get("/api/trading/preflight").get_json()["status"])
        audit = client.get("/api/trading/validation/audit").get_json()
        self.assertIn(audit["status"], {"READY", "NOT_RUN", "NO_SNAPSHOT"})

    def test_detail_carries_continuity_explanation_and_diagnosis(self):
        identity = self.snap["records"][0]["opportunity_id"]
        body = self.app.test_client().get(f"/api/trading/opportunities/{identity}").get_json()
        self.assertIn("events", body["continuity"])
        self.assertIn("strength", body["explanation"])
        self.assertIn("cards", body["diagnosis"])


if __name__ == "__main__":
    unittest.main()
