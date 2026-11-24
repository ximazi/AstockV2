"""Daily deliverable invariants: assembled from artefacts, stamped from evidence, never a proof by default."""
import json
from pathlib import Path
import tempfile
import unittest

from .trading import brief

DAY = "2026-09-18"


LIVE = "FORWARD_RECORDED_AFTER_CLOSE"
REBUILT = "DATED_SOURCE_RECONSTRUCTION_NOT_PROOF_OF_LIVE_DISCOVERY"


def artefacts(day=DAY, mature=0, rows=2, protocols=True, drop=(), basis=LIVE, rebuilt_rows=0):
    top = [{"symbol": "sh_600010", "name": "甲", "opportunity_id": "o1", "theme": "t", "route_ids": "R1",
            "state": "MARKET_WAIT", "strength_rank": 1, "participation_rank": 3, "strength_score": 90.0,
            "rank_change": -2, "blockers": ["MARKET_WAIT"]},
           {"symbol": "sh_600011", "name": "乙", "opportunity_id": "o2", "theme": "t", "route_ids": "",
            "state": "PLAN_REVIEW", "strength_rank": 2, "participation_rank": 1, "strength_score": 80.0,
            "rank_change": None, "blockers": []}]
    board = {"as_of": day, "snapshot_id": "snap", "strength_top": top, "participation_top": list(reversed(top)),
             "explained": {"o1": {"verdict": "STRONG_BUT_NOT_ACTIONABLE", "verdict_reading": "市场闸门",
                                  "change": {"attribution": "COHORT_STRENGTHENED", "reading": "比较域更强"}},
                           "o2": {"verdict": "ACTIONABLE_FOR_REVIEW", "verdict_reading": "可复核",
                                  "change": {"attribution": "NO_PRIOR_ROUND", "reading": ""}}}}
    cards = {"cards": [{"symbol": "sh_600010", "cards": [{"route": "R1", "verdict": "FAILED", "first_blocking_step": "retention",
                                                           "stopped_status": "FAIL", "next_condition": "等待新冲击", "steps": []}]},
                       {"symbol": "sh_600011", "cards": []}],
             "routes": [{"route": "R1", "label": "反查", "total": 1, "verdicts": {"FAILED": 1},
                         "first_failures": {"retention:FAIL": 1}, "cases": {"FAILED": [{"symbol": "sh_600010"}]}}]}
    chain = {"summary": {"by_event": {"DISCOVERED": 2}},
             "events": [{"opportunity_id": "o1", "event": "WAITING", "reason": "市场闸门"},
                        {"opportunity_id": "o2", "event": "DISCOVERED", "reason": "首次"}]}
    live = {"rows": rows, "mature_rows": mature, "immature_rows": rows - mature, "independent_days": 1 if rows else 0}
    by_basis = {basis: live}
    if rebuilt_rows:
        by_basis[REBUILT] = {"rows": rebuilt_rows, "mature_rows": rebuilt_rows, "immature_rows": 0, "independent_days": 3}
    review = {"horizons": [5, 10],
              "summary": {"rows": rows + rebuilt_rows, "mature_rows": mature + rebuilt_rows,
                          "immature_rows": rows - mature, "by_basis": by_basis},
              "comparisons": [{"group": g, "horizon": 5, "signal_day": day, "basis": basis, "selected_n": 3, "mature_n": mature,
                               "independent_opportunities": 3, "mean": 1.0 if mature else None,
                               "positive_fraction": .5 if mature else None, "worst": -1.0 if mature else None,
                               "complete": bool(mature)} for g in ("strength_top3", "all_candidates")]}
    audit = {"protocols": {"protocols": [{"protocol": "p", "frozen_days": 7 if protocols else 0}]},
             "missed": {"status": "WINDOW_IMMATURE", "rule": {"definition": "20日≥25%"}, "summary": None,
                        "cases": [], "notice": "未成熟"}}
    ledgers = {"signal": {"trade_set": [], "eligible": 0, "excluded_r3_research": 0},
               "auto": {"live": False, "plans": 0, "fills": 0, "open_positions": 0, "placed": [], "reconciliation": "UNCONFIGURED"},
               "manual": {"status": "UNCONFIGURED", "performance": "UNCONFIGURED"}}
    files = {"ranking_explained.json": board, "route_diagnostics.json": cards, "opportunity_continuity.json": chain,
             "ranking_validation.json": review, "validation_audit.json": audit, "books_daily.json": ledgers}
    return {k: v for k, v in files.items() if k not in drop}


class BriefTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, day, files):
        folder = self.root / "results/mainwave_engine" / day.replace("-", "")
        folder.mkdir(parents=True, exist_ok=True)
        for name, payload in files.items():
            (folder / name).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def test_four_sections_each_carry_three_independent_stamps(self):
        self.write(DAY, artefacts())
        result = brief.build(DAY, self.root)
        self.assertEqual(set(result["sections"]), {"top", "diagnosis", "cross_day", "outcomes", "books"})
        for section in result["sections"].values():
            stamp = section["stamp"]
            self.assertIn(stamp["software"], brief.STATUS["software"])
            self.assertIn(stamp["historical"], brief.STATUS["historical"])
            self.assertIn(stamp["forward"], brief.STATUS["forward"])

    def test_output_alone_is_never_proof(self):
        self.write(DAY, artefacts())
        result = brief.build(DAY, self.root)
        self.assertFalse(result["screening_ability_proven"])
        self.assertIn("有输出不等于", result["banner"])

    def test_forward_stamp_follows_maturity(self):
        self.write(DAY, artefacts(mature=0, rows=2))
        self.assertEqual(brief.build(DAY, self.root)["sections"]["top"]["stamp"]["forward"], "RECORDING")
        self.write(DAY, artefacts(mature=3, rows=3))
        self.assertEqual(brief.build(DAY, self.root)["sections"]["top"]["stamp"]["forward"], "MATURE")
        self.write(DAY, artefacts(rows=0))
        self.assertEqual(brief.build(DAY, self.root)["sections"]["top"]["stamp"]["forward"], "NOT_STARTED")

    def test_reconstruction_rows_never_make_the_forward_stamp_mature(self):
        # 2244 matured reconstruction rows next to zero live rows: forward has not even started.
        self.write(DAY, artefacts(rows=0, rebuilt_rows=2244))
        stamp = brief.build(DAY, self.root)["sections"]["top"]["stamp"]
        self.assertEqual(stamp["forward"], "NOT_STARTED")
        self.assertIn("回建 2244 行另计", stamp["forward_note"])
        # Rows recorded with the reconstruction basis alone are not a live recording either.
        self.write(DAY, artefacts(rows=5, mature=5, basis=REBUILT))
        self.assertEqual(brief.build(DAY, self.root)["sections"]["top"]["stamp"]["forward"], "NOT_STARTED")

    def test_historical_stamp_is_partial_at_most_from_frozen_protocols(self):
        self.write(DAY, artefacts(protocols=True))
        self.assertEqual(brief.build(DAY, self.root)["sections"]["outcomes"]["stamp"]["historical"], "PARTIAL")
        self.write(DAY, artefacts(protocols=False))
        self.assertEqual(brief.build(DAY, self.root)["sections"]["outcomes"]["stamp"]["historical"], "NOT_DONE")

    def test_missing_artefact_is_named_not_blanked(self):
        self.write(DAY, artefacts(drop=("ranking_validation.json",)))
        result = brief.build(DAY, self.root)
        self.assertIn("ranking_validation.json", result["missing_inputs"])
        self.assertEqual(result["sections"]["outcomes"]["stamp"]["software"], "MISSING_INPUT")
        self.assertEqual(result["sections"]["top"]["stamp"]["forward"], "NOT_STARTED")

    def test_top_rows_join_diagnosis_change_and_continuity(self):
        self.write(DAY, artefacts())
        first = brief.build(DAY, self.root)["sections"]["top"]["strength"][0]
        self.assertEqual(first["diagnosis"][0]["stopped_at"], "retention")
        self.assertEqual(first["change_attribution"], "COHORT_STRENGTHENED")
        self.assertEqual(first["continuity_event"], "WAITING")
        self.assertEqual(first["blockers"], ["市场闸门"])

    def test_cross_day_uses_the_latest_earlier_board_or_says_none(self):
        self.write(DAY, artefacts())
        alone = brief.build(DAY, self.root)["sections"]["cross_day"]
        self.assertEqual(alone["comparison_basis"], "无前日对照")
        earlier = artefacts("2026-09-17")
        earlier["ranking_explained.json"]["strength_top"][0]["symbol"] = "sh_600099"
        self.write("2026-09-17", earlier)
        paired = brief.build(DAY, self.root)["sections"]["cross_day"]
        self.assertEqual(paired["previous_day"], "2026-09-17")
        self.assertIn("sh_600010", paired["entered_top"])
        self.assertIn("sh_600099", paired["left_top"])

    def test_outcomes_keep_process_and_result_apart(self):
        self.write(DAY, artefacts())
        outcomes = brief.build(DAY, self.root)["sections"]["outcomes"]
        self.assertTrue(all(c["reading"] == "未成熟，不记零" for c in outcomes["cohorts"]))
        self.assertEqual(outcomes["missed"]["status"], "WINDOW_IMMATURE")
        self.assertIn("过程侧", outcomes["process_note"])

    def test_rank_history_older_than_chain_is_flagged(self):
        self.write(DAY, artefacts())
        files = artefacts()
        files["opportunity_continuity.json"]["events"][0]["event"] = "DISCOVERED"
        self.write(DAY, files)
        x = brief.build(DAY, self.root)["sections"]["cross_day"]
        self.assertIn("时间基线不同", x["chain_note"])
        self.assertIn("注意：", brief.render(brief.build(DAY, self.root)))

    def test_render_contains_banner_and_all_three_labels(self):
        self.write(DAY, artefacts())
        text = brief.render(brief.build(DAY, self.root))
        self.assertIn(brief.BANNER, text)
        for word in ("软件已实现", "历史验证", "前向"):
            self.assertIn(word, text)
        self.assertIn("筛选能力已证明：**否**", text)


if __name__ == "__main__":
    unittest.main()
