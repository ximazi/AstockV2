from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from mainwave_engine import historical_cases as HC
from mainwave_engine import phase_events as PE
from mainwave_engine import price_cycle_setups as PCS
from mainwave_engine import stock_analyzer as SA
from mainwave_engine import stock_lifecycle as SL
from mainwave_engine.test_phase_events import PhaseEventTests


def case_timeline(control=False):
    dates = pd.bdate_range("2025-10-01", periods=120).strftime("%Y-%m-%d")
    frame = pd.DataFrame({"date": dates, "cycle_id": "", "cycle_started_at": "",
                          "cycle_stage": "WATCH", "setup_id": ""})
    frame.loc[30:, "cycle_id"] = "sh_600000:" + dates[30]
    frame.loc[30:, "cycle_started_at"] = dates[30]
    frame.loc[30, "cycle_stage"] = "IGNITION"
    frame.loc[31:60, "cycle_stage"] = "MARKUP"
    frame.loc[41:46, "cycle_stage"] = "PULLBACK"
    frame.loc[61:68, "cycle_stage"] = "DISTRIBUTION_RISK"
    frame.loc[69:100, "cycle_stage"] = "FAILED"
    frame.loc[101:, "cycle_stage"] = "REBUILDING"
    if control:
        frame.loc[34:, "cycle_stage"] = "FAILED"
    return frame


class HistoricalCaseTests(unittest.TestCase):
    def test_closed_cycle_requires_repair_and_context(self):
        frame = case_timeline()
        cases = HC.discover("sh_600000", frame, "2025-09-04")
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["category"], "COMPLETE")
        self.assertIn("REBUILDING", cases[0]["stages"])
        self.assertEqual(HC.discover("sh_600000", frame.iloc[:100], "2025-09-04"), [])
        self.assertEqual(HC.discover("sh_600000", frame, "2026-09-04"), [])

    def test_failure_control_is_not_full_cycle(self):
        row = HC.discover("sh_600000", case_timeline(True), "2025-09-04")[0]
        self.assertEqual(row["category"], "FAILED_LAUNCH")
        self.assertLess(row["markup_sessions"], 5)

    def test_trend_already_visible_is_not_observed_launch(self):
        frame = case_timeline()
        frame.loc[30, "cycle_stage"] = "MARKUP"
        self.assertEqual(HC.discover("sh_600000", frame, "2025-09-04"), [])

    def test_read_api_is_cached_and_rejects_unknown_or_stale_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {"cycle_version": SL.VERSION, "setup_version": PCS.VERSION,
                        "phase_version": PE.VERSION, "analyzer_version": SA.VERSION,
                        "cases": [{"id": "sh_600000_20251001"}]}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "sh_600000_20251001.json").write_text('{"ok": true}', encoding="utf-8")
            with patch.object(HC, "build", side_effect=AssertionError("GET must not scan")):
                self.assertEqual(HC.case_detail("sh_600000_20251001", root), {"ok": True})
                with self.assertRaises(ValueError):
                    HC.case_detail("../positions_v2", root)
                manifest["cycle_version"] = "old"
                (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaises(ValueError):
                    HC.case_detail("sh_600000_20251001", root)

    def test_empty_library(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(HC.case_list(Path(directory))["cases"], [])

    def test_adjusted_chart_ema_warmup_not_viewport_dependent(self):
        classified = PE.classify_frame(PhaseEventTests.price_frame())
        long = SA._adjusted_chart(classified, 180)
        short = SA._adjusted_chart(classified, 30)
        self.assertEqual(long[-30:], short)

    def test_history_without_archived_report_never_uses_current_value_or_episodes(self):
        raw = PhaseEventTests.price_frame()
        classified = PE.classify_frame(raw)
        day = str(raw["date"].iloc[-1].date())
        timeline = SL.replay("sh_600000", classified)
        cycle = json.loads(timeline.tail(1).to_json(orient="records"))[0]
        with patch.object(SA, "latest_report_date", side_effect=FileNotFoundError()), \
             patch.object(SA, "_data_frame", return_value=(raw, classified)) as prices, \
             patch.object(SA, "_mapping", return_value={}), \
             patch.object(SA, "_read_csv", return_value=pd.DataFrame()), \
             patch.object(SA, "_episodes", side_effect=AssertionError("future store")), \
             patch.object(SL, "analyze", return_value=(cycle, timeline)), \
             patch.object(PE, "latest_snapshot", return_value={}), \
             patch.object(SA.CV, "evaluate_candidates", side_effect=AssertionError("future calibration")):
            result = SA.analyze("600000", day, persist=True)
        prices.assert_called_once_with("sh_600000", day)
        self.assertEqual(result["meta"]["report_date"], day)
        self.assertEqual(result["decision"]["execution_permission"], "HISTORICAL_ONLY")
        self.assertEqual(result["potential"]["sample_n"], 0)
        self.assertEqual(result["evidence"]["episodes"], {})
        self.assertTrue(result["chart"]["stage_segments"])

    def test_flask_history_routes(self):
        from mainwave_visual_app import app
        client = app.test_client()
        with patch.object(HC, "case_list", return_value={"cases": []}):
            self.assertEqual(client.get("/api/history-cases").json, {"cases": []})
        with patch.object(HC, "case_detail", side_effect=ValueError("missing")):
            self.assertEqual(client.get("/api/history-cases/unknown").status_code, 400)


if __name__ == "__main__":
    unittest.main()
