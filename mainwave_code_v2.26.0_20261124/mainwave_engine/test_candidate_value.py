from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mainwave_engine import candidate_value as CV
from mainwave_engine import phase_events as PE
from mainwave_engine.stock_analyzer import normalize_symbol


class CandidateValueTest(unittest.TestCase):
    def _calibration(self, path: Path) -> None:
        stats = {
            "n": 1000, "mfe20_p25": 0.02, "mfe20_median": 0.08,
            "mfe20_p75": 0.18, "mae20_p25": -0.07,
            "close_ret20_median": 0.01, "touch15_20": 0.30,
            "mfe40_median": 0.12,
        }
        path.write_text(json.dumps({
            "event_detector": PE.VERSION, "minimum_context_n": 200,
            "contexts": {"BASELINE": stats, "TECH_MID_BUILDING": stats},
        }), encoding="utf-8")

    def test_risk_veto_precedes_large_historical_potential(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            calibration = Path(temp) / "calibration.json"
            self._calibration(calibration)
            rows = pd.DataFrame([
                {"symbol": "sh_999998", "stock_lifecycle": "READY",
                 "discovery_path": "LURK", "theme_source": "POINT_IN_TIME_MAPPING",
                 "trigger_distance_pct": 1, "stop_distance_pct": 5, "r20": 8,
                 "action": "FORWARD_WATCH", "broad_phase": "TRANSITION"},
                {"symbol": "sh_999999", "stock_lifecycle": "READY",
                 "discovery_path": "LURK", "theme_source": "POINT_IN_TIME_MAPPING",
                 "trigger_distance_pct": 1, "stop_distance_pct": 5, "r20": 8,
                 "action": "FORWARD_WATCH", "broad_phase": "DISTRIBUTION",
                 "phase_observation_action": "RISK_REVIEW"},
            ])
            result = CV.evaluate_candidates(
                "2026-08-28", rows, "GREEN", calibration_path=calibration
            )
        self.assertEqual(result.iloc[0]["value_tier"], "A_READY")
        self.assertEqual(result.iloc[-1]["value_tier"], "D_AVOID")

    def test_market_gate_does_not_erase_structural_grade(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            calibration = Path(temp) / "calibration.json"
            self._calibration(calibration)
            row = pd.DataFrame([{
                "symbol": "sh_999998", "stock_lifecycle": "READY",
                "discovery_path": "LURK", "theme_source": "POINT_IN_TIME_MAPPING",
                "trigger_distance_pct": 1, "stop_distance_pct": 5, "r20": 8,
                "action": "MARKET_RISK_WATCH_ONLY", "broad_phase": "TRANSITION",
            }])
            result = CV.evaluate_candidates(
                "2026-08-28", row, "ORANGE", calibration_path=calibration
            ).iloc[0]
        self.assertEqual(result["value_tier"], "A_READY")
        self.assertEqual(result["execution_permission"], "MARKET_WATCH_ONLY")

    def test_symbol_normalization(self) -> None:
        self.assertEqual(normalize_symbol("600325"), "sh_600325")
        self.assertEqual(normalize_symbol("300305.SZ"), "sz_300305")
        self.assertEqual(normalize_symbol("sh_688820"), "sh_688820")

    def test_production_started_stock_cannot_reenter_as_early_setup(self):
        row = {"stock_lifecycle": "READY", "trigger_distance_pct": 1,
               "cycle_stage": "PULLBACK", "production_trigger_at": "2026-09-01"}
        self.assertEqual(CV._buy_state(row, "CLEAR")[0], "WATCH_LATE")
        row["cycle_stage"] = "FAILED"
        self.assertEqual(CV._risk_state(row)[0], "HARD_RISK")

    def test_reconstructed_cycle_is_not_a_new_trading_gate(self):
        for missing in ("", None, float("nan")):
            row = {"stock_lifecycle": "READY", "trigger_distance_pct": 1,
                   "cycle_stage": "PULLBACK", "production_trigger_at": missing}
            self.assertEqual(CV._buy_state(row, "CLEAR")[0], "BUY_ZONE")

    def test_stale_cycle_data_blocks_current_buy_assessment(self):
        self.assertEqual(CV._risk_state({"cycle_data_status": "STALE"})[0], "HARD_RISK")


if __name__ == "__main__":
    unittest.main()
