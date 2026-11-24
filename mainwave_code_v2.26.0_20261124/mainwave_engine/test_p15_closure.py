from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from mainwave_engine import candidate_report as REPORT
from mainwave_engine import decision_bridge as BRIDGE
from mainwave_engine import p1_daily as P1
from mainwave_engine import pipeline as PIPE
from mainwave_engine import recent_review as REVIEW
from mainwave_engine import theme_proposals as PROPOSALS
from mainwave_engine import theme_horizon as HORIZON


class P15ClosureTests(unittest.TestCase):
    def test_attention_marks_internal_leading_instead_of_overwriting_with_external_low(self) -> None:
        timeline = pd.DataFrame([
            {"pulse_members": 0, "breadth": 0.20},
            {"pulse_members": 0, "breadth": 0.22},
            {"pulse_members": 1, "breadth": 0.25},
            {"pulse_members": 1, "breadth": 0.30},
            {"pulse_members": 2, "breadth": 0.35},
            {"pulse_members": 2, "breadth": 0.45},
        ])
        external = {
            "external_attention_level": "LOW", "external_attention_slope": "FLAT",
            "external_matches": "", "external_heat": 0.0, "external_limit_count": 0,
            "external_data_status": "FRESH_NO_MATCH",
        }
        with patch.object(P1, "_external_attention", return_value=external):
            result = P1._theme_attention(
                {"id": "demo", "label": "Demo"}, {"stage": "FERMENTING"},
                timeline, "2026-08-28")
        self.assertEqual(result["attention_relation"], "INTERNAL_LEADING")
        self.assertEqual(result["attention_level"], "MEDIUM")
        self.assertEqual(result["attention_slope"], "RISING")

    def test_l0_red_blocks_execution_but_preserves_discovery(self) -> None:
        focus = pd.DataFrame([{
            "symbol": "sh_600001", "name": "A", "theme_id": "demo",
            "theme_episode_id": "demo:C1:2026-08-01", "episode_id": "capital-1",
            "intersection_class": "A", "role": "information_leader",
        }])
        with tempfile.TemporaryDirectory() as directory, patch.object(
                BRIDGE, "_l0", return_value={
                    "state": "RED", "allow": False, "cap": 0.0, "panic": 2, "banner": "test"
                }):
            frame, summary = BRIDGE.build("2026-08-28", focus, Path(directory))
        self.assertEqual(len(frame), 1)
        self.assertTrue(bool(frame.iloc[0]["discovery_recorded"]))
        self.assertEqual(frame.iloc[0]["permission"], "BLOCKED_BY_L0")
        self.assertEqual(summary["orders_created"], 0)

    def test_candidate_diagnostics_respects_as_of(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(REPORT, "ROOT", Path(directory)):
            data = Path(directory) / "data"
            data.mkdir()
            frame = pd.DataFrame({
                "date": pd.date_range("2026-01-01", periods=90, freq="D").strftime("%Y-%m-%d"),
                "close": range(1, 91), "volume": [100] * 90, "pct_chg": [0.0] * 90,
            })
            frame.to_csv(data / "sh_600001_daily.csv", index=False)
            stats = REPORT._stock_stats("sh_600001", "2026-03-11")
        self.assertAlmostEqual(stats["r20"], (70 / 50 - 1) * 100)

    def test_pipeline_records_failed_step_and_no_success_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
                PIPE, "OUT_ROOT", Path(directory)), patch.object(
                PIPE.subprocess, "run", side_effect=lambda command, **kwargs:
                SimpleNamespace(returncode=7 if command[2] == "mainwave_engine.chain_pulse" else 0)):
            code = PIPE.run("2026-08-28")
            output = Path(directory) / "20260828"
            status = json.loads((output / "mainwave_run_status.json").read_text(encoding="utf-8"))
            self.assertFalse((output / "_SUCCESS").exists())
        self.assertEqual(code, 7)
        self.assertEqual(status["status"], "FAILED")
        self.assertEqual(status["failed_step"], "chain_pulse")

    def test_pipeline_missing_calendar_stops_before_scanning_or_trading(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(
                PIPE, "OUT_ROOT", Path(directory)), patch.object(
                PIPE.subprocess, "run", return_value=SimpleNamespace(returncode=1)) as launch:
            self.assertEqual(PIPE.run("2026-11-18"), 1)
            self.assertEqual(launch.call_count, 1)
            self.assertEqual(launch.call_args.args[0][2], "mainwave_engine.trading.preflight")
            output = Path(directory) / "20261118"
            status = json.loads((output / "mainwave_run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["failed_step"], "calendar_preflight")
            self.assertFalse((output / "_SUCCESS").exists())

    def test_recent_review_excludes_inactive_theme(self) -> None:
        episode = {"shadow_stage": "ABSORBING", "retention_state": "RETAINED"}
        metrics = {"r5": 2.0, "r20": 8.0, "ma20_bias": 1.0}
        self.assertEqual(REVIEW._follow_state(episode, metrics, "RETREAT"), "THEME_INACTIVE")

    def test_recent_review_marks_twenty_day_extension(self) -> None:
        episode = {"shadow_stage": "ABSORBING", "retention_state": "RETAINED"}
        metrics = {"r5": 1.0, "r20": 35.0, "ma20_bias": 8.0}
        self.assertEqual(
            REVIEW._follow_state(episode, metrics, "FERMENTING"),
            "EXTENDED_WAIT_PULLBACK",
        )

    def test_future_concept_mapping_is_not_used_for_historical_run(self) -> None:
        self.assertEqual(
            PROPOSALS._mapping_status("2026-08-29", "2026-08-28"),
            "FUTURE_UNAVAILABLE",
        )

    def test_theme_horizon_preheat_window(self) -> None:
        self.assertEqual(
            HORIZON.calendar_state("2026-08-28", "2026-09-10", "2026-10-10", 20),
            "PREHEAT",
        )

    def test_theme_horizon_repair_override_wins(self) -> None:
        self.assertEqual(
            HORIZON.decide("ACTIVE", "PROPOSED", "BUILDING", repair=True),
            ("REPAIR_ONLY", "REPAIR_ONLY"),
        )

    def test_theme_horizon_active_carrier_is_now_focus(self) -> None:
        self.assertEqual(
            HORIZON.decide("ACTIVE", "PROPOSED", "BUILDING"),
            ("ACTIVE_WINDOW_WITH_CARRIER", "NOW_FOCUS"),
        )

    def test_theme_horizon_expired_is_closed(self) -> None:
        self.assertEqual(
            HORIZON.decide("EXPIRED", "PROPOSED", "BUILDING"),
            ("EXPIRED", "CLOSED"),
        )

    def test_confirmed_climate_catalyst_raises_research_not_execution(self) -> None:
        self.assertEqual(
            HORIZON.research_priority("BUILD_WATCH", "CONFIRMED_STRONG", "ACTIVE"),
            "HIGH",
        )
        self.assertEqual(
            HORIZON.decide("ACTIVE", "WAIT_CARRIER", "NONE"),
            ("ACTIVE_WAIT_CARRIER", "BUILD_WATCH"),
        )

    def test_external_catalyst_requires_scheduled_refresh(self) -> None:
        theme = {"external_catalyst": {
            "state": "CONFIRMED_STRONG", "as_of": "2026-08-13",
            "review_on": "2026-09-10", "valid_until": "2027-02-28",
        }}
        self.assertEqual(
            HORIZON.external_catalyst_state(theme, "2026-09-10"),
            "STALE_REVIEW_REQUIRED",
        )

    def test_theme_proposal_applies_climate_research_context(self) -> None:
        config = {"effective_from": "2026-08-28", "themes": [{
            "id": "agri", "label": "Agriculture", "aliases": ["农药化肥"],
            "window_start": "2026-08-20", "window_end": "2026-10-31",
            "preheat_days": 10,
            "external_catalyst": {
                "state": "CONFIRMED_STRONG", "as_of": "2026-08-13",
                "review_on": "2026-09-10", "valid_until": "2027-02-28",
            },
        }]}
        context = PROPOSALS._calendar_context("农药化肥", "2026-08-28", "WAIT_CARRIER", config)
        self.assertEqual(context["research_priority"], "HIGH")
        self.assertEqual(context["external_catalyst_state"], "CONFIRMED_STRONG")

    def test_high_research_probe_can_still_be_rejected_as_extended(self) -> None:
        episode = {"shadow_stage": "PROBED", "retention_state": "UNCERTAIN"}
        metrics = {"r5": 16.0, "r20": 40.0, "ma20_bias": 25.0}
        self.assertEqual(REVIEW._follow_state(episode, metrics), "EXTENDED_PROBE")

    def test_theme_calendar_is_not_backfilled_before_effective_date(self) -> None:
        context = PROPOSALS._calendar_context(
            "农药化肥", "2026-08-27", "WAIT_CARRIER",
            {"effective_from": "2026-08-28", "themes": []},
        )
        self.assertEqual(context["calendar_state"], "NONE")
        self.assertEqual(context["research_priority"], "BASE")


if __name__ == "__main__":
    unittest.main()
