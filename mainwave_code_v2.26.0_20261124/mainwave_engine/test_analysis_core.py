import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from mainwave_engine import analysis_core as AC
from mainwave_engine import market_context as MC
from mainwave_engine import pipeline as PIPE


def fixture():
    market = {"as_of": "2026-09-04", "state": "GREEN", "structure": "BULL", "temperature": "NORMAL",
              "data_status": "FRESH", "issues": [], "reversion_research_relevant": False}
    data = {
        "meta": {"symbol": "sh_600001", "report_date": "2026-09-04", "price_date": "2026-09-04",
                 "mode": "DAILY_ANALYSIS", "in_daily_pool": False},
        "lifecycle": {"current": "PULLBACK", "current_label": "回踩", "cycle_id": "cycle-1", "data_status": "FRESH",
                      "production_trigger_at": "2026-08-01"},
        "price": {"close": 10.2}, "theme": {"belonging_theme": "test"},
        "turnover": {"turnover_relative20": 1.2},
        "breakout": {"breakout_id": "break-1", "breakout_status": "ADVANCING"},
        "setups": {"current": {"setup_kind": "FIRST_PULLBACK", "setup_first_known": True,
                               "setup_status": "CONFIRMED", "setup_id": "setup-1",
                               "setup_breakout_id": "break-1", "setup_breakout_accepted_at": "2026-08-20",
                               "setup_breakout_advanced_at": "2026-08-24", "setup_adjustment_from": "2026-08-27",
                               "setup_trigger_price": 10, "setup_invalid_price": 9,
                               "setup_frozen_atr_price": 1, "setup_confirmed_at": "2026-09-04"}},
        "chart": {"bars": [{"date": d} for d in ["2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"]]},
    }
    return data, market


class AnalysisCoreTests(unittest.TestCase):
    def acceptance_fixture(self):
        data, market = fixture()
        data["setups"]["current"].update(setup_status="HISTORY_FAILED")
        data["lifecycle"]["current"] = "IGNITION"
        data["breakout"].update(breakout_status="ACCEPTED", breakout_accepted_at="2026-09-04")
        data["route_evidence"] = {"confirmation_pass": True, "route_requirements_met": True,
                                  "eligible": True, "accepted_at": "2026-09-04",
                                  "reference_close": 10.2, "atr_price": 1, "risks": []}
        return data, market

    def test_fresh_acceptance_reviews_half_initial_with_active_price_band(self):
        data, market = self.acceptance_fixture()
        result = AC.evaluate(data, market)
        self.assertEqual(result["active_strategy"], "S4")
        self.assertEqual(result["advice"], "PRIORITY_REVIEW")
        self.assertTrue(result["review_window"]["active"])
        self.assertIsNotNone(result["review_window"]["upper"])
        self.assertEqual(result["buy_plan"]["initial_fraction"], .5)
        self.assertTrue(result["buy_plan"]["initial_review"])
        self.assertFalse(result["orders_enabled"])

    def test_direct_acceptance_never_overrides_parent_or_supply_or_market_risks(self):
        data, market = self.acceptance_fixture()
        data["route_evidence"]["risks"] = ["SUPPLY_PRESSURE"]
        self.assertEqual(AC.evaluate(data, market)["advice"], "RISK_WATCH")
        data["route_evidence"]["risks"] = []
        data["lifecycle"]["current"] = "FAILED"
        self.assertNotEqual(AC.evaluate(data, market)["active_strategy"], "S4")
        data["lifecycle"]["current"] = "IGNITION"
        market["state"] = "ORANGE"
        self.assertEqual(AC.evaluate(data, market)["advice"], "MARKET_WAIT")

    def test_route_preparation_gate_also_applies_to_existing_pullback(self):
        data, market = fixture()
        data["route_evidence"] = {"route_requirements_met": False}
        self.assertEqual(AC.evaluate(data, market)["advice"], "WATCH")

    def test_r2_theme_cannot_be_bypassed_by_appending_r1_label(self):
        data, market = fixture()
        data["route_evidence"] = {"route_ids": "R2", "route_requirements_met": True}
        self.assertEqual(AC.evaluate(data, market)["advice"], "WATCH")
        data["theme"]["lifecycle"] = "FERMENTING"
        self.assertEqual(AC.evaluate(data, market)["advice"], "PRIORITY_REVIEW")
        data["theme"]["lifecycle"] = ""
        data["route_evidence"]["route_ids"] = "R2|R1"
        self.assertEqual(AC.evaluate(data, market)["advice"], "WATCH")
        data["route_evidence"]["independent_price_evidence"] = True
        self.assertEqual(AC.evaluate(data, market)["advice"], "PRIORITY_REVIEW")

    def test_old_acceptance_does_not_reissue_daily_s4(self):
        data, market = self.acceptance_fixture()
        data["route_evidence"]["accepted_at"] = "2026-09-03"
        self.assertNotEqual(AC.evaluate(data, market)["active_strategy"], "S4")

    def test_confirmed_first_retest_not_blocked_by_prior_launch_or_pool_absence(self):
        data, market = fixture()
        result = AC.evaluate(data, market)
        self.assertEqual(result["advice"], "PRIORITY_REVIEW")
        self.assertEqual(result["active_strategy"], "S2")
        self.assertFalse(result["orders_enabled"])
        self.assertEqual(result["mode"], "ANALYSIS_ONLY")
        self.assertNotIn("expected_return", result)

    def test_missing_fundamentals_does_not_veto_price_hypothesis(self):
        data, market = fixture()
        result = AC.evaluate(data, market)
        self.assertEqual([r["code"] for r in result["dimensions"]], [f"D{i}" for i in range(1, 7)])
        self.assertEqual(result["dimensions"][0]["status"], "UNKNOWN")
        self.assertEqual(result["mechanism"]["code"], "M0")
        self.assertEqual(result["advice"], "PRIORITY_REVIEW")

    def test_unknown_first_identity_routes_to_continuation(self):
        data, market = fixture()
        data["setups"]["current"]["setup_first_known"] = False
        self.assertEqual(AC.evaluate(data, market)["active_strategy"], "S3")

    def test_overheat_is_overlay_not_a_new_accumulation_cycle(self):
        data, market = fixture()
        data["lifecycle"]["current"] = "OVERHEATED"
        result = AC.evaluate(data, market)
        self.assertEqual(result["state_vector"]["phase"], "ADVANCE")
        self.assertEqual(result["state_vector"]["cycle_id"], "cycle-1")
        self.assertEqual(result["advice"], "RISK_WATCH")

    def test_market_caution_preserves_stock_structure(self):
        data, market = fixture()
        market.update(state="ORANGE", structure="BEAR", reversion_research_relevant=True)
        result = AC.evaluate(data, market)
        self.assertEqual(result["advice"], "MARKET_WAIT")
        self.assertEqual(result["structural_advice"], "PRIORITY_REVIEW")
        self.assertEqual(result["strategies"][-1]["status"], "RESEARCH_RELEVANT")

    def test_theme_risk_is_not_lost_in_new_analysis(self):
        data, market = fixture()
        data["theme"]["risk"] = {"theme_risk_action": "BLOCK_NEW_ENTRY", "theme_risk_reason": "event risk"}
        result = AC.evaluate(data, market)
        self.assertEqual(result["advice"], "RISK_WATCH")
        self.assertTrue(any(r["source"] == "theme_risk" for r in result["state_vector"]["risk_overlays"]))
        self.assertEqual(result["structural_advice"], "PRIORITY_REVIEW")

    def test_price_below_frozen_stop_is_not_a_price_wait(self):
        data, market = fixture()
        data["price"]["close"] = 8.5
        self.assertEqual(AC.evaluate(data, market)["advice"], "RISK_WATCH")

    def test_risks_sort_after_market_waiting_opportunities(self):
        self.assertLess(AC.ADVICE_ORDER["MARKET_WAIT"], AC.ADVICE_ORDER["RISK_WATCH"])

    def test_stale_market_or_stock_waits(self):
        for location in ("stock", "market"):
            data, market = fixture()
            if location == "stock":
                data["meta"]["price_date"] = "2026-09-03"
            else:
                market["as_of"] = "2026-09-07"
            self.assertEqual(AC.evaluate(data, market)["advice"], "DATA_WAIT")

    def test_missing_risk_anchor_never_qualifies(self):
        for missing in (None, float("nan"), 0, -1):
            data, market = fixture()
            data["setups"]["current"]["setup_invalid_price"] = missing
            self.assertEqual(AC.evaluate(data, market)["advice"], "DATA_WAIT")

    def test_no_chase_or_negative_width_band(self):
        data, market = fixture()
        data["price"]["close"] = 11
        self.assertEqual(AC.evaluate(data, market)["advice"], "WAIT_PRICE")
        data["setups"]["current"]["setup_invalid_price"] = 7
        result = AC.evaluate(data, market)
        self.assertIsNone(result["review_window"]["lower"])
        self.assertEqual(result["review_window"]["trigger"], 10)
        self.assertEqual(result["advice"], "WAIT_PRICE")

    def test_confirmation_expiry_counts_sessions_and_rejects_future_dates(self):
        data, market = fixture()
        data["setups"]["current"].update(setup_status="HISTORY_CONFIRMED", setup_confirmed_at="2026-09-01")
        self.assertEqual(AC.evaluate(data, market)["advice"], "TREND_WATCH")
        data["setups"]["current"]["setup_confirmed_at"] = "2026-09-07"
        self.assertNotEqual(AC.evaluate(data, market)["advice"], "PRIORITY_REVIEW")

    def test_history_never_issues_current_advice_and_evaluation_is_pure(self):
        data, market = fixture()
        data["meta"]["mode"] = "HISTORICAL_RECONSTRUCTION"
        before = copy.deepcopy(data)
        result = AC.evaluate(data, market)
        self.assertEqual(result["advice"], "HISTORICAL_ONLY")
        self.assertEqual(data, before)
        self.assertEqual(result, AC.evaluate(data, market))

    def test_pipeline_contains_only_new_engine_and_closure_follows_analysis(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(PIPE, "OUT_ROOT", Path(temp)), patch.object(
                PIPE.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            self.assertEqual(PIPE.run("2026-09-04"), 0)
            commands = [c.args[0] for c in run.call_args_list]
            self.assertEqual(commands[0][2], "mainwave_engine.trading.preflight")
            self.assertEqual(commands[1][2], "mainwave_engine.virtual_portfolio")
            self.assertEqual(commands[2][2], "mainwave_engine.market_context")
            self.assertEqual(commands[3][2], "mainwave_engine.trading.risk_daily")
            # Closure follows analysis; the after-close review and its V2 report come last and only read what they froze.
            self.assertEqual(commands[-7][2], "mainwave_engine.analysis_daily")
            self.assertEqual(commands[-6][2], "mainwave_engine.trading.daily")
            self.assertEqual(commands[-5][2], "mainwave_engine.strong_trend")
            self.assertEqual(commands[-4][2], "mainwave_engine.scenario_scan")
            self.assertEqual(commands[-3][2], "mainwave_engine.eod_review")
            self.assertEqual(commands[-2][2], "mainwave_engine.limit_theme")
            self.assertEqual(commands[-1][2], "mainwave_engine.fupan")
            modules = [command[2] for command in commands]
            self.assertEqual(len(modules), len(set(modules)))
            for module in ("virtual_portfolio", "route_scan", "three_direction_report", "analysis_daily"):
                self.assertEqual(modules.count(f"mainwave_engine.{module}"), 1)
            self.assertNotIn("strategy_tracker", json.dumps(commands))
            self.assertTrue((Path(temp) / "20260904" / "_SUCCESS").exists())


class MarketContextTests(unittest.TestCase):
    def market(self):
        return pd.DataFrame([{"date": pd.Timestamp("2026-09-04"), "structure": "BULL",
                              "temperature": "NORMAL", "market_context": "BULL"}])

    def source(self, path):
        dates = pd.bdate_range(end="2026-09-04", periods=120).strftime("%Y-%m-%d")
        return pd.DataFrame({"date": dates, "close": 10, "avg_pct": 0.5, "up_count": 3000,
                             "down_count": 2000, "limit_up": 50, "limit_down": 10})

    def test_uses_own_data_sources_not_old_banner(self):
        with patch.object(MC.pd, "read_csv", side_effect=self.source) as read, patch.object(
                MC, "build_market_frame", return_value=self.market()):
            result = MC.build("2026-09-04")
        self.assertEqual(result["state"], "GREEN")
        self.assertNotIn("strategy_tracker", str(read.call_args_list))

    def test_stale_source_does_not_default_to_green(self):
        def source(path):
            frame = self.source(path)
            return frame.iloc[:-1] if "000688" in str(path) else frame
        with patch.object(MC.pd, "read_csv", side_effect=source), patch.object(
                MC, "build_market_frame", return_value=self.market()):
            result = MC.build("2026-09-04")
        self.assertEqual(result["state"], "UNKNOWN")
        self.assertTrue(result["issues"])

    def test_wrong_date_cache_not_reused(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(MC, "OUT_ROOT", Path(temp)), patch.object(
                MC, "build", return_value={"rebuilt": True}) as build:
            path = Path(temp) / "20260904" / "market_context.json"
            path.parent.mkdir()
            path.write_text(json.dumps({"version": MC.VERSION, "as_of": "2026-09-07"}), encoding="utf-8")
            self.assertEqual(MC.load("2026-09-04"), {"rebuilt": True})
            build.assert_called_once()


if __name__ == "__main__":
    unittest.main()
