from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from mainwave_engine import theme_risk as TR
from mainwave_engine import theme_semantics as TS
from mainwave_engine import three_direction_report as REPORT


class ThemeSemanticsTests(unittest.TestCase):
    def test_unanimous_specific_concept_relabels_broad_theme(self) -> None:
        mapping = {symbol: {"concepts": ["broad"]} for symbol in ("a", "b", "c")}
        auxiliary = {symbol: {"specific"} for symbol in ("a", "b", "c")}
        result = TS.resolve_event_theme("broad", ["a", "b", "c"], mapping, auxiliary)
        self.assertEqual(result["resolved_theme"], "specific")
        self.assertEqual(result["semantic_status"], "LEADER_UNANIMOUS_RELABEL")

    def test_partial_overlap_does_not_relabel(self) -> None:
        mapping = {symbol: {"concepts": ["copper"]} for symbol in ("a", "b", "c", "d")}
        auxiliary = {"a": {"gold"}, "b": {"gold"}, "c": {"gold"}, "d": set()}
        result = TS.resolve_event_theme("copper", ["a", "b", "c", "d"], mapping, auxiliary)
        self.assertEqual(result["resolved_theme"], "copper")
        self.assertEqual(result["semantic_status"], "RAW_THEME")

    def test_generic_concept_is_deprioritized_but_retained(self) -> None:
        symbol = "sz_000001"
        concepts = {
            "融资融券": {"stocks": [], "heat_score": 90, "limit_count": 8},
            "液冷服务器": {"stocks": [], "heat_score": 60, "limit_count": 3},
        }
        mapping = {
            symbol: {
                "concepts": ["融资融券", "液冷服务器"],
                "_dynamic_concepts": ["液冷服务器"],
            },
        }
        with patch.object(REPORT, "_hotspot_context", return_value=(concepts, "FRESH")), \
                patch.object(TS, "load_mapping_as_of", return_value=(
                    mapping, {"age_days": 0, "status": "FRESH"}
                )), \
                patch.object(TS, "load_aux_relationships", return_value=(
                    {}, {}, {"status": "EMPTY"}
                )):
            rows, _ = REPORT._symbol_theme_matches("2026-09-02")
        self.assertEqual(rows[symbol][0]["theme"], "液冷服务器")
        self.assertEqual(rows[symbol][0]["source"], "DYNAMIC_HOT_MAPPING")
        self.assertEqual(REPORT._related_themes(rows[symbol]), "液冷服务器|融资融券")

    def test_daily_dynamic_mapping_is_point_in_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "concept_mapping.json"
            history = root / "history"
            dynamic = root / "dynamic"
            history.mkdir()
            dynamic.mkdir()
            base.write_text(json.dumps({
                "generated": "2026-08-01 18:00:00",
                "mapping": {"sz_000001": {"name": "A", "concepts": ["基础题材"]}},
            }, ensure_ascii=False), encoding="utf-8")
            (dynamic / "concept_dynamic_20260831.json").write_text(json.dumps({
                "generated": "2026-08-31 18:00:00",
                "captured_concepts": ["液冷服务器"],
                "mapping": {"sz_000001": {"name": "A", "concepts": ["液冷服务器"]}},
            }, ensure_ascii=False), encoding="utf-8")
            with patch.object(TS, "MAPPING", base), \
                    patch.object(TS, "MAPPING_HISTORY", history), \
                    patch.object(TS, "DYNAMIC_MAPPING_HISTORY", dynamic):
                before, before_meta = TS.load_mapping_as_of("2026-08-30")
                after, after_meta = TS.load_mapping_as_of("2026-08-31")
        self.assertEqual(before["sz_000001"]["concepts"], ["基础题材"])
        self.assertEqual(after["sz_000001"]["concepts"], ["基础题材", "液冷服务器"])
        self.assertEqual(after["sz_000001"]["_dynamic_concepts"], ["液冷服务器"])
        self.assertEqual(before_meta["dynamic_overlay_files"], 0)
        self.assertEqual(after_meta["dynamic_overlay_files"], 1)


class ThemeRiskTests(unittest.TestCase):
    def test_precious_metal_shock_only_applies_from_next_session(self) -> None:
        self.assertEqual(TR.evaluate("黄金概念", "2026-08-28")["theme_risk_action"], "ALLOW")
        self.assertEqual(
            TR.evaluate("黄金概念", "2026-08-31")["theme_risk_action"],
            "BLOCK_NEW_ENTRY",
        )

    def test_industrial_metal_is_confirmation_not_blanket_block(self) -> None:
        self.assertEqual(
            TR.evaluate("铜", "2026-08-31")["theme_risk_action"],
            "REQUIRE_EXTERNAL_CONFIRMATION",
        )
        self.assertEqual(
            TR.evaluate("电子布/覆铜板", "2026-08-31")["theme_risk_action"],
            "ALLOW",
        )


class ThreeDirectionTests(unittest.TestCase):
    def test_friday_report_targets_monday_decision(self) -> None:
        self.assertEqual(REPORT.next_decision_date("2026-08-28"), "2026-08-31")

    def test_reverse_trace_distinguishes_current_from_historical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            reverse = output / "weekly_reverse_trace"
            reverse.mkdir()
            pd.DataFrame([
                {
                    "raw_theme": "服饰", "theme": "黄金概念", "t0": "2026-08-26",
                    "symbol": "sz_000001", "name": "A", "role": "IGNITION_CARRIER",
                    "first_probe": "2026-08-20", "probe_lead_sessions": 4,
                    "first_qualified": "2026-08-25", "qualified_lead_sessions": 1,
                    "pre_qualification": "QUALIFIED_LURK",
                    "current_qualification": "QUALIFIED_LURK", "bottleneck": "QUALIFIED_ON_EVE",
                },
                {
                    "raw_theme": "服饰", "theme": "黄金概念", "t0": "2026-08-26",
                    "symbol": "sz_000002", "name": "B", "role": "EVE_QUALIFIED_LURK",
                    "first_probe": "2026-08-18", "probe_lead_sessions": 6,
                    "first_qualified": "2026-08-22", "qualified_lead_sessions": 2,
                    "pre_qualification": "QUALIFIED_LURK",
                    "current_qualification": "NO_ACTIVE_EPISODE", "bottleneck": "QUALIFIED_ON_EVE",
                },
            ]).to_csv(reverse / "pre_ignition_carriers.csv", index=False, encoding="utf-8-sig")
            result = REPORT.build_direction2("2026-08-28", "2026-08-31", output)
        actions = dict(zip(result["symbol"], result["action"]))
        self.assertEqual(actions["sz_000001"], "BLOCKED_BY_THEME_RISK")
        self.assertEqual(actions["sz_000002"], "HISTORICAL_EVIDENCE_ONLY")

    def test_dynamic_theme_memory_survives_week_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "20260828"
            output.mkdir()
            pd.DataFrame([{
                "symbol": "sz_000001", "carrier_qualification": "QUALIFIED_LURK",
            }]).to_csv(output / "carrier_qualification.csv", index=False, encoding="utf-8-sig")
            current = [{
                "as_of": "2026-08-28", "decision_date": "2026-08-31",
                "raw_theme": "agri", "theme": "agri", "source": "WEEKLY_DYNAMIC_CHAIN",
                "semantic_status": "RAW_THEME", "lifecycle": "DIFFUSING",
                "t0": "2026-08-28", "event_type": "NEW_IGNITION_30D",
                "follow_through": "PENDING", "information_leader": "sz_000001",
                "limit_count_t0": 3, "post_week_max": 0,
                "current_qualified_carriers": 1,
                "current_carrier_symbols": "sz_000001", "member_symbols": "sz_000001",
                "ignition_threshold": 3, "action": "PROPAGATION_CONFIRMED",
                **TR.evaluate("agri", "2026-08-31"),
            }]
            with patch.object(REPORT, "DYNAMIC_STORE", root / "store.json"), \
                    patch.object(REPORT, "DYNAMIC_HISTORY", root / "history.csv"), \
                    patch.object(REPORT, "_hotspot_context",
                                 return_value=({"agri": {"limit_count": 3, "heat_score": 80}}, "FRESH")):
                REPORT._dynamic_memory("2026-08-28", "2026-08-31", current, output, "GREEN")
                carried = REPORT._dynamic_memory("2026-08-31", "2026-09-01", [], output, "GREEN")
        self.assertEqual(len(carried), 1)
        self.assertEqual(carried[0]["source"], "DYNAMIC_CHAIN_MEMORY")
        self.assertEqual(carried[0]["lifecycle"], "DIFFUSING")
        self.assertEqual(carried[0]["action"], "PROPAGATION_CONFIRMED")

    def test_expansion_is_monitored_but_never_becomes_a_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            reverse = output / "weekly_reverse_trace"
            reverse.mkdir()
            pd.DataFrame([{
                "date": "2026-08-31", "theme": "液冷服务器",
                "resolved_theme": "液冷服务器", "event_type": "EXPANSION",
                "follow_through": "HELD", "limit_count": 10,
                "post_week_max": 3, "ignition_threshold": 7,
                "limit_stocks": "sz_leader", "semantic_status": "RAW_THEME",
            }]).to_csv(
                reverse / "weekly_theme_events.csv", index=False, encoding="utf-8-sig"
            )
            pd.DataFrame([{
                "symbol": "sz_lurk", "carrier_qualification": "QUALIFIED_LURK",
            }]).to_csv(
                output / "carrier_qualification.csv", index=False, encoding="utf-8-sig"
            )
            mapping = {
                "sz_lurk": {"concepts": ["液冷服务器"]},
                "sz_leader": {"concepts": ["液冷服务器"]},
            }
            with patch.object(TS, "load_mapping_as_of", return_value=(mapping, {})), \
                    patch.object(REPORT, "_dynamic_memory",
                                 side_effect=lambda _a, _d, rows, _o, _m: rows):
                result = REPORT.build_direction3(
                    "2026-09-01", "2026-09-02", output, "GREEN"
                )
            with patch.object(REPORT, "_display_mapping", return_value={}), \
                    patch.object(REPORT, "_stage_context", return_value={}):
                weekly = REPORT.build_weekly_candidates(
                    "2026-09-01", pd.DataFrame(), result
                )
        self.assertEqual(result.iloc[0]["source"], "WEEKLY_EXPANSION_CHAIN")
        self.assertEqual(result.iloc[0]["current_qualified_carriers"], 1)
        self.assertEqual(result.iloc[0]["action"], "EXPANSION_MONITOR_NO_CHASE")
        self.assertTrue(weekly.empty)

    def test_weekly_table_separates_discovery_path_from_stock_lifecycle(self) -> None:
        direction1 = pd.DataFrame([{
            "symbol": "sz_lurk", "name": "Lurk", "theme": "",
            "theme_match_source": "", "theme_lifecycle": "",
            "shadow_stage": "READY", "trigger_distance_pct": 1.2,
            "stop_distance_pct": 4.0, "action": "FORWARD_WATCH",
        }])
        direction3 = pd.DataFrame([{
            "source": "WEEKLY_DYNAMIC_CHAIN", "information_leader": "sz_ignition",
            "theme": "agri", "raw_theme": "agri", "lifecycle": "DIFFUSING",
            "current_carrier_symbols": "sz_ignition|sz_diffusion",
            "structural_action": "PROPAGATION_CONFIRMED",
            "action": "PROPAGATION_CONFIRMED",
        }])
        metrics = {
            "sz_lurk": {"close": 10, "ema20_bias": 1, "r5": 2, "r20": 3},
            "sz_ignition": {"close": 11, "ema20_bias": 5, "r5": 12, "r20": 15},
            "sz_diffusion": {"close": 9, "ema20_bias": 2, "r5": 4, "r20": 6},
        }
        mapping = {
            "sz_lurk": {"name": "Lurk", "industry": "materials"},
            "sz_ignition": {"name": "Ignition", "industry": "agri"},
            "sz_diffusion": {"name": "Diffusion", "industry": "agri"},
        }
        with patch.object(REPORT, "_display_mapping", return_value=mapping), \
                patch.object(REPORT, "_stage_context", return_value={}), \
                patch.object(REPORT, "_stock_metrics",
                             side_effect=lambda symbol, *_: metrics[symbol]):
            result = REPORT.build_weekly_candidates(
                "2026-08-28", direction1, direction3
            )
        by_symbol = result.set_index("symbol")
        self.assertEqual(by_symbol.loc["sz_lurk", "discovery_path"], "LURK")
        self.assertEqual(by_symbol.loc["sz_lurk", "stock_lifecycle"], "READY")
        self.assertEqual(by_symbol.loc["sz_lurk", "belonging_theme"], "materials")
        self.assertEqual(by_symbol.loc["sz_ignition", "discovery_path"], "IGNITION")
        self.assertEqual(by_symbol.loc["sz_ignition", "stock_lifecycle"], "ACCELERATION")
        self.assertEqual(by_symbol.loc["sz_diffusion", "discovery_path"], "PROPAGATION")
        self.assertEqual(by_symbol.loc["sz_diffusion", "theme_lifecycle"], "DIFFUSING")

    def test_diffusing_theme_without_carrier_is_not_a_diffusion_stock(self) -> None:
        direction3 = pd.DataFrame([{
            "source": "WEEKLY_DYNAMIC_CHAIN", "information_leader": "sz_leader",
            "theme": "agri", "raw_theme": "agri", "lifecycle": "DIFFUSING",
            "current_carrier_symbols": "", "structural_action": "WAIT_CARRIER",
            "action": "MARKET_RISK_WATCH_ONLY",
        }])
        mapping = {"sz_leader": {"name": "Leader", "industry": "agri"}}
        with patch.object(REPORT, "_display_mapping", return_value=mapping), \
                patch.object(REPORT, "_stage_context", return_value={}), \
                patch.object(REPORT, "_stock_metrics", return_value={
                    "close": 10, "ema20_bias": 1, "r5": 2, "r20": 3,
                }):
            result = REPORT.build_weekly_candidates(
                "2026-08-28", pd.DataFrame(), direction3
            )
        self.assertEqual(result["discovery_path"].tolist(), ["IGNITION"])


if __name__ == "__main__":
    unittest.main()
