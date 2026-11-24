import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from mainwave_engine import analysis_core as AC, analysis_daily as AD
from mainwave_engine import breakout_events as BE, evidence_evolution as EE
from mainwave_engine import price_cycle_setups as PCS, three_direction_report as TD
from mainwave_engine.test_analysis_core import fixture
from mainwave_engine.test_price_cycle_setups import fixture as price_fixture


class BreakoutEventTests(unittest.TestCase):
    def test_attempt_accept_advance_then_linked_retest_without_parent_mutation(self):
        frame, parent = price_fixture()
        result = PCS.annotate(frame, parent)
        events = result[result.breakout_event.ne("")]
        self.assertEqual(events.breakout_event.tolist(), ["ATTEMPT", "ACCEPTED", "ADVANCING"])
        setup = result[result.setup_status.eq("FORMING")].iloc[0]
        self.assertLess(setup.setup_breakout_accepted_at, setup.setup_breakout_advanced_at)
        self.assertLess(setup.setup_breakout_advanced_at, setup.setup_adjustment_from)
        self.assertEqual(setup.setup_breakout_id, events.iloc[0].breakout_id)
        self.assertTrue(setup.setup_first_known)
        pd.testing.assert_frame_equal(result[parent.columns], parent)

    def test_acceptance_does_not_look_ahead_and_boundary_is_frozen(self):
        frame, parent = price_fixture()
        full = BE.annotate(frame, parent)
        for end in (20, 42, 43, 45, 65):
            pd.testing.assert_frame_equal(BE.annotate(frame.iloc[:end], parent.iloc[:end]),
                                          full.iloc[:end], check_dtype=False)
        self.assertEqual(full.iloc[41].breakout_status, "ATTEMPT")
        self.assertEqual(full.iloc[42].breakout_status, "ACCEPTED")
        self.assertEqual(full[full.breakout_id.ne("")].breakout_boundary_index.nunique(), 1)

    def test_failed_attempt_is_rejected_not_accepted(self):
        frame, parent = price_fixture()
        frame.loc[42, "return_index"] = 98
        result = BE.annotate(frame, parent)
        self.assertEqual(result.iloc[42].breakout_event, "REJECTED")
        self.assertEqual(result.iloc[42].breakout_accepted_at, "")

    def test_attempt_expires_in_active_sessions(self):
        frame, parent = price_fixture()
        frame.loc[42:48, "return_index"] = 100.3
        result = BE.annotate(frame, parent)
        self.assertEqual(result.iloc[47].breakout_event, "EXPIRED")

    def test_missing_reference_cannot_establish_first_identity(self):
        frame, parent = price_fixture()
        frame.loc[:50, "atr20"] = np.nan
        result = PCS.annotate(frame, parent)
        # A later breakout during an existing adjustment cannot rewrite its origin.
        formed = result[result.setup_status.eq("FORMING")].iloc[0]
        self.assertFalse(formed.setup_first_known)
        self.assertEqual(formed.setup_kind, "PULLBACK_UNKNOWN")


class CoreBoundaryTests(unittest.TestCase):
    def test_market_overlay_does_not_promote_repair_or_hide_missing_evidence(self):
        data, market = fixture()
        market["state"] = "ORANGE"
        data["lifecycle"]["current"] = "REBUILDING"
        self.assertEqual(AC.evaluate(data, market)["advice"], "WATCH")
        data["lifecycle"]["current"] = "PULLBACK"
        data.pop("breakout")
        self.assertEqual(AC.evaluate(data, market)["advice"], "DATA_WAIT")

    def test_missing_mismatched_or_future_chain_cannot_qualify(self):
        for change in ("missing", "mismatch", "future"):
            data, market = fixture()
            if change == "missing":
                data.pop("breakout")
            elif change == "mismatch":
                data["breakout"]["breakout_id"] = "another-event"
            else:
                data["setups"]["current"]["setup_breakout_advanced_at"] = "2026-09-05"
            result = AC.evaluate(data, market)
            self.assertEqual(result["advice"], "DATA_WAIT")
            self.assertNotEqual(result["active_strategy"], "S2")
            self.assertIsNone(result["review_window"]["lower"])

    def test_expired_or_failed_cycle_does_not_show_old_entry_window(self):
        for stage in ("PULLBACK", "REBUILDING"):
            data, market = fixture()
            data["lifecycle"]["current"] = stage
            data["setups"]["current"].update(setup_status="HISTORY_CONFIRMED", setup_confirmed_at="2026-09-01")
            result = AC.evaluate(data, market)
            self.assertEqual(result["active_strategy"], "NONE")
            self.assertIsNone(result["review_window"]["lower"])
            self.assertIsNone(result["review_window"]["invalid"])


class EvolutionTests(unittest.TestCase):
    def core(self, date="2026-09-04"):
        data, market = fixture()
        data["meta"].update(report_date=date, price_date=date)
        market["as_of"] = date
        core = AC.evaluate(data, market)
        core["evolution"] = EE.compare(core)
        return core

    def test_origin_frozen_and_current_rerun_idempotent(self):
        first, latest = self.core(), self.core("2026-09-07")
        before = copy.deepcopy(first)
        latest["evolution"] = EE.compare(latest, first)
        self.assertEqual(first, before)
        self.assertEqual(latest["evolution"]["origin"]["as_of"], "2026-09-04")
        self.assertEqual(latest["evolution"], EE.compare(latest, first))
        self.assertEqual(EE.compare(latest, latest)["previous_as_of"], "")
        self.assertEqual(latest["evolution"]["confirmation_event_key"], "")

    def test_unknown_data_is_not_a_failed_hypothesis(self):
        first, latest = self.core(), self.core("2026-09-07")
        latest["checks"][0]["status"] = "UNKNOWN"
        result = EE.compare(latest, first)
        self.assertEqual(result["changes"][0]["change"], "UNKNOWN")

    def test_new_cycle_resets_but_preparation_to_launch_retains_origin(self):
        first, latest = self.core(), self.core("2026-09-07")
        latest["state_vector"]["cycle_id"] = "cycle-2"
        self.assertTrue(EE.compare(latest, first)["baseline_reset"])
        first["evolution"]["latest"]["cycle_id"] = ""
        result = EE.compare(latest, first)
        self.assertFalse(result["baseline_reset"])
        self.assertEqual(result["origin"]["as_of"], "2026-09-04")

    def test_old_version_and_future_snapshot_are_not_comparable(self):
        first, latest = self.core(), self.core("2026-09-07")
        first["version"] = "old"
        self.assertFalse(EE.compare(latest, first)["previous"])
        self.assertFalse(EE.compare(self.core(), latest)["previous"])

    def test_reconstruction_cannot_masquerade_as_live_previous_evidence(self):
        first, latest = self.core(), self.core("2026-09-07")
        first["observation_mode"] = "HISTORICAL_RECONSTRUCTION"
        self.assertIsNone(EE.compare(latest, first)["previous"])

    def test_loader_uses_only_strictly_prior_matching_version(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for date in ("2026-09-04", "2026-09-07", "2026-09-08"):
                core = self.core(date)
                folder = root / date.replace("-", "")
                folder.mkdir()
                (folder / "analysis_evidence.json").write_text(json.dumps({"records": {core["symbol"]: core}}), encoding="utf-8")
            previous = EE.load_previous(root, "sh_600001", "2026-09-07", AC.VERSION)
            self.assertEqual(previous["as_of"], "2026-09-04")


class CandidateCoverageTests(unittest.TestCase):
    def test_forward_pool_is_retained_before_display_and_empty_is_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            pd.DataFrame([{"symbol": f"sh_{i:06d}", "carrier_qualification": "QUALIFIED_LURK"}
                          for i in range(22)]).to_csv(output / "carrier_qualification.csv", index=False)
            result = TD.build_direction1("2026-09-07", "2026-09-08", output, {})
            self.assertEqual(len(result), 22)
            pd.DataFrame([{"symbol": "sh_600000", "carrier_qualification": "BUILDING"}]).to_csv(
                output / "carrier_qualification.csv", index=False)
            self.assertTrue(TD.build_direction1("2026-09-07", "2026-09-08", output, {}).empty)

    def test_departed_candidates_and_failed_analyses_are_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            current, previous = root / "20260907", root / "20260904"
            current.mkdir()
            previous.mkdir()
            pd.DataFrame(columns=["symbol"]).to_csv(current / "mainwave_discovery_candidates.csv", index=False)
            core = AC.evaluate(*fixture())
            core["evolution"] = EE.compare(core)
            (previous / "analysis_evidence.json").write_text(json.dumps({"version": AC.VERSION,
                "records": {core["symbol"]: core}, "pending_symbols": ["sh_600002"]}), encoding="utf-8")
            with patch.object(AD.SA, "OUT_ROOT", root), patch.object(AD.MC, "write", return_value=fixture()[1]), patch.object(
                    AD.SA, "analyze", side_effect=ValueError("temporarily unavailable")) as analyze:
                result = AD.run("2026-09-07")
            self.assertEqual(analyze.call_count, 2)
            self.assertEqual(result["summary"]["tracked_extra"], 2)
            self.assertEqual(result["records"], [])
            stored = json.loads((current / "analysis_evidence.json").read_text(encoding="utf-8"))
            self.assertEqual(set(stored["pending_symbols"]), {"sh_600001", "sh_600002"})

    def test_ignition_slots_cannot_eliminate_forward_candidates(self):
        forward = pd.DataFrame([{"symbol": f"sz_{i:06d}", "shadow_stage": "READY"} for i in range(20)])
        ignition = pd.DataFrame([{"source": "WEEKLY_DYNAMIC_CHAIN", "information_leader": f"sh_{i:06d}"}
                                 for i in range(20)])
        with patch.object(TD, "_display_mapping", return_value={}), patch.object(TD, "_stage_context", return_value={}), patch.object(
                TD, "_phase_context", return_value={}), patch.object(TD, "_stock_metrics", return_value={
                    "close": 10, "ema20_bias": 1, "r5": 1, "r20": 1}):
            result = TD.build_weekly_candidates("2026-09-07", forward, ignition)
        self.assertEqual(len(result), 40)
        self.assertEqual(result.discovery_path.eq("LURK").sum(), 20)

    def test_analyze_all_before_display_and_audit_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "20260904"
            output.mkdir()
            pd.DataFrame({"symbol": [f"sh_{i:06d}" for i in range(18)]}).to_csv(output / "mainwave_discovery_candidates.csv", index=False)

            def analyze(symbol, *args, **kwargs):
                if symbol == "sh_000017":
                    raise ValueError("missing bars")
                data, market = fixture()
                data["meta"]["symbol"] = symbol
                data.update(identity={"name": symbol})
                data["price"]["ema20_bias_pct"] = 1
                data["lifecycle"]["discovery_path_label"] = "埋伏"
                core = AC.evaluate(data, market)
                core["evolution"] = EE.compare(core)
                data["analysis_core"] = core
                return data

            with patch.object(AD.SA, "OUT_ROOT", root), patch.object(AD.MC, "write", return_value=fixture()[1]), patch.object(
                    AD.SA, "analyze", side_effect=analyze) as run:
                result = AD.run("2026-09-04")
            self.assertEqual(run.call_count, 18)
            self.assertEqual(len(result["records"]), 15)
            self.assertEqual(result["summary"]["failed"], 1)
            self.assertEqual(len(pd.read_csv(output / "analysis_board.csv")), 17)
            self.assertEqual(len(pd.read_csv(output / "analysis_candidate_audit.csv")), 18)
            self.assertEqual(len(json.loads((output / "analysis_evidence.json").read_text(encoding="utf-8"))["records"]), 17)

    def test_tracking_retains_core_evidence_without_full_chart_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            current, previous = root / "20260907", root / "20260904"
            current.mkdir()
            previous.mkdir()
            pd.DataFrame({"symbol": ["sh_600002"]}).to_csv(
                current / "mainwave_discovery_candidates.csv", index=False)
            prior = AC.evaluate(*fixture())
            prior["evolution"] = EE.compare(prior)
            (previous / "analysis_evidence.json").write_text(
                json.dumps({"records": {prior["symbol"]: prior}}), encoding="utf-8")

            def analyze(symbol, *args, **kwargs):
                data, market = fixture()
                data["meta"].update(symbol=symbol, report_date="2026-09-07", price_date="2026-09-07")
                data["identity"] = {"name": symbol}
                data["price"]["ema20_bias_pct"] = 1
                data["lifecycle"]["discovery_path_label"] = "test"
                core = AC.evaluate(data, market)
                core["evolution"] = EE.compare(core)
                data["analysis_core"] = core
                if kwargs["persist"]:
                    (current / f"stock_analysis_{symbol}.json").write_text(json.dumps(data), encoding="utf-8")
                return data

            with patch.object(AD.SA, "OUT_ROOT", root), patch.object(AD.MC, "write", return_value=fixture()[1]), patch.object(
                    AD.SA, "analyze", side_effect=analyze):
                result = AD.run("2026-09-07")
            self.assertEqual(result["summary"]["analyzed"], 2)
            self.assertFalse((current / "stock_analysis_sh_600002.json").exists())
            self.assertFalse((current / "stock_analysis_sh_600001.json").exists())
            stored = json.loads((current / "analysis_evidence.json").read_text(encoding="utf-8"))
            self.assertEqual(set(stored["records"]), {"sh_600001", "sh_600002"})
            self.assertEqual(EE.load_previous(root, "sh_600001", "2026-09-08", AC.VERSION)["as_of"], "2026-09-07")


if __name__ == "__main__":
    unittest.main()
