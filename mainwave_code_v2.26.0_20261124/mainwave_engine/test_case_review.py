import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from mainwave_engine import case_review as CR


class CaseReviewTests(unittest.TestCase):
    def fixture(self):
        dates = pd.bdate_range("2026-06-01", periods=9).strftime("%Y-%m-%d").tolist()
        frame = pd.DataFrame([dict(date=d, open=10.4, high=10.8, low=10.1,
                                   close=10.3, return_index=10.3, amount=1e8) for d in dates])
        rows = [dict(date=d, cycle_id="cycle1", cycle_stage="MARKUP", breakout_id="b1",
                     breakout_status="ADVANCING", setup_status="NONE") for d in dates]
        rows[3].update(setup_status="CONFIRMED", setup_confirmed_at=dates[3],
            setup_trigger_index=10, setup_invalid_index=9, setup_frozen_atr=1,
            setup_breakout_accepted_at=dates[0], setup_breakout_advanced_at=dates[1],
            setup_adjustment_from=dates[2], setup_breakout_id="b1")
        rows[6]["cycle_stage"] = "FAILED"
        payload = {"case": {"symbol": "sh_600000", "start": dates[0], "cycle_id": "cycle1"},
                   "lifecycle": {"history": rows}, "chart": {"bars": [{"date": d} for d in dates]}}
        return payload, frame, dates

    def test_fill_after_confirmation_and_exit_after_failure(self):
        p, f, d = self.fixture()
        r = CR.review(p, f)
        t = r["trade"]
        self.assertEqual(t["status"], "CLOSED")
        self.assertEqual((t["entry"]["signal_at"], t["entry"]["date"]), (d[3], d[4]))
        self.assertEqual((t["exit"]["signal_at"], t["exit"]["date"]), (d[6], d[7]))
        self.assertEqual(t["gross_return_pct"], 0)
        self.assertFalse(r["profitability_validated"])

    def test_no_entry_is_not_zero_return(self):
        p, f, _ = self.fixture()
        p["lifecycle"]["history"][3]["setup_status"] = "NONE"
        t = CR.review(p, f)["trade"]
        self.assertEqual(t["status"], "NO_ENTRY")
        self.assertIsNone(t["gross_return_pct"])
        self.assertIsNone(t["unrealized_return_pct"])

    def test_reject_gap_without_chasing(self):
        p, f, d = self.fixture()
        f.loc[4, "open"] = 10.7
        t = CR.review(p, f)["trade"]
        self.assertEqual(t["status"], "NO_ENTRY")
        self.assertEqual(t["rejections"][0]["date"], d[4])

    def test_not_before_identification_or_from_different_cycle(self):
        p, f, d = self.fixture()
        p["case"]["target_ignition_at"] = d[5]
        self.assertIsNone(CR.review(p, f)["trade"]["entry"])
        p["case"]["target_ignition_at"] = d[0]
        p["lifecycle"]["history"][3]["cycle_id"] = "old-cycle"
        self.assertIsNone(CR.review(p, f)["trade"]["entry"])

    def test_same_chain_required(self):
        p, f, _ = self.fixture()
        p["lifecycle"]["history"][3]["setup_breakout_id"] = "other-breakout"
        self.assertIsNone(CR.review(p, f)["trade"]["entry"])

    def test_one_price_session_delays_exit(self):
        p, f, d = self.fixture()
        f.loc[7, ["open", "high", "low", "close"]] = 9
        t = CR.review(p, f)["trade"]
        self.assertEqual(t["exit"]["date"], d[8])
        self.assertEqual(t["exit"]["signal_at"], d[6])

    def test_exit_gap_uses_actual_next_open_not_frozen_stop(self):
        p, f, _ = self.fixture()
        f.loc[7, "open"] = 8
        t = CR.review(p, f)["trade"]
        self.assertEqual(t["exit"]["raw_price"], 8)
        self.assertAlmostEqual(t["gross_return_pct"], (8 / 10.4 - 1) * 100)

    def test_failure_on_entry_close_exits_no_earlier_than_next_day(self):
        p, f, d = self.fixture()
        p["lifecycle"]["history"][4]["cycle_stage"] = "FAILED"
        t = CR.review(p, f)["trade"]
        self.assertEqual(t["entry"]["date"], d[4])
        self.assertEqual(t["exit"]["signal_at"], d[4])
        self.assertEqual(t["exit"]["date"], d[5])

    def test_corporate_action_return_uses_index_not_raw_price(self):
        p, f, _ = self.fixture()
        f.loc[5:, ["open", "high", "low", "close"]] /= 2
        t = CR.review(p, f)["trade"]
        self.assertAlmostEqual(t["gross_return_pct"], 0)
        self.assertAlmostEqual(t["entry"]["raw_price"], 10.4)
        self.assertAlmostEqual(t["exit"]["raw_price"], 5.2)
        self.assertAlmostEqual(t["entry"]["chart_price"], 5.2)

    def test_prefix_states_no_same_close_fill_or_forced_exit(self):
        p, f, _ = self.fixture()
        for length, state in [(4, "ENTRY_PENDING"), (5, "OPEN"), (7, "EXIT_PENDING"), (9, "CLOSED")]:
            prefix = copy.deepcopy(p)
            prefix["chart"]["bars"] = prefix["chart"]["bars"][:length]
            prefix["lifecycle"]["history"] = prefix["lifecycle"]["history"][:length]
            t = CR.review(prefix, f.iloc[:length])["trade"]
            self.assertEqual(t["status"], state)
            if state != "CLOSED":
                self.assertIsNone(t["gross_return_pct"])

    def test_audit_target_and_theme_are_distinct_and_not_route_two(self):
        p, f, d = self.fixture()
        p["case"].update(t0=d[2], target_ignition_at=d[1], discovery_audit={"first_signal": d[0]})
        r = CR.review(p, f)
        self.assertEqual(r["route"]["id"], "R1")
        self.assertEqual(r["events"][0]["kind"], "TARGET_IGNITION")
        self.assertEqual(r["events"][1]["kind"], "THEME_IGNITION")
        self.assertFalse(r["route"]["production_hit_proven"])

    def test_route_three_requires_dated_link_evidence(self):
        p, f, d = self.fixture()
        context = {"id": "R3", "source": "dated-research", "theme_confirmed_at": d[0], "propagation_at": d[1]}
        p["case"]["route_context"] = context
        self.assertIsNone(CR.review(p, f)["trade"]["entry"])
        context.update(leader_symbol="sh_600001", link_evidence="dated business linkage")
        self.assertIsNotNone(CR.review(p, f)["trade"]["entry"])

    def test_route_two_waits_for_theme_even_after_discovery(self):
        p, f, d = self.fixture()
        p["case"]["route_context"] = {"id": "R2", "source": "dated-pool", "discovered_at": d[0], "theme_confirmed_at": d[5]}
        self.assertIsNone(CR.review(p, f)["trade"]["entry"])

    def test_refresh_preserves_selection_and_never_rescans(self):
        from mainwave_engine import historical_cases as HC
        p, f, d = self.fixture()
        p["case"].update(id="fixture", sessions=len(d), window_start=d[0], window_end=d[-1])
        p["setups"] = {"history": []}
        manifest = {"version": "old", "as_of": d[-1], "source_sha256": {"old-audit": "frozen"}, "cases": [p["case"]]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "fixture.json").write_text(json.dumps(p), encoding="utf-8")
            with patch.object(HC.SA, "analyze", return_value=copy.deepcopy(p)), patch.object(HC.SA, "_data_frame", return_value=(f, f)), patch.object(HC, "verify", return_value={"status": "PASS"}), patch.object(HC, "build", side_effect=AssertionError("no rescan")):
                self.assertEqual(HC.refresh(root)["status"], "PASS")
            saved = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["source_sha256"], manifest["source_sha256"])
            self.assertEqual(saved["as_of"], manifest["as_of"])
            self.assertEqual(saved["cases"][0]["id"], "fixture")
            self.assertEqual(saved["cases"][0]["window_end"], d[-1])
            self.assertEqual(saved["review_version"], CR.VERSION)


if __name__ == "__main__":
    unittest.main()
