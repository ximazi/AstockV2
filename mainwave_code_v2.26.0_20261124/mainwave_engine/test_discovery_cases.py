from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from mainwave_engine import discovery_cases as DC
from mainwave_engine import historical_cases as HC


def fixture(states=None):
    dates = ["2026-07-27", "2026-07-28", "2026-07-29", "2026-07-30", "2026-07-31"]
    trace = pd.DataFrame({"date": dates, "state": states or ["ARMED", "READY", "FORMING", "OUT", "READY"],
                          "capital_stage": "COILED", "prefilter": "True", "supply_ready": "True",
                          "structure_ready": "True"})
    row = {"audit_case": "test", "symbol": "sh_600000", "name": "sample", "industry": "sample",
           "theme_label": "sample", "t0": dates[3], "first_armed": dates[0],
           "first_capital_stage": dates[0], "covered_before_t0": "True", "bottleneck": "",
           "membership_source": "historical_backfill", "first_limit_date": dates[3]}
    return row, trace


class DiscoveryCaseTests(unittest.TestCase):
    def test_historical_hit_is_not_continuous_or_production_success(self):
        row, trace = fixture()
        result = DC.audit_case(row, trace, {})
        audit = result["discovery_audit"]
        self.assertEqual(result["category"], "DISCOVERY_HIT")
        self.assertEqual(audit["first_signal"], "2026-07-27")
        self.assertEqual(audit["last_signal_before_t0"], "2026-07-28")
        self.assertEqual(audit["lead_sessions"], 3)
        self.assertEqual(audit["gap_sessions_to_t0"], 2)
        self.assertFalse(audit["covered_on_eve"])
        self.assertFalse(audit["continuous_from_first_to_eve"])
        self.assertFalse(audit["production_hit_proven"])
        self.assertFalse(audit["same_cycle_success_proven"])

    def test_continuous_coverage_includes_triggered_as_in_original_audit(self):
        row, trace = fixture(["ARMED", "TRIGGERED", "READY", "OUT", "OUT"])
        audit = DC.audit_case(row, trace, {})["discovery_audit"]
        self.assertTrue(audit["covered_on_eve"])
        self.assertTrue(audit["continuous_from_first_to_eve"])

    def test_reentry_is_not_continuous_coverage(self):
        row, trace = fixture(["ARMED", "FAILED", "READY", "READY", "READY"])
        audit = DC.audit_case(row, trace, {})["discovery_audit"]
        self.assertTrue(audit["covered_on_eve"])
        self.assertFalse(audit["continuous_from_first_to_eve"])

    def test_ignition_day_or_later_signal_does_not_rescue_a_miss(self):
        row, trace = fixture(["OUT", "FORMING", "OUT", "READY", "ARMED"])
        row.update(first_armed="", covered_before_t0="False")
        result = DC.audit_case(row, trace, {})
        self.assertEqual(result["category"], "DISCOVERY_MISS")
        self.assertIsNone(result["discovery_audit"]["lead_sessions"])
        self.assertEqual(result["discovery_audit"]["last_signal_before_t0"], "")

    def test_reject_summary_disagreement_duplicates_and_missing_t0(self):
        row, trace = fixture()
        bad = dict(row, first_armed="2026-07-28")
        with self.assertRaises(ValueError):
            DC.audit_case(bad, trace, {})
        with self.assertRaises(ValueError):
            DC.audit_case(row, pd.concat([trace, trace.iloc[:1]]), {})
        with self.assertRaises(ValueError):
            DC.audit_case(row, trace.iloc[:3], {})

    def test_libraries_merge_without_overwriting_or_scanning_and_reject_unknown_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "discovery_audit"
            child.mkdir()
            versions = {"cycle_version": HC.SL.VERSION, "setup_version": HC.PCS.VERSION,
                        "phase_version": HC.PE.VERSION, "analyzer_version": HC.SA.VERSION}
            for base, case_id in ((root, "cycle_example"), (child, "audit_example")):
                manifest = dict(versions, as_of="2026-09-07", cases=[{"id": case_id}])
                (base / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                (base / f"{case_id}.json").write_text(json.dumps({"id": case_id}), encoding="utf-8")
            with patch.object(HC, "build", side_effect=AssertionError("no scan")):
                self.assertEqual(len(HC.case_list(root)["cases"]), 2)
                self.assertEqual(HC.case_detail("audit_example", root), {"id": "audit_example"})
                self.assertEqual(HC.case_detail("audit_example", child), {"id": "audit_example"})
                self.assertEqual(HC.case_detail("cycle_example", root), {"id": "cycle_example"})
                with self.assertRaises(ValueError):
                    HC.case_detail("../positions", root)
            manifest["cycle_version"] = "old"
            (child / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(HC.case_list(root)["stale"])
            with self.assertRaises(ValueError):
                HC.case_detail("audit_example", root)
            self.assertEqual(HC.case_detail("cycle_example", root)["id"], "cycle_example")


if __name__ == "__main__":
    unittest.main()
