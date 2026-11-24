import itertools
import unittest

import pandas as pd

from mainwave_engine import preparation_track as PT
from mainwave_engine import route_contracts as RC
from mainwave_engine import test_analysis_core as fixtures
from mainwave_engine import analysis_core as AC


class ContractTests(unittest.TestCase):
    def test_route_order_never_changes_entry_decision(self):
        data, market = fixtures.fixture()
        outcomes = []
        for routes in itertools.permutations(["R1", "R2", "R3"]):
            data["route_evidence"] = {"route_ids": "|".join(routes), "route_requirements_met": True}
            core = AC.evaluate(data, market)
            outcomes.append((core["entry_contract"], core["advice"]))
        self.assertTrue(all(o == outcomes[0] for o in outcomes))
        self.assertEqual(outcomes[0][1], "WATCH")

    def test_real_price_confirmation_does_not_need_a_theme(self):
        route = {"route_ids": "R3|R2", "independent_price_evidence": True}
        result = RC.entry_context(route, {})
        self.assertTrue(result["requirements_met"])
        self.assertEqual(result["contract"], "PRICE_INDEPENDENT")

    def test_dated_proxy_is_not_a_verified_chain(self):
        result = RC.entry_context({"route_ids": "R3"}, {"lifecycle": "IGNITED"})
        self.assertFalse(result["requirements_met"])
        self.assertIn("STRICT_CHAIN_UNVERIFIED", result["missing"])

    def test_secondary_r3_clue_cannot_veto_valid_r2_contract(self):
        result = RC.entry_context({"route_ids": "R3|R2", "route_requirements_met": True}, {"lifecycle": "FERMENTING"})
        self.assertTrue(result["requirements_met"])
        self.assertEqual(result["contract"], "THEME_REQUIRED")

    def test_price_evidence_requires_identity_dates_and_demand(self):
        row = {"eligible": True, "date": "2026-09-09", "breakout_id": "one",
               "breakout_status": "ACCEPTED", "breakout_accepted_at": "2026-09-09", "bias_atr": 1., "demand": True}
        self.assertTrue(RC.price_confirmation(row))
        for field, value in [("breakout_id", ""), ("breakout_accepted_at", "2026-09-10"),
                             ("demand", False), ("supply_warning", True)]:
            self.assertFalse(RC.price_confirmation({**row, field: value}))


class PreparationTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=40),
                             "breakout_id": "", "breakout_status": "NONE", "breakout_event": "",
                             "probe_at": "2025-12-31", "prepared": True, "eligible": True,
                             "structure_floor": 9., "atr20": 1., "return_index": 10.,
                             "amount_contract": .7, "turnover_contract": .7, "range_contract": 1.})

    def test_daily_threshold_flip_keeps_first_seen_and_ttl(self):
        f = self.frame()
        f.loc[1:3, "prepared"] = False
        r = PT.annotate(f, "sh_600001")
        self.assertEqual(r.preparation_id.nunique(), 1)
        self.assertEqual(r.preparation_first_seen.nunique(), 1)
        self.assertEqual(r.preparation_state.iloc[31], "EXPIRED")
        self.assertFalse(r.preparation_valid.iloc[39])
        self.assertTrue(r.activity_contraction.iloc[0])
        self.assertFalse(r.volatility_contraction.iloc[0])

    def test_parent_failure_cannot_be_rescued_by_a_new_event(self):
        f = self.frame()
        f.loc[2:4, "breakout_id"] = "old"
        f.loc[2, ["breakout_event", "breakout_status"]] = ["ATTEMPT", "ATTEMPT"]
        f.loc[3:4, "breakout_status"] = "FAILED"
        f.loc[5:, ["breakout_id", "breakout_status"]] = ["new", "ACCEPTED"]
        r = PT.annotate(f, "sh_600001")
        self.assertEqual(r.preparation_state.iloc[5], "INVALID")
        f.loc[6:, "probe_at"] = "2026-01-08"
        r = PT.annotate(f, "sh_600001")
        self.assertNotEqual(r.preparation_id.iloc[0], r.preparation_id.iloc[6])
        self.assertEqual(r.preparation_origin.iloc[6], "POST_ACCEPTANCE")

    def test_attempt_preparation_is_not_pre_attempt_warning(self):
        f = self.frame()
        f["breakout_status"], f["breakout_id"] = "ATTEMPT", "one"
        full = PT.annotate(f, "sh_600001")
        self.assertEqual(full.preparation_origin.iloc[0], "ATTEMPT_UNDERWAY")
        pd.testing.assert_frame_equal(full.iloc[:20], PT.annotate(f.iloc[:20], "sh_600001"))
