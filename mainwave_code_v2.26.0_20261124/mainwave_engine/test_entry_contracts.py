"""Entry contracts: drafts only from complete daily facts, never approved, intraday contracts stay research."""
import unittest

from . import entry_contracts as EC


def rec(symbol="sh_1", accepted_at="2026-09-22", status="ACCEPTED", close=10.0, atr=0.4, invalid=9.4, stage="MARKUP", setup="CONFIRMED",
        bias=1.0, cl=0.8, rel=1.2, lower=9.6, upper=10.3):
    return {"symbol": symbol, "opportunity_id": symbol + ":c",
            "source_record": {"accepted_at": accepted_at, "breakout_status": status, "close": close, "invalid": invalid, "route_bias_atr": bias,
                              "review_lower": lower, "review_upper": upper},
            "evidence": {"route_evidence": {"accepted_at": accepted_at, "atr_price": atr}, "state_vector": {"stage": stage, "setup_status": setup}},
            "features": {"close_location": cl, "relative_turnover": rel}}


class ContractTests(unittest.TestCase):
    def test_permissions_are_declared(self):
        self.assertEqual({k: v["permission"] for k, v in EC.CONTRACTS.items()},
                         {"B1": "SHADOW", "B2": "PAPER_APPROVED", "B3": "SHADOW", "B4": "RESEARCH", "B5": "RESEARCH"})

    def test_b1_only_on_acceptance_day_with_invalidation(self):
        d = EC.draft_b1(rec(), "2026-09-22", "2026-09-23")
        self.assertEqual((d["contract"], d["permission"], d["ready"], d["entry_max"], d["invalidation"]), ("B1", "SHADOW", True, 10.2, 9.4))
        self.assertEqual((d["valid_from"][:10], d["expires_at"][:10]), ("2026-09-23", "2026-09-23"))
        self.assertIsNone(EC.draft_b1(rec(accepted_at="2026-09-21"), "2026-09-22", "2026-09-23"))
        self.assertIsNone(EC.draft_b1(rec(status="ATTEMPT"), "2026-09-22", "2026-09-23"))
        gap = EC.draft_b1(rec(invalid=None), "2026-09-22", "2026-09-23")
        self.assertFalse(gap["ready"])
        self.assertEqual(gap["missing"], ["invalidation"])
        filled = EC.draft_b1(rec(invalid=None), "2026-09-22", "2026-09-23", route_row={"invalid_price": 9.3})
        self.assertTrue(filled["ready"])
        bad = EC.draft_b1(rec(invalid=10.5), "2026-09-22", "2026-09-23")
        self.assertFalse(bad["ready"])
        self.assertEqual(bad["missing"], ["invalidation_below_close"])

    def test_b3_needs_in_band_continuation_with_live_setup_and_absorbed_close(self):
        d = EC.draft_b3(rec(accepted_at="2026-09-01"), "2026-09-22", "2026-09-23")
        self.assertEqual((d["contract"], d["ready"], d["consolidation_low"]), ("B3", True, 9.6))
        self.assertIsNone(EC.draft_b3(rec(stage="IGNITION"), "2026-09-22", "2026-09-23"))
        self.assertIsNone(EC.draft_b3(rec(bias=2.5), "2026-09-22", "2026-09-23"))
        weak = EC.draft_b3(rec(cl=0.3), "2026-09-22", "2026-09-23")
        self.assertIn("re_ignition_close", weak["missing"])

    def test_drafts_and_summary_never_carry_approval(self):
        ds = EC.drafts(rec(), "2026-09-22", "2026-09-23")
        self.assertEqual([d["contract"] for d in ds], ["B1", "B3"])
        self.assertTrue(all("approval" not in d and d["permission"] != "PAPER_APPROVED" for d in ds))
        s = EC.summarize(ds)
        self.assertEqual((s["B1"]["drafts"], s["B1"]["ready"], s["B3"]["permission"]), (1, 1, "SHADOW"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
