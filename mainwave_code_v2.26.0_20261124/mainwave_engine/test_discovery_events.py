"""Discovery register: every source counted on the full union, states are observation states, names never vanish."""
import unittest

from . import discovery_events as DE
from .trading import market_policy as MP


def cand(symbol, state="WAIT_SETUP", stage="MARKUP", setup="NONE"):
    return {"symbol": symbol, "name": symbol, "theme": "T", "state": state, "route_ids": "R1", "tracking_only": False,
            "evidence": {"state_vector": {"stage": stage, "setup_status": setup}}}


class RegisterTests(unittest.TestCase):
    def test_union_counts_and_states(self):
        records = [cand("a", "PLAN_REVIEW"), cand("b", setup="FORMING"), cand("c", "DATA_WAIT"), cand("d", stage="FAILED")]
        watch = [{"symbol": "b", "name": "b", "industry": "T"}, {"symbol": "w1", "name": "w1", "industry": "U"}, {"symbol": "w2", "name": "w2", "industry": "U"}]
        pinned = [{"symbol": "p1", "name": "p1"}, {"symbol": "a", "name": "a"}]
        plans = [{"symbol": "a", "name": "a", "plan_id": "P", "approved": True, "expired": False, "open_position": False}]
        reg = DE.build("2026-09-22", records, [{"symbol": "a", "shadow_rank": 1, "reading": "r"}], watch, pinned, plans)
        s = reg["summary"]
        self.assertEqual((s["candidates"], s["watch"], s["overlap"], s["outside"], s["union"], s["pinned"]), (4, 3, 1, 2, 6, 2))
        by = {e["symbol"]: e for e in reg["events"]}
        self.assertEqual(by["a"]["state"], "PLAN_APPROVED_WAIT_TRIGGER")
        self.assertEqual(sorted(by["a"]["sources"]), ["ENGINE_CANDIDATE", "PAPER_BOOK", "USER_PINNED"])
        self.assertEqual(by["b"]["state"], "SETUP_FORMING")
        self.assertEqual(by["c"]["state"], "DATA_WAIT")
        self.assertEqual(by["d"]["state"], "INVALIDATED")
        self.assertEqual(by["w1"]["state"], "EVIDENCE_PENDING")
        self.assertIn("冻结触发价与失效价", by["w1"]["missing"])
        self.assertEqual(by["p1"]["state"], "EVIDENCE_PENDING")
        self.assertIn("必保留", by["p1"]["not_reviewed_reason"])
        self.assertEqual(s["total"], 7)

    def test_continuity_deactivation_and_reactivation(self):
        day1 = DE.build("2026-09-21", [cand("a"), cand("b")], None, None, None, None)
        day2 = DE.build("2026-09-22", [cand("a")], None, None, None, None, previous=day1)
        by = {e["symbol"]: e for e in day2["events"]}
        self.assertEqual(by["b"]["state"], "DEACTIVATED")
        self.assertEqual(by["a"]["first_seen"], "2026-09-21")
        self.assertEqual(day2["summary"]["deactivated"], 1)
        day3 = DE.build("2026-09-23", [cand("a"), cand("b")], None, None, None, None, previous=day2)
        by3 = {e["symbol"]: e for e in day3["events"]}
        self.assertTrue(by3["b"]["reactivated"])
        self.assertEqual(by3["b"]["first_seen"], "2026-09-21")
        self.assertEqual(day3["summary"]["reactivated"], 1)


class MarketPolicyTests(unittest.TestCase):
    def test_dimensions_and_engine_policy_keep_frozen_gate(self):
        hot = {k: {"status": "OK", "reading": "热"} for k in ("limit_up", "pool_mean_pct", "pool_up_share", "open_premium_pct", "promotion_1_to_2", "seal_failure_rate", "limit_down")}
        out = MP.evaluate({"structure": "RANGE", "temperature": "NORMAL", "state": "ORANGE", "data_status": "FRESH"}, hot, {"yesterday_pool_negative": {"count": 5}}, "2026-09-21")
        d, p = out["dimensions"], out["policy"]
        self.assertEqual((d["relay_sentiment"]["reading"], d["loss_effect"]["reading"], d["system_health"]["reading"]), ("HOT", "CONTAINED", "OK"))
        self.assertEqual((p["frozen_gate"], p["frozen_allows_new_plans"], p["E1"]["shadow"], p["E3"]["shadow"], p["E2"]["shadow"]), ("ORANGE", False, "ALLOW_SHADOW", "ALLOW_SHADOW", "RESEARCH_ONLY"))
        self.assertEqual(p["divergence_from_frozen"], ["E1", "E3"])
        blocked = MP.evaluate({"structure": "BULL", "state": "GREEN", "data_status": "STALE"})
        self.assertEqual((blocked["policy"]["hard_gate"], blocked["policy"]["E1"]["shadow"]), ("BLOCK", "BLOCK"))
        unknown = MP.evaluate({"structure": "BEAR", "state": "ORANGE", "data_status": "FRESH"})
        self.assertEqual((unknown["dimensions"]["relay_sentiment"]["reading"], unknown["policy"]["E1"]["shadow"]), ("UNKNOWN", "WAIT"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
