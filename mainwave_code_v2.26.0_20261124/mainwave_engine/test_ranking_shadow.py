"""Shadow screening v2: layered readings, stated order, structured change, dated route order, batch-dated
re-eligibility, cohort wording, full counts before truncation, shortlist ≠ allocation."""
import copy
import unittest

from .trading import ranking_shadow as RS


def record(symbol, theme="T", state="WAIT_SETUP", stage="MARKUP", setup_status="NONE", local_role="CONTINUATION", location="UNKNOWN",
           bias=None, close_location=0.5, rel=1.0, excess5=1.0, excess20=5.0, routes="", prep="EXPIRED", pre_attempt=None,
           prepared_at="", attempt_at="", tracking=False, blockers=None, change="", rank=None, alpha="PRICE_ALPHA", return20=10.0,
           as_of="2026-09-21"):
    return {"opportunity_id": symbol + ":c", "symbol": symbol, "name": symbol, "theme": theme, "state": state,
            "blockers": blockers if blockers is not None else (["WAIT_PULLBACK_CONFIRMATION"] if state == "WAIT_SETUP" else []),
            "tracking_only": tracking, "route_ids": routes, "alpha_scope": alpha, "price_date": as_of,
            "strength_rank": rank, "participation_rank": rank,
            "features": {"status": "READY", "close_location": close_location, "relative_turnover": rel, "market_excess5": excess5, "market_excess20": excess20,
                         "peer_excess20": None, "peer_count": 0, "return20": return20, "drawdown20": 0.0},
            "evidence": {"as_of": as_of, "state_vector": {"stage": stage, "setup_status": setup_status, "location": location},
                         "route_evidence": {"preparation_state": prep, "demand": True, "activity_contraction": False, "bias_atr": bias,
                                            "prepared_before_attempt": pre_attempt, "prepared_at": prepared_at, "attempt_at": attempt_at}},
            "source_record": {"route_bias_atr": bias, "local_role": local_role, "evidence_change": change, "route_status": "WAIT_ACCEPTANCE", "preparation_state": prep}}


def row(rec, previous=None, as_of="2026-09-21", cohorts=None, market=None):
    return RS.read(rec, cohorts or {}, market or {}, previous, as_of)


class LayerTests(unittest.TestCase):
    def test_layers_are_read_separately(self):
        r = record("a", stage="MARKUP", setup_status="CONFIRMED", local_role="FIRST_RETEST", bias=1.2, close_location=0.8, rel=1.5, routes="R2|R1",
                   prepared_at="2026-09-01", attempt_at="2026-09-10", prep="ADVANCING")
        self.assertEqual(RS.stage_of(r), ("EARLY", "MARKUP"))
        self.assertEqual(RS.position_of(r), ("IN_BAND", 1.2))
        self.assertEqual(RS.route_of(r), "R2_THEN_R1")
        self.assertEqual(RS.trigger_of(r["features"]), "ABSORBED")
        self.assertTrue(RS.preparation_of(r)["live"])
        self.assertEqual(RS.stage_of(record("b", stage="OVERHEATED"))[0], "LATE")
        self.assertEqual(RS.stage_of(record("c", stage="REBUILDING"))[0], "REPAIRING")
        self.assertEqual(RS.stage_of(record("d", stage="FAILED"))[0], "FAILED")
        self.assertEqual(RS.position_of(record("e", bias=3.1))[0], "EXTENDED")
        self.assertEqual(RS.position_of(record("f", location="EXTENDED", bias=1.0))[0], "EXTENDED")
        self.assertEqual(RS.trigger_of({"close_location": 0.2, "relative_turnover": 2.0}), "WEAK_ACTIVE")
        self.assertEqual(RS.route_of(record("g", routes="R3")), "R3")

    def test_route_order_needs_dates(self):
        self.assertEqual(RS.route_of(record("a", routes="R1|R2")), "R1_AND_R2_UNORDERED")  # coexistence proves no order
        self.assertEqual(RS.route_of(record("b", routes="R1|R2", prepared_at="2026-09-12", attempt_at="2026-09-10")), "R1_AND_R2_UNORDERED")
        self.assertEqual(RS.route_of(record("c", routes="R1|R2", prepared_at="2026-09-01", attempt_at="2026-09-10")), "R2_THEN_R1")
        self.assertEqual(RS.route_of(record("d", routes="R1", pre_attempt=True)), "R2_THEN_R1")
        self.assertEqual(RS.route_of(record("e", routes="R1")), "R1")

    def test_change_from_structured_fields_not_prose(self):
        early = row(record("a", stage="IGNITION", bias=1.0))
        cont = row(record("a", stage="MARKUP", bias=1.0), previous=early)
        self.assertEqual(cont["change"], "PROGRESSING")  # normal lifecycle progress is not weakening
        late = row(record("a", stage="OVERHEATED", bias=1.0), previous=cont)
        self.assertEqual(late["change"], "WEAKENING")
        repaired = row(record("a", stage="IGNITION", bias=0.5), previous=row(record("a", stage="REBUILDING")))
        self.assertEqual(repaired["change"], "NEW")
        extended = row(record("a", stage="MARKUP", bias=3.0), previous=cont)
        self.assertEqual(extended["change"], "WEAKENING")
        # without a previous row only unambiguous prose is read; a negated 改善 is not an improvement
        self.assertEqual(row(record("b", change="无阶段或市场风险警报：改善"))["change"], "IMPROVING")
        self.assertEqual(row(record("b", change="证据未改善"))["change"], "UNKNOWN")
        self.assertEqual(row(record("b", change="需求延续证据：减弱或时效消退"))["change"], "WEAKENING")
        self.assertEqual(row(record("b", change="新周期，重新建立证据基线"))["change"], "NEW")
        self.assertEqual(row(record("b", change=""))["change"], "UNKNOWN")

    def test_tracking_re_eligibility_is_batch_dated(self):
        live = record("a", tracking=True, setup_status="CONFIRMED", blockers=["WAIT_PULLBACK_CONFIRMATION", "TRACKING_ONLY"])
        self.assertTrue(RS.re_eligible(live, "2026-09-21"))
        self.assertFalse(RS.re_eligible(live, "2026-09-23"))  # a 09-21 record is not fresh in a 09-23 batch
        self.assertFalse(RS.re_eligible(live, None))
        stale_evidence = record("b", tracking=True, setup_status="CONFIRMED", blockers=["TRACKING_ONLY"])
        stale_evidence["evidence"]["as_of"] = "2026-09-18"
        self.assertFalse(RS.re_eligible(stale_evidence, "2026-09-21"))
        self.assertEqual(RS.state_without(live, {"TRACKING_ONLY"}), ("WAIT_SETUP", ["WAIT_PULLBACK_CONFIRMATION"]))
        self.assertFalse(RS.re_eligible(record("c", tracking=True, setup_status="NONE", blockers=["TRACKING_ONLY"]), "2026-09-21"))
        self.assertFalse(RS.re_eligible(record("d", tracking=True, setup_status="CONFIRMED", blockers=["STRUCTURE_RISK", "TRACKING_ONLY"]), "2026-09-21"))

    def test_cohort_reading_is_named_for_the_cohort(self):
        rows = [record(f"s{i}", theme="X", close_location=0.9, rel=1.2, excess5=2.0, stage="MARKUP") for i in range(3)]
        cohorts = RS.theme_cohorts(rows)
        self.assertEqual(cohorts["X"]["state"], "STRONG")
        self.assertEqual(cohorts["X"]["basis"], "CANDIDATE_COHORT_ONLY_NOT_MARKET_BREADTH")
        fading = [record(f"f{i}", theme="Y", close_location=0.1, rel=2.0, excess5=-3.0) for i in range(3)]
        self.assertEqual(RS.theme_cohorts(fading)["Y"]["state"], "WEAK")
        self.assertEqual(RS.theme_cohorts(rows[:2])["X"]["state"], "UNKNOWN")
        # conditions first, then the top cut: a fourth qualifying theme is not displaced by a non-qualifying stronger one
        many = []
        for t, ex, cl in (("A", 9.0, 0.1), ("B", 5.0, 0.9), ("C", 4.0, 0.9), ("D", 3.0, 0.9), ("E", 2.0, 0.9)):
            many += [record(f"{t}{i}", theme=t, close_location=cl, rel=1.2, excess5=ex, stage="MARKUP") for i in range(3)]
        c = RS.theme_cohorts(many)
        self.assertEqual([t for t in "ABCDE" if c[t]["state"] == "STRONG"], ["B", "C", "D"])
        self.assertIn("候选群体", RS.LABELS["mainline"]["STRONG"])

    def test_market_theme_context_from_full_membership_facts(self):
        facts = {"as_of": "2026-09-21", "top_by_limit_up": [{"theme": "半导体", "members": 200, "limit_up": 12, "up": 150, "breadth": {"value": 0.75}}],
                 "top_by_limit_share": [{"theme": "租售同权", "members": 24, "limit_up": 4, "up": 24, "breadth": {"value": 1.0}}],
                 "top_by_amount_share": [{"theme": "银行", "members": 40, "limit_up": 0, "up": 10, "breadth": {"value": 0.25}}]}
        ctx = RS.market_theme_context(facts)
        self.assertEqual((ctx["半导体"]["state"], ctx["租售同权"]["state"], ctx["银行"]["state"]), ("STRONG", "STRONG", "WEAK"))
        self.assertEqual(ctx["半导体"]["as_of"], "2026-09-21")
        self.assertEqual(RS.market_theme_context(None), {})


class OrderTests(unittest.TestCase):
    def test_order_prefers_early_in_band_over_extended_runner(self):
        runner = record("run", stage="MARKUP", bias=3.5, location="EXTENDED", close_location=1.0, rel=3.0, excess5=20.0, excess20=50.0, rank=1, routes="R1")
        early = record("early", stage="IGNITION", bias=0.8, close_location=0.7, rel=1.3, excess5=3.0, excess20=8.0, rank=2, routes="R2|R1",
                       prepared_at="2026-09-01", attempt_at="2026-09-18", prep="ADVANCING")
        shadow = RS.evaluate({"as_of": "2026-09-21", "records": [runner, early]})
        by = {r["symbol"]: r for r in shadow["rows"]}
        self.assertEqual(by["run"]["variant_ranks"]["V0_frozen"], 1)
        self.assertEqual(by["early"]["variant_ranks"]["V2_position"], 1)
        self.assertEqual(shadow["rows"][0]["symbol"], "early")
        self.assertIn("早期/刚启动", by["early"]["reading"])
        self.assertIn("偏离过大", by["run"]["reading"])
        self.assertIn("先准备后点火（有日期）", by["early"]["reading"])
        self.assertEqual(shadow["basis"], "SHADOW_ONLY_NOT_CALIBRATED_NO_ORDERS")
        self.assertIn("偏好写明", shadow["order"])

    def test_market_theme_outranks_cohort_when_known(self):
        facts = {"as_of": "2026-09-21", "top_by_limit_up": [{"theme": "X", "members": 100, "limit_up": 10, "up": 80, "breadth": {"value": 0.8}}]}
        a = record("a", theme="X", stage="IGNITION", bias=1.0)
        b = record("b", theme="Y", stage="IGNITION", bias=1.0)
        shadow = RS.evaluate({"as_of": "2026-09-21", "records": [a, b]}, theme_facts=facts)
        self.assertEqual(shadow["rows"][0]["symbol"], "a")
        self.assertEqual(shadow["rows"][0]["market_theme"], "STRONG")
        self.assertIn("全市场题材活跃", shadow["rows"][0]["reading"])

    def test_state_gate_still_comes_first(self):
        plan = record("plan", state="PLAN_REVIEW", stage="OVERHEATED", bias=2.5, rank=5)
        wait = record("wait", state="WAIT_SETUP", stage="IGNITION", bias=0.5, rank=1)
        shadow = RS.evaluate({"as_of": "2026-09-21", "records": [plan, wait]})
        self.assertEqual(shadow["rows"][0]["symbol"], "plan")

    def test_shortlist_is_one_per_security_in_band_and_not_an_allocation(self):
        a = record("a", state="PLAN_REVIEW", theme="X", stage="IGNITION", bias=0.5)
        a2 = record("a", state="PLAN_REVIEW", theme="Y", stage="IGNITION", bias=0.5)
        a2["opportunity_id"] = "a:other"
        c = record("c", state="WAIT_SETUP", theme="Y", stage="MARKUP", bias=1.0)
        d = record("d", state="PLAN_REVIEW", theme="Z", stage="MARKUP", bias=3.0)
        u = record("u", state="PLAN_REVIEW", theme="W", stage="IGNITION", bias=None)
        r3 = record("r", state="PLAN_REVIEW", theme="V", stage="IGNITION", bias=0.3, routes="R3", alpha="R3_RESEARCH")
        rep = record("rep", state="WAIT_SETUP", theme="U", stage="REBUILDING", bias=0.5)
        shadow = RS.evaluate({"as_of": "2026-09-21", "records": [a, a2, c, d, u, r3, rep]})
        chosen = [x["symbol"] for x in RS.select_review_shortlist(shadow["rows"])]
        self.assertEqual(chosen, ["a", "c"])  # same security once; unknown position, extended, R3-only, repairing excluded
        self.assertEqual([x["symbol"] for x in shadow["summary"]["review_shortlist"]], chosen)
        self.assertFalse(hasattr(RS, "choose"))

    def test_counts_before_truncation(self):
        rows = [record(f"c{i}") for i in range(3)]
        watch = [{"symbol": f"c{i}"} for i in range(2)] + [{"symbol": f"n{i}"} for i in range(100)]
        shadow = RS.evaluate({"as_of": "2026-09-21", "records": rows}, preparation_watch=watch)
        pw = shadow["summary"]["preparation_watch"]
        self.assertEqual((pw["candidates"], pw["watch"], pw["overlap"], pw["outside"], pw["union"]), (3, 102, 2, 100, 103))
        self.assertEqual(shadow["preparation_watch_outside_total"], 100)
        self.assertEqual(len(shadow["preparation_watch_outside"]), 20)

    def test_multiday_uses_previous_shadow_and_never_touches_input(self):
        rec = record("a", stage="MARKUP")
        frozen_copy = copy.deepcopy(rec)
        previous = {"as_of": "2026-09-18", "rows": [{"opportunity_id": "a:c", "stage_bucket": "LATE", "position": "UNKNOWN", "trigger": "NEUTRAL",
                                                      "preparation": {"live": False}, "shadow_rank": 7}]}
        shadow = RS.evaluate({"as_of": "2026-09-21", "records": [rec]}, previous)
        self.assertEqual(shadow["rows"][0]["change"], "IMPROVING")
        self.assertEqual(shadow["rows"][0]["previous_shadow_rank"], 7)
        self.assertEqual(rec, frozen_copy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
