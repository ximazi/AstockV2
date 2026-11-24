import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from . import hotspot_relay as H
from . import hotspot_funnel as F


DAY, PREV = "2026-11-17", "2026-11-16"


def row(symbol, day, theme="华为算力", status="CORROBORATED"):
    return {"symbol": symbol, "name": symbol, "attribution": {"known_at": day, "theme": theme, "status": status}}


class RelayTests(unittest.TestCase):
    def result(self, rows, prior_rows, perf, previous=None):
        symbols = [r["symbol"] for r in prior_rows]
        facts = {"previous_day": PREV, "boards": {"U": [r["symbol"] for r in rows]},
                 "yesterday_pool": {"basis": "FROZEN_PREVIOUS_REVIEW", "members": symbols, "performance": [
                     {"symbol": s, "status": "OK", "close_vs_pre": v} for s, v in perf.items()]}}
        old = {"rows": prior_rows, "relay": {"qualified_watch": {"b": "B"}}} if previous is None else previous
        with patch.object(H, "_frozen", return_value=(old, None)), patch.object(H, "_carriers", return_value=(None, {})):
            return H.build(DAY, ".", facts, {"rows": rows}, {"market_allowed": False, "eligible_count": 0})

    def test_diffusion_requires_independent_peer_and_preserves_veto(self):
        r = self.result([row("a", DAY), row("b", DAY, "华为昇腾具身智能算力")], [row("a", PREV)], {"a": 3})
        self.assertEqual(r["chains"][0]["state"], "接力扩散观察")
        self.assertEqual(r["chains"][0]["new_board_members"], ["b"])
        self.assertEqual(r["watch_hits_today"], ["b"])
        self.assertFalse(r["market_allowed"] or r["orders_enabled"])

    def test_relabelled_old_winner_is_not_new_board(self):
        r = self.result([row("a", DAY), row("b", DAY)], [row("a", PREV), row("b", PREV, "其他")], {"a": 3, "b": 1})
        self.assertNotIn("接力扩散观察", [x["state"] for x in r["chains"]])

    def test_old_leader_is_not_diffusion_and_loss_does_not_prove_theme_death(self):
        old = [row("a", PREV)]
        self.assertEqual(self.result([row("a", DAY)], old, {"a": 10})["chains"][0]["state"], "仅原班封板")
        self.assertEqual(self.result([], old, {"a": -9})["chains"][0]["state"], "旧核心负反馈")
        self.assertEqual(self.result([], old, {})["chains"][0]["state"], "旧核心反馈不完整")

    def test_no_future_weak_or_missing_evidence(self):
        r = self.result([row("a", "2026-11-18"), row("b", DAY, status="INSUFFICIENT")], [], {}, previous={})
        self.assertEqual((r["attributed"], r["limit_count"], r["chains"]), (0, 2, []))
        self.assertIsNone(r["watch_hits_today"])

    def test_ignore_late_rebuild_even_when_latest(self):
        with TemporaryDirectory() as tmp:
            for ver, rendered in [("v001", PREV+"T23:30:00+08:00"), ("v002", DAY+"T01:30:00+08:00"),
                                  ("v003", DAY+"T09:15:00+08:00")]:
                folder = Path(tmp) / "fupan" / ver
                folder.mkdir(parents=True)
                (folder / "report.json").write_text(json.dumps({"meta": {"forward_frozen": True,
                    "trade_date": PREV, "target_date": DAY, "rendered_at": rendered}}))
                (folder / "hotspot_core.json").write_text(json.dumps({"as_of": PREV}))
            self.assertEqual(H._frozen(Path(tmp), PREV)[1].parent.name, "v002")
            self.assertEqual(H._frozen(Path(tmp), PREV, PREV+"T23:59:59+08:00")[1].parent.name, "v001")

    def test_quiet_session_carries_only_fresh_routes_without_inventing_an_exit(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "route_candidates.csv"
            path.write_text("symbol,as_of,data_status,route_id,route_status,route_status_label,breakout_status\n"
                            f"a,{DAY},FRESH,R1,WAIT_PRICE,wait,PULLBACK\n", encoding="utf-8")
            cutoff = DAY+"T23:00:00+08:00"
            path.with_suffix(".published.json").write_text(json.dumps({
                "as_of": DAY, "sha256": H.file_sha(path), "available_at": DAY+"T21:00:00+08:00"}))
            os.utime(path, (1, 1))  # A copied file must not change the historical decision.
            old = {"relay": {"followup": {"a": {"last_board": "2026-11-13"}}}}
            keep, gaps, _ = H._followup(Path(tmp), DAY, PREV, old, ["b"], [], cutoff)
            self.assertEqual((keep["a"]["last_board"], keep["a"]["routes"][0]["state"], gaps),
                             ("2026-11-13", "WAIT_PRICE", ["b"]))
            self.assertEqual(keep["b"]["tracking_state"], "NO_FRESH_ROUTE")
            self.assertEqual(H._followup(Path(tmp), DAY, PREV, old, [], [], DAY+"T14:30:00+08:00")[2], "DATA_WAIT")
            path.write_text("changed")
            self.assertEqual(H._followup(Path(tmp), DAY, PREV, old, [], [], cutoff)[2], "DATA_WAIT")

    def test_funnel_separates_permission_from_roles_and_does_not_read_future_evidence(self):
        stamp = DAY + "T22:30:00+08:00"
        core = {"rows": [{"symbol": s, "amount_yi": 3-i} for i,s in enumerate("abc")]}
        attr = {"as_of": DAY, "available_at": stamp, "scope": {"full_limit_up_coverage": True},
                "rows": [{"symbol": s, "status": "CORROBORATED", "primary_theme": t} for s,t in
                         [("a","华为算力"), ("b","华为昇腾具身智能算力")]]}
        admission = {"as_of": DAY, "market_allowed": False, "rows": [{"symbol":"a","eligible":True}], "top":["a"]}
        def run(cutoff):
            return F.build(DAY,cutoff,{"boards":{"U":list("abc")}},core,attr,admission,{"followup":{}})
        r = run(stamp)
        self.assertEqual((r["role_watch"],r["participation_top"],r["grades"]), (["a","b"],[],{"CORROBORATED":2,"DATA_WAIT":1}))
        self.assertEqual(r["rows"][0]["chain_id"],r["rows"][1]["chain_id"])
        self.assertEqual(r["r3_shadow"][0]["stage"],"NO_VERIFIED_INDEPENDENT_MEMBERS")
        with self.assertRaisesRegex(ValueError, "post-close"):
            run(DAY+"T14:30:00+08:00")
        self.assertEqual(run(DAY+"T22:00:00+08:00")["grades"],{"DATA_WAIT":3})
        admission["market_allowed"] = True
        self.assertEqual(run(stamp)["participation_top"], ["a"])

    def test_review_uses_equal_prior_budget_and_does_not_zero_missing_outcomes(self):
        previous = {"focus":["x","b","a"],"funnel":{"version":F.VERSION,"as_of":PREV,"role_watch":["a"]}}
        facts = {"previous_day":PREV,"yesterday_pool":{"basis":"FROZEN_PREVIOUS_REVIEW","members":["a","b"],
                 "performance":[{"symbol":"a","status":"OK","close_vs_pre":2}]}}
        review = F.forward_check(previous,facts)
        self.assertEqual(review["equal_budget"],1)
        self.assertEqual((review["old_focus"]["symbols"],review["old_focus"]["mean_pct"]),(["b"],None))
        self.assertEqual(review["shadow_roles"]["mean_pct"],2)
        self.assertEqual(F.forward_check({},facts)["status"], "NO_COMPARABLE_FROZEN_PAIR")
