import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from . import first_board_watch as W
from . import hotspot_funnel as F
from .eod_review import file_sha


DAYS = ["2026-11-12", "2026-11-13", "2026-11-16", "2026-11-17"]
DAY, NEXT = DAYS[-1], "2026-11-18"
STAMP = DAY + "T23:00:00+08:00"
HELD = {"above_ma20_now": "True", "reclaim_failed": "False", "bias_atr": "0.73"}
FAILED = {**HELD, "above_ma20_now": "False", "reclaim_failed": "True"}


class FirstBoardWatchTests(unittest.TestCase):
    def pair(self, prior=HELD, current=HELD, height=1, grade="CORROBORATED", high=20.82, low=18.41):
        core = {"rows": [{"symbol": "a", "close": high, "high": high, "low": low}]}
        facts = {"boards": {"U": ["a", "b"]}, "ladder": {"heights": {"a": height, "b": 1}}}
        attribution = {"as_of": DAY, "available_at": STAMP, "scope": {"full_limit_up_coverage": True},
                       "rows": [{"symbol": "a", "primary_theme": "theme", "status": "CORROBORATED"},
                                {"symbol": "b", "primary_theme": "theme", "status": grade}]}
        funnel = F.build(DAY, STAMP, facts, core, attribution, {"as_of": DAY, "market_allowed": False}, {"followup": {}})
        pools = {DAYS[0]: {"a": prior}, DAYS[1]: {"a": FAILED}, DAYS[2]: {}, DAY: {"a": current}}
        return W.build(DAY, NEXT, DAYS[:-1], pools, core, funnel), funnel

    def test_two_sources_meet_outside_formal_route_without_entry_permission(self):
        watch, funnel = self.pair()
        self.assertEqual(funnel["participation_top"], [])
        self.assertEqual(watch["review_on"], NEXT)
        self.assertFalse(watch["orders_enabled"])
        pick, = watch["picks"]
        self.assertEqual((pick["symbol"], pick["route_state"], pick["peers"]), ("a", "NOT_IN_CANDIDATES", ["b"]))
        self.assertEqual(pick["failed_reclaims"], [DAYS[1]])
        row, = W.table_rows(watch, [])
        self.assertEqual(len(row), 8)
        self.assertIn(NEXT, row[6])

    def test_first_board_only_no_fabricated_preparation_or_theme(self):
        for change in ({"prior": FAILED}, {"current": FAILED}, {"height": 2},
                       {"grade": "DISPUTED"}, {"low": 20.82}, {"current": {**HELD, "bias_atr": "2.01"}}):
            with self.subTest(change=change):
                self.assertEqual(self.pair(**change)[0]["picks"], [])

    def test_preparation_receipts_bound_each_prior_session_before_next_open(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for day, when in [(DAYS[0], DAYS[0]+"T22:00:00+08:00"), (DAYS[1], DAYS[2]+"T12:00:00+08:00"),
                              (DAY, DAY+"T21:00:00+08:00")]:
                path = root / day.replace("-", "") / "preparation_watch.csv"
                path.parent.mkdir()
                path.write_text(f"as_of,symbol,detector\n{day},a,P5_union\n", encoding="utf-8")
                path.with_suffix(".published.json").write_text(json.dumps({"as_of": day,
                    "available_at": when, "sha256": file_sha(path)}), encoding="utf-8")
            last = root / DAY.replace("-", "") / "preparation_watch.csv"
            self.assertEqual(set(W.load_pools(root, DAYS, STAMP)), {DAYS[0], DAY})
            last.write_text("modified", encoding="utf-8")
            self.assertEqual(set(W.load_pools(root, DAYS, STAMP)), {DAYS[0]})

    def test_at_most_one_per_theme_and_three_overall(self):
        symbols = [str(i) for i in range(8)]
        core = {"rows": [{"symbol": s, "close": 20, "high": 20, "low": 19} for s in symbols]}
        funnel = {"attribution_available_at": STAMP, "rows": [{"symbol": s, "name": s, "height": 1,
                  "grade": "CORROBORATED", "chain": str(int(s) // 2)} for s in symbols]}
        pool = dict.fromkeys(symbols, HELD)
        watch = W.build(DAY, NEXT, DAYS[:-1], {DAYS[0]: pool, DAY: pool}, core, funnel)
        self.assertEqual(watch["qualified_count"], 8)
        self.assertEqual([r["symbol"] for r in watch["picks"]], ["0", "2", "4"])

    def test_p5_seal_intersection_keeps_all_prior_hits_even_without_today_p5(self):
        symbols = list("abcdefgh")
        quote = pd.DataFrame([{"date": DAY, "close": 20.82, "low": 20.82, "high": 20.82}])
        rows = {s: {"frame": quote} for s in symbols}
        boards = {"U": list("abcdefg") + ["a"], "Z": ["h"], "reconciliation": {"U": {"local_only": ["b"]}}}
        heights = {**dict.fromkeys("abcdef", 1), "g": 2}
        pools = {DAYS[0]: dict.fromkeys("abcdegh", HELD), DAYS[1]: {"a": FAILED}, DAY: {"f": HELD}}
        watch = W.limit_up_matches(DAY, DAYS[:-1], pools, boards, heights, rows)
        self.assertEqual(watch["focus_count"], 4)
        self.assertEqual(watch["closing_limit_up_count"], 7)
        by = {r["symbol"]: r for r in watch["rows"]}
        self.assertEqual(set(by), set("abcdefg"))
        self.assertEqual(by["a"]["evidence"][-1]["state"], "RECLAIM_FAILED")
        self.assertTrue(by["a"]["one_price"] and by["b"]["board_disputed"])
        self.assertEqual(by["b"]["priority"], "BOARD_DATA_DISPUTED")
        self.assertEqual(by["f"]["priority"], "SAME_DAY_P5_ONLY")
        self.assertEqual(by["g"]["priority"], "PRIOR_P5_OTHER_BOARD")
        self.assertFalse(watch["orders_enabled"])
        self.assertIn(DAYS[2], watch["unverified_dates"])
        self.assertEqual({r["symbol"] for r in W.limit_up_followup(watch, NEXT, {})}, set("acde"))

    def test_next_day_feedback_does_not_invent_fills_or_roll_over(self):
        watch, _ = self.pair()
        previous = {"funnel": {"first_board_watch": watch}}
        facts = {"as_of": NEXT, "previous_day": DAY}
        bar = {"date": NEXT, "valid": True, "open": 22.90, "high": 22.90, "low": 22.90,
               "close": 22.90, "preclose": 20.82}
        for changes, state in [({}, "ONE_PRICE_NO_CONFIRMED_FILL"), ({"low": 18.0}, "STRUCTURE_LOST"),
                               ({"low": 21.0}, "NO_PULLBACK_WINDOW"), ({"low": 20.0, "close": 20.5}, "WEAK_CLOSE"),
                               ({"preclose": 19.0}, "DATA_WAIT")]:
            with self.subTest(state=state):
                result, = W.followup(previous, facts, {"a": {"frame": pd.DataFrame([{**bar, **changes}])}})
                self.assertEqual(result["state"], state)
                self.assertTrue(result["expired"])
                self.assertFalse(result["orders_enabled"])
        self.assertEqual(W.followup(previous, {**facts, "as_of": "2026-11-19"}, {}), [])
