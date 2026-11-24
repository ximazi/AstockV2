from copy import deepcopy
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mainwave_engine import intraday_review as R, intraday_price as P, intraday_quotes as Q
from mainwave_engine import test_intraday_decisions as fixtures


class PriceReviewTests(unittest.TestCase):
    def setUp(self):
        fixtures.DecisionTests.setUp(self)
        self.bundle["raw"] = [{"api": "rt_min", "records": [dict(r, time=str(self.at)) for r in self.rows]}]
        self.limits = [{"ts_code": Q.ts_code(s), "trade_date": "20260918", "pre_close": 10.,
                        "up_limit": 11., "down_limit": 9.} for s in self.symbols]
        self.sources = {"previous_day": "2026-09-17",
                        "limits": {"status": "AVAILABLE", "as_of": "2026-09-18", "retrieved_at": str(self.at), "records": self.limits},
                        "boards": {"status": "AVAILABLE", "as_of": "2026-09-17", "retrieved_at": str(self.at),
                                   "records": [{"ts_code": Q.ts_code(s), "trade_date": "20260917", "limit_times": 1} for s in self.symbols[:3]]}}
        self.bundle["tail_sources"] = self.sources
        self.positions, self.prior = [], {}
        for seed in self.histories.values():
            seed.update(ema10=10., preparation={"status": "READY", "as_of": "2026-09-17", "close": 10.,
                        "ema10": 10., "ema20": 10., "atr20": .5, "low20": 9., "high20": 10.2,
                        "ema20_rising": True, "liquid": True, "risk_flags": [], "evidence": ["retained"]})

    def report(self, scope="market"):
        return R.evaluate(self.bundle, self.symbols, "2026-09-17", self.baseline, self.positions, self.prior,
                          self.histories, {}, self.at, self.membership, scope)

    def test_prices_enable_observations_not_full_confirmation(self):
        r = self.report()
        p = r["price_review"]
        self.assertEqual(p["market"]["covered"], 12)
        self.assertEqual((p["market"]["up"], p["market"]["down"]), (6, 6))
        self.assertAlmostEqual(p["market"]["median_pct"], 1.5)
        theme = next(t for t in p["themes"]["records"] if t["name"] == "液冷")
        self.assertEqual(theme["state"], "PRICE_HOT")
        self.assertIsNone(theme["amount_yi"])
        self.assertIsNone(theme["above_vwap_ratio"])
        self.assertEqual(r["market"]["state"], "UNKNOWN")
        self.assertEqual(r["summary"]["full_snapshots"], 0)
        self.assertFalse(r["candidates"]["paper_review_symbols"])
        self.assertTrue(p["candidates"])
        for row in r["records"]:
            self.assertEqual(row["quote"]["quality"], "PRICE_ONLY")
            self.assertIsNone(row["quote"]["low"])
            self.assertIsNone(row["metrics"]["volume_vs_full_day5"])
            self.assertFalse(row["paper_review_eligible"])
        self.assertFalse(p["orders_enabled"])
        self.assertFalse(p["market"]["new_review_allowed"])

    def test_no_reference_never_falls_back_to_unadjusted_previous_close(self):
        for key, value in (("status", "UNAVAILABLE"), ("as_of", "2026-09-17"),
                           ("retrieved_at", "2026-09-18 14:36")):
            with self.subTest(key=key):
                source = deepcopy(self.sources)
                source["limits"][key] = value
                self.assertFalse(P.reference_prices(source, "2026-09-18", self.at))
        self.sources["limits"]["status"] = "UNAVAILABLE"
        p = self.report()["price_review"]
        self.assertEqual(p["market"]["covered"], 0)
        self.assertIsNone(p["market"]["median_pct"])
        self.assertFalse(p["candidates"])

    def test_invalid_or_duplicate_reference_is_excluded(self):
        self.limits.append(dict(self.limits[0]))
        self.limits[1]["trade_date"] = "20260917"
        self.limits[2]["pre_close"] = 0
        self.limits[3]["pre_close"] = float("nan")
        p = self.report()["price_review"]
        self.assertEqual(p["market"]["covered"], 8)
        self.assertEqual(p["market"]["status"], "SAMPLE_ONLY")
        self.assertIsNone(p["previous_boards"]["mean_pct"])

    def test_price_breadth_does_not_require_long_history(self):
        self.histories[self.symbols[0]] = {"status": "HISTORY_WAIT"}
        p = self.report()["price_review"]
        self.assertEqual(p["market"]["covered"], 12)
        self.assertNotIn(self.symbols[0], [r["symbol"] for r in p["candidates"]])

    def test_stale_and_future_prices_do_not_contribute(self):
        self.bundle["raw"][0]["records"][0]["time"] = "2026-09-18 14:30"
        self.bundle["raw"][0]["records"][1]["time"] = "2026-09-18 14:36"
        self.assertEqual(self.report()["price_review"]["market"]["covered"], 10)

    def test_pool_and_unknown_membership_cannot_claim_market_hot_theme(self):
        p = self.report("pool")["price_review"]
        self.assertEqual(p["market"]["status"], "SAMPLE_ONLY")
        self.assertEqual(p["themes"]["status"], "DATA_WAIT")
        self.membership["meta"]["status"] = "STALE_RICH"
        self.assertEqual(self.report()["price_review"]["themes"]["status"], "DATA_WAIT")

    def test_future_membership_and_incomplete_domain_cannot_claim_relative_heat(self):
        self.membership["meta"]["as_of"] = "2026-09-21"
        self.assertEqual(self.report()["price_review"]["themes"]["status"], "DATA_WAIT")
        self.membership["meta"]["as_of"] = "2026-09-17"
        self.bundle["raw"][0]["records"] = self.bundle["raw"][0]["records"][:6]
        self.assertEqual(self.report()["price_review"]["themes"]["status"], "DATA_WAIT")

    def test_future_or_incomplete_preparation_cannot_supply_candidates(self):
        for seed in self.histories.values():
            seed["preparation"]["as_of"] = "2026-09-18"
        self.assertFalse(self.report()["price_review"]["candidates"])
        for seed in self.histories.values():
            seed["preparation"]["as_of"] = "2026-09-17"
        for row in self.baseline["records"]:
            row["structural_advice"] = "HISTORICAL_ONLY"
        self.assertFalse(self.report()["price_review"]["candidates"])

    def test_ex_right_reference_revalues_price_ema_and_holding_boundaries(self):
        self.limits[0]["pre_close"] = 5.
        self.bundle["raw"][0]["records"][0].update(open=5., high=5.3, low=4.9, close=5.2)
        self.positions = [{"id": "p", "symbol": self.symbols[0], "entry_date": "2026-09-18"}]
        self.prior = {"p": {"status": "HOLD", "hard_floor": 9.5, "soft_floor": 10.2}}
        r = self.report()
        h = r["price_review"]["holdings"][0]
        self.assertAlmostEqual(h["pct_chg"], 4.)
        self.assertEqual(h["hard_floor"], 4.75)
        self.assertEqual(h["soft_floor"], 5.1)
        self.assertTrue(h["t1_locked"])
        self.assertEqual(r["holdings"][0]["status"], "DATA_WAIT")
        self.assertFalse(h["orders_enabled"])
        self.assertNotIn(self.symbols[0], [c["symbol"] for c in r["price_review"]["candidates"]])

    def test_holding_alerts_and_missing_prices_are_not_hidden(self):
        self.positions = [{"id": "p", "symbol": self.symbols[0], "entry_date": "2026-09-16"}]
        self.prior = {"p": {"status": "HOLD", "hard_floor": 10.45, "soft_floor": 10.5}}
        self.assertIn("硬底线", self.report()["price_review"]["holdings"][0]["reading"])
        self.prior["p"]["hard_floor"] = 10.3
        self.assertIn("收盘确认", self.report()["price_review"]["holdings"][0]["reading"])
        self.prior["p"]["status"] = "SELL"
        self.bundle["raw"][0]["records"] = self.bundle["raw"][0]["records"][1:]
        h = self.report()["price_review"]["holdings"][0]
        self.assertIsNone(h["price"])
        self.assertIsNone(h["hard_floor"])
        self.assertIn("原退出警报", h["reading"])

    def test_candidate_risk_pressure_chase_and_distance_veto(self):
        self.baseline["records"][0]["value_tier"] = "RISK_WATCH"
        self.histories[self.symbols[1]]["preparation"]["pressure"] = True
        self.baseline["records"][2]["review_upper"] = 10.3
        self.histories[self.symbols[3]]["preparation"]["atr20"] = .01
        self.baseline["records"][4]["invalid"] = 10.45
        self.histories[self.symbols[5]]["preparation"]["liquid"] = False
        self.assertFalse(self.report()["price_review"]["candidates"])

    def test_focus_is_capped_and_not_a_buy_permission(self):
        p = self.report()["price_review"]
        self.assertEqual(p["candidate_count"], 6)
        self.assertEqual(len(p["candidates"]), 2)
        self.assertTrue(all(not c["paper_review_eligible"] and not c["orders_enabled"] for c in p["candidates"]))

    def test_previous_cohort_uses_valid_denominator_and_not_seal_confirmation(self):
        self.bundle["raw"][0]["records"][0].update(high=11., close=11.)
        self.bundle["raw"][0]["records"] = [r for r in self.bundle["raw"][0]["records"] if r["ts_code"] != Q.ts_code(self.symbols[1])]
        c = self.report()["price_review"]["previous_boards"]
        self.assertEqual((c["covered"], c["total"], c["at_limit"]), (2, 3, 1))
        self.assertAlmostEqual(c["mean_pct"], 7.)
        self.assertEqual(c["positive_ratio"], 1.)
        self.sources["boards"]["status"] = "UNAVAILABLE"
        c = self.report()["price_review"]["previous_boards"]
        self.assertIsNone(c["total"])
        self.assertIsNone(c["mean_pct"])

    def test_expiry_keeps_history_but_marks_price_observations_stale(self):
        r = self.report()
        R.check_expiry(r, self.at)
        picks = deepcopy(r["price_review"]["candidates"])
        R.check_expiry(r, self.at + timedelta(minutes=3))
        self.assertEqual(r["price_review"]["mode"], "STALE")
        self.assertEqual(r["price_review"]["candidates"], picks)
        self.assertFalse(r["candidates"]["paper_review_symbols"])

    def test_price_report_and_page_render_without_mutating_archive(self):
        from mainwave_visual_app import app
        with tempfile.TemporaryDirectory() as tmp, patch.dict(app.config, INTRADAY_ROOT=Path(tmp)):
            report = self.report()
            R.check_expiry(report, self.at)
            path = R.publish(self.bundle, report, Path(tmp))
            self.assertIn("价格级观察", (path / "review.md").read_text(encoding="utf-8"))
            before = (path / "review.json").read_bytes()
            with patch.object(Q, "now_china", return_value=self.at.to_pydatetime()):
                response = app.test_client().get("/intraday?date=2026-09-18")
                self.assertEqual(response.status_code, 200)
                self.assertIn('id="price-review"', response.get_data(as_text=True))
                self.assertIn("价格强势", response.get_data(as_text=True))
            self.assertEqual(before, (path / "review.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
