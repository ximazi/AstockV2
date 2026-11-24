from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from mainwave_engine import potential_pool as P, tail_selection as S, intraday_review as R, intraday_quotes as Q
from mainwave_engine.test_intraday_decisions import DecisionTests


def seed():
    return {"status": "READY", "as_of": "2026-09-17", "previous_day": "2026-09-16",
            "close": 10.2, "ema10": 10., "ema20": 9.8, "atr20": .4,
            "volume20": 100., "high20": 10.5, "low20": 9., "trend_up": True,
            "ema20_rising": True, "pressure": False, "ignition_date": "2026-09-12", "ignition_floor": 9.7,
            "liquid": True, "evidence": [{"key": "冲击保留", "date": "2026-09-12"}],
            "risk_flags": [], "score": 2, "metrics": {}, "entry": {}}


def pullback():
    return {"open": 10.16, "close": 10.2, "low": 9.96, "high": 10.25, "volume": 110., "pre_close": 10.2}


class PoolTests(unittest.TestCase):
    def test_partial_evidence_enters_without_market_gate_or_buy(self):
        p = P.build({"sh_600000": seed()}, "2026-09-17", at="2026-09-17 21:00")
        r = p["records"][0]
        self.assertEqual(r["state"], "POTENTIAL")
        self.assertEqual(len(r["evidence"]), 1)
        self.assertFalse(r["orders_enabled"])
        self.assertEqual(r["entry"], {})

    def test_ignition_bonus_requires_prior_session_membership_not_today_admission(self):
        s = seed()
        s["entry"] = {"ignition": True}
        old = P.build({"sh_600000": s}, "2026-09-17", at="2026-09-17 21:00")
        self.assertEqual(old["records"][0]["ignition_bonus"], 0)
        s.update(as_of="2026-09-18", previous_day="2026-09-17")
        new = P.build({"sh_600000": s}, "2026-09-18", old, at="2026-09-18 21:00")
        self.assertEqual(new["records"][0]["ignition_bonus"], 2)
        self.assertEqual(new["records"][0]["first_seen"], "2026-09-17")
        old["as_of"] = "2026-09-16"
        self.assertEqual(P.build({"sh_600000": s}, "2026-09-18", old)["records"][0]["ignition_bonus"], 0)

    def test_risk_missing_and_expired_evidence_are_distinct(self):
        s = seed()
        old = P.build({"sh_600000": s}, "2026-09-17")
        s.update(as_of="2026-09-18", previous_day="2026-09-17", risk_flags=["供给压力"])
        self.assertEqual(P.build({"sh_600000": s}, "2026-09-18", old)["records"][0]["state"], "RISK_WATCH")
        self.assertEqual(P.build({}, "2026-09-18", old)["records"][0]["state"], "DATA_WAIT")
        s.update(evidence=[], risk_flags=[])
        self.assertEqual(P.build({"sh_600000": s}, "2026-09-18", old)["records"][0]["state"], "EXITED")

    def test_ema10_wick_recovery_and_prior_ignition_combine(self):
        r = P.entry_checks(seed(), pullback())
        self.assertTrue(r["pullback"])
        self.assertTrue(r["combined"])
        self.assertFalse(r["ignition"])
        self.assertGreater(r["lower_wick_ratio"], .25)

    def test_deep_fall_downtrend_no_wick_or_failed_recovery_are_not_buys(self):
        for changes in ({"low": 9.2}, {"low": 10.15}, {"close": 9.98}):
            self.assertFalse(P.entry_checks(seed(), {**pullback(), **changes})["pullback"])
        self.assertFalse(P.entry_checks({**seed(), "trend_up": False}, pullback())["pullback"])
        self.assertFalse(P.entry_checks({**seed(), "ignition_floor": 10.}, pullback())["combined"])
        self.assertFalse(P.entry_checks(seed(), {**pullback(), "volume": None})["pullback"])

    def test_adjustment_invariance_and_risk_not_offset_by_signals(self):
        a = P.entry_checks(seed(), pullback())
        b = P.entry_checks(seed(), {k: v/2 if k != "volume" else v for k, v in pullback().items()})
        self.assertEqual(a["pullback"], b["pullback"])
        self.assertAlmostEqual(a["ema10"] / 2, b["ema10"])
        self.assertAlmostEqual(a["lower_wick_ratio"], b["lower_wick_ratio"])
        self.assertFalse(P.entry_checks({**seed(), "pressure": True}, pullback())["pullback"])

    def test_ignition_is_strong_breakout_not_any_large_candle(self):
        bar = {"open": 10.3, "high": 10.85, "low": 10.2, "close": 10.8, "pre_close": 10.2, "volume": 180.}
        self.assertTrue(P.entry_checks(seed(), bar)["ignition"])
        for changes in ({"high": 11.9}, {"close": 10.4}, {"volume": 80.}, {"close": 11.8, "high": 11.9}):
            self.assertFalse(P.entry_checks(seed(), {**bar, **changes})["ignition"])

    def history(self):
        close = pd.Series([10 + i * .025 for i in range(100)])
        return pd.DataFrame({"date": pd.bdate_range(end="2026-09-17", periods=100),
                             "close": close, "open": close-.02, "high": close+.06, "low": close-.04,
                             "preclose": close.shift().fillna(close), "pct_chg": close.pct_change().fillna(0)*100,
                             "volume": 5e6, "amount": close*5e6, "turnover": 3.})

    def test_feature_seed_has_no_future_and_uses_adjusted_price(self):
        h = self.history()
        a = P.feature_seed(h, "sh_600000", "2026-09-17")
        future = h.iloc[-1:].assign(date=pd.Timestamp("2026-09-18"), close=999., pct_chg=100.)
        self.assertEqual(a, P.feature_seed(pd.concat([h, future]), "sh_600000", "2026-09-17"))
        split = h.copy()
        for k in ("open", "high", "low", "close", "preclose"):
            split.loc[split.index >= 90, k] /= 2
        b = P.feature_seed(split, "sh_600000", "2026-09-17")
        self.assertEqual(a["evidence"], b["evidence"])
        self.assertAlmostEqual(a["ema10"]/2, b["ema10"])
        self.assertAlmostEqual(a["metrics"]["position60"], b["metrics"]["position60"])
        self.assertEqual(P.feature_seed(h.iloc[:-1], "sh_600000", "2026-09-17")["status"], "HISTORY_WAIT")

    def test_zero_or_invalid_data_never_creates_a_good_seed(self):
        h = self.history()
        for col, value in (("low", 0), ("volume", float("inf")), ("pct_chg", -101)):
            bad = h.copy()
            bad.loc[99, col] = value
            self.assertEqual(P.feature_seed(bad, "sh_600000", "2026-09-17")["status"], "HISTORY_WAIT")
        self.assertEqual(P.feature_seed(h.assign(volume=0.), "sh_600000", "2026-09-17")["status"], "HISTORY_WAIT")

    def test_reference_and_pool_reuse_same_history_read(self):
        R._reference_from_source.cache_clear()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "daily/20260917"
            folder.mkdir(parents=True)
            (folder / "_SUCCESS").touch()
            R._write(folder / "mainwave_run_status.json", {"as_of": "2026-09-17", "status": "SUCCESS", "finished_at": "2026-09-17T20:00:00"})
            with patch.object(R, "read_history", return_value=self.history()) as read, patch.object(R.TS, "load_mapping_as_of", return_value=({}, {})):
                R.prepare_reference("2026-09-17", ["sh_600000"], root, root / "daily")
            self.assertEqual(read.call_count, 1)
            first = P.load(root, "2026-09-17")
            self.assertEqual(first["summary"]["universe"], 1)
            self.assertFalse(P.latest(root, "2026-09-17", "2026-09-17 14:30"))

    def test_source_change_refreshes_missing_and_valid_cached_features(self):
        R._reference_from_source.cache_clear()
        with patch.object(R, "source_fingerprint", return_value=[1, 100]), patch.object(R, "read_history", return_value=pd.DataFrame()) as read:
            missing = R.load_reference("sh_600000", "2026-09-17")
            self.assertEqual(R.load_reference("sh_600000", "2026-09-17", missing), missing)
            self.assertEqual(read.call_count, 1)
        with patch.object(R, "source_fingerprint", return_value=[2, 200]), patch.object(R, "read_history", return_value=self.history()) as read:
            ready = R.load_reference("sh_600000", "2026-09-17", missing)
            self.assertEqual(ready["status"], "READY")
            self.assertEqual(read.call_count, 1)
        with patch.object(R, "source_fingerprint", side_effect=[[3, 300], [4, 400]]), patch.object(R, "read_history", return_value=self.history()):
            self.assertEqual(R.load_reference("sh_600000", "2026-09-17", ready), {"status": "HISTORY_WAIT"})

    def test_same_day_preparation_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reference").mkdir()
            with R.FileLock(root / "reference/prepare_20260917.lock"), patch.object(R, "_prepare_reference") as prepare:
                with self.assertRaisesRegex(ValueError, "正在构建"):
                    R.prepare_reference("2026-09-17", [], root, root / "daily")
                prepare.assert_not_called()


class TailTests(unittest.TestCase):
    def setUp(self):
        f = DecisionTests()
        f.setUp()
        f.complete_minutes()
        self.report = f.report()
        self.row = next(r for r in self.report["records"] if r["symbol"] == "sh_600000")
        self.row["preparation"] = seed()
        self.row["quote"].update(pullback())
        self.row["minute_context"]["reclaim_confirmed"] = True
        self.pool = P.build({"sh_600000": seed()}, "2026-09-17", at="2026-09-17 21:00")

    def selected(self):
        return S.build(self.report, self.pool, "2026-09-17")

    def test_new_strategy_not_dependent_on_old_route_permission(self):
        self.row.update(route_ids="", prior_advice="", status="WATCH")
        r = self.selected()["records"][0]
        self.assertEqual(r["status"], "SETUP_REVIEW")
        self.assertIn("点火后回踩", r["strategies"])
        self.assertEqual(r["route_ids"], "")
        self.assertFalse(r["orders_enabled"])

    def test_data_market_and_sequence_gates(self):
        self.row["minute_context"]["reclaim_confirmed"] = False
        self.assertFalse(self.selected()["paper_review_symbols"])
        self.row["minute_context"]["reclaim_confirmed"] = True
        self.report["market"]["new_review_allowed"] = False
        self.assertEqual(self.selected()["records"][0]["status"], "MARKET_WAIT")
        self.row["quote"]["quality"] = "PRICE_ONLY"
        self.assertEqual(self.selected()["records"][0]["status"], "DATA_WAIT")

    def test_future_or_wrong_day_pool_never_earns_bonus(self):
        self.row["quote"].update(open=10.3, high=10.85, low=10.2, close=10.8, volume=180.)
        self.assertEqual(self.selected()["records"][0]["pool_bonus"], 2)
        self.pool["known_at"] = "2026-09-18 16:00"
        self.assertEqual(self.selected()["records"][0]["pool_bonus"], 0)
        self.pool["known_at"] = "2026-09-17 21:00"
        self.pool["as_of"] = "2026-09-18"
        self.assertEqual(self.selected()["records"][0]["pool_bonus"], 0)

    def test_board_and_leader_roles_are_not_orders_or_proof_of_leadership(self):
        self.report["tail_observation"] = {
            "leaders": [{"symbol": self.row["symbol"], "state": "DAILY_LEADER"}],
            "boards": [{"symbol": self.row["symbol"], "state": "SECOND_PRICE", "group": "TODAY_PROMOTION"}]}
        r = self.selected()["records"][0]
        self.assertIn("1进2晋级观察", r["strategies"])
        self.assertIn("龙头候选回踩", r["strategies"])
        self.assertFalse(r["paper_review_eligible"])
        self.assertFalse(r["leadership_confirmed"])
        self.row["quote"].update(open=10.3, high=10.85, low=10.2, close=10.8, volume=180.)
        self.assertIn("日线领头候选", self.selected()["records"][0]["strategies"])
        self.assertNotIn("龙头候选回踩", self.selected()["records"][0]["strategies"])

    def test_selection_is_readonly_and_expiry_revokes_review(self):
        before = deepcopy(self.report)
        selection = self.selected()
        self.assertEqual(before, self.report)
        self.assertTrue(selection["paper_review_symbols"])
        S.expire(selection)
        self.assertEqual(selection["mode"], "STALE")
        self.assertFalse(selection["paper_review_symbols"])
        self.assertFalse(selection["records"][0]["paper_review_eligible"])

    def test_unscanned_first_board_is_not_lost(self):
        self.report["tail_observation"] = {"boards": [{"symbol": "sz_300001", "state": "DATA_WAIT", "group": "TODAY_PROMOTION"}]}
        row = next(r for r in self.selected()["records"] if r["symbol"] == "sz_300001")
        self.assertEqual(row["status"], "DATA_WAIT")
        self.assertIn("1进2晋级观察", row["strategies"])

    def test_board_missing_limits_never_bypasses_new_strategy_gate(self):
        self.report["tail_observation"] = {"boards": [{"symbol": self.row["symbol"], "state": "DATA_WAIT", "group": "TODAY_PROMOTION"}]}
        self.assertEqual(self.selected()["records"][0]["status"], "DATA_WAIT")
        self.assertFalse(self.selected()["paper_review_symbols"])

    def test_existing_holdings_are_not_new_buys_and_exit_risk_has_priority(self):
        for state in ("HOLD_OBSERVATION", "SELL_REVIEW", "REDUCE_REVIEW"):
            self.report["holdings"] = [{"symbol": self.row["symbol"], "status": state, "add_allowed": False}]
            self.assertFalse(self.selected()["paper_review_symbols"])
            if state != "HOLD_OBSERVATION":
                self.assertEqual(self.selected()["records"][0]["status"], "RISK_WATCH")

    def test_old_route_missing_data_cannot_hide_known_risk(self):
        f = DecisionTests()
        f.setUp()
        f.complete_minutes()
        for tier in ("DATA_WAIT", "HISTORICAL_ONLY"):
            f.baseline["records"][0].update(value_tier=tier, invalid=10.5, structural_advice="RISK_WATCH")
            row = next(r for r in f.report()["records"] if r["symbol"] == self.row["symbol"])
            self.assertEqual(row["status"], "RISK_WATCH")
            self.assertIn("当前价格跌破已知结构失效边界", row["risk_flags"])
            self.row["risk_flags"] = row["risk_flags"]
            self.row["status"] = "DATA_WAIT"
            self.assertEqual(self.selected()["records"][0]["status"], "RISK_WATCH")
            self.assertFalse(self.selected()["paper_review_symbols"])

    def test_minute_reclaim_requires_completed_dip_then_recovery(self):
        f = DecisionTests()
        f.setUp()
        f.complete_minutes()
        minute = pd.DataFrame(f.bundle["raw"][2]["records"])
        minute.loc[:, ["open", "close"]] = 10.
        minute["high"], minute["low"] = 10.01, 9.99
        minute["amount"] = minute.vol * minute.close
        self.assertFalse(Q.minute_context(minute, "600000.SH", f.at)["reclaim_confirmed"])
        path = [10.2, 10.18, 10.1, 9.95, 9.9, 9.96, 10., 10.03, 10.04, 10.05, 10.06, 10.08, 10.1, 10.12, 10.2, 10.2]
        for i, close in zip(minute.index[-16:], path):
            minute.loc[i, ["open", "close", "high", "low", "amount"]] = [close, close, close+.01, close-.01, close*100]
        self.assertTrue(Q.minute_context(minute, "600000.SH", f.at)["reclaim_confirmed"])
        minute.loc[minute.index[-2], "close"] = 10.1
        minute.loc[minute.index[-2], "low"] = 10.09
        self.assertFalse(Q.minute_context(minute, "600000.SH", f.at)["reclaim_confirmed"])


if __name__ == "__main__":
    unittest.main()
