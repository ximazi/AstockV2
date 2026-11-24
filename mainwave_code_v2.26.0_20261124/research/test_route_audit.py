import unittest
import pandas as pd

from research.route_clock import classify_gap
from research.route_followup import arrays, simulate
from research import test_route_followup as fixtures
from research.route_diagnostic_audit import cohort_bridge
from research.route_followup import trade_stats
from research.route_preparation_audit import continuous_before_attempt


class ClockAndTraceTests(unittest.TestCase):
    def test_continuous_coverage_requires_same_alive_pre_attempt_track(self):
        row = {"preparation_valid": True, "preparation_origin": "PRE_ATTEMPT", "preparation_parent_id": "one",
               "breakout_id": "one", "preparation_first_seen": "2026-01-01", "breakout_attempt_at": "2026-01-05"}
        self.assertTrue(continuous_before_attempt(row))
        for change in ({"preparation_valid": False}, {"breakout_id": "two"},
                       {"preparation_origin": "POST_ACCEPTANCE"}, {"preparation_first_seen": "2026-01-05"}):
            self.assertFalse(continuous_before_attempt({**row, **change}))

    def test_open_or_unverifiable_trades_are_not_zero_profit_signals(self):
        f = pd.DataFrame({"status": ["CLOSED", "HOLDING_DATA_UNVERIFIED"], "net": [1., None],
                          "held": [3., None], "reason": ["TRAIL_STOP", ""], "partials": [0., None]})
        stats = trade_stats(f)
        self.assertEqual(stats["mean"], 1.)
        self.assertEqual(stats["unresolved"], 1)
        self.assertTrue(pd.isna(stats["closed_net_sum_per_signal"]))

    def test_bridge_separates_year_end_purge_from_policy_filter(self):
        events = pd.DataFrame({"symbol": ["a", "b", "c"], "date": ["2024-12-20", "2025-01-10", "2025-01-10"],
                               "accepted_bias2": [True, True, False], "r20": [30., 25., 40.],
                               "outcome20_at": ["2025-01-20", "2025-02-20", "2025-02-20"]})
        coverage = events.rename(columns={"outcome20_at": "outcome_at"}).assign(variant="strict_early", year=[2024, 2025, 2025])
        bridge, summary = cohort_bridge(events, coverage)
        self.assertEqual(summary["shared"], 1)
        self.assertEqual(summary["r1_only"], 1)
        self.assertEqual(summary["r2_only"], 1)
        self.assertEqual(bridge.event_key.nunique(), 3)

    def test_holiday_is_not_missing_session(self):
        calendar = ("2024-09-30", "2024-10-08", "2024-10-09")
        self.assertEqual(classify_gap(calendar[0], calendar[1], calendar)["status"], "NEXT_EXCHANGE_SESSION")
        self.assertEqual(classify_gap(calendar[0], calendar[2], calendar)["status"], "UNEXPLAINED_SESSION_GAP")
        self.assertEqual(classify_gap(calendar[0], calendar[2], calendar, [calendar[1]])["status"], "VERIFIED_SUSPENSION")
        self.assertEqual(classify_gap("2024-09-27", calendar[2], calendar)["status"], "CALENDAR_UNVERIFIED")

    def test_activation_applies_next_day_and_stop_never_falls(self):
        f = fixtures.FollowupTests().fixture()
        f["index_low"] = 99.
        f["low10"] = 101.
        f.loc[12, "index_high"] = 105.
        trace = []
        result = simulate(arrays(f), 10, "sh_600001", "activated20", trace=trace,
                          calendar=tuple(f.date.dt.strftime("%Y-%m-%d")))
        self.assertEqual(result["exit_date"], str(f.date.iloc[13].date()))
        self.assertEqual(trace[1]["stop_before"], 96.)
        self.assertEqual(trace[1]["stop_next"], 101.)
        self.assertEqual(trace[1]["fills"], [])
        self.assertTrue(all(r["stop_next"] >= r["stop_before"] for r in trace))

    def test_unexplained_holding_gap_retains_unresolved_position(self):
        f = fixtures.FollowupTests().fixture()
        f["index_low"] = 99.
        calendar = tuple(f.date.dt.strftime("%Y-%m-%d"))
        removed = calendar[12]
        f = f.drop(index=12).reset_index(drop=True)
        result = simulate(arrays(f), 10, "sh_600001", "hold20", calendar=calendar)
        self.assertEqual(result["status"], "HOLDING_DATA_UNVERIFIED")
        self.assertEqual(result["remaining"], 1.)
        self.assertNotIn("net", result)
        f.attrs["suspended_days"] = [removed]
        self.assertEqual(simulate(arrays(f), 10, "sh_600001", "hold20", calendar=calendar)["status"], "CLOSED")

    def test_no_sell_on_entry_day_and_cash_reconciles(self):
        f = fixtures.FollowupTests().fixture()
        f.loc[12, "index_open"] = 92.
        trace = []
        result = simulate(arrays(f), 10, "sh_600001", "stage36_20", trace=trace,
                          calendar=tuple(f.date.dt.strftime("%Y-%m-%d")))
        self.assertFalse(trace[0]["sellable"])
        self.assertEqual(trace[0]["fills"], [])
        self.assertTrue(trace[0]["pending_after"])
        self.assertAlmostEqual(result["gross"] - result["cost_pct"], result["net"])
