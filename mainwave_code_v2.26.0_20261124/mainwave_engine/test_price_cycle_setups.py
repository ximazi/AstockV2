from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from mainwave_engine import price_cycle_setups as P


def fixture():
    n = 110
    close = np.full(n, 100.0)
    close[40:56] = np.linspace(100, 120, 16)
    close[56:62] = [119, 118, 117, 117.6, 118.5, 120]
    close[62:] = 120
    frame = pd.DataFrame({
        "return_index": close, "close": close, "open": close,
        "high": close + 0.4, "low": close - 0.4, "atr20": 2.0,
        "turnover": 10.0, "close_location": 0.7,
    })
    frame.loc[56:60, "turnover"] = 3.0
    frame.loc[61, "turnover"] = 6.0
    dates = pd.bdate_range("2025-01-01", periods=n).strftime("%Y-%m-%d")
    timeline = pd.DataFrame({
        "symbol": "sh_600000", "date": dates,
        "cycle_id": ["" if i < 40 else "sh_600000:launch" for i in range(n)],
        "cycle_stage": ["WATCH" if i < 40 else "MARKUP" for i in range(n)],
        "cycle_stage_label": ["结构观察" if i < 40 else "上涨延续" for i in range(n)],
        "cycle_reason": "NO_NEW_EVIDENCE", "cycle_reason_label": "观察",
        "turnover_relative20": 1.0,
    })
    timeline.loc[40, "cycle_reason"] = "PRICE_BREAKOUT_OBSERVATION"
    return frame, timeline


class PriceCycleSetupTests(unittest.TestCase):
    def test_first_pullback_confirmation_is_not_repeated(self):
        frame, timeline = fixture()
        result = P.annotate(frame, timeline)
        formed = result[result.setup_status.eq("FORMING")].iloc[0]
        self.assertEqual(formed.setup_kind, "FIRST_PULLBACK")
        confirmed = result[result.setup_status.eq("CONFIRMED")]
        self.assertEqual(len(confirmed), 1)
        self.assertTrue(result.iloc[62].setup_status.startswith("HISTORY_"))
        self.assertTrue(result.setup_permission.eq("OBSERVATION_ONLY").all())
        pd.testing.assert_frame_equal(result[timeline.columns], timeline)

    def test_prefix_and_repeat_stability_including_no_setup_prefix(self):
        frame, timeline = fixture()
        full = P.annotate(frame, timeline)
        for end in (35, 58, 60, 63, 100):
            pd.testing.assert_frame_equal(P.annotate(frame.iloc[:end], timeline.iloc[:end]),
                                          full.iloc[:end], check_dtype=False)
        pd.testing.assert_frame_equal(P.annotate(frame, timeline), full)

    def test_unknown_start_never_claims_first(self):
        frame, timeline = fixture()
        timeline.loc[40, "cycle_reason"] = "TREND_ALREADY_VISIBLE"
        result = P.annotate(frame, timeline)
        self.assertEqual(result[result.setup_status.eq("FORMING")].iloc[0].setup_kind,
                         "PULLBACK_UNKNOWN")

    def test_parent_failure_invalidates_child(self):
        frame, timeline = fixture()
        timeline.loc[60:, "cycle_stage"] = "FAILED"
        result = P.annotate(frame, timeline)
        self.assertEqual(result.iloc[60].setup_status, "FAILED")
        self.assertEqual(result.iloc[61].setup_status, "HISTORY_FAILED")
        self.assertFalse(result.setup_status.eq("CONFIRMED").any())

    def test_new_child_after_failure_is_not_first_again(self):
        frame, timeline = fixture()
        frame.loc[60, ["return_index", "close", "open", "high", "low"]] = [115, 115, 115, 115.4, 114.6]
        frame.loc[61:76, "return_index"] = np.linspace(116, 135, 16)
        frame.loc[77:82, "return_index"] = [134, 133, 132, 132.6, 133.5, 135]
        frame.loc[61:, "close"] = frame.loc[61:, "return_index"]
        frame.loc[61:, "high"] = frame.loc[61:, "close"] + 0.4
        frame.loc[61:, "low"] = frame.loc[61:, "close"] - 0.4
        result = P.annotate(frame, timeline)
        setups = P.setup_history(result)
        self.assertEqual(setups.iloc[0].setup_outcome, "FAILED")
        self.assertEqual(setups.iloc[1].setup_kind, "LATER_PULLBACK")

    def test_missing_turnover_never_confirms_demand(self):
        frame, timeline = fixture()
        frame["turnover"] = np.nan
        result = P.annotate(frame, timeline)
        self.assertFalse(result.setup_status.eq("CONFIRMED").any())
        self.assertTrue(result.speed_evidence.eq("MISSING").all())

    def test_platform_is_distinct_and_boundaries_frozen(self):
        frame, timeline = fixture()
        frame.loc[56:63, "return_index"] = [120, 119.8, 120, 119.9, 120, 119.9, 120, 120]
        frame.loc[64:, "return_index"] = 123
        for key in ("close", "open"):
            frame[key] = frame.return_index
        frame["high"], frame["low"] = frame.close + 0.2, frame.close - 0.2
        frame.loc[56:63, "turnover"] = 3.0
        frame.loc[64, "turnover"] = 6.0
        result = P.annotate(frame, timeline)
        formed = result[result.setup_status.eq("FORMING")]
        self.assertEqual(formed.iloc[0].setup_kind, "BASE_BREAK")
        self.assertEqual(formed.setup_trigger_index.nunique(), 1)
        self.assertEqual(result.iloc[64].setup_status, "CONFIRMED")

    def test_speed_distinguishes_equal_total_returns(self):
        a, _ = fixture()
        b = a.copy()
        a["return_index"] = np.r_[np.full(95, 100), np.arange(101, 106),
                                    np.arange(107, 116, 2), np.arange(119, 136, 4)]
        b["return_index"] = np.r_[np.full(95, 100), np.arange(104, 121, 4),
                                    np.arange(122, 131, 2), np.arange(131, 136)]
        self.assertEqual(a.iloc[-1].return_index, b.iloc[-1].return_index)
        self.assertGreater(P.speed_frame(a).iloc[-1].speed_change, 0)
        self.assertLess(P.speed_frame(b).iloc[-1].speed_change, 0)

    def test_segments_partition_timeline_without_overlap(self):
        _, timeline = fixture()
        spans = P.segments(timeline)
        self.assertEqual(sum(s["sessions"] for s in spans), len(timeline))
        self.assertLess(spans[0]["end"], spans[1]["start"])


if __name__ == "__main__":
    unittest.main()
