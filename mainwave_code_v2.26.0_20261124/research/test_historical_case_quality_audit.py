import unittest

import pandas as pd

from research.historical_case_quality_audit import path_metrics


class PathMetricsTests(unittest.TestCase):
    def fixture(self):
        dates = pd.bdate_range("2026-01-05", periods=21)
        return pd.DataFrame({"date": dates, "open": 100., "high": 120.,
                             "low": 90., "close": 100., "return_index": 100.})

    def test_event_day_range_is_not_counted_as_future_opportunity(self):
        frame = self.fixture()
        frame.loc[0, ["high", "low"]] = [1000, 1]
        result = path_metrics(frame, "2026-01-05")
        self.assertAlmostEqual(result["mfe20_pct"], 20)
        self.assertAlmostEqual(result["mae20_pct"], -10)

    def test_horizon_excludes_event_day_and_requires_full_observation(self):
        frame = self.fixture()
        frame.loc[20, "return_index"] = 110
        self.assertAlmostEqual(path_metrics(frame, "2026-01-05")["r20_pct"], 10)
        self.assertIsNone(path_metrics(frame.iloc[:20], "2026-01-05")["r20_pct"])

    def test_split_does_not_look_like_a_price_crash(self):
        frame = self.fixture()
        before = path_metrics(frame, "2026-01-05")
        frame.loc[10:, ["open", "high", "low", "close"]] /= 2
        after = path_metrics(frame, "2026-01-05")
        for key in ("r20_pct", "mfe20_pct", "mae20_pct"):
            self.assertAlmostEqual(before[key], after[key])

    def test_missing_event_is_not_silently_shifted(self):
        with self.assertRaisesRegex(ValueError, "Missing"):
            path_metrics(self.fixture(), "2026-01-04")


if __name__ == "__main__":
    unittest.main()
