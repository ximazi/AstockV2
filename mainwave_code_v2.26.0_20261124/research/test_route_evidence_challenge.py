import unittest

import numpy as np
import pandas as pd

from research.route_evidence_challenge import snapshot


class SnapshotTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame({
            "date": pd.bdate_range("2026-01-01", periods=90),
            "open": 100., "high": 101., "low": 99., "close": 100.,
            "volume": 100., "amount": 1e8, "pct_chg": 0., "turnover": 2.,
        })

    def test_future_prices_do_not_change_snapshot(self):
        frame = self.fixture()
        date = str(frame.iloc[70].date.date())
        expected = snapshot(frame.iloc[:71], "sh_600001", date)
        frame.loc[71:, ["open", "high", "low", "close", "amount", "pct_chg"]] = 9999
        self.assertEqual(snapshot(frame, "sh_600001", date), expected)

    def test_bullish_bodies_can_coexist_with_falling_closes(self):
        frame = self.fixture()
        for i in range(66, 70):
            close = 99 - (i - 66) * 2
            frame.loc[i, ["open", "close", "high", "low"]] = [close - 1, close, close + 1, close - 2]
        result = snapshot(frame, "sh_600001", str(frame.iloc[70].date.date()))
        self.assertTrue(result["legacy_yang"])
        self.assertEqual(result["legacy_prior_bullish_bodies"], 4)
        self.assertTrue(np.all(np.diff(frame.close.iloc[66:70]) < 0))

    def test_event_limit_proxy_is_not_prior_limit(self):
        frame = self.fixture()
        frame.loc[70, ["open", "high", "low", "close", "pct_chg"]] = [108, 110, 107, 110, 10]
        result = snapshot(frame, "sh_600001", str(frame.iloc[70].date.date()))
        self.assertFalse(result["legacy_limit_proxy"])

    def test_missing_session_does_not_shift_event(self):
        frame = self.fixture()
        with self.assertRaisesRegex(ValueError, "Missing active"):
            snapshot(frame.drop(index=70), "sh_600001", str(frame.iloc[70].date.date()))


if __name__ == "__main__":
    unittest.main()
