from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from mainwave_engine import phase_events as PE
from research.mainwave_phase_event_study import _forward_extreme


class PhaseEventTests(unittest.TestCase):
    @staticmethod
    def price_frame(size: int = 180) -> pd.DataFrame:
        index = np.arange(size)
        close = 10 + index * 0.01 + np.sin(index / 7) * 0.2
        return pd.DataFrame({
            "date": pd.bdate_range("2025-01-01", periods=size),
            "open": close - 0.03,
            "high": close + 0.15,
            "low": close - 0.15,
            "close": close,
            "amount": 1e8 * (1 + 0.05 * np.cos(index / 5)),
            "pct_chg": pd.Series(close).pct_change().fillna(0).to_numpy() * 100,
        })

    def test_classification_is_prefix_invariant(self) -> None:
        raw = self.price_frame()
        prefix = PE.classify_frame(raw.iloc[:150])
        full = PE.classify_frame(raw).iloc[:150]
        columns = [*PE.EVENT_COLUMNS, "broad_phase", "effort_result"]
        pd.testing.assert_frame_equal(
            prefix[columns].reset_index(drop=True),
            full[columns].reset_index(drop=True),
        )

    def test_latest_snapshot_ignores_rejected_newer_event(self) -> None:
        classified = self.price_frame(80)
        for event in PE.EVENT_COLUMNS:
            classified[event] = False
        classified["broad_phase"] = "MARKUP"
        classified["effort_result"] = "DEMAND_EXPANSION"
        classified.loc[70, "LPS"] = True
        classified.loc[75, "SPRING"] = True
        validation = {
            "events": {"LPS": {"status": "PASS"}, "SPRING": {"status": "REJECT"}},
            "phases": {"MARKUP": {"status": "PASS"}},
            "efforts": {"DEMAND_EXPANSION": {"status": "PASS"}},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.price_frame(80).to_csv(root / "sh_600000_daily.csv", index=False)
            with patch.object(PE, "classify_frame", return_value=classified), \
                    patch.object(PE, "_validation", return_value=validation):
                result = PE.latest_snapshot("sh_600000", "2025-04-30", root)
        self.assertEqual(result["latest_event"], "LPS")
        self.assertEqual(result["latest_event_raw"], "SPRING")
        self.assertEqual(result["observation_action"], "LPS_CONFIRMATION_WATCH")

    def test_forward_window_excludes_anchor_day(self) -> None:
        values = pd.Series(range(50), dtype=float)
        maximum = _forward_extreme(values, "max")
        self.assertEqual(maximum.iloc[0], 40.0)
        self.assertEqual(maximum.iloc[9], 49.0)
        self.assertTrue(np.isnan(maximum.iloc[10]))


if __name__ == "__main__":
    unittest.main()
