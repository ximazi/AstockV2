import unittest

import pandas as pd

from mainwave_engine import breakout_events as BE
from mainwave_engine.test_price_cycle_setups import fixture


class BreakoutEventTests(unittest.TestCase):
    def test_parent_labels_cannot_hide_or_invalidate_price_evidence(self):
        frame, timeline = fixture()
        normal = BE.annotate(frame, timeline)
        timeline["cycle_stage"] = "FAILED"
        timeline["cycle_id"] = [f"arbitrary:{i}" for i in range(len(timeline))]
        failed = BE.annotate(frame, timeline)
        fields = [c for c in normal if c.startswith("breakout_")]
        pd.testing.assert_frame_equal(normal[fields], failed[fields])
        self.assertTrue(failed.breakout_event.eq("ACCEPTED").any())

    def test_acceptance_is_not_backdated(self):
        frame, timeline = fixture()
        full = BE.annotate(frame, timeline)
        i = full.index[full.breakout_event.eq("ATTEMPT")][0]
        prefix = BE.annotate(frame.iloc[:i + 1], timeline.iloc[:i + 1])
        self.assertEqual(prefix.iloc[-1].breakout_accepted_at, "")
        pd.testing.assert_frame_equal(prefix, full.iloc[:i + 1])


if __name__ == "__main__":
    unittest.main()
