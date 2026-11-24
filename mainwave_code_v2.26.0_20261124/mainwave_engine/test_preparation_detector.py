"""Preparation detectors are causal, gated by the -1..2 ATR band, and read only what the frame already holds."""
import unittest

import pandas as pd

from . import preparation_detector as PD


def frame(n=60, reclaim_at=None, bias=0.5, eligible=True, amount_ratio=1.0, contract=0.8):
    """A synthetic route frame: price drifts below its 20-day mean, then reclaims it at `reclaim_at`."""
    idx = []
    level = 100.0
    for i in range(n):
        if reclaim_at is not None and i >= reclaim_at:
            level += 1.5
        elif i >= 20:
            level -= 0.3
        idx.append(level)
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    return pd.DataFrame({"date": dates, "return_index": idx, "eligible": eligible, "bias_atr": bias, "turnover_contract": contract,
                         "range_contract": contract, "retained": True, "prepared": False, "amount_ratio": amount_ratio})


class DetectorTests(unittest.TestCase):
    def test_reclaim_fires_only_after_a_long_stay_below_and_within_band(self):
        f = frame(reclaim_at=45)
        ann = PD.annotate(f)
        self.assertFalse(ann.P4_reclaim.iloc[:40].any())
        self.assertTrue(ann.P4_reclaim.iloc[45:55].any())
        extended = frame(reclaim_at=45, bias=2.5)
        self.assertFalse(PD.annotate(extended).P4_reclaim.any())
        ineligible = frame(reclaim_at=45, eligible=False)
        self.assertFalse(PD.annotate(ineligible).P4_reclaim.any())
        climax = frame(reclaim_at=45, amount_ratio=3.0)
        self.assertFalse(PD.annotate(climax).P4_reclaim.any())

    def test_repair_needs_turning_mean_and_calm_activity(self):
        f = frame(reclaim_at=40)
        ann = PD.annotate(f)
        self.assertTrue(ann.P1_repair.iloc[50:].any())
        noisy = frame(reclaim_at=40, contract=1.2)
        self.assertFalse(PD.annotate(noisy).P1_repair.any())

    def test_relative_strength_variant_uses_benchmark_by_date(self):
        f = frame(reclaim_at=40)
        strong_bench = pd.Series(50.0, index=f.date.dt.strftime("%Y-%m-%d").values)  # market up 50% in 20 days: stock never leads
        weak_bench = pd.Series(-50.0, index=f.date.dt.strftime("%Y-%m-%d").values)
        self.assertFalse(PD.annotate(f, strong_bench).P2_repair_rs.any())
        self.assertTrue(PD.annotate(f, weak_bench).P2_repair_rs.iloc[50:].any())
        self.assertFalse(PD.annotate(f).P2_repair_rs.any())  # no benchmark → unknown, never true

    def test_union_and_latest(self):
        f = frame(reclaim_at=45)
        ann = PD.annotate(f)
        self.assertTrue((ann.P5_union == (ann.P2_repair_rs | ann.P4_reclaim)).all())
        last = PD.latest(f)
        self.assertEqual(set(PD.VARIANTS) <= set(last), True)
        self.assertEqual(PD.latest(pd.DataFrame())["P4_reclaim"], False)
        self.assertEqual(last["version"], PD.VERSION)

    def test_causality(self):
        f = frame(reclaim_at=45)
        full = PD.annotate(f)
        prefix = PD.annotate(f.iloc[:50].copy())
        for v in PD.VARIANTS:
            self.assertEqual(full[v].iloc[:50].tolist(), prefix[v].tolist(), v)


if __name__ == "__main__":
    unittest.main(verbosity=2)
