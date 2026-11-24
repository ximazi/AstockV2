import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from mainwave_engine import s3_scan as S3


def s3_frame(dry=0.5, burst=2.5, jump=1.03):
    """An uptrend, a 40-session quiet base on low volume, then a breakout day on heavy volume."""
    closes = list(100 * np.cumprod(np.full(130, 1.008)))
    top = closes[-1]
    closes += [top * (0.97 + 0.02 * (k % 2)) for k in range(40)]
    closes.append(top * jump)
    c = np.array(closes)
    vol = np.r_[np.full(130, 1000.0), np.full(40, 1000.0 * dry), [1000.0 * dry * burst]]
    return pd.DataFrame({"open": c / 1.005, "high": c * 1.005, "low": c * 0.995, "close": c, "volume": vol})


class SignalTests(unittest.TestCase):
    def test_quiet_base_then_heavy_breakout_is_s3(self):
        s = S3.signal(s3_frame())
        self.assertIsNotNone(s)
        self.assertGreaterEqual(s["base_len"], 7)
        self.assertLessEqual(s["base_pre"], 0.8)
        self.assertGreaterEqual(s["sig_base"], 2)

    def test_no_volume_dry_up_or_no_burst_is_not_s3(self):
        self.assertIsNone(S3.signal(s3_frame(dry=1.0)))
        self.assertIsNone(S3.signal(s3_frame(burst=1.5)))

    def test_close_inside_the_base_is_not_a_breakout(self):
        self.assertIsNone(S3.signal(s3_frame(jump=0.99)))

    def test_structure_classes_use_only_past_bars(self):
        f = s3_frame()
        cls, facts = S3.classify(f, 40)
        self.assertIn(cls, S3.CLASS_NOTE)
        self.assertEqual(cls, "趋势内整理释放", facts)


class ApiTests(unittest.TestCase):
    def test_scenario_endpoints(self):
        import mainwave_visual_app as web
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "20260924"
            folder.mkdir()
            (folder / "s3_observations.json").write_text(json.dumps({"as_of": "2026-09-24", "observations": [{"symbol": "sh_600661"}]}),
                                                         encoding="utf-8")
            with patch.object(web, "OUT_ROOT", Path(tmp)), web.app.test_client() as client:
                self.assertEqual(client.get("/api/scenario/s3?date=2026-09-24").json["observations"][0]["symbol"], "sh_600661")
                self.assertTrue(client.get("/api/scenario/s4?date=2026-09-24").json["missing"])
                self.assertEqual(client.get("/api/scenario/x?date=2026-09-24").status_code, 404)
                self.assertIn(client.get("/scenario-cases/s9").status_code, (404,))


if __name__ == "__main__":
    unittest.main()
