import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from mainwave_engine import index_trend as IT


def index_frame(dates, closes):
    return pd.DataFrame({"ts_code": "000300.SH", "trade_date": dates, "close": closes})


class RegimeTests(unittest.TestCase):
    def write(self, tmp, dates, closes):
        path = Path(tmp) / IT.INDEX_FILE
        index_frame(dates, closes).to_csv(path, index=False)
        return Path(tmp)

    def rising(self):
        dates = pd.bdate_range("2024-01-01", periods=300).strftime("%Y-%m-%d").tolist()
        closes = [100.0 * (1.004 ** i) for i in range(300)]
        return dates, closes

    def falling(self):
        dates = pd.bdate_range("2024-01-01", periods=300).strftime("%Y-%m-%d").tolist()
        closes = [300.0 * (0.996 ** i) for i in range(300)]
        return dates, closes

    def test_a_rising_index_blocks_new_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            dates, closes = self.rising()
            self.write(tmp, dates, closes)
            out = IT.regime(dates[-1], Path(tmp))
            self.assertEqual(out["status"], "FRESH")
            self.assertEqual(out["regime"], "UP")
            self.assertEqual(out["gate"], "BLOCK")
            self.assertGreater(out["slope"], 0)

    def test_a_falling_index_allows_new_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            dates, closes = self.falling()
            self.write(tmp, dates, closes)
            out = IT.regime(dates[-1], Path(tmp))
            self.assertEqual(out["status"], "FRESH")
            self.assertEqual(out["regime"], "DOWN")
            self.assertEqual(out["gate"], "ALLOW")

    def test_only_data_up_to_the_day_is_used(self):
        """当天的斜率不能用之后的数据算出来。"""
        with tempfile.TemporaryDirectory() as tmp:
            dates, closes = self.rising()
            self.write(tmp, dates, closes)
            early = IT.regime(dates[-1], Path(tmp))
            self.assertEqual(early["regime"], "UP")
            # 在最后一天之后追加暴跌, 早期那天的斜率不能被改写
            path = Path(tmp) / IT.INDEX_FILE
            frame = index_frame(dates, closes)
            extra = pd.bdate_range(dates[-1], periods=40).strftime("%Y-%m-%d").tolist()[1:]
            frame = pd.concat([frame, index_frame(extra, [closes[-1] * (0.9 ** i) for i in range(len(extra))])],
                              ignore_index=True)
            frame.to_csv(path, index=False)
            again = IT.regime(dates[-1], Path(tmp))
            self.assertEqual(again["slope"], early["slope"])
            self.assertEqual(again["regime"], "UP")

    def test_stale_data_does_not_pass(self):
        """指数数据不是当天 -> 未知 -> 不开新仓, 不能按旧数据放行。"""
        with tempfile.TemporaryDirectory() as tmp:
            dates, closes = self.rising()
            self.write(tmp, dates, closes)
            out = IT.regime(dates[-1], Path(tmp), )
            self.assertEqual(out["status"], "FRESH")
            stale_day = pd.bdate_range(dates[-1], periods=5).strftime("%Y-%m-%d").tolist()[-1]
            out = IT.regime(stale_day, Path(tmp))
            self.assertEqual(out["status"], "STALE")
            self.assertEqual(out["gate"], "UNKNOWN")
            self.assertIn("不开 S4 新仓", out["text"])

    def test_missing_file_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = IT.regime("2026-11-19", Path(tmp))
            self.assertEqual(out["status"], "MISSING")
            self.assertEqual(out["gate"], "UNKNOWN")

    def test_short_history_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            dates, closes = self.rising()
            self.write(tmp, dates[:100], closes[:100])
            out = IT.regime(dates[99], Path(tmp))
            self.assertEqual(out["status"], "INSUFFICIENT_HISTORY")
            self.assertEqual(out["gate"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
