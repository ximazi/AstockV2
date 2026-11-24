"""The index updater waits for indices published later than others (2026-10-15: 000300 / 399330 / 399673)."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("index_update", ROOT / "update_a_share_major_indices.py")
U = importlib.util.module_from_spec(spec)
spec.loader.exec_module(U)


def write(folder, stem, last):
    days = pd.bdate_range(end=last, periods=3).strftime("%Y-%m-%d")
    pd.DataFrame({"ts_code": stem, "trade_date": days, "close": 1.0}).to_csv(Path(folder) / f"{stem}_index.csv", index=False)
    return Path(folder) / f"{stem}_index.csv"


class LaggingTests(unittest.TestCase):
    def test_only_indices_behind_a_published_trading_day_are_waited_for(self):
        with tempfile.TemporaryDirectory() as tmp:
            done = write(tmp, "000852_SH", "2026-10-15")
            late = write(tmp, "000300_SH", "2026-10-14")
            self.assertEqual(U.lagging([done, late], "20261015"), [late])
            self.assertEqual(U.lagging([done, late], "20261016"), [], "无人有该日数据：非交易日或补跑，不等待")

    def test_main_retries_until_published_and_reports_stale_otherwise(self):
        with tempfile.TemporaryDirectory() as tmp:
            done = write(tmp, "000852_SH", "2026-10-15")
            late = write(tmp, "000300_SH", "2026-10-14")
            calls = []

            def fake_update(pro, path, end_date, throttle):
                calls.append(path.stem.replace("_index", ""))
                if calls[-1] == "000300_SH" and calls.count("000300_SH") >= 2:
                    write(tmp, "000300_SH", "2026-10-15")
                    return "000300.SH", "updated", 1
                return path.stem, "no_new_rows", 0

            argv = ["x", "--end-date", "20261015", "--retry-wait", "0"]
            with patch.object(U, "DATA_DIR", Path(tmp)), patch.object(U, "update_one", side_effect=fake_update), \
                    patch.object(U.ts, "pro_api"), patch.object(U, "load_dotenv"), patch.dict("os.environ", {"TUSHARE_TOKEN": "t"}), \
                    patch("sys.argv", argv):
                self.assertEqual(U.main(), 0)
            self.assertEqual(calls.count("000300_SH"), 2, "第一轮没有，重试一次后拿到")
            write(tmp, "000300_SH", "2026-10-14")
            with patch.object(U, "DATA_DIR", Path(tmp)), patch.object(U, "update_one", return_value=("x", "no_new_rows", 0)), \
                    patch.object(U.ts, "pro_api"), patch.object(U, "load_dotenv"), patch.dict("os.environ", {"TUSHARE_TOKEN": "t"}), \
                    patch("sys.argv", argv + ["--retries", "2"]):
                self.assertEqual(U.main(), 2, "重试完仍缺，返回非零，daily_full 会显示告警")


if __name__ == "__main__":
    unittest.main()
