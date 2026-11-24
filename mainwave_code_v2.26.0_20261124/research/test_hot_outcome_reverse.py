import unittest
from collections import defaultdict
from tempfile import TemporaryDirectory
from pathlib import Path

import pandas as pd

from research.hot_outcome_reverse import BASE, KINDS, audit, event_rows, scan_archive, tape


class HotOutcomeReverseTests(unittest.TestCase):
    def setUp(self):
        self.days = list(pd.bdate_range("2026-05-01", periods=42).strftime("%Y-%m-%d"))

    def test_three_boards_once_and_suspension_breaks_streak(self):
        f = pd.DataFrame({
            "date": self.days, "symbol": "sh_600000", "pct_chg": 1.0, "amount_rank": .2,
            "sealed": [i in (20, 21, 22, 23, 27, 28, 30, 31) for i in range(42)],
        }).drop(index=29)
        self.assertEqual(list(event_rows(f, self.days, self.days[18])),
                         [("sh_600000", KINDS[0], 20, 22)])

    def test_surge_requires_activity_and_rearms_only_after_gap(self):
        f = pd.DataFrame({
            "date": self.days, "symbol": "sh_600000", "pct_chg": 5.0,
            "sealed": False, "amount_rank": [.95 if i in (20, 22, 28) else .10 for i in range(42)],
        })
        self.assertEqual(list(event_rows(f, self.days, self.days[18])),
                         [("sh_600000", KINDS[1], 16, 20), ("sh_600000", KINDS[1], 24, 28)])

    def test_round_to_cent_and_exclude_st(self):
        with TemporaryDirectory() as temp:
            data = Path(temp)
            for symbol, prices in (("sh_600000", [11.06, 11.05]), ("sz_300123", [12.06, 12.05])):
                f = pd.DataFrame({"date": self.days[:23], "preclose": 10.05,
                                  "close": [10.3] * 20 + prices + [prices[0]], "volume": 1, "amount": 100,
                                  "tradestatus": 1, "pct_chg": 10, "isST": [0] * 22 + [1]})
                f.to_csv(data / (symbol + "_daily.csv"), index=False)
            actual, skipped = tape(data, self.days)
        self.assertEqual(skipped, [])
        self.assertEqual(list(actual.sealed), [True, False, True, False])

    def test_forming_is_context_only_and_no_archive_is_not_a_miss(self):
        symbol, begin, anchor = "sh_600000", 22, 24
        with TemporaryDirectory() as temp:
            root = Path(temp) / "results"
            folder = root / "mainwave_engine" / self.days[begin].replace("-", "")
            folder.mkdir(parents=True)
            (folder / "trajectory.csv").write_text(
                f"date,symbol,state,data_fresh\n{self.days[begin]},{symbol},FORMING,True\n", encoding="utf-8"
            )
            coverage, history = scan_archive(root, self.days, defaultdict(set, {begin: {symbol}}))
        prices = pd.DataFrame({"symbol": symbol, "date": self.days, "pct_chg": 10., "close": 10.})
        events = [(symbol, KINDS[0], begin, anchor)]
        ((row, records),) = list(audit(events, history, coverage, self.days, prices))
        self.assertEqual(row["new_result"], "ARCHIVE_GAP")
        self.assertEqual(records[0]["positive"], 0)
        for i in range(begin - 10, anchor + 1):
            coverage[i].update(BASE)
        ((row, _),) = list(audit(events, history, coverage, self.days, prices))
        self.assertEqual(row["new_result"], "NOT_SEEN_TO_ANCHOR")


if __name__ == "__main__":
    unittest.main()
