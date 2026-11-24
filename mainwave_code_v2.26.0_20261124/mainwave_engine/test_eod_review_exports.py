"""Supplement exports run on the synthetic review universe: every file lands, numbers agree with the facts."""
import json
from pathlib import Path
import unittest

import pandas as pd

from . import eod_review as R, eod_review_exports as X
from .test_eod_review import Universe, DAY


class Exports(Universe):
    def test_exports_are_written_and_reconcile_with_the_facts(self):
        facts, manifest = self.build()
        pointer = R.persist(DAY, facts, manifest, self.out)
        out = X.run(DAY, self.data, self.out, reference_root=self.root, mapping_path=str(self.mapping), days=3, use_tushare=False)
        names = sorted(p.name for p in out.iterdir())
        for expected in ("A_pool_yesterday.csv", "A_limit_lists_today.csv", "A_reconciliation_differences.csv", "A_promotion_1_to_2_reconcile.json",
                         "A_field_definitions.md", "B_history_5d.csv", "B_promotion_history.json", "B_broke_performance.csv", "C_theme_members.csv",
                         "C_key_stocks_daily_60d.csv", "C_limit_process_today.csv", "D_candidates_plan_review.json", "E_top_list_status.json",
                         "README.md", "manifest.json"):
            self.assertIn(expected, names)
        pool = pd.read_csv(out / "A_pool_yesterday.csv")
        self.assertEqual(sorted(pool.symbol), facts["yesterday_pool"]["members"])
        lists = pd.read_csv(out / "A_limit_lists_today.csv")
        self.assertEqual(sorted(lists[lists.kind == "U"].symbol), facts["boards"]["U"])
        self.assertEqual(sorted(lists[lists.kind == "Z"].symbol), facts["boards"]["Z"])
        hist = pd.read_csv(out / "B_history_5d.csv")
        today = hist[hist.day == DAY].iloc[0]
        self.assertEqual(int(today.limit_up), facts["boards"]["counts"]["U"])
        self.assertEqual(int(today.limit_down), facts["boards"]["counts"]["D"])
        self.assertEqual(int(today.pool_n), facts["yesterday_pool"]["n_valid"])
        recon = json.loads((out / "A_promotion_1_to_2_reconcile.json").read_text(encoding="utf-8"))
        self.assertEqual(recon["sentiment_table"]["value"], facts["sentiment"]["metrics"]["promotion_1_to_2"]["today"])
        status = json.loads((out / "E_top_list_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "NOT_COLLECTED")
        readme = (out / "README.md").read_text(encoding="utf-8")
        self.assertIn("未采集 / 未实施", readme)
        self.assertIn("NOT_COLLECTED", (out / "C_limit_process_today.csv").read_text(encoding="utf-8"))
        self.assertIn(pointer["version"], json.loads((out / "manifest.json").read_text(encoding="utf-8"))["review_version"])


if __name__ == "__main__":
    unittest.main()
