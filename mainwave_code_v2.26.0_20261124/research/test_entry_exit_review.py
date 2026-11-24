import unittest
import json
from pathlib import Path
import tempfile

import pandas as pd

from mainwave_engine import pullback_plan as PP
from research import route_followup as RF
from research.entry_exit_review import waiting_trial, trial
from research import test_route_followup as fixtures


class EntryExitTests(unittest.TestCase):
    def frame(self):
        f = fixtures.FollowupTests().fixture()
        f["index_low"], f["plan_structure"] = 99., 98.
        f["amount"], f["return_index"] = 100., 100.
        f["close"] = 100.
        f.loc[20, "breakout_event"] = "ACCEPTED"
        return f

    def test_retest_cannot_confirm_on_test_bar(self):
        f = self.frame()
        f.loc[21, "amount"] = 60.
        f.loc[22, "index_close"] = 101.
        p = PP.annotate(f)
        self.assertEqual(p.loc[21, "plan_status"], "TESTING")
        self.assertEqual(p.loc[22, "plan_status"], "CONFIRMED")
        self.assertEqual(p.loc[22, "plan_kind"], "ACCEPTANCE_RETEST")
        self.assertEqual(p.loc[23, "plan_status"], "WAIT_PULLBACK")

    def test_prefix_invariance_and_new_reentry_identity(self):
        f = self.frame()
        f.loc[[21, 24], "amount"] = 60.
        f.loc[[22, 25], "index_close"] = 101.
        p = PP.annotate(f)
        pd.testing.assert_frame_equal(p.iloc[:23], PP.annotate(f.iloc[:23]))
        self.assertNotEqual(p.loc[22, "plan_id"], p.loc[25, "plan_id"])
        self.assertEqual(p.loc[25, "plan_status"], "CONFIRMED")

    def test_strong_recovery_requires_price_and_amount(self):
        f = self.frame()
        f.loc[21, "amount"] = 60.
        f.loc[22, "index_close"] = 101.
        f.loc[23, ["index_close", "amount_ratio"]] = [103., 1.1]
        p = PP.annotate(f, strong_reclaim=True)
        self.assertEqual(p.loc[22, "plan_status"], "TESTING")
        self.assertEqual(p.loc[23, "plan_status"], "CONFIRMED")
        pd.testing.assert_frame_equal(p.iloc[:23], PP.annotate(f.iloc[:23], strong_reclaim=True))

    def test_failed_structure_does_not_later_reuse_test(self):
        f = self.frame()
        f.loc[21, "amount"] = 60.
        f.loc[22, "index_close"] = 97.
        f.loc[23, "index_close"] = 101.
        p = PP.annotate(f)
        self.assertEqual(p.loc[22, "plan_status"], "FAILED")
        self.assertNotEqual(p.loc[23, "plan_status"], "CONFIRMED")

    def test_opening_guard_uses_signal_metrics(self):
        f = self.frame()
        f.loc[11, ["ma20", "atr20"]] = [110., 20.]
        a = RF.arrays(f)
        self.assertEqual(PP.opening_check(a, 10, 105.), "ENTRY_OPEN_BIAS")
        self.assertEqual(PP.opening_check(a, 10, 102.), "ENTRY_OPEN_GAP")
        self.assertEqual(PP.opening_check(a, 10, 97.), "ENTRY_OPEN_STRUCTURE")
        self.assertEqual(PP.opening_check(a, 10, 100.), "")

    def test_intraday_test_is_not_soft_structure_failure(self):
        f = self.frame()
        f.loc[12, "index_low"] = 97.
        calendar = tuple(f.date.dt.strftime("%Y-%m-%d"))
        r = RF.simulate(RF.arrays(f), 10, "sh_600001", "phase36_20", calendar=calendar)
        self.assertEqual(r["reason"], "HORIZON")

    def test_soft_structure_close_exits_next_open(self):
        f = self.frame()
        f.loc[12, ["index_low", "index_close"]] = [97., 97.5]
        f.loc[13, ["open", "index_open"]] = [98.5, 98.5]
        calendar = tuple(f.date.dt.strftime("%Y-%m-%d"))
        trace = []
        r = RF.simulate(RF.arrays(f), 10, "sh_600001", "phase36_20", calendar=calendar, trace=trace)
        self.assertEqual(r["reason"], "STRUCTURE_CLOSE_EXIT")
        self.assertEqual(r["exit_date"], calendar[13])
        self.assertEqual(r["exit_index"], 98.5)
        self.assertEqual(trace[-2]["fills"], [])

    def test_hard_stop_tplus1_stays_in_force(self):
        f = self.frame()
        f.loc[11, "index_low"] = 95.
        f.loc[12, ["open", "index_open"]] = [94., 94.]
        r = RF.simulate(RF.arrays(f), 10, "sh_600001", "phase36_20",
                        calendar=tuple(f.date.dt.strftime("%Y-%m-%d")))
        self.assertEqual(r["reason"], "DEFERRED_STOP")
        self.assertAlmostEqual(r["net"], -6.3)

    def test_mature_warning_not_initial_upper_wick(self):
        f = self.frame()
        f.loc[12:, "index_high"] = 108.
        f.loc[12:, "amount_ratio"] = 1.5
        f.loc[12:, "close_location"] = .2
        a = RF.arrays(f)
        self.assertFalse(PP.profit_warning(a, 12, 11, 100., 2., 108.))
        self.assertFalse(PP.profit_warning(a, 16, 11, 100., 2., 104.))
        self.assertTrue(PP.profit_warning(a, 16, 11, 100., 2., 108.))

    def test_profit_exit_preserves_partial_and_next_open(self):
        f = self.frame()
        f.loc[12:16, "index_high"] = 108.
        f.loc[15, ["amount_ratio", "close_location"]] = [1.5, .2]
        f.loc[16, ["open", "index_open"]] = [103., 103.]
        calendar = tuple(f.date.dt.strftime("%Y-%m-%d"))
        trace = []
        r = RF.simulate(RF.arrays(f), 10, "sh_600001", "profit36_20", calendar=calendar, trace=trace)
        self.assertEqual(r["exit_date"], calendar[16])
        self.assertEqual(r["partials"], 1)
        self.assertEqual(r["reason"], "MATURE_WEAK_CLOSE_EXIT")
        self.assertAlmostEqual(r["net"], 4.2)
        self.assertEqual(trace[-1]["fills"][0]["weight"], .5)

    def test_rejected_open_waits_for_new_confirmation_not_next_open(self):
        f = self.frame()
        f["plan_status"], f["plan_test_at"], f["plan_id"], f["plan_kind"] = "WAIT_PULLBACK", "", "", ""
        calendar = tuple(f.date.dt.strftime("%Y-%m-%d"))
        f.loc[12, ["plan_status", "plan_test_at"]] = ["CONFIRMED", calendar[11]]
        f.loc[13, ["open", "index_open"]] = [105., 105.]
        r = waiting_trial(RF.arrays(f), 10, "sh_600001", calendar)
        self.assertFalse(r["trace"])
        self.assertEqual(r["rejected"][0]["status"], "ENTRY_OPEN_BIAS")

    def test_unsettled_trade_still_reports_actual_entry(self):
        f = self.frame().iloc[:14]
        calendar = tuple(f.date.dt.strftime("%Y-%m-%d"))
        r = trial(RF.arrays(f), 10, "sh_600001", "phase36_20", calendar)
        self.assertEqual(r["status"], "CENSORED")
        self.assertEqual(r["entry_date"], calendar[11])
        self.assertNotIn("net", r)

    def test_reentry_can_use_new_confirmation_on_exit_day_but_not_same_plan(self):
        f = self.frame()
        f["plan_status"], f["plan_test_at"], f["plan_id"], f["plan_kind"] = "WAIT_PULLBACK", "", "", ""
        calendar = tuple(f.date.dt.strftime("%Y-%m-%d"))
        f.loc[15, ["plan_status", "plan_test_at", "plan_id"]] = ["CONFIRMED", calendar[13], "new-plan"]
        a = RF.arrays(f)
        r = waiting_trial(a, 10, "sh_600001", calendar, start=15, used_plans={"previous-plan"})
        self.assertEqual(r["entry_date"], calendar[16])
        r = waiting_trial(a, 10, "sh_600001", calendar, start=15, used_plans={"new-plan"})
        self.assertFalse(r["trace"])

    def test_attachment_absent_or_incomplete_not_exposed(self):
        from mainwave_engine import entry_exit_study as ES
        summary = {"symbol": "sh_600001", "signal_date": "2026-01-05"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(ES.load(summary, root))
            folder = root / ES.DIRECTORY
            folder.mkdir()
            (folder / "manifest.json").write_text("{}")
            path = folder / "sh_600001_2026-01-05.json"
            path.write_text(json.dumps({"symbol": "sh_600001", "original_signal": "2026-01-05"}))
            self.assertEqual(ES.load(summary, root)["symbol"], "sh_600001")
            path.write_text('{"symbol":"wrong"}')
            with self.assertRaises(ValueError):
                ES.load(summary, root)
            with self.assertRaises(ValueError):
                ES.load({**summary, "symbol": "../escape"}, root)


if __name__ == "__main__":
    unittest.main()
