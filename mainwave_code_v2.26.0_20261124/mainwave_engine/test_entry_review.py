import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import entry_review as ER, theme_risk as TR

DAY = "2026-10-27"
AT = DAY + "T23:30:00+08:00"


class EntryReviewTests(unittest.TestCase):
    def setUp(self):
        self.record = {"symbol": "sh_600001", "price_date": DAY, "state": "PLAN_REVIEW", "paper_plan_eligible": True,
                       "blockers": [], "window": {"active": True}, "participation_rank": 1, "theme": "机械"}
        self.snapshot = {"as_of": DAY, "market": {"as_of": DAY, "data_status": "FRESH", "trend_review_allowed": True},
                         "records": [self.record]}

    def evaluate(self, mapping=None):
        return ER.build(DAY, self.snapshot, mapping or {}, ["sh_600001"], AT)

    def test_valid_entry_and_no_mutation(self):
        original = copy.deepcopy(self.snapshot)
        result = self.evaluate()
        self.assertEqual(result["top"], ["sh_600001"])
        self.assertFalse(result["orders_enabled"])
        self.assertEqual(self.snapshot, original)

    def test_strength_does_not_override_closed_market(self):
        self.snapshot["market"]["trend_review_allowed"] = False
        self.assertEqual(self.evaluate()["top"], [])

    def test_blocker_stale_record_expired_window_and_missing_permission(self):
        for key, value in (("blockers", ["STRUCTURE_RISK"]), ("price_date", "2026-10-26"),
                           ("window", {"active": False}), ("paper_plan_eligible", False)):
            with self.subTest(key=key):
                old = self.record[key]
                self.record[key] = value
                self.assertEqual(self.evaluate()["top"], [])
                self.record[key] = old

    def test_secondary_concept_risk_not_only_industry(self):
        result = self.evaluate({"sh_600001": {"concepts": ["消费", "人形机器人"]}})
        self.assertIn("THEME_RISK_REVIEW", result["rows"][0]["reasons"])
        self.assertEqual(result["top"], [])

    def test_max_two_and_keep_original_participation_order(self):
        self.snapshot["records"] = [{**self.record, "symbol": f"sh_60000{i}", "participation_rank": 4-i} for i in range(1, 4)]
        result = ER.build(DAY, self.snapshot, {}, [r["symbol"] for r in self.snapshot["records"]], AT)
        self.assertEqual(result["top"], ["sh_600003", "sh_600002"])
        self.assertEqual(result["eligible_count"], 3)

    def test_risk_respects_known_at_effective_date_and_expiry(self):
        self.assertEqual(TR.evaluate("机器人概念", DAY, DAY + "T21:00:00+08:00")["theme_risk_action"], "ALLOW")
        self.assertEqual(TR.evaluate("机器人概念", "2026-10-26")["theme_risk_action"], "ALLOW")
        self.assertEqual(TR.evaluate("机器人概念", "2026-10-31")["theme_risk_action"], "ALLOW")
        self.assertEqual(TR.evaluate("人形机器人", DAY, AT)["theme_risk_action"], "REQUIRE_EXTERNAL_CONFIRMATION")

    def test_unreadable_risk_config_is_not_allow(self):
        with patch.object(TR, "_read", return_value={"error": "ValueError"}):
            self.assertNotEqual(TR.evaluate("机械", DAY)["theme_risk_action"], "ALLOW")


class CoolingTests(unittest.TestCase):
    def check(self, missing=False, mapping_changed=False):
        days = ["2026-10-23", "2026-10-26", DAY]
        facts = [{"as_of": d, "previous_day": days[i-1] if i else "2026-10-22", "known_at": d + "T20:00:00+08:00",
                  "themes": {"top_by_limit_up": [{"theme": "机器人", "limit_up": [42, 22, 12][i],
                                                    "breadth": {"value": [.90, .34, .11][i]}}]}}
                 for i, d in enumerate(days)]
        if missing:
            facts[1]["themes"] = {}
        with tempfile.TemporaryDirectory() as tmp:
            manifests = [{"sources": {"concept_mapping": {"sha256": "other" if i == 1 and mapping_changed else "same"}}} for i in range(3)]
            for i in range(2):
                p = Path(tmp) / days[i]
                p.mkdir()
                (p / "manifest.json").write_text(json.dumps(manifests[i]), encoding="utf-8")
            def prior(root, day, known_at):
                i = days.index(day)
                return facts[i], {"path": str(Path(tmp) / day / "facts.json")}
            with patch.object(ER.R, "previous_review", side_effect=prior):
                return ER.cooling(facts[-1], manifests[-1], tmp, AT)

    def test_consecutive_cooling(self):
        self.assertEqual(self.check()["rows"][0]["status"], "COOLING")

    def test_absent_theme_is_unknown_not_zero(self):
        self.assertEqual(self.check(missing=True)["rows"][0]["status"], "UNKNOWN")

    def test_changed_mapping_is_not_comparable(self):
        self.assertEqual(self.check(mapping_changed=True)["rows"][0]["status"], "UNKNOWN")
