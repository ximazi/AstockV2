"""Attribution corrections must be sourced, dated and shared; no future or industry-wide inference."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import mock_open, patch

from . import data as D, s4_scan as S4, stock_analyzer as SA, theme_semantics as T

DAY = "2026-10-29"


class MembershipEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "links.json"
        self.link = {"symbol": "sz_000678", "themes": ["机器人概念", "人形机器人"],
                     "relation_kind": "MARKET_NARRATIVE", "known_at": DAY + "T15:18:33+08:00",
                     "effective_from": DAY, "published_on": "2026-10-27", "review_on": "2026-11-05",
                     "source_urls": ["https://example.com/announcement"], "evidence": "产品应用与市场叙事分开",
                     "risk_note": "非机器人主营"}
        self.mapping = {s: {"name": s, "concepts": ["汽车配件", "减速器"]} for s in ("sz_000678", "sh_600001")}
        self.write()
        self.patcher = patch.object(T, "MEMBERSHIP_EVIDENCE", self.path)
        self.patcher.start()
        D.concepts_map.cache_clear()

    def tearDown(self):
        D.concepts_map.cache_clear()
        self.patcher.stop()
        self.temp.cleanup()

    def write(self, link=None):
        self.path.write_text(json.dumps({"links": [self.link if link is None else link]}), encoding="utf-8")

    def test_only_named_stock_and_no_mutation_or_industry_expansion(self):
        original = copy.deepcopy(self.mapping)
        updated, meta = T.apply_membership_evidence(self.mapping, DAY)
        self.assertIn("人形机器人", updated["sz_000678"]["concepts"])
        self.assertEqual(updated["sh_600001"], original["sh_600001"])
        self.assertEqual(updated["sz_000678"]["_membership_evidence"], [self.link])
        self.assertEqual(self.mapping, original)
        self.assertEqual(len(meta["applied"]), 1)

    def test_future_knowledge_not_backfilled_even_after_research_date(self):
        updated, meta = T.apply_membership_evidence(self.mapping, "2026-10-28", DAY + "T22:00:00+08:00")
        self.assertEqual(updated, self.mapping)
        self.assertEqual(meta["applied"], [])

    def test_same_day_knowledge_cutoff(self):
        early, _ = T.apply_membership_evidence(self.mapping, DAY, DAY + "T14:30:00+08:00")
        late, _ = T.apply_membership_evidence(self.mapping, DAY, DAY + "T15:18:33+08:00")
        self.assertEqual(early, self.mapping)
        self.assertIn("机器人概念", late["sz_000678"]["concepts"])

    def test_expired_link_is_not_reused(self):
        self.write({**self.link, "expires_after": DAY})
        updated, _ = T.apply_membership_evidence(self.mapping, "2026-10-30")
        self.assertEqual(updated, self.mapping)

    def test_missing_sources_unknown_type_naive_clock_or_false_publication_rejected(self):
        for change in ({"source_urls": []}, {"relation_kind": "guess"}, {"known_at": DAY + "T15:18:33"},
                       {"published_on": "2026-10-30"}, {"themes": "机器人概念"}):
            self.write({**self.link, **change})
            with self.subTest(change=change):
                updated, meta = T.apply_membership_evidence(self.mapping, DAY)
                self.assertEqual(updated, self.mapping)
                self.assertTrue(meta["errors"])

    def test_invalid_file_is_explicit_and_keeps_original_map(self):
        self.path.write_text("broken", encoding="utf-8")
        updated, meta = T.apply_membership_evidence(self.mapping, DAY)
        self.assertEqual(updated, self.mapping)
        self.assertTrue(meta["errors"])

    def test_point_in_time_loader_consumes_same_evidence(self):
        base = self.root / "base.json"
        base.write_text(json.dumps({"generated": "2026-10-27", "mapping": self.mapping}), encoding="utf-8")
        with patch.object(T, "MAPPING", base), patch.object(T, "MAPPING_HISTORY", self.root / "history"), \
                patch.object(T, "DYNAMIC_MAPPING_HISTORY", self.root / "dynamic"):
            old, _ = T.load_mapping_as_of("2026-10-28")
            new, meta = T.load_mapping_as_of(DAY)
        self.assertNotIn("机器人概念", old["sz_000678"]["concepts"])
        self.assertIn("机器人概念", new["sz_000678"]["concepts"])
        self.assertEqual(len(meta["membership_evidence"]["applied"]), 1)

    def test_daily_concept_map_has_date_separated_cache(self):
        with patch("builtins.open", mock_open()), patch.object(D.json, "load", return_value={"mapping": self.mapping}):
            old = D.concepts_map("2026-10-28")
            new = D.concepts_map(DAY)
        self.assertNotIn("机器人概念", old["sz_000678"])
        self.assertIn("机器人概念", new["sz_000678"])

    def test_single_stock_reader_consumes_same_evidence(self):
        with patch.object(SA, "_read_json", return_value={"mapping": self.mapping}):
            old = SA._mapping("sz_000678", "2026-10-28")
            new = SA._mapping("sz_000678", DAY)
        self.assertNotIn("机器人概念", old["concepts"])
        self.assertIn("机器人概念", new["concepts"])

    def test_s4_risk_check_sees_corrected_theme_before_any_virtual_add(self):
        folder = self.root / DAY.replace("-", "")
        folder.mkdir()
        (folder / "s4_opportunities.json").write_text(json.dumps({"as_of": DAY, "buy_advised": True,
            "recommended": [{"symbol": "sz_000678", "industry": "汽车配件"}],
            "market_risk": {"as_of": DAY, "level": "NORMAL"}}), encoding="utf-8")
        (self.root / "concept_mapping.json").write_text(json.dumps({"mapping": self.mapping}), encoding="utf-8")
        def risk(themes, day):
            self.assertIn("机器人概念", themes)
            return {"theme_risk_action": "BLOCK_NEW_ENTRY", "theme_risk_reason": "test risk"}
        with patch.object(S4, "ROOT", self.root), patch.object(S4.TR, "evaluate_many", side_effect=risk), \
                patch.object(S4.VP, "add") as add:
            with self.assertRaisesRegex(ValueError, "test risk"):
                S4.pick(DAY, "000678", out_root=self.root)
            add.assert_not_called()


if __name__ == "__main__":
    unittest.main()
