import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from research.route_legacy_archive import march_events, replay, symbol
from research import test_route_followup as fixtures


class LegacyArchiveTests(unittest.TestCase):
    def test_symbol_normalization(self):
        self.assertEqual(symbol("600001.SH"), "sh_600001")
        self.assertEqual(symbol("sz_300001"), "sz_300001")
        self.assertEqual(symbol("bad"), "")

    def test_internal_date_must_match_snapshot_filename(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            p = root / "concept_cache" / "concepts_20260311.json"
            p.parent.mkdir()
            obj = {"date": "20260312", "concept_stocks": {"demo": ["600001.SH", "600002.SH", "600003.SH"]},
                   "concept_info": {"demo": {"limit_up_count": 4}}}
            p.write_text(json.dumps(obj), encoding="utf-8")
            self.assertEqual(march_events(root), [])
            obj["date"] = "20260311"
            p.write_text(json.dumps(obj), encoding="utf-8")
            events = march_events(root)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["anchor_type"], "HEAT_OBSERVED_NOT_FIRST_IGNITION")

    def test_expired_preparation_cannot_back_later_cycle(self):
        frames = {}
        members = ["sh_600001", "sh_600002", "sh_600003"]
        for index, s in enumerate(members):
            f = fixtures.FollowupTests().fixture()
            f["return_index"], f["atr_pct"], f["bias_atr"] = 100., 2., 0.
            f["ret5"] = .1 if index == 0 else 0.
            f["prepared"], f["accepted_bias2"] = True, False
            if index:
                f.loc[22, "breakout_status"] = "REJECTED"
                f.loc[23:, "breakout_id"] = "new"
                f.loc[23, "accepted_bias2"] = True
            frames[s] = f
        events = [{"as_of": str(frames[members[0]].date.iloc[20].date()), "theme": "demo",
                   "members": members, "source": "fixture", "anchor_type": "HEAT_OBSERVED_NOT_FIRST_IGNITION"}]
        result = replay(events, frames, "accepted_or_reclaim")
        self.assertFalse(result.selected.any())
        self.assertTrue(result.status.eq("STRUCTURE_BROKEN").all())


if __name__ == "__main__":
    unittest.main()
