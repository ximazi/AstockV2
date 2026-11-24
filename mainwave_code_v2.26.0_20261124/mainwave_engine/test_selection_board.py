from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mainwave_engine import selection_board as SB


class SelectionBoardTests(unittest.TestCase):
    def test_counts_and_top_five(self):
        data = SB.selection_list()
        self.assertEqual(data["count"], 36)
        self.assertEqual(len({row["symbol"] for row in data["records"]}), 36)
        self.assertEqual([batch["count"] for batch in data["batches"]], [6, 13, 15, 5, 5])
        self.assertEqual([row["symbol"] for row in data["records"][:5]],
                         ["sz_000738", "sz_000519", "sh_688023", "sh_603686", "sz_301252"])

    def test_history_and_risk_not_erased(self):
        rows = {row["symbol"]: row for row in SB.selection_list()["records"]}
        self.assertEqual(len(rows["sh_603444"]["selections"]), 2)
        self.assertEqual(rows["sh_603444"]["selections"][0]["snapshot"]["route"],
                         "LOCAL_FIRST_PULLBACK_CONFIRMED")
        self.assertIn("BLOCKED", rows["sz_300364"]["snapshot"]["engine_permission"])
        self.assertEqual(rows["sz_300731"]["group"], "WATCH")
        self.assertIn("未入前5", rows["sz_300731"]["note"])
        for symbol in ("sz_000417", "sz_001222", "sz_300852", "sh_603161", "sh_603122"):
            self.assertNotIn(symbol, rows)

    def test_snapshots_have_consistent_dates_and_sources(self):
        data = SB.selection_list()
        json.dumps(data, allow_nan=False)
        ids = [batch["id"] for batch in data["batches"]]
        self.assertEqual(len(ids), len(set(ids)))
        for batch in data["batches"]:
            self.assertTrue((SB.REGISTRY.parent.parent / batch["report"]).is_file())
        for row in data["records"]:
            self.assertRegex(row["symbol"], r"^(sh|sz)_\d{6}$")
            self.assertTrue(row["theme"] and row["note"])
            self.assertEqual(len({item["id"] for item in row["selections"]}), len(row["selections"]))
            for item in row["selections"]:
                self.assertEqual(item["as_of"], item["snapshot"]["price_date"])
                self.assertLessEqual(item["as_of"], item["recorded_at"])
                self.assertGreater(item["snapshot"]["close"], 0)
                self.assertTrue(item["snapshot"]["cycle"])

    def test_later_review_can_downgrade(self):
        registry = json.loads(SB.REGISTRY.read_text(encoding="utf-8"))
        old = registry["batches"][-1]
        entry = {**old["records"][0], "group": "RESERVE", "note": "downgraded"}
        registry["batches"].append({**old, "id": "later_review", "records": [entry]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(json.dumps(registry), encoding="utf-8")
            row = next(row for row in SB.selection_list(path)["records"]
                       if row["symbol"] == entry["symbol"])
        self.assertEqual(row["group"], "RESERVE")
        self.assertEqual(len(row["selections"]), 3)
        self.assertEqual(row["selections"][-2]["group"], "CORE_WATCH")

    def test_empty_and_missing_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.json"
            path.write_text(json.dumps({"version": "test", "updated_at": "", "scope": "",
                                        "batches": []}), encoding="utf-8")
            self.assertEqual(SB.selection_list(path)["records"], [])
            with self.assertRaises(FileNotFoundError):
                SB.selection_list(Path(directory) / "missing.json")

    def test_real_routes_and_sidebar_wiring(self):
        import mainwave_visual_app as visual

        client = visual.app.test_client()
        with patch.object(visual, "analyze", side_effect=AssertionError("list must not analyze")), \
             patch.object(visual, "candidate_list", side_effect=AssertionError("list must not scan")):
            response = client.get("/api/selections")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["count"], 36)
        self.assertEqual(response.json["permission"], "RESEARCH_ONLY_NO_ORDER")
        self.assertEqual(client.post("/api/selections").status_code, 405)
        with patch.object(visual, "latest_report_date", return_value="2026-09-04"):
            page = client.get("/").get_data(as_text=True)
        for marker in ('id="selectionSource"', 'id="selectionBatch"', 'id="selectionSearch"',
                       "'/api/selections'", "switchSource('selections')"):
            self.assertIn(marker, page)
        with patch.object(visual, "selection_list", side_effect=FileNotFoundError("missing")):
            self.assertEqual(client.get("/api/selections").status_code, 500)


if __name__ == "__main__":
    unittest.main()
