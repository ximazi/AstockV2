import json
import unittest
from unittest.mock import patch
import zipfile

import pandas as pd

from mainwave_engine import review_workbench as RW, review_catalog as RC, review_batch as RB
from mainwave_engine import test_review_workbench as fixtures


class ReviewBulkTests(unittest.TestCase):
    setUp = fixtures.ReviewWorkbenchTests.setUp
    body = fixtures.ReviewWorkbenchTests.body

    def catalog(self, staged_net=None):
        source = self.root / "source"
        source.mkdir()
        frozen = self.root / "catalog_source" / "data"
        frozen.mkdir(parents=True)
        prices = b"date,close\n2026-01-05,6.9\n"
        (frozen / "sh_600866_daily.csv").write_bytes(prices)
        rows = [{"year": 2026, "symbol": "sh_600866", "date": "2026-01-05", "exit": p,
                 "status": "CLOSED" if p == "legacy20" else "CENSORED", "net": 1.99 if p == "legacy20" else None}
                for p in sorted(RC.POLICIES)]
        if staged_net is not None:
            next(row for row in rows if row["exit"] == "stage36_20").update(status="CLOSED", net=staged_net)
        rows.append({**rows[0], "year": 2025, "date": "2025-01-05"})
        pd.DataFrame(rows).to_csv(source / "r1_exit_replay.csv.gz", index=False)
        (source / "exit_replay_manifest.json").write_bytes(RW.encoded({"as_of": "2026-02-28", "inputs": [{"symbol": "sh_600866", "price_hash": RW.digest(prices)}]}))
        with patch.object(RW.SA, "_read_json", return_value={"mapping": {"sh_600866": {"name": "星湖科技"}}}), \
             patch("research.route_clock.sessions", return_value=("2026-01-05", "2026-01-06")):
            return RC.build(self.root, source)

    def run_batch(self, kind, ids=None, reviews=None, cancel=False):
        job_id = "a" * 32
        folder = RB.job_dir(job_id, self.root)
        folder.mkdir(parents=True)
        ids = ids or [self.case_id]
        state = {"id": job_id, "kind": kind, "status": "QUEUED", "total": len(ids),
                 "completed": 0, "succeeded": 0, "failed": 0, "current": ""}
        if cancel:
            (folder / "cancel").touch()
        RB.run(ids, kind, reviews or {}, state, self.root)
        return RB.status(job_id, self.root), folder / "bundle.zip"

    def test_catalog_keeps_all_policies_nonclosed_and_original_case(self):
        catalog = self.catalog()
        self.assertEqual(catalog["signals"], 1)
        self.assertEqual(len(catalog["entries"]), 3)
        self.assertEqual(RW.detail(self.case_id, self.root), self.case)
        with patch.object(RW, "detail", side_effect=AssertionError("Listing must not load full cases")):
            listing = RW.case_list(self.root)
        self.assertEqual(len(listing["cases"]), 1)
        self.assertEqual(listing["raw_cases"], 3)
        self.assertEqual(listing["cases"][0]["exit_selection"]["pending_count"], 2)
        self.assertEqual(RC.build(self.root, self.root / "missing"), catalog)

    def test_catalog_detects_tampering_and_materializes_from_frozen_source(self):
        catalog = self.catalog()
        cid = next(cid for cid in catalog["entries"] if cid.endswith("stage36_20"))
        with patch.object(RW, "import_r1_case") as importer:
            RC.materialize(cid, self.root)
            self.assertEqual(importer.call_args.args[3], self.root / "catalog_source")
        catalog["signals"] = 999
        (self.root / "catalog.json").write_bytes(RW.encoded(catalog))
        with self.assertRaisesRegex(ValueError, "校验失败"):
            RC.load(self.root)

    def test_bulk_validates_unknown_duplicate_and_empty_ids(self):
        for ids in ([], "abc", [self.case_id, self.case_id], ["../../bad"], [None]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                RB.validate_ids(ids, self.root)
        self.assertEqual(RB.validate_ids([self.case_id], self.root), [self.case_id])
        with self.assertRaises(ValueError):
            RB.job_dir("../../bad", self.root)

    def test_case_bulk_contains_png_and_manifest_and_reuses_cache(self):
        state, path = self.run_batch("case")
        self.assertEqual(state["status"], "COMPLETE")
        with zipfile.ZipFile(path) as archive:
            self.assertIsNone(archive.testzip())
            self.assertIn(f"{self.case_id}/kline_full.png", archive.namelist())
            self.assertIn(f"{self.case_id}/price_volume_context.json", archive.namelist())
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertEqual(manifest["requested_ids"], [self.case_id])
        with patch.object(RW, "archive", side_effect=AssertionError("Must reuse")):
            self.assertTrue(RB.case_archive(self.case_id, self.root).is_file())

    def test_case_export_cache_invalidates_when_observer_changes(self):
        from mainwave_engine import price_volume_context as PV
        from pathlib import Path
        first = RB.case_archive(self.case_id, self.root)
        original = Path.read_bytes
        def changed(path):
            content = original(path)
            return content + b"\n# observer update\n" if path == Path(PV.__file__) else content
        with patch.object(Path, "read_bytes", changed):
            second = RB.case_archive(self.case_id, self.root)
        self.assertNotEqual(first, second)
        self.assertTrue(first.is_file())
        self.assertTrue(second.is_file())

    def test_review_batch_freezes_saved_revision_and_includes_history(self):
        body = self.body()
        body["summary"] = "版本一人工意见"
        saved = RW.save_review(self.case_id, body, self.root)
        body.update(revision=1, summary="任务开始后修改")
        RW.save_review(self.case_id, body, self.root)
        with patch.object(RW, "archive", side_effect=AssertionError("Reviews need no charts")):
            state, path = self.run_batch("review", reviews={self.case_id: saved})
        self.assertEqual(state["status"], "COMPLETE")
        with zipfile.ZipFile(path) as archive:
            self.assertEqual(json.loads(archive.read("all_reviews.json"))[0]["summary"], "版本一人工意见")
            self.assertEqual(len(json.loads(archive.read(f"{self.case_id}/review_history.json"))), 1)
            self.assertFalse(any(n.endswith(".png") for n in archive.namelist()))
        self.assertEqual(RW.load_review(self.case_id, self.root)["revision"], 2)

    def test_unreviewed_catalog_export_stays_blank_without_materialization(self):
        catalog = self.catalog(staged_net=4.5)
        cid = next(cid for cid in catalog["entries"] if cid.endswith("stage36_20"))
        with patch.object(RW, "detail", side_effect=AssertionError("Blank reviews need no replay")):
            state, path = self.run_batch("review", [cid])
        self.assertEqual(state["status"], "COMPLETE")
        with zipfile.ZipFile(path) as archive:
            review = json.loads(archive.read("all_reviews.json"))[0]
            self.assertEqual((review["revision"], review["summary"], review["annotations"]), (0, "", []))
            reference = json.loads(archive.read(f"{cid}/case_reference.json"))
            self.assertEqual(reference["catalog_hash"], catalog["catalog_hash"])
            self.assertIsNone(reference["case_hash"])

    def test_best_policy_uses_closed_net_and_deterministic_ties(self):
        import copy
        def rows(nets, statuses=("CLOSED",) * 3):
            return [{"id": p, "route": "R1", "symbol": "sh_600866", "signal_date": "2026-01-05",
                     "exit_policy": p, "trade_status": status, "net_pct": net}
                    for p, net, status in zip(RC.POLICY_ORDER, nets, statuses)]
        for nets, expected in [((1, 5, 3), "stage36_20"), ((-4, -2, -1), "structure36_20"),
                               ((2, 2, 2), "legacy20"), ((0, -1, -2), "legacy20")]:
            data = rows(nets)
            before = copy.deepcopy(data)
            chosen = RC.best_per_signal(list(reversed(data)))[0]
            self.assertEqual(chosen["exit_policy"], expected)
            self.assertEqual(data, before)
        chosen = RC.best_per_signal(rows((-5, None, None), ("CLOSED", "ENTRY_LIMIT_UP", "CENSORED")))[0]
        self.assertEqual(chosen["net_pct"], -5)
        self.assertEqual(chosen["exit_selection"]["label"], "已平仓中暂优，仍有方案待定")
        for statuses in [("CENSORED",) * 3, ("ENTRY_LIMIT_UP",) * 3, ("HOLDING_DATA_UNVERIFIED",) * 3]:
            chosen = RC.best_per_signal(rows((None,) * 3, statuses))[0]
            self.assertEqual(chosen["exit_policy"], "legacy20")
            self.assertIsNone(chosen["net_pct"])
            self.assertNotIn("最优", chosen["exit_selection"]["label"])
        self.assertEqual(RC.best_per_signal(rows((float('nan'), 1, float('inf'))))[0]["exit_policy"], "stage36_20")

    def test_prior_policy_reviews_remain_separate_and_export_as_appendix(self):
        saved = RW.save_review(self.case_id, {**self.body(), "summary": "原legacy交易分析"}, self.root)
        self.catalog(staged_net=4.5)
        row = RW.case_list(self.root)["cases"][0]
        self.assertEqual(row["exit_policy"], "stage36_20")
        self.assertEqual(row["revision"], 0)
        self.assertEqual(row["exit_selection"]["related_reviews"][0]["id"], self.case_id)
        with self.assertRaises(ValueError):
            RB.validate_ids([self.case_id, row["id"]], self.root)
        self.assertEqual(RB.validate_ids([row["id"]], self.root), [row["id"]])
        later = RW.save_review(self.case_id, {**self.body(), "revision": 1, "summary": "导出提交后的修改"}, self.root)
        state, path = self.run_batch("review", [row["id"]], {self.case_id: saved})
        self.assertEqual(state["status"], "COMPLETE")
        with zipfile.ZipFile(path) as archive:
            self.assertEqual(json.loads(archive.read("all_reviews.json"))[0]["summary"], "")
            self.assertEqual(json.loads(archive.read(f"{row['id']}/other_policy_reviews.json")), [saved])
            self.assertEqual(json.loads(archive.read(f"{row['id']}/exit_selection.json"))["related_reviews"][0]["revision"], 1)
            self.assertIn("事后择优", archive.read(f"{row['id']}/review.md").decode())
        self.assertEqual(RW.load_review(self.case_id, self.root), later)
        self.assertEqual(RW.detail(self.case_id, self.root), self.case)

    def test_case_export_labels_hindsight_without_changing_frozen_snapshot(self):
        import io
        self.catalog()
        with zipfile.ZipFile(io.BytesIO(RW.archive(self.case_id, "case", self.root))) as archive:
            self.assertEqual(json.loads(archive.read("case.json")), self.case)
            self.assertEqual(json.loads(archive.read("exit_selection.json"))["selected_id"], self.case_id)
            self.assertIn("事后择优", archive.read("README.md").decode())

    def test_review_status_follows_signal_without_relabeling_selected_review(self):
        body = {**self.body(), "summary": "原交易复盘内容", "entry_notes": "原买点判断"}
        draft = RW.save_review(self.case_id, body, self.root)
        self.catalog(staged_net=4.5)
        row = RW.case_list(self.root)["cases"][0]
        self.assertEqual(row["signal_review_status"], "DRAFT")
        reviewed = RW.save_review(self.case_id, {**body, "revision": 1, "status": "REVIEWED"}, self.root)
        row = RW.case_list(self.root)["cases"][0]
        self.assertEqual(row["signal_review_status"], "REVIEWED")
        self.assertEqual((row["exit_policy"], row["revision"], row["review_status"]), ("stage36_20", 0, "DRAFT"))
        self.assertEqual(row["exit_selection"]["related_reviews"][0]["review_status"], "REVIEWED")
        self.assertEqual(reviewed["revision"], 2)
        self.assertEqual(reviewed["summary"], draft["summary"])
        self.assertEqual(reviewed["entry_notes"], draft["entry_notes"])
        from contextlib import closing
        import sqlite3
        with closing(sqlite3.connect(self.root / "reviews.sqlite3")) as conn:
            history = [json.loads(r[0]) for r in conn.execute("SELECT payload FROM reviews WHERE case_id=? ORDER BY revision", (self.case_id,))]
        self.assertEqual(history, [draft, reviewed])
        self.assertEqual(RW.detail(self.case_id, self.root), self.case)

    def test_list_order_reviewed_then_newest_unreviewed_regardless_of_outcome(self):
        def row(cid, net, status="CLOSED", reviewed="UNREVIEWED", day="2026-01-05"):
            return {"id": cid, "symbol": cid, "signal_date": day, "net_pct": net,
                    "trade_status": status, "signal_review_status": reviewed}
        rows = [row("profit", 5, day="2026-09-09"), row("reviewed_profit", 10, reviewed="REVIEWED", day="2026-02-01"),
                row("pending", None, "CENSORED", day="2026-09-11"), row("loss", -1),
                row("draft_loss", -5, reviewed="DRAFT", day="2026-09-10"),
                row("reviewed_loss", -3, reviewed="REVIEWED"), row("flat", 0, day="2026-08-01"),
                row("unfilled", None, "ENTRY_LIMIT_UP", day="2026-09-11")]
        self.assertEqual([r["id"] for r in sorted(reversed(rows), key=RC.review_order)],
                         ["reviewed_loss", "reviewed_profit", "pending", "unfilled", "draft_loss", "profit", "flat", "loss"])

    def test_newest_order_is_applied_to_archive_only_lists(self):
        import copy
        for suffix, day in (("newest", "2026-09-11"), ("middle", "2026-05-01")):
            case = copy.deepcopy(self.case)
            cid = f"R1_sh_600866_{suffix}_legacy20"
            case["summary"].update(id=cid, signal_date=day)
            case.pop("case_hash")
            case["case_hash"] = RW.digest(RW.encoded(case))
            (self.root / "cases" / f"{cid}.json").write_bytes(RW.encoded(case))
        body = self.body()
        body.update(status="REVIEWED", summary="保留已复盘置顶")
        RW.save_review(self.case_id, body, self.root)
        rows = RW.case_list(self.root)["cases"]
        self.assertEqual([r["signal_date"] for r in rows], ["2026-01-05", "2026-09-11", "2026-05-01"])
        self.assertEqual(rows[0]["review_status"], "REVIEWED")

    def test_cancelled_and_failed_items_are_explicit_in_manifest(self):
        state, path = self.run_batch("case", cancel=True)
        self.assertEqual(state["status"], "CANCELLED")
        with zipfile.ZipFile(path) as archive:
            self.assertEqual(json.loads(archive.read("manifest.json"))["results"][0]["result"], "CANCELLED")
        with patch.object(RB, "case_archive", side_effect=ValueError("input mismatch")):
            state["status"] = "QUEUED"
            state["completed"] = 0
            (path.parent / "cancel").unlink()
            RB.run([self.case_id], "case", {}, state, self.root)
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["status"], "PARTIAL")
            self.assertEqual(manifest["failed"], 1)
            self.assertIn("input mismatch", manifest["results"][0]["error"])

    def test_http_batch_validation_and_status(self):
        from mainwave_visual_app import app
        client = app.test_client()
        self.assertEqual(client.post("/api/review-exports", json=[]).status_code, 400)
        self.assertEqual(client.post("/api/review-exports", json={}, headers={"Origin": "https://elsewhere.example"}).status_code, 403)
        with patch.object(RB, "start", return_value={"id": "a" * 32, "status": "QUEUED"}) as start:
            self.assertEqual(client.post("/api/review-exports", json={"kind": "review", "case_ids": [self.case_id]}).status_code, 202)
            start.assert_called_once()
        with patch.object(RB, "status", return_value={"status": "RUNNING"}):
            self.assertEqual(client.get("/api/review-exports/" + "a" * 32 + "/download").status_code, 409)

    def test_status_polling_and_atomic_writes_can_run_together(self):
        from concurrent.futures import ThreadPoolExecutor
        job_id = "b" * 32
        folder = RB.job_dir(job_id, self.root)
        folder.mkdir(parents=True)
        RB.write_status(folder, {"id": job_id, "status": "COMPLETE", "completed": 0})
        def update():
            for n in range(100):
                RB.write_status(folder, {"id": job_id, "status": "COMPLETE", "completed": n})
        def poll():
            for _ in range(100):
                self.assertEqual(RB.status(job_id, self.root)["id"], job_id)
        with ThreadPoolExecutor(max_workers=2) as pool:
            writer, reader = pool.submit(update), pool.submit(poll)
            writer.result()
            reader.result()


if __name__ == "__main__":
    unittest.main()
