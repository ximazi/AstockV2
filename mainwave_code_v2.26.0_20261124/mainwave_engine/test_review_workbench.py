import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image

from mainwave_engine import review_workbench as RW


class ReviewWorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.case_id = "R1_sh_600866_20260105_legacy20"
        case = {"summary": {"id": self.case_id, "symbol": "sh_600866", "name": "星湖科技", "route": "R1",
                            "signal_date": "2026-01-05", "window_start": "2025-12-01", "window_end": "2026-02-28"},
                "source_row": {"net": 1.99}, "notice": RW.NOTICE, "selection": "按日期及代码选首条，不按收益",
                "daily": [{"date": "2026-01-05", "cycle_stage": "IGNITION"}],
                "raw_history": [{"date": "2026-01-05", "close": 6.9}], "units": {"close": "人民币"},
                "execution_trace": [{"date": "2026-01-06", "fills": []}]}
        dates = [f"2026-01-{day:02}" for day in (5, 6, 7, 8, 9, 12, 13, 14, 15, 16)]
        case["view"] = {"chart": {"bars": [
            {"date": day, "open": 6.8, "high": 7.1, "low": 6.7, "close": 6.9,
             "ema10": 6.85, "ema20": 6.8, "ema60": 6.75, "amount": 1.2} for day in dates],
            "stage_segments": [{"start": dates[0], "end": dates[-1], "stage": "PULLBACK", "label": "整理", "color": "#c9a02b"}]},
            "case_review": {"events": [
                {"kind": "TARGET_IGNITION", "date": dates[0], "raw_price": 10, "chart_price": 6.9},
                {"kind": "BUY", "date": dates[1], "raw_price": 10.1, "chart_price": 6.91},
                {"kind": "SELL", "date": dates[4], "raw_price": 10.3, "chart_price": 7.01}]}}
        case["case_hash"] = RW.digest(RW.encoded(case))
        self.case = case
        (self.root / "cases").mkdir()
        (self.root / "cases" / f"{self.case_id}.json").write_bytes(RW.encoded(case))

    def body(self):
        body = RW.blank_review(self.case)
        body.pop("updated_at")
        return body

    def test_empty_review_never_fabricates_human_conclusion(self):
        self.assertEqual(RW.load_review(self.case_id, self.root)["revision"], 0)
        self.assertFalse((self.root / "reviews.sqlite3").exists())
        self.assertEqual(len(RW.case_list(self.root)["cases"]), 1)

    def test_failure_selection_is_deterministic_closed_losing_and_deduplicated(self):
        import pandas as pd
        base = {"year": 2026, "exit": "legacy20", "status": "CLOSED", "net": -2,
                "date": "2026-01-05", "held": 3, "mfe": 1, "reason": "TRAIL_STOP"}
        rows = [{**base, "symbol": "sh_600001", **change} for change in (
            {"year": 2025}, {"status": "CENSORED"}, {"net": 1}, {"exit": "staged"})]
        rows += [{**base, "symbol": "sh_600002"},
                 {**base, "symbol": "sh_600002", "date": "2026-01-06", "mfe": 10},
                 {**base, "symbol": "sh_600003", "date": "2026-01-07", "held": 10, "mfe": 9},
                 {**base, "symbol": "sh_600004", "reason": "DEFERRED_STOP"}]
        frame = pd.DataFrame(rows)
        selected = RW.failure_samples(frame)
        self.assertEqual([r[0]["symbol"] for r in selected], ["sh_600002", "sh_600003", "sh_600004"])
        self.assertEqual(selected, RW.failure_samples(frame.sample(frac=1, random_state=17)))
        self.assertTrue(all("事后交易结果" in item[1] for item in selected))

    def test_missing_failure_group_does_not_fabricate_case(self):
        import pandas as pd
        frame = pd.DataFrame([{"year": 2026, "exit": "legacy20", "status": "CLOSED", "net": 1,
                               "date": "2026-01-05", "symbol": "sh_600001", "reason": "TRAIL_STOP", "held": 3, "mfe": 2}])
        with self.assertRaisesRegex(ValueError, "不补造"):
            RW.failure_samples(frame)

    def test_reimport_preserves_frozen_case_and_human_revision(self):
        body = self.body()
        body["summary"] = "已有人工判断必须保留"
        saved = RW.save_review(self.case_id, body, self.root)
        row = {"symbol": "sh_600866", "date": "2026-01-05", "year": 2026, "exit": "legacy20"}
        RW.import_r1_case(row, "新的描述不能覆盖", self.root, self.root / "no_source", "新分类")
        self.assertEqual(RW.detail(self.case_id, self.root), self.case)
        self.assertEqual(RW.load_review(self.case_id, self.root), saved)

    def test_persistence_and_conflict_preserve_prior_revisions(self):
        body = self.body()
        body.update(summary="人工判断，不覆盖系统", status="REVIEWED")
        saved = RW.save_review(self.case_id, body, self.root)
        self.assertEqual(saved["revision"], 1)
        self.assertEqual(RW.load_review(self.case_id, self.root)["summary"], body["summary"])
        with self.assertRaises(RW.Conflict):
            RW.save_review(self.case_id, body, self.root)
        body.update(revision=1, summary="第二轮补充")
        RW.save_review(self.case_id, body, self.root)
        with zipfile.ZipFile(io.BytesIO(RW.archive(self.case_id, "review", self.root))) as z:
            history = json.loads(z.read("review_history.json"))
            self.assertEqual([r["revision"] for r in history], [1, 2])
            self.assertIn("第二轮补充", z.read("review.md").decode())
        self.assertEqual(RW.detail(self.case_id, self.root), self.case)

    def test_annotations_validate_window_and_persist_unicode(self):
        body = self.body()
        annotation = {"start": "2026-01-05", "end": "2026-01-12", "stage": "PULLBACK", "note": "人工：回踩后需观察承接"}
        body["annotations"] = [annotation]
        self.assertEqual(RW.save_review(self.case_id, body, self.root)["annotations"], [annotation])
        for invalid in [{"start": "2025-01-01"}, {"end": "2026-01-01"}, {"stage": []}, {"note": 9}, {"start": "not-date"}]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                RW.validate_review({**body, "annotations": [annotation | invalid]}, self.case)

    def test_invalid_payloads_are_rejected_before_writing(self):
        for changes in [{"status": []}, {"verdict": {}}, {"summary": 7}, {"revision": True}, {"annotations": {}},
                        {"status": "REVIEWED", "summary": " "}, {"summary": "x" * 20001}, {"unknown": "field"}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                RW.save_review(self.case_id, self.body() | changes, self.root)
        with self.assertRaises(RW.Conflict):
            RW.save_review(self.case_id, self.body() | {"case_hash": "different"}, self.root)

    def test_case_export_has_whole_data_and_no_review_mixup(self):
        data = RW.archive(self.case_id, "case", self.root)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            self.assertEqual(json.loads(z.read("case.json")), self.case)
            self.assertIn("2026-01-05", z.read("lifecycle_daily.csv").decode("utf-8-sig"))
            self.assertIn("raw_history.csv", z.namelist())
            self.assertNotIn("review.json", z.namelist())
            for name in ("kline_full.png", "kline_trade_focus.png"):
                with Image.open(io.BytesIO(z.read(name))) as picture:
                    self.assertEqual(picture.size, (2560, 1440))
                    self.assertEqual(picture.info["CaseSHA256"], self.case["case_hash"])
                    self.assertNotEqual(picture.convert("RGB").getextrema(), ((255, 255),) * 3)
            self.assertEqual(json.loads(z.read("chart_exports.json"))["case_hash"], self.case["case_hash"])

    def test_chart_labels_prices_and_header_do_not_overlap(self):
        from mainwave_engine.review_chart import figure_for
        fig = figure_for(self.case, True)
        self.addCleanup(fig.clear)
        fig.canvas.draw()
        ax = fig.axes[0]
        labels = [t for t in ax.texts if "回测" in t.get_text() or "点火代理" in t.get_text()]
        self.assertEqual(len(labels), 3)
        self.assertTrue(any("2026-01-06\n原价 10.100" in t.get_text() for t in labels))
        self.assertEqual(labels[1].xy, (1, 6.91))
        renderer = fig.canvas.get_renderer()
        boxes = [t.get_bbox_patch().get_window_extent(renderer) for t in labels]
        self.assertFalse(any(left.overlaps(right) for i, left in enumerate(boxes) for right in boxes[i + 1:]))
        title, subtitle = [t.get_window_extent(renderer) for t in fig.texts[:2]]
        legend = fig.legends[0].get_window_extent(renderer)
        self.assertFalse(title.overlaps(subtitle))
        self.assertFalse(subtitle.overlaps(legend))

    def test_focus_preserves_every_event_and_no_exit_is_not_fabricated(self):
        from mainwave_engine.review_chart import chart_window, figure_for
        import pandas as pd
        case = copy.deepcopy(self.case)
        bars = [{**case["view"]["chart"]["bars"][0], "date": day.strftime("%Y-%m-%d")}
                for day in pd.bdate_range("2026-01-05", periods=200)]
        case["view"]["chart"]["bars"] = bars
        case["view"]["case_review"]["events"] = [
            {"kind": "BUY", "date": bars[60]["date"], "raw_price": 10, "chart_price": 6.9},
            {"kind": "SELL", "date": bars[95]["date"], "raw_price": 11, "chart_price": 7.0}]
        focused = chart_window(case, True)
        self.assertEqual(focused[0]["date"], bars[35]["date"])
        self.assertEqual(focused[-1]["date"], bars[120]["date"])
        case["view"]["case_review"]["events"] = []
        fig = figure_for(case, True)
        self.addCleanup(fig.clear)
        self.assertFalse(any("回测卖出" in t.get_text() for t in fig.axes[0].texts))

    def test_case_tampering_and_path_traversal_rejected(self):
        for name in ("../secret", "..\\secret", "bad/id"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                RW.detail(name, self.root)
        altered = copy.deepcopy(self.case)
        altered["source_row"]["net"] = 99
        (self.root / "cases" / f"{self.case_id}.json").write_bytes(RW.encoded(altered))
        with self.assertRaisesRegex(ValueError, "校验失败"):
            RW.detail(self.case_id, self.root)

    def test_http_roundtrip_and_export_headers(self):
        from mainwave_visual_app import app
        original_detail, original_load, original_save, original_archive = RW.detail, RW.load_review, RW.save_review, RW.archive
        # Route defaults use the real archive; wrappers isolate HTTP tests completely.
        with patch.object(RW, "detail", side_effect=lambda cid, root=None: original_detail(cid, self.root)), \
             patch.object(RW, "load_review", side_effect=lambda cid, root=None: original_load(cid, self.root)), \
             patch.object(RW, "save_review", side_effect=lambda cid, body: original_save(cid, body, self.root)), \
             patch.object(RW, "archive", side_effect=lambda cid, kind: original_archive(cid, kind, self.root)):
            client = app.test_client()
            path = f"/api/reviews/{self.case_id}"
            self.assertEqual(client.get(path).status_code, 200)
            self.assertEqual(client.post(path, json=self.body()).status_code, 200)
            self.assertEqual(client.post(path, json=self.body()).status_code, 409)
            self.assertEqual(client.post(path, json=self.body(), headers={"Origin": "https://elsewhere.example"}).status_code, 403)
            self.assertEqual(client.post(path, data="not json").status_code, 400)
            self.assertEqual(client.post(path, json=[1]).status_code, 400)
            output = client.get(path + "/export/review")
            self.assertEqual(output.status_code, 200)
            self.assertIn("attachment", output.headers["Content-Disposition"])
            self.assertEqual(output.mimetype, "application/zip")


if __name__ == "__main__":
    unittest.main()
