"""Attribution is evidence analysis, not static-label inference or order authority."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from flask import Flask

from . import eod_review as R, hotspot_core as H, intraday_quotes as Q, pipeline
from .limit_theme import review as L, sources as S
from .trading import store, web

DAY = "2026-10-28"
NOW = DAY + "T21:00:00+08:00"
SYMBOL = "sz_000678"


def material(identity, kind, origin="issuer"):
    return {"id": identity, "kind": kind, "origin": origin, "origin_group": origin,
            "title": "测试报道", "text": "襄阳轴承的机器人相关业务仍在拓展阶段，收入规模较小。", "url": "https://example.com/source",
            "published_at": DAY + "T16:00:00+08:00", "collected_at": NOW}


def packet():
    return {"as_of": DAY, "symbol": SYMBOL, "name": "襄阳轴承", "height": 5,
            "legacy_concepts": ["汽车配件"], "coverage": {}, "gaps": [], "knowledge_cutoff": NOW,
            "materials": [material("M1", "MEDIA_BODY", "财联社"), material("M2", "MEDIA_BODY", "证券时报"),
                          material("A1", "ISSUER_BODY"), material("L1", "LHB")]}


def answer(p):
    cite = lambda identity: {"evidence_id": identity, "quote": "机器人相关业务仍在拓展阶段"}
    return {"symbol": p["symbol"], "primary_theme": "机器人", "status": "CORROBORATED",
            "hypotheses": [{"theme": "机器人", "relation_kind": "MARKET_NARRATIVE", "mechanism": "市场叙事，不代表主营",
                            "support": [cite("M1"), cite("M2")], "counter": []}],
            "business_reality": {"status": "LIMITED", "summary": "业务有限", "citations": [cite("A1")]},
            "lhb_reading": {"summary": "交易背景不能证明动机", "citations": [cite("L1")]},
            "unknowns": ["规模订单未知"], "next_checks": ["核对后续公告"]}


class ValidationTest(unittest.TestCase):
    def test_new_theme_can_differ_from_old_tags_without_mutation(self):
        p = packet()
        original = copy.deepcopy(p)
        out = L.check(answer(p), p)
        self.assertEqual(out["status"], "CORROBORATED")
        self.assertNotIn(out["primary_theme"], p["legacy_concepts"])
        self.assertEqual(p, original)

    def test_reprints_summaries_and_lhb_do_not_create_confirmation(self):
        for kind in ("MEDIA_SUMMARY", "LHB", "VENDOR_REASON"):
            p = packet()
            p["materials"][1]["kind"] = kind
            out = L.check(answer(p), p)
            self.assertEqual(out["status"], "INSUFFICIENT")
            self.assertTrue(out["verification"]["host_downgraded"])
        p = packet()
        p["materials"][1]["origin_group"] = p["materials"][0]["origin_group"]
        self.assertEqual(L.check(answer(p), p)["status"], "INSUFFICIENT")

    def test_issuer_body_and_media_can_cross_check_but_title_cannot_prove_business(self):
        p = packet()
        out = answer(p)
        out["hypotheses"][0]["support"][1]["evidence_id"] = "A1"
        self.assertEqual(L.check(out, p)["status"], "CORROBORATED")
        p["materials"][2]["kind"] = "ISSUER_TITLE"
        with self.assertRaisesRegex(ValueError, "issuer original"):
            L.check(out, p)

    def test_fake_quote_unknown_id_and_trading_fields_are_rejected(self):
        for change in ("quote", "id", "trade"):
            p = packet()
            out = answer(p)
            if change == "quote":
                out["hypotheses"][0]["support"][0]["quote"] = "根本不存在的已量产事实"
            elif change == "id":
                out["hypotheses"][0]["support"][0]["evidence_id"] = "invented"
            else:
                out["target_price"] = 100
            with self.subTest(change=change), self.assertRaises(ValueError):
                L.check(out, p)

    def test_render_fields_and_business_relation_require_valid_evidence(self):
        p = packet()
        for change in ("business", "lhb", "hypothesis", "relation"):
            out = answer(p)
            if change == "business":
                out["business_reality"].pop("summary")
            elif change == "lhb":
                out["lhb_reading"].pop("summary")
            elif change == "hypothesis":
                out["hypotheses"] = [None]
            else:
                out["hypotheses"][0]["relation_kind"] = "CONFIRMED_BUSINESS"
            with self.subTest(change=change), self.assertRaises(ValueError):
                L.check(out, p)

    def test_render_escapes_external_material(self):
        p = packet()
        out = L.check(answer(p), p)
        out.update(name="<script>alert(1)</script>", height=5, legacy_concepts=[], gaps=[])
        page = L.render({"as_of": DAY, "notice": "test", "status": "SUCCESS", "scope": {"universe": 1},
                         "summary": {}, "known_at": NOW, "mode": "SAME_DAY_REVIEW", "rows": [out], "materials": {SYMBOL: p["materials"]}})
        self.assertNotIn("<script>alert(1)</script>", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertIn("原文与待核", page)


class CacheTest(unittest.TestCase):
    def test_bad_batch_isolated_and_only_validated_rows_cached(self):
        ps = [{**packet(), "symbol": s} for s in ("a", "b", "c")]
        called = []
        def client(system, user, cfg):
            group = json.loads(user)["packets"]
            called.append([p["symbol"] for p in group])
            if len(group) > 1 or group[0]["symbol"] == "b":
                return "{}{}", {}
            return json.dumps({"rows": [answer(group[0])]}), {}
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {"provider":"codex-cli", "model":"test"}
            rows, meta = L.analyze(ps, Path(tmp), cfg, client)
            self.assertEqual(called, [["a","b","c"],["a"],["b"],["c"]])
            self.assertEqual([r["processing"] for r in rows], ["VALIDATED","MODEL_FAILED","VALIDATED"])
            self.assertEqual((meta["single_retries"],len(meta["errors"])), (3,1))
            called.clear()
            _, meta = L.analyze(ps, Path(tmp), cfg, client)
            self.assertEqual((called,meta["cache_hits"]), ([["b"]],2))

    def test_missing_row_does_not_requery_siblings(self):
        ps = [{**packet(),"symbol":s} for s in ("a","b")]
        called = []
        def client(system,user,cfg):
            group = json.loads(user)["packets"]
            called.append([p["symbol"] for p in group])
            return json.dumps({"rows":[answer(group[0])]}), {}
        with tempfile.TemporaryDirectory() as tmp:
            rows,_ = L.analyze(ps,Path(tmp),{"provider":"codex-cli","model":"test"},client)
        self.assertEqual(called, [["a","b"],["b"]])
        self.assertTrue(all(r["status"]=="CORROBORATED" for r in rows))

    def test_source_failure_is_cached_and_can_retry_after_ttl(self):
        with tempfile.TemporaryDirectory() as temp:
            c = S.Collector(temp, DAY, now=NOW)
            calls = []
            def fail():
                calls.append(1)
                raise ValueError("no permission")
            self.assertEqual(c.cache("news", fail)["status"], "UNAVAILABLE")
            c.cache("news", fail)
            self.assertEqual(len(calls), 1)
            c.now += Q.pd.Timedelta(minutes=16)
            c.cache("news", fail)
            self.assertEqual(len(calls), 2)

    def test_model_resume_revalidates_and_does_not_call_twice(self):
        p = packet()
        with tempfile.TemporaryDirectory() as temp:
            client = lambda s, u, cfg: (json.dumps({"rows": [answer(p)]}), {"input_tokens": 10})
            with patch.object(L.model_cli, "call", side_effect=client) as model:
                first, _ = L.analyze([p], Path(temp), {"provider": "codex-cli", "model": "test"})
                p["knowledge_cutoff"] = "2026-10-28T21:05:00+08:00"
                p["publication_cutoff"] = "2026-10-28T21:05:00+08:00"
                for m in p["materials"]:
                    m["collected_at"] = p["knowledge_cutoff"]
                again, meta = L.analyze([p], Path(temp), {"provider": "codex-cli", "model": "test"})
            self.assertEqual(model.call_count, 1)
            self.assertEqual(meta["cache_hits"], 1)
            self.assertEqual(first, again)

    def test_invalid_model_coverage_or_timeout_keeps_unknown_stock(self):
        for response in ('{"rows":[]}', '{"rows":[{"symbol":"sh_600001"}]}'):
            with tempfile.TemporaryDirectory() as temp:
                rows, meta = L.analyze([packet()], Path(temp), {"provider": "codex-cli", "model": "test"},
                                       lambda s, u, cfg: (response, {}))
                self.assertEqual(rows[0]["status"], "INSUFFICIENT")
                self.assertTrue(meta["errors"])

    def test_contract_error_has_at_most_one_repair_and_keeps_other_rows(self):
        p = packet()
        bad = answer(p)
        bad["primary_theme"] = "a paragraph rather than a theme"
        calls = []
        def model(s, u, cfg):
            calls.append(json.loads(u))
            return json.dumps({"rows": [bad if len(calls) == 1 else answer(p)]}), {}
        with tempfile.TemporaryDirectory() as temp:
            rows, meta = L.analyze([p], Path(temp), {"provider": "codex-cli", "model": "test"}, model)
        self.assertEqual((len(calls), meta["repairs"]), (2, 1))
        self.assertEqual(rows[0]["status"], "CORROBORATED")
        self.assertIn("validation_error", calls[1])

    def test_repeated_provider_failure_opens_bounded_circuit(self):
        with tempfile.TemporaryDirectory() as temp:
            c = S.Collector(temp, DAY, now=NOW)
            calls = []
            def fail():
                calls.append(1)
                raise ValueError("provider unavailable")
            for i in range(10):
                result = c.cache(["media", str(i)], fail)
            self.assertEqual(len(calls), 3)
            self.assertIn("CIRCUIT", result["error"])

    def test_no_external_material_does_not_infer_from_static_tags(self):
        p = packet()
        p["materials"] = [material("L1", "LHB")]
        with tempfile.TemporaryDirectory() as temp, patch.object(L.model_cli, "call") as model:
            rows, _ = L.analyze([p], Path(temp), {"provider": "codex-cli", "model": "test"})
        model.assert_not_called()
        self.assertEqual(rows[0]["primary_theme"], "待核")


class CollectorTest(unittest.TestCase):
    def test_future_article_is_not_queried_and_empty_global_lhb_is_not_not_on_list(self):
        with tempfile.TemporaryDirectory() as temp:
            c = S.Collector(temp, DAY, now=NOW)
            future = {"status": "AVAILABLE", "data": [{"title": "襄阳轴承涨停", "content": "机器人", "date": "2026-10-29 09:30:00", "mediaName": "财联社", "code": "1"}]}
            market = {a: {"status": "AVAILABLE", "data": [], "collected_at": NOW} for a in ("top_list", "top_inst")}
            with patch.object(c, "news", return_value=future), patch.object(c, "article") as article, \
                    patch.object(c, "announcements", return_value={"status": "AVAILABLE", "data": {"records": []}}):
                p = c.packet({"symbol": SYMBOL, "name": "襄阳轴承"}, market)
            article.assert_not_called()
            self.assertEqual(p["coverage"]["top_list"], "EMPTY_UNVERIFIED")
            self.assertEqual(p["materials"], [])

    def test_material_rejects_future_publication_and_unapproved_urls(self):
        with tempfile.TemporaryDirectory() as temp:
            c = S.Collector(temp, DAY, now=NOW)
            with self.assertRaises(ValueError):
                c.material("MEDIA_BODY", "财联社", "title", "body", "url", "2026-10-29 09:00:00")
            for url in ("http://127.0.0.1", "https://evil.example/", "file:///secret"):
                with self.subTest(url=url), self.assertRaises(ValueError):
                    c.request("GET", url)

    def test_future_filing_is_not_downloaded(self):
        with tempfile.TemporaryDirectory() as temp:
            c = S.Collector(temp, DAY, now=NOW)
            record = {"secCode": "000678", "announcementTitle": "future filing",
                      "announcementTime": int(Q.stamp("2026-10-29T00:00:00+08:00").timestamp() * 1000),
                      "adjunctUrl": "finalpage/2026-10-29/100.PDF"}
            with patch.object(c, "news", return_value={"status": "AVAILABLE", "data": []}), \
                    patch.object(c, "announcements", return_value={"status": "AVAILABLE", "data": {"records": [record]}}), \
                    patch.object(c, "filing") as filing:
                c.packet({"symbol": SYMBOL, "name": SYMBOL}, {})
            filing.assert_not_called()

    def test_reprint_group_and_business_material_levels(self):
        with tempfile.TemporaryDirectory() as temp:
            c = S.Collector(temp, DAY, now=NOW)
            rows = [{"title": "襄阳轴承涨停", "content": "襄阳轴承", "date": NOW, "mediaName": origin, "code": str(i)} for i, origin in enumerate(("财联社", "证券时报"))]
            with patch.object(c, "news", return_value={"status": "AVAILABLE", "data": rows, "collected_at": NOW}), \
                    patch.object(c, "article", return_value={"status": "AVAILABLE", "collected_at": NOW, "data": {"text": "襄阳轴承机器人概念涨停，相关业务处于拓展阶段。", "publisher": "UNKNOWN"}}), \
                    patch.object(c, "announcements", return_value={"status": "AVAILABLE", "data": {"records": []}}):
                p = c.packet({"symbol": SYMBOL, "name": "襄阳轴承"}, {})
            self.assertEqual(len(p["materials"]), 2)
            self.assertEqual(p["materials"][0]["origin_group"], p["materials"][1]["origin_group"])
            self.assertIn("reprint_of", p["materials"][1])


class RunnerTest(unittest.TestCase):
    def seed(self, root, universe):
        source = root / "facts"
        R._write(source / "facts.json", {"as_of": DAY, "boards": {"U": universe}, "ladder": {"heights": {s: 1 for s in universe}}})
        R._write(root / "20261028/eod_review/latest.json", {"path": str(source)})

    def test_full_coverage_and_partial_scope_never_publish_false_full_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.seed(root, [SYMBOL, "sh_600001"])
            class Fake:
                def market_sources(self): return {}
                def packet(self, stock, market): return {**packet(), **stock}
            def model(s, u, cfg):
                ps = json.loads(u)["packets"]
                return json.dumps({"rows": [answer(p) for p in ps]}), {}
            report = L.run(DAY, root, collector=Fake(), cfg={"provider": "codex-cli", "model": "test"}, client=model, now=NOW)
            self.assertEqual(len(report["rows"]), 2)
            self.assertTrue((root / "20261028/theme_attribution.json").exists())
            original = (root / "20261028/theme_attribution.json").read_bytes()
            L.run(DAY, root, [SYMBOL], collector=Fake(), cfg={"provider": "off", "model": "test"}, now=NOW)
            self.assertEqual((root / "20261028/theme_attribution.json").read_bytes(), original)
            self.assertFalse(report["orders_enabled"])

    def test_pipeline_runs_attribution_between_facts_and_fupan_without_blocking_core(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(pipeline, "OUT_ROOT", Path(temp)), \
                patch.object(pipeline.subprocess, "run") as launch:
            launch.return_value.returncode = 0
            pipeline.run(DAY)
            modules = [call.args[0][2] for call in launch.call_args_list]
            self.assertLess(modules.index("mainwave_engine.eod_review"), modules.index("mainwave_engine.limit_theme"))
            self.assertLess(modules.index("mainwave_engine.limit_theme"), modules.index("mainwave_engine.fupan"))

    def test_web_read_does_not_invoke_network_or_model(self):
        app = Flask(__name__)
        app.register_blueprint(web.bp)
        with tempfile.TemporaryDirectory() as temp, patch.object(store, "ROOT", Path(temp)), patch.object(L.model_cli, "call") as model:
            client = app.test_client()
            self.assertEqual(client.get("/api/trading/limit-theme?as_of=" + DAY).get_json()["status"], "NO_REPORT")
            self.assertEqual(client.get("/limit-themes?as_of=" + DAY).status_code, 404)
            model.assert_not_called()

    def test_web_serves_frozen_report_without_recalculation(self):
        app = Flask(__name__)
        app.register_blueprint(web.bp)
        with tempfile.TemporaryDirectory() as temp, patch.object(store, "ROOT", Path(temp)), patch.object(L.model_cli, "call") as model:
            target = Path(temp) / "results/mainwave_engine/20261028/limit_theme/v001"
            R._write(target.parent / "latest.json", {"path": str(target)})
            R._write(target / "report.json", {"as_of": DAY, "rows": [], "status": "PARTIAL"})
            (target / "report.html").write_text("<!doctype html><title>Frozen attribution</title>", encoding="utf-8")
            client = app.test_client()
            self.assertEqual(client.get("/api/trading/limit-theme?as_of=" + DAY).get_json()["status"], "PARTIAL")
            response = client.get("/limit-themes?as_of=" + DAY)
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"Frozen attribution", response.data)
            self.assertEqual(client.get("/limit-themes?as_of=not-a-date").status_code, 400)
            model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
