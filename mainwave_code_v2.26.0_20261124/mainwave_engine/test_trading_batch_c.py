"""Batch C invariants: AI evidence validation, offline determinism, intraday linkage."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from flask import Flask

from .trading import assessment, intraday_link, modes, runner, store, web
from .test_trading_system import AT, DAY, payload

VALID = {"verdict": "SUPPORT", "supporting_evidence": [{"evidence_id": "E01"}],
         "counter_evidence": [], "unknowns": [], "alternative_explanations": [],
         "mode_fit": "TREND_PULLBACK", "next_review_event": "次日开盘复核",
         "expires_at": "2026-09-19T14:00:00+08:00"}
NOW = "2026-09-18T18:00:00+08:00"


def pack_for(snapshot, record=None):
    record = record or snapshot["records"][0]
    return assessment.bundle(record, snapshot, {"cards": []}, modes.evaluations(record))


class BundleTest(unittest.TestCase):
    def setUp(self):
        self.snap = payload()
        self.snap["as_of"] = DAY

    def test_every_item_is_addressable_and_dated(self):
        pack = pack_for(self.snap)
        self.assertTrue(pack["evidence"])
        for item in pack["evidence"]:
            self.assertRegex(item["id"], r"^E\d{2}$")
            self.assertEqual(item["as_of"], DAY)
            self.assertTrue(item["source"])

    def test_gaps_are_listed_rather_than_filled_in(self):
        pack = pack_for(self.snap)
        self.assertIn("路径诊断卡", pack["missing"])

    def test_bundle_carries_the_injection_policy_and_no_credentials(self):
        pack = pack_for(self.snap)
        self.assertEqual(pack["injection_policy"], assessment.INJECTION_NOTICE)
        self.assertNotIn("token", store.encoded(pack).lower())
        self.assertIn("不提供任何交易密钥", pack["account_risk_summary"])

    def test_same_facts_produce_the_same_hash(self):
        self.assertEqual(pack_for(self.snap)["evidence_hash"], pack_for(self.snap)["evidence_hash"])


class ValidationTest(unittest.TestCase):
    def setUp(self):
        self.snap = payload()
        self.snap["as_of"] = DAY
        self.pack = pack_for(self.snap)

    def check(self, output, at=NOW):
        return assessment.validate(output, self.pack, at)

    def test_well_formed_output_passes(self):
        self.assertTrue(self.check(VALID)["valid"])

    def test_forged_citation_is_rejected(self):
        result = self.check({**VALID, "supporting_evidence": [{"evidence_id": "E99"}]})
        self.assertFalse(result["valid"])
        self.assertTrue(any("引用不存在" in p for p in result["problems"]))

    def test_future_dated_evidence_is_rejected(self):
        pack = dict(self.pack)
        pack["evidence"] = [{**pack["evidence"][0], "as_of": "2026-09-30"}]
        result = assessment.validate(VALID, pack, NOW)
        self.assertFalse(result["valid"])
        self.assertTrue(any("超出时点" in p for p in result["problems"]))

    def test_uncalibrated_probability_field_is_refused(self):
        result = self.check({**VALID, "win_rate": .7})
        self.assertFalse(result["valid"])
        self.assertTrue(any("禁止字段" in p for p in result["problems"]))

    def test_conclusion_without_any_citation_is_refused(self):
        result = self.check({**VALID, "supporting_evidence": [], "counter_evidence": []})
        self.assertFalse(result["valid"])
        self.assertTrue(any("未引用" in p for p in result["problems"]))

    def test_expired_conclusion_is_not_valid(self):
        result = self.check({**VALID, "expires_at": "2026-09-18T09:00:00+08:00"})
        self.assertEqual(result["status"], "EXPIRED")

    def test_missing_expiry_on_a_directional_verdict_is_refused(self):
        result = self.check({k: v for k, v in VALID.items() if k != "expires_at"})
        self.assertFalse(result["valid"])

    def test_over_long_validity_is_refused(self):
        result = self.check({**VALID, "expires_at": "2026-09-30T09:00:00+08:00"})
        self.assertFalse(result["valid"])
        self.assertTrue(any("有效期超过" in p for p in result["problems"]))

    def test_malformed_structure_is_refused(self):
        self.assertEqual(self.check(["not", "a", "dict"])["status"], "INVALID_STRUCTURE")

    def test_illegal_verdict_is_refused(self):
        self.assertFalse(self.check({**VALID, "verdict": "BUY_NOW"})["valid"])


class OfflineTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "t.sqlite3"
        self.snap = payload()
        self.snap["as_of"] = DAY
        self.pack = pack_for(self.snap)

    def tearDown(self):
        self.temp.cleanup()

    def test_offline_is_deterministic_unknown(self):
        result = assessment.assess(self.pack, db=self.db, at=NOW)
        self.assertEqual(result["assessment"]["verdict"], "UNKNOWN")
        self.assertEqual(result["assessment"]["model_version"], assessment.OFFLINE)

    def test_result_is_cached_by_evidence_hash(self):
        first = assessment.assess(self.pack, db=self.db, at=NOW)
        second = assessment.assess(self.pack, db=self.db, at=NOW)
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(first["output_hash"], second["output_hash"])

    def test_timeout_degrades_to_unknown_and_records_the_reason(self):
        def slow(pack, **_):
            raise TimeoutError()
        result = assessment.assess(self.pack, slow, db=self.db, at=NOW, model="slow")
        self.assertEqual(result["status"], "TIMEOUT")
        self.assertEqual(result["assessment"]["verdict"], "UNKNOWN")

    def test_invalid_client_output_degrades_instead_of_being_trusted(self):
        def bad(pack, **_):
            return {**VALID, "supporting_evidence": [{"evidence_id": "E99"}]}
        result = assessment.assess(self.pack, bad, db=self.db, at=NOW, model="bad")
        self.assertEqual(result["assessment"]["verdict"], "UNKNOWN")
        self.assertEqual(result["validation"]["applied"], "DEGRADED_TO_UNKNOWN")

    def test_run_metadata_is_recorded(self):
        result = assessment.assess(self.pack, db=self.db, at=NOW)
        for key in ("model", "prompt_version", "input_hash", "output_hash", "latency_seconds", "cost"):
            self.assertIn(key, result)

    def test_gate_only_pauses_entries_when_ai_was_declared_required(self):
        result = assessment.assess(self.pack, db=self.db, at=NOW)
        self.assertTrue(assessment.gate(result, required=False)["new_entry_allowed"])
        gated = assessment.gate(result, required=True)
        self.assertFalse(gated["new_entry_allowed"])
        self.assertTrue(gated["position_risk_continues"])

    def test_shadow_keeps_rules_assisted_and_veto_apart(self):
        record = self.snap["records"][0]
        stored = [{"opportunity_id": record["opportunity_id"],
                   "assessment": {"verdict": "AGAINST"}}]
        result = assessment.shadow(self.snap["records"], stored)
        self.assertTrue(result["pure_rules"])
        self.assertNotIn(record["symbol"], [r["symbol"] for r in result["ai_veto"]])
        self.assertIn(record["symbol"], [r["symbol"] for r in result["vetoed_from_top"]])
        self.assertTrue(result["disagreement"])


class IntradayLinkTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / "t.sqlite3"
        with patch.object(store, "now", return_value=AT):
            self.snap = store.persist(payload(), self.db)

    def tearDown(self):
        self.temp.cleanup()

    def write_report(self, records):
        folder = self.root / "intraday" / DAY.replace("-", "")
        folder.mkdir(parents=True, exist_ok=True)
        review = folder / "review.json"
        review.write_text(json.dumps({"as_of": DAY, "observed_at": f"{DAY}T14:35:00+08:00",
                                      "records": records, "market": {"state": "REVIEW_ONLY"}}),
                          encoding="utf-8")
        (folder / "latest.json").write_text(json.dumps({"as_of": DAY, "review": str(review)}), encoding="utf-8")
        return self.root / "intraday"

    def test_modes_stay_separate_and_intraday_is_never_executable(self):
        root = self.write_report([])
        result = intraday_link.link(DAY, self.db, root)
        row = result["records"][0]
        kinds = {m["mode"] for m in row["modes"]}
        self.assertEqual(kinds, {modes.TREND_PULLBACK, modes.STRENGTH_CONFIRM, modes.OVERNIGHT})
        for entry in row["modes"]:
            if entry["source"] == "INTRADAY_1430":
                self.assertFalse(entry["executable"])
                self.assertTrue(entry["provisional"])

    def test_data_wait_intraday_is_not_counted_as_a_difference(self):
        symbol = self.snap["records"][0]["symbol"]
        root = self.write_report([{"symbol": symbol, "status": "DATA_WAIT", "quote": None}])
        result = intraday_link.link(DAY, self.db, root)
        self.assertEqual(result["summary"]["differences"], 0)
        self.assertEqual(result["summary"]["comparable"], 0)

    def test_comparable_disagreement_is_recorded_with_close_as_authority(self):
        symbol = self.snap["records"][0]["symbol"]
        root = self.write_report([{"symbol": symbol, "status": "SELL_REVIEW",
                                   "quote": {"close": 10.0, "quality": "FULL"}}])
        result = intraday_link.link(DAY, self.db, root)
        self.assertEqual(result["summary"]["comparable"], 1)
        row = next(r for r in result["records"] if r["symbol"] == symbol)
        if row.get("difference"):
            self.assertIn("以收盘为准", row["difference"]["reading"])

    def test_missing_closing_snapshot_refuses_to_stand_alone(self):
        empty = Path(self.temp.name) / "empty.sqlite3"
        result = intraday_link.link(DAY, empty, self.write_report([]))
        self.assertEqual(result["status"], "NO_CLOSING_SNAPSHOT")

    def test_repeat_1430_schedule_does_not_run_twice(self):
        root = self.write_report([])
        first = intraday_link.session(DAY, self.db, root, at=AT)
        self.assertEqual(first["status"], "READY")
        self.assertEqual(first["risk_first"][0], "PENDING_EXITS")
        # Another live process holds the lock; this process must not evaluate again.
        with patch.object(runner, "owner", return_value="other:2"):
            runner.acquire(f"intraday:{DAY}", self.db, at=AT)
        second = intraday_link.session(DAY, self.db, root, at=AT)
        self.assertEqual(second["status"], "ANOTHER_RUN_IN_PROGRESS")
        self.assertEqual(second["lock"]["owner"], "other:2")

    def test_session_records_a_resumable_checkpoint(self):
        intraday_link.session(DAY, self.db, self.write_report([]), at=AT)
        self.assertTrue(runner.completed(DAY, intraday_link.STAGE, "link", self.db))


class BatchCWebTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "t.sqlite3"
        with patch.object(store, "now", return_value=AT):
            self.snap = store.persist(payload(), self.db)
        self.app = Flask(__name__, template_folder=str(store.ROOT / "templates"))
        self.app.config["TRADING_DB"] = self.db
        self.app.register_blueprint(web.bp)

    def tearDown(self):
        self.temp.cleanup()

    def test_assessment_endpoints_answer_offline(self):
        client = self.app.test_client()
        identity = self.snap["records"][0]["opportunity_id"]
        body = client.get(f"/api/trading/assessment/{identity}").get_json()
        self.assertIn("bundle", body)
        self.assertEqual(body["injection_policy"], assessment.INJECTION_NOTICE)
        with patch.object(assessment.model_cli, "config", return_value={"provider": "off"}):
            result = client.post(f"/api/trading/assessment/{identity}",
                                 json={}, headers={"Origin": "http://localhost"})
        self.assertEqual(result.get_json()["assessment"]["verdict"], "UNKNOWN")

    def test_only_explicit_assessment_post_calls_configured_model(self):
        client = self.app.test_client()
        identity = self.snap["records"][0]["opportunity_id"]
        with patch.object(assessment, "configured_assess", return_value={"status": "OK"}) as configured:
            self.assertEqual(client.get(f"/api/trading/assessment/{identity}").status_code, 200)
            configured.assert_not_called()
            result = client.post(f"/api/trading/assessment/{identity}", json={}, headers={"Origin": "http://localhost"})
            self.assertEqual(result.get_json()["status"], "OK")
            configured.assert_called_once()

    def test_intraday_endpoint_reports_without_an_intraday_run(self):
        body = self.app.test_client().get("/api/trading/intraday").get_json()
        self.assertIn(body["status"], {"NO_INTRADAY_RUN", "READY", "NO_SNAPSHOT"})


if __name__ == "__main__":
    unittest.main()
