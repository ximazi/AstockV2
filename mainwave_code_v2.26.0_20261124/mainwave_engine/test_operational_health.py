import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from . import etf_tracker as ET, pipeline
from .fupan import __main__ as CLI
from .trading import jev
from .test_etf_tracker import fixture


class JevProxyTests(unittest.TestCase):
    def test_proxy_precedence_and_explicit_routing(self):
        response = Mock(status_code=200)
        response.json.return_value = {"answers": {}}
        with patch("dotenv.load_dotenv"), patch.dict(os.environ, {"JEV_PROXY": "http://jev:7890", "FUPAN_PROXY": "http://other:7890"}, clear=True), \
                patch("requests.post", return_value=response) as post:
            client = jev.Client(key="test-secret")
            self.assertEqual(client.proxy, "http://jev:7890")
            client.ask({}, {})
            self.assertEqual(post.call_args.kwargs["proxies"], {"http": "http://jev:7890", "https": "http://jev:7890"})
            self.assertEqual(jev.Client(key="test", proxy="http://explicit:8888").proxy, "http://explicit:8888")
            self.assertNotIn("HTTP_PROXY", os.environ)

    def test_fupan_proxy_fallback(self):
        with patch("dotenv.load_dotenv"), patch.dict(os.environ, {"FUPAN_PROXY": "http://fallback:7890"}, clear=True):
            self.assertEqual(jev.Client(key="test").proxy, "http://fallback:7890")

    def test_service_denial_circuits_and_redacts_key(self):
        response = Mock(status_code=451, text="test-secret: unavailable")
        with patch("requests.post", return_value=response) as post:
            client = jev.Client(key="test-secret", proxy="http://local:7890")
            for _ in range(2):
                with self.assertRaises(jev.JevError) as ctx:
                    client.ask({}, {})
                self.assertNotIn("test-secret", str(ctx.exception))
            self.assertEqual(post.call_count, 1)

    def test_network_error_does_not_expose_proxy_credentials(self):
        with patch("requests.post", side_effect=requests.ConnectionError("http://user:password@local")):
            with self.assertRaises(jev.JevError) as ctx:
                jev.Client(key="test", proxy="http://local").ask({}, {})
        self.assertNotIn("password", str(ctx.exception))


class ETFRetryTests(unittest.TestCase):
    def test_transient_failure_recovers(self):
        daily, adj = fixture()
        pro = Mock()
        pro.fund_daily.side_effect = [requests.ConnectionError("secret"), daily]
        pro.fund_adj.return_value = adj
        pro.fund_nav.return_value = daily.rename(columns={"trade_date": "nav_date"}).assign(ann_date="20261027")
        with tempfile.TemporaryDirectory() as tmp, patch.object(ET.time, "sleep") as sleep:
            errors = ET.refresh_one(ET.registry()["items"][0], "2026-10-27", pro, Path(tmp))
        self.assertEqual(errors, [])
        self.assertEqual(pro.fund_daily.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_retry_is_bounded_and_nontransport_errors_not_retried(self):
        pro = Mock()
        pro.fund_daily.side_effect = requests.Timeout("secret")
        pro.fund_adj.side_effect = RuntimeError("secret")
        pro.fund_nav.side_effect = ValueError("secret")
        with tempfile.TemporaryDirectory() as tmp, patch.object(ET.time, "sleep"):
            errors = ET.refresh_one(ET.registry()["items"][0], "2026-10-27", pro, Path(tmp))
        self.assertEqual(pro.fund_daily.call_count, 3)
        self.assertEqual(pro.fund_adj.call_count, 1)
        self.assertEqual(len(errors), 3)
        self.assertNotIn("secret", str(errors))


class DeliveryTests(unittest.TestCase):
    def test_ai_failure_preserves_rule_report_but_marks_partial(self):
        pointer = {"version": "v001", "reused": False, "mode": "FROZEN_EOD", "stocks": 0, "issues": 0, "required_symbols": []}
        for ai_status, expected in (("FAILED", "PARTIAL"), ("ALREADY_MERGED", "SUCCESS"), ("MERGED", "SUCCESS")):
            with tempfile.TemporaryDirectory() as tmp, patch("sys.argv", ["fupan", "--date", "2026-10-27", "--out-root", tmp, "--analyze"]), \
                    patch.object(CLI, "build", return_value=({}, {})), patch.object(CLI, "persist", return_value=pointer), \
                    patch("mainwave_engine.fupan.analyzer.run", return_value={"status": ai_status}):
                self.assertEqual(CLI.main(), 0)
                status = json.loads((Path(tmp) / "20261027/fupan/status.json").read_text(encoding="utf-8"))
                self.assertEqual(status["status"], expected)
                self.assertTrue(status["rule_report_available"])

    def test_jev_and_data_degradation_reach_pipeline_status(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, "OUT_ROOT", Path(tmp)):
            output = Path(tmp) / "20261027"
            output.mkdir()
            def run(command, check=False):
                if command[2] == "mainwave_engine.trading.daily":
                    (output / "jev_shadow.json").write_text(json.dumps({"as_of": "2026-10-27", "status": "UNAVAILABLE"}), encoding="utf-8")
                    (output / "closure_preflight.json").write_text(json.dumps({"as_of": "2026-10-27", "status": "DEGRADED"}), encoding="utf-8")
                if command[2] in ("mainwave_engine.eod_review", "mainwave_engine.limit_theme", "mainwave_engine.fupan"):
                    name = command[2].split(".")[-1]
                    pipeline._write(output / name / "status.json", {"as_of": "2026-10-27", "status": "SUCCESS"})
                if command[2] == "mainwave_engine.board_relay":
                    pipeline._write(output / "board_relay/status.json", {"as_of": "2026-10-27", "status": "PARTIAL",
                                                                        "gaps": ["LIMIT_TIMES"]})
                return SimpleNamespace(returncode=0)
            with patch.object(pipeline.subprocess, "run", side_effect=run):
                self.assertEqual(pipeline.run("2026-10-27"), 0)
            status = json.loads((output / "mainwave_run_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "SUCCESS")
            self.assertEqual(status["delivery_status"], "DEGRADED")
            self.assertEqual(len(status["warnings"]), 2)
            relay = next(r for r in status["steps"] if r["name"] == "board_relay")
            self.assertEqual((relay["review_status"], relay["gaps"]), ("PARTIAL", ["LIMIT_TIMES"]))
