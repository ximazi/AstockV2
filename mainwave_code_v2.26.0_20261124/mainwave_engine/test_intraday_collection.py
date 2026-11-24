"""Account throttling, fallback coverage and price-only UI regression tests."""
import os
import tempfile
import unittest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch

from . import intraday_quotes as Q, intraday_review as R


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.clock = [1000.]
        for target, value in (("monotonic", lambda: self.clock[0]), ("sleep", self.sleep)):
            p = patch.object(Q.time, target, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch.dict(os.environ, TUSHARE_TOKEN="test-token")
        p.start()
        self.addCleanup(p.stop)
        self.client = Q.Client()
        self.addCleanup(self.client.session.close)
        self.requests = []
        self.rate_failures = 0
        self.truncate = False
        self.client.session.post = Mock(side_effect=self.respond)
        self.at = Q.stamp("2026-09-18 14:35")

    def sleep(self, seconds):
        self.clock[0] += seconds

    def respond(self, url, json, **kwargs):
        api = json["api_name"]
        self.requests.append((self.clock[0], api, json["params"]))
        response = Mock()
        if api != "rt_min":
            payload = {"code": 40203, "msg": "没有接口访问权限"}
        elif self.rate_failures:
            self.rate_failures -= 1
            payload = {"code": 40203, "msg": "访问接口频率超限(5次/分钟)"}
        else:
            codes = json["params"].get("ts_code", "600000.SH").split(",")
            if self.truncate:
                codes = codes[:-1]
            payload = {"code": 0, "data": {"fields": ["ts_code", "time", "close"],
                       "items": [[c, "2026-09-18 14:35:00", 10.] for c in codes]}}
        response.json.return_value = payload
        return response

    def test_sixth_minute_call_waits_for_rolling_window(self):
        for _ in range(6):
            self.client.query("rt_min")
        times = [t for t, _, _ in self.requests]
        self.assertGreaterEqual(times[5] - times[0], 61)

    def test_deadline_prevents_request_after_required_wait(self):
        for _ in range(5):
            self.client.query("rt_min")
        with self.assertRaises(Q.ApiError) as error:
            self.client.query("rt_min", deadline=1020)
        self.assertEqual(error.exception.code, "BUDGET")
        self.assertEqual(len(self.requests), 5)

    def test_rate_limit_retries_and_is_not_permanent_denial(self):
        self.rate_failures = 1
        result = Q.collect(self.client, ["sh_600000"], [], self.at)
        self.assertEqual(len(result["raw"]), 1)
        calls = [t for t, api, _ in self.requests if api == "rt_min"]
        self.assertEqual(len(calls), 2)
        self.assertGreaterEqual(calls[1] - calls[0], 61)
        self.assertNotIn("rt_min", self.client.denied)
        self.assertIn("rt_k", self.client.denied)
        self.client.query("rt_min")

    def test_rate_limit_cannot_force_retry_past_budget(self):
        self.rate_failures = 1
        result = Q.collect(self.client, ["sh_600000"], [], self.at, budget_seconds=10)
        self.assertFalse(result["raw"])
        self.assertTrue(any("BUDGET" in s for s in result["issues"]))
        self.assertEqual(sum(api == "rt_min" for _, api, _ in self.requests), 1)

    def test_repeated_limits_resume_current_batch_beyond_100_seconds(self):
        symbols = [f"sh_{600000+i}" for i in range(1001)]
        failed = [0]

        def respond(url, json, **kwargs):
            if json["api_name"] == "rt_min" and json["params"]["ts_code"] == "601000.SH" and failed[0] < 2:
                self.rate_failures = 1
                failed[0] += 1
            return self.respond(url, json, **kwargs)

        self.client.session.post.side_effect = respond
        with patch("builtins.print") as log:
            result = Q.collect(self.client, symbols, [], self.at)
        calls = [p["ts_code"] for _, api, p in self.requests if api == "rt_min"]
        self.assertEqual(calls.count(",".join(Q.ts_code(s) for s in symbols[:1000])), 1)
        self.assertEqual(calls[1:], ["601000.SH"] * 3)
        self.assertGreater(self.clock[0] - 1000, 100)
        self.assertEqual(result["quote_collection"]["status"], "COMPLETE")
        self.assertEqual(result["quote_collection"]["received"], 1001)
        self.assertFalse(result["quote_collection"]["missing_codes"])
        self.assertTrue(any("wait" in str(c) for c in log.call_args_list))

    def test_repeated_limits_stop_at_window_end_and_report_unfinished_symbols(self):
        self.rate_failures = 100
        result = Q.collect(self.client, ["sh_600000"], [], "2026-09-18 15:58")
        self.assertLessEqual(self.clock[0] - 1000, 120)
        self.assertEqual(sum(api == "rt_min" for _, api, _ in self.requests), 2)
        self.assertEqual(result["quote_collection"]["stop_reason"], "WINDOW_END")
        self.assertEqual(result["quote_collection"]["missing_codes"], ["600000.SH"])

    def test_daily_quota_during_collection_preserves_first_batch(self):
        symbols = [f"sh_{600000+i}" for i in range(1001)]

        def respond(url, json, **kwargs):
            response = self.respond(url, json, **kwargs)
            if json["api_name"] == "rt_min" and json["params"]["ts_code"] == "601000.SH":
                response.json.return_value = {"code": 40203, "msg": "访问接口频率超限(10次/天)"}
            return response

        self.client.session.post.side_effect = respond
        result = Q.collect(self.client, symbols, [], self.at)
        self.assertEqual(sum(api == "rt_min" for _, api, _ in self.requests), 2)
        self.assertEqual(result["quote_collection"]["received"], 1000)
        self.assertEqual(result["quote_collection"]["stop_reason"], "DAILY_QUOTA")
        self.assertEqual(result["quote_collection"]["missing_codes"], ["601000.SH"])

    def test_partial_response_retries_only_missing_codes(self):
        truncated = [False]

        def respond(url, json, **kwargs):
            self.truncate = json["api_name"] == "rt_min" and not truncated[0]
            response = self.respond(url, json, **kwargs)
            truncated[0] = truncated[0] or self.truncate
            return response

        self.client.session.post.side_effect = respond
        result = Q.collect(self.client, ["sh_600000", "sh_600001"], [], self.at)
        calls = [p["ts_code"] for _, api, p in self.requests if api == "rt_min"]
        self.assertEqual(calls, ["600000.SH,600001.SH", "600001.SH"])
        self.assertEqual(result["quote_collection"]["status"], "COMPLETE")
        self.assertEqual(result["quote_collection"]["received"], 2)
        self.assertFalse(any("missing" in issue for issue in result["issues"]))

    def test_transport_failure_retries_current_batch_without_restarting(self):
        symbols = [f"sh_{600000+i}" for i in range(1001)]
        failures = [0]

        def respond(url, json, **kwargs):
            if json["api_name"] == "rt_min" and json["params"]["ts_code"] == "601000.SH" and failures[0] < 2:
                failures[0] += 1
                self.requests.append((self.clock[0], "rt_min", json["params"]))
                raise Q.requests.Timeout("request timed out")
            return self.respond(url, json, **kwargs)

        self.client.session.post.side_effect = respond
        result = Q.collect(self.client, symbols, [], self.at)
        calls = [p["ts_code"] for _, api, p in self.requests if api == "rt_min"]
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[1:], ["601000.SH"] * 3)
        self.assertEqual(result["quote_collection"]["status"], "COMPLETE")

    def test_persistent_transport_failure_stops_without_claiming_complete(self):
        def respond(url, json, **kwargs):
            if json["api_name"] == "rt_min":
                self.requests.append((self.clock[0], "rt_min", json["params"]))
                raise Q.requests.Timeout("request timed out")
            return self.respond(url, json, **kwargs)

        self.client.session.post.side_effect = respond
        result = Q.collect(self.client, ["sh_600000"], [], self.at)
        self.assertEqual(sum(api == "rt_min" for _, api, _ in self.requests), 3)
        self.assertEqual(result["quote_collection"]["stop_reason"], "TRANSPORT")
        self.assertEqual(result["quote_collection"]["missing_codes"], ["600000.SH"])

    def test_missing_quotes_have_bounded_retries_and_never_claim_complete(self):
        self.truncate = True
        result = Q.collect(self.client, ["sh_600000", "sh_600001"], [], self.at)
        calls = [p["ts_code"] for _, api, p in self.requests if api == "rt_min"]
        self.assertEqual(calls, ["600000.SH,600001.SH", "600001.SH", "600001.SH"])
        self.assertEqual(result["quote_collection"]["status"], "PARTIAL")
        self.assertEqual(result["quote_collection"]["stop_reason"], "MISSING_QUOTES")
        self.assertEqual(result["quote_collection"]["missing_codes"], ["600001.SH"])

    def test_daily_quota_is_not_retried_or_mistaken_for_permanent_denial(self):
        response = Mock()
        response.json.return_value = {"code": 40203, "msg": "访问接口频率超限(10次/天)"}
        self.client.session.post.side_effect = None
        self.client.session.post.return_value = response
        with patch.object(Q, "now_china", return_value=self.at.to_pydatetime()):
            for _ in range(2):
                with self.assertRaises(Q.ApiError) as error:
                    self.client.query("rt_min")
                self.assertTrue(error.exception.daily_limit)
                self.assertFalse(error.exception.rate_limited)
        self.assertEqual(self.client.session.post.call_count, 1)
        self.assertNotIn("rt_min", self.client.denied)
        self.client.session.post.side_effect = self.respond
        with patch.object(Q, "now_china", return_value=(self.at + pd.Timedelta(days=1)).to_pydatetime()):
            self.assertEqual(len(self.client.query("rt_min")), 1)

    def test_fallback_batches_cover_5018_symbols_and_holdings_go_first(self):
        symbols = [f"sh_{600000+i}" for i in range(5017)] + ["sz_000863"]
        result = Q.collect(self.client, symbols, [], self.at, priority_symbols=["sz_000863"])
        rows = [r for b in result["raw"] for r in b["records"]]
        self.assertEqual(len(rows), 5018)
        self.assertEqual(len({r["ts_code"] for r in rows}), 5018)
        self.assertEqual(rows[0]["ts_code"], "000863.SZ")
        self.assertEqual([len(b["records"]) for b in result["raw"]], [1000, 1000, 1000, 1000, 1000, 18])
        self.assertNotIn("deadline", self.requests[0][2])
        self.assertLess(self.clock[0] - 1000, 100)

    def test_truncated_response_is_not_silently_counted_as_complete(self):
        self.truncate = True
        result = Q.collect(self.client, ["sh_600000", "sh_600001"], [], self.at)
        self.assertEqual(len(result["raw"][0]["records"]), 1)
        self.assertTrue(any("missing 1/2" in s for s in result["issues"]))

    def test_window_end_stops_collection(self):
        result = Q.collect(self.client, ["sh_600000"], [], "2026-09-18 16:00")
        self.assertFalse(self.requests)
        self.assertFalse(result["raw"])

    def test_after_close_collection_waits_and_finishes(self):
        self.rate_failures = 1
        result = Q.collect(self.client, ["sh_600000"], [], "2026-09-18 15:30")
        calls = [t for t, api, _ in self.requests if api == "rt_min"]
        self.assertEqual(len(calls), 2)
        self.assertGreaterEqual(calls[1] - calls[0], 61)
        self.assertEqual(result["quote_collection"]["status"], "COMPLETE")


class PriceDisplayTests(unittest.TestCase):
    def test_complete_sweep_does_not_make_old_quotes_fresh(self):
        from .test_intraday import IntradayTests
        fixture = IntradayTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        coverage = {"status": "COMPLETE", "requested": 1, "received": 1,
                    "missing_codes": [], "stop_reason": "ALL_RECEIVED"}
        bundle = {"raw": [{"api": "rt_min", "records": [fixture.row]}], "issues": [], "quote_collection": coverage}
        report = R.evaluate(bundle, ["sh_600000"], "2026-09-17", {}, [fixture.position],
                            {"p": fixture.prior}, {"sh_600000": fixture.history}, {},
                            fixture.at + pd.Timedelta(minutes=3))
        self.assertEqual(report["summary"]["fresh_quotes"], 0)
        self.assertFalse(report["market"]["new_review_allowed"])
        self.assertEqual(report["holdings"][0]["status"], "DATA_WAIT")
        self.assertEqual(report["quote_collection"], coverage)
        with tempfile.TemporaryDirectory() as tmp:
            output = R.publish(bundle, report, Path(tmp))
            content = (output / "review.md").read_text(encoding="utf-8")
            self.assertIn("已返回 1/1", content)
            self.assertIn("新鲜报价 0", content)
            self.assertEqual(bundle["raw"][0]["records"][0]["time"], "2026-09-18 14:35:00")

    def test_price_only_is_visible_but_does_not_become_a_risk_decision(self):
        from .test_intraday import IntradayTests
        fixture = IntradayTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        bundle = {"raw": [{"api": "rt_min", "records": [fixture.row]}], "issues": []}
        report = R.evaluate(bundle, ["sh_600000"], "2026-09-17", {}, [fixture.position],
                            {"p": fixture.prior}, {"sh_600000": fixture.history}, {}, fixture.at)
        holding = report["holdings"][0]
        self.assertEqual(holding["latest_price"], 10.4)
        self.assertEqual(holding["status"], "DATA_WAIT")
        self.assertFalse(report["market"]["new_review_allowed"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            R.check_expiry(report, fixture.at)
            out = R.publish(bundle, report, root)
            original = (out / "review.json").read_bytes()
            view = R.dashboard("2026-09-18", root, fixture.at + pd.Timedelta(minutes=3))
            self.assertEqual(view["collection"]["prices"], 1)
            self.assertEqual(view["collection"]["full"], 0)
            self.assertEqual(view["collection"]["holding_prices"], 1)
            self.assertEqual(view["mode"], "STALE")
            self.assertEqual(view["price_samples"][0]["quote"]["close"], 10.4)
            from mainwave_visual_app import app
            with patch.dict(app.config, INTRADAY_ROOT=root), patch.object(Q, "now_china", return_value=Q.stamp("2026-09-18 14:40").to_pydatetime()):
                html = app.test_client().get('/intraday?date=2026-09-18').get_data(as_text=True)
            for text in ("本轮采集状态", "仅分钟参考价", "本轮价格记录", "不是当前实时行情", "10.40"):
                self.assertIn(text, html)
            self.assertEqual((out / "review.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
