from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from mainwave_engine import intraday_quotes as Q, intraday_review as R


class IntradayTests(unittest.TestCase):
    def setUp(self):
        self.at = Q.stamp("2026-09-18 14:35:00")
        self.row = {"ts_code": "600000.SH", "trade_time": "2026-09-18 14:35:00", "time": "2026-09-18 14:35:00",
                    "pre_close": 10., "open": 10., "high": 10.5, "low": 9.9, "close": 10.4,
                    "vol": 1000000., "amount": 10200000.}
        self.quote = Q.normalize(pd.DataFrame([self.row]), "rt_k", self.at)[0]["600000.SH"]
        dates = pd.bdate_range(end="2026-09-17", periods=90)
        self.history = pd.DataFrame({"date": dates, "open": 10., "high": 10.1, "low": 9.9,
                                     "close": 10., "preclose": 10., "pct_chg": 0., "volume": 2000000., "amount": 20000000.})
        self.position = {"id": "p", "symbol": "sh_600000", "name": "Test", "entry_date": "2026-09-16"}
        self.prior = {"status": "HOLD", "hard_floor": 9.5, "soft_floor": 10.2}
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_minute_bar_never_becomes_day_volume_or_ohlc(self):
        quotes, errors = Q.normalize(pd.DataFrame([self.row]), "rt_min", self.at)
        self.assertFalse(errors)
        q = quotes["600000.SH"]
        self.assertEqual(q["close"], 10.4)
        for key in ("pre_close", "volume", "amount", "vwap", "open", "high", "low"):
            self.assertIsNone(q[key])

    def test_wrong_day_stale_future_missing_time_are_rejected(self):
        for time, reason in (("2026-09-17 15:00:00", "WRONG_DAY"), ("2026-09-18 14:30:00", "STALE_QUOTE"),
                             ("2026-09-18 14:36:00", "FUTURE_QUOTE"), (None, "MISSING_OR_INVALID_TIMESTAMP")):
            q, errors = Q.normalize(pd.DataFrame([{**self.row, "trade_time": time}]), "rt_k", self.at)
            self.assertFalse(q)
            self.assertEqual(errors[0]["reason"], reason)

    def test_invalid_ohlc_and_nonfinite_volume_are_rejected(self):
        for update in ({"high": 9.}, {"vol": float("inf")}, {"amount": -1}, {"pre_close": 0}, {"close": float("nan")}):
            self.assertFalse(Q.normalize(pd.DataFrame([{**self.row, **update}]), "rt_k", self.at)[0])

    def test_duplicate_quote_uses_latest_timestamp(self):
        old = {**self.row, "trade_time": "2026-09-18 14:34:00", "close": 10.2}
        q, _ = Q.normalize(pd.DataFrame([self.row, old]), "rt_k", self.at)
        self.assertEqual(q["600000.SH"]["close"], 10.4)

    def test_lunch_not_counted_as_trading_minutes(self):
        self.assertEqual(Q.session_minutes("2026-09-18 14:30:00"), 210)
        self.assertEqual(Q.session_minutes("2026-09-18 12:30:00"), 120)

    def test_partial_minutes_cannot_claim_full_day_vwap(self):
        frame = pd.DataFrame([{**self.row, "freq": "1MIN"}])
        result = Q.minute_context(frame, "600000.SH", self.at)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertIsNone(result["full_day_vwap"])

    def minute_frame(self):
        times = pd.date_range("2026-09-18 09:31", "2026-09-18 11:30", freq="min").union(
            pd.date_range("2026-09-18 13:01", "2026-09-18 14:35", freq="min"))
        return pd.DataFrame({"ts_code": "600000.SH", "time": times, "open": 10., "high": 10.5,
                             "low": 9.9, "close": 10.4, "vol": 100., "amount": 1020.})

    def test_minute_vwap_complete_sequence_and_dedup(self):
        f = self.minute_frame()
        result = Q.minute_context(pd.concat([f, f.tail(1)]), "600000.SH", self.at)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["missing_minutes"], 0)
        self.assertAlmostEqual(result["full_day_vwap"], 10.2)
        self.assertEqual(result["above_vwap_ratio"], 1.)
        self.assertEqual(Q.minute_context(f.drop(index=50), "600000.SH", self.at)["status"], "INCOMPLETE")

    def test_history_slice_excludes_same_day_final_bar_and_future(self):
        extra = pd.DataFrame({"date": pd.to_datetime(["2026-09-18", "2026-09-21"]), "close": [50., 60.]})
        pd.concat([self.history, extra]).to_csv(self.root / "sh_600000_daily.csv", index=False)
        f = R.read_history("sh_600000", "2026-09-18", self.root)
        self.assertEqual(len(f), len(self.history))
        self.assertEqual(f.close.iloc[-1], 10.)

    def test_metrics_price_only_is_unverified_and_volume_not_projected(self):
        q = Q.normalize(pd.DataFrame([self.row]), "rt_min", self.at)[0]["600000.SH"]
        result = R.quick_metrics(self.history, q, "2026-09-17")
        self.assertEqual(result["status"], "PRICE_REFERENCE_UNVERIFIED")
        self.assertIsNone(result["volume_vs_full_day5"])
        result = R.quick_metrics(self.history, self.quote, "2026-09-17", {
            "trade_date": "20260917", "float_share": 10000., "total_share": 20000.})
        self.assertAlmostEqual(result["volume_vs_full_day5"], .5)
        self.assertAlmostEqual(result["turnover_pct_estimate"], 1.)
        self.assertIsNone(result["volume_ratio_same_time"])

    def test_ex_right_reference_revalues_ema_and_risk_floors(self):
        q = {**self.quote, "pre_close": 5., "close": 5.2, "low": 5.}
        m = R.quick_metrics(self.history, q, "2026-09-17")
        normal = R.quick_metrics(self.history, self.quote, "2026-09-17")
        self.assertAlmostEqual(m["ema20"], normal["ema20"] / 2)
        r = R.holding_review(self.position, self.prior, q, m)
        self.assertEqual(r["status"], "HOLD_OBSERVATION")
        self.assertAlmostEqual(r["hard_floor"], 4.75)

    def test_cached_ema_exactly_matches_full_series_calculation(self):
        f = self.history.copy()
        f["pct_chg"] = [(-1 if i % 3 == 0 else 1.2) for i in range(len(f))]
        index = (1 + f.pct_chg / 100).cumprod() * 100
        f["close"] = index / 10
        f["high"] = f.close * 1.02
        reference = float(f.close.iloc[-1]) / 2
        q = {**self.quote, "pre_close": reference, "close": reference * 1.04}
        seed = R.reference_features(f, "2026-09-17")
        result = R.quick_metrics(seed, q, "2026-09-17")
        current = index.iloc[-1] * 1.04
        joined = pd.concat([index, pd.Series([current])], ignore_index=True)
        for span in (20, 60):
            expected = joined.ewm(span=span, adjust=False).mean().iloc[-1] * q["close"] / current
            self.assertAlmostEqual(result[f"ema{span}"], expected, places=10)
        self.assertEqual(R.quick_metrics(seed, q, "2026-09-16")["status"], "HISTORY_WAIT")

    def test_prior_observation_window_prevents_chasing(self):
        bundle = {"raw": [{"api": "rt_k", "records": [self.row]},
                          {"api": "rt_idx_k", "records": [{**self.row, "ts_code": c} for c in Q.INDEX_CODES]},
                          {"api": "rt_min_daily", "records": self.minute_frame().to_dict("records")}], "issues": []}
        old = {"symbol": "sh_600000", "structural_advice": "PRIORITY_REVIEW", "review_lower": 10., "review_upper": 10.3, "invalid": 9.5}
        baseline = {"market": {"data_status": "FRESH", "trend_review_allowed": True}, "records": [old]}
        report = R.evaluate(bundle, ["sh_600000"], "2026-09-17", baseline, [], {}, {"sh_600000": self.history}, {}, self.at)
        self.assertEqual(report["records"][0]["status"], "WAIT_PRICE")
        old["review_upper"] = 10.5
        report = R.evaluate(bundle, ["sh_600000"], "2026-09-17", baseline, [], {}, {"sh_600000": self.history}, {}, self.at)
        self.assertEqual(report["records"][0]["status"], "PROVISIONAL_REVIEW")

    def test_soft_floor_does_not_count_as_confirmed_close(self):
        q = {**self.quote, "close": 10., "low": 9.8}
        r = R.holding_review(self.position, self.prior, q, {"status": "PROVISIONAL", "reference_scale": 1, "previous_date": "2026-09-17"})
        self.assertEqual(r["status"], "WAIT_CLOSE_CONFIRMATION")
        self.assertNotIn("soft_breaches", r)

    def test_hard_stop_prior_exit_t1_and_no_orders(self):
        m = {"status": "PROVISIONAL", "reference_scale": 1, "previous_date": "2026-09-17"}
        for q, old in (({**self.quote, "low": 9.4}, self.prior), (self.quote, {**self.prior, "status": "SELL"})):
            r = R.holding_review(self.position, old, q, m)
            self.assertEqual(r["status"], "SELL_REVIEW")
            self.assertFalse(r["orders_enabled"])
        r = R.holding_review({**self.position, "entry_date": "2026-09-18"}, self.prior, self.quote, m)
        self.assertEqual(r["status"], "T1_LOCKED")
        self.assertEqual(R.holding_review(self.position, {**self.prior, "status": "SELL"}, None, {})["prior_status"], "SELL")

    def test_missing_indices_cannot_open_market_gate(self):
        r = R.market_review({"600000.SH": self.quote}, ["sh_600000"], {}, {"data_status": "FRESH", "trend_review_allowed": True})
        self.assertEqual(r["state"], "UNKNOWN")
        self.assertFalse(r["new_review_allowed"])

    def test_market_prior_cannot_be_upgraded_by_intraday_rally(self):
        indices = {code: self.quote for code in Q.INDEX_CODES}
        r = R.market_review({"600000.SH": self.quote}, ["sh_600000"], indices, {"data_status": "FRESH", "trend_review_allowed": False})
        self.assertEqual(r["state"], "RESTRICTED")

    def report(self):
        bundle = {"raw": [{"api": "rt_k", "records": [self.row]}], "issues": []}
        report = R.evaluate(bundle, ["sh_600000"], "2026-09-17", {}, [self.position], {"p": self.prior},
                            {"sh_600000": self.history}, {}, self.at)
        return bundle, report

    def test_publish_immutable_snapshots_and_no_official_mutation(self):
        bundle, report = self.report()
        a = R.publish(bundle, report, self.root)
        before = (a / "review.json").read_bytes()
        b = R.publish(bundle, report, self.root)
        self.assertNotEqual(a, b)
        self.assertEqual(before, (a / "review.json").read_bytes())
        self.assertEqual(R.effective("2026-09-18", self.root, self.at)["mode"], "INTRADAY_PROVISIONAL")
        self.assertEqual(R.effective("2026-09-18", self.root, self.at + pd.Timedelta(minutes=3))["mode"], "STALE")

    def test_finalize_requires_fresh_success_then_preserves_intraday(self):
        bundle, report = self.report()
        output = R.publish(bundle, report, self.root)
        original = (output / "review.json").read_bytes()
        daily = self.root / "daily"
        day = daily / "20260918"
        day.mkdir(parents=True)
        data = self.root / "data"
        data.mkdir()
        pd.DataFrame([{"date": "2026-09-18", "close": 10.6}]).to_csv(data / "sh_600000_daily.csv", index=False)
        final = lambda: R.finalize("2026-09-18", self.root, daily, data, "2026-09-18 17:00")
        self.assertEqual(final()["status"], "PENDING_FINAL")
        R._write(day / "mainwave_run_status.json", {"as_of": "2026-09-18", "status": "SUCCESS", "finished_at": "2026-09-18T16:00:00"})
        R._write(day / "analysis_universe.json", {"as_of": "2026-09-18", "summary": {"status": "SUCCESS"}, "market": {"data_status": "FRESH"}, "records": []})
        R._write(day / "virtual_positions.json", {"as_of": "2026-09-18", "summary": {"status": "SUCCESS"}})
        (day / "_SUCCESS").write_text("ok")
        result = final()
        self.assertEqual(result["status"], "FINAL")
        self.assertAlmostEqual(result["comparisons"][0]["change_after_snapshot_pct"], (10.6 / 10.4 - 1) * 100)
        self.assertEqual(R.effective("2026-09-18", self.root)["mode"], "FINAL")
        self.assertEqual(original, (output / "review.json").read_bytes())

    def test_finalize_lists_long_suspended_instead_of_failing(self):
        bundle, report = self.report()
        R.publish(bundle, report, self.root)
        daily = self.root / "daily"
        day = daily / "20260918"
        day.mkdir(parents=True)
        data = self.root / "data"
        data.mkdir()
        pd.DataFrame([{"date": "2026-09-10", "close": 10.6}]).to_csv(data / "sh_600000_daily.csv", index=False)
        pd.DataFrame({"trade_date": ["2026-09-16", "2026-09-17", "2026-09-18"]}).to_csv(data / "000300_SH_index.csv", index=False)
        R._write(day / "mainwave_run_status.json", {"as_of": "2026-09-18", "status": "SUCCESS", "finished_at": "2026-09-18T16:00:00"})
        R._write(day / "analysis_universe.json", {"as_of": "2026-09-18", "summary": {"status": "SUCCESS"}, "market": {"data_status": "FRESH"}, "records": []})
        R._write(day / "virtual_positions.json", {"as_of": "2026-09-18", "summary": {"status": "SUCCESS"}})
        (day / "_SUCCESS").write_text("ok")
        result = R.finalize("2026-09-18", self.root, daily, data, "2026-09-18 17:00")
        self.assertEqual(result["status"], "FINAL")
        self.assertEqual(result["suspended_no_close"], [{"symbol": "sh_600000", "last_bar": "2026-09-10"}])

    def test_outside_window_does_not_fetch_or_write(self):
        for clock in ("09:00", "09:29:59", "11:31", "12:00", "12:59:59", "16:00", "16:01"):
            with self.subTest(clock=clock), patch.object(Q, "now_china", return_value=Q.stamp(f"2026-09-18 {clock}").to_pydatetime()), patch.object(Q, "Client") as client:
                for runner in (R.run, R.watch):
                    with self.assertRaises(ValueError):
                        runner("2026-09-18")
                client.assert_not_called()

    def test_expanded_start_window_reaches_calendar_check(self):
        for clock in ("09:30", "10:15", "11:30", "13:00", "14:00", "15:00", "15:30", "15:59:59"):
            with self.subTest(clock=clock), patch.object(Q, "now_china", return_value=Q.stamp(f"2026-09-18 {clock}").to_pydatetime()), patch.object(Q, "Client") as client:
                client.return_value.query.side_effect = RuntimeError("calendar reached")
                with self.assertRaisesRegex(RuntimeError, "calendar reached"):
                    R.run("2026-09-18")
                self.assertEqual(client.return_value.query.call_args.args[0], "trade_cal")

    def test_expanded_window_still_rejects_historical_dates(self):
        with patch.object(Q, "now_china", return_value=Q.stamp("2026-09-18 15:30").to_pydatetime()), patch.object(Q, "Client") as client:
            for runner in (R.run, R.watch):
                with self.assertRaises(ValueError):
                    runner("2026-09-17")
            client.assert_not_called()

    def test_watch_accepts_expanded_window_and_stops_at_1600(self):
        for clock in ("09:30", "13:00", "15:30", "15:59:59"):
            current = [Q.stamp(f"2026-09-18 {clock}").to_pydatetime()]

            def finish(*args):
                current[0] = Q.stamp("2026-09-18 16:00").to_pydatetime()
                return {"summary": {"status": "PARTIAL"}}

            with self.subTest(clock=clock), patch.object(Q, "now_china", side_effect=lambda: current[0]), \
                    patch.object(Q, "Client"), patch.object(R, "run", side_effect=finish) as run, patch.object(R.time, "sleep") as sleep:
                self.assertEqual(R.watch("2026-09-18"), 2)
                run.assert_called_once()
                sleep.assert_not_called()

    def test_watch_waits_through_the_midday_break_without_calling_the_api(self):
        clocks = iter(["2026-09-18 11:45", "2026-09-18 16:00"])
        current = [Q.stamp("2026-09-18 11:00").to_pydatetime()]

        def tick():
            return current[0]

        def advance(*_):
            current[0] = Q.stamp(next(clocks)).to_pydatetime()

        def one_run(*args):
            current[0] = Q.stamp("2026-09-18 11:45").to_pydatetime()
            return {"summary": {"status": "PROVISIONAL"}}

        with patch.object(Q, "now_china", side_effect=tick), patch.object(Q, "Client"),                 patch.object(R, "run", side_effect=one_run) as run, patch.object(R.time, "sleep", side_effect=advance) as sleep:
            current[0] = Q.stamp("2026-09-18 11:00").to_pydatetime()
            R.watch("2026-09-18")
        run.assert_called_once()  # the 11:45 cycles only slept
        self.assertTrue(any(c.args == (60,) for c in sleep.call_args_list))

    def test_collection_window_does_not_expand_participation_window(self):
        for clock in ("09:30", "11:00", "13:00", "14:29", "14:57", "15:01", "15:30"):
            with self.subTest(clock=clock):
                _, report = self.report()
                R.check_expiry(report, f"2026-09-18 {clock}")
                self.assertFalse(report["market"]["entry_window_open"])
                self.assertFalse(report["market"]["new_review_allowed"])

    def test_collection_falls_back_once_and_never_loops_on_denied_minutes(self):
        class FakeClient:
            audit = []
            calls = []

            def query(client, api, **params):
                client.calls.append(api)
                if api != "rt_min":
                    raise Q.ApiError(api, 40203, "denied")
                return pd.DataFrame([self.row])

        client = FakeClient()
        bundle = Q.collect(client, ["sh_600000"], ["sh_600000", "sz_000001"], self.at)
        self.assertEqual(client.calls, ["rt_k", "rt_min", "rt_idx_k", "rt_min_daily"])
        self.assertEqual(bundle["raw"][0]["api"], "rt_min")

    def test_expiry_after_calculation_blocks_actions(self):
        _, report = self.report()
        R.check_expiry(report, self.at + pd.Timedelta(minutes=3))
        self.assertEqual(report["summary"]["status"], "EXPIRED")
        self.assertFalse(report["market"]["new_review_allowed"])
        self.assertEqual(report["records"][0]["status"], "DATA_WAIT")
        self.assertEqual(report["holdings"][0]["status"], "DATA_WAIT")

    def test_bad_history_and_new_price_hits_remain_separate_from_market_permission(self):
        self.assertEqual(R.quick_metrics(self.history.drop(columns="volume"), self.quote, "2026-09-17")["status"], "HISTORY_WAIT")
        _, report = self.report()
        self.assertTrue(report["records"][0]["new_price_breakout"])
        self.assertEqual(report["records"][0]["status"], "MARKET_WAIT")
        self.assertEqual(report["summary"]["new_price_discoveries"], 1)

    def test_integration_run_uses_reference_history_and_isolated_output(self):
        day = self.root / "daily" / "20260917"
        day.mkdir(parents=True)
        (day / "_SUCCESS").write_text("ok")
        R._write(day / "analysis_universe.json", {"as_of": "2026-09-17", "records": [], "market": {"data_status": "FRESH", "trend_review_allowed": True}})
        history_copy = self.history.copy(deep=True)

        class FakeClient:
            audit = []

            def query(client, api, **params):
                if api == "trade_cal":
                    return pd.DataFrame({"cal_date": ["20260917", "20260918"], "is_open": [1, 1]})
                if api == "daily_basic":
                    return pd.DataFrame([{"ts_code": "600000.SH", "trade_date": "20260917", "float_share": 10000., "total_share": 10000.}])
                if api == "rt_k":
                    return pd.DataFrame([self.row])
                if api == "rt_idx_k":
                    return pd.DataFrame([{**self.row, "ts_code": code} for code in Q.INDEX_CODES])
                return pd.DataFrame()

        # Function defaults bind their original roots; explicitly redirect publishing too.
        publish = R.publish
        D = R.D
        with patch.object(Q, "now_china", return_value=self.at.to_pydatetime()), \
                patch.object(R, "ROOT", self.root / "intraday"), patch.object(R, "DAILY", day.parent), \
                patch.object(R, "read_positions", return_value=[]), patch.object(D, "list_universe", return_value=["sh_600000"]), \
                patch.object(R.TS, "load_mapping_as_of", return_value=({}, {"status": "MISSING"})), \
                patch.object(R, "read_history", return_value=self.history), \
                patch.object(R, "publish", side_effect=lambda b, r: publish(b, r, self.root / "intraday")):
            result = R.run("2026-09-18", client=FakeClient())
        self.assertEqual(result["summary"]["status"], "PARTIAL")
        self.assertEqual(result["themes"]["status"], "DATA_WAIT")
        self.assertEqual(result["records"][0]["status"], "NEW_PRICE_DISCOVERY")
        pd.testing.assert_frame_equal(history_copy, self.history)
        self.assertFalse((self.root / "daily" / "20260918").exists())


if __name__ == "__main__":
    unittest.main()
