"""After-close review facts on a small deterministic universe: every definition recomputable by hand."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from . import eod_review as R

DAY = "2026-09-18"


def sessions(n=30, end=DAY):
    return pd.bdate_range(end=end, periods=n).strftime("%Y-%m-%d").tolist()


def bars(closes, opens=None, highs=None, lows=None, volumes=None, preclose_override=None, st_last=False, turnover=2.0):
    """Flat history where each bar's preclose is the prior close, unless a corporate action overrides it."""
    days = sessions(len(closes))
    rows, prior = [], closes[0]
    for i, (day, close) in enumerate(zip(days, closes)):
        pre = prior if i else close
        if preclose_override and day in preclose_override:
            pre = preclose_override[day]
        o = (opens or {}).get(day, close)
        h = (highs or {}).get(day, max(o, close))
        lo = (lows or {}).get(day, min(o, close))
        v = (volumes or {}).get(day, 1_000_000)
        rows.append(dict(date=day, open=o, high=h, low=lo, close=close, preclose=pre, volume=v, amount=v * close,
                         adjustflag=3, turnover=turnover, tradestatus=1 if v else 0, pct_chg=round((close / pre - 1) * 100, 2),
                         isST=1 if (st_last and i == len(closes) - 1) else 0))
        prior = close
    return pd.DataFrame(rows)


class Universe(unittest.TestCase):
    """A: 2-board limit-up. B (20% board): yesterday U, today touched and failed. C: limit-down. D: yesterday U, suspended.
    E: ST. F: gap-up reversal. G: yesterday U, corporate action today. H: flat. I: 20-day breakdown."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.out = self.root / "results/mainwave_engine"
        days = sessions()
        y, t = days[-2], days[-1]
        pd.DataFrame({"trade_date": days, "close": 100.}).to_csv(self.data / "000300_SH_index.csv", index=False)
        flat = [10.0] * 28
        frames = {
            "sh_600001": bars(flat + [11.0, 12.1], highs={y: 11.0, t: 12.1}),
            "sz_300002": bars(flat + [12.0, 11.5], highs={y: 12.0, t: 14.4}, lows={t: 11.4}),
            "sh_600003": bars(flat + [10.0, 9.0], lows={t: 9.0}),
            "sz_000004": bars(flat + [11.0, 11.0], highs={y: 11.0}, volumes={t: 0}),
            "sh_688005": bars(flat + [10.0, 10.2], st_last=True),
            "sz_000006": bars(flat + [10.0, 9.9], opens={t: 10.3}, highs={t: 10.35}),
            "sh_600007": bars(flat + [11.0, 5.4], highs={y: 11.0}, opens={t: 5.6}, preclose_override={t: 5.5}),
            "sz_000008": bars(flat + [10.0, 10.1]),
            "sz_000009": bars(flat + [10.0, 9.5]),
            "sz_399001": bars(flat + [10.0, 10.0]),
        }
        for symbol, frame in frames.items():
            frame.to_csv(self.data / f"{symbol}_daily.csv", index=False)
        robot = ["sh_600001", "sz_300002", "sh_600003", "sz_000004", "sz_000006", "sz_000008"]
        mapping = {s: {"name": s[-4:], "industry": "x", "concepts": ["机器人", "融资融券"] if s in robot else ["银行", "融资券融"]}
                   for s in frames}
        self.mapping = self.root / "concept_mapping.json"
        self.mapping.write_text(json.dumps({"generated": "t", "mapping": mapping}, ensure_ascii=False), encoding="utf-8")
        self.registry = self.root / "theme_registry.json"
        self.registry.write_text(json.dumps({"themes": [{"id": "robot", "label": "机器人链", "active_from": "2026-09-01",
                                                          "members": {"sh_600001": "A", "sz_300002": "B"}}]}, ensure_ascii=False), encoding="utf-8")
        (self.out / DAY.replace("-", "")).mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def build(self, day=DAY):
        return R.build(day, self.data, self.out, self.out, use_tushare=False, mapping_path=str(self.mapping),
                       registry_path=str(self.registry), now=pd.Timestamp(day + "T16:00:00+08:00").to_pydatetime())


class Definitions(Universe):
    def test_p5_first_board_is_published_by_fact_review_without_theme_or_today_p5(self):
        prior = sessions()[-2]
        bars([10.] * 29 + [11.], opens={DAY: 10.}, lows={DAY: 9.9}).to_csv(
            self.data / "sz_000008_daily.csv", index=False)
        path = self.out / prior.replace("-", "") / "preparation_watch.csv"
        path.parent.mkdir()
        path.write_text(f"symbol,detector,as_of,above_ma20_now,reclaim_failed\nsz_000008,P5_union,{prior},True,False\n",
                        encoding="utf-8")
        R._write(path.with_suffix(".published.json"), {"as_of": prior, "sha256": R.file_sha(path),
                                                      "available_at": prior + "T21:00:00+08:00"})
        facts, manifest = self.build()
        p5 = facts["p5_limit_up"]
        self.assertEqual(p5["focus_count"], 1)
        self.assertIn(DAY, p5["unverified_dates"])
        self.assertEqual(p5["rows"][0]["symbol"], "sz_000008")
        self.assertIn("P5首板重点", R.render(facts, manifest))
        self.assertEqual(R.compact(facts, manifest)["p5_limit_up"], p5)

    def test_limit_prices_round_half_up_per_board(self):
        self.assertEqual(R.limit_price(10.05, .10), 11.06)
        self.assertEqual(R.limit_price(10.0, .20), 12.0)
        self.assertEqual(R.limit_price(10.05, -.10), 9.05)
        self.assertEqual(R.board_of("sh_688005"), "STAR")
        self.assertEqual(R.board_of("sz_399001"), "")

    def test_domain_and_board_sets(self):
        facts, _ = self.build()
        self.assertEqual(facts["coverage"]["excluded"], {"ST": 1, "SUSPENDED": 1})
        self.assertEqual(facts["coverage"]["in_domain"], 7)
        b = facts["boards"]
        self.assertEqual(b["U"], ["sh_600001"])
        self.assertEqual(b["Z"], ["sz_300002"])
        self.assertEqual(b["D"], ["sh_600003"])
        self.assertEqual(b["by_board"]["CHINEXT"], {"Z": 1, "domain": 1})
        self.assertEqual(b["seal_failure_rate"]["value"], .5)
        self.assertEqual(b["seal_failure_rate"]["denominator"], 2)
        self.assertEqual(b["open_rate"]["status"], R.NO_SAMPLE, "缺开板字段不计算")

    def test_yesterday_pool_is_judged_member_by_member(self):
        facts, _ = self.build()
        y = facts["yesterday_pool"]
        self.assertEqual(y["basis"], "RECONSTRUCTED_FROM_BARS_NOT_FROZEN")
        self.assertEqual(y["members"], ["sh_600001", "sh_600007", "sz_000004", "sz_300002"])
        self.assertEqual(y["excluded"], {"SUSPENDED_OR_MISSING": 1, "CORPORATE_ACTION_PENDING": 1})
        self.assertEqual(y["n_valid"], 2)
        self.assertEqual(y["continued_U"], 1)
        self.assertEqual(y["close_vs_pre"]["n"], 2)
        self.assertEqual(y["close_vs_pre"]["worst"], round((11.5 / 12 - 1) * 100, 3))
        self.assertEqual(y["high_vs_pre"]["best"], 20.0, "日内最高单列，不是可实现收益")

    def test_ladder_heights_and_promotion_denominators(self):
        facts, _ = self.build()
        l = facts["ladder"]
        self.assertEqual(l["heights"], {"sh_600001": 2})
        self.assertEqual(l["by_height"], {"2": 1})
        promo = l["promotion_from_previous"]["1"]
        self.assertEqual((promo["pool"], promo["denominator"], promo["numerator"], promo["uncheckable"]), (4, 3, 1, 1))
        self.assertEqual(promo["continued"], ["sh_600001"])
        self.assertIn("sz_300002", promo["broke"])

    def test_theme_structure_counts_generic_tags_but_never_ranks_them(self):
        facts, _ = self.build()
        t = facts["themes"]
        robot = next(x for x in t["top_by_limit_up"] if x["theme"] == "机器人")
        # In-domain members A, B, C, F, H (D is suspended): A limit-up, A and H up.
        self.assertEqual((robot["members"], robot["limit_up"], robot["up"]), (5, 1, 2))
        self.assertNotIn("融资融券", [x["theme"] for x in t["top_by_limit_up"]])
        # 机器人 and 融资融券 reach five in-domain members, 银行 does not; the registered chain is listed regardless.
        self.assertEqual(t["themes_counted"], 3, "机器人、融资融券（通用标签也计数）与紧链")
        chain = t["chains"][0]
        self.assertEqual((chain["theme"], chain["members"]), ("紧链:机器人链", 2))
        self.assertEqual(t["overlap_of_limit_ups"], {"1": 1})

    def test_activity_and_loss_layers(self):
        facts, _ = self.build()
        a = facts["activity"]
        self.assertEqual(a["top20_share"]["value"], 1.0)
        self.assertEqual(a["close_location_buckets"].get("weak", 0) >= 1, True)
        events = facts["loss_effects"]["events"]
        self.assertEqual(events["gap_up_reversal"]["symbols"], ["sz_000006"])
        # Flat history puts every prior low at 10.00, so any close under it is a breakdown; the split (G) is not.
        self.assertEqual(events["breakdown_20d"]["symbols"], ["sh_600003", "sz_000006", "sz_000009"])
        self.assertEqual(events["limit_down"]["symbols"], ["sh_600003"])
        self.assertEqual(events["yesterday_pool_negative"]["symbols"], ["sz_300002"])
        self.assertIn("by_position", events["limit_down"])

    def test_bellwethers_are_roles_with_questions_not_a_buy_list(self):
        facts, _ = self.build()
        picks = facts["lists"]["bellwethers"]
        roles = {p["role"]: p["symbol"] for p in picks}
        self.assertEqual(roles["HEIGHT_LEADER"], "sh_600001")
        self.assertEqual(roles["RELAY_FAILURE"], "sz_300002")
        self.assertEqual(roles["LIMIT_DOWN"], "sh_600003")
        self.assertTrue(all(p["question"] and "非买入推荐" in p["note"] for p in picks))
        self.assertLessEqual(len(picks), 8)
        self.assertEqual(facts["lists"]["review_note"], "暂无条件齐备标的")


class Sentiment(Universe):
    def test_relay_sentiment_reads_each_metric_against_its_own_window(self):
        facts, _ = self.build()
        se = facts["sentiment"]
        self.assertEqual(se["window_days"][-1], DAY)
        m = se["metrics"]
        self.assertEqual(m["limit_up"]["today"], 1)
        self.assertEqual(m["limit_down"]["today"], 1)
        self.assertEqual(m["seal_failure_rate"]["today"], .5)
        # Yesterday's pool judged today: A +10% (held), B -4.17%; D suspended and G's split are excluded.
        self.assertEqual(m["pool_up_share"]["today"], .5)
        self.assertAlmostEqual(m["pool_mean_pct"]["today"], round((10.0 + (11.5 / 12 - 1) * 100) / 2, 3), places=2)
        self.assertEqual(m["promotion_1_to_2"]["today"], .5)
        self.assertEqual(m["max_height"]["today"], 2)
        # Flat history: today's limit-up count is the window maximum, so the reading is hot; readings never merge.
        self.assertEqual(m["limit_up"]["reading"], "热")
        self.assertIn("不合成一个总分", se["note"])
        self.assertNotIn("score", se)


class Denominators(Universe):
    def test_zero_denominators_read_as_no_sample(self):
        for path in self.data.glob("*_daily.csv"):
            if path.name.startswith(("sh_6000", "sz_3000", "sz_0000", "sh_6880")):
                path.unlink()
        bars([10.0] * 30).to_csv(self.data / "sh_600011_daily.csv", index=False)
        facts, _ = self.build()
        self.assertEqual(facts["boards"]["seal_failure_rate"]["status"], R.NO_SAMPLE)
        self.assertEqual(facts["yesterday_pool"]["close_vs_pre"]["status"], R.NO_SAMPLE)
        self.assertEqual(facts["ladder"]["promotion_from_previous"], {})


class ReferenceReconciliation(Universe):
    def test_reference_lists_and_prices_are_reconciled_stock_by_stock(self):
        panel, _ = R.load_panel(self.data, DAY)
        rows, _ = R.day_rows(panel, DAY)
        reference = {"status": "AVAILABLE", "issues": [], "files": {},
                     "limit_prices": {"sh_600001": {"up": 12.1, "down": 9.9}, "sz_000008": {"up": 11.5, "down": 9.5}},
                     "lists": {"U": {"status": "AVAILABLE", "records": [{"ts_code": "600001.SH", "open_times": 0}, {"ts_code": "600099.SH", "open_times": 2}]},
                               "Z": {"status": "AVAILABLE", "records": [{"ts_code": "300002.SZ", "open_times": 1}]},
                               "D": {"status": "UNAVAILABLE", "issues": ["no permission"]}}}
        disagreements = R.apply_reference_prices(rows, reference["limit_prices"])
        self.assertEqual([d["symbol"] for d in disagreements], ["sz_000008"])
        boards = R.boards_fact(rows, DAY, reference)
        self.assertEqual(boards["reconciliation"]["U"]["both"], 1)
        self.assertEqual(boards["reconciliation"]["U"]["reference_only"], ["sh_600099"])
        self.assertEqual(boards["reconciliation"]["Z"]["both"], 1)
        self.assertEqual(boards["reconciliation"]["D"]["status"], "REFERENCE_UNAVAILABLE")
        self.assertEqual(boards["open_rate"]["value"], 0)
        self.assertEqual(boards["open_rate"]["denominator"], 1)
        self.assertEqual(boards["open_rate"]["open_times_total"], 0)


class Versions(Universe):
    def test_same_inputs_reuse_and_changed_inputs_version_up(self):
        facts, manifest = self.build()
        first = R.persist(DAY, facts, manifest, self.out)
        self.assertEqual(first["version"], "v001")
        self.assertFalse(first["reused"])
        self.assertTrue((Path(first["path"]) / "report.md").exists())
        again = R.persist(DAY, *self.build(), self.out)
        self.assertTrue(again["reused"])
        self.assertEqual(again["version"], "v001")
        self.mapping.write_text(self.mapping.read_text(encoding="utf-8").replace("机器人", "人形机器人"), encoding="utf-8")
        third = R.persist(DAY, *self.build(), self.out)
        self.assertEqual(third["version"], "v002")
        self.assertTrue((Path(first["path"]) / "facts.json").exists(), "旧版本保留")
        self.assertEqual(json.loads((self.out / DAY.replace("-", "") / "eod_review/latest.json").read_text(encoding="utf-8"))["version"], "v002")

    def test_next_day_freezes_yesterday_pool_from_the_previous_review(self):
        facts, manifest = self.build()
        R.persist(DAY, facts, manifest, self.out)
        days = sessions(31, end="2026-09-21")
        for path in self.data.glob("*_daily.csv"):
            f = pd.read_csv(path)
            last = f.iloc[-1].to_dict()
            last.update(date="2026-09-21", preclose=last["close"], open=last["close"], high=last["close"], low=last["close"], pct_chg=0.0, volume=1_000_000)
            pd.concat([f, pd.DataFrame([last])]).to_csv(path, index=False)
        pd.DataFrame({"trade_date": days, "close": 100.}).to_csv(self.data / "000300_SH_index.csv", index=False)
        (self.out / "20260921").mkdir()
        nxt, _ = self.build("2026-09-21")
        self.assertEqual(nxt["yesterday_pool"]["basis"], "FROZEN_PREVIOUS_REVIEW")
        self.assertEqual(nxt["yesterday_pool"]["members"], ["sh_600001"])
        self.assertEqual(nxt["ladder"]["promotion_from_previous"]["2"]["pool"], 1)

    def test_report_and_manifest_name_gaps_and_sections(self):
        facts, manifest = self.build()
        text = R.render(facts, manifest)
        for heading in ("## 市场与接力", "## 连板梯队", "## 题材地图", "## 活跃与风险", "## 三类清单", "未实施（P1）"):
            self.assertIn(heading, text)
        self.assertIn("昨日涨停母池由日线重建", "；".join(manifest["gaps"]))
        self.assertEqual(manifest["sections"]["conclusions"], "NOT_IMPLEMENTED_P1")
        self.assertEqual(manifest["stage"], "PARTIAL_DATA")
        self.assertEqual(manifest["sections"]["market"], "MISSING_INPUT")
        R._write(self.out / DAY.replace("-", "") / "market_context.json", {"data_status": "FRESH"})
        _, with_context = self.build()
        self.assertEqual(with_context["stage"], "PRICE_FINAL_REFERENCE_PENDING")


class Boundaries(unittest.TestCase):
    def test_optional_review_failure_warns_without_failing_core(self):
        from types import SimpleNamespace
        from . import pipeline
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def launch(command, **_):
                if command[2] == "mainwave_engine.eod_review":
                    R._write(root / "20260918/eod_review/status.json", {"as_of": DAY, "status": "FAILED", "error": "test"})
                    return SimpleNamespace(returncode=1)
                if command[2] == "mainwave_engine.fupan":
                    R._write(root / "20260918/fupan/status.json", {"as_of": DAY, "status": "SUCCESS"})
                return SimpleNamespace(returncode=0)
            with patch.object(pipeline, "OUT_ROOT", root), patch.object(pipeline.subprocess, "run", side_effect=launch):
                self.assertEqual(pipeline.run(DAY), 0)
            status = R.read_json(root / "20260918/mainwave_run_status.json")
            self.assertEqual(status["status"], "SUCCESS")
            self.assertEqual(status["delivery_status"], "DEGRADED")
            review = next(s for s in status["steps"] if s["name"] == "eod_review")
            self.assertEqual((review["name"], review["status"], review["exit_code"]), ("eod_review", "WARNING", 1))
            self.assertEqual((status["steps"][-1]["name"], status["steps"][-1]["status"]), ("fupan", "SUCCESS"))
            self.assertTrue(status["warnings"])
            self.assertEqual(R.read_json(root / "20260918/_SUCCESS")["delivery_status"], "DEGRADED")

    def test_review_never_imports_the_ledger(self):
        source = Path(R.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from .trading", source)
        self.assertNotIn("import paper", source)

    def test_failure_is_recorded_with_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "data").mkdir()
            import sys
            argv = sys.argv
            sys.argv = ["eod_review", "--date", DAY, "--data-root", str(root / "data"), "--out-root", str(root / "out"), "--no-tushare"]
            try:
                self.assertNotEqual(R.main(), 0)
            finally:
                sys.argv = argv
            status = json.loads((root / "out" / DAY.replace("-", "") / "eod_review/status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "FAILED")
            self.assertIn("市场日历", status["error"])


class RegressionBoundaries(Universe):
    def test_known_at_is_after_inputs_have_been_read(self):
        started = pd.Timestamp(DAY + "T16:00:00+08:00").to_pydatetime()
        completed = pd.Timestamp(DAY + "T16:04:00+08:00").to_pydatetime()
        with patch.object(R, "datetime") as clock:
            clock.now.side_effect = [started, completed]
            facts, manifest = R.build(DAY, self.data, self.out, self.out, use_tushare=False,
                                      mapping_path=str(self.mapping), registry_path=str(self.registry))
        self.assertEqual(pd.Timestamp(facts["known_at"]), pd.Timestamp(completed))
        self.assertEqual(manifest["known_at"], facts["known_at"])

    def test_corrected_bar_invalidates_cached_report(self):
        facts, manifest = self.build()
        first = R.persist(DAY, facts, manifest, self.out)
        path = self.data / "sh_600001_daily.csv"
        frame = pd.read_csv(path)
        frame.loc[frame.index[-1], "close"] = 11.9
        frame.loc[frame.index[-1], "low"] = 11.9
        frame.to_csv(path, index=False)
        revised, changed = self.build()
        self.assertNotEqual(manifest["input_hash"], changed["input_hash"])
        second = R.persist(DAY, revised, changed, self.out)
        self.assertEqual(second["version"], "v002")
        self.assertEqual(R.read_json(Path(first["path"]) / "facts.json")["boards"]["counts"]["U"], 1)
        self.assertEqual(R.read_json(Path(second["path"]) / "facts.json")["boards"]["counts"]["U"], 0)

    def test_index_correction_invalidates_cache(self):
        _, manifest = self.build()
        path = self.data / "000300_SH_index.csv"
        frame = pd.read_csv(path)
        frame.loc[frame.index[-1], "close"] = 101
        frame.to_csv(path, index=False)
        facts, changed = self.build()
        self.assertEqual(facts["market"]["index"]["return_pct"], 1)
        self.assertNotEqual(manifest["input_hash"], changed["input_hash"])

    def test_invalid_index_does_not_mark_market_ready(self):
        path = self.data / "000300_SH_index.csv"
        frame = pd.read_csv(path)
        frame.loc[frame.index[-1], "close"] = float("nan")
        frame.to_csv(path, index=False)
        facts, manifest = self.build()
        self.assertIsNone(facts["market"]["index"])
        self.assertEqual(manifest["sections"]["market"], "MISSING_INPUT")
        self.assertEqual(manifest["stage"], "PARTIAL_DATA")

    def test_prior_version_is_fingerprinted_and_future_version_not_used(self):
        yesterday = sessions()[-2]
        prior, prior_manifest = self.build(yesterday)
        R.persist(yesterday, prior, prior_manifest, self.out)
        initial, initial_manifest = self.build()
        prior["boards"]["U"].remove("sh_600001")
        prior["ladder"]["heights"].pop("sh_600001")
        prior["known_at"] = DAY + "T15:30:00+08:00"
        prior_manifest.update(input_hash="prior-correction", known_at=prior["known_at"])
        R.persist(yesterday, prior, prior_manifest, self.out)
        corrected, corrected_manifest = self.build()
        self.assertNotEqual(initial_manifest["input_hash"], corrected_manifest["input_hash"])
        self.assertNotEqual(initial["yesterday_pool"]["members"], corrected["yesterday_pool"]["members"])
        self.assertEqual(corrected["yesterday_pool"]["basis"], "HISTORICAL_PREVIOUS_REVIEW")
        self.assertEqual(corrected["ladder"]["heights"]["sh_600001"], 1)
        self.assertEqual(corrected_manifest["sources"]["p5_preparation"]["prior_review"]["version"], "v001")
        prior["boards"]["U"] = []
        prior["known_at"] = "2026-09-21T16:00:00+08:00"
        prior_manifest.update(input_hash="future-correction", known_at=prior["known_at"])
        R.persist(yesterday, prior, prior_manifest, self.out)
        again, _ = self.build()
        self.assertEqual(again["yesterday_pool"]["members"], corrected["yesterday_pool"]["members"])

    def test_missing_today_does_not_remove_yesterday_member(self):
        path = self.data / "sz_000004_daily.csv"
        frame = pd.read_csv(path)
        frame[frame.date.ne(DAY)].to_csv(path, index=False)
        facts, _ = self.build()
        self.assertIn("sz_000004", facts["yesterday_pool"]["members"])
        self.assertEqual(facts["yesterday_pool"]["excluded"]["SUSPENDED_OR_MISSING"], 1)

    def test_empty_panel_is_data_wait_not_zero_market(self):
        with patch.object(R, "load_panel", return_value=({}, [])):
            with self.assertRaisesRegex(ValueError, "DATA_WAIT"):
                self.build()

    def test_invalid_quote_and_unknown_st_are_excluded(self):
        for symbol, field in (("sh_600001", "close"), ("sz_000008", "isST")):
            path = self.data / (symbol + "_daily.csv")
            frame = pd.read_csv(path)
            frame.loc[frame.index[-1], field] = float("nan")
            frame.to_csv(path, index=False)
        facts, manifest = self.build()
        self.assertEqual(facts["coverage"]["excluded"]["INVALID_DATA"], 2)
        self.assertEqual(manifest["sections"]["boards"], "PARTIAL")
        self.assertNotIn("sh_600001", facts["boards"]["U"])
        self.assertIsNone(R.number(float("inf")))

    def test_missing_metadata_column_is_not_assumed_nonst(self):
        path = self.data / "sh_600001_daily.csv"
        pd.read_csv(path).drop(columns=["isST"]).to_csv(path, index=False)
        facts, _ = self.build()
        self.assertEqual(facts["coverage"]["excluded"]["INVALID_DATA"], 1)
        self.assertNotIn("sh_600001", facts["boards"]["U"])

    def test_open_rate_excludes_out_of_domain_and_duplicate_rows(self):
        panel, _ = R.load_panel(self.data, DAY)
        rows, _ = R.day_rows(panel, DAY)
        records = [{"ts_code": "600001.SH", "open_times": 0},
                   {"ts_code": "600001.SH", "open_times": 0},
                   {"ts_code": "920298.BJ", "open_times": 2},
                   {"ts_code": "688005.SH", "open_times": 2}]
        result = R.boards_fact(rows, DAY, {"lists": {"U": {"status": "AVAILABLE", "records": records}}})
        self.assertEqual(result["open_rate"]["denominator"], 1)
        self.assertEqual(result["open_rate"]["numerator"], 0)

    def test_long_suspended_is_not_a_gap_but_one_missing_day_is(self):
        stale = bars([10.] * 30).iloc[:-2]      # last bar two sessions before DAY
        missing_today = bars([10.] * 30).iloc[:-1]  # last bar on the previous session
        days = sessions()
        rows, excluded = R.day_rows({"sh_600001": stale, "sh_600002": missing_today}, DAY, days[-2])
        self.assertEqual(rows["sh_600001"]["reason"], "LONG_SUSPENDED")
        self.assertEqual(rows["sh_600002"]["reason"], "NOT_TRADED_OR_MISSING")
        self.assertEqual(excluded, {"LONG_SUSPENDED": 1, "NOT_TRADED_OR_MISSING": 1})

    def test_special_limit_uses_frozen_previous_height(self):
        frame = bars([10.] * 28 + [10.5, 11.03])
        rows, _ = R.day_rows({"sh_600182": frame}, DAY)
        R.apply_reference_prices(rows, {"sh_600182": {"up": 11.03, "down": 9.98}})
        ladder = R.ladder_fact(rows, sessions(), DAY, {"1": ["sh_600182"]})
        self.assertEqual(ladder["heights"]["sh_600182"], 2)
        self.assertEqual(ladder["promotion_from_previous"]["1"]["numerator"], 1)

    def test_complete_reference_missing_price_does_not_assume_normal_limit(self):
        rows, _ = R.day_rows({"sh_600001": bars([10.] * 29 + [11.])}, DAY)
        R.apply_reference_prices(rows, {}, reference_complete=True)
        self.assertFalse(bool(rows["sh_600001"]["frame"].iloc[-1].is_U))
        self.assertEqual(rows["sh_600001"]["price_basis"], "LIMIT_RULE_UNKNOWN")

    def test_failed_rerun_api_exposes_stale_report(self):
        from flask import Flask
        from .trading import web
        facts, manifest = self.build()
        R.persist(DAY, facts, manifest, self.out)
        R._write(self.out / DAY.replace("-", "") / "eod_review/status.json",
                 {"as_of": DAY, "status": "FAILED", "error": "synthetic source failure"})
        app = Flask(__name__)
        app.register_blueprint(web.bp)
        with patch.object(web.store, "ROOT", self.root):
            client = app.test_client()
            result = client.get("/api/trading/eod_review?as_of=" + DAY).get_json()
            exported = client.get("/api/trading/eod_review/report.md?as_of=" + DAY).get_data(as_text=True)
        self.assertEqual(result["status"], "STALE_REVIEW")
        self.assertEqual(result["latest_attempt"]["status"], "FAILED")
        self.assertIn("synthetic source failure", result["note"])
        self.assertIn("synthetic source failure", exported)

    def test_cached_cli_summary_matches_saved_manifest(self):
        facts, manifest = self.build()
        R.persist(DAY, facts, manifest, self.out)
        import sys
        argv = ["eod_review", "--date", DAY, "--out-root", str(self.out), "--no-tushare"]
        revised_manifest = {**manifest, "known_at": "2026-09-21T16:00:00+08:00"}
        with patch.object(sys, "argv", argv), patch.object(R, "build", return_value=(facts, revised_manifest)):
            self.assertEqual(R.main(), 0)
        status = R.read_json(self.out / DAY.replace("-", "") / "eod_review/status.json")
        self.assertEqual(status["known_at"], manifest["known_at"])


if __name__ == "__main__":
    unittest.main()
