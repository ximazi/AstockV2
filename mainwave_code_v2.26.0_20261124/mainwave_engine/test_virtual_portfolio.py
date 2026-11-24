from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from mainwave_engine import virtual_portfolio as VP, exit_policy as XP
from mainwave_engine import test_exit_policy as exit_tests


class VirtualPortfolioTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "positions.sqlite3"
        self.f, self.calendar = exit_tests.ExitPolicyTests().fixture()
        self.f["low"] = self.f.index_low
        self.position = {"id": "paper", "symbol": "sh_600001", "entry_date": self.calendar[70],
                         "entry_price": 100., "atr_fraction": .02, "status": "OPEN"}

    def evaluate(self, end):
        frame = self.f.iloc[:end].copy()
        with patch.object(XP, "enrich", side_effect=lambda f, _: f):
            return VP.evaluate(self.position, frame, self.calendar[end - 1], self.calendar, pd.DataFrame())

    def add(self, index=70):
        with patch.object(VP, "load_frame", return_value=self.f.iloc[:index + 1].copy()), \
                patch.object(VP, "market_calendar", return_value=self.calendar), \
                patch.object(VP.SA, "_mapping", return_value={"name": "Test"}), \
                patch.object(VP.SA, "_read_json", return_value={}):
            return VP.add("600001", self.calendar[index], self.db)

    def test_entry_close_ignores_prior_intraday_low_and_high(self):
        self.f.loc[70, ["index_low", "index_high"]] = [90., 120.]
        r = self.evaluate(71)
        self.assertEqual(r["status"], "HOLD")
        self.assertEqual(r["hard_floor"], 96.)
        self.assertEqual(r["events"], [])

    def test_hard_risk_latches_after_recovery_without_auto_sale(self):
        self.f.loc[71, "index_low"] = 95.
        r = self.evaluate(76)
        self.assertEqual(r["status"], "SELL")
        self.assertEqual(r["pending_reason"], "INITIAL_STOP")
        self.assertEqual(len(r["events"]), 1)
        self.assertEqual(self.position["status"], "OPEN")

    def test_strong_soft_confirmation_and_intraday_tolerance(self):
        self.f.loc[71, "index_high"] = 105.
        self.f.loc[72, ["exit_strong", "index_low"]] = [True, 98.]
        self.assertEqual(self.evaluate(73)["status"], "HOLD")
        self.f.loc[73:74, ["index_low", "index_close"]] = [97., 98.]
        self.assertEqual(self.evaluate(74)["status"], "SOFT_CONFIRMING")
        self.assertEqual(self.evaluate(75)["status"], "SELL")

    def test_partial_protection_is_advice_not_a_fill(self):
        self.f.loc[71, "index_high"] = 107.
        r = self.evaluate(75)
        self.assertEqual(r["status"], "REDUCE")
        self.assertEqual(r["events"][0]["reason"], "TARGET_3ATR")
        self.assertEqual(self.position["status"], "OPEN")

    def test_close_structural_failure_has_priority(self):
        self.f.loc[72, ["index_close", "index_low", "amount_ratio", "close_location"]] = [98., 97., 2., .2]
        r = self.evaluate(73)
        self.assertEqual(r["pending_reason"], "EXIT_STRUCTURE")

    def test_day20_extension_and_horizon_reuse_policy(self):
        r = self.evaluate(90)
        self.assertEqual(r["pending_reason"], "TIME_REVIEW_EXIT")
        self.f.loc[71:, ["index_close", "index_high", "index_low", "exit_strong"]] = [104., 105., 103., True]
        self.assertEqual(self.evaluate(90)["status"], "HOLD")
        self.assertEqual(self.evaluate(130)["pending_reason"], "HORIZON")

    def test_missing_stale_nonfinite_and_revised_prices_are_not_safe(self):
        self.f = self.f.drop(index=72).reset_index(drop=True)
        self.assertEqual(self.evaluate(75)["status"], "DATA_WAIT")
        self.f, self.calendar = exit_tests.ExitPolicyTests().fixture()
        self.f.loc[74, "index_low"] = float("inf")
        self.assertEqual(self.evaluate(75)["status"], "DATA_WAIT")
        self.f.loc[74, "index_low"] = 99.
        self.f.loc[70, "close"] = 101.
        self.assertIn("修订", self.evaluate(75)["reason"])

    def test_known_suspension_retains_pending_action(self):
        self.f.loc[71, "index_low"] = 95.
        frame = self.f.iloc[:72].copy()
        frame.attrs['suspended_days'] = self.calendar[72:75]
        with patch.object(XP, 'enrich', side_effect=lambda f, _: f):
            r = VP.evaluate(self.position, frame, self.calendar[74], self.calendar, pd.DataFrame())
        self.assertEqual(r['status'], 'DATA_WAIT')
        self.assertEqual(r['pending_action'], 'SELL')

    def test_adjustment_changes_raw_price_not_economic_return_or_fixed_risk(self):
        self.f.loc[71:, 'close'] = 50.
        r = self.evaluate(75)
        self.assertEqual(r['status'], 'HOLD')
        self.assertEqual(r['return_pct'], 0.)
        self.assertEqual(r['raw_return_pct'], -50.)
        self.assertEqual(r['hard_floor'], 48.)

    def test_future_rows_cannot_change_past_decision(self):
        with patch.object(XP, 'enrich', side_effect=lambda f, _: f):
            short = VP.evaluate(self.position, self.f.iloc[:75], self.calendar[74], self.calendar, pd.DataFrame())
            self.f.loc[76:, 'index_low'] = 1.
            full = VP.evaluate(self.position, self.f, self.calendar[74], self.calendar, pd.DataFrame())
        self.assertEqual(short, full)

    def test_add_uses_server_close_and_duplicate_keeps_original_entry(self):
        first = self.add()
        again = self.add(75)
        self.assertTrue(first['created'])
        self.assertFalse(again['created'])
        self.assertEqual(first['position'], again['position'])
        self.assertEqual(first['position']['entry_price'], 100.)
        self.assertEqual(len(VP.positions(self.db)), 1)

    def test_concurrent_add_only_creates_one_open_position(self):
        with patch.object(VP, 'close_quote', return_value=(self.f, self.f.iloc[70])), \
                patch.object(VP.SA, '_mapping', return_value={}), patch.object(VP.SA, '_read_json', return_value={}):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda _: VP.add('600001', self.calendar[70], self.db), range(2)))
        self.assertEqual(sum(r['created'] for r in results), 1)
        self.assertEqual(len(VP.positions(self.db)), 1)

    def test_stale_quote_future_date_and_tplus1_are_rejected(self):
        with patch.object(VP, 'load_frame', return_value=self.f.iloc[:70]), self.assertRaises(ValueError):
            VP.close_quote('sh_600001', self.calendar[70])
        with self.assertRaises(ValueError):
            VP.checked_date('2099-01-01')
        p = self.add()['position']
        with self.assertRaises(ValueError):
            VP.close_position(p['id'], p['entry_date'], self.db)

    def test_manual_sale_preserves_history_allows_new_episode_and_rejects_locked(self):
        p = self.add()['position']
        self.f.loc[75, 'high'] = 90.
        with patch.object(VP, 'close_quote', return_value=(self.f, self.f.iloc[75])):
            with self.assertRaises(ValueError):
                VP.close_position(p['id'], self.calendar[75], self.db)
        self.f.loc[75, 'high'] = 101.
        with patch.object(VP, 'close_quote', return_value=(self.f, self.f.iloc[75])):
            closed = VP.close_position(p['id'], self.calendar[75], self.db)
        self.assertEqual(closed['entry_price'], p['entry_price'])
        self.assertEqual(closed['status'], 'CLOSED')
        self.assertEqual(VP.close_position(p['id'], self.calendar[76], self.db), closed)
        self.assertTrue(self.add(76)['created'])
        self.assertEqual(len(VP.positions(self.db)), 2)

    def test_daily_scans_non_candidates_once_per_day_and_backfill_does_not_replace_latest(self):
        p = self.add()['position']
        def observation(pos, day):
            return {'as_of': day, 'status':'SELL', 'reason':'risk', 'events':[]}
        with patch.object(VP, 'scan_position', side_effect=observation) as scan:
            for day in (self.calendar[75], self.calendar[75], self.calendar[74]):
                payload = VP.run(day, self.db, Path(self.tmp.name) / 'reports')
                self.assertEqual(payload['summary']['SELL'], 1)
            self.assertEqual(scan.call_count, 3)
        listed = VP.listing(self.db)['positions'][0]
        self.assertEqual(listed['observation']['as_of'], self.calendar[75])
        self.assertEqual(listed['status'], 'OPEN')
        with closing(VP.connect(self.db)) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM observations').fetchone()[0], 2)
        with patch.object(VP, 'close_quote', return_value=(self.f, self.f.iloc[74])), self.assertRaises(VP.Conflict):
            VP.close_position(p['id'], self.calendar[74], self.db)
        with patch.object(VP, 'scan_position', side_effect=ValueError('missing')):
            payload = VP.run(self.calendar[76], self.db, Path(self.tmp.name) / 'reports')
        self.assertEqual(payload['summary']['status'], 'PARTIAL')
        self.assertEqual(VP.listing(self.db)['positions'][0]['last_actionable_observation']['status'], 'SELL')

    def test_api_validates_price_fields_origin_and_persists(self):
        from mainwave_visual_app import app
        with patch.dict(app.config, VIRTUAL_PORTFOLIO_DB=self.db), app.test_client() as client:
            self.assertEqual(client.get('/api/virtual-positions').json['positions'], [])
            for body in ({'symbol':'600001','date':self.calendar[70],'price':1}, {'symbol':1,'date':'2026-01-01'}):
                self.assertEqual(client.post('/api/virtual-positions', json=body).status_code, 400)
            self.assertEqual(client.post('/api/virtual-positions', json={}, headers={'Origin':'https://foreign.example'}).status_code, 403)
            self.add()
            response = client.post('/api/virtual-positions', json={'symbol':'600001','date':self.calendar[70]})
            self.assertEqual(response.status_code, 200)
            self.assertFalse(response.json['created'])
            self.assertEqual(len(client.get('/api/virtual-positions').json['positions']), 1)


if __name__ == '__main__':
    unittest.main()


class S4LadTests(unittest.TestCase):
    """S4_LAD advice on the paper close entry: 1/3 at +10%, 1/3 at +20%, rest on EMA20 / distribution / stop."""

    def setUp(self):
        self.f, self.calendar = exit_tests.ExitPolicyTests().fixture()
        self.f["volume"] = 100.
        self.position = {"id": "s4", "symbol": "sh_600001", "entry_date": self.calendar[70], "entry_price": 100.,
                         "atr_fraction": .02, "status": "OPEN", "strategy": "S4_LAD"}

    def evaluate(self, end):
        return VP.evaluate(self.position, self.f.iloc[:end].copy(), self.calendar[end - 1], self.calendar, pd.DataFrame())

    def test_quiet_hold(self):
        r = self.evaluate(75)
        self.assertEqual((r["status"], r["lad_remaining"], r["strategy"]), ("HOLD", 1, "S4_LAD"))
        self.assertAlmostEqual(r["hard_floor"], 90.)

    def test_first_target_is_a_reduce_advice_then_hold(self):
        self.f.loc[72, "index_high"] = 111.
        self.assertEqual(self.evaluate(73)["status"], "REDUCE")
        r = self.evaluate(74)
        self.assertEqual(r["status"], "HOLD")
        self.assertAlmostEqual(r["lad_remaining"], 2 / 3, places=4)
        self.assertAlmostEqual(r["lad_return_pct"], (1 / 3 * 1.10 + 2 / 3 * 1.0 - 1) * 100, places=6)

    def test_market_trend_exit_sells_the_rest_next_open(self):
        r = VP.evaluate(self.position, self.f.iloc[:75].copy(), self.calendar[74], self.calendar, pd.DataFrame(), g1={self.calendar[73]})
        self.assertEqual(r["status"], "SELL")
        self.assertIn("大盘趋势清仓", r["pending_reason"])
        self.assertEqual(r["events"][-1]["date"], self.calendar[73])
        self.assertEqual(self.evaluate(75)["status"], "HOLD", "没有 G1 的日子照旧持有")

    def test_stop_sells_the_rest(self):
        self.f.loc[72, "index_low"] = 89.
        r = self.evaluate(76)
        self.assertEqual(r["status"], "SELL")
        self.assertAlmostEqual(r["lad_return_pct"], -10.)
        self.assertEqual(self.position["status"], "OPEN", "只是建议，不自动平仓")

    def test_two_closes_under_ema20_exit_next_open(self):
        self.f.loc[60:69, "index_low"] = 90.  # a deeper pullback low, so only the EMA20 rule can fire
        self.f.loc[76:77, ["index_close", "index_low", "close"]] = 96.
        r = self.evaluate(78)
        self.assertEqual(r["status"], "SELL")
        self.assertEqual(r["pending_reason"], "两日收盘跌破 EMA20×0.97")

    def test_relaunch_candle_erased_sells_half_then_structure_denial_sells_the_rest(self):
        self.f.loc[60:69, "index_low"] = 92.
        self.f.loc[72, ["index_close", "close", "index_low"]] = 98.  # below the signal day's open (100)
        r = self.evaluate(73)
        self.assertEqual(r["status"], "REDUCE")
        self.assertIn("卖一半", r["reason"])
        r = self.evaluate(74)
        self.assertAlmostEqual(r["lad_remaining"], 0.5)
        self.f.loc[74, ["index_close", "close", "index_low"]] = 91.  # below the pullback low (92), above the stop (90)
        r = self.evaluate(75)
        self.assertEqual((r["status"], r["pending_reason"]), ("SELL", "收盘跌破回踩低点（回踩事件被否定）"))
        self.assertAlmostEqual(r["pullback_low"], 92.)

    def test_targets_after_the_warning_half_never_sell_more_than_is_held(self):
        # 2026-09-29 review probe: warning half, then +10% and +20% used to leave -1/6
        self.f.loc[60:69, "index_low"] = 92.
        self.f.loc[71, ["index_close", "close"]] = 98.                    # below the signal open: sell half next open
        self.f.loc[72, ["index_open", "index_high", "index_close", "close"]] = [99., 112., 110., 110.]
        self.f.loc[73, ["index_open", "index_high", "index_low", "index_close", "close"]] = [115., 125., 114., 122., 122.]
        r = self.evaluate(74)
        self.assertEqual(r["lad_remaining"], 0)
        self.assertEqual(r["status"], "SELL")
        self.assertIn("分批止盈后已无剩余", r["reason"])
        self.assertAlmostEqual(r["lad_return_pct"], (0.5 * .99 + 1 / 3 * 1.10 + 1 / 6 * 1.20 - 1) * 100, places=6)
        for end in range(72, 75):
            self.assertTrue(0 <= self.evaluate(end)["lad_remaining"] <= 1)

    def test_a_stored_version_replays_its_own_rules_and_g1_only_from_the_migration(self):
        g1 = {self.calendar[72]}
        def review(version, migration=VP.G1_MIGRATION):
            position = {**self.position, "exit_policy_version": version}
            with patch.object(VP, "G1_MIGRATION", migration):
                return VP.evaluate(position, self.f.iloc[:74].copy(), self.calendar[73], self.calendar, pd.DataFrame(), g1=g1)
        v3 = review("s4_lad_v3_20260929")
        self.assertEqual((v3["status"], v3["policy_version"], v3["g1_rule"]), ("SELL", "s4_lad_v3_20260929", "入场版本自带"))
        v2 = review("s4_lad_v2_20260926")
        self.assertEqual((v2["status"], v2["policy_version"]), ("HOLD", "s4_lad_v2_20260926"), "迁移日之前的 G1 不回放进旧版本")
        self.assertIn("迁移", v2["g1_rule"])
        self.assertEqual(review("s4_lad_v2_20260926", migration=self.calendar[72])["status"], "SELL", "迁移日起的 G1 适用")
        self.f.loc[71, ["index_close", "close"]] = 98.
        self.assertEqual(review("s4_lad_v1_20260925")["lad_remaining"], 1, "v1 没有分级否定")
        self.assertEqual(review("s4_lad_v9")["status"], "DATA_WAIT", "未知版本不用现行规则替代")

    def test_stored_anchors_are_used_instead_of_rebuilt_ones(self):
        self.f.loc[60:69, "index_low"] = 92.
        self.f.loc[72, ["index_close", "close", "index_low"]] = 94.
        rebuilt = self.evaluate(73)
        self.assertEqual((rebuilt["status"], rebuilt["anchor_source"]), ("REDUCE", "按行情重建"), "重建锚点：信号开盘 100、回踩低点 92，只触发减半")
        self.position.update(s4_pullback_low=95., s4_signal_open=90.)
        frozen = self.evaluate(73)
        self.assertEqual((frozen["status"], frozen["anchor_source"]), ("SELL", "记入时冻结"))
        self.assertEqual(frozen["pending_reason"], "收盘跌破回踩低点（回踩事件被否定）")
        self.assertAlmostEqual(frozen["pullback_low"], 95.)

    def test_stop_before_denial_keeps_the_hard_floor(self):
        self.f.loc[71, "index_low"] = 89.
        self.assertEqual(self.evaluate(72)["reason"].split("（")[-1], "止损 −10%）")

    def test_distribution_only_counts_after_a_ten_percent_close(self):
        self.f.loc[60:69, "index_low"] = 90.
        self.f.loc[70, "index_open"] = 94.  # the relaunch opened at 94, so 94.5 does not erase it
        self.f.loc[72, ["index_open", "index_close", "close", "volume"]] = [100., 94.5, 94.5, 300.]
        self.assertEqual(self.evaluate(73)["status"], "HOLD", "未盈利 10% 前放量大阴不算出货")
        self.f.loc[72, ["index_open", "index_close", "close", "volume", "index_high"]] = [100., 111., 111., 100., 111.]
        self.f.loc[73, ["index_open", "index_close", "close", "volume"]] = [111., 104., 104., 300.]
        r = self.evaluate(74)
        self.assertEqual((r["status"], r["pending_reason"]), ("SELL", "放量大阴线"))

    def test_add_keeps_the_s4_event_identity(self):
        db = Path(tempfile.mkdtemp()) / "vp.sqlite3"
        with patch.object(VP, "load_frame", return_value=self.f.iloc[:71].copy()), \
                patch.object(VP, "market_calendar", return_value=self.calendar), \
                patch.object(VP.SA, "_mapping", return_value={"name": "Test"}), \
                patch.object(VP.SA, "_read_json", return_value={}):
            p = VP.add("600001", self.calendar[70], db, strategy="S4_LAD",
                       meta={"s4_event_id": "sh_600001:2026-04-10", "other": 1})["position"]
        self.assertEqual((p["s4_event_id"], "other" in p, p["exit_policy_version"]), ("sh_600001:2026-04-10", False, VP.LAD_VERSION))

    def test_add_rejects_unknown_strategy(self):
        with self.assertRaises(ValueError):
            VP.add("600001", self.calendar[70], Path(tempfile.gettempdir()) / "never.sqlite3", strategy="X")
