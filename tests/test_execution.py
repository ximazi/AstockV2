"""成交模型的不变量测试。

覆盖 2026-11-24 修掉的四个撮合缺陷:
  1. 买侧也扣滑点;
  2. 到期日封死跌停不再整笔丢弃, 顺延到能成交那天;
  3. 停牌日不成交;
  4. 复权空间: 除权日的假跳空不再当成真跳空, 限价单改成盘中触及也能成交。
"""
import unittest

import pandas as pd

from astockv2.execution import (ExitParams, simulate_entry, simulate_entry_limit,
                                simulate_exit, with_indicators)


def flat(n=60, close=100.0, preclose=None, limit_pct=0.10, tradestatus=1,
         volume=1000.0, pct_chg=0.0):
    '''一根没有波动的日线, 用来当底稿。'''
    rows = []
    for _ in range(n):
        rows.append({'open': close, 'high': close * 1.01, 'low': close * 0.99,
                     'close': close, 'preclose': close if preclose is None else preclose,
                     'volume': volume, 'tradestatus': tradestatus,
                     'pct_chg': pct_chg, 'amount': close * volume,
                     'turnover': 1.0})
    return rows


def build(rows, limit_pct=0.10):
    df = pd.DataFrame(rows)
    df['date'] = pd.date_range('2026-01-05', periods=len(df), freq='B')
    df['isST'] = 0
    out = with_indicators(df)
    out['limit_pct'] = limit_pct
    out.attrs['limit_pct'] = limit_pct
    return out


class EntryTests(unittest.TestCase):
    def test_limit_up_open_is_refused(self):
        rows = flat(40)
        rows[30]['close'] = 100.0
        rows[30]['preclose'] = 100.0
        # 次日开盘即涨停
        rows[31].update({'open': 110.0, 'high': 110.0, 'low': 110.0, 'close': 110.0,
                         'preclose': 100.0})
        b = build(rows)
        e = simulate_entry(b, 30, ExitParams(use_adjusted=False))
        self.assertEqual(e['status'], 'ENTRY_LIMIT_UP')

    def test_ex_dividend_drop_is_not_a_fake_gap_in_adjusted_space(self):
        rows = flat(40)
        rows[30].update({'open': 100.0, 'high': 100.0, 'low': 100.0, 'close': 100.0,
                         'preclose': 100.0})
        # 次日除权: 原始价掉 10%, 但当日真实涨跌幅是 0
        rows[31].update({'open': 90.0, 'high': 90.0, 'low': 90.0, 'close': 90.0,
                         'preclose': 90.0, 'pct_chg': 0.0})
        b = build(rows)
        raw = simulate_entry(b, 30, ExitParams(use_adjusted=False))
        adj = simulate_entry(b, 30, ExitParams(use_adjusted=True))
        self.assertEqual(raw['status'], 'OK')
        self.assertEqual(adj['status'], 'OK')
        self.assertAlmostEqual(raw['entry_price'], 90.0, places=6)
        self.assertAlmostEqual(adj['entry_price'], 100.0, places=6,
                               msg='复权空间的入场价不应被除权缺口带偏')

    def test_adjusted_series_is_continuous_across_ex_dividend(self):
        rows = flat(40)
        rows[20].update({'close': 100.0, 'open': 100.0, 'high': 100.0, 'low': 100.0})
        rows[21].update({'close': 90.0, 'open': 90.0, 'high': 90.0, 'low': 90.0,
                         'preclose': 90.0, 'pct_chg': 0.0})
        b = build(rows)
        r = b['adj_close'].pct_change().dropna()
        self.assertTrue((r.abs() < 1e-9).all(),
                        '复权序列里除权日不应出现断层: ' + str(r.abs().max()))

    def test_limit_order_fills_intraday_when_low_touches_the_limit(self):
        rows = flat(40)
        rows[30].update({'close': 100.0, 'open': 100.0, 'high': 100.0, 'low': 100.0,
                         'preclose': 100.0})
        # 次日高开 3%, 但盘中回落到 100 以下 -> 限价单能成交
        rows[31].update({'open': 103.0, 'high': 104.0, 'low': 99.0, 'close': 101.0,
                         'preclose': 100.0})
        b = build(rows)
        e = simulate_entry_limit(b, 30, ExitParams(use_adjusted=False),
                                 max_chase=0.0, wait_bars=1)
        self.assertEqual(e['status'], 'OK')
        self.assertAlmostEqual(e['entry_price'], 100.0, places=6,
                               msg='盘中触及限价应按限价成交')

    def test_limit_order_is_not_filled_on_a_one_word_limit_up(self):
        rows = flat(40)
        rows[30].update({'close': 100.0, 'open': 100.0, 'high': 100.0, 'low': 100.0,
                         'preclose': 100.0})
        rows[31].update({'open': 110.0, 'high': 110.0, 'low': 110.0, 'close': 110.0,
                         'preclose': 100.0})
        b = build(rows)
        # 只给次日一天: 一字涨停, 挂单排不上, 不成交
        e0 = simulate_entry_limit(b, 30, ExitParams(use_adjusted=False),
                                  max_chase=0.0, wait_bars=0)
        self.assertEqual(e0['status'], 'ENTRY_LIMIT_NOT_FILLED')
        # 多等一天: 一字涨停那天跳过, 第二天正常成交
        e1 = simulate_entry_limit(b, 30, ExitParams(use_adjusted=False),
                                  max_chase=0.0, wait_bars=1)
        self.assertEqual(e1['status'], 'OK')
        self.assertEqual(e1['waited'], 1, '一字涨停当天不应成交')


class ExitTests(unittest.TestCase):
    def _run(self, rows, params, entry_idx=30, stop=50.0, price=100.0):
        b = build(rows)
        atr0 = float(b['atr20'].iloc[entry_idx])
        return b, simulate_exit(b, entry_idx, price, stop, params, atr0)

    def test_buy_side_slippage_is_charged(self):
        rows = flat(60)
        p0 = ExitParams(stop_atr=2.0, targets=(), horizon=5, cost_bp=30.0,
                        slippage_bp=0.0, use_adjusted=False)
        p3 = ExitParams(stop_atr=2.0, targets=(), horizon=5, cost_bp=30.0,
                        slippage_bp=3.0, use_adjusted=False)
        _, r0 = self._run(rows, p0)
        _, r3 = self._run(rows, p3)
        self.assertEqual(r0['status'], 'CLOSED')
        self.assertEqual(r3['status'], 'CLOSED')
        self.assertLess(r3['net_ret'], r0['net_ret'],
                        '买侧滑点必须扣掉, 否则往返成本被少算一半')

    def test_locked_limit_down_at_horizon_is_deferred_not_dropped(self):
        rows = flat(60)
        # 到期日封死跌停
        rows[34].update({'open': 90.0, 'high': 90.0, 'low': 90.0, 'close': 90.0,
                         'preclose': 100.0})
        rows[35].update({'open': 95.0, 'high': 95.0, 'low': 95.0, 'close': 95.0,
                         'preclose': 90.0})
        p = ExitParams(stop_atr=2.0, targets=(), horizon=5, cost_bp=30.0,
                       slippage_bp=0.0, use_adjusted=False)
        b, res = self._run(rows, p)
        self.assertEqual(res['status'], 'CLOSED',
                         '到期卖不掉应该顺延, 不能整笔丢掉')
        self.assertEqual(res['reason'], 'HORIZON_DEFERRED')
        self.assertEqual(int(res['exit_idx']), 35)

    def test_suspended_bar_is_not_tradable(self):
        rows = flat(60)
        rows[34].update({'tradestatus': 0, 'volume': 0.0})
        rows[35].update({'open': 95.0, 'high': 95.0, 'low': 95.0, 'close': 95.0,
                         'preclose': 100.0})
        p = ExitParams(stop_atr=2.0, targets=(), horizon=5, cost_bp=30.0,
                       slippage_bp=0.0, use_adjusted=False)
        b, res = self._run(rows, p)
        self.assertEqual(res['status'], 'CLOSED')
        self.assertEqual(int(res['exit_idx']), 35,
                         '停牌日不能成交, 顺延到下一个能成交的日子')

    def test_truncated_forward_window_is_flagged(self):
        rows = flat(40)
        p = ExitParams(stop_atr=2.0, targets=(), horizon=60, cost_bp=30.0,
                       slippage_bp=0.0, use_adjusted=False)
        b, res = self._run(rows, p)
        self.assertEqual(res['status'], 'CLOSED')
        self.assertTrue(res['data_truncated'],
                        '数据不够走完持仓周期时必须标记出来')

    def test_full_horizon_window_is_not_flagged(self):
        rows = flat(60)
        p = ExitParams(stop_atr=2.0, targets=(), horizon=5, cost_bp=30.0,
                       slippage_bp=0.0, use_adjusted=False)
        b, res = self._run(rows, p)
        self.assertEqual(res['status'], 'CLOSED')
        self.assertFalse(res['data_truncated'])


if __name__ == '__main__':
    unittest.main()
