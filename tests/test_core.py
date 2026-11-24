"""核心不变量测试。

跑法:
    python -m pytest tests -q
"""
from __future__ import annotations

import pandas as pd

from astockv2.bars import bars, index_of
from astockv2.config import V1_RESULTS
from astockv2.execution import ExitParams, limit_pct, simulate_exit

LEDGER = V1_RESULTS / '20261119_validated'


def test_limit_pct():
    assert limit_pct('sh_600000') == 0.10
    assert limit_pct('sz_300001') == 0.20
    assert limit_pct('sh_688001') == 0.20
    assert limit_pct('sh_600001', is_st=True) == 0.05


def test_atr_matches_v1_definition():
    # ATR20 = true_range.shift(1).rolling(20, min_periods=15).mean()
    b = bars('sh_600000')
    prev = b['close'].shift(1)
    tr = pd.concat([b['high'] - b['low'],
                    (b['high'] - prev).abs(),
                    (b['low'] - prev).abs()], axis=1).max(axis=1)
    expect = tr.shift(1).rolling(20, min_periods=15).mean()
    got = b['atr20']
    diff = (got.dropna() - expect.dropna()).abs().max()
    assert diff < 1e-9, diff


def test_ledger_loads():
    f = LEDGER / 'ledger_2026_B_30bp.csv'
    df = pd.read_csv(f)
    assert 'price' in df.columns
    assert len(df[df['status'] == 'CLOSED']) == 28


def test_baseline_reproduces_v1_ledger():
    '''基线退出必须能复现 v1 账本, 否则后面的变体比较没有意义。'''
    from astockv2.exit_lab import V0, load_trades, rerun
    trades = load_trades(LEDGER)
    base = rerun(trades, V0)
    assert len(base) >= 150
    same_date = (base['exit_date'].astype(str) == base['ledger_exit'].astype(str)).mean()
    assert same_date > 0.95, same_date
    diff = (base['pnl'] - base['ledger_pnl']).abs()
    assert diff.median() < 1.0, diff.median()
