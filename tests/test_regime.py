'''市场环境分层的不变量测试。

跑法:
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
'''
from __future__ import annotations

import numpy as np
import pandas as pd

from astockv2.regime import (attach_regime, load_index, prev_trading_day_map,
                             regime_frame)


def test_index_loads_and_is_sorted():
    df = load_index('hs300')
    assert len(df) > 1000
    assert df['date'].is_monotonic_increasing
    assert df['close'].notna().all()


def test_ema_slope_definition():
    '''slope = EMA100[t] / EMA100[t-lookback] - 1, 纯上涨序列应为正。'''
    up = np.linspace(100, 200, 300)
    ema = pd.Series(up).ewm(span=100, adjust=False).mean()
    assert ((ema / ema.shift(10) - 1).dropna() > 0).all()
    down = np.linspace(200, 100, 300)
    ema2 = pd.Series(down).ewm(span=100, adjust=False).mean()
    assert ((ema2 / ema2.shift(10) - 1).dropna() < 0).all()


def test_regime_labels_are_two_valued():
    rf = regime_frame('hs300', 100, 10)
    assert set(rf['regime'].unique()) <= {'DOWN', 'UP', 'UNKNOWN'}
    assert rf['regime'].iloc[0] == 'UNKNOWN'


def test_attach_regime_marks_unknown():
    df = pd.DataFrame({'anchor_day': ['2024-01-02', '1999-01-01']})
    out = attach_regime(df, 'anchor_day', {'2024-01-02': 'UP'})
    assert list(out['regime']) == ['UP', 'UNKNOWN']


def test_prev_trading_day_map():
    pm = prev_trading_day_map('hs300')
    days = list(load_index('hs300')['date'].dt.strftime('%Y-%m-%d'))
    assert pm[days[1]] == days[0]
    assert days[0] not in pm
