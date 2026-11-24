"""统一的成交与退出模型。

v1 的研究撮合 (research/route_followup.py:simulate) 已经处理了 T+1、
涨跌停锁死、退出受阻和分批止盈, 但有两处已知缺陷:
  1. 完全没有滑点, 五个 harness 都没有这个参数;
  2. 分批止盈按 max(开盘价, 目标价) 成交, 默认盘中穿过目标位就能成交,
     不扣滑点, 也不模拟跳空穿过。
这里把模型抽成一份可参数化的实现, 让退出规则实验和对照组实验共用同一套
成交假设, 避免两个实验的差异其实来自成交口径。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from .bars import load

# 板块涨跌停幅度: 主板 10%, 创业板/科创板 20%, ST 5%
LIMIT_MAIN = 0.10
LIMIT_GROWTH = 0.20
LIMIT_ST = 0.05


def limit_pct(symbol: str, is_st: bool = False) -> float:
    if is_st:
        return LIMIT_ST
    return LIMIT_GROWTH if symbol[3:5] in ('30', '68') else LIMIT_MAIN


def with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """补上 ATR20 与 10 日最低价。

    ATR20 定义与 v1 一致 (mainwave_engine/phase_events.py:85):
    true_range.shift(1).rolling(20, min_periods=15).mean()
    """
    out = df.copy()
    prev_close = out['close'].shift(1)
    tr = pd.concat([
        out['high'] - out['low'],
        (out['high'] - prev_close).abs(),
        (out['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out['atr20'] = tr.shift(1).rolling(20, min_periods=15).mean()
    out['low10'] = out['low'].rolling(10, min_periods=5).min()
    out['high20_prev'] = out['high'].shift(1).rolling(20, min_periods=15).max()
    out['preclose_calc'] = prev_close
    return out


@dataclass
class ExitParams:
    '''退出规则参数。默认值即 v1 的 structure36_20 口径。'''

    stop_atr: float = 2.0          # 初始止损距离 (ATR)
    stop_source: str = 'ledger'    # ledger 用账本冻结的止损; atr 用 stop_atr 重算
    stop_on_close: bool = False    # True 时止损按收盘价判定, 不按盘中最低价
    use_structure: bool = True     # 是否再取突破无效位与 ATR 止损中更严者
    trail: str = 'low10'           # none / low10 / atr
    trail_atr: float = 2.5         # trail='atr' 时的回撤距离
    activate_atr: float = 2.0      # 浮盈达到多少 ATR 后启动移动止损
    targets: tuple = ((3.0, 0.5), (6.0, 0.25))   # (目标 ATR 倍数, 减仓比例)
    horizon: int = 20              # 持仓根数上限
    cost_bp: float = 30.0          # 双边总成本
    slippage_bp: float = 3.0       # 单边滑点, 在成交价上直接扣
    fill_target_at_open: bool = False  # True 时盘中穿越目标价按开盘价保守成交


def _slip(price: float, slippage_bp: float, side: str) -> float:
    adj = price * (slippage_bp / 10000.0)
    return price - adj if side == 'sell' else price + adj


def simulate_exit(bars: pd.DataFrame, entry_idx: int, entry_price: float,
                  initial_stop: float, params: ExitParams,
                  atr0: float) -> dict:
    '''在一根已含指标的火线标的上模拟退出。

    口径与 v1 research/route_followup.py:simulate 的 structure36_20 分支一致:
    - T+1: 入场当日触及止损只能挂到次日开盘;
    - 跌停锁死时不能卖, 顺延到下一个能成交的开盘;
    - 分批止盈按目标价成交, 叠加滑点;
    - 移动止损在当日收盘后更新, 次日才生效。
    '''
    n = len(bars)
    end = min(entry_idx + params.horizon - 1, n - 1)
    stop = float(initial_stop)
    remaining = 1.0
    cash = 0.0
    peak = float(entry_price)
    trough = float(entry_price)
    activated = False
    pending = False
    pending_exit = ''
    targets = list(params.targets)
    reason = ''
    exit_idx = None
    exit_price = float('nan')
    for j in range(entry_idx, end + 1):
        op = float(bars['open'].iloc[j])
        hi = float(bars['high'].iloc[j])
        lo = float(bars['low'].iloc[j])
        cl = float(bars['close'].iloc[j])
        pre = float(bars['preclose'].iloc[j]) if j > 0 else op
        lim = (float(bars['limit_pct'].iloc[j])
               if 'limit_pct' in bars.columns
               else bars.attrs.get('limit_pct', 0.10))
        lower = math.floor(pre * (1 - lim) * 100 + .5) / 100
        locked = (hi <= lower + .001) or (op <= lower + .001)
        fill = float('nan')
        breach = (cl <= stop) if params.stop_on_close else (lo <= stop)
        if j == entry_idx:
            if breach:
                pending = True
        elif pending and not locked:
            reason, fill = 'DEFERRED_STOP', op
        elif pending_exit and not locked:
            reason, fill = pending_exit, op
        elif breach:
            if locked:
                pending = True
            else:
                reason = 'INITIAL_STOP' if stop == initial_stop else 'TRAIL_STOP'
                fill = min(op, stop) if not params.stop_on_close else cl
        if reason:
            fill = _slip(fill, params.slippage_bp, 'sell')
            cash += remaining * fill / entry_price
            remaining = 0.0
            exit_idx, exit_price = j, fill
        elif j > entry_idx and not locked:
            for target_atr, weight in list(targets):
                if remaining <= 1e-10:
                    break
                target = entry_price + target_atr * atr0
                if hi >= target:
                    px = op if params.fill_target_at_open else max(op, target)
                    px = _slip(px, params.slippage_bp, 'sell')
                    cash += weight * px / entry_price
                    remaining -= weight
                    targets.remove((target_atr, weight))
        if remaining > 1e-10 and j == end:
            if locked:
                pending_exit = 'HORIZON_DEFERRED'
                continue
            fill = _slip(cl, params.slippage_bp, 'sell')
            cash += remaining * fill / entry_price
            remaining = 0.0
            reason = 'HORIZON'
            exit_idx, exit_price = j, fill
        if remaining <= 1e-10:
            break
        peak = max(peak, hi)
        trough = min(trough, lo)
        activated = activated or peak >= entry_price + params.activate_atr * atr0
        if activated:
            if params.trail == 'low10':
                lv = bars['low10'].iloc[j]
                if lv == lv:
                    stop = max(stop, float(lv))
            elif params.trail == 'atr':
                stop = max(stop, peak - params.trail_atr * atr0)
    if remaining > 1e-10:
        return {'status': 'OPEN', 'reason': '', 'exit_idx': None,
                'exit_price': float('nan'), 'gross_ret': 0.0, 'net_ret': 0.0,
                'mfe': peak / entry_price - 1.0, 'mae': trough / entry_price - 1.0,
                'bars_held': 0}
    gross = cash - 1.0
    net = gross - params.cost_bp / 10000.0
    den = entry_price - (float(initial_stop))
    return {
        'status': 'CLOSED',
        'reason': reason,
        'exit_idx': exit_idx,
        'exit_date': str(bars['date'].iloc[exit_idx].date()),
        'exit_price': exit_price,
        'gross_ret': gross,
        'net_ret': net,
        'net_r': net * entry_price / den if den else float('nan'),
        'mfe': peak / entry_price - 1.0,
        'mae': trough / entry_price - 1.0,
        'bars_held': int(exit_idx - entry_idx + 1),
    }


def simulate_entry(bars: pd.DataFrame, signal_idx: int, params: ExitParams,
                   structure_stop=None) -> dict:
    '''次日开盘入场判定, 口径与 v1 research/route_followup.py:simulate 一致。

    不成交的情形: 无次日交易数据、停牌、ST、开盘涨停、跳空超过 1 ATR、
    开盘已跌破结构止损位。
    '''
    entry_idx = signal_idx + 1
    if entry_idx >= len(bars):
        return {'status': 'NO_ENTRY_BAR'}
    row = bars.iloc[entry_idx]
    prev = bars.iloc[signal_idx]
    if int(row['tradestatus']) != 1 or float(row['volume']) <= 0:
        return {'status': 'ENTRY_SUSPENDED'}
    if float(row.get('isST', 0) or 0) != 0:
        return {'status': 'ENTRY_ST'}
    atr0 = float(bars['atr20'].iloc[signal_idx])
    if not atr0 or atr0 != atr0 or atr0 <= 0:
        return {'status': 'INVALID_ATR'}
    price = float(row['open'])
    lim = (float(row['limit_pct'])
           if 'limit_pct' in bars.columns
           else bars.attrs.get('limit_pct', 0.10))
    pre = float(row['preclose']) if row['preclose'] == row['preclose'] else float(prev['close'])
    upper = math.floor(pre * (1 + lim) * 100 + .5) / 100
    if price >= upper - 0.001:
        return {'status': 'ENTRY_LIMIT_UP'}
    signal_close = float(prev['close'])
    if price - signal_close > atr0:
        return {'status': 'ENTRY_GAP'}
    initial_stop = price - params.stop_atr * atr0
    if structure_stop is not None:
        initial_stop = max(initial_stop, float(structure_stop))
    if price <= initial_stop:
        return {'status': 'ENTRY_BELOW_STRUCTURE'}
    return {
        'status': 'OK',
        'entry_idx': entry_idx,
        'entry_date': str(row['date'].date()),
        'entry_price': price,
        'atr0': atr0,
        'initial_stop': initial_stop,
        'risk_per_share': price - initial_stop,
    }


def simulate_entry_limit(bars: pd.DataFrame, signal_idx: int, params: ExitParams,
                         max_chase: float = 0.0, wait_bars: int = 5,
                         structure_stop=None) -> dict:
    '''限价入场: 不高于信号日收盘才买, 最多等 wait_bars 根。

    这是"不追高"的可执行版本。信号日收盘价是决策时已知的价格, 挂一个
    不高于它的限价单在真实盘口里是能做到的。

    max_chase=0 表示买价不得高于信号日收盘; wait_bars=0 表示次日不满足
    就直接放弃。
    '''
    n = len(bars)
    signal_close = float(bars['close'].iloc[signal_idx])
    limit_price = signal_close * (1.0 + max_chase)
    last = min(signal_idx + 1 + wait_bars, n - 1)
    for j in range(signal_idx + 1, last + 1):
        row = bars.iloc[j]
        if int(row['tradestatus']) != 1 or float(row['volume']) <= 0:
            continue
        if float(row.get('isST', 0) or 0) != 0:
            continue
        price = float(row['open'])
        if price > limit_price:
            continue
        lim = (float(row['limit_pct'])
               if 'limit_pct' in bars.columns
               else bars.attrs.get('limit_pct', 0.10))
        pre = float(row['preclose']) if row['preclose'] == row['preclose'] else float(bars['close'].iloc[j - 1])
        upper = math.floor(pre * (1 + lim) * 100 + .5) / 100
        if price >= upper - 0.001:
            continue
        atr0 = float(bars['atr20'].iloc[j - 1])
        if not atr0 or atr0 != atr0 or atr0 <= 0:
            continue
        if j - signal_idx > 1 and price - float(bars['close'].iloc[j - 1]) > atr0:
            continue
        initial_stop = price - params.stop_atr * atr0
        if structure_stop is not None:
            initial_stop = max(initial_stop, float(structure_stop))
        if price <= initial_stop:
            continue
        return {
            'status': 'OK',
            'entry_idx': j,
            'entry_date': str(row['date'].date()),
            'entry_price': price,
            'atr0': atr0,
            'initial_stop': initial_stop,
            'risk_per_share': price - initial_stop,
            'waited': j - signal_idx - 1,
        }
    return {'status': 'ENTRY_LIMIT_NOT_FILLED'}
