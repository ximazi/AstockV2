"""统一的成交与退出模型。

v1 的研究撮合 (research/route_followup.py:simulate) 已经处理了 T+1、
涨跌停锁死、退出受阻和分批止盈, 但有几处已知缺陷:

  1. 完全没有滑点;
  2. 分批止盈按 max(开盘价, 目标价) 成交, 不扣滑点, 也不模拟跳空穿过;
  3. 用的是未复权价: 除权除息日的价格跳空会被当成真实跳空, ATR 与止损
     都会被算错;
  4. 持仓到期遇到跌停锁死时, 这笔交易会被整笔丢掉 (而不是顺延到能成交的
     那天), 等于把卖不掉的亏损单从样本里删掉。

这里把模型抽成一份可参数化的实现, 让退出规则实验和对照组实验共用同一套
成交假设, 避免两个实验的差异其实来自成交口径。

价格空间的两套口径 (重要):
  * 原始价 (raw): 涨跌停价、限价单、交易所口径的一切, 都用原始价;
  * 复权空间 (adjusted): ATR、止损位、止盈位, 用因果复权价, 避免除权
    日的假跳空。ExitParams.use_adjusted 决定用哪一套。
原始价上的 atr20 / low10 / high20_prev 保持不变, 因为 exit_lab 的基线
对账必须与 v1 用同一套价格。
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
    """补上因果复权价、ATR20 与 10 日最低价。

    复权序列用 pct_chg (缺失时用 close/preclose-1) 累乘, 以第一根收盘价为
    基准, 因此 adj_close[i]/adj_close[i-1] 恒等于当日真实涨跌幅, 除权日
    不会留下价格断层。原始价上的 atr20 / low10 / high20_prev 语义不变。
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

    close = pd.to_numeric(out['close'], errors='coerce')
    open_ = pd.to_numeric(out['open'], errors='coerce')
    high = pd.to_numeric(out['high'], errors='coerce')
    low = pd.to_numeric(out['low'], errors='coerce')
    if 'preclose' in out.columns:
        pre = pd.to_numeric(out['preclose'], errors='coerce')
    else:
        pre = close.shift(1)
    if 'pct_chg' in out.columns:
        ret = pd.to_numeric(out['pct_chg'], errors='coerce') / 100.0
    else:
        ret = pd.Series(float('nan'), index=out.index)
    fallback = (close / pre - 1.0).where(pre > 0)
    ret = ret.fillna(fallback).fillna(0.0).clip(-0.95, 3.0)
    if len(ret):
        ret = ret.copy()
        ret.iloc[0] = 0.0          # 第一根没有前一日, 不计入
    base = float(close.iloc[0]) if len(close) and close.iloc[0] == close.iloc[0] else 1.0
    adj_close = base * (1.0 + ret).cumprod()
    scale = adj_close / close.replace(0, float('nan'))
    out['adj_scale'] = scale
    out['adj_close'] = adj_close
    out['adj_open'] = open_ * scale
    out['adj_high'] = high * scale
    out['adj_low'] = low * scale

    adj_prev = out['adj_close'].shift(1)
    tr_adj = pd.concat([
        out['adj_high'] - out['adj_low'],
        (out['adj_high'] - adj_prev).abs(),
        (out['adj_low'] - adj_prev).abs(),
    ], axis=1).max(axis=1)
    out['atr20_adj'] = tr_adj.shift(1).rolling(20, min_periods=15).mean()
    out['low10_adj'] = out['adj_low'].rolling(10, min_periods=5).min()
    out['high20_prev_adj'] = out['adj_high'].shift(1).rolling(20, min_periods=15).max()
    return out


@dataclass
class ExitParams:
    '''退出规则参数。默认启用复权空间与双边滑点。'''
    stop_atr: float = 2.0          # 初始止损距离 (ATR)
    stop_source: str = 'ledger'    # ledger 用账本冻结的止损; atr 用 stop_atr 重算
    stop_on_close: bool = False    # True 时止损按收盘价判定, 不按盘中最低价
    use_structure: bool = True     # 是否再取突破无效位与 ATR 止损中更严者
    trail: str = 'low10'           # none / low10 / atr
    trail_atr: float = 2.5         # trail='atr' 时的回撤距离
    activate_atr: float = 2.0      # 浮盈达到多少 ATR 后启动移动止损
    targets: tuple = ((3.0, 0.5), (6.0, 0.25))   # (目标 ATR 倍数, 减仓比例)
    targets_pct: tuple = ()        # ((涨幅, 减仓比例), ...) 固定百分比止盈
    stop_pct: float = 0.0          # >0 时用固定百分比止损, 优先于 ATR 止损
    horizon: int = 20              # 持仓根数上限
    cost_bp: float = 30.0          # 双边总成本
    slippage_bp: float = 3.0       # 单边滑点, 在成交价上直接扣
    fill_target_at_open: bool = False  # True 时盘中穿越目标价按开盘价保守成交
    use_adjusted: bool = True      # False 时用原始价 (对账 v1 旧账本时用)


def _slip(price: float, slippage_bp: float, side: str) -> float:
    adj = price * (slippage_bp / 10000.0)
    return price - adj if side == 'sell' else price + adj


def _st_flag(row) -> float:
    v = float(row.get('isST', 0) or 0)
    return 0.0 if v != v else v


def _limit_pct_of(bars: pd.DataFrame, j: int) -> float:
    if 'limit_pct' in bars.columns:
        return float(bars['limit_pct'].iloc[j])
    return float(bars.attrs.get('limit_pct', LIMIT_MAIN))


def simulate_exit(bars: pd.DataFrame, entry_idx: int, entry_price: float,
                  initial_stop: float, params: ExitParams,
                  atr0: float) -> dict:
    '''退出撮合: 双边滑点、区分开盘跌停与全天封死、停牌不成交、到期顺延。

    口径与 v1 research/route_followup.py:simulate 的 structure36_20 分支一致:
    - T+1: 入场当日触及止损只能挂到次日开盘;
    - 跌停锁死时不能卖, 顺延到下一个能成交的开盘;
    - 分批止盈按目标价成交, 叠加滑点;
    - 移动止损在当日收盘后更新, 次日才生效。

    与 v1 的差别 (都是修正, 不是放宽):
    - 买侧也扣滑点;
    - 到期日封死跌停不再整笔丢弃, 顺延到能成交那天;
    - 停牌日不成交, 顺延;
    - 默认在复权空间计算 ATR 与止损。
    '''
    n = len(bars)
    end = min(entry_idx + params.horizon - 1, n - 1)
    # 数据不够走完整个持仓周期: 这笔的持有期被截断, 与跑满周期的交易不可比
    data_truncated = bool(entry_idx + params.horizon - 1 > n - 1)
    use_adj = params.use_adjusted and 'adj_open' in bars.columns
    if use_adj:
        col_o, col_h, col_l, col_c = 'adj_open', 'adj_high', 'adj_low', 'adj_close'
        col_low10 = 'low10_adj'
    else:
        col_o, col_h, col_l, col_c = 'open', 'high', 'low', 'close'
        col_low10 = 'low10'

    buy_cost = _slip(float(entry_price), params.slippage_bp, 'buy')
    stop = float(initial_stop)
    remaining = 1.0
    cash = 0.0
    peak = float(entry_price)
    trough = float(entry_price)
    activated = False
    pending = False
    pending_exit = ''
    targets = []
    for atr_mult, weight in params.targets:
        targets.append((entry_price + atr_mult * atr0, weight))
    for pct, weight in params.targets_pct:
        targets.append((entry_price * (1.0 + pct), weight))
    reason = ''
    exit_idx = None
    exit_price = float('nan')

    for j in range(entry_idx, n):
        if j > end and not (pending or pending_exit):
            break
        # 停牌或无成交量的日子不能成交
        tradable = (int(bars['tradestatus'].iloc[j]) == 1
                    and float(bars['volume'].iloc[j]) > 0)
        if not tradable:
            if j >= end and remaining > 1e-10:
                pending_exit = pending_exit or 'HORIZON_DEFERRED'
            continue

        op = float(bars[col_o].iloc[j])
        hi = float(bars[col_h].iloc[j])
        lo = float(bars[col_l].iloc[j])
        cl = float(bars[col_c].iloc[j])
        raw_op = float(bars['open'].iloc[j])
        raw_hi = float(bars['high'].iloc[j])
        pre = float(bars['preclose'].iloc[j]) if j > 0 else raw_op
        lim = _limit_pct_of(bars, j)
        lower = math.floor(pre * (1 - lim) * 100 + .5) / 100
        open_locked = raw_op <= lower + .001      # 开盘即跌停, 开盘卖不掉
        day_locked = raw_hi <= lower + .001       # 全天封死跌停 (蕴含 open_locked)
        fill = float('nan')
        breach = (cl <= stop) if params.stop_on_close else (lo <= stop)

        # 持仓期最高/最低, 含离场当根
        peak = max(peak, hi)
        trough = min(trough, lo)

        if j == entry_idx:
            if breach:
                pending = True                    # T+1, 当日不能卖
        elif pending and not open_locked:
            reason, fill = 'DEFERRED_STOP', op
        elif pending_exit and not open_locked:
            reason, fill = pending_exit, op
        elif breach:
            if day_locked:
                pending = True                    # 封死跌停, 顺延
            else:
                reason = 'INITIAL_STOP' if stop == initial_stop else 'TRAIL_STOP'
                fill = min(op, stop) if not params.stop_on_close else cl

        if reason:
            fill = _slip(fill, params.slippage_bp, 'sell')
            cash += remaining * fill / buy_cost
            remaining = 0.0
            exit_idx, exit_price = j, fill
        elif j > entry_idx and not day_locked:
            for level, weight in list(targets):
                if remaining <= 1e-10:
                    break
                if hi >= level:
                    px = op if params.fill_target_at_open else max(op, level)
                    px = _slip(px, params.slippage_bp, 'sell')
                    cash += weight * px / buy_cost
                    remaining -= weight
                    targets.remove((level, weight))

        if remaining > 1e-10 and j == end:
            if day_locked:
                pending_exit = pending_exit or 'HORIZON_DEFERRED'
            else:
                fill = _slip(cl, params.slippage_bp, 'sell')
                cash += remaining * fill / buy_cost
                remaining = 0.0
                reason = 'HORIZON'
                exit_idx, exit_price = j, fill

        if remaining <= 1e-10:
            break

        activated = activated or peak >= entry_price + params.activate_atr * atr0
        if activated:
            if params.trail == 'low10':
                lv = bars[col_low10].iloc[j]
                if lv == lv:
                    stop = max(stop, float(lv))
            elif params.trail == 'atr':
                stop = max(stop, peak - params.trail_atr * atr0)

    if remaining > 1e-10:
        # 数据不够走完持仓周期, 这笔无法评估, 记 OPEN 由调用方显式处理
        return {'status': 'OPEN', 'reason': '', 'exit_idx': None,
                'exit_price': float('nan'), 'gross_ret': 0.0, 'net_ret': 0.0,
                'mfe': peak / entry_price - 1.0, 'mae': trough / entry_price - 1.0,
                'bars_held': 0, 'deferred': bool(pending_exit),
                'data_truncated': data_truncated}
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
        'deferred': reason == 'HORIZON_DEFERRED',
        'data_truncated': data_truncated,
    }


def simulate_entry(bars: pd.DataFrame, signal_idx: int, params: ExitParams,
                   structure_stop=None) -> dict:
    '''次日开盘入场判定, 口径与 v1 research/route_followup.py:simulate 一致。

    不成交的情形: 无次日交易数据、停牌、ST、开盘涨停、跳空超过 1 ATR、
    开盘已跌破结构止损位。跳空与止损用复权空间判定, 避免除权日的假跳空。
    '''
    entry_idx = signal_idx + 1
    if entry_idx >= len(bars):
        return {'status': 'NO_ENTRY_BAR'}
    row = bars.iloc[entry_idx]
    prev = bars.iloc[signal_idx]
    if int(row['tradestatus']) != 1 or float(row['volume']) <= 0:
        return {'status': 'ENTRY_SUSPENDED'}
    if _st_flag(row) != 0:
        return {'status': 'ENTRY_ST'}
    use_adj = params.use_adjusted and 'adj_open' in bars.columns
    atr0 = float(bars['atr20_adj' if use_adj else 'atr20'].iloc[signal_idx])
    if not atr0 or atr0 != atr0 or atr0 <= 0:
        return {'status': 'INVALID_ATR'}
    raw_open = float(row['open'])
    lim = _limit_pct_of(bars, entry_idx)
    pre = float(row['preclose']) if row['preclose'] == row['preclose'] else float(prev['close'])
    upper = math.floor(pre * (1 + lim) * 100 + .5) / 100
    if raw_open >= upper - 0.001:
        return {'status': 'ENTRY_LIMIT_UP'}
    price = float(row['adj_open']) if use_adj else raw_open
    signal_close = float(prev['adj_close']) if use_adj else float(prev['close'])
    if price - signal_close > atr0:
        return {'status': 'ENTRY_GAP'}
    initial_stop = price - params.stop_atr * atr0
    if params.stop_pct and params.stop_pct > 0:
        initial_stop = price * (1.0 - params.stop_pct)
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
    '''真限价挂单: 不高于信号日收盘才买, 最多等 wait_bars 根。

    这是"不追高"的可执行版本。信号日收盘价是决策时已知的价格, 挂一个
    不高于它的限价单在真实盘口里是能做到的。

    限价按原始价挂 (交易所口径), 因此:
    - 开盘价 <= 限价 -> 按开盘价成交 (比限价更好);
    - 盘中最低价触及限价 -> 按限价成交;
    - 一字涨停全天没有低于涨停价的成交 -> 不成交。
    max_chase=0 表示买价不得高于信号日收盘; wait_bars=0 表示次日不满足
    就直接放弃。
    '''
    n = len(bars)
    raw_signal_close = float(bars['close'].iloc[signal_idx])
    limit_raw = raw_signal_close * (1.0 + max_chase)
    last = min(signal_idx + 1 + wait_bars, n - 1)
    for j in range(signal_idx + 1, last + 1):
        row = bars.iloc[j]
        if int(row['tradestatus']) != 1 or float(row['volume']) <= 0:
            continue
        if _st_flag(row) != 0:
            continue
        raw_open = float(row['open'])
        raw_low = float(row['low'])
        lim = _limit_pct_of(bars, j)
        pre = float(row['preclose']) if row['preclose'] == row['preclose'] else float(bars['close'].iloc[j - 1])
        upper = math.floor(pre * (1 + lim) * 100 + .5) / 100
        if raw_low >= upper - 0.001:
            continue                      # 一字涨停, 全天没有低于涨停价的成交
        if raw_open <= limit_raw:
            if raw_open >= upper - 0.001:
                continue                  # 开盘涨停, 挂单排不上
            fill_raw = raw_open
        elif raw_low <= limit_raw:
            fill_raw = limit_raw
        else:
            continue
        use_adj = params.use_adjusted and 'adj_open' in bars.columns
        atr0 = float(bars['atr20_adj' if use_adj else 'atr20'].iloc[j - 1])
        if not atr0 or atr0 != atr0 or atr0 <= 0:
            continue
        price = fill_raw * float(bars['adj_scale'].iloc[j]) if use_adj else fill_raw
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
