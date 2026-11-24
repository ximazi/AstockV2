"""市场环境分层: 核心指数 EMA100 斜率。

A股只能做多, 指数中期趋势向下时, 个股信号再准也难赚钱。这个模块把
"信号日处在什么市场环境"量化出来, 并且用来回答两个问题:

  1. 过去的信号日是不是大量落在差环境里;
  2. 把市场环境当成混杂因素剔除之后, 信号的优势还剩多少。

判据只用当日收盘已知的信息: EMA100 与它自己在 lookback 根之前的比值。
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .config import V1_DATA

INDEX_FILES = {
    'hs300': '000300_SH_index.csv',
    'csi1000': '000852_SH_index.csv',
    'sse': '000001_SH_index.csv',
    'szcomp': '399001_SZ_index.csv',
    'csi500': '000905_SH_index.csv',
}


def load_index(name: str) -> pd.DataFrame:
    '''指数日线, 只取日期与收盘。'''
    path = V1_DATA / INDEX_FILES[name]
    df = pd.read_csv(path, dtype={'trade_date': str})
    df['date'] = pd.to_datetime(df['trade_date'].astype(str))
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    return df[['date', 'close']].dropna().sort_values('date').reset_index(drop=True)


def regime_frame(name: str = 'hs300', span: int = 100, lookback: int = 10,
                 flat_band: float = 0.0) -> pd.DataFrame:
    '''核心指数的环境标签。

    slope = EMA100[t] / EMA100[t-lookback] - 1
    DOWN: slope < -flat_band, UP: 其余。
    '''
    df = load_index(name)
    ema = df['close'].ewm(span=span, adjust=False).mean()
    slope = ema / ema.shift(lookback) - 1
    out = df.copy()
    out['ema'] = ema
    out['slope'] = slope
    out['regime'] = np.where(slope < -flat_band, 'DOWN', 'UP')
    out.loc[slope.isna(), 'regime'] = 'UNKNOWN'
    return out


def regime_map(name: str = 'hs300', span: int = 100, lookback: int = 10,
               flat_band: float = 0.0) -> dict:
    r = regime_frame(name, span, lookback, flat_band)
    return dict(zip(r['date'].dt.strftime('%Y-%m-%d'), r['regime']))


def attach_regime(df: pd.DataFrame, day_col: str, rmap: dict) -> pd.DataFrame:
    '''给一臂的交易打上市场环境标签。'''
    out = df.copy()
    out['regime'] = out[day_col].astype(str).map(rmap)
    out['regime'] = out['regime'].fillna('UNKNOWN')
    return out


def mix(df: pd.DataFrame) -> pd.Series:
    '''各环境的交易占比。'''
    if not len(df):
        return pd.Series(dtype=float)
    return df['regime'].value_counts(normalize=True).sort_index()


def by_regime(df: pd.DataFrame, col: str = 'ret_pct') -> pd.DataFrame:
    '''按环境分层的笔数、胜率、平均收益。'''
    if not len(df):
        return pd.DataFrame()
    g = df.groupby('regime')
    out = g.agg(笔数=(col, 'size'),
                平均收益=('ret_pct', 'mean'),
                中位收益=('ret_pct', 'median')).reset_index()
    out['胜率%'] = g['ret_pct'].apply(lambda s: (s > 0).mean() * 100).values
    out['平均收益'] = out['平均收益'] * 100
    out['中位收益'] = out['中位收益'] * 100
    return out


def standardize(traded: pd.DataFrame, control: pd.DataFrame) -> tuple:
    '''把对照组按真实成交的环境构成重新加权, 再比。

    返回 (点估计差值, 各环境明细)。这是把市场环境当混杂因素剔除:
    如果差值仍然为负, 那么环境不是原因。
    '''
    rows = []
    diff = 0.0
    weight = 0.0
    for reg, g in traded.groupby('regime'):
        t = float(g['ret_pct'].mean())
        c = control[control['regime'] == reg]
        cm = float(c['ret_pct'].mean()) if len(c) else float('nan')
        w = len(g) / len(traded)
        rows.append({
            '环境': reg,
            '真实笔数': len(g),
            '真实平均%': round(t * 100, 3),
            '对照笔数': len(c),
            '对照平均%': round(cm * 100, 3) if len(c) else float('nan'),
            '差值%': round((t - cm) * 100, 3) if len(c) else float('nan'),
            '权重': round(w, 3),
        })
        if len(c) and cm == cm:
            diff += w * (t - cm)
            weight += w
    if weight > 0:
        diff = diff / weight
    return diff, pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--control-dir', default=str('D:/AICode/AStockV2/results/control'))
    ap.add_argument('--s4-dir', default=str('D:/AICode/AStockV2/results/s4_arm'))
    ap.add_argument('--ledger-dir', default=str('D:/AICode/Astock/results/research/mainwave_path/20261119_validated'))
    ap.add_argument('--out', default=str('D:/AICode/AStockV2/results/regime'))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cdir = Path(args.control_dir)
    sdir = Path(args.s4_dir)

    from .exit_lab import load_trades
    from .verdict import block_boot

    rf = regime_frame('hs300')
    rf = rf[rf['date'] >= '2024-01-01']
    rmap = dict(zip(rf['date'].dt.strftime('%Y-%m-%d'), rf['regime']))
    slopemap = dict(zip(rf['date'].dt.strftime('%Y-%m-%d'), rf['slope']))
    print('=== 核心指数环境 (沪深300, EMA100, 10 日斜率) ===')
    print(f"2024-01-01 起共 {len(rf)} 个交易日: DOWN {int((rf['regime']=='DOWN').sum())} "
          f"({(rf['regime']=='DOWN').mean()*100:.0f}%), "
          f"UP {int((rf['regime']=='UP').sum())} ({(rf['regime']=='UP').mean()*100:.0f}%)")
    print()

    arms = {}
    from .placebo_audit import mark_overlap
    t0 = pd.read_csv(cdir / 'T0_真实信号.csv')
    t0['ret_pct'] = t0['net_ret']
    arms['R1 真实信号 (重建)'] = attach_regime(t0, 'anchor_day', rmap)
    c2 = pd.read_csv(cdir / 'C2_同股异日.csv')
    c2['ret_pct'] = c2['net_ret']
    c2 = mark_overlap(t0, c2)
    c2_clean = c2[~c2['overlaps_any']]
    arms['R1 同股异日'] = attach_regime(c2_clean, 'anchor_day', rmap)
    arms['R1 同股异日·全部(含窗口重叠)'] = attach_regime(c2, 'anchor_day', rmap)
    for nm, f in (('R1 同日随机标的', 'C1_同日随机.csv'),
                  ('R1 同池未选中', 'C3_同池未选中.csv')):
        d = pd.read_csv(cdir / f)
        d['ret_pct'] = d['net_ret']
        arms[nm] = attach_regime(d, 'anchor_day', rmap)
    sa = pd.read_csv(sdir / 's4_arm.csv')
    sa['ret_pct'] = sa['net_ret']
    arms['S4 信号'] = attach_regime(sa, 'anchor_day', rmap)
    sp = pd.read_csv(sdir / 's4_placebo.csv')
    sp['ret_pct'] = sp['net_ret']
    sp = mark_overlap(sa, sp)
    sp_clean = sp[~sp['overlaps_any']]
    arms['S4 同股异日'] = attach_regime(sp_clean, 'anchor_day', rmap)
    arms['S4 同股异日·全部(含窗口重叠)'] = attach_regime(sp, 'anchor_day', rmap)

    print('=== 各臂的市场环境构成 ===')
    rows = []
    for nm, d in arms.items():
        m = mix(d)
        rows.append({'臂': nm, '笔数': len(d),
                     'DOWN%': round(m.get('DOWN', 0) * 100, 1),
                     'UP%': round(m.get('UP', 0) * 100, 1),
                     'DOWN笔数': int((d['regime'] == 'DOWN').sum())})
    mixdf = pd.DataFrame(rows)
    print(mixdf.to_string(index=False))
    print()

    print('=== 真实成交按环境分层 (R1) ===')
    print(by_regime(arms['R1 真实信号 (重建)']).to_string(index=False))
    print()
    print('=== 同股异日安慰剂按环境分层 ===')
    print(by_regime(arms['R1 同股异日']).to_string(index=False))
    print()
    print('=== S4 信号按环境分层 ===')
    print(by_regime(arms['S4 信号']).to_string(index=False))
    print()
    print('=== S4 同股异日安慰剂按环境分层 ===')
    print(by_regime(arms['S4 同股异日']).to_string(index=False))
    print()
    print('=== 剔除环境混杂后的差值 ===')
    print('主判据用干净桶 (安慰剂持仓窗口不含任何信号行情); 旧口径一并列出')
    for tag, a, b, b_all in (('R1', arms['R1 真实信号 (重建)'], arms['R1 同股异日'],
                              arms['R1 同股异日·全部(含窗口重叠)']),
                             ('S4', arms['S4 信号'], arms['S4 同股异日'],
                              arms['S4 同股异日·全部(含窗口重叠)'])):
        d, detail = standardize(a, b)
        print(f'--- {tag}: 按真实成交的环境构成加权后的差值 (干净桶) {d*100:+.3f}% ---')
        print(detail.to_string(index=False))
        print()
        for reg in ('DOWN', 'UP'):
            aa = a[a['regime'] == reg]
            bb = b[b['regime'] == reg]
            if len(aa) < 10 or len(bb) < 10:
                print(f'  {reg}: 样本不足 (真实 {len(aa)}, 对照 {len(bb)})')
                continue
            obs, lo, hi, days = block_boot(aa, bb)
            print(f'  {reg}: 差值 {obs*100:+.3f}%  95%区间 [{lo*100:+.3f}%, {hi*100:+.3f}%]  '
                  f'可比交易日 {days}')
        print()
        print(f'  [旧口径参考] {tag} 信号 vs 全部安慰剂 (含窗口重叠):')
        for reg in ('DOWN', 'UP'):
            aa = a[a['regime'] == reg]
            bb = b_all[b_all['regime'] == reg]
            if len(aa) < 10 or len(bb) < 10:
                continue
            obs, lo, hi, days = block_boot(aa, bb)
            print(f'    {reg}: 差值 {obs*100:+.3f}%  95%区间 [{lo*100:+.3f}%, {hi*100:+.3f}%]  '
                  f'可比交易日 {days}')
        print()

    print('=== 假如把 DOWN 环境的日子全部不做 (R1, 用冻结账本的原始金额) ===')
    trades = load_trades(Path(args.ledger_dir))
    trades['ret_pct'] = trades['pnl'] / trades['notional']
    trades['regime'] = trades['day'].map(rmap).fillna('UNKNOWN')
    for reg in ('DOWN', 'UP'):
        g = trades[trades['regime'] == reg]
        if not len(g):
            continue
        print(f'  {reg}: {len(g)} 笔, 合计盈亏 {g["pnl"].sum():.0f} 元, '
              f'平均每笔 {g["ret_pct"].mean()*100:.3f}%, '
              f'胜率 {(g["ret_pct"]>0).mean()*100:.1f}%')
    up = trades[trades['regime'] == 'UP']
    print(f'  只做 UP: {len(up)} 笔, 合计盈亏 {up["pnl"].sum():.0f} 元, '
          f'平均每笔 {up["ret_pct"].mean()*100:.3f}%')
    print()

    print('=== 假如把 DOWN 环境的日子全部不做 (S4) ===')
    for nm in ('S4 信号', 'S4 同股异日'):
        d = arms[nm]
        for reg in ('DOWN', 'UP'):
            g = d[d['regime'] == reg]
            if not len(g):
                continue
            print(f'  {nm} {reg}: {len(g)} 笔, 平均每笔 {g["ret_pct"].mean()*100:.3f}%, '
                  f'胜率 {(g["ret_pct"]>0).mean()*100:.1f}%')
    print()

    mixdf.to_csv(out / 'arm_regime_mix.csv', index=False, encoding='utf-8-sig')
    rf.to_csv(out / 'regime_series_hs300.csv', index=False, encoding='utf-8-sig')
    print('输出: ' + str(out))


if __name__ == '__main__':
    main()


def prev_trading_day_map(name: str = 'hs300') -> dict:
    '''交易日的前一个交易日, 用来把"买入日"还原成"决策日"。'''
    df = load_index(name)
    days = df['date'].dt.strftime('%Y-%m-%d').tolist()
    return {d: days[i - 1] for i, d in enumerate(days) if i > 0}


def decision_regime(df: pd.DataFrame, entry_col: str, prevmap: dict,
                    rmap: dict) -> pd.Series:
    '''决策日 = 买入日的前一个交易日; 环境按决策日判定。'''
    dec = df[entry_col].astype(str).map(prevmap)
    return dec.map(rmap).fillna('UNKNOWN')


def within_signal_diff(a: pd.DataFrame, tag: str, n: int = 3000,
                        seed: int = 20261120) -> dict:
    '''同一臂内部: UP 环境 vs DOWN 环境的平均收益差, 按交易日分块 bootstrap。'''
    up = a[a['regime'] == 'UP']
    dn = a[a['regime'] == 'DOWN']
    if len(up) < 10 or len(dn) < 10:
        return {}
    obs = float(up['ret_pct'].mean()) - float(dn['ret_pct'].mean())
    rng = random.Random(seed)
    by_day = {}
    for d, g in a.groupby('anchor_day'):
        by_day[d] = (g['regime'].iloc[0], g['ret_pct'].tolist())
    days = sorted(by_day)
    diffs = []
    for _ in range(n):
        pick = [days[rng.randrange(len(days))] for _ in range(len(days))]
        us, ds = [], []
        for d in pick:
            reg, vals = by_day[d]
            (us if reg == 'UP' else ds).extend(vals)
        if us and ds:
            diffs.append(sum(us) / len(us) - sum(ds) / len(ds))
    diffs.sort()
    return {
        '臂': tag,
        'UP笔数': len(up),
        'DOWN笔数': len(dn),
        'UP平均%': round(float(up['ret_pct'].mean()) * 100, 3),
        'DOWN平均%': round(float(dn['ret_pct'].mean()) * 100, 3),
        '差值%': round(obs * 100, 3),
        '区间下界%': round(diffs[int(len(diffs) * 0.025)] * 100, 3),
        '区间上界%': round(diffs[int(len(diffs) * 0.975)] * 100, 3),
    }


def robustness(arms_spec: list, indices: tuple = ('hs300', 'csi1000', 'sse'),
               spans: tuple = (50, 100, 200), lookbacks: tuple = (5, 10, 20)) -> pd.DataFrame:
    '''不同指数、不同 EMA 长度、不同回看长度下, UP/DOWN 差值是否稳定。'''
    rows = []
    for idx in indices:
        for span in spans:
            for lb in lookbacks:
                rf = regime_frame(idx, span, lb)
                rmap = dict(zip(rf['date'].dt.strftime('%Y-%m-%d'), rf['regime']))
                for tag, path, daycol in arms_spec:
                    df = pd.read_csv(path)
                    if 'ret_pct' not in df.columns:
                        df['ret_pct'] = df['net_ret']
                    df = attach_regime(df, daycol, rmap)
                    r = within_signal_diff(df, tag, n=800)
                    if r:
                        r['指数'] = idx
                        r['EMA'] = span
                        r['回看'] = lb
                        rows.append(r)
    return pd.DataFrame(rows)
