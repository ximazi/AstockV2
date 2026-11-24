"""S4 模式用同一套仪表盘验收。

v1 给 S4 的成绩是"20 日内先触及 +10% 而非 -10% 的比例 45.8% / 48.2%"。
那是价格触达率, 不是扣费后的盈亏, 而且不区分同一天买入的股票里
哪些其实更好。这里把 S4 信号用与 R1 完全相同的入场规则、退出模型、
成本和安慰剂对照重跑一遍, 直接回答"这套形态有没有可重复的优势"。
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from .bars import bars as load_bars
from .bars import index_of
from .config import V1_ROOT, V2_RESULTS
from .execution import ExitParams

S4_LABELLED = V1_ROOT / 'results' / 'research' / 's4_features' / 's4_labelled.csv'


def load_signals(path: Path = S4_LABELLED) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={'symbol': str, 'date': str})
    df = df.rename(columns={'date': 'day'})
    df = df[['symbol', 'day', 'label', 'year', 'win']]
    df = df.drop_duplicates(subset=['symbol', 'day'])
    return df.reset_index(drop=True)


def arm_s4(signals: pd.DataFrame, params: ExitParams, sample: int | None = None,
           seed: int = 20261120) -> pd.DataFrame:
    '''S4 信号用统一入场与退出规则重跑。'''
    from .control import run_entry
    rng = random.Random(seed)
    rows = []
    sig = signals
    if sample and len(sig) > sample:
        sig = sig.sample(sample, random_state=seed).reset_index(drop=True)
    for r in sig.itertuples(index=False):
        b = load_bars(r.symbol)
        si = index_of(b, r.day)
        if si < 0:
            continue
        res = run_entry(b, si, params)
        if not res:
            continue
        res.update({'arm': 'S4_信号', 'symbol': r.symbol, 'anchor_day': r.day,
                    'label': r.label, 'year': r.year})
        rows.append(res)
    return pd.DataFrame(rows)


def arm_s4_placebo(signals: pd.DataFrame, params: ExitParams, k: int = 3,
                   seed: int = 20261120) -> pd.DataFrame:
    '''同一只股票、同一年、换一个随机交易日入场。'''
    from .control import run_entry
    rng = random.Random(seed + 1)
    rows = []
    for r in signals.itertuples(index=False):
        b = load_bars(r.symbol)
        si = index_of(b, r.day)
        if si < 0:
            continue
        year = str(r.day)[:4]
        days = [i for i in range(21, len(b) - 1) if str(b['date'].iloc[i].date())[:4] == year]
        if len(days) < 5:
            continue
        for ci in rng.sample(days, min(k, len(days))):
            if abs(ci - si) < 3:
                continue
            res = run_entry(b, ci, params, source_idx=si)
            if res:
                res.update({'arm': 'S4_同股异日', 'symbol': r.symbol,
                            'anchor_day': str(b['date'].iloc[ci].date()),
                            'label': '', 'year': r.year})
                rows.append(res)
    return pd.DataFrame(rows)


def report_arms(tag: str, a: pd.DataFrame, c: pd.DataFrame) -> None:
    '''信号臂 vs 安慰剂, 安慰剂按"窗口含不含信号行情"拆开。'''
    from .verdict import block_boot, split_overlap
    clean, overlap = split_overlap(c)
    print(f'--- {tag} ---')
    if len(a):
        print(f'  信号 {len(a)} 笔, 平均每笔 {a["net_ret"].mean()*100:+.3f}%, '
              f'胜率 {(a["net_ret"]>0).mean()*100:.1f}%, 平均 {a["net_r"].mean():.3f} R')
    for name, df in (('干净桶 CLEAN', clean),
                     ('窗口含信号 OVERLAP', overlap),
                     ('全部安慰剂', c)):
        if not len(df):
            continue
        print(f'  安慰剂[{name}] {len(df)} 笔, 平均每笔 {df["net_ret"].mean()*100:+.3f}%, '
              f'胜率 {(df["net_ret"]>0).mean()*100:.1f}%')
        if len(a) and len(df) >= 20:
            a2 = a.copy()
            a2['ret_pct'] = a2['net_ret']
            b2 = df.copy()
            b2['ret_pct'] = b2['net_ret']
            obs, lo, hi, days = block_boot(a2, b2)
            print(f'    信号 - {name}: {obs*100:+.3f}%  '
                  f'95%[{lo*100:+.3f}%, {hi*100:+.3f}%]  可比交易日 {days}')
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=0, help='只取前 N 个信号做冒烟; 0 = 全量')
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--out', default=str(V2_RESULTS / 's4_arm'))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    params = ExitParams(stop_atr=2.0, trail='low10', activate_atr=2.0,
                         targets=((3.0, 0.5), (6.0, 0.25)), horizon=20,
                         cost_bp=30.0, slippage_bp=3.0)

    # S4 原装退出: -10% 止损, +10%/+20% 各卖 1/3, 剩余 1/3 持有到 60 日上限
    lad = ExitParams(stop_pct=0.10, trail='none', activate_atr=2.0,
                     targets=(), targets_pct=((0.10, 1.0/3), (0.20, 1.0/3)),
                     horizon=60, cost_bp=30.0, slippage_bp=3.0)

    sig = load_signals()
    if args.sample:
        sig = sig.head(args.sample)
    print(f'S4 信号 {len(sig)} 条')

    a = arm_s4(sig, params)
    print(f'S4 信号重跑成交 {len(a)} 笔')
    c = arm_s4_placebo(sig, params, args.k)
    print(f'S4 同股异日对照 {len(c)} 笔')
    from .control import universe as uni
    d = arm_s4_same_day(sig, params, uni(), args.k)
    print(f'S4 同日随机标的对照 {len(d)} 笔')
    print()

    if len(a):
        print('=== S4 信号 ===')
        print(f"平均每笔 {a['net_ret'].mean()*100:.3f}%  胜率 {(a['net_ret']>0).mean()*100:.1f}%  "
              f"平均 {a['net_r'].mean():.3f} R  笔数 {len(a)}")
        if 'label' in a.columns:
            g = a.groupby('label')['net_ret'].agg(['size', 'mean'])
            g['胜率%'] = a.groupby('label')['net_ret'].apply(lambda s: (s > 0).mean() * 100)
            g['平均%'] = g['mean'] * 100
            print()
            print('=== 触达标签 vs 实际盈亏 ===')
            print(g[['size', '胜率%', '平均%']].to_string())
    report_arms('S4 走 R1 的 structure36_20 退出', a, c)

    if len(a) and len(d):
        from .verdict import block_boot
        a2 = a.copy()
        a2['ret_pct'] = a2['net_ret']
        d2 = d.copy()
        d2['ret_pct'] = d2['net_ret']
        obs, lo, hi, days = block_boot(a2, d2)
        print(f'S4 信号 - 同日随机标的: 差值 {obs*100:+.3f}%  '
              f'95%区间 [{lo*100:+.3f}%, {hi*100:+.3f}%]  可比交易日 {days}')
        print()

    print('=== S4 原装 S4_LAD 退出 (-10% 止损, +10%/+20% 各卖 1/3, 60 日上限) ===')
    la = arm_s4(sig, lad)
    lc = arm_s4_placebo(sig, lad, args.k)
    print(f'S4 信号 {len(la)} 笔, 同股异日对照 {len(lc)} 笔')
    report_arms('S4 原装 S4_LAD', la, lc)

    a.to_csv(out / 's4_arm.csv', index=False, encoding='utf-8-sig')
    c.to_csv(out / 's4_placebo.csv', index=False, encoding='utf-8-sig')
    d.to_csv(out / 's4_same_day.csv', index=False, encoding='utf-8-sig')
    la.to_csv(out / 's4_lad_arm.csv', index=False, encoding='utf-8-sig')
    lc.to_csv(out / 's4_lad_placebo.csv', index=False, encoding='utf-8-sig')
    print()
    print('输出: ' + str(out))


if __name__ == '__main__':
    main()


def arm_s4_same_day(signals: pd.DataFrame, params: ExitParams, universe: list,
                    k: int = 3, seed: int = 20261120) -> pd.DataFrame:
    '''同一天换一只随机股票: 把"选股"和"择时"分开看。'''
    from .control import run_entry
    rng = random.Random(seed + 2)
    rows = []
    for r in signals.itertuples(index=False):
        b = load_bars(r.symbol)
        si = index_of(b, r.day)
        if si < 0:
            continue
        for sym in rng.sample(universe, k):
            if sym == r.symbol:
                continue
            cb = load_bars(sym)
            ci = index_of(cb, r.day)
            if ci < 0:
                continue
            res = run_entry(cb, ci, params)
            if res:
                res.update({'arm': 'S4_同日随机', 'symbol': sym,
                            'anchor_day': r.day, 'label': '', 'year': r.year})
                rows.append(res)
    return pd.DataFrame(rows)
