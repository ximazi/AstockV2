"""逐笔特征表: 只用信号日之前的信息, 看有没有任何可观测特征能区分好坏。

关键做法不是看特征在真实成交里有没有区分度, 而是看它在安慰剂群体里
是不是同样有效。某个特征如果在随机入场里也有效, 那它是市场因子, 不是
信号带来的优势; 只有在真实成交里明显更强, 才算系统自己的东西。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .bars import bars as load_bars
from .bars import index_of


def entry_features(symbol: str, signal_day: str, entry_date: str) -> dict:
    '''信号日收盘可知的全部特征, 加次日开盘才知道的跳空。'''
    b = load_bars(symbol)
    si = index_of(b, signal_day)
    if si < 21:
        return {}
    ei = index_of(b, entry_date)
    if ei <= 0:
        return {}
    close = b['close']
    atr = float(b['atr20'].iloc[si])
    if not atr or atr != atr:
        return {}
    ma20 = float(close.iloc[si - 19:si + 1].mean())
    px = float(close.iloc[si])
    up = b['close'].iloc[max(0, si - 9):si + 1] / b['preclose'].iloc[max(0, si - 9):si + 1] - 1
    limit_up = (up > 0.095).sum()
    f = {
        'bias_atr': (px - ma20) / atr,
        'atr_pct': atr / px * 100,
        'ret5': px / float(close.iloc[si - 5]) - 1,
        'ret20': px / float(close.iloc[si - 20]) - 1,
        'dist_high20': px / float(b['high20_prev'].iloc[si]) - 1,
        'turnover_rel': float(b['turnover'].iloc[si]) / float(b['turnover'].iloc[max(0, si - 20):si + 1].median()),
        'amount_log': float(np.log(max(1.0, b['amount'].iloc[max(0, si - 20):si + 1].median()))),
        'signal_pct': float(b['pct_chg'].iloc[si]) / 100.0,
        'limit_up_10d': int(limit_up),
        'gap': float(b['open'].iloc[ei]) / px - 1,
        'weekday': int(b['date'].iloc[si].weekday()),
        'month': int(b['date'].iloc[si].month),
        'sig_date': str(b['date'].iloc[si].date()),
    }
    return f


QUANTS = 5


def add_features(df: pd.DataFrame, day_col: str = 'anchor_day') -> pd.DataFrame:
    '''给对照表补上特征。'''
    rows = []
    for r in df.itertuples(index=False):
        day = getattr(r, day_col)
        entry = getattr(r, 'entry_date', None)
        if not entry:
            continue
        f = entry_features(r.symbol, day, entry)
        if f:
            f['net_r'] = r.net_r
            f['net_ret'] = r.net_ret
            f['arm'] = r.arm
            rows.append(f)
    return pd.DataFrame(rows)


def quintile_table(traded: pd.DataFrame, control: pd.DataFrame,
                   feature: str) -> pd.DataFrame:
    '''按特征分位, 比较真实成交与安慰剂群体。

    只有当真实成交明显好于同分位的安慰剂时, 这个特征才算系统自己的优势。
    '''
    t = traded.dropna(subset=[feature]).copy()
    c = control.dropna(subset=[feature]).copy()
    if not len(t) or not len(c):
        return pd.DataFrame()
    edges = sorted({float(c[feature].quantile(q / QUANTS)) for q in range(1, QUANTS)})
    t['bin'] = pd.cut(t[feature], [-np.inf] + edges + [np.inf], labels=False)
    c['bin'] = pd.cut(c[feature], [-np.inf] + edges + [np.inf], labels=False)
    rows = []
    for b in range(QUANTS):
        tb = t[t['bin'] == b]
        cb = c[c['bin'] == b]
        if not len(tb):
            continue
        rows.append({
            '分位': b + 1,
            '特征区间': f'{(tb[feature].min()):.3f} ~ {(tb[feature].max()):.3f}',
            '真实笔数': len(tb),
            '真实平均R': round(float(tb['net_r'].mean()), 3),
            '真实胜率%': round(float((tb['net_ret'] > 0).mean()) * 100, 1),
            '对照笔数': len(cb),
            '对照平均R': round(float(cb['net_r'].mean()), 3) if len(cb) else float('nan'),
            '差值R': round(float(tb['net_r'].mean()) - (float(cb['net_r'].mean()) if len(cb) else 0), 3),
        })
    return pd.DataFrame(rows)


FEATURES = ['bias_atr', 'atr_pct', 'ret5', 'ret20', 'dist_high20',
            'turnover_rel', 'amount_log', 'signal_pct', 'limit_up_10d', 'gap']


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--control-dir', default=str('D:/AICode/AStockV2/results/control'))
    ap.add_argument('--out', default=str('D:/AICode/AStockV2/results/features'))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cdir = Path(args.control_dir)

    traded = pd.read_csv(cdir / 'T0_真实信号.csv')
    ctrl = pd.read_csv(cdir / 'C2_同股异日.csv')

    t = add_features(traded)
    c = add_features(ctrl)
    print(f'真实成交带特征 {len(t)} 笔, 安慰剂带特征 {len(c)} 笔')
    print()
    t.to_csv(out / 'traded_features.csv', index=False, encoding='utf-8-sig')
    c.to_csv(out / 'control_features.csv', index=False, encoding='utf-8-sig')

    rows = []
    for f in FEATURES:
        tab = quintile_table(t, c, f)
        if not len(tab):
            continue
        tab.insert(0, '特征', f)
        rows.append(tab)
    all_tab = pd.concat(rows, ignore_index=True)
    all_tab.to_csv(out / 'feature_quintiles.csv', index=False, encoding='utf-8-sig')
    for f in FEATURES:
        tab = quintile_table(t, c, f)
        if not len(tab):
            continue
        print(f'=== {f} ===')
        print(tab.to_string(index=False))
        print()
    print('输出: ' + str(out))


if __name__ == '__main__':
    main()
