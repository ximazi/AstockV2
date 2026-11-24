"""退出规则实验台: 在冻结的入场集合上比较不同退出规则。

用法:
    python run_exit_lab.py

第一件事是复现 v1 的基线退出 (structure36_20), 用账本里的离场日、离场原因
和盈亏逐笔对账。复现不过, 后面的变体比较就没有意义。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from astockv2.bars import bars as load_bars
from astockv2.bars import index_of
from astockv2.config import V1_RESULTS, V2_RESULTS
from astockv2.diagnose import load_ledger
from astockv2.execution import ExitParams, simulate_exit

DEFAULT_LEDGER_DIR = V1_RESULTS / '20261119_validated'


V0 = ExitParams(stop_atr=2.0, trail='low10', activate_atr=2.0,
                targets=((3.0, 0.5), (6.0, 0.25)), horizon=20,
                cost_bp=30.0, slippage_bp=0.0, use_adjusted=False)


def fidelity_params() -> ExitParams:
    '''与 v1 账本对账用的口径: 原始价、无滑点。

    v1 记账用的是未复权价且没有滑点, 所以复现基线必须锁在这套口径上;
    退出规则变体则跑在生产口径 (复权 + 滑点) 上, 与验收链一致。
    '''
    return ExitParams(stop_atr=2.0, trail='low10', activate_atr=2.0,
                      targets=((3.0, 0.5), (6.0, 0.25)), horizon=20,
                      cost_bp=30.0, slippage_bp=0.0, use_adjusted=False)


def load_trades(ledger_dir: Path) -> pd.DataFrame:
    files = sorted(ledger_dir.glob('ledger_*_30bp.csv'))
    rows = []
    for f in files:
        stem = f.stem.replace('ledger_', '')
        year, arm, *_ = stem.split('_')
        df = load_ledger(f)
        df['year'] = year
        df['arm'] = arm
        rows.append(df)
    all_df = pd.concat(rows, ignore_index=True)
    return all_df[all_df['status'] == 'CLOSED'].reset_index(drop=True)


def rerun(trades: pd.DataFrame, params: ExitParams) -> pd.DataFrame:
    '''用给定退出规则重跑每笔冻结入场。

    账本冻结的是原始价的入场价与止损位。复权口径下要把这两个价格按入场日
    的复权因子折算过去, 否则止损距离会错。
    '''
    recs = []
    for r in trades.itertuples(index=False):
        b = load_bars(r.symbol)
        ei = index_of(b, r.entry)
        if ei < 1:
            continue
        use_adj = params.use_adjusted and 'adj_open' in b.columns
        atr0 = float(b['atr20_adj' if use_adj else 'atr20'].iloc[ei - 1])
        if not atr0 or atr0 != atr0:
            continue
        entry_price = float(r.price)
        stop = float(r.price - (r.risk / r.qty))
        if use_adj:
            scale = float(b['adj_scale'].iloc[ei])
            entry_price = entry_price * scale
            stop = stop * scale
        if params.stop_source == 'atr':
            stop = entry_price - params.stop_atr * atr0
        res = simulate_exit(b, ei, entry_price, stop, params, atr0)
        res.update({'symbol': r.symbol, 'day': r.day, 'year': r.year,
                    'arm': r.arm, 'entry': r.entry, 'ledger_exit': r.exit,
                    'ledger_reason': r.reason, 'ledger_pnl': r.pnl,
                    'notional': r.notional})
        if res['status'] == 'CLOSED':
            res['pnl'] = res['net_ret'] * r.notional
            res['net_r'] = res['net_ret'] * entry_price / (entry_price - stop)
        recs.append(res)
    return pd.DataFrame(recs)


def fidelity(base: pd.DataFrame) -> None:
    '''与账本逐笔对账: 离场日、离场原因、盈亏。'''
    print('=== 基线复现对账 (v1 structure36_20, 无滑点) ===')
    n = len(base)
    same_date = (base['exit_date'].astype(str) == base['ledger_exit'].astype(str)).sum()
    same_reason = (base['reason'] == base['ledger_reason']).sum()
    diff = (base['pnl'] - base['ledger_pnl']).abs()
    print(f'成交 {n} 笔, 离场日一致 {same_date} ({same_date/n*100:.0f}%), '
          f'离场原因一致 {same_reason} ({same_reason/n*100:.0f}%)')
    print(f'盈亏绝对差: 中位 {diff.median():.2f} 元, 均值 {diff.mean():.2f} 元, '
          f'最大 {diff.max():.2f} 元')
    tol = base['notional'].abs() * 0.002
    within = int((diff < tol.clip(lower=1.0)).sum())
    print(f'盈亏差在名义金额 0.2% 以内: {within} 笔 ({within/n*100:.0f}%)')
    print()
    bad = base[base['reason'] != base['ledger_reason']]
    if len(bad):
        print('原因不一致的样本 (前 15 笔):')
        cols = ['symbol', 'entry', 'ledger_exit', 'ledger_reason', 'ledger_pnl',
                'exit_date', 'reason', 'pnl']
        print(bad[cols].head(15).to_string(index=False))
    print()


def variants() -> dict:
    '''退出规则变体集合。V0 是 v1 基线, 其余只改退出, 入场保持冻结。'''
    base = dict(stop_atr=2.0, trail='low10', activate_atr=2.0,
                targets=((3.0, 0.5), (6.0, 0.25)), horizon=20, cost_bp=30.0)
    v = {}
    v['V0_基线'] = ExitParams(**base, slippage_bp=0.0)
    v['V0s_基线加滑点'] = ExitParams(**base, slippage_bp=3.0)
    v['V1_止损放宽3ATR'] = ExitParams(**{**base, 'stop_atr': 3.0},
                                stop_source='atr', slippage_bp=3.0)
    v['V2_止损放宽4ATR'] = ExitParams(**{**base, 'stop_atr': 4.0},
                                stop_source='atr', slippage_bp=3.0)
    v['V3_不止盈'] = ExitParams(**{**base, 'targets': ()}, slippage_bp=3.0)
    v['V4_ATR移动止损'] = ExitParams(**{**base, 'trail': 'atr', 'trail_atr': 2.5},
                                slippage_bp=3.0)
    v['V5_只持有到期'] = ExitParams(**{**base, 'trail': 'none', 'targets': ()},
                               slippage_bp=3.0)
    v['V6_收盘止损'] = ExitParams(**base, stop_on_close=True, slippage_bp=3.0)
    v['V7_远目标'] = ExitParams(**{**base, 'targets': ((5.0, 0.5), (10.0, 0.25))},
                             slippage_bp=3.0)
    v['V8_宽止损且不止盈'] = ExitParams(**{**base, 'targets': (), 'stop_atr': 3.0},
                                  stop_source='atr', slippage_bp=3.0)
    return v


def agg(df: pd.DataFrame) -> dict:
    if not len(df):
        return {}
    pnl = df['pnl']
    top = pnl.max()
    return {
        '笔数': len(df),
        '胜率': round(float((pnl > 0).mean()) * 100, 1),
        '平均R': round(float(df['net_r'].mean()), 3),
        '中位R': round(float(df['net_r'].median()), 3),
        '合计盈亏': round(float(pnl.sum()), 0),
        '平均盈亏': round(float(pnl.mean()), 0),
        '去掉最大赢家': round(float(pnl.sum() - top), 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ledger-dir', default=str(DEFAULT_LEDGER_DIR))
    ap.add_argument('--out', default=str(V2_RESULTS / 'exit_lab'))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    trades = load_trades(Path(args.ledger_dir))
    print(f'冻结成交 {len(trades)} 笔')
    print()

    base = rerun(trades, fidelity_params())
    fidelity(base)
    base.to_csv(out / 'baseline_replay.csv', index=False, encoding='utf-8-sig')

    rows = []
    per_year = []
    for name, params in variants().items():
        df = rerun(trades, params)
        if not len(df):
            continue
        row = {'变体': name}
        row.update(agg(df))
        rows.append(row)
        for (y, a), g in df.groupby(['year', 'arm']):
            r = {'变体': name, '年': y, '臂': a}
            r.update(agg(g))
            per_year.append(r)
        df.to_csv(out / f'variant_{name}.csv', index=False, encoding='utf-8-sig')

    print('=== 退出规则变体对比 (入场冻结, 只改退出; 复权价 + 3bp 滑点) ===')
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print('=== 分年分臂 ===')
    print(pd.DataFrame(per_year).to_string(index=False))
    print()
    pd.DataFrame(rows).to_csv(out / 'variant_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(per_year).to_csv(out / 'variant_summary_by_year.csv', index=False, encoding='utf-8-sig')
    print('输出: ' + str(out))


if __name__ == '__main__':
    main()
