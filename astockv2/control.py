"""安慰剂对照实验: 信号到底有没有价值。

固定 159 笔真实成交不动, 只把"为什么在这一天买这一只"拿掉, 看结果差多少。

  C1 同日随机标的: 同一天, 从全市场随机抽一只同样能成交的股票。
     回答: 这些日子本身是不是普遍赚钱 (市场漂移)。
  C2 同股异日: 同一只股票, 换成同年的另一个随机交易日入场。
     回答: 是"这一天这只股票"带来了结果, 还是"这只股票"本身。

两组对照用完全相同的入场规则和退出模型, 因此差异只能来自信号本身。
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from .bars import bars as load_bars
from .bars import index_of
from .config import V1_DATA
from .execution import ExitParams, simulate_entry, simulate_exit

UNIVERSE_RE = r'^(sh_(?:60|68)|sz_(?:00|30))\d{4}_daily\.csv$'


def universe() -> list:
    '''与 v1 route_scan 相同的股票域: 沪深主板/创业板/科创板。'''
    out = []
    for p in sorted(V1_DATA.glob('*_daily.csv')):
        if p.stem.endswith('_daily'):
            sym = p.stem[:-6]
            if sym[:2] == 'sh' and sym[3:5] in ('60', '68'):
                out.append(sym)
            elif sym[:2] == 'sz' and sym[3:5] in ('00', '30'):
                out.append(sym)
    return out


def liquid_enough(bars: pd.DataFrame, idx: int) -> bool:
    '''v1 的流动性口径: 前 20 日成交额中位数 >= 3e7。'''
    if idx < 21:
        return False
    amt = bars['amount'].iloc[idx - 20:idx]
    return float(amt.median()) >= 3e7


def run_entry(bars: pd.DataFrame, signal_idx: int, params: ExitParams):
    '''按统一入场规则尝试一笔, 成交则返回退出模拟结果。'''
    if signal_idx < 21 or signal_idx >= len(bars) - 1:
        return None
    if not liquid_enough(bars, signal_idx):
        return None
    e = simulate_entry(bars, signal_idx, params)
    if e['status'] != 'OK':
        return None
    res = simulate_exit(bars, e['entry_idx'], e['entry_price'],
                        e['initial_stop'], params, e['atr0'])
    if res['status'] != 'CLOSED':
        return None
    res['entry_date'] = e['entry_date']
    res['entry_price'] = e['entry_price']
    return res


def control_same_day(trades: pd.DataFrame, params: ExitParams, universe: list,
                     k: int, seed: int = 20261120) -> pd.DataFrame:
    '''C1: 同一天, 随机抽别的股票。'''
    rng = random.Random(seed)
    rows = []
    for r in trades.itertuples(index=False):
        b = load_bars(r.symbol)
        si = index_of(b, r.day)
        if si < 0:
            continue
        picks = rng.sample(universe, k)
        for sym in picks:
            if sym == r.symbol:
                continue
            cb = load_bars(sym)
            ci = index_of(cb, r.day)
            if ci < 0:
                continue
            res = run_entry(cb, ci, params)
            if res:
                res.update({'arm': 'C1_同日随机', 'anchor': r.symbol,
                            'anchor_day': r.day, 'symbol': sym})
                rows.append(res)
    return pd.DataFrame(rows)


def control_same_stock(trades: pd.DataFrame, params: ExitParams,
                        k: int, seed: int = 20261120) -> pd.DataFrame:
    '''C2: 同一只股票, 换成同年的另一个随机交易日。'''
    rng = random.Random(seed)
    rows = []
    for r in trades.itertuples(index=False):
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
            res = run_entry(b, ci, params)
            if res:
                res.update({'arm': 'C2_同股异日', 'anchor': r.symbol,
                            'anchor_day': str(b['date'].iloc[ci].date()),
                            'symbol': r.symbol})
                rows.append(res)
    return pd.DataFrame(rows)


def arm_signal(trades: pd.DataFrame, params: ExitParams) -> pd.DataFrame:
    '''T0: 真实信号, 但用与对照组完全相同的构造方式重跑一遍。'''
    rows = []
    for r in trades.itertuples(index=False):
        b = load_bars(r.symbol)
        si = index_of(b, r.day)
        if si < 0:
            continue
        res = run_entry(b, si, params)
        if res:
            res.update({'arm': 'T0_真实信号', 'anchor': r.symbol,
                        'anchor_day': r.day, 'symbol': r.symbol})
            rows.append(res)
    return pd.DataFrame(rows)


def boot_diff(a: pd.DataFrame, b: pd.DataFrame, key: str = 'anchor_day',
              n: int = 3000, seed: int = 20261120) -> tuple:
    '''按交易日分块 bootstrap, 给出均值差的置信区间。

    同一天的多笔交易高度相关, 所以按日期重采样, 不能把每笔当独立样本。
    '''
    rng = random.Random(seed)
    ga = a.groupby(key)['net_ret'].mean()
    gb = b.groupby(key)['net_ret'].mean()
    common = sorted(set(ga.index) & set(gb.index))
    if len(common) < 3:
        return (float('nan'), float('nan'), float('nan'))
    da = ga.loc[common].to_numpy()
    db = gb.loc[common].to_numpy()
    obs = da.mean() - db.mean()
    diffs = []
    m = len(common)
    for _ in range(n):
        idx = [rng.randrange(m) for _ in range(m)]
        diffs.append(da[idx].mean() - db[idx].mean())
    diffs.sort()
    return (obs, diffs[int(n * 0.025)], diffs[int(n * 0.975)])


def summarize_arms(arms: dict) -> pd.DataFrame:
    rows = []
    for name, df in arms.items():
        if not len(df):
            continue
        ret = df['net_ret']
        rows.append({
            '臂': name,
            '笔数': len(df),
            '胜率%': round(float((ret > 0).mean()) * 100, 1),
            '平均收益%': round(float(ret.mean()) * 100, 3),
            '中位收益%': round(float(ret.median()) * 100, 3),
            '标准差%': round(float(ret.std()) * 100, 3),
            '最好%': round(float(ret.max()) * 100, 2),
            '最差%': round(float(ret.min()) * 100, 2),
            '平均R': round(float(df['net_r'].mean()), 3),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ledger-dir', default=str('D:/AICode/Astock/results/research/mainwave_path/20261119_validated'))
    ap.add_argument('--out', default=str('D:/AICode/AStockV2/results/control'))
    ap.add_argument('--k', type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    params = ExitParams(stop_atr=2.0, trail='low10', activate_atr=2.0,
                         targets=((3.0, 0.5), (6.0, 0.25)), horizon=20,
                         cost_bp=30.0, slippage_bp=3.0)

    from .exit_lab import load_trades
    trades = load_trades(Path(args.ledger_dir))
    print(f'真实成交 {len(trades)} 笔, 每笔抽 {args.k} 个对照')
    print()

    t0 = arm_signal(trades, params)
    print(f'T0 真实信号重跑: {len(t0)} 笔')
    c1 = control_same_day(trades, params, universe(), args.k)
    print(f'C1 同日随机: {len(c1)} 笔')
    c2 = control_same_stock(trades, params, args.k)
    print(f'C2 同股异日: {len(c2)} 笔')
    c3 = arm_not_selected(Path(args.ledger_dir), params)
    print(f'C3 同池未选中: {len(c3)} 笔')
    print()

    arms = {'T0_真实信号': t0, 'C1_同日随机': c1, 'C2_同股异日': c2,
            'C3_同池未选中': c3}
    print('=== 安慰剂对照 ===')
    print(summarize_arms(arms).to_string(index=False))
    print()
    for other in ('C1_同日随机', 'C2_同股异日', 'C3_同池未选中'):
        obs, lo, hi = boot_diff(t0, arms[other])
        print(f'T0 - {other}: 均值差 {obs*100:+.3f}%  95%区间 [{lo*100:+.3f}%, {hi*100:+.3f}%]')
    print()
    for name, df in arms.items():
        df.to_csv(out / f'{name}.csv', index=False, encoding='utf-8-sig')
    print('输出: ' + str(out))


if __name__ == '__main__':
    main()


def arm_not_selected(ledger_dir: Path, params: ExitParams,
                     keep_status: tuple = ('PORTFOLIO_CAP', 'RISK_OR_LOT')) -> pd.DataFrame:
    '''C3: 同一天、同一个候选池里, 因为没抢到席位而没成交的信号。

    这组对照直接回答"两席择优有没有挑出更好的那两只":
    如果 T0 不比 C3 好, 说明排序没有把好机会挑出来。
    '''
    rows = []
    for f in sorted(ledger_dir.glob('ledger_*_30bp.csv')):
        arm = f.stem.replace('ledger_', '').split('_')[1]
        df = pd.read_csv(f, dtype=str)
        if 'status' not in df.columns:
            continue
        for r in df.itertuples(index=False):
            if r.status not in keep_status:
                continue
            if not r.symbol or not r.day:
                continue
            b = load_bars(r.symbol)
            si = index_of(b, r.day)
            if si < 0:
                continue
            res = run_entry(b, si, params)
            if res:
                res.update({'arm': 'C3_同池未选中', 'anchor': r.symbol,
                            'anchor_day': r.day, 'symbol': r.symbol,
                            'ledger_arm': arm, 'reject_reason': r.status})
                rows.append(res)
    return pd.DataFrame(rows)
