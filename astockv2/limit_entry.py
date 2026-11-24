"""入场价格实验: "不追高"能不能修好 S4。

问题来自市场环境分层: 在指数单边上行时, S4 追启动日开盘, 系统性比同一只
股票随机某天买还差 1%/笔。这里直接改入场规则, 其余全部不变:

  次日开盘买 (现用)  ->  不高于信号日收盘的限价买, 最多等 5 根

对照组是同一只股票、同一年、随机某天, 用同样的入场规则, 所以比较的仍然
只是"什么时候买"。
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from .bars import bars as load_bars
from .bars import index_of
from .execution import ExitParams, simulate_entry_limit, simulate_exit
from .s4_arm import load_signals

V2 = Path('D:/AICode/AStockV2')


def run_limit(signals: pd.DataFrame, params: ExitParams, max_chase: float,
              wait_bars: int, k: int = 2, seed: int = 20261120,
              same_stock: bool = True) -> tuple:
    '''返回 (信号臂, 安慰剂臂)。'''
    rng = random.Random(seed)
    sig_rows, pbo_rows = [], []
    for r in signals.itertuples(index=False):
        b = load_bars(r.symbol)
        si = index_of(b, r.day)
        if si < 0:
            continue
        e = simulate_entry_limit(b, si, params, max_chase, wait_bars)
        if e['status'] == 'OK':
            res = simulate_exit(b, e['entry_idx'], e['entry_price'],
                                e['initial_stop'], params, e['atr0'])
            if res['status'] == 'CLOSED':
                res.update({'arm': 'S4 信号', 'symbol': r.symbol,
                            'anchor_day': r.day, 'label': r.label,
                            'waited': e['waited']})
                sig_rows.append(res)
        if not same_stock:
            continue
        year = str(r.day)[:4]
        days = [i for i in range(21, len(b) - 1)
                if str(b['date'].iloc[i].date())[:4] == year]
        if len(days) < 5:
            continue
        for ci in rng.sample(days, min(k, len(days))):
            if abs(ci - si) < 3:
                continue
            e2 = simulate_entry_limit(b, ci, params, max_chase, wait_bars)
            if e2['status'] != 'OK':
                continue
            res2 = simulate_exit(b, e2['entry_idx'], e2['entry_price'],
                                 e2['initial_stop'], params, e2['atr0'])
            if res2['status'] == 'CLOSED':
                res2.update({'arm': 'S4 同股异日', 'symbol': r.symbol,
                             'anchor_day': str(b['date'].iloc[ci].date()),
                             'label': '', 'waited': e2['waited']})
                pbo_rows.append(res2)
    return pd.DataFrame(sig_rows), pd.DataFrame(pbo_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=0)
    ap.add_argument('--k', type=int, default=2)
    ap.add_argument('--out', default=str(V2 / 'results' / 'limit_entry'))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    params = ExitParams(stop_atr=2.0, trail='low10', activate_atr=2.0,
                         targets=((3.0, 0.5), (6.0, 0.25)), horizon=20,
                         cost_bp=30.0, slippage_bp=3.0)
    sig = load_signals()
    if args.sample:
        sig = sig.head(args.sample)
    print(f'S4 信号 {len(sig)} 条')

    from .regime import attach_regime, regime_frame
    from .verdict import block_boot
    rf = regime_frame('hs300', 100, 10)
    rmap = dict(zip(rf['date'].dt.strftime('%Y-%m-%d'), rf['regime']))

    for tag, chase, wait in (('限价不高于信号收盘, 最多等5根', 0.0, 5),
                             ('限价不高于信号收盘+0.5%, 最多等5根', 0.005, 5),
                             ('限价不高于信号收盘, 当天不满足就放弃', 0.0, 0)):
        a, b = run_limit(sig, params, chase, wait, args.k)
        if not len(a):
            print(f'{tag}: 无成交')
            continue
        a['ret_pct'] = a['net_ret']
        b['ret_pct'] = b['net_ret']
        a = attach_regime(a, 'anchor_day', rmap)
        b = attach_regime(b, 'anchor_day', rmap)
        print(f'=== {tag} ===')
        print(f'  信号 {len(a)} 笔 (成交率 {len(a)/len(sig)*100:.0f}%), '
              f'平均每笔 {a["net_ret"].mean()*100:+.3f}%, 胜率 {(a["net_ret"]>0).mean()*100:.1f}%')
        print(f'  安慰剂 {len(b)} 笔, 平均每笔 {b["net_ret"].mean()*100:+.3f}%, '
              f'胜率 {(b["net_ret"]>0).mean()*100:.1f}%')
        obs, lo, hi, days = block_boot(a, b)
        print(f'  信号 - 安慰剂: {obs*100:+.3f}%  95%区间 [{lo*100:+.3f}%, {hi*100:+.3f}%]  '
              f'可比交易日 {days}')
        for reg in ('DOWN', 'UP'):
            aa, bb = a[a['regime'] == reg], b[b['regime'] == reg]
            if len(aa) < 20 or len(bb) < 20:
                continue
            o2, l2, h2, _ = block_boot(aa, bb)
            print(f'    {reg}: 信号 {len(aa)} 笔 {aa["net_ret"].mean()*100:+.3f}%  '
                  f'对照 {len(bb)} 笔 {bb["net_ret"].mean()*100:+.3f}%  '
                  f'差值 {o2*100:+.3f}% 区间 [{l2*100:+.3f}%, {h2*100:+.3f}%]')
        a.to_csv(out / f'limit_{chase}_{wait}_arm.csv', index=False, encoding='utf-8-sig')
        b.to_csv(out / f'limit_{chase}_{wait}_placebo.csv', index=False, encoding='utf-8-sig')
        print()
    print('输出: ' + str(out))


if __name__ == '__main__':
    main()
