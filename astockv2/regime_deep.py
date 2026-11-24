"""市场环境的深挖: 环境到底是原因, 还是只是同期涨跌的影子。

回答三个问题:
  1. 信号日是不是大量落在指数中期趋势向下的环境里;
  2. 把环境当混杂因素剔除后, 信号的优势还剩多少;
  3. 如果把差环境的日子全部不做, 账户会变成什么样, 那是 beta 还是 edge。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .regime import (attach_regime, by_regime, load_index, mix,
                     prev_trading_day_map, regime_frame, standardize,
                     within_signal_diff)

V2 = Path('D:/AICode/AStockV2')


def load_arms(control_dir: Path, s4_dir: Path, rmap: dict, prevmap: dict) -> dict:
    '''载入各臂, 并按"决策日"打环境标签。

    决策日 = 发出信号的那一天 (真实臂), 或安慰剂随机挑的那一天。
    对照臂的买入日往前推一个交易日, 才是它自己的决策日。
    '''
    arms = {}
    from .verdict import split_overlap
    t0 = pd.read_csv(control_dir / 'T0_真实信号.csv')
    t0['ret_pct'] = t0['net_ret']
    arms['R1 真实信号'] = attach_regime(t0, 'anchor_day', rmap)
    c1 = pd.read_csv(control_dir / 'C1_同日随机.csv')
    c1['ret_pct'] = c1['net_ret']
    arms['R1 同日随机标的'] = attach_regime(c1, 'anchor_day', rmap)
    c2 = pd.read_csv(control_dir / 'C2_同股异日.csv')
    c2['ret_pct'] = c2['net_ret']
    c2_clean, c2_over = split_overlap(c2)
    arms['R1 同股异日'] = attach_regime(c2_clean, 'anchor_day', rmap)
    arms['R1 同股异日·窗口含信号'] = attach_regime(c2_over, 'anchor_day', rmap)
    c3 = pd.read_csv(control_dir / 'C3_同池未选中.csv')
    c3['ret_pct'] = c3['net_ret']
    arms['R1 同池未选中'] = attach_regime(c3, 'anchor_day', rmap)
    sa = pd.read_csv(s4_dir / 's4_arm.csv')
    sa['ret_pct'] = sa['net_ret']
    arms['S4 信号'] = attach_regime(sa, 'anchor_day', rmap)
    sp = pd.read_csv(s4_dir / 's4_placebo.csv')
    sp['ret_pct'] = sp['net_ret']
    sp_clean, sp_over = split_overlap(sp)
    arms['S4 同股异日'] = attach_regime(sp_clean, 'anchor_day', rmap)
    arms['S4 同股异日·窗口含信号'] = attach_regime(sp_over, 'anchor_day', rmap)
    return arms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--control-dir', default=str(V2 / 'results' / 'control'))
    ap.add_argument('--s4-dir', default=str(V2 / 'results' / 's4_arm'))
    ap.add_argument('--ledger-dir', default=str('D:/AICode/Astock/results/research/mainwave_path/20261119_validated'))
    ap.add_argument('--out', default=str(V2 / 'results' / 'regime'))
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
    prevmap = prev_trading_day_map('hs300')
    arms = load_arms(cdir, sdir, rmap, prevmap)

    print('=== 一、各臂的市场环境构成 (决策日口径) ===')
    rows = []
    for nm, d in arms.items():
        m = mix(d)
        rows.append({'臂': nm, '笔数': len(d),
                     'DOWN%': round(m.get('DOWN', 0) * 100, 1),
                     'UP%': round(m.get('UP', 0) * 100, 1)})
    mixdf = pd.DataFrame(rows)
    print(mixdf.to_string(index=False))
    print()

    print('=== 二、同一臂内部: UP 环境比 DOWN 环境好多少 ===')
    diffs = []
    for nm, d in arms.items():
        r = within_signal_diff(d, nm)
        if r:
            diffs.append(r)
    diffdf = pd.DataFrame(diffs)
    print(diffdf.to_string(index=False))
    print()

    print('=== 三、剔除环境混杂后的差值 (真实信号 vs 同股异日) ===')
    for tag, a, b in (('R1', arms['R1 真实信号'], arms['R1 同股异日']),
                      ('S4', arms['S4 信号'], arms['S4 同股异日'])):
        d, detail = standardize(a, b)
        print(f'--- {tag}: 标准化差值 {d*100:+.3f}% ---')
        print(detail.to_string(index=False))
        for reg in ('DOWN', 'UP'):
            aa, bb = a[a['regime'] == reg], b[b['regime'] == reg]
            if len(aa) < 10 or len(bb) < 10:
                print(f'  {reg}: 样本不足')
                continue
            obs, lo, hi, days = block_boot(aa, bb)
            print(f'  {reg}: 差值 {obs*100:+.3f}%  95%区间 [{lo*100:+.3f}%, {hi*100:+.3f}%]  '
                  f'可比交易日 {days}')
        print()

    print('=== 四、年份 x 环境 交叉表 ===')
    trades = load_trades(Path(args.ledger_dir))
    trades['ret_pct'] = trades['pnl'] / trades['notional']
    trades['regime'] = trades['day'].map(rmap).fillna('UNKNOWN')
    trades['year'] = trades['day'].astype(str).str[:4]
    xt = trades.pivot_table(index='year', columns='regime', values='ret_pct',
                            aggfunc=['size', 'mean'])
    print(xt.to_string())
    print()

    print('=== 五、如果只做 DOWN 环境 (门禁反事实) ===')
    for tag, a, b in (('R1', arms['R1 真实信号'], arms['R1 同股异日']),
                      ('S4', arms['S4 信号'], arms['S4 同股异日'])):
        for reg in ('DOWN', 'UP'):
            aa, bb = a[a['regime'] == reg], b[b['regime'] == reg]
            if not len(aa):
                continue
            obs, lo, hi, days = block_boot(aa, bb)
            print(f'  {tag} {reg}: 信号 {len(aa)} 笔 平均 {aa["ret_pct"].mean()*100:+.3f}%  '
                  f'| 安慰剂 {len(bb)} 笔 平均 {bb["ret_pct"].mean()*100:+.3f}%  '
                  f'| 差值 {obs*100:+.3f}% 区间 [{lo*100:+.3f}%, {hi*100:+.3f}%]')
    print()
    mixdf.to_csv(out / 'arm_regime_mix_decision.csv', index=False, encoding='utf-8-sig')
    diffdf.to_csv(out / 'within_arm_up_vs_down.csv', index=False, encoding='utf-8-sig')
    print('输出: ' + str(out))


if __name__ == '__main__':
    main()


def robustness(cdir: Path, sdir: Path, indices=('hs300', 'csi1000', 'sse'),
               spans=(50, 100, 200), lookbacks=(5, 10, 20)) -> pd.DataFrame:
    '''不同指数 / EMA 长度 / 回看长度下, "DOWN 环境是否更好"是否稳定。'''
    prevmap = prev_trading_day_map('hs300')
    rows = []
    for idx in indices:
        for span in spans:
            for lb in lookbacks:
                rf = regime_frame(idx, span, lb)
                rmap = dict(zip(rf['date'].dt.strftime('%Y-%m-%d'), rf['regime']))
                specs = [
                    ('R1 真实信号', cdir / 'T0_真实信号.csv', 'anchor_day', None),
                    ('S4 信号', sdir / 's4_arm.csv', 'anchor_day', None),
                    ('S4 同股异日', sdir / 's4_placebo.csv', 'entry_date', prevmap),
                ]
                for tag, path, daycol, pm in specs:
                    df = pd.read_csv(path)
                    df['ret_pct'] = df['net_ret']
                    if pm:
                        df['decision_day'] = df[daycol].astype(str).map(pm)
                        daycol = 'decision_day'
                    df = attach_regime(df, daycol, rmap)
                    r = within_signal_diff(df, tag, n=600)
                    if r:
                        r['指数'] = idx
                        r['EMA'] = span
                        r['回看'] = lb
                        rows.append(r)
    return pd.DataFrame(rows)


def main2() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--control-dir', default=str(V2 / 'results' / 'control'))
    ap.add_argument('--s4-dir', default=str(V2 / 'results' / 's4_arm'))
    ap.add_argument('--out', default=str(V2 / 'results' / 'regime'))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print('=== 稳健性: 27 组参数 (3 指数 x 3 EMA 长度 x 3 回看) ===')
    print('看的是同一个臂内部 UP 环境比 DOWN 环境好多少 (差值为负 = DOWN 更好)')
    tab = robustness(Path(args.control_dir), Path(args.s4_dir))
    tab.to_csv(out / 'regime_robustness.csv', index=False, encoding='utf-8-sig')
    for tag in tab['臂'].unique():
        sub = tab[tab['臂'] == tag]
        neg = int((sub['差值%'] < 0).sum())
        sig = int((sub['区间上界%'] < 0).sum())
        print(f'  {tag}: {len(sub)} 组参数中 {neg} 组 DOWN 更好, 其中 {sig} 组显著')
        print('    ' + sub[['指数', 'EMA', '回看', 'UP平均%', 'DOWN平均%', '差值%',
                          '区间下界%', '区间上界%']].to_string(index=False).replace('\n', '\n    '))
    print()
    print('输出: ' + str(out / 'regime_robustness.csv'))


if __name__ == '__main__':
    main2()


def differential_robustness(sdir: Path, indices=('hs300', 'csi1000', 'sse'),
                            spans=(50, 100, 200), lookbacks=(5, 10, 20)) -> pd.DataFrame:
    '''每个参数组合下, 分环境算 "S4 信号 - 同股异日安慰剂" 的差值与区间。

    直接复用 placebo_audit.config_scan, 保证两个工具的口径不会各说各话。
    对照有三条口径: 干净桶(按源信号) / 干净桶(按同年任一信号) / 全部安慰剂。
    旧口径(全部)让对照组提前上车拿走信号自己的行情, 基准被抬高。
    '''
    from .placebo_audit import config_scan
    sig = pd.read_csv(sdir / 's4_arm.csv')
    pbo = pd.read_csv(sdir / 's4_placebo.csv')
    return config_scan(sig, pbo, indices=indices, spans=spans, lookbacks=lookbacks)


def main3() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--s4-dir', default=str(V2 / 'results' / 's4_arm'))
    ap.add_argument('--out', default=str(V2 / 'results' / 'regime'))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print('=== S4 信号 - 同股异日安慰剂, 分环境, 27 组参数 ===')
    print('两条口径并排: 干净桶 = 安慰剂持仓窗口不含信号行情 (主判据); 全部 = 旧口径')
    print()
    tab = differential_robustness(Path(args.s4_dir))
    tab.to_csv(out / 's4_regime_differential.csv', index=False, encoding='utf-8-sig')
    for bucket in tab['对照口径'].unique():
        for reg in ('DOWN', 'UP'):
            sub = tab[(tab['对照口径'] == bucket) & (tab['环境'] == reg)]
            if not len(sub):
                continue
            pos = int((sub['差值%'] > 0).sum())
            sig_pos = int((sub['区间下界%'] > 0).sum())
            sig_neg = int((sub['区间上界%'] < 0).sum())
            print(f'  [{bucket}] {reg}: {len(sub)} 组参数, 差值为正 {pos} 组, '
                  f'显著为正 {sig_pos} 组, 显著为负 {sig_neg} 组')
            print('    ' + sub[['指数', 'EMA', '回看', '信号笔数', '信号平均%', '对照笔数',
                              '对照平均%', '差值%', '区间下界%', '区间上界%']]
                  .to_string(index=False).replace('\n', '\n    '))
        print()
    print('输出: ' + str(out / 's4_regime_differential.csv'))


if __name__ == '__main__':
    main3()
