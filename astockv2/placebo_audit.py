"""安慰剂口径审计: 回答"对照组是不是被污染了"。

背景 (2026-11-24): 之前的验收把"同股异日"安慰剂与真实信号直接比大小。
事后发现安慰剂里有一批交易, 它的持仓窗口正好含住那次信号的行情 —— 抽样
本身是均匀的, 没有未来信息, 但**持有期与信号重叠**, 于是它买的是同一波
行情的更早位置, 收益远高于普通交易日, 把基准抬高了。

实测 (S4, 5,403 个信号 vs 10,204 笔安慰剂):
    窗口含信号行情的那批   n=1,494  平均 +7.373%
    窗口不含信号行情的那批 n=8,710  平均 +0.663%
    真实信号                       平均 +0.546%
安慰剂总体平均 +1.651% 与信号之间的差距, 约 87% 由重叠那批贡献。

所以任何"信号 vs 安慰剂"的结论, 必须同时报两条口径:
    干净桶  安慰剂持仓窗口不含任何信号行情 —— 主判据口径;
    全部    旧口径, 把重叠那批也算进来。
本模块把两条口径并排算出来, 并在 27 组环境参数下各算一遍。
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from .config import V2_RESULTS
from .regime import regime_frame

SEP = '=' * 78


def mark_overlap(sig: pd.DataFrame, pbo: pd.DataFrame) -> pd.DataFrame:
    '''标出两类"安慰剂持仓窗口含住信号行情"的行。

    overlaps_source  窗口含住**它自己对应的那次信号**的行情 —— 抽样时就写下
                     的标记 (control.py / s4_arm.py 里的 overlaps_source_signal)。
    overlaps_any     窗口含住**同一只股票同一年里任意一次信号**的行情。
                     同一只股票常常有多次信号, 一笔安慰剂虽然是为信号 A 抽的,
                     窗口里却可能撞上信号 B 的那一波。

    两种口径都报: 越宽的口径剔得越多, 基准越保守。
    '''
    pbo = pbo.copy()
    pbo['anchor_day'] = pbo['anchor_day'].astype(str)
    pbo['exit_date'] = pbo['exit_date'].astype(str)
    sig_days: dict = {}
    for sym, day in sig[['symbol', 'anchor_day']].itertuples(index=False):
        sig_days.setdefault(sym, []).append(str(day))
    for v in sig_days.values():
        v.sort()

    any_flags = []
    for sym, day, exit_day in pbo[['symbol', 'anchor_day', 'exit_date']].itertuples(index=False):
        any_flags.append(any(day <= d <= exit_day for d in sig_days.get(sym, [])))
    pbo['overlaps_any'] = any_flags

    if 'overlaps_source_signal' in pbo.columns:
        pbo['overlaps_source'] = (pbo['overlaps_source_signal'].astype(str).str.lower()
                                  .isin(('true', '1', 'yes')))
    elif 'source_signal_day' in pbo.columns and 'entry_date' in pbo.columns:
        pbo['overlaps_source'] = [
            any(str(e) <= str(s) <= str(x)
                for s in sig_days.get(sym, []) if str(s) == str(src))
            for sym, e, src, x in pbo[['symbol', 'entry_date', 'source_signal_day',
                                       'exit_date']].itertuples(index=False)]
    else:
        pbo['overlaps_source'] = pbo['overlaps_any']
    return pbo


def bucket_table(sig: pd.DataFrame, pbo: pd.DataFrame, rmap: dict | None = None) -> pd.DataFrame:
    '''把安慰剂按"相对信号日先后"与"窗口是否含信号"拆成几桶。'''
    sig = sig.copy()
    pbo = pbo.copy()
    for df in (sig, pbo):
        df['anchor_day'] = df['anchor_day'].astype(str)
        df['year'] = df['anchor_day'].str[:4]
        if rmap:
            df['regime'] = df['anchor_day'].map(rmap).fillna('UNKNOWN')
    sig_days: dict = {}
    for sym, day in sig[['symbol', 'anchor_day']].itertuples(index=False):
        sig_days.setdefault(sym, []).append(day)
    for v in sig_days.values():
        v.sort()
    rows = []
    for sym, day, exit_day in pbo[['symbol', 'anchor_day', 'exit_date']].itertuples(index=False):
        days = sig_days.get(sym, [])
        same_year = [d for d in days if d[:4] == str(day)[:4]]
        if not same_year:
            rows.append('NO_SIG')
        elif any(str(day) <= d <= str(exit_day) for d in same_year):
            rows.append('OVERLAP_窗口含信号')
        elif str(day) < min(same_year):
            rows.append('EARLY_信号前且窗口不含')
        elif str(day) > max(same_year):
            rows.append('LATE_信号后')
        else:
            rows.append('MID_多信号之间')
    pbo['bucket'] = rows
    out = []
    for name, df in (('真实信号', sig),) + tuple(
            (b, pbo[pbo['bucket'] == b]) for b in
            ['OVERLAP_窗口含信号', 'EARLY_信号前且窗口不含', 'MID_多信号之间',
             'LATE_信号后', 'NO_SIG']):
        if not len(df):
            continue
        out.append({'桶': name, '笔数': len(df),
                    '平均%': round(float(df['net_ret'].mean()) * 100, 3),
                    '中位%': round(float(df['net_ret'].median()) * 100, 3),
                    '胜率%': round(float((df['net_ret'] > 0).mean()) * 100, 1)})
    return pd.DataFrame(out)


def arm_diff(sig: pd.DataFrame, pbo: pd.DataFrame) -> dict:
    '''信号 vs 安慰剂的各口径差值与区间。'''
    from .verdict import block_boot
    a = sig.copy()
    a['ret_pct'] = a['net_ret']
    res = {}
    buckets = (
        ('干净桶(按源信号)', pbo[~pbo['overlaps_source']]),
        ('干净桶(按同年任一信号)', pbo[~pbo['overlaps_any']]),
        ('窗口含源信号', pbo[pbo['overlaps_source']]),
        ('全部安慰剂', pbo),
    )
    for name, df in buckets:
        if not len(df) or not len(a):
            continue
        b = df.copy()
        b['ret_pct'] = b['net_ret']
        obs, lo, hi, days = block_boot(a, b)
        res[name] = {'差值%': round(obs * 100, 3),
                     '区间下界%': round(lo * 100, 3),
                     '区间上界%': round(hi * 100, 3),
                     '可比交易日': days,
                     '对照笔数': len(df),
                     '对照平均%': round(float(df['net_ret'].mean()) * 100, 3),
                     '信号平均%': round(float(a['net_ret'].mean()) * 100, 3)}
    return res


def config_scan(sig: pd.DataFrame, pbo: pd.DataFrame,
                indices=('hs300', 'csi1000', 'sse'),
                spans=(50, 100, 200), lookbacks=(5, 10, 20)) -> pd.DataFrame:
    '''27 组环境参数下, 分环境算信号与两个安慰剂桶的差值。

    参数空间与 2026-11-20 那次分层检验一致 (3 指数 x 3 EMA x 3 回看), 这样
    两版的数字可以直接对照。
    '''
    from .verdict import block_boot
    rows = []
    pbo = mark_overlap(sig, pbo)
    for idx in indices:
        for span in spans:
            for lb in lookbacks:
                rf = regime_frame(idx, span, lb)
                rmap = dict(zip(rf['date'].dt.strftime('%Y-%m-%d'), rf['regime']))
                a = sig.copy()
                a['anchor_day'] = a['anchor_day'].astype(str)
                a['regime'] = a['anchor_day'].map(rmap).fillna('UNKNOWN')
                b = pbo.copy()
                b['regime'] = b['anchor_day'].map(rmap).fillna('UNKNOWN')
                for reg in ('DOWN', 'UP'):
                    aa = a[a['regime'] == reg]
                    if len(aa) < 20:
                        continue
                    for label, mask in (('干净桶(按源信号)', ~b['overlaps_source']),
                                        ('干净桶(按同年任一信号)', ~b['overlaps_any']),
                                        ('全部安慰剂', b['overlaps_any'] | ~b['overlaps_any'])):
                        bb = b[mask & (b['regime'] == reg)]
                        if len(bb) < 20:
                            continue
                        x = aa.copy()
                        x['ret_pct'] = x['net_ret']
                        y = bb.copy()
                        y['ret_pct'] = y['net_ret']
                        obs, lo, hi, _ = block_boot(x, y)
                        rows.append({
                            '指数': idx, 'EMA': span, '回看': lb, '环境': reg,
                            '对照口径': label,
                            '信号笔数': len(aa), '对照笔数': len(bb),
                            '信号平均%': round(float(aa['net_ret'].mean()) * 100, 3),
                            '对照平均%': round(float(bb['net_ret'].mean()) * 100, 3),
                            '差值%': round(obs * 100, 3),
                            '区间下界%': round(lo * 100, 3),
                            '区间上界%': round(hi * 100, 3),
                        })
    return pd.DataFrame(rows)


def analyze(name: str, sig_path: Path, pbo_path: Path, out_dir: Path,
            do_config_scan: bool = True) -> None:
    from .verdict import split_overlap
    sig = pd.read_csv(sig_path)
    pbo = pd.read_csv(pbo_path)
    sig['anchor_day'] = sig['anchor_day'].astype(str)
    pbo['anchor_day'] = pbo['anchor_day'].astype(str)
    pbo['exit_date'] = pbo['exit_date'].astype(str)
    pbo = mark_overlap(sig, pbo)

    print(SEP)
    print(f'{name}   信号 n={len(sig)}   安慰剂 n={len(pbo)}')
    print(SEP)
    print()
    print('--- 安慰剂分桶 ---')
    print(bucket_table(sig, pbo).to_string(index=False))
    print()
    print('--- 信号 vs 安慰剂 (两条口径) ---')
    for k, v in arm_diff(sig, pbo).items():
        print(f'  {k:<12} 信号 {v["信号平均%"]:+.3f}%  对照 {v["对照平均%"]:+.3f}% '
              f'({v["对照笔数"]} 笔)  差值 {v["差值%"]:+.3f}%  '
              f'95%[{v["区间下界%"]:+.3f}%, {v["区间上界%"]:+.3f}%]  '
              f'可比交易日 {v["可比交易日"]}')
    print()

    if do_config_scan:
        print('--- 27 组环境参数扫描 ---')
        tab = config_scan(sig, pbo)
        if len(tab):
            tab.to_csv(out_dir / f'placebo_config_scan_{name.split()[0]}.csv',
                        index=False, encoding='utf-8-sig')
            for bucket in tab['对照口径'].unique():
                for reg in ('DOWN', 'UP'):
                    sub = tab[(tab['对照口径'] == bucket) & (tab['环境'] == reg)]
                    if not len(sub):
                        continue
                    sig_pos = int((sub['区间下界%'] > 0).sum())
                    sig_neg = int((sub['区间上界%'] < 0).sum())
                    print(f'  [{bucket}] {reg}: {len(sub)} 组, '
                          f'显著为正 {sig_pos} 组, 显著为负 {sig_neg} 组, '
                          f'无显著差异 {len(sub) - sig_pos - sig_neg} 组')
        print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--control-dir', default=str(V2_RESULTS / 'control'))
    ap.add_argument('--s4-dir', default=str(V2_RESULTS / 's4_arm'))
    ap.add_argument('--out', default=str(V2_RESULTS / 'placebo_audit'))
    ap.add_argument('--no-config-scan', action='store_true')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cdir = Path(args.control_dir)
    sdir = Path(args.s4_dir)

    if (sdir / 's4_arm.csv').exists():
        analyze('S4 信号 vs 同股异日', sdir / 's4_arm.csv',
                sdir / 's4_placebo.csv', out, not args.no_config_scan)
    if (cdir / 'T0_真实信号.csv').exists():
        analyze('R1 真实信号 vs 同股异日', cdir / 'T0_真实信号.csv',
                cdir / 'C2_同股异日.csv', out, not args.no_config_scan)
    print('输出: ' + str(out))


if __name__ == '__main__':
    main()
