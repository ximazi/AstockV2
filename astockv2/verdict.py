"""验收裁决: 把"这个规则到底行不行"变成一个能自动判定的门槛。

裁决口径在看结果之前写死在这里, 不允许事后改阈值。

  主判据: 真实成交的每笔平均收益, 必须显著高于"同一只股票、同一年、
          换一个随机交易日入场"的安慰剂对照, 且 95% 区间下界 > 0。
  次判据: 必须同时高于"同一天换一只随机股票"的对照。
  稳定性: 三个年度里至少两年为正。
  稳健性: 去掉最大赢家之后仍然为正。

两条口径上的修正 (2026-11-24):

1. 对照组按"持仓窗口里含不含那次信号的行情"再拆一次:
     OVERLAP 窗口里含信号 -> 这笔安慰剂买的就是同一波行情, 只是买得更早;
     CLEAN   窗口里不含信号 -> 才是这只股票普通某一天的干净基准。
   主判据用 CLEAN。安慰剂抽样本身是均匀的, 没有未来信息, 但如果把
   OVERLAP 混进来, 等于允许对照组"提前上车"去拿走信号自己的那段行情,
   基准会被抬高 (实测 +7.2% vs +0.7%), 裁决就会把"买得早"误判成
   "信号没用"。
2. 区间用 20 个交易日的循环块 bootstrap。持仓周期是 20 日, 逐日独立
   重采样会把重叠的持仓当成独立样本, 区间会偏窄。

任何一条不满足, 裁决就是不通过。系统可以继续跑, 但不得据此扩大发现面
或切换正式规则。
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import pandas as pd


CI_LOW = 0.0        # 主判据: 差值 95% 区间下界必须大于 0
MIN_POSITIVE_YEARS = 2   # 稳定性: 至少两年为正
BOOT = 3000
SEED = 20261120
BLOCK = 20          # 循环块长度, 与 20 日持仓周期一致


def _resolve_col(a: pd.DataFrame, b: pd.DataFrame, col: str) -> str:
    '''列名兜底: 统一用 ret_pct, 没有就用净收益列 net_ret。'''
    if col in a.columns and col in b.columns:
        return col
    for alt in ('ret_pct', 'net_ret'):
        if alt in a.columns and alt in b.columns:
            return alt
    raise KeyError(f'两臂都没有可用的收益列: {col}')


def block_boot(a: pd.DataFrame, b: pd.DataFrame, col: str = 'ret_pct',
               key: str = 'anchor_day', n: int = BOOT,
               seed: int = SEED, block_size: int = BLOCK) -> tuple:
    '''20 交易日循环块 bootstrap 的均值差与 95% 区间。

    同一天的多笔交易高度相关, 而且 20 日持仓互相重叠, 所以按交易日
    分块重采样, 不能把每笔当独立样本。

    统计量是"两臂整体均值之差"。每次重采样抽一组交易日 (两臂并集),
    再把落在这些日子上的交易各自汇总体重算均值, 这样两臂始终用同一组
    抽样日, 区间和"整体均值差"是同一个量。
    '''
    col = _resolve_col(a, b, col)
    rng = random.Random(seed)
    va = a.groupby(key)[col].apply(lambda s: [v for v in s if v == v]).to_dict()
    vb = b.groupby(key)[col].apply(lambda s: [v for v in s if v == v]).to_dict()
    days = sorted(set(va) | set(vb))
    m = len(days)
    if m < 5:
        return (float('nan'), float('nan'), float('nan'), m)
    obs = float(a[col].mean()) - float(b[col].mean())
    blk = min(block_size, m)
    k_blocks = math.ceil(m / blk)
    # 按天预聚合: 一次重采样只需要把抽到的天的和与个数加起来, 不用重建交易列表
    sum_a = [sum(va.get(d, ())) for d in days]
    num_a = [len(va.get(d, ())) for d in days]
    sum_b = [sum(vb.get(d, ())) for d in days]
    num_b = [len(vb.get(d, ())) for d in days]
    diffs = []
    for _ in range(n):
        tot_a = tot_b = 0.0
        cnt_a = cnt_b = 0
        taken = 0
        for _ in range(k_blocks):
            start = rng.randrange(m)
            for offset in range(blk):
                if taken >= m:
                    break
                i = (start + offset) % m
                tot_a += sum_a[i]
                cnt_a += num_a[i]
                tot_b += sum_b[i]
                cnt_b += num_b[i]
                taken += 1
        if not cnt_a or not cnt_b:
            continue
        diffs.append(tot_a / cnt_a - tot_b / cnt_b)
    diffs.sort()
    if not diffs:
        return (obs, float('nan'), float('nan'), m)
    return (obs, diffs[int(len(diffs) * 0.025)], diffs[int(len(diffs) * 0.975)], m)


def split_overlap(control: pd.DataFrame) -> tuple:
    '''把安慰剂拆成 (CLEAN, OVERLAP)。

    没有 overlap 标记的旧结果一律当 CLEAN (保守: 不因为缺标记而抬高基准)。
    '''
    if 'overlaps_source_signal' not in control.columns:
        return control, control.iloc[0:0]
    flag = control['overlaps_source_signal'].astype(str).str.lower().isin(
        ('true', '1', 'yes'))
    return control[~flag], control[flag]


def yearly_diff(traded: pd.DataFrame, control: pd.DataFrame,
                col: str = 'ret_pct') -> pd.Series:
    '''按年给出真实成交减安慰剂的差值。'''
    t = traded.copy()
    c = control.copy()
    t['year'] = t['anchor_day'].astype(str).str[:4]
    c['year'] = c['anchor_day'].astype(str).str[:4]
    a = t.groupby('year')[col].mean()
    b = c.groupby('year')[col].mean()
    return (a - b).dropna()


def _check(item: str, obs: float, lo: float, hi: float, days: int,
           require: str, passed: bool | None = None) -> dict:
    return {
        '项': item,
        '差值%': round(obs * 100, 3),
        '区间下界%': round(lo * 100, 3),
        '区间上界%': round(hi * 100, 3),
        '可比交易日': days,
        '通过': bool(lo > CI_LOW) if passed is None else bool(passed),
        '要求': require,
    }


def judge(traded: pd.DataFrame, ctrl_same_stock: pd.DataFrame,
          ctrl_same_day: pd.DataFrame, label: str = 'R1') -> dict:
    '''按预先写死的口径给出裁决。

    traded 用同一套成交模型重跑出来的真实信号臂 (T0), 与对照组同口径。
    '''
    clean, overlap = split_overlap(ctrl_same_stock)
    checks = []

    obs, lo, hi, days = block_boot(traded, clean)
    checks.append(_check(
        f'{label} 主判据: 真实成交 - 同股异日安慰剂(干净桶)',
        obs, lo, hi, days, '95% 区间下界 > 0'))

    if len(overlap):
        obs, lo, hi, days = block_boot(traded, overlap)
        checks.append(_check(
            f'{label} 参考: 真实成交 - 同股异日安慰剂(窗口含信号行情)',
            obs, lo, hi, days,
            '参考项, 不参与总裁决 (这批对照买的是同一波行情的更早位置)'))

    obs2, lo2, hi2, days2 = block_boot(traded, ctrl_same_day)
    checks.append(_check(
        f'{label} 次判据: 真实成交 - 同日随机标的安慰剂',
        obs2, lo2, hi2, days2, '95% 区间下界 > 0'))

    yr = yearly_diff(traded, clean)
    pos = int((yr > 0).sum())
    detail = ', '.join(f'{y} {v*100:+.2f}%' for y, v in yr.items())
    checks.append({
        '项': f'{label} 稳定性: 分年差值为正的年份数',
        '差值%': round(float(yr.mean()) * 100, 3),
        '区间下界%': float('nan'),
        '区间上界%': float('nan'),
        '可比交易日': len(yr),
        '通过': bool(pos >= MIN_POSITIVE_YEARS),
        '要求': f'至少 {MIN_POSITIVE_YEARS} 年为正 (实际 {pos}/{len(yr)}; {detail})',
    })

    t = traded.copy()
    t['ret_ex_top'] = t['ret_pct']
    best = t['ret_pct'].idxmax()
    t.loc[best, 'ret_ex_top'] = float('nan')
    clean2 = clean.copy()
    clean2['ret_ex_top'] = clean2['ret_pct']
    obs3, lo3, hi3, days3 = block_boot(t, clean2, col='ret_ex_top')
    checks.append(_check(
        f'{label} 稳健性: 去掉最大赢家后仍优于安慰剂',
        obs3, lo3, hi3, days3, '95% 区间下界 > 0'))

    hard = [c for c in checks if not c['项'].startswith(f'{label} 参考')]
    passed = all(c['通过'] for c in hard)
    return {
        '裁决': '通过' if passed else '不通过',
        '判据': checks,
        '口径': {
            '主判据对照': ('同一只股票、同一年、换一个随机交易日入场, '
                        '且该笔持仓窗口不含任何真实信号行情 (干净桶)'),
            '参考对照': '同股异日中持仓窗口含真实信号行情的那部分 (买的是同一波行情的更早位置)',
            '次判据对照': '同一天、从全市场随机抽一只同样能成交的股票',
            '成本': '双边 30bp + 单边 3bp 滑点 (买卖两侧都扣)',
            '价格空间': '复权空间 (ATR 与止损用因果复权价, 涨跌停与限价用原始价)',
            '区间': f'按交易日 {BLOCK} 日循环块 bootstrap {BOOT} 次, 95%',
        },
    }


def to_markdown(v: dict) -> str:
    lines = []
    lines.append('# 验收裁决')
    lines.append('')
    for key in ('裁决_R1', '裁决_S4'):
        if key not in v:
            continue
        sub = v[key]
        lines.append(f"## {key}: {sub['裁决']}")
        lines.append('')
        lines.append('| 判据 | 差值 | 95% 区间 | 可比交易日 | 通过 | 要求 |')
        lines.append('|---|---:|---|---:|---|---|')
        for c in sub['判据']:
            is_ref = ' 参考' in c['项']
            mark = '—' if is_ref else ('是' if c['通过'] else '否')
            lines.append(f"| {c['项']} | {c['差值%']}% | [{c['区间下界%']}%, {c['区间上界%']}%] "
                         f"| {c['可比交易日']} | {mark} | {c['要求']} |")
        lines.append('')
        lines.append('### 口径')
        lines.append('')
        for k, val in sub['口径'].items():
            lines.append(f'- **{k}**: {val}')
        lines.append('')
    lines.append(f"**总裁决: {v.get('总裁决', '')}**")
    lines.append('')
    return '\n'.join(lines)


def judge_s4(s4_dir: Path) -> dict | None:
    '''S4 那一臂的裁决, 前提是已经跑过 run_s4_arm.py。'''
    arm = s4_dir / 's4_arm.csv'
    pbo = s4_dir / 's4_placebo.csv'
    day = s4_dir / 's4_same_day.csv'
    if not (arm.exists() and pbo.exists()):
        return None
    a = pd.read_csv(arm)
    b = pd.read_csv(pbo)
    a2 = a.copy()
    a2['ret_pct'] = a2['net_ret']
    b2 = b.copy()
    b2['ret_pct'] = b2['net_ret']
    ctrl_day = None
    if day.exists():
        c = pd.read_csv(day)
        c2 = c.copy()
        c2['ret_pct'] = c2['net_ret']
        ctrl_day = c2
    empty = pd.DataFrame({'anchor_day': [], 'ret_pct': []})
    return judge(a2, b2, ctrl_day if ctrl_day is not None else empty, label='S4')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ledger-dir', default=str('D:/AICode/Astock/results/research/mainwave_path/20261119_validated'))
    ap.add_argument('--control-dir', default=str('D:/AICode/AStockV2/results/control'))
    ap.add_argument('--s4-dir', default=str('D:/AICode/AStockV2/results/s4_arm'))
    ap.add_argument('--out', default=str('D:/AICode/AStockV2/results/verdict'))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cdir = Path(args.control_dir)

    from .exit_lab import load_trades
    trades = load_trades(Path(args.ledger_dir))
    ledger = trades.copy()
    ledger['ret_pct'] = ledger['pnl'] / ledger['notional']
    ledger['anchor_day'] = ledger['day']
    ledger = ledger[['anchor_day', 'symbol', 'ret_pct', 'pnl', 'notional', 'year', 'arm']]

    t0 = pd.read_csv(cdir / 'T0_真实信号.csv')
    t0['ret_pct'] = t0['net_ret']
    c2 = pd.read_csv(cdir / 'C2_同股异日.csv')
    c1 = pd.read_csv(cdir / 'C1_同日随机.csv')
    for c in (c1, c2):
        c['ret_pct'] = c['net_ret']
    c3_path = cdir / 'C3_同池未选中.csv'
    c3 = pd.read_csv(c3_path) if c3_path.exists() else pd.DataFrame()
    if len(c3):
        c3['ret_pct'] = c3['net_ret']

    print(f'冻结账本 {len(ledger)} 笔, 账本实际每笔 {ledger["ret_pct"].mean()*100:.3f}%')
    print(f'同口径重跑真实信号 (T0) {len(t0)} 笔, 每笔 {t0["ret_pct"].mean()*100:.3f}%')
    print(f'同股异日对照 {len(c2)} 笔, 每笔 {c2["ret_pct"].mean()*100:.3f}%')
    print(f'同日随机对照 {len(c1)} 笔, 每笔 {c1["ret_pct"].mean()*100:.3f}%')
    if len(c2) and 'overlaps_source_signal' in c2.columns:
        flag = c2['overlaps_source_signal'].astype(str).str.lower().isin(('true', '1', 'yes'))
        print(f'  其中窗口含信号行情 (OVERLAP) {int(flag.sum())} 笔, '
              f'每笔 {c2.loc[flag, "ret_pct"].mean()*100:.3f}%')
        print(f'  干净桶 (CLEAN) {int((~flag).sum())} 笔, '
              f'每笔 {c2.loc[~flag, "ret_pct"].mean()*100:.3f}%')
    print()

    v1 = judge(t0, c2, c1, label='R1')
    v1['口径']['样本'] = f'2024-2026 冻结账本的 {len(t0)} 笔成交, 用同一套成交模型重跑'
    if len(c3):
        obs, lo, hi, days_c3 = block_boot(t0, c3)
        v1['判据'].append({
            '项': 'R1 参考: 真实成交 - 同池未选中候选',
            '差值%': round(obs * 100, 3),
            '区间下界%': round(lo * 100, 3),
            '区间上界%': round(hi * 100, 3),
            '可比交易日': days_c3,
            '通过': bool(lo > CI_LOW),
            '要求': '95% 区间下界 > 0 (参考项, 不参与总裁决)',
        })
    # 口径参考: 冻结账本的实际盈亏 vs 同口径安慰剂
    obs, lo, hi, days_l = block_boot(ledger, c2)
    v1['判据'].append({
        '项': 'R1 参考: 冻结账本实际盈亏 - 同股异日安慰剂(旧口径)',
        '差值%': round(obs * 100, 3),
        '区间下界%': round(lo * 100, 3),
        '区间上界%': round(hi * 100, 3),
        '可比交易日': days_l,
        '通过': bool(lo > CI_LOW),
        '要求': '参考项, 不参与总裁决 (账本成交口径与对照组不同)',
    })

    v2 = judge_s4(Path(args.s4_dir))
    out_v = {'裁决_R1': v1}
    if v2:
        out_v['裁决_S4'] = v2
    out_v['总裁决'] = '通过' if all(
        out_v[k]['裁决'] == '通过' for k in ('裁决_R1', '裁决_S4') if k in out_v) else '不通过'

    for key in ('裁决_R1', '裁决_S4'):
        if key not in out_v:
            continue
        print(f'=== {key}: {out_v[key]["裁决"]} ===')
        for c in out_v[key]['判据']:
            is_ref = ' 参考' in c['项']
            mark = '参考' if is_ref else ('通过' if c['通过'] else '不通过')
            print(f"[{mark}] {c['项']}: 差值 {c['差值%']}% "
                  f"区间 [{c['区间下界%']}%, {c['区间上界%']}%] 要求: {c['要求']}")
        print()
    print('总裁决: ' + out_v['总裁决'])
    print()

    (out / 'verdict.json').write_text(json.dumps(out_v, ensure_ascii=False, indent=2),
                                       encoding='utf-8')
    (out / 'verdict.md').write_text(to_markdown(out_v), encoding='utf-8')
    t0.to_csv(out / 'traded_arm.csv', index=False, encoding='utf-8-sig')
    print('输出: ' + str(out))


if __name__ == '__main__':
    main()
