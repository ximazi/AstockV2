"""验收裁决: 把"这个规则到底行不行"变成一个能自动判定的门槛。

裁决口径在看结果之前写死在这里, 不允许事后改阈值。

  主判据: 真实成交的每笔平均收益, 必须显著高于"同一只股票、同一年、
          换一个随机交易日入场"的安慰剂对照。
          用按交易日分块的 bootstrap 给 95% 区间, 下界必须大于 0。
  次判据: 必须同时高于"同一天换一只随机股票"的对照。
  稳定性: 三个年度里至少两年为正。
  稳健性: 去掉最大赢家之后仍然为正。

任何一条不满足, 裁决就是不通过。系统可以继续跑, 但不得据此扩大发现面
或切换正式规则。
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd


CI_LOW = 0.0        # 主判据: 差值 95% 区间下界必须大于 0
MIN_POSITIVE_YEARS = 2   # 稳定性: 至少两年为正
BOOT = 3000
SEED = 20261120


def block_boot(a: pd.DataFrame, b: pd.DataFrame, col: str = 'ret_pct',
               key: str = 'anchor_day', n: int = BOOT,
               seed: int = SEED) -> tuple:
    '''按交易日分块 bootstrap 的均值差与 95% 区间。

    同一天的多笔交易高度相关, 按日期重采样, 不能把每笔当独立样本。

    统计量是"两臂整体均值之差"。每次重采样抽一组交易日 (两臂并集),
    再把落在这些日子上的交易各自汇总体重算均值, 这样两臂始终用同一组
    抽样日, 区间和"整体均值差"是同一个量。
    '''
    rng = random.Random(seed)
    va = a.groupby(key)[col].apply(lambda s: [v for v in s if v == v]).to_dict()
    vb = b.groupby(key)[col].apply(lambda s: [v for v in s if v == v]).to_dict()
    days = sorted(set(va) | set(vb))
    if len(days) < 5:
        return (float('nan'), float('nan'), float('nan'), len(days))
    obs = float(a[col].mean()) - float(b[col].mean())
    m = len(days)
    diffs = []
    for _ in range(n):
        idx = [days[rng.randrange(m)] for _ in range(m)]
        sa = [v for d in idx for v in va.get(d, ())]
        sb = [v for d in idx for v in vb.get(d, ())]
        if not sa or not sb:
            continue
        diffs.append(sum(sa) / len(sa) - sum(sb) / len(sb))
    diffs.sort()
    if not diffs:
        return (obs, float('nan'), float('nan'), m)
    return (obs, diffs[int(len(diffs) * 0.025)], diffs[int(len(diffs) * 0.975)], m)


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


def judge(traded: pd.DataFrame, ctrl_same_stock: pd.DataFrame,
          ctrl_same_day: pd.DataFrame) -> dict:
    '''按预先写死的口径给出裁决。'''
    checks = []
    obs, lo, hi, days = block_boot(traded, ctrl_same_stock)
    checks.append({
        '项': '主判据: 真实成交 - 同股异日安慰剂',
        '差值%': round(obs * 100, 3),
        '区间下界%': round(lo * 100, 3),
        '区间上界%': round(hi * 100, 3),
        '可比交易日': days,
        '通过': bool(lo > CI_LOW),
        '要求': '95% 区间下界 > 0',
    })
    obs2, lo2, hi2, days2 = block_boot(traded, ctrl_same_day)
    checks.append({
        '项': '次判据: 真实成交 - 同日随机标的安慰剂',
        '差值%': round(obs2 * 100, 3),
        '区间下界%': round(lo2 * 100, 3),
        '区间上界%': round(hi2 * 100, 3),
        '可比交易日': days2,
        '通过': bool(lo2 > CI_LOW),
        '要求': '95% 区间下界 > 0',
    })
    yr = yearly_diff(traded, ctrl_same_stock)
    pos = int((yr > 0).sum())
    detail = ', '.join(f'{y} {v*100:+.2f}%' for y, v in yr.items())
    checks.append({
        '项': '稳定性: 分年差值为正的年份数',
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
    ex_top = t['ret_ex_top'].mean()
    ctrl_same_stock = ctrl_same_stock.copy()
    ctrl_same_stock['ret_ex_top'] = ctrl_same_stock['ret_pct']
    obs3, lo3, hi3, days3 = block_boot(t, ctrl_same_stock, col='ret_ex_top')
    checks.append({
        '项': '稳健性: 去掉最大赢家后仍优于安慰剂',
        '差值%': round(obs3 * 100, 3),
        '区间下界%': round(lo3 * 100, 3),
        '区间上界%': round(hi3 * 100, 3),
        '可比交易日': days3,
        '通过': bool(lo3 > CI_LOW),
        '要求': '95% 区间下界 > 0',
    })

    passed = all(c['通过'] for c in checks)
    return {
        '裁决': '通过' if passed else '不通过',
        '判据': checks,
        '口径': {
            '主判据对照': '同一只股票、同一年、换一个随机交易日入场',
            '次判据对照': '同一天、从全市场随机抽一只同样能成交的股票',
            '成本': '双边 30bp + 单边 3bp 滑点',
            '样本': '2024-2026 冻结账本的 159 笔成交',
            '区间': f'按交易日分块 bootstrap {BOOT} 次, 95%',
        },
    }


def to_markdown(v: dict) -> str:
    lines = []
    lines.append('# 验收裁决')
    lines.append('')
    lines.append(f"**裁决: {v['裁决']}**")
    lines.append('')
    lines.append('| 判据 | 差值 | 95% 区间 | 可比交易日 | 通过 | 要求 |')
    lines.append('|---|---:|---|---:|---|---|')
    for c in v['判据']:
        lines.append(f"| {c['项']} | {c['差值%']}% | [{c['区间下界%']}%, {c['区间上界%']}%] "
                     f"| {c['可比交易日']} | {'是' if c['通过'] else '否'} | {c['要求']} |")
    lines.append('')
    lines.append('## 口径')
    lines.append('')
    for k, val in v['口径'].items():
        lines.append(f'- **{k}**: {val}')
    lines.append('')
    return '\n'.join(lines)


def judge_s4(s4_dir: Path) -> list:
    '''S4 那一臂的裁决, 前提是已经跑过 run_s4_arm.py。'''
    arm = s4_dir / 's4_arm.csv'
    pbo = s4_dir / 's4_placebo.csv'
    day = s4_dir / 's4_same_day.csv'
    if not (arm.exists() and pbo.exists()):
        return []
    a = pd.read_csv(arm)
    b = pd.read_csv(pbo)
    out = []
    a2 = a.copy()
    a2['ret_pct'] = a2['net_ret']
    b2 = b.copy()
    b2['ret_pct'] = b2['net_ret']
    obs, lo, hi, days = block_boot(a2, b2)
    out.append({
        '项': 'S4 主判据: S4 信号 - 同股异日安慰剂',
        '差值%': round(obs * 100, 3),
        '区间下界%': round(lo * 100, 3),
        '区间上界%': round(hi * 100, 3),
        '可比交易日': days,
        '通过': bool(lo > CI_LOW),
        '要求': '95% 区间下界 > 0',
    })
    if day.exists():
        c = pd.read_csv(day)
        c2 = c.copy()
        c2['ret_pct'] = c2['net_ret']
        obs2, lo2, hi2, days2 = block_boot(a2, c2)
        out.append({
            '项': 'S4 次判据: S4 信号 - 同日随机标的安慰剂',
            '差值%': round(obs2 * 100, 3),
            '区间下界%': round(lo2 * 100, 3),
            '区间上界%': round(hi2 * 100, 3),
            '可比交易日': days2,
            '通过': bool(lo2 > CI_LOW),
            '要求': '95% 区间下界 > 0',
        })
    return out


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
    t = trades.copy()
    t['ret_pct'] = t['pnl'] / t['notional']
    t['anchor_day'] = t['day']
    t = t[['anchor_day', 'symbol', 'ret_pct', 'pnl', 'notional', 'year', 'arm']]

    c2 = pd.read_csv(cdir / 'C2_同股异日.csv')
    c1 = pd.read_csv(cdir / 'C1_同日随机.csv')
    for c in (c1, c2):
        c['ret_pct'] = c['net_ret']
    c3_path = cdir / 'C3_同池未选中.csv'
    c3 = pd.read_csv(c3_path) if c3_path.exists() else pd.DataFrame()
    if len(c3):
        c3['ret_pct'] = c3['net_ret']

    print(f'真实成交 {len(t)} 笔, 平均每笔 {t["ret_pct"].mean()*100:.3f}%')
    print(f'同股异日对照 {len(c2)} 笔, 平均每笔 {c2["ret_pct"].mean()*100:.3f}%')
    print(f'同日随机对照 {len(c1)} 笔, 平均每笔 {c1["ret_pct"].mean()*100:.3f}%')
    print()

    v = judge(t, c2, c1)
    if len(c3):
        obs, lo, hi, days_c3 = block_boot(t, c3)
        v['判据'].append({
            '项': '参考: 真实成交 - 同池未选中候选',
            '差值%': round(obs * 100, 3),
            '区间下界%': round(lo * 100, 3),
            '区间上界%': round(hi * 100, 3),
            '可比交易日': days_c3,
            '通过': bool(lo > CI_LOW),
            '要求': '95% 区间下界 > 0 (参考项, 不参与总裁决)',
        })
    s4 = judge_s4(Path(args.s4_dir))
    if s4:
        v['判据'].extend(s4)
        hard = [c for c in v['判据'] if not c['项'].startswith('参考')]
        v['裁决'] = '通过' if all(c['通过'] for c in hard) else '不通过'
    print('=== 验收裁决 ===')
    for c in v['判据']:
        mark = '通过' if c['通过'] else '不通过'
        print(f"[{mark}] {c['项']}: 差值 {c['差值%']}% "
              f"区间 [{c['区间下界%']}%, {c['区间上界%']}%] 要求: {c['要求']}")
    print()
    print('裁决: ' + v['裁决'])
    print()

    (out / 'verdict.json').write_text(json.dumps(v, ensure_ascii=False, indent=2),
                                       encoding='utf-8')
    (out / 'verdict.md').write_text(to_markdown(v), encoding='utf-8')
    t.to_csv(out / 'traded_arm.csv', index=False, encoding='utf-8-sig')
    print('输出: ' + str(out))


if __name__ == '__main__':
    main()
