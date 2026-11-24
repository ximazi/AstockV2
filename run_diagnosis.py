# 两席账本逐笔路径分解入口。
# 用法: python run_diagnosis.py [--ledger-dir DIR] [--out DIR]
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from astockv2.config import V1_RESULTS, V2_RESULTS
from astockv2.diagnose import enrich, load_ledger, summarize

DEFAULT_LEDGER_DIR = V1_RESULTS / "20261119_validated"


def parse_year_arm(path: Path):
    stem = path.stem.replace("ledger_", "")
    year, arm, *_ = stem.split("_")
    return int(year), arm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ledger-dir', default=str(DEFAULT_LEDGER_DIR))
    ap.add_argument('--out', default=str(V2_RESULTS / 'diagnosis'))
    args = ap.parse_args()

    src = Path(args.ledger_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob('ledger_*_30bp.csv'))
    if not files:
        raise SystemExit('没有找到账本: ' + str(src))

    all_rows = []
    for f in files:
        year, arm = parse_year_arm(f)
        df = load_ledger(f)
        df['year'] = year
        df['arm'] = arm
        all_rows.append(enrich(df))

    trades = pd.concat(all_rows, ignore_index=True)
    trades.to_csv(out / 'trades_enriched.csv', index=False, encoding='utf-8-sig')
    return trades, out


def report(trades, out) -> None:
    closed = trades[trades['status'] == 'CLOSED']
    print(f"账本 {trades.groupby(['year', 'arm']).ngroups} 份, 候选 {len(trades)} 行, 成交 {len(closed)} 笔")
    print()
    print('=== 按年 / 臂 ===')
    summary, modes = summarize(trades, ['year', 'arm'])
    print(summary.to_string(index=False))
    print()
    print('=== 失败模式分布 (笔数 / 盈亏) ===')
    print(modes.to_string(index=False))
    print()
    print('=== 全部成交笔汇总 ===')
    s, m = summarize(trades, [])
    print(s.to_string(index=False))
    print()
    print(m.to_string(index=False))
    print()
    print('输出: ' + str(out / 'trades_enriched.csv'))


def r_report(trades) -> None:
    """R 倍数视角: 把账户结果拆成入场问题和出场问题。"""
    d = trades[trades['status'] == 'CLOSED'].copy()
    if not len(d):
        return
    d = d[d['realized_r'].notna()]
    win = d[d['pnl'] > 0]
    loss = d[d['pnl'] <= 0]
    print('=== R 倍数分解 ===')
    print(f"总成交 {len(d)} 笔, 胜率 {len(win)/len(d)*100:.1f}%, "
          f"平均实现 R {d['realized_r'].mean():.3f}, 中位 R {d['realized_r'].median():.3f}")
    print(f"平均单笔风险 {d['risk'].mean():.0f} 元, 平均盈亏 {d['pnl'].mean():.0f} 元")
    print()
    print('亏损单:')
    print(f"  笔数 {len(loss)}, 平均实现 R {loss['realized_r'].mean():.3f}, "
          f"平均持仓期最大浮盈 {loss['mfe_r_hold'].mean():.3f} R, "
          f"中位 {loss['mfe_r_hold'].median():.3f} R")
    for thr in (0.25, 0.5, 1.0):
        n = int((loss['mfe_r_hold'] >= thr).sum())
        print(f'  持仓期曾浮盈 >= {thr:.2f} R 的亏损单: {n} 笔 ({n/max(len(loss),1)*100:.0f}%), '
              f'合计 {loss[loss["mfe_r_hold"] >= thr]["pnl"].sum():.0f} 元')
    print()
    print('盈利单:')
    print(f"  笔数 {len(win)}, 平均实现 R {win['realized_r'].mean():.3f}, "
          f"平均持仓期最大浮盈 {win['mfe_r_hold'].mean():.3f} R, "
          f"平均捕获率 {win['capture'].mean():.2f}")
    print()
    print('组合期望 (按 R):')
    exp = d['realized_r'].mean()
    print(f'  每笔期望 {exp:+.3f} R, 每年 {len(d)} 笔样本')


if __name__ == '__main__':
    tr, o = main()
    report(tr, o)
    print()
    r_report(tr)
