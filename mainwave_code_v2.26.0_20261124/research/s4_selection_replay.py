"""Account replay picking S4 candidates by the features that passed research/s4_selection_features.py (fixed there
before running): signal-day close location (lower is better) and latest quarterly sales growth (higher is better).
Score of a candidate = mean of its within-day percentile ranks among the candidates entering that day (equal weights,
nothing fitted; a missing feature counts 0.5). Account 5 x 20% with the G1 entry filter, 100 seeds, as in
research/account_replay.py; compared with random picks, each feature alone, and the reverse composite (sanity check).
python research/s4_selection_replay.py -> results/research/s4_selection/replay.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import account_replay as AR  # noqa: E402

OUT = ROOT / "results/research/s4_selection"


def scores(trades, feat, cols):
    """Within the candidates sharing an entry day: percentile rank per feature (direction applied), averaged."""
    t = pd.DataFrame({"i": range(len(trades)), "symbol": [x["symbol"] for x in trades], "date": [x["signal"] for x in trades],
                      "entry": [x["entry"] for x in trades]})
    t = t.merge(feat, on=["symbol", "date"], how="left")
    parts = []
    for col, sign in cols:
        parts.append((sign * t[col]).groupby(t.entry).rank(pct=True).fillna(0.5))
    s = sum(parts) / len(parts)
    return dict(zip(t.i, s))


def main():
    feat = pd.read_csv(OUT / "features.csv", encoding="utf-8-sig")[["symbol", "date", "clv", "q_sales_yoy"]]
    trades = AR.s4_trades()
    bench = pd.read_csv(ROOT / "data/000852_SH_index.csv")
    bench["d"] = pd.to_datetime(bench.trade_date).dt.strftime("%Y-%m-%d")
    first = min(x["entry"] for x in trades)
    last = max(max(fd for fd, _, _ in x["fills"]) for x in trades)
    calendar = [d for d in bench.d if first <= d <= last]
    variants = {"随机挑选": None,
                "组合分（收盘位置低 + 营收增速高）": scores(trades, feat, [("clv", -1), ("q_sales_yoy", 1)]),
                "只按收盘位置低": scores(trades, feat, [("clv", -1)]),
                "只按营收增速高": scores(trades, feat, [("q_sales_yoy", 1)]),
                "反向组合（对照）": scores(trades, feat, [("clv", 1), ("q_sales_yoy", -1)])}
    rows = []
    for name, sc in variants.items():
        runs = []
        for seed in range(AR.SEEDS):
            nav, ex, done = AR.run_account(trades, 5, 0.20, True, seed, calendar, score=sc)
            runs.append(AR.metrics(nav, ex, done))
        m = pd.DataFrame(runs)
        rec = {"挑选方式": name}
        for col in ("总收益%", "2025%", "2026%", "最大回撤%", "成交笔数", "胜率%", "盈亏比", "每笔期望%", "最长连亏"):
            rec[col] = round(m[col].median(), 2)
        rec["总收益 5%~95%"] = f"{m['总收益%'].quantile(0.05):.1f} ~ {m['总收益%'].quantile(0.95):.1f}"
        rec["优于随机中位的种子%"] = None
        rows.append(rec)
        if name == "随机挑选":
            rand_median = m["总收益%"].median()
        rec["优于随机中位的种子%"] = round((m["总收益%"] > rand_median).mean() * 100)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "replay.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 260)
    print(r.to_string(index=False))


if __name__ == "__main__":
    main()
