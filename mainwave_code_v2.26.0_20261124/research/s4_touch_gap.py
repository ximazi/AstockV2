"""S4: does the distance between the EMA20 touch and the relaunch matter? (external review 2026-09-26, E2)

For every kept S4 event: GAP = sessions since the last qualifying touch (low <= EMA20 x 1.02 with close >= EMA20 x
0.99, the S4 definition), and BIAS = close / EMA20 - 1 on the signal day. Buckets fixed before looking:
GAP 1-2 / 3-5 / 6-10, BIAS < 4% / 4-8% / >= 8%. Outcomes: LAD exit return (T+1 fixed), the volatility label
(+2.5 ATR before -2.5 ATR) and the -10% stop within 20 sessions. Per year, with counts.
python research/s4_touch_gap.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import four_dim_v2 as V2  # noqa: E402


def main():
    ev = pd.read_csv(ROOT / "results/research/s4_features/exclusion_events.csv", encoding="utf-8-sig", dtype={"year": str})
    ev = ev[~ev["排除"].astype(bool)]
    lad = pd.read_csv(ROOT / "results/research/exit_s4/trades.csv", encoding="utf-8-sig")
    lad = lad[lad.rule == "LAD"][["symbol", "date", "ret"]]
    rows = []
    for sym, g in ev.groupby("symbol"):
        raw, f = V2.frame(ROOT / "data" / f"{sym}_daily.csv")
        c, l = f.close.astype(float), f.low.astype(float)
        ema = c.ewm(span=20, adjust=False).mean()
        touch = ((l <= ema * 1.02) & (c >= ema * 0.99)).to_numpy()
        pos = {d: i for i, d in enumerate(f.date)}
        for r in g.itertuples():
            t = pos[r.date]
            gap = next((k for k in range(1, 11) if touch[t - k]), None)
            rows.append({"symbol": sym, "date": r.date, "year": r.year, "gap": gap, "bias": float(c.iloc[t] / ema.iloc[t] - 1),
                         "atr_win": r.atr_win, "stop": r.stop})
    d = pd.DataFrame(rows).merge(lad, on=["symbol", "date"], how="left")
    d["GAP档"] = pd.cut(d.gap, [0, 2, 5, 10], labels=["1-2日", "3-5日", "6-10日"])
    d["偏离档"] = pd.cut(d.bias, [-1, 0.04, 0.08, 1], labels=["<4%", "4-8%", ">=8%"])
    pd.set_option("display.width", 200)
    for col in ("GAP档", "偏离档"):
        t = d.groupby(["year", col], observed=True).agg(笔数=("ret", "size"), LAD平均=("ret", "mean"), LAD中位=("ret", "median"),
                                                          ATR胜率=("atr_win", "mean"), 止损率=("stop", "mean"))
        t[["LAD平均", "LAD中位", "ATR胜率", "止损率"]] = (t[["LAD平均", "LAD中位", "ATR胜率", "止损率"]].astype(float) * 100).round(1)
        print(t.to_string(), "\n")


if __name__ == "__main__":
    main()
