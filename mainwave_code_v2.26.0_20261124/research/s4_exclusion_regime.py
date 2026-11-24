"""S4 (pullback to EMA20 then relaunch): stack the three exclusion rules, and split by market environment.

Sample: every pullback-to-EMA20 event 2025-01..2026-08-24, including those the top-divergence rule removed, buys in
2026-03 / 2026-07 excluded. Entry next open (gap >= 5% / limit-up open / not traded skipped), as in s4_feature_discrimination.
Exclusion rules (fixed before looking at the stacked result):
  R1 top divergence (the user's definition, the `diverge` flag already used to form S4)
  R2 MACD green bars growing (DIF < DEA and the histogram lower than 2 sessions ago)
  R3 close more than 15% under the 60-session high (first bounce after a top, heavy overhead supply)
Environment at the signal close (CSI 1000 and all-A breadth, point in time, thresholds fixed a priori):
  trend    STRONG close > MA20 > MA60 / WEAK close < MA20 < MA60 / MIXED otherwise
  breadth  share of A-shares above their MA60: HIGH >= 60% / MID 40-60% / LOW < 40%
Outcomes within 20 sessions: reach +10% / +15% before the -10% stop, the -10% stop touched, and the volatility-neutral
label (+2.5 ATR20 before -2.5 ATR20; a session touching both counts as a loss).
python research/s4_exclusion_regime.py -> results/research/s4_features/exclusion_*.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import four_dim_v2 as V2  # noqa: E402
import s4_feature_discrimination as S  # noqa: E402
import s4_macd_state as M  # noqa: E402

OUT = ROOT / "results/research/s4_features"


def outcome(f, t, atr):
    e = float(f.open.iloc[t + 1])
    stop, best, atr_win = False, 0.0, None
    for k in range(t + 1, t + 21):
        lo, hi = float(f.low.iloc[k]), float(f.high.iloc[k])
        if atr_win is None:
            if lo <= e - 2.5 * atr:
                atr_win = False
            elif hi >= e + 2.5 * atr:
                atr_win = True
        if lo <= e * 0.90:
            stop = True
            break
        best = max(best, hi / e - 1)
    return {"win10": best >= 0.10, "win15": best >= 0.15, "stop": stop, "atr_win": bool(atr_win)}


def environment():
    m = pd.read_csv(ROOT / "results/research/regime/market_regime_daily.csv", encoding="utf-8-sig")
    c = m.csi1000
    ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
    m["趋势"] = "MIXED"
    m.loc[(c > ma20) & (ma20 > ma60), "趋势"] = "STRONG"
    m.loc[(c < ma20) & (ma20 < ma60), "趋势"] = "WEAK"
    b = m.breadth_above_ma60
    m["宽度"] = pd.cut(b, [-1, 0.4, 0.6, 2], right=False, labels=["LOW", "MID", "HIGH"]).astype(str)
    return m.set_index("date")[["趋势", "宽度"]]


def build():
    ev = pd.read_csv(ROOT / "results/research/four_dim_v2/events_excess.csv", encoding="utf-8-sig")
    b = lambda s: s.astype(str).str.lower().isin(["true", "1", "1.0"])
    ev = ev[b(ev.pullback_ema20) & ev.date.between("2025-01-01", "2026-08-24") & ~ev.date.str[:7].isin(S.EXCLUDE)]
    ev["R1"] = b(ev.diverge)
    rows = []
    for sym, g in ev.groupby("symbol"):
        raw, f = V2.frame(ROOT / "data" / f"{sym}_daily.csv")
        x = M.features(f)
        ind = S.indicators(raw, f)
        high60 = f.high.astype(float).rolling(60).max()
        pos = {d: i for i, d in enumerate(f.date)}
        for d, r1 in zip(g.date, g.R1):
            t = pos.get(d)
            if t is None or S.label(raw, f, t, sym) is None:
                continue
            rows.append({"symbol": sym, "date": d, "year": d[:4], "R1": r1, "R2": x.state.iloc[t] == "GREEN_DOWN",
                         "R3": float(f.close.iloc[t]) / float(high60.iloc[t]) - 1 < -0.15,
                         **outcome(f, t, float(ind.V1_atr_pct.iloc[t]) * float(f.close.iloc[t]))})
    d = pd.DataFrame(rows).join(environment(), on="date")
    d["排除"] = d.R1 | d.R2 | d.R3
    d.to_csv(OUT / "exclusion_events.csv", index=False, encoding="utf-8-sig")
    return d


def stats(s):
    return {"样本": len(s), "达+10%": round(s.win10.mean() * 100, 1), "达+15%": round(s.win15.mean() * 100, 1),
            "止损率": round(s.stop.mean() * 100, 1), "ATR成功": round(s.atr_win.mean() * 100, 1)}


def report(d):
    pd.set_option("display.width", 250)
    rules = [("全部", d.index == d.index), ("R1 顶背离", d.R1), ("R2 绿柱放大", d.R2), ("R3 距60日高>15%", d.R3),
             ("任一命中(剔除)", d.排除), ("保留", ~d.排除)]
    t1 = pd.DataFrame([{"年份": y, "分组": name, **stats(d[m & (d.year == y)])}
                       for y in ("2025", "2026") for name, m in rules])
    for y in ("2025", "2026"):
        n = (d.year == y).sum()
        t1.loc[t1.年份 == y, "占比%"] = (t1.loc[t1.年份 == y, "样本"] / n * 100).round(1)
    t1.to_csv(OUT / "exclusion_rules.csv", index=False, encoding="utf-8-sig")
    print(t1.to_string(index=False), "\n")
    for dim in ("趋势", "宽度"):
        rec = []
        for y in ("2025", "2026"):
            for k, g in d[d.year == y].groupby(dim):
                rec.append({"年份": y, dim: k, "天数占比%": None, **{f"全部{a}": v for a, v in stats(g).items()},
                            **{f"保留{a}": v for a, v in stats(g[~g.排除]).items() if a != "样本"},
                            "保留样本": int((~g.排除).sum())})
        t = pd.DataFrame(rec).drop(columns="天数占比%")
        t.to_csv(OUT / f"exclusion_by_{dim}.csv", index=False, encoding="utf-8-sig")
        print(t.to_string(index=False), "\n")


if __name__ == "__main__":
    report(pd.read_csv(OUT / "exclusion_events.csv", encoding="utf-8-sig", dtype={"year": str})
           if len(sys.argv) > 1 and sys.argv[1] == "--report" else build())
