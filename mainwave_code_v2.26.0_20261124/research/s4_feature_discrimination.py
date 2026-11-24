"""S4 (pullback to EMA20 then relaunch) as the main scenario: which indicators separate the winners?

Labels (entry = next open; skip gap >= 5% / limit-up open / not traded; stop when the low touches -10%):
  WIN15  +15% reached within 20 sessions before the stop
  WIN10  +10% reached (not +15%) before the stop
  FAIL   everything else
Indicators at the signal close, 2-3 per dimension:
  TREND   T1 MA20/MA60-1, T2 MA60 20-session slope, T3 sessions since MA20 crossed above MA60
  MOMENTUM M1 20-session return, M2 DIF/close, M3 MACD histogram change vs 3 sessions earlier (/close), M4 RSI14
  VOLATILITY V1 ATR20/close, V2 ATR5/ATR20, V3 pullback depth (20-session high to lowest low of the last 10)
  VOLUME  Q1 signal-day volume / prior 20 mean, Q2 pullback volume (last 10 before signal) / the 20 before that,
          Q3 turnover
Separation: AUC of (WIN10 or WIN15) vs FAIL, and hit rate (P >= +10%) by quintile, each per year; an indicator counts
only if its direction agrees in 2025 and 2026. 2025-01..2026-08 signals, buys in 2026-03 / 2026-07 excluded.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import four_dim_v2 as V2  # noqa: E402

OUT = ROOT / "results/research/s4_features"
EXCLUDE = {"2026-03", "2026-07"}
LABELS = {"T1_ma20_ma60": "趋势 T1 MA20高出MA60", "T2_ma60_slope": "趋势 T2 MA60 20日斜率", "T3_trend_age": "趋势 T3 金叉(20上穿60)天数",
          "M1_ret20": "动量 M1 20日涨幅", "M2_dif_pct": "动量 M2 DIF/股价", "M3_hist_accel": "动量 M3 红柱3日变化", "M4_rsi14": "动量 M4 RSI14",
          "V1_atr_pct": "波动 V1 ATR20/股价", "V2_atr5_20": "波动 V2 ATR5/ATR20", "V3_pullback_depth": "波动 V3 回踩深度",
          "Q1_vol_ratio": "量能 Q1 信号日量比", "Q2_pullback_vol": "量能 Q2 回踩期缩量比", "Q3_turnover": "量能 Q3 换手率"}


def indicators(raw, f):
    c, h, l, v = (f[x].astype(float) for x in ("close", "high", "low", "volume"))
    ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
    above = (ma20 > ma60).astype(int)
    age = above.groupby((above != above.shift()).cumsum()).cumsum() * above
    dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = 2 * (dif - dif.ewm(span=9, adjust=False).mean())
    delta = c.diff()
    up, dn = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(), (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    prev = c.shift(1)
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr20, atr5 = tr.ewm(alpha=1 / 20, adjust=False).mean(), tr.ewm(alpha=1 / 5, adjust=False).mean()
    return pd.DataFrame({
        "T1_ma20_ma60": ma20 / ma60 - 1, "T2_ma60_slope": ma60 / ma60.shift(20) - 1, "T3_trend_age": age,
        "M1_ret20": c / c.shift(20) - 1, "M2_dif_pct": dif / c, "M3_hist_accel": (hist - hist.shift(3)) / c, "M4_rsi14": 100 - 100 / (1 + up / dn),
        "V1_atr_pct": atr20 / c, "V2_atr5_20": atr5 / atr20, "V3_pullback_depth": l.shift(1).rolling(10).min() / h.shift(1).rolling(20).max() - 1,
        "Q1_vol_ratio": v / v.shift(1).rolling(20).mean(), "Q2_pullback_vol": v.shift(1).rolling(10).mean() / v.shift(11).rolling(20).mean(),
        "Q3_turnover": raw.turnover.astype(float) if "turnover" in raw else np.nan})


def label(raw, f, t, symbol):
    j = t + 1
    if j + 19 >= len(f):
        return None
    lim = 0.2 if symbol[3:5] in ("30", "68") else 0.1
    r = raw.iloc[j]
    if float(r.tradestatus) != 1 or float(r.volume) <= 0 or float(r.open) >= round(float(r.preclose) * (1 + lim), 2) - 0.001 \
            or float(f.open.iloc[j]) >= float(f.close.iloc[t]) * 1.05:
        return None
    e, best = float(f.open.iloc[j]), 0.0
    for k in range(j, j + 20):
        if f.low.iloc[k] <= e * 0.90:
            break
        best = max(best, float(f.high.iloc[k]) / e - 1)
    return "WIN15" if best >= 0.15 else "WIN10" if best >= 0.10 else "FAIL"


def auc(x, y):
    """P(score of a random winner > score of a random loser), ties half."""
    x, y = np.asarray(x), np.asarray(y).astype(bool)
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    if y.sum() == 0 or (~y).sum() == 0:
        return np.nan
    r = pd.Series(x).rank().to_numpy()
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def main():
    ev = pd.read_csv(ROOT / "results/research/four_dim_v2/events_excess.csv", encoding="utf-8-sig")
    b = lambda s: s.astype(str).str.lower().isin(["true", "1", "1.0"])
    ev = ev[b(ev.pullback_ema20) & ~b(ev.diverge) & ev.date.between("2025-01-01", "2026-08-24") & ~ev.date.str[:7].isin(EXCLUDE)]
    rows = []
    for sym, g in ev.groupby("symbol"):
        raw, f = V2.frame(ROOT / "data" / f"{sym}_daily.csv")
        if "turnover" not in raw:
            full = pd.read_csv(ROOT / "data" / f"{sym}_daily.csv", usecols=["date", "turnover"])
            full["date"] = full.date.astype(str).str[:10]
            raw = raw.merge(full.drop_duplicates("date"), on="date", how="left")
        x = indicators(raw, f)
        pos = {d: i for i, d in enumerate(f.date)}
        for d in g.date:
            t = pos.get(d)
            lab = label(raw, f, t, sym) if t is not None else None
            if lab:
                rows.append({"symbol": sym, "date": d, "label": lab, **x.iloc[t].to_dict()})
    d = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "s4_labelled.csv", index=False, encoding="utf-8-sig")
    report(d)


def report(d):
    d = d.copy()
    d["year"] = d.date.str[:4]
    d["win"] = d.label.isin(["WIN10", "WIN15"])
    res = []
    for col, name in LABELS.items():
        rec = {"指标": name}
        for grp in ("WIN15", "WIN10", "FAIL"):
            rec[f"{grp}中位"] = round(float(d.loc[d.label == grp, col].median()), 4)
        for y in ("2025", "2026"):
            g = d[d.year == y]
            rec[f"AUC{y}"] = round(float(auc(g[col], g.win)), 3)
            q = pd.qcut(g[col].rank(method="first"), 5, labels=False)
            hit = g.groupby(q).win.mean()
            rec[f"{y}五分位命中率%"] = "/".join(f"{v * 100:.0f}" for v in hit)
        a25, a26 = rec["AUC2025"], rec["AUC2026"]
        rec["两年方向一致"] = bool((a25 - 0.5) * (a26 - 0.5) > 0 and min(abs(a25 - 0.5), abs(a26 - 0.5)) >= 0.02)
        res.append(rec)
    t = pd.DataFrame(res)
    counts = d.groupby(["year", "label"]).size().unstack(fill_value=0)
    t.to_csv(OUT / "discrimination.csv", index=False, encoding="utf-8-sig")
    (OUT / "counts.json").write_text(counts.to_json(force_ascii=False), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(counts.to_string())
    print(t.to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report(pd.read_csv(OUT / "s4_labelled.csv", encoding="utf-8-sig"))
    else:
        main()
