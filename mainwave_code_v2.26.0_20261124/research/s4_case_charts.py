"""Build one HTML page of S4 (pullback to EMA20 then relaunch) candlestick charts: 10 winners (+15% before the -10%
stop within 20 sessions) and 10 losers (the -10% stop hit), each marking the buy, the highest point and the stop.
Prices are shown in the signal day's traded basis. Cases: 2025-01..2026-08, holdings not touching 2026-03 / 2026-07,
no names now carrying ST, at most one per month, most recent first.
python research/s4_case_charts.py -> results/research/s4_features/s4_cases.html
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
from mainwave_engine import strong_trend as ST  # noqa: E402

BASE = ROOT / "results/research/s4_features"
EXCLUDE = {"2026-03", "2026-07"}
PRE, POST = 60, 25


def path_points(f, t):
    j = t + 1
    e = float(f.open.iloc[j])
    peak_k, stop_k = j, None
    for k in range(j, j + 20):
        if f.low.iloc[k] <= e * 0.90:
            stop_k = k
            break
        if f.high.iloc[k] > f.high.iloc[peak_k]:
            peak_k = k
    return j, e, peak_k, stop_k


def case(sym, day, name, row):
    raw, f = V2.frame(ROOT / "data" / f"{sym}_daily.csv")
    t = int(f.index[f.date == day][0])
    scale = float(f.adj_factor.iloc[t]) if "adj_factor" in f else 1.0  # continuous -> signal-day traded basis
    j, e, peak_k, stop_k = path_points(f, t)
    lo, hi = max(0, t - PRE), min(len(f), t + POST + 1)
    c = f.close
    ema20, ma60 = c.ewm(span=20, adjust=False).mean(), c.rolling(60).mean()
    dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    macd = 2 * (dif - dea)
    p = lambda x: round(float(x) / scale, 3)
    bars = [[f.date.iloc[i], p(f.open.iloc[i]), p(f.close.iloc[i]), p(f.low.iloc[i]), p(f.high.iloc[i]), int(f.volume.iloc[i]) if f.volume.iloc[i] == f.volume.iloc[i] else 0] for i in range(lo, hi)]
    zero = [f.date.iloc[i] for i in range(j, min(len(f), j + 20)) if not f.volume.iloc[i] > 0]
    return {"name": name, "code": sym[3:], "signal": day, "entry_day": f.date.iloc[j], "entry": p(e), "zero_volume_days": zero,
            "peak_day": f.date.iloc[peak_k], "peak": p(f.high.iloc[peak_k]), "peak_pct": round((float(f.high.iloc[peak_k]) / e - 1) * 100, 1),
            "stop_price": p(e * 0.90), "stop_day": f.date.iloc[stop_k] if stop_k is not None else None,
            "end_day": f.date.iloc[min(j + 19, len(f) - 1)],
            "bars": bars, "ema20": [p(x) if x == x else None for x in ema20.iloc[lo:hi]],
            "ma60": [p(x) if x == x else None for x in ma60.iloc[lo:hi]],
            "dif": [p(x) for x in dif.iloc[lo:hi]], "dea": [p(x) for x in dea.iloc[lo:hi]], "macd": [p(x) for x in macd.iloc[lo:hi]],
            "ind": {"MA60斜率": f"{row.T2_ma60_slope * 100:.1f}%", "MA20高出MA60": f"{row.T1_ma20_ma60 * 100:.1f}%",
                    "20日涨幅": f"{row.M1_ret20 * 100:.1f}%", "DIF/股价": f"{row.M2_dif_pct * 100:.1f}%", "RSI14": f"{row.M4_rsi14:.0f}",
                    "ATR/股价": f"{row.V1_atr_pct * 100:.1f}%", "回踩深度": f"{row.V3_pullback_depth * 100:.1f}%",
                    "信号日量比": f"{row.Q1_vol_ratio:.2f}", "回踩缩量比": f"{row.Q2_pullback_vol:.2f}", "换手率": f"{row.Q3_turnover:.1f}%"}}


def window_clear(sym, day):
    """True when neither the signal nor any of the 20 holding sessions falls in an excluded month (holidays included)."""
    raw, f = V2.frame(ROOT / "data" / f"{sym}_daily.csv")
    t = int(f.index[f.date == day][0])
    return not any(x[:7] in EXCLUDE for x in f.date.iloc[t:t + 21])


def st_on(sym, day):
    """ST status recorded on that day (point in time), not the name today (external review 2026-09-26)."""
    raw = pd.read_csv(ROOT / "data" / f"{sym}_daily.csv", usecols=["date", "isST"])
    hit = raw[raw.date.astype(str).str[:10] == day]
    return bool(len(hit) and float(hit.isST.iloc[0] or 0) == 1)


def pick(d, names, per_month=1):
    d = d[[not st_on(r.symbol, r.date) for r in d.itertuples()]].copy()
    d["month"] = d.date.str[:7]
    d = d[~d.month.isin(EXCLUDE)].sort_values("date", ascending=False).drop_duplicates("symbol")
    d = d.groupby("month", group_keys=False).head(per_month).sort_values("date", ascending=False)
    return d[[window_clear(r.symbol, r.date) for r in d.itertuples()]]


def outcomes(card, sym, day, label):
    """Keep the price-path label apart from what a trade under the rules made (external review 2026-09-26)."""
    import accept_failure as AF
    import exit_study_s4 as E
    f, x = E.prepare(sym)
    t = int(f.index[f.date == day][0])
    after = "，之后又触及 −10%" if card["stop_day"] else ""
    path = ("先到 +15%" + after if label == "WIN15" else "先到 +10%" + after if label == "WIN10"
            else "先触及 −10%" if card["stop_day"] else "都未触及")
    fmt = lambda r: f"{r['ret'] * 100:+.1f}%（{r['reason']}，{r['days']} 日）" if r else "—"
    fail, warn = AF.s4_checks(x, t)["B2h"]
    card["ind"] = {"零成交日(持有窗口)": "、".join(card["zero_volume_days"]) or "无", "价格路径（20日）": path, "交易结果·分批止盈": fmt(AF.lad(x, t)),
                   "交易结果·分批止盈+否定即走(B2h)": fmt(AF.lad(x, t, fail, warn)), **card["ind"]}
    return card


def main():
    names = ST.names(ROOT)
    d = pd.read_csv(BASE / "s4_labelled.csv", encoding="utf-8-sig")
    wins = pick(d[d.label == "WIN15"], names).head(10)
    losers = []
    for r in pick(d[d.label == "FAIL"], names, per_month=2).itertuples():
        raw, f = V2.frame(ROOT / "data" / f"{r.symbol}_daily.csv")
        t = int(f.index[f.date == r.date][0])
        if path_points(f, t)[3] is not None:
            losers.append(r)
        if len(losers) == 10:
            break
    data = {"win": [outcomes(case(r.symbol, r.date, names.get(r.symbol, ""), r), r.symbol, r.date, r.label) for r in wins.itertuples()],
            "fail": [outcomes(case(r.symbol, r.date, names.get(r.symbol, ""), r), r.symbol, r.date, r.label) for r in losers]}
    template = (Path(__file__).resolve().parent / "s4_case_charts_template.html").read_text(encoding="utf-8")
    out = BASE / "s4_cases.html"
    out.write_text(template.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False)), encoding="utf-8")
    print(out, len(data["win"]), len(data["fail"]), round(out.stat().st_size / 1024), "KB")


if __name__ == "__main__":
    main()
