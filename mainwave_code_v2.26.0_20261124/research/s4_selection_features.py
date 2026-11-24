"""Which S4 candidates of the same day do better? (user 2026-10-11: find the features of stocks with better results to
improve screening). Rules fixed before running; research only.

Why within the day: the account picks a few of ~10 S4 signals a day (research/account_replay.py), and luck in that
pick dominates the result. Earlier feature studies pooled all trades, where day-to-day market swings swamp any
stock-level difference. Here every feature is judged only against the other candidates of the same signal day.
Sample: S4 trades of research/accept_failure.py (2025: 3344, 2026: 1044). Outcome: the production contract v3 return
(B2h+G1); the plain LAD return is reported beside it.
Features, all known at the signal close (gap_open is known at the next open, before the buy):
  existing  T1 MA20/MA60, T2 MA60 slope, T3 trend age, M1 20-day return, M2 DIF/price, M3 histogram change, M4 RSI14,
            V1 ATR%, V2 ATR5/ATR20, V3 pullback depth, Q1 volume ratio, Q2 pullback volume, Q3 turnover,
            to_high (to the 60-day high), high_age, I5 / I20 industry strength, C_LIM / C5 concept heat,
            MF1 / MF5 / MPB main money flow
  new       RS20 / RS60 return minus CSI 1000's; hi250 distance to the 250-day high; lim30 limit-up days in 30 sessions;
            clv close location of the signal day; body signal candle body; pct signal-day change; amt20 log mean
            amount of 20 sessions; price log raw close; pb_days sessions from the 20-day high to the signal; up10 up
            days in 10; gap_open next open / signal close - 1; ind_crowd S4 signals of the same industry that day
  fundamentals (latest report announced before the signal day, results/r4_fundamentals_panel.csv): roe,
            netprofit_yoy, q_profit_yoy, or_yoy, q_sales_yoy, grossprofit_margin, debt_to_assets
Per feature and year, on days with >= 5 candidates having the feature: the Spearman rank correlation of feature and
outcome inside each day (daily IC), its mean and t = mean / sd * sqrt(days); and the excess of the top 2 and bottom 2
of the day over the day's mean outcome.
Pass (both directions allowed, the sign set by 2025): 2025 |t| >= 2; 2026 the same sign with t >= 1; top-2 excess with
the sign's direction in both years. About 40 features: ~0.3 false passes expected by chance.
If two or more pass, a composite = mean of their within-day percentile ranks (signs from 2025, equal weights, nothing
fitted) is checked the same way, and in the account replay (5 x 20%, G1 entry filter) picking by composite instead of at
random. Day level, separately: does the number of S4 signals on a day relate to that day's mean outcome (crowding)?
python research/s4_selection_features.py -> results/research/s4_selection/
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/research"
OUT = RES / "s4_selection"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exit_study_s4 as E  # noqa: E402

Y = "B2h+G1"
MIN_DAY = 5


def base_table():
    t = pd.read_csv(RES / "accept_failure/S4_trades.csv", encoding="utf-8-sig", dtype={"year": str})[["symbol", "date", "year", "LAD", Y]]
    lab = pd.read_csv(RES / "s4_features/s4_labelled.csv", encoding="utf-8-sig").drop(columns=["label", "atr_label", "year", "win"])
    macd = pd.read_csv(RES / "s4_features/macd_state_events.csv", encoding="utf-8-sig")[["symbol", "date", "to_high", "high_age"]]
    th = pd.read_csv(RES / "s4_theme_money/events.csv", encoding="utf-8-sig")[["symbol", "date", "I5", "I20", "C_LIM", "C5", "M1", "M5", "MPB"]]
    th = th.rename(columns={"M1": "MF1", "M5": "MF5"})
    return t.merge(lab, on=["symbol", "date"], how="left").merge(macd, on=["symbol", "date"], how="left").merge(th, on=["symbol", "date"], how="left")


def price_features(t):
    idx = pd.read_csv(ROOT / "data/000852_SH_index.csv")
    idx = pd.Series(idx.close.to_numpy(), index=pd.to_datetime(idx.trade_date).dt.strftime("%Y-%m-%d"))
    mp = json.loads((ROOT / "concept_mapping.json").read_text(encoding="utf-8")).get("mapping", {})
    rows = []
    for sym, g in t.groupby("symbol"):
        f, x = E.prepare(sym)
        raw = pd.read_csv(ROOT / f"data/{sym}_daily.csv", usecols=["date", "close", "amount", "pct_chg"])
        raw["date"] = raw.date.astype(str).str[:10]
        raw = raw.drop_duplicates("date").set_index("date")
        dates = list(x.date)
        pos = {d: i for i, d in enumerate(dates)}
        c, h, l, o = x.c, x.h, x.l, x.o
        lim = 0.198 if sym.startswith(("sz_30", "sh_68")) else 0.098
        for d in g.date:
            i = pos.get(d)
            if i is None or i < 250 or i + 1 >= len(x):
                continue
            ic = idx.reindex(dates[i - 60:i + 1]).to_numpy()
            r = raw.reindex(dates[i - 29:i + 1])
            hi20 = int(np.argmax(h.iloc[i - 20:i + 1].to_numpy()))
            rng = h.iloc[i] - l.iloc[i]
            rows.append({"symbol": sym, "date": d,
                         "RS20": c.iloc[i] / c.iloc[i - 20] - 1 - (ic[-1] / ic[-21] - 1),
                         "RS60": c.iloc[i] / c.iloc[i - 60] - 1 - (ic[-1] / ic[0] - 1),
                         "hi250": c.iloc[i] / h.iloc[i - 249:i + 1].max() - 1,
                         "lim30": int((r.pct_chg / 100 >= lim).sum()),
                         "clv": (c.iloc[i] - l.iloc[i]) / rng if rng > 0 else np.nan,
                         "body": (c.iloc[i] - o.iloc[i]) / c.iloc[i - 1], "pct": c.iloc[i] / c.iloc[i - 1] - 1,
                         "amt20": np.log(r.amount.iloc[-20:].mean()) if r.amount.iloc[-20:].notna().any() else np.nan,
                         "price": np.log(r.close.iloc[-1]) if pd.notna(r.close.iloc[-1]) else np.nan,
                         "pb_days": 20 - hi20, "up10": int((c.iloc[i - 9:i + 1].to_numpy() > c.iloc[i - 10:i].to_numpy()).sum()),
                         "gap_open": o.iloc[i + 1] / c.iloc[i] - 1, "industry": (mp.get(sym) or {}).get("industry") or ""})
    p = pd.DataFrame(rows)
    p["ind_crowd"] = p.groupby(["date", "industry"]).symbol.transform("count") - 1
    return p


def fundamentals(t):
    fin = pd.read_csv(RES.parent / "r4_fundamentals_panel.csv")
    fin = fin.dropna(subset=["ann_date"])
    fin["symbol"] = fin.ts_code.str[-2:].str.lower() + "_" + fin.ts_code.str[:6]
    fin["ann"] = pd.to_datetime(fin.ann_date.astype(int).astype(str))
    fin = fin.sort_values("ann")
    q = t[["symbol", "date"]].copy()
    q["d"] = pd.to_datetime(q.date) - pd.Timedelta(days=1)
    q = q.sort_values("d")
    cols = ["roe", "netprofit_yoy", "q_profit_yoy", "or_yoy", "q_sales_yoy", "grossprofit_margin", "debt_to_assets"]
    m = pd.merge_asof(q, fin[["symbol", "ann"] + cols], left_on="d", right_on="ann", by="symbol", direction="backward")
    return m[["symbol", "date"] + cols]


def daily_ic(d, col, y):
    ics, tops, bots = [], [], []
    for _, g in d[["date", col, y]].dropna().groupby("date"):
        if len(g) < MIN_DAY or g[col].nunique() < 3:
            continue
        ics.append(g[col].rank().corr(g[y].rank()))
        s = g.sort_values(col)
        tops.append(s[y].iloc[-2:].mean() - g[y].mean())
        bots.append(s[y].iloc[:2].mean() - g[y].mean())
    ics = np.array([v for v in ics if v == v])
    n = len(ics)
    t = ics.mean() / ics.std(ddof=1) * np.sqrt(n) if n > 2 and ics.std(ddof=1) > 0 else np.nan
    return {"天数": n, "IC": round(ics.mean(), 3) if n else np.nan, "t": round(t, 2),
            "前2超额%": round(np.mean(tops) * 100, 2) if tops else np.nan, "后2超额%": round(np.mean(bots) * 100, 2) if bots else np.nan}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = base_table()
    d = t.merge(price_features(t), on=["symbol", "date"], how="left").merge(fundamentals(t), on=["symbol", "date"], how="left")
    d.to_csv(OUT / "features.csv", index=False, encoding="utf-8-sig")
    feats = [c for c in d.columns if c not in ("symbol", "date", "year", "LAD", Y, "industry")]
    rows = []
    for col in feats:
        rec = {"特征": col, "覆盖%": round(d[col].notna().mean() * 100)}
        for y in ("2025", "2026"):
            s = daily_ic(d[d.year == y], col, Y)
            rec.update({f"{y} {k}": v for k, v in s.items()})
            rec[f"{y} LAD IC"] = daily_ic(d[d.year == y], col, "LAD")["IC"]
        sign = np.sign(rec["2025 IC"])
        top = "前2超额%" if sign > 0 else "后2超额%"
        ok = (abs(rec["2025 t"]) >= 2 and np.sign(rec["2026 IC"]) == sign and sign * rec["2026 t"] >= 1
              and sign * (rec[f"2025 {top}"] if sign > 0 else -rec[f"2025 {top}"]) > 0
              and sign * (rec[f"2026 {top}"] if sign > 0 else -rec[f"2026 {top}"]) > 0)
        rec["通过"] = "是（越大越好）" if ok and sign > 0 else "是（越小越好）" if ok else ""
        rows.append(rec)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "within_day_ic.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 100)
    print(r.to_string(index=False))
    # crowding: day mean outcome vs number of signals
    day = d.groupby(["year", "date"]).agg(n=(Y, "size"), mean=(Y, "mean")).reset_index()
    for y, g in day.groupby("year"):
        q = pd.qcut(g.n, 3, labels=["少", "中", "多"], duplicates="drop")
        print(y, "信号数与当日均值的秩相关", round(g.n.rank().corr(g["mean"].rank()), 3),
              {k: (int(v.n.median()), round(v["mean"].mean() * 100, 2)) for k, v in g.groupby(q, observed=True)})
    passed = r[r["通过"] != ""]
    print("通过:", passed["特征"].tolist())


if __name__ == "__main__":
    main()
