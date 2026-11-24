"""S4 ranking with information beyond the daily chart: sector / theme strength and main-force money flow.
Directions fixed by the strategy logic before looking (theme = amplifier, money = base; higher = better):
  I5 / I20  the stock's industry 5 / 20-session median-return rank among industries that day (0..1)
  C_LIM     limit-up count that day in the stock's strongest concept (concept with the most limit-ups, >= 10 members)
  C5        best 5-session mean-return rank among the stock's concepts (>= 10 members)
  M1        main-force net inflow on the signal day / that day's total traded amount (lg + elg orders, Tushare moneyflow)
  M5        main-force net inflow over the last 5 sessions / their amount
  MPB       main-force net inflow over the 10 sessions before the signal (the pullback) / their amount
Memberships: concept_mapping.json as of today (a mild look-ahead for concepts; industries rarely change).
Sample: S4 kept by R1-R3 (exclusion_events.csv). Labels: +10% before -10% within 20 sessions (win10), volatility label
(+2.5 ATR first, atr_win), and the LAD exit return (exit_study_s4.py). AUC and quintile means per year.
python research/s4_theme_money.py -> results/research/s4_theme_money/
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import crash_bottom as B  # noqa: E402
from s4_feature_discrimination import auc  # noqa: E402

OUT = ROOT / "results/research/s4_theme_money"
MF = ROOT / "results/research/moneyflow"
FEATS = {"I5": "行业5日强度分位", "I20": "行业20日强度分位", "C_LIM": "最强概念当日涨停数", "C5": "概念5日强度分位",
         "M1": "信号日主力净流入/成交额", "M5": "5日主力净流入/成交额", "MPB": "回踩10日主力净流入/成交额"}


def moneyflow(dates):
    from dotenv import load_dotenv
    import tushare as ts
    load_dotenv(ROOT / ".env")
    pro = ts.pro_api(os.getenv("TUSHARE_TOKEN"))
    MF.mkdir(parents=True, exist_ok=True)
    frames = []
    for d in dates:
        path = MF / f"{d}.csv"
        if not path.exists():
            for attempt in range(3):
                try:
                    df = pro.moneyflow(trade_date=d.replace("-", ""))
                    break
                except Exception:  # rate limit / network: wait and retry
                    time.sleep(20)
            else:
                continue
            df.to_csv(path, index=False)
        df = pd.read_csv(path)
        if df.empty:
            continue
        amt = df[[c for c in df.columns if c.startswith(("buy_", "sell_")) and c.endswith("_amount")]].sum(axis=1) / 2
        net = df.buy_lg_amount + df.buy_elg_amount - df.sell_lg_amount - df.sell_elg_amount
        sym = df.ts_code.str[-2:].str.lower() + "_" + df.ts_code.str[:6]
        frames.append(pd.DataFrame({"date": d, "symbol": sym, "net": net, "amt": amt}))
    m = pd.concat(frames)
    return m.pivot(index="date", columns="symbol", values="net"), m.pivot(index="date", columns="symbol", values="amt")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = pd.read_csv(ROOT / "results/research/s4_features/exclusion_events.csv", encoding="utf-8-sig", dtype={"year": str})
    ev = ev[~ev["排除"].astype(bool)].copy()
    lad = pd.read_csv(ROOT / "results/research/exit_s4/trades.csv", encoding="utf-8-sig")
    ev = ev.merge(lad[lad.rule == "LAD"][["symbol", "date", "ret"]].rename(columns={"ret": "lad_ret"}), on=["symbol", "date"], how="left")

    panel = pd.read_csv(ROOT / "results/research/crash/market_panel.csv", encoding="utf-8-sig", index_col=0)
    px = B.stock_panel(panel.index)
    ret = px / px.shift(1) - 1
    mp = json.loads((ROOT / "concept_mapping.json").read_text(encoding="utf-8"))["mapping"]
    ind = {s: v.get("industry") for s, v in mp.items() if v.get("industry")}
    concepts = {s: v.get("concepts") or [] for s, v in mp.items()}
    # industry strength
    cols = [s for s in px.columns if s in ind]
    by_ind = {}
    for s in cols:
        by_ind.setdefault(ind[s], []).append(s)
    ind_ret = pd.DataFrame({k: ret[v].median(axis=1) for k, v in by_ind.items() if len(v) >= 5})
    lr = np.log1p(ind_ret)
    ind5, ind20 = lr.rolling(5).sum().rank(axis=1, pct=True), lr.rolling(20).sum().rank(axis=1, pct=True)
    # concept strength
    by_con = {}
    for s, cs in concepts.items():
        if s in px.columns:
            for c in cs:
                by_con.setdefault(c, []).append(s)
    by_con = {k: v for k, v in by_con.items() if len(v) >= 10}
    lim = {k: (ret[v] >= 0.095).sum(axis=1) for k, v in by_con.items()}
    con5 = np.log1p(pd.DataFrame({k: ret[v].mean(axis=1) for k, v in by_con.items()})).rolling(5).sum().rank(axis=1, pct=True)
    lim = pd.DataFrame(lim)

    days = sorted(ev.date.unique())
    need = sorted({d for d in panel.index if "2024-12-01" <= d <= days[-1]})
    net, amt = moneyflow(need)
    net5, amt5 = net.rolling(5).sum(), amt.rolling(5).sum()
    netpb, amtpb = net.shift(1).rolling(10).sum(), amt.shift(1).rolling(10).sum()

    def val(frame, d, col):
        try:
            return float(frame.at[d, col])
        except KeyError:
            return np.nan

    rows = []
    for r in ev.itertuples():
        cs = [c for c in concepts.get(r.symbol, []) if c in by_con]
        rows.append({"I5": val(ind5, r.date, ind.get(r.symbol)), "I20": val(ind20, r.date, ind.get(r.symbol)),
                     "C_LIM": max([val(lim, r.date, c) for c in cs], default=np.nan),
                     "C5": max([val(con5, r.date, c) for c in cs], default=np.nan),
                     "M1": val(net, r.date, r.symbol) / val(amt, r.date, r.symbol),
                     "M5": val(net5, r.date, r.symbol) / val(amt5, r.date, r.symbol),
                     "MPB": val(netpb, r.date, r.symbol) / val(amtpb, r.date, r.symbol)})
    d = pd.concat([ev.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    d.to_csv(OUT / "events.csv", index=False, encoding="utf-8-sig")
    report(d)


def report(d):
    out = []
    for col, name in FEATS.items():
        rec = {"指标": f"{col} {name}", "覆盖%": round(d[col].notna().mean() * 100)}
        for y in ("2025", "2026"):
            g = d[(d.year == y) & d[col].notna()]
            rec[f"AUC达10 {y}"] = round(float(auc(g[col], g.win10)), 3)
            rec[f"AUC ATR {y}"] = round(float(auc(g[col], g.atr_win)), 3)
            q = pd.qcut(g[col].rank(method="first"), 5, labels=False)
            rec[f"LAD收益五分位 {y}"] = "/".join(f"{v * 100:.1f}" for v in g.groupby(q).lad_ret.mean())
            rec[f"达10五分位 {y}"] = "/".join(f"{v * 100:.0f}" for v in g.groupby(q).win10.mean())
        out.append(rec)
    t = pd.DataFrame(out)
    t.to_csv(OUT / "discrimination.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_colwidth", 60)
    print(t.to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report(pd.read_csv(OUT / "events.csv", encoding="utf-8-sig", dtype={"year": str}))
    else:
        main()
