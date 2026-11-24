"""Scenario catalogue 2025-01..2026-09: index monthly returns, a new BOTTOM_REVERSAL detector (from 002967 on
2026-09-23, user's description), and every scenario we test with its conditions, 2025-2026 statistics and
typical success / failure cases for joint review. Entry = next open (not limit-up, traded); outcome = 20 sessions:
excess vs CSI1000, MFE, MAE, +10% before -7%.

BOTTOM_REVERSAL (fixed before looking at outcomes)
  big fall      120-session high to the later low >= 20% down, that low at least 20 sessions before the trigger
  slowing       min MACD DIF of the last 40 sessions > min DIF of the 40 sessions before them
  volume dries  mean volume of the last 20 sessions <= 0.7 x mean volume of sessions t-60..t-21
  range         last 20 sessions (max high - min low) / min low < 15%
  trigger       MACD golden cross today, DIF between -1.5% and +0.5% of price (near the zero axis),
                volume >= 1.3 x prior 20-session mean ("volume starts to expand"), close > MA20.
                The first draft used 1.5x and missed the labelled case (002967 on 2026-09-23: 1.49x); both are reported.
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
import four_dim_v3 as V3  # noqa: E402
from mainwave_engine import strong_trend as ST  # noqa: E402

OUT = ROOT / "results/research/scenarios_2025_2026"
START, LAST = "2025-01-01", "2026-09-23"
# User 2026-09-24: sharp market-fall months are excluded from case study and statistics (no buying in them).
# Early de-risking during such falls and bottom-fishing after them are separate, deferred research topics.
EXCLUDE_MONTHS = {"2026-03", "2026-07"}
INDICES = {"上证指数": "000001_SH", "深证成指": "399001_SZ", "创业板指": "399006_SZ", "沪深300": "000300_SH",
           "中证500": "000905_SH", "中证1000": "000852_SH", "中证2000": "399303_SZ", "科创50": "000688_SH"}


def monthly():
    table = {}
    for name, code in INDICES.items():
        f = pd.read_csv(ROOT / "data" / f"{code}_index.csv")
        f["date"] = pd.to_datetime(f.trade_date.astype(str))
        s = f.sort_values("date").set_index("date")["close"].resample("ME").last()
        table[name] = (s.pct_change() * 100).round(2)
    t = pd.DataFrame(table)
    t = t[(t.index >= "2025-01-01") & (t.index <= LAST)]
    t.index = t.index.strftime("%Y-%m")
    reg = pd.read_csv(ROOT / "results/research/regime/regime_monthly.csv", encoding="utf-8-sig", index_col=0)
    t["市场宽度(站上MA60)"] = (reg.breadth.reindex(t.index) * 100).round(0)
    t["环境标签"] = reg.regime.reindex(t.index)
    return t


def bottom_reversal(f, vol_mult=1.3, ma_rule="strict"):
    c, h, l, v = (f[x].astype(float) for x in ("close", "high", "low", "volume"))
    dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    high120 = h.rolling(120).max()
    low_since = l.rolling(100).min()
    low_age = l.rolling(100).apply(lambda x: len(x) - 1 - int(np.argmin(x)), raw=True)
    fall = low_since / high120.shift(20) - 1 <= -0.20
    slowing = dif.rolling(40).min() > dif.shift(40).rolling(40).min()
    dries = v.shift(1).rolling(20).mean() <= 0.7 * v.shift(21).rolling(40).mean()
    rng = (h.shift(1).rolling(20).max() - l.shift(1).rolling(20).min()) / l.shift(1).rolling(20).min() < 0.15
    cross = (dif > dea) & (dif.shift(1) <= dea.shift(1))
    zero = (dif / c).between(-0.015, 0.005)
    volx = v >= vol_mult * v.shift(1).rolling(20).mean()
    ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
    above = c > ma20
    # User 2026-09-24: MA20 must be above MA60. "near" is a reference variant (MA20 >= 99% of MA60 and rising).
    ma_ok = (ma20 > ma60) if ma_rule == "strict" else ((ma20 >= 0.99 * ma60) & (ma20 > ma20.shift(5)))
    sig = fall & (low_age >= 20) & slowing & dries & rng & cross & zero & volx & above & ma_ok
    return sig.fillna(False).to_numpy(), pd.DataFrame({"dif_pct": dif / c, "drawdown": low_since / high120.shift(20) - 1,
                                                        "low_age": low_age, "vol_ratio": v / v.shift(1).rolling(20).mean()})


def scan_bottom(mkt, vol_mult=1.3, ma_rule="strict"):
    rows = []
    for path in sorted((ROOT / "data").glob("s[hz]_*_daily.csv")):
        sym = path.name[:9]
        if sym[:5] not in ("sh_60", "sz_00"):
            continue
        try:
            raw, f = V2.frame(path)
        except (OSError, ValueError, KeyError):
            continue
        if len(f) < 250:
            continue
        sig, extra = bottom_reversal(f, vol_mult, ma_rule)
        window = (f.date.between(START, LAST) & (raw.isST.fillna(0) != 1)).to_numpy()
        idx = np.nonzero(sig & window)[0]
        fw = V3.forward(raw, f, sym, idx, f.date.tolist(), mkt)
        for t in idx:
            label = "MA20>MA60" if ma_rule == "strict" else "参考:MA20≥0.99MA60且上行"
            rec = {"symbol": sym, "date": f.date.iloc[t], "scenario": f"S5 底部反转({label})", **extra.iloc[t].to_dict()}
            rec.update(fw.get(t, {"filled": None}))
            rows.append(rec)
    return pd.DataFrame(rows)


def other_scenarios():
    v2 = pd.read_csv(ROOT / "results/research/four_dim_v2/events_excess.csv", encoding="utf-8-sig")
    pools = []
    for tag in ("_2021", ""):
        p = pd.read_csv(ROOT / f"results/research/case_pool/20260923{tag}/event_domain.csv", encoding="utf-8-sig",
                        usecols=["symbol", "date", "kind", "entry_type", "domain", "plan_eligible"])
        pools.append(p)
    kinds = pd.concat(pools).drop_duplicates(["symbol", "date"])
    v2 = v2.merge(kinds[["symbol", "date", "kind", "entry_type"]], on=["symbol", "date"], how="left")
    for c in ("core_event", "pullback_ema20", "diverge", "not_extended", "trend"):
        v2[c] = v2[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    v2 = v2.rename(columns={"ex20": "ex20"})
    s1 = v2[v2.core_event & (v2.entry_type == "BREAKOUT")].assign(scenario="S1 新结构突破(CT20)")
    s2 = v2[v2.core_event & (v2.entry_type == "REACCELERATION")].assign(scenario="S2 调整后再转强(RA3)")
    s4 = v2[v2.pullback_ema20 & ~v2.diverge].assign(scenario="S4 回踩EMA20再启动")
    v3 = pd.read_csv(ROOT / "results/research/four_dim_v3/events.csv", encoding="utf-8-sig")
    for c in ("shrink", "expand", "not_extended", "diverge", "leading_theme", "filled"):
        v3[c] = v3[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    s3 = v3[v3.shrink & v3.expand & v3.not_extended & ~v3.diverge].assign(scenario="S3 平台突破基础组(整理≥7天<15%+缩量后倍量)")
    return [s1, s2, s3, s4]


def stats(g):
    g = g[g.filled.astype(str).str.lower().isin(["true", "1", "1.0"])]
    if not len(g):
        return {"n": 0}
    return {"n": int(len(g)), "20日超额均值%": round(float(g.ex20.mean()) * 100, 2), "20日超额中位%": round(float(g.ex20.median()) * 100, 2),
            "跑赢大盘": round(float((g.ex20 > 0).mean()), 3), "最大浮盈中位%": round(float(g.mfe20.median()) * 100, 2),
            "最大浮亏中位%": round(float(g.mae20.median()) * 100, 2), "先+10%": round(float((g["first"] == "UP").mean()), 3),
            "先-7%": round(float((g["first"] == "DOWN").mean()), 3)}


def detail(sym, day, name):
    """Concrete time points for one case: raw prices for entry and day 20, path extremes with dates."""
    raw, f = V2.frame(ROOT / "data" / f"{sym}_daily.csv")
    t = int(f.index[f.date == day][0])
    j, k = t + 1, t + 20
    e_adj = float(f.open.iloc[j])
    path = f.iloc[j:k + 1]
    hi, lo = path.high.idxmax(), path.low.idxmin()
    idx = V3.market()
    return {"股票": name, "代码": sym, "信号日": day, "信号日收盘": round(float(raw.close.iloc[t]), 2),
            "买入日": f.date.iloc[j], "买入价(开盘)": round(float(raw.open.iloc[j]), 2),
            "第20日": f.date.iloc[k], "第20日收盘": round(float(raw.close.iloc[k]), 2),
            "20日涨幅%": round((float(f.close.iloc[k]) / e_adj - 1) * 100, 1),
            "同期中证1000%": round(idx(day, 20) * 100, 1),
            "最高日": f.date.iloc[hi], "最大浮盈%": round((float(f.high.iloc[hi]) / e_adj - 1) * 100, 1),
            "最低日": f.date.iloc[lo], "最大浮亏%": round((float(f.low.iloc[lo]) / e_adj - 1) * 100, 1)}


def examples(g, names, k=4):
    """Typical cases spread over months (at most one per month, most recent months first), holdings touching crash
    months and names
    now carrying ST excluded. Success: +10% came first and 20-session excess +10%..+30%; failure: -7% came first and
    excess -8%..-20%."""
    g = g[g.filled.astype(str).str.lower().isin(["true", "1", "1.0"])].copy()
    g = g[~g.symbol.map(lambda s: "ST" in str(names.get(s, "")).upper())]
    g["month"] = g.date.str[:7]
    # cases whose 20-session holding reaches into a crash month say more about the market than the pattern
    later = (pd.to_datetime(g.date) + pd.Timedelta(days=30)).dt.strftime("%Y-%m")
    g = g[~g.month.isin(EXCLUDE_MONTHS) & ~later.isin(EXCLUDE_MONTHS)]
    good = g[(g["first"] == "UP") & g.ex20.between(0.10, 0.30)]
    bad = g[(g["first"] == "DOWN") & g.ex20.between(-0.20, -0.08)]

    def pick(x):
        x = x.sort_values(["month", "ex20"], ascending=[False, False]).drop_duplicates("month").drop_duplicates("symbol").head(k)
        return [detail(r.symbol, r.date, names.get(r.symbol, "")) for r in x.itertuples()]
    return {"成功": pick(good), "失败": pick(bad)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = monthly()
    t.to_csv(OUT / "index_monthly.csv", encoding="utf-8-sig")
    print(t.to_string())
    mkt = V3.market()
    bottom = scan_bottom(mkt, 1.3, "strict")
    bottom.to_csv(OUT / "bottom_reversal.csv", index=False, encoding="utf-8-sig")
    bottom15 = scan_bottom(mkt, 1.3, "near")
    names = ST.names(ROOT)
    report = {}
    for g in other_scenarios() + [bottom, bottom15]:
        name = g.scenario.iloc[0]
        g = g[g.date.between(START, "2026-08-24") & ~g.date.str[:7].isin(EXCLUDE_MONTHS)]
        report[name] = {"all": stats(g), "2025": stats(g[g.date < "2026-01-01"]), "2026": stats(g[g.date >= "2026-01-01"]),
                        "cases": examples(g, names)}
        print(name, json.dumps(report[name], ensure_ascii=False))
    (OUT / "scenarios.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    hit = bottom[(bottom.symbol == "sz_002967")]
    print("002967 bottom signals:", hit[["date", "dif_pct", "drawdown", "low_age", "vol_ratio"]].to_dict("records"))


if __name__ == "__main__":
    main()
