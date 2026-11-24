# -*- coding: utf-8 -*-
"""题材发酵 × 主升浪 考古研究 (2026-08-14, 用户指令)
Q1 主升浪(TREND episode)相对题材点火日T0的时间分布 → 谁领先谁、共振占比
Q2 题材点火后第d天进场(低位成员 vs 已启动成员)的前向收益/胜率 → 最优入场时机
题材点火 T0 = 概念内单日涨停家数≥2 且 此前10日内无(≥2)日 (冷→热第一天)
数据: data/*.csv + concept_mapping.json + results/mainwave_episodes_{2025,2026}.csv
"""
import io, sys, json, re
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path("D:/AICode/Astock")
SCRATCH = Path(__file__).parent
STOCK_RE = re.compile(r"^(sh_60|sh_68|sz_00|sz_30)\d{4}$")

# ── 1. 载入 ───────────────────────────────────────────────
mapping = json.load(open(ROOT / "concept_mapping.json", encoding="utf-8"))["mapping"]
concepts_of = {k: [c for c in v.get("concepts", []) if c] for k, v in mapping.items()}
members = {}
for sym, cs in concepts_of.items():
    for c in cs:
        members.setdefault(c, []).append(sym)
members = {c: ms for c, ms in members.items() if len(ms) >= 8}
print(f"概念数(≥8成员): {len(members)}")

cal = pd.read_csv(ROOT / "data/000001_SH_index.csv", usecols=["trade_date"])
CAL = sorted({str(x) if "-" in str(x) else f"{str(x)[:4]}-{str(x)[4:6]}-{str(x)[6:8]}"
              for x in cal["trade_date"]})
CAL = [d for d in CAL if d >= "2024-10-01"]
IDX = {d: i for i, d in enumerate(CAL)}
print(f"市场日历: {CAL[0]} ~ {CAL[-1]} 共{len(CAL)}日")

closes, pcts = {}, {}
n_loaded = 0
for p in (ROOT / "data").glob("*_daily.csv"):
    sym = p.name[:-len("_daily.csv")]
    if not STOCK_RE.match(sym):
        continue
    try:
        df = pd.read_csv(p, usecols=["date", "close", "pct_chg"])
    except Exception:
        continue
    df = df[df["date"] >= "2024-10-01"]
    if len(df) < 60:
        continue
    c = np.full(len(CAL), np.nan)
    q = np.full(len(CAL), np.nan)
    for dt, cl, pc in zip(df["date"], df["close"], df["pct_chg"]):
        i = IDX.get(str(dt))
        if i is not None:
            c[i], q[i] = cl, pc
    closes[sym], pcts[sym] = c, q
    n_loaded += 1
print(f"股票载入: {n_loaded}")

def lim_of(sym):
    return 20.0 if sym.startswith(("sh_68", "sz_30")) else 10.0

# ── 2. 概念日级热度 + 题材波(点火T0) ─────────────────────
T = len(CAL)
mkt = np.full(T, np.nan)
allp = np.vstack([pcts[s] for s in pcts])
mkt = np.nanmedian(allp, axis=0)

waves = []   # (concept, t0_idx)
lu_mat = {}
for cname, ms in members.items():
    ms2 = [s for s in ms if s in pcts]
    if len(ms2) < 8:
        continue
    lu = np.zeros(T)
    for s in ms2:
        thr = lim_of(s) * 0.985
        lu += (pcts[s] >= thr).astype(float)
    lu_mat[cname] = lu
    for t in range(12, T - 45):
        if lu[t] >= 2 and lu[t - 10:t].max() < 2:
            waves.append((cname, t))
print(f"题材点火波次: {len(waves)} (来自{len(lu_mat)}个概念)")

# ── 3. Q1: TREND episode 相对 T0 的时间分布 ───────────────
eps = pd.concat([pd.read_csv(ROOT / f"results/mainwave_episodes_{y}.csv") for y in (2025, 2026)])
eps = eps[eps["cls"] == "TREND"].copy()
eps["t_idx"] = eps["start"].map(lambda d: IDX.get(str(d)))
eps = eps.dropna(subset=["t_idx"])
eps["t_idx"] = eps["t_idx"].astype(int)
print(f"TREND episodes(在日历内): {len(eps)}")

t0_by_concept = {}
for cname, t0 in waves:
    t0_by_concept.setdefault(cname, []).append(t0)

deltas, no_theme = [], 0
for r in eps.itertuples():
    sym = r.stem
    best = None
    for cname in concepts_of.get(sym, []):
        for t0 in t0_by_concept.get(cname, []):
            d = r.t_idx - t0
            if -20 <= d <= 40 and (best is None or abs(d) < abs(best)):
                best = d
    if best is None:
        no_theme += 1
    else:
        deltas.append(best)
deltas = np.array(deltas)
print("\n== Q1 主升浪起点 vs 最近题材点火日 T0 ==")
print(f"有题材波配对: {len(deltas)} ({100*len(deltas)/len(eps):.0f}%) | 无任何题材波(±20/40日内): {no_theme} ({100*no_theme/len(eps):.0f}%)")
bins = [(-20, -6, "点火前6日以上(股先动)"), (-5, -1, "点火前1-5日(股微先动)"),
        (0, 0, "点火当日"), (1, 3, "点火后1-3日"), (4, 8, "点火后4-8日"),
        (9, 15, "点火后9-15日"), (16, 40, "点火16日后(晚期)")]
for lo, hi, lb in bins:
    m = ((deltas >= lo) & (deltas <= hi)).sum()
    print(f"  {lb:<22} {m:5d}  ({100*m/len(deltas):.1f}%)")

# ── 4. Q2: 点火后第d天进场的前向收益(低位 vs 已启动) ─────
rows = []
for cname, t0 in waves:
    ms2 = [s for s in members[cname] if s in pcts]
    for d in (0, 1, 2, 3, 5, 8, 13):
        te = t0 + d
        if te + 20 >= T:
            continue
        for s in ms2:
            c = closes[s]
            if np.isnan(c[te]) or np.isnan(c[te - 1]):
                continue
            base = c[te - 21] if te >= 21 and np.isfinite(c[te - 21]) else np.nan
            if np.isnan(base):
                continue
            prior20 = c[te] / base - 1
            f10 = c[te + 10] / c[te] - 1 if np.isfinite(c[te + 10]) else np.nan
            f20 = c[te + 20] / c[te] - 1 if np.isfinite(c[te + 20]) else np.nan
            if np.isnan(f10):
                continue
            rows.append((d, "低位滞涨" if prior20 < 0.10 else "已启动",
                         f10 * 100, (f20 * 100) if np.isfinite(f20) else np.nan))
R = pd.DataFrame(rows, columns=["stage", "pos", "f10", "f20"])
print(f"\n== Q2 点火后第d天进场 × 成员位置 (样本{len(R)}) ==")
g = R.groupby(["stage", "pos"]).agg(n=("f10", "size"), f10均=("f10", "mean"),
                                    f10中位=("f10", "median"),
                                    胜率10=("f10", lambda x: 100 * (x > 0).mean()),
                                    f20均=("f20", "mean")).round(2)
print(g.to_string())
# 基准: 全市场同持有期
mk = pd.Series(mkt).fillna(0) / 100 + 1
mret10 = (mk.rolling(10).apply(np.prod, raw=True) - 1) * 100
print(f"\n基准: 全市场中位股任意日持有10日 ≈ {mret10.mean():.2f}%")
R.to_csv(SCRATCH / "theme_entry_stats.csv", index=False)
np.save(SCRATCH / "deltas.npy", deltas)
print("done")
