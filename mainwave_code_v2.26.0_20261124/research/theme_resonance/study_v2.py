# -*- coding: utf-8 -*-
"""题材发酵 × 主升浪 研究 v2 (2026-08-14, 按第7轮评审方法学修订)

修订(相对 study_v1):
  1 留一法: 判个股与题材点火的时序关系时, 点火计数排除该股自身(消内生性);
  2 事件加权: Q2 统计先按题材事件聚合(每事件一票), 再对事件做区块自助法95%CI;
  3 基准对齐: 收益一律为【同入场日、同持有期的全市场均值超额】, 均对均;
  4 D0-D15 完整逐日曲线(不再挑窗口);
  5 成员四分类(近似系统分组): 蓄势补涨近似(低位+贴近40日阻力) / 弱低位(低位无结构)
    / 中间已启动(10-25%) / 已扩张(≥25%);
  6 元数据落盘(数据范围/参数/git版本), 全部产物在本目录, 可复算。
运行: python research/theme_resonance/study_v2.py  (工作目录=仓库根)
"""
import io, sys, json, re, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
OUT = Path("research/theme_resonance")
STOCK_RE = re.compile(r"^(sh_60|sh_68|sz_00|sz_30)\d{4}$")
PARAMS = {"ignite_lu": 2, "cold_window": 10, "pair_lo": -20, "pair_hi": 40,
          "stages": list(range(0, 16)), "boot_n": 1000, "seed": 20260814}

mapping = json.load(open(ROOT / "concept_mapping.json", encoding="utf-8"))["mapping"]
concepts_of = {k: [c for c in v.get("concepts", []) if c] for k, v in mapping.items()}
members = {}
for sym, cs in concepts_of.items():
    for c in cs:
        members.setdefault(c, []).append(sym)
members = {c: ms for c, ms in members.items() if len(ms) >= 8}

cal = pd.read_csv(ROOT / "data/000001_SH_index.csv", usecols=["trade_date"])
CAL = sorted({str(x) if "-" in str(x) else f"{str(x)[:4]}-{str(x)[4:6]}-{str(x)[6:8]}"
              for x in cal["trade_date"]})
CAL = [d for d in CAL if d >= "2024-10-01"]
IDX = {d: i for i, d in enumerate(CAL)}
T = len(CAL)

closes, highs, pcts = {}, {}, {}
for p in (ROOT / "data").glob("*_daily.csv"):
    sym = p.name[:-len("_daily.csv")]
    if not STOCK_RE.match(sym):
        continue
    try:
        df = pd.read_csv(p, usecols=["date", "close", "high", "pct_chg"])
    except Exception:
        continue
    df = df[df["date"] >= "2024-10-01"]
    if len(df) < 60:
        continue
    c = np.full(T, np.nan); h = np.full(T, np.nan); q = np.full(T, np.nan)
    for dt, cl, hi, pc in zip(df["date"], df["close"], df["high"], df["pct_chg"]):
        i = IDX.get(str(dt))
        if i is not None:
            c[i], h[i], q[i] = cl, hi, pc
    closes[sym], highs[sym], pcts[sym] = c, h, q
print(f"股票 {len(closes)} | 日历 {CAL[0]}~{CAL[-1]} {T}日 | 概念 {len(members)}")

def lim_of(s): return 20.0 if s.startswith(("sh_68", "sz_30")) else 10.0

# 全市场逐日前向均值(基准, 修订3)
F10 = np.full(T, np.nan); F20 = np.full(T, np.nan)
s10 = np.zeros(T); n10 = np.zeros(T); s20 = np.zeros(T); n20 = np.zeros(T)
for s, c in closes.items():
    r10 = np.full(T, np.nan); r20 = np.full(T, np.nan)
    r10[:-10] = c[10:] / c[:-10] - 1
    r20[:-20] = c[20:] / c[:-20] - 1
    m = np.isfinite(r10); s10[m] += r10[m]; n10[m] += 1
    m = np.isfinite(r20); s20[m] += r20[m]; n20[m] += 1
F10 = np.where(n10 > 100, s10 / np.maximum(n10, 1), np.nan)
F20 = np.where(n20 > 100, s20 / np.maximum(n20, 1), np.nan)

# 概念逐日涨停名单(留一法的基础, 修订1)
lu_sets = {}
for cname, ms in members.items():
    ms2 = [s for s in ms if s in pcts]
    if len(ms2) < 8:
        continue
    day_sets = [set() for _ in range(T)]
    for s in ms2:
        thr = lim_of(s) * 0.985
        for t in np.where(pcts[s] >= thr)[0]:
            day_sets[t].add(s)
    lu_sets[cname] = day_sets

def waves_for(cname, exclude=None):
    ds = lu_sets[cname]
    lu = np.array([len(d - {exclude}) if exclude else len(d) for d in ds])
    K, W = PARAMS["ignite_lu"], PARAMS["cold_window"]
    return [t for t in range(W + 2, T - 45) if lu[t] >= K and lu[t - W:t].max() < K]

# ── Q1: 留一法配对 ────────────────────────────────────────
eps = pd.concat([pd.read_csv(ROOT / f"results/mainwave_episodes_{y}.csv") for y in (2025, 2026)])
eps = eps[eps["cls"] == "TREND"].copy()
eps["t_idx"] = eps["start"].map(lambda d: IDX.get(str(d)))
eps = eps.dropna(subset=["t_idx"]); eps["t_idx"] = eps["t_idx"].astype(int)
N_ALL = len(eps)

wave_cache = {}
deltas = []
for r in eps.itertuples():
    best = None
    for cname in concepts_of.get(r.stem, []):
        if cname not in lu_sets:
            continue
        key = (cname, r.stem)
        if key not in wave_cache:
            wave_cache[key] = waves_for(cname, exclude=r.stem)  # 留一
        for t0 in wave_cache[key]:
            d = r.t_idx - t0
            if PARAMS["pair_lo"] <= d <= PARAMS["pair_hi"] and (best is None or abs(d) < abs(best)):
                best = d
    if best is not None:
        deltas.append(best)
deltas = np.array(deltas)
paired = len(deltas)
lead = int((deltas < 0).sum())
print(f"\nQ1(留一法): TREND总数{N_ALL} | 配对{paired}({100*paired/N_ALL:.0f}%) | "
      f"配对中先于点火 {lead}({100*lead/paired:.1f}%) = 全体的 {100*lead/N_ALL:.1f}%")
q1 = pd.DataFrame({"delta": deltas})
q1.to_csv(OUT / "v2_q1_deltas.csv", index=False)
for lo, hi, lb in [(-20, -6, "前6日+"), (-5, -1, "前1-5日"), (0, 0, "当日"), (1, 3, "后1-3日"),
                   (4, 8, "后4-8日"), (9, 15, "后9-15日"), (16, 40, "后16日+")]:
    m = ((deltas >= lo) & (deltas <= hi)).sum()
    print(f"  {lb:<8} {m:5d} 配对中{100*m/paired:5.1f}% | 全体中{100*m/N_ALL:4.1f}%")

# ── Q2: 事件加权 + 超额 + 四分类 + D0-15 ─────────────────
rng = np.random.default_rng(PARAMS["seed"])
waves_full = [(c, t0) for c in lu_sets for t0 in waves_for(c)]
print(f"\n题材波(全量口径): {len(waves_full)}")

def classify(s, te):
    c = closes[s]
    base = c[te - 21] if te >= 21 and np.isfinite(c[te - 21]) else np.nan
    if np.isnan(base) or np.isnan(c[te]):
        return None
    prior = c[te] / base - 1
    hh = np.nanmax(highs[s][max(0, te - 40):te]) if te > 5 else np.nan
    near = np.isfinite(hh) and c[te] >= 0.9 * hh
    if prior < 0.10:
        return "蓄势补涨近似" if near else "弱低位无结构"
    return "已扩张" if prior >= 0.25 else "中间已启动"

ev_rows = []
for cname, t0 in waves_full:
    ms2 = [s for s in members[cname] if s in closes]
    for d in PARAMS["stages"]:
        te = t0 + d
        if te + 20 >= T or not np.isfinite(F10[te]):
            continue
        bucket = {}
        for s in ms2:
            cl = classify(s, te)
            if cl is None:
                continue
            c = closes[s]
            if not (np.isfinite(c[te]) and np.isfinite(c[te + 10])):
                continue
            ex10 = (c[te + 10] / c[te] - 1) - F10[te]
            ex20 = ((c[te + 20] / c[te] - 1) - F20[te]) if (np.isfinite(c[te + 20]) and np.isfinite(F20[te])) else np.nan
            bucket.setdefault(cl, []).append((ex10, ex20))
        for cl, vals in bucket.items():
            if len(vals) >= 3:  # 每事件每类≥3只才计一票
                a = np.array(vals, dtype=float)
                ev_rows.append((cname, CAL[t0], d, cl, len(vals),
                                float(np.nanmean(a[:, 0]) * 100), float(np.nanmean(a[:, 1]) * 100)))
EV = pd.DataFrame(ev_rows, columns=["concept", "t0", "stage", "cls", "n_members", "ex10", "ex20"])
EV.to_csv(OUT / "v2_q2_event_level.csv", index=False)

print("\nQ2(事件加权超额, 均对均; 括号=区块自助法95%CI, 单位pp):")
summary = []
for cl in ["蓄势补涨近似", "弱低位无结构", "中间已启动", "已扩张"]:
    line = [cl]
    for d in [0, 2, 5, 8, 12, 15]:
        sub = EV[(EV["cls"] == cl) & (EV["stage"] == d)]["ex10"].values
        if len(sub) < 30:
            line.append("n<30")
            continue
        boots = [np.mean(rng.choice(sub, len(sub), replace=True)) for _ in range(PARAMS["boot_n"])]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        sig = "*" if lo > 0 or hi < 0 else " "
        line.append(f"{np.mean(sub):+.2f}[{lo:+.2f},{hi:+.2f}]{sig}")
        summary.append((cl, d, len(sub), round(float(np.mean(sub)), 3), round(lo, 3), round(hi, 3)))
    print("  " + " | ".join(str(x) for x in line))
pd.DataFrame(summary, columns=["cls", "stage", "n_events", "mean_ex10", "ci_lo", "ci_hi"]).to_csv(
    OUT / "v2_q2_summary.csv", index=False)

# D0-15 曲线(蓄势补涨近似)
curve = EV[EV["cls"] == "蓄势补涨近似"].groupby("stage")["ex10"].agg(["count", "mean"]).round(3)
print("\n蓄势补涨近似 · D0-D15 逐日曲线(事件数/超额均值pp):")
print(curve.to_string())
curve.to_csv(OUT / "v2_curve_armedlike.csv")

git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
meta = {"run": "study_v2", "data_range": [CAL[0], CAL[-1]], "stocks": len(closes),
        "concepts": len(lu_sets), "waves": len(waves_full), "trend_episodes": int(N_ALL),
        "paired_loo": int(paired), "params": PARAMS, "git": git}
(OUT / "v2_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\nmeta -> {OUT}/v2_meta.json | git {git}")
