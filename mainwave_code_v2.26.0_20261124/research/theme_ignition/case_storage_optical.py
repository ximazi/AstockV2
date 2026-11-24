# -*- coding: utf-8 -*-
"""题材点火机制考古: 存储 & 光模块 2024-2026 全程时间线 (2026-08-16 用户任务)

问题: 为什么某些题材会点火爆发? 前期潜伏吸筹与题材点火的关系是什么?
方法: 对两条已走完完整周期的链, 重建四时钟时间线:
  F-钟 基本面拐点(R4季度利润, 按公告日)
  A-钟 吸筹信号(链中位 vol_ratio 缩量 + 指数在MA60上方的占比)
  L-钟 龙头脉冲(个股 +7% 日且其后25日再涨25%)
  I-钟 批量涨停点火(链内单日涨停>=3 且前10日无)
  B-钟 宽度确认(链内 >MA20 成员占比首次>70%并维持3日)
输出: research/theme_ignition/timeline_{storage,optical}.csv + 控制台时间线
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
OUT = Path("research/theme_ignition")
START = "2024-01-01"

CHAINS = {
    "storage": {"sh_603986": "兆易创新", "sz_300223": "北京君正", "sh_688008": "澜起科技",
                "sh_688525": "佰维存储", "sz_301308": "江波龙", "sz_001309": "德明利",
                "sz_300475": "香农芯创", "sz_300302": "同有科技", "sh_688123": "聚辰股份",
                "sh_688110": "东芯股份", "sh_688766": "普冉股份", "sh_688416": "恒烁股份"},
    "optical": {"sz_300308": "中际旭创", "sz_300502": "新易盛", "sz_300394": "天孚通信",
                "sz_002281": "光迅科技", "sz_300570": "太辰光", "sz_000988": "华工科技",
                "sh_688498": "源杰科技", "sh_688313": "仕佳光子", "sh_688195": "腾景科技",
                "sz_300620": "光库科技", "sh_688205": "德科立"},
}


def load(sym):
    df = pd.read_csv(ROOT / f"data/{sym}_daily.csv", usecols=["date", "close", "volume", "pct_chg"])
    return df[df["date"] >= START].reset_index(drop=True)


def lim_of(s):
    return 19.7 if s.startswith(("sh_68", "sz_30")) else 9.85


for chain, syms in CHAINS.items():
    data = {s: load(s) for s in syms}
    data = {s: d for s, d in data.items() if len(d) > 200}
    cal = sorted(set().union(*[set(d["date"]) for d in data.values()]))
    idx = {d: i for i, d in enumerate(cal)}
    T = len(cal)
    C = {}
    V = {}
    P = {}
    for s, d in data.items():
        c = np.full(T, np.nan); v = np.full(T, np.nan); p = np.full(T, np.nan)
        for dt, cl, vo, pc in zip(d["date"], d["close"], d["volume"], d["pct_chg"]):
            c[idx[dt]], v[idx[dt]], p[idx[dt]] = cl, vo, pc
        C[s], V[s], P[s] = pd.Series(c).ffill().values, v, p

    # 链等权指数
    rets = np.nanmean(np.vstack([pd.Series(P[s]).fillna(0).values for s in C]), axis=0)
    index = np.cumprod(1 + rets / 100)
    ma60 = pd.Series(index).rolling(60).mean().values

    # B-钟: >MA20 占比
    above = np.zeros(T)
    for s in C:
        ma20 = pd.Series(C[s]).rolling(20).mean().values
        above += (C[s] > ma20).astype(float)
    above /= len(C)

    # I-钟: 批量涨停
    lu = np.zeros(T)
    for s in P:
        lu += (pd.Series(P[s]).fillna(0).values >= lim_of(s)).astype(float)
    ignitions = [t for t in range(10, T) if lu[t] >= 3 and lu[t - 10:t].max() < 3]

    # L-钟: 龙头脉冲(每股可多次, 60日冷却)
    launches = []
    for s in C:
        p = pd.Series(P[s]).fillna(0).values
        c = C[s]
        last = -999
        for t in range(20, T - 25):
            if t - last >= 60 and p[t] >= 7 and np.nanmax(c[t:t + 26]) / c[t] - 1 >= 0.25:
                launches.append((cal[t], syms[s]))
                last = t
    launches.sort()

    # A-钟: 吸筹窗口(缩量 + 指数守MA60上方) — 逐月统计
    vol_ratio = np.full(T, np.nan)
    med_v = {s: pd.Series(V[s]) for s in V}
    vr_each = []
    for s in V:
        r = med_v[s].rolling(10).mean() / med_v[s].rolling(60).mean()
        vr_each.append(r.values)
    vol_ratio = np.nanmedian(np.vstack(vr_each), axis=0)

    rows = []
    for t in range(T):
        rows.append({"date": cal[t], "index": round(index[t], 3),
                     "above_ma20": round(above[t], 2), "lu": int(lu[t]),
                     "vol_ratio": round(float(vol_ratio[t]), 2) if np.isfinite(vol_ratio[t]) else np.nan,
                     "above_ma60_idx": int(index[t] > ma60[t]) if np.isfinite(ma60[t]) else 0})
    TL = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    TL.to_csv(OUT / f"timeline_{chain}.csv", index=False)

    print(f"\n===== {chain} ({len(C)}只) =====")
    print("I-钟 批量涨停点火日(>=3家,10日冷窗):", [cal[t] for t in ignitions])
    # 主周期 = 2025 年以后的首次点火
    major = next((t for t in ignitions if cal[t] >= "2025-01-01"), None)
    if major is not None:
        d0 = cal[major]
        print(f"\n-- 主周期点火日 {d0} --")
        pre = [(d, n) for d, n in launches if -180 <= (cal.index(d) - major) < 0]
        post = [(d, n) for d, n in launches if 0 <= (cal.index(d) - major) <= 40]
        print("  点火前180日内的龙头脉冲(先行者):", pre)
        print("  点火后40日内的脉冲(跟随扩散):", post)
        print("  点火前逐月画像(vr=量比 idx=指数/点火日 宽度=>MA20占比 守MA60):")
        for back in range(12, 0, -1):
            lo, hi = major - back * 21, major - (back - 1) * 21
            if lo < 0:
                continue
            seg = TL.iloc[lo:hi]
            print(f"    T-{back:2d}月: vr={seg['vol_ratio'].median():.2f} idx={seg['index'].median()/TL['index'].iloc[major]:.2f} "
                  f"宽度={seg['above_ma20'].median():.2f} 守MA60={seg['above_ma60_idx'].mean():.2f}")
        peak_t = int(np.nanargmax(index[major:min(major + 200, T)])) + major
        print(f"  点火日指数=1 → 其后峰值 {index[peak_t]/index[major]:.2f}x @ {cal[peak_t]} (点火后{peak_t-major}交易日)")
        print(f"  现在({cal[-1]}) 指数相对点火日 {index[-1]/index[major]:.2f}x, 相对峰值 {index[-1]/index[peak_t]:.2f}x")

# R4 基本面拐点(F-钟)
r4 = pd.read_csv(ROOT / "results/r4_fundamentals_panel.csv",
                 usecols=["ts_code", "ann_date", "end_date", "q_profit_yoy", "q_sales_yoy"])
r4["c6"] = r4["ts_code"].str[:6]
print("\n===== F-钟: 链中位单季利润增速(按报告期) =====")
for chain, syms in CHAINS.items():
    c6s = [s[3:] for s in syms]
    sub = r4[r4["c6"].isin(c6s) & (r4["end_date"] >= 20240331)]
    g = sub.groupby("end_date").agg(q_profit=("q_profit_yoy", "median"), q_sales=("q_sales_yoy", "median"),
                                    n=("c6", "size"), ann=("ann_date", "max")).round(1)
    print(f"-- {chain}")
    print(g.to_string())
