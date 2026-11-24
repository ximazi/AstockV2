# -*- coding: utf-8 -*-
"""题材点火对照研究 (P1-4, 2026-08-27): 五阶段前置条件的判别力检验

问题: "坑+缩量枯竭+龙头脉冲串 → 点火"是否只是幸存者叙事?
方法: 在全部行业(>=15成员)× 2022-07~2026-05 上, 用统一纪律采样四组条件事件,
     测每组 60 个交易日内的点火转化率:
  S0 基线: 每 60 日系统采样(无条件)
  S1 坑:   行业等权指数距 250 日高 <= -25%
  S2 坑+枯竭: S1 且 近60日内成员量比中位曾 <= 0.75
  S3 坑+枯竭+脉冲串: S2 且 60日窗内"脉冲且未回吐"成员数 >= max(3, 5%成员)
点火 = 单日涨停家数 >= max(3, 3%成员) 且前 10 日无 (冷→热)。
事件纪律: 条件当日成立且前 20 日不成立才计一次; 每事件后 60 日冷却。
输出: control_episodes.csv + 汇总; S3 未点火事件 = 失败对照组清单。
口径警告: 行业是紧链的代理(94行业口径偏宽), 结论按方向读, 数值不外推到紧链。
"""
import io
import re
import sys
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = Path(".")
OUT = Path("research/theme_ignition")
STOCK_RE = re.compile(r"^(sh_60|sh_68|sz_00|sz_30)\d{4}$")
START = "2021-01-01"
EP_LO, EP_HI = "2022-07-01", "2026-05-15"   # 事件起始范围(保证60日结果窗)
FWD = 60
CLEAN, COOL = 20, 60

cal = pd.read_csv(ROOT / "data/000001_SH_index.csv", usecols=["trade_date"])
CAL = sorted({str(x) if "-" in str(x) else f"{str(x)[:4]}-{str(x)[4:6]}-{str(x)[6:8]}"
              for x in cal["trade_date"]})
CAL = [d for d in CAL if d >= START]
T = len(CAL)

mapping = json.load(open(ROOT / "concept_mapping.json", encoding="utf-8"))["mapping"]
members = {}
for s, v in mapping.items():
    if STOCK_RE.match(s) and v.get("industry"):
        members.setdefault(v["industry"], []).append(s)
members = {k: v for k, v in members.items() if len(v) >= 15}
print(f"行业(>=15成员): {len(members)} | 日历 {CAL[0]}~{CAL[-1]} {T}日")

print("载入个股数据...")
DATA = {}
for ind, syms in members.items():
    for s in syms:
        if s in DATA:
            continue
        try:
            df = pd.read_csv(ROOT / f"data/{s}_daily.csv", usecols=["date", "close", "volume", "pct_chg"])
        except Exception:
            continue
        df = df[df["date"] >= START].set_index("date").reindex(CAL)
        if df["close"].notna().sum() < 250:
            continue
        DATA[s] = (df["close"].values, df["volume"].values, df["pct_chg"].values)
print(f"载入 {len(DATA)} 只")


def lim_of(s):
    return 19.7 if s.startswith(("sh_68", "sz_30")) else 9.85


ep_rows = []
for ind, syms in members.items():
    syms = [s for s in syms if s in DATA]
    n = len(syms)
    if n < 15:
        continue
    P = np.vstack([DATA[s][2] for s in syms])          # pct
    C = np.vstack([pd.Series(DATA[s][0]).ffill().values for s in syms])
    V = np.vstack([DATA[s][1] for s in syms])
    idx = np.cumprod(1 + np.nan_to_num(np.nanmean(np.where(np.isfinite(P), P, np.nan), axis=0)) / 100)
    hi250 = pd.Series(idx).rolling(250, min_periods=100).max().values
    pit = idx / hi250 - 1 <= -0.25

    vr = np.full((n, T), np.nan)
    for i in range(n):
        v = pd.Series(V[i])
        vr[i] = (v.rolling(10).mean() / v.rolling(60).mean()).values
    med_vr = np.nanmedian(vr, axis=0)
    exhaust = pd.Series(med_vr).rolling(60, min_periods=20).min().values <= 0.75

    n_pulse = np.zeros(T)
    for i in range(n):
        pulse = P[i] >= 7.0
        arr = np.where(pulse, C[i], np.nan)
        rmin = pd.Series(arr).rolling(60, min_periods=1).min().values
        n_pulse += (np.isfinite(rmin) & (C[i] >= rmin)).astype(float)
    k_pulse = max(3, math.ceil(0.05 * n))
    train = n_pulse >= k_pulse

    lim = np.array([lim_of(s) for s in syms])[:, None]
    lu = np.nansum(P >= lim, axis=0)
    k_ign = max(3, math.ceil(0.03 * n))
    ign = np.zeros(T, bool)
    for t in range(10, T):
        if lu[t] >= k_ign and lu[t - 10:t].max() < k_ign:
            ign[t] = True
    ign_fwd = np.array([ign[t + 1:t + 1 + FWD].any() if t + 1 + FWD <= T else False
                        for t in range(T)])

    conds = {"S0": np.ones(T, bool), "S1": pit, "S2": pit & exhaust, "S3": pit & exhaust & train}
    lo = next((i for i, d in enumerate(CAL) if d >= EP_LO), T)
    hi = next((i for i, d in enumerate(CAL) if d > EP_HI), T)
    for sk, cond in conds.items():
        last_end = -1
        t = max(lo, 310)
        while t < min(hi, T - FWD - 1):
            if sk == "S0":
                take = t >= last_end
            else:
                take = cond[t] and not cond[max(0, t - CLEAN):t].any() and t >= last_end
            if take:
                fwd_ret = idx[min(t + FWD, T - 1)] / idx[t] - 1
                # 首次点火等待天数
                w = np.where(ign[t + 1:t + 1 + FWD])[0]
                ep_rows.append({"industry": ind, "n_members": n, "set": sk, "date": CAL[t],
                                "ignited": bool(ign_fwd[t]), "days_to_ign": int(w[0] + 1) if len(w) else "",
                                "fwd60_idx_ret": round(float(fwd_ret) * 100, 1),
                                "n_pulse": int(n_pulse[t]), "med_vr60min": round(float(pd.Series(med_vr).rolling(60, min_periods=20).min().values[t]), 2)})
                last_end = t + COOL
                t += COOL if sk == "S0" else 1
                continue
            t += 1

EP = pd.DataFrame(ep_rows)
OUT.mkdir(parents=True, exist_ok=True)
EP.to_csv(OUT / "control_episodes.csv", index=False, encoding="utf-8-sig")

print("\n== 60日内点火转化率(事件纪律统一) ==")
g = EP.groupby("set").agg(n_ep=("ignited", "size"), 转化率=("ignited", "mean"),
                          fwd60均=("fwd60_idx_ret", "mean"))
g["转化率"] = (g["转化率"] * 100).round(1)
g["fwd60均"] = g["fwd60均"].round(1)
print(g.to_string())
s3 = EP[EP["set"] == "S3"]
if len(s3):
    conv = s3[s3["ignited"]]
    fail = s3[~s3["ignited"]]
    print(f"\nS3 事件 {len(s3)}: 点火 {len(conv)}, 未点火 {len(fail)}")
    if len(conv):
        print(f"  点火者等待天数: 中位 {conv['days_to_ign'].median():.0f} 日 | fwd60指数 {conv['fwd60_idx_ret'].mean():+.1f}%")
    if len(fail):
        print(f"  失败组 fwd60指数 {fail['fwd60_idx_ret'].mean():+.1f}% | 清单(近期优先):")
        for _, r in fail.sort_values("date", ascending=False).head(15).iterrows():
            print(f"    {r['date']} {r['industry']}(n={r['n_members']}) 脉冲{r['n_pulse']} fwd60 {r['fwd60_idx_ret']:+.1f}%")
print("\n-> research/theme_ignition/control_episodes.csv")
