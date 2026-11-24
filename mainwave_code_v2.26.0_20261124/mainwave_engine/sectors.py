# -*- coding: utf-8 -*-
"""L1 板块五态 + 龙头点火 (方案 §3特征6/7 + T4补全; 2026-08-13 按审核修订).

修订: 全市场中位数用原始个股面板(非行业中位的中位) / MA20 在完整序列上先算再截窗 /
     120日分位需满120有效观测才授予 / 新增 leader_ignitions(梯队传导输入, 只用前一日RS判龙头)。
已知限制(留档): 行业归属用当前 concept_mapping 回套历史(无点时行业库), 板块广度含此近似。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WINDOW = 140  # 120 分位窗 + 状态判定余量


def build_breadth(per_stock: dict[str, pd.DataFrame], group_map: dict[str, list[str]],
                  lim_of: dict[str, float]) -> tuple[pd.DataFrame, dict]:
    """聚合 分组×日期 广度面板 + 全市场每日中位收益。
    v1.1: 分组=行业+概念题材(概念名带 "C:" 前缀), 一套机制双层运行(P0-3)。
    per_stock 含当日停牌股(其历史仍参与广度, 减存活偏差)。"""
    rows = []
    mrows = []
    for sym, df in per_stock.items():
        groups = group_map.get(sym)
        if not groups:
            continue
        ma20_full = df["close"].rolling(20).mean()  # 完整序列先算, 防截断假0 (审核P2)
        d = df.tail(WINDOW)
        ma20 = ma20_full.tail(WINDOW)
        lim = lim_of[sym]
        limit_up = d["pct_chg"] >= lim * 0.985
        touch = d["high"] >= d["preclose"] * (1 + lim / 100) * 0.998
        for dt, pct, c, m20, lu, tc in zip(d["date"], d["pct_chg"], d["close"], ma20, limit_up, touch):
            above = 1.0 if (pd.notna(m20) and c > m20) else 0.0
            mrows.append((str(dt), pct))  # 市场中位: 每股每日只计一次(概念复制不得重复计票)
            for g in groups:
                rows.append((str(dt), g, pct, above, 1.0 if lu else 0.0, 1.0 if tc else 0.0))
    raw = pd.DataFrame(rows, columns=["date", "industry", "pct", "above_ma20", "lu", "touch"])
    med = (pd.DataFrame(mrows, columns=["date", "pct"])
           .groupby("date")["pct"].median().rename("mkt_med"))  # 全市场真中位 (审核P2)
    mkt_med = med.to_dict()
    raw = raw.merge(med, on="date")
    raw["beat"] = (raw["pct"] > raw["mkt_med"]).astype(float)
    g = raw.groupby(["date", "industry"]).agg(
        n=("pct", "size"), beat=("beat", "mean"), ma20b=("above_ma20", "mean"),
        lub=("lu", "mean"), lu_n=("lu", "sum"), touch_n=("touch", "sum"),
        ret=("pct", "median")).reset_index()
    g = g[g["n"] >= 8]
    g = g.merge(med, on="date")
    return g, mkt_med


def _pct_rank_120(s: pd.Series) -> pd.Series:
    # 审核修订: 必须满 120 有效观测才产生分位 (min_periods=120)
    return s.rolling(120, min_periods=120).apply(lambda x: float(np.mean(x <= x.iloc[-1])), raw=False)


def sector_states(panel: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["industry", "state", "B", "dB2", "exc5", "heat", "n"])
    if panel.empty:
        return empty
    out = []
    for ind, d in panel.groupby("industry"):
        d = d.sort_values("date").reset_index(drop=True)
        if len(d) < 123:  # 120分位 + B[t-2]
            continue
        B = (_pct_rank_120(d["beat"]) + _pct_rank_120(d["ma20b"]) + _pct_rank_120(d["lub"])) / 3
        exc = d["ret"] - d["mkt_med"]
        exc3 = exc.rolling(3).sum()
        exc5 = exc.rolling(5).sum()
        heat = _pct_rank_120(exc5)
        seal = np.where(d["touch_n"] > 0, d["lu_n"] / d["touch_n"], 1.0)
        brk_q = _pct_rank_120(pd.Series(1 - seal))
        i = len(d) - 1
        b0, b1, b2 = B.iloc[i], B.iloc[i - 1], B.iloc[i - 2]
        if pd.isna(b0) or pd.isna(b2) or pd.isna(heat.iloc[i]):
            continue  # 分位未满120观测, 不判状态
        emerging = (b2 <= 0.50 and b2 < b1 < b0 and (b0 - b2) >= 0.15
                    and exc3.iloc[i] > 0 and b0 < 0.85 and heat.iloc[i] < 0.90)
        crowded = ((b0 >= 0.85 or heat.iloc[i] >= 0.90)
                   and sum([bool(brk_q.iloc[i] >= 0.80), d["ret"].iloc[i] <= 0, b0 < b1]) >= 2)
        retreat = ((b0 - b2) <= -0.15 and exc3.iloc[i] < 0
                   and (d["ret"].iloc[i] < 0 or brk_q.iloc[i] >= 0.70))
        active = (0.55 <= b0 < 0.85 and exc5.iloc[i] > 0
                  and (b0 - b2) >= -0.05 and brk_q.iloc[i] < 0.80)
        state = ("CROWDED" if crowded else "RETREAT" if retreat else
                 "EMERGING" if emerging else "ACTIVE" if active else "NEUTRAL")
        out.append({"industry": ind, "state": state, "B": round(float(b0), 3),
                    "dB2": round(float(b0 - b2), 3), "exc5": round(float(exc5.iloc[i]), 2),
                    "heat": round(float(heat.iloc[i]), 2), "n": int(d["n"].iloc[i])})
    return pd.DataFrame(out).sort_values("B", ascending=False) if out else empty


def leader_ignitions(per_stock: dict[str, pd.DataFrame], group_map: dict[str, list[str]],
                     lim_of: dict[str, float], run_date: str) -> dict[str, dict]:
    """梯队传导输入(特征7): 龙头=【前一日】20日RS top3(只用前日信息, 防用当日涨幅倒推龙头);
    点火=龙头近3日强冲击。复审#8: 按分组(行业+概念)构建, AI/存储/机器人等跨行业题材可传导。
    返回 {group: {date, leader}}。"""
    by_g: dict[str, list] = {}
    for sym, df in per_stock.items():
        if str(df["date"].iloc[-1]) != run_date:
            continue
        groups = group_map.get(sym)
        if not groups or len(df) < 25:
            continue
        c = df["close"].values
        pct = df["pct_chg"].values
        dates = [str(x) for x in df["date"]]
        r20_prev = c[-2] / c[-22] - 1
        lim = lim_of[sym]
        thr = (0.60 if lim >= 20 else 0.70) * lim
        imps = [dates[-k] for k in (1, 2, 3) if len(pct) >= k and pct[-k] >= thr]
        for g in groups:
            by_g.setdefault(g, []).append((sym, r20_prev, imps))
    out: dict[str, dict] = {}
    for g, rows in by_g.items():
        if len(rows) < 8:
            continue
        rows.sort(key=lambda x: -x[1])
        best = None
        for sym, _, imps in rows[:3]:
            for d in imps:
                if best is None or d > best["date"]:
                    best = {"date": d, "leader": sym}
        if best:
            out[g] = best
    return out
