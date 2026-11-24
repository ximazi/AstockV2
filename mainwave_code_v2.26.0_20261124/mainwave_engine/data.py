# -*- coding: utf-8 -*-
"""数据访问层: 股票域 / 日线读取 / 行业映射 / 基本面快照 (方案 §3 约定)."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
STOCK_RE = re.compile(r"^(sh_60|sh_68|sz_00|sz_30)\d{4}$")

COLS = ["date", "open", "high", "low", "close", "preclose", "volume", "amount", "turnover", "pct_chg"]


def limit_pct(symbol: str) -> float:
    """涨跌停幅度: 科创(sh_68)/创业板(sz_30)=20%, 主板=10%。"""
    return 20.0 if symbol.startswith(("sh_68", "sz_30")) else 10.0


@lru_cache(maxsize=1)
def industry_map() -> dict[str, str]:
    m = json.load(open("concept_mapping.json", encoding="utf-8"))["mapping"]
    return {k: v.get("industry", "?") for k, v in m.items() if v.get("industry", "?") != "?"}


@lru_cache(maxsize=8)
def concepts_map(as_of: str = "") -> dict[str, list[str]]:
    """个股→概念题材列表(跨行业主线层, v1.1 P0-3)。"""
    m = json.load(open("concept_mapping.json", encoding="utf-8"))["mapping"]
    if as_of:
        from .theme_semantics import apply_membership_evidence
        m, _ = apply_membership_evidence(m, as_of)
    return {k: [c for c in v.get("concepts", []) if c] for k, v in m.items()}


@lru_cache(maxsize=1)
def name_map() -> dict[str, str]:
    m = json.load(open("concept_mapping.json", encoding="utf-8"))["mapping"]
    return {k: v.get("name", "") for k, v in m.items()}


@lru_cache(maxsize=1)
def st_set() -> frozenset[str]:
    """ST/退市排除: 用名称 SSOT(concept_mapping), 不信逐行 isST(7月实证该旗标失真)。"""
    bad = {s for s, n in name_map().items() if ("ST" in n.upper()) or ("退" in n)}
    return frozenset(bad)


def list_universe() -> list[str]:
    # 退市股会被 concept_mapping 整条移除, 名称查不到 => 名称型 ST 过滤看不见它们,
    # 而本地 CSV 仍在(僵尸数据, 末日停在退市前)。故要求"映射中有名字"才算在册。
    live = name_map()
    out = []
    for p in DATA_DIR.glob("*_daily.csv"):
        sym = p.name[: -len("_daily.csv")]
        if STOCK_RE.match(sym) and sym in live and sym not in st_set():
            out.append(sym)
    return sorted(out)


def load_tail(symbol: str, rows: int = 320, as_of: str | None = None) -> pd.DataFrame | None:
    """读取尾部 rows 根日线(as_of 切片=回放/考古的前提, 防未来函数)。None=不可用。"""
    p = DATA_DIR / f"{symbol}_daily.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, usecols=lambda c: c in COLS)
    except Exception:
        return None
    if len(df) < 70 or "close" not in df.columns:
        return None
    if as_of:
        df = df[df["date"].astype(str) <= as_of]
        if len(df) < 70:
            return None
    df = df.tail(rows).reset_index(drop=True)
    for c in ("open", "high", "low", "close", "preclose", "amount", "turnover", "pct_chg"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if df["close"].isna().any() or (df["close"] <= 0).any():
        return None
    return df


R4_PANEL = Path("results/r4_fundamentals_panel.csv")


@lru_cache(maxsize=1)
def r4_panel() -> pd.DataFrame:
    """R4 季度业绩面板(2019Q1~, 含 ann_date 可点时取数)。空表=不可用。"""
    if not R4_PANEL.exists():
        return pd.DataFrame()
    df = pd.read_csv(R4_PANEL, usecols=["ts_code", "ann_date", "end_date",
                                        "q_sales_yoy", "q_profit_yoy", "or_yoy", "netprofit_yoy"])
    df["code6"] = df["ts_code"].astype(str).str[:6]
    df["ann_date"] = pd.to_numeric(df["ann_date"], errors="coerce")
    for c in ("q_sales_yoy", "q_profit_yoy", "or_yoy", "netprofit_yoy"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["code6", "end_date"])


def fundamental_view(c6: str, as_of: str) -> dict | None:
    """as_of 时点已披露的最近两季(点时: ann_date<=as_of, 防未来函数)。
    返回 {"latest": row, "prev": row|None} 或 None(无数据)。"""
    panel = r4_panel()
    if panel.empty or not c6:
        return None
    cutoff = float(as_of.replace("-", ""))
    d = panel[(panel["code6"] == c6) & (panel["ann_date"] <= cutoff)]
    if d.empty:
        return None
    d = d.sort_values("end_date")
    latest = d.iloc[-1].to_dict()
    prev = d.iloc[-2].to_dict() if len(d) >= 2 else None
    return {"latest": latest, "prev": prev}


def code6(symbol: str) -> str:
    m = re.search(r"(\d{6})", symbol)
    return m.group(1) if m else ""


# ── 序列工具 ─────────────────────────────────────────────
def atr(df: pd.DataFrame, period: int) -> np.ndarray:
    h, l, pc = df["high"].values, df["low"].values, df["preclose"].values
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(period).mean().values


def close_location(df: pd.DataFrame) -> np.ndarray:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    rng = np.where(h - l <= 0, np.nan, h - l)
    return np.clip((c - l) / rng, 0, 1)
