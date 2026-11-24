# -*- coding: utf-8 -*-
"""Publish the P1 canonical focus report and raw READY/ARMED diagnostics.

The canonical candidate report mirrors p1_closure_report.md. The unbounded
READY/ARMED pool is written separately and is never a final-pick list.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")
POOL_COLUMNS = [
    "symbol", "name", "industry", "route", "structure", "supply",
    "close", "trigger_price", "invalid_price",
]
DIAGNOSTIC_COLUMNS = [
    "symbol", "name", "industry", "theme", "pool", "route", "structure", "supply",
    "close", "trigger", "invalid", "dist_pct", "stop_pct", "sector_state",
    "sector_B", "sector_dB2", "r20", "off_hi250", "ma20_bias", "vol_ratio",
    "q_profit_yoy", "or_yoy",
]


def _read_pool(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        # Older daily runs wrote BOM/newline only when this pool had no rows.
        print(f"[candidate_report] legacy empty pool: {path.name}; rows=0")
        return pd.DataFrame(columns=POOL_COLUMNS)
    missing = sorted(set(POOL_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing columns: {', '.join(missing)}")
    return frame


def _theme_of(industry: str, themes: list) -> str:
    for t in themes:
        if any(k in industry for k in t["keywords"]):
            return t["theme"]
    return ""


def _stock_stats(sym: str, as_of: str) -> dict:
    try:
        df = pd.read_csv(ROOT / f"data/{sym}_daily.csv",
                         usecols=["date", "close", "volume", "pct_chg"])
        df["date"] = df["date"].astype(str).str[:10]
        df = df[df["date"] <= as_of].tail(260)
    except Exception:
        return {}
    c = df["close"].values
    v = df["volume"].values
    if len(c) < 60:
        return {}
    ma20 = pd.Series(c).rolling(20).mean().iloc[-1]
    return {
        "r20": (c[-1] / c[-21] - 1) * 100 if len(c) > 21 else np.nan,
        "r60": (c[-1] / c[-61] - 1) * 100 if len(c) > 61 else np.nan,
        "off_hi250": (c[-1] / np.nanmax(c) - 1) * 100,
        "ma20_bias": (c[-1] / ma20 - 1) * 100,
        "vol_ratio": float(np.nanmean(v[-10:]) / np.nanmean(v[-60:])) if np.nanmean(v[-60:]) > 0 else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)  # YYYY-MM-DD 或 YYYYMMDD
    args = ap.parse_args()
    dash = args.date if "-" in args.date else f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}"
    compact = dash.replace("-", "")
    out_dir = ROOT / f"results/mainwave_engine/{compact}"
    if not out_dir.exists():
        print(f"[candidate_report] {out_dir} 不存在, 跳过")
        return

    focus_path = out_dir / "intersection_candidates.csv"
    closure_path = out_dir / "p1_closure_report.md"
    if not focus_path.exists() or not closure_path.exists():
        raise RuntimeError("P1 closure outputs are missing; run p1_daily before candidate_report")
    focus = pd.read_csv(focus_path, encoding="utf-8-sig")

    ready = _read_pool(out_dir / "ready_table.csv")
    armed = _read_pool(out_dir / "armed_watch.csv")
    sect = pd.read_csv(out_dir / "sector_states.csv")
    gcol = "group" if "group" in sect.columns else sect.columns[0]
    sect = sect[~sect[gcol].astype(str).str.startswith("C:")].set_index(gcol)

    themes = json.load(open(ROOT / "external_theme_prior.json", encoding="utf-8"))["themes"]

    # R4 基本面(点时: ann_date<=报告日)
    r4 = pd.read_csv(ROOT / "results/r4_fundamentals_panel.csv",
                     usecols=["ts_code", "ann_date", "q_profit_yoy", "or_yoy", "netprofit_yoy"])
    r4["c6"] = r4["ts_code"].str[:6]
    r4 = r4[r4["ann_date"].astype(float) <= float(compact)]
    r4 = r4.sort_values("ann_date").groupby("c6").tail(1).set_index("c6")

    ready["pool"] = "READY"
    armed["pool"] = "ARMED"
    pools = [frame for frame in (ready, armed) if not frame.empty]
    allc = pd.concat(pools, ignore_index=True) if pools else pd.DataFrame()

    rows = []
    for _, r in allc.iterrows():
        st = _stock_stats(r["symbol"], dash)
        c6 = r["symbol"][3:]
        f = r4.loc[c6] if c6 in r4.index else None
        ind = str(r["industry"])
        srow = sect.loc[ind] if ind in sect.index else None
        rows.append({
            "symbol": r["symbol"], "name": r["name"], "industry": ind,
            "theme": _theme_of(ind, themes) or ind,
            "pool": r["pool"], "route": r["route"], "structure": r["structure"],
            "supply": r["supply"], "close": r["close"],
            "trigger": r["trigger_price"], "invalid": r["invalid_price"],
            "dist_pct": round((r["trigger_price"] / r["close"] - 1) * 100, 2),
            "stop_pct": round((1 - r["invalid_price"] / r["close"]) * 100, 2),
            "sector_state": srow["state"] if srow is not None else "?",
            "sector_B": round(float(srow["B"]), 2) if srow is not None else np.nan,
            "sector_dB2": round(float(srow["dB2"]), 3) if srow is not None else np.nan,
            "r20": round(st.get("r20", np.nan), 1), "off_hi250": round(st.get("off_hi250", np.nan), 1),
            "ma20_bias": round(st.get("ma20_bias", np.nan), 1),
            "vol_ratio": round(st.get("vol_ratio", np.nan), 2),
            "q_profit_yoy": round(float(f["q_profit_yoy"]), 1) if f is not None and pd.notna(f["q_profit_yoy"]) else np.nan,
            "or_yoy": round(float(f["or_yoy"]), 1) if f is not None and pd.notna(f["or_yoy"]) else np.nan,
        })
    D = pd.DataFrame(rows, columns=DIAGNOSTIC_COLUMNS)
    quality_path = out_dir / "carrier_qualification.csv"
    if quality_path.exists() and len(D):
        quality = pd.read_csv(quality_path, encoding="utf-8-sig")
        quality_columns = ["symbol", "carrier_qualification", "carrier_tier", "gate_failures"]
        D = D.merge(quality[[column for column in quality_columns if column in quality.columns]],
                    on="symbol", how="left")
    if "carrier_qualification" not in D:
        D["carrier_qualification"] = "DATA_INCOMPLETE"
        D["carrier_tier"] = "NOT_QUALIFIED"
        D["gate_failures"] = "QUALIFICATION_OUTPUT_MISSING"

    # 题材聚类: 同行业候选数 × 板块热度趋势
    ag = D.groupby("industry").agg(
        n=("symbol", "size"), ready=("pool", lambda x: (x == "READY").sum()),
        B=("sector_B", "first"), dB2=("sector_dB2", "first"), state=("sector_state", "first"),
        vol_ratio_med=("vol_ratio", "median"), r20_med=("r20", "median"),
    ).sort_values(["n", "B"], ascending=False)

    lines = [f"# 主升浪原始ARMED/READY诊断池（非最终候选）  {dash}", ""]
    lines += [f"- P1最终焦点: {len(focus)}只；本表仅诊断READY {len(ready)} / ARMED {len(armed)} = {len(D)}只。",
              f"- 原始池覆盖 {D['industry'].nunique()} 个行业，不受每题材3只/全市场15只的最终焦点上限约束。",
              f"- 读法: 同一行业内多只同时蓄势 = 板块级潜伏聚集, 是题材点火前的重点观察对象;",
              f"  vol_ratio<1 = 近10日较60日缩量(吸筹静默), dB2>0 = 板块热度分位在抬升。", ""]

    lines += ["## 一、题材聚类(按候选密度排序)", "",
              ag.reset_index().to_markdown(index=False), ""]

    hot = ag[(ag["n"] >= 3) & (ag["dB2"] > 0)]
    if len(hot):
        lines += ["**⚑ 聚集且板块热度抬升(潜在下一个点火题材)**: "
                  + ", ".join(f"{i}({int(r['n'])}只,B{r['B']:.2f}↑)" for i, r in hot.iterrows()), ""]

    lines += ["## 二、候选个股全表(按 题材聚类规模→距触发 排序)", ""]
    order = {ind: i for i, ind in enumerate(ag.index)}
    D["_o"] = D["industry"].map(order)
    Ds = D.sort_values(["_o", "dist_pct"]).drop(columns="_o")
    cols = ["symbol", "name", "industry", "pool", "route", "carrier_qualification",
            "carrier_tier", "structure", "supply", "close",
            "trigger", "dist_pct", "stop_pct", "sector_state", "sector_B", "r20",
            "off_hi250", "ma20_bias", "vol_ratio", "q_profit_yoy", "or_yoy", "gate_failures"]
    lines += [Ds[cols].to_markdown(index=False), ""]

    flags = D[(D["stop_pct"] > 8) | (D["sector_state"] == "RETREAT")]
    lines += ["## 三、红旗区(止损>8% 或 板块RETREAT, 触发也不宜执行)", "",
              (flags[["symbol", "name", "industry", "pool", "stop_pct", "sector_state"]]
               .to_markdown(index=False) if len(flags) else "(无)"), ""]

    lines += ["## 四、口径备注", "",
              "- 本报告只做发现层解读, 不构成交易指令; 执行纪律见方案§9.5(买前四问/止损>8%不买/ORANGE减半);",
              "- q_profit_yoy/or_yoy 为 R4 面板点时数据(公告日<=报告日); vol_ratio 用未复权 volume;",
              "- theme 列: external_theme_prior 关键词命中, 未命中回退行业名。"]

    gate_flags = D[D["carrier_qualification"] != "QUALIFIED_LURK"]
    lines += ["", "## Carrier hard-gate exclusions", "",
              (gate_flags[["symbol", "name", "pool", "carrier_qualification", "gate_failures"]]
               .to_markdown(index=False) if len(gate_flags) else "None."), ""]

    raw_path = out_dir / "candidate_pool_diagnostics.md"
    raw_path.write_text("\n".join(lines), encoding="utf-8")
    canonical_path = out_dir / "candidate_report.md"
    canonical_path.write_text(closure_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[candidate_report] final_focus={len(focus)}; raw_pool={len(D)} "
          f"-> {canonical_path} + {raw_path}")


if __name__ == "__main__":
    main()
