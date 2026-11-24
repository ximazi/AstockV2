# -*- coding: utf-8 -*-
"""Unconditional historical calibration for candidate potential ranges."""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from mainwave_engine import phase_events as PE
from strategy_tracker_v2.universe import universe_rejection_reason


VERSION = "mainwave_value_calibration_v2_20260830"
OUT = Path("results/mainwave_value_calibration")
MIN_AMOUNT = 5e7
START = pd.Timestamp("2022-01-01")
TRAIN_END = pd.Timestamp("2025-12-31")
OOS_START = pd.Timestamp("2026-01-01")
OOS_END = pd.Timestamp("2026-06-30")


def _forward(values: pd.Series, sessions: int, kind: str) -> pd.Series:
    shifted = values.shift(-1).iloc[::-1]
    roller = shifted.rolling(sessions, min_periods=sessions)
    result = roller.max() if kind == "max" else roller.min()
    return result.iloc[::-1]


def context_key(phase: str, event: str, effort: str,
                range_position60: float, ret20: float) -> str:
    if event in {"LPS", "UTAD"}:
        return f"EVENT_{event}"
    if effort in {"DEMAND_EXPANSION", "DISTRIBUTION_PRESSURE"}:
        return f"EFFORT_{effort}"
    if phase in {"MARKUP", "CLIMAX", "DISTRIBUTION"}:
        return f"PHASE_{phase}"
    position = "LOW" if range_position60 < 0.35 else (
        "MID" if range_position60 < 0.75 else "HIGH")
    momentum = "WEAK" if ret20 < 0 else (
        "FLAT" if ret20 < 0.05 else ("BUILDING" if ret20 < 0.15 else "EXTENDED"))
    return f"TECH_{position}_{momentum}"


def _stock_rows(path: str) -> list[dict]:
    symbol = os.path.basename(path).replace("_daily.csv", "")
    if universe_rejection_reason(symbol):
        return []
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, pd.errors.EmptyDataError):
        return []
    frame = PE.classify_frame(raw)
    if frame.empty:
        return []
    close = frame["return_index"]
    for sessions in (20, 40):
        frame[f"max{sessions}"] = _forward(close, sessions, "max") / close - 1
        frame[f"min{sessions}"] = _forward(close, sessions, "min") / close - 1
        frame[f"ret{sessions}_forward"] = close.shift(-sessions) / close - 1
    frame["recent_LPS"] = frame["LPS"].rolling(10, min_periods=1).max().astype(bool)
    frame["recent_UTAD"] = frame["UTAD"].rolling(10, min_periods=1).max().astype(bool)
    eligible = (
        (frame["date"] >= START) & (frame["date"] <= OOS_END)
        & (frame["amount"] >= MIN_AMOUNT) & frame["max40"].notna()
    )
    event_day = frame["recent_LPS"] | frame["recent_UTAD"] \
        | frame["effort_result"].isin(["DEMAND_EXPANSION", "DISTRIBUTION_PRESSURE"])
    cadence = pd.Series(np.arange(len(frame)) % 10 == 0, index=frame.index)
    selected = frame[eligible & (event_day | cadence)].copy()
    rows = []
    for row in selected.to_dict("records"):
        event = "UTAD" if row["recent_UTAD"] else ("LPS" if row["recent_LPS"] else "")
        key = context_key(
            str(row["broad_phase"]), event, str(row["effort_result"]),
            float(row.get("range_position60") or 0.5), float(row.get("ret20") or 0.0),
        )
        rows.append({
            "symbol": symbol, "date": str(pd.Timestamp(row["date"]).date()),
            "sample": "TRAIN" if row["date"] <= TRAIN_END else "OOS_2026H1",
            "context_key": key, "phase": row["broad_phase"], "event": event,
            "effort": row["effort_result"], "ret20": row["ret20"],
            "range_position60": row["range_position60"],
            **{name: row[name] for name in [
                "max20", "min20", "ret20_forward", "max40", "min40", "ret40_forward"]},
        })
    return rows


def _stats(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby("context_key"):
        row = {"context_key": key, "n": len(group)}
        for sessions in (20, 40):
            row.update({
                f"mfe{sessions}_p25": group[f"max{sessions}"].quantile(0.25),
                f"mfe{sessions}_median": group[f"max{sessions}"].median(),
                f"mfe{sessions}_p75": group[f"max{sessions}"].quantile(0.75),
                f"mae{sessions}_p25": group[f"min{sessions}"].quantile(0.25),
                f"close_ret{sessions}_median": group[f"ret{sessions}_forward"].median(),
                f"touch15_{sessions}": group[f"max{sessions}"].ge(0.15).mean(),
                f"touch25_{sessions}": group[f"max{sessions}"].ge(0.25).mean(),
            })
        rows.append(row)
    baseline = {"context_key": "BASELINE", "n": len(frame)}
    for sessions in (20, 40):
        baseline.update({
            f"mfe{sessions}_p25": frame[f"max{sessions}"].quantile(0.25),
            f"mfe{sessions}_median": frame[f"max{sessions}"].median(),
            f"mfe{sessions}_p75": frame[f"max{sessions}"].quantile(0.75),
            f"mae{sessions}_p25": frame[f"min{sessions}"].quantile(0.25),
            f"close_ret{sessions}_median": frame[f"ret{sessions}_forward"].median(),
            f"touch15_{sessions}": frame[f"max{sessions}"].ge(0.15).mean(),
            f"touch25_{sessions}": frame[f"max{sessions}"].ge(0.25).mean(),
        })
    rows.append(baseline)
    return pd.DataFrame(rows).sort_values("context_key").reset_index(drop=True)


def _oos_check(oos: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lookup = calibration.set_index("context_key")
    baseline = lookup.loc["BASELINE"]
    for key, group in oos.groupby("context_key"):
        trained = lookup.loc[key] if key in lookup.index and lookup.loc[key, "n"] >= 200 else baseline
        rows.append({
            "context_key": key, "n": len(group), "calibration_source": trained.name,
            "pred_mfe20_median": trained["mfe20_median"],
            "actual_mfe20_median": group["max20"].median(),
            "pred_mfe40_median": trained["mfe40_median"],
            "actual_mfe40_median": group["max40"].median(),
            "mfe20_iqr_coverage": group["max20"].between(
                trained["mfe20_p25"], trained["mfe20_p75"]).mean(),
        })
    return pd.DataFrame(rows)


def run(max_files: int = 0) -> int:
    files = sorted(glob.glob("data/*_daily.csv"))
    if max_files:
        files = files[:max_files]
    rows: list[dict] = []
    for number, path in enumerate(files, 1):
        rows.extend(_stock_rows(path))
        if number % 500 == 0:
            print(f"[value_calibration] files={number}/{len(files)} rows={len(rows)}")
    samples = pd.DataFrame(rows)
    if samples.empty:
        raise RuntimeError("no value calibration samples")
    train = samples[samples["sample"] == "TRAIN"]
    oos = samples[samples["sample"] == "OOS_2026H1"]
    calibration = _stats(train)
    oos_check = _oos_check(oos, calibration)
    OUT.mkdir(parents=True, exist_ok=True)
    samples.to_csv(OUT / "samples.csv", index=False, encoding="utf-8-sig")
    calibration.to_csv(OUT / "calibration.csv", index=False, encoding="utf-8-sig")
    oos_check.to_csv(OUT / "oos_check.csv", index=False, encoding="utf-8-sig")
    payload = {
        "version": VERSION, "event_detector": PE.VERSION,
        "train_rows": len(train), "oos_rows": len(oos),
        "minimum_context_n": 200,
        "contexts": calibration.set_index("context_key").round(6).to_dict("index"),
    }
    (OUT / "calibration.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    shown = calibration.copy()
    percentage = [column for column in shown if column not in {"context_key", "n"}]
    shown[percentage] = (shown[percentage] * 100).round(1)
    check = oos_check.copy()
    for column in [c for c in check if c not in {"context_key", "n", "calibration_source"}]:
        check[column] = (check[column] * 100).round(1)
    report = "\n".join([
        "# 主升浪候选潜在空间历史校准", "",
        f"- 版本：`{VERSION}`；训练样本 {len(train):,}，2026H1 OOS {len(oos):,}。",
        "- 样本为所有流动性合格日期的固定10日抽样，并额外保留已验证事件/量价关系日期。",
        "- MFE是未来窗口最大有利波动，不等于可实现收益；MAE是最大不利波动。",
        "- 当前工具只输出历史分位区间，不输出承诺收益率。", "",
        "## 冻结校准（数值为百分比）", "", shown.to_markdown(index=False), "",
        "## 2026H1 OOS检查（数值为百分比）", "", check.to_markdown(index=False), "",
    ])
    (OUT / "report.md").write_text(report, encoding="utf-8")
    print(f"[value_calibration] train={len(train)} oos={len(oos)} -> {OUT / 'report.md'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()
    return run(args.max_files)


if __name__ == "__main__":
    raise SystemExit(main())
