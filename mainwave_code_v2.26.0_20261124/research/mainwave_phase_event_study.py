# -*- coding: utf-8 -*-
"""Historical validation for broad phases and Wyckoff-style price-volume events.

The study uses 2022-2024 for definition/development, freezes the rules, validates
on 2025, and reports 2026 H1 as out-of-sample. It does not modify production
state transitions.
"""
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


VERSION = "mainwave_phase_event_study_v2_20260830"
OUT = Path("results/mainwave_phase_event_study")
FWD = 40
MIN_GAIN = 0.25
MAX_UNDERWATER = -0.15
CONTROL_MAX_GAIN = 0.10
MIN_AMOUNT = 5e7
COOLDOWN = 20
SPLITS = {
    "DEV_2022_2024": (pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31")),
    "VALID_2025": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31")),
    "OOS_2026H1": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-30")),
}
SUPPORT_WINDOWS = {
    "SPRING": (-20, -1),
    "TEST": (-20, -1),
    "SOS": (-2, 2),
    "LPS": (-5, 10),
}


def _forward_extreme(values: pd.Series, kind: str) -> pd.Series:
    shifted = values.shift(-1).iloc[::-1]
    roller = shifted.rolling(FWD, min_periods=FWD)
    result = roller.max() if kind == "max" else roller.min()
    return result.iloc[::-1]


def _cooldown(indices: list[int]) -> list[int]:
    selected: list[int] = []
    last = -10**9
    for index in indices:
        if index - last >= COOLDOWN:
            selected.append(index)
            last = index
    return selected


def _even_sample(indices: list[int], count: int) -> list[int]:
    if count <= 0 or not indices:
        return []
    if len(indices) <= count:
        return indices
    positions = np.linspace(0, len(indices) - 1, count).round().astype(int)
    return [indices[i] for i in positions]


def _has(frame: pd.DataFrame, event: str, anchor: int, lo: int, hi: int) -> bool:
    start = max(0, anchor + lo)
    end = min(len(frame), anchor + hi + 1)
    return bool(frame[event].iloc[start:end].any()) if end > start else False


def _sequence(frame: pd.DataFrame, first: str, second: str,
              anchor: int, lo: int, hi: int) -> bool:
    start = max(0, anchor + lo)
    end = min(len(frame), anchor + hi + 1)
    window = frame.iloc[start:end]
    first_positions = np.flatnonzero(window[first].to_numpy(bool))
    second_positions = np.flatnonzero(window[second].to_numpy(bool))
    return bool(len(first_positions) and len(second_positions)
                and np.any(first_positions[:, None] < second_positions[None, :]))


def _mode(values: pd.Series) -> str:
    mode = values.mode()
    return str(mode.iloc[0]) if len(mode) else "TRANSITION"


def _sample_row(symbol: str, frame: pd.DataFrame, anchor: int,
                split: str, label: str) -> dict:
    close = frame["return_index"].to_numpy(float)
    future = close[anchor + 1:anchor + 1 + FWD]
    peak = anchor + 1 + int(np.argmax(future))
    peak_end = min(len(frame), peak + 11)
    before = frame.iloc[max(0, anchor - 10):anchor]
    after_peak = frame.iloc[peak:peak_end]
    row = {
        "split": split,
        "label": label,
        "symbol": symbol,
        "date": str(frame.at[anchor, "date"].date()),
        "max_gain40": float(np.max(future) / close[anchor] - 1),
        "min_path40": float(np.min(future) / close[anchor] - 1),
        "phase_pre_start": str(frame.at[anchor - 1, "broad_phase"]),
        "phase_post_peak": _mode(after_peak["broad_phase"]),
        "effort_pre_demand": bool((before["effort_result"] == "DEMAND_EXPANSION").any()),
        "effort_pre_absorption": bool((before["effort_result"] == "ABSORPTION").any()),
        "effort_pre_no_supply": bool((before["effort_result"] == "NO_SUPPLY").any()),
        "effort_pre_distribution": bool((before["effort_result"] == "DISTRIBUTION_PRESSURE").any()),
        "effort_post_distribution": bool(
            (after_peak["effort_result"] == "DISTRIBUTION_PRESSURE").any()),
        "sequence_spring_test": _sequence(frame, "SPRING", "TEST", anchor, -30, -1),
        "sequence_sos_lps": _sequence(frame, "SOS", "LPS", anchor, -5, 10),
        "post_peak_UTAD": bool(after_peak["UTAD"].any()),
        "post_peak_SOW": bool(after_peak["SOW"].any()),
    }
    for event, (lo, hi) in SUPPORT_WINDOWS.items():
        row[f"event_{event}"] = _has(frame, event, anchor, lo, hi)
    return row


def _split_name(date: pd.Timestamp) -> str:
    for name, (lo, hi) in SPLITS.items():
        if lo <= date <= hi:
            return name
    return ""


def _stock_samples(path: str) -> list[dict]:
    symbol = os.path.basename(path).replace("_daily.csv", "")
    if universe_rejection_reason(symbol):
        return []
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, pd.errors.EmptyDataError):
        return []
    frame = PE.classify_frame(raw)
    if frame.empty or len(frame) < 130:
        return []
    frame["fwd_max"] = _forward_extreme(frame["return_index"], "max")
    frame["fwd_min"] = _forward_extreme(frame["return_index"], "min")
    frame["gain40"] = frame["fwd_max"] / frame["return_index"] - 1
    frame["path40"] = frame["fwd_min"] / frame["return_index"] - 1
    if "pct_chg" in frame.columns:
        limit = pd.to_numeric(frame["pct_chg"], errors="coerce").ge(9.5)
    else:
        limit = frame["close"].pct_change().ge(0.095)
    frame["limit11"] = limit.rolling(11, min_periods=1, center=True).sum()
    frame["split"] = frame["date"].map(_split_name)
    liquid = frame["amount"] >= MIN_AMOUNT
    eligible = frame["split"].ne("") & liquid & frame["fwd_max"].notna()
    trend = (frame["ret60"] < 0.80) & (frame["limit11"] < 4)
    positive_mask = (
        eligible & trend & (frame["gain40"] >= MIN_GAIN)
        & (frame["path40"] >= MAX_UNDERWATER)
    )
    control_mask = (
        eligible & trend & (frame["gain40"] < CONTROL_MAX_GAIN)
        & (frame["path40"] >= MAX_UNDERWATER)
    )
    rows: list[dict] = []
    for split in SPLITS:
        positive = _cooldown(frame.index[positive_mask & frame["split"].eq(split)].tolist())
        if not positive:
            continue
        blocked = np.zeros(len(frame), dtype=bool)
        for index in positive:
            blocked[max(0, index - FWD):min(len(frame), index + FWD + 1)] = True
        controls = frame.index[control_mask & frame["split"].eq(split) & ~blocked].tolist()
        controls = _even_sample(_cooldown(controls), len(positive))
        for index in positive:
            rows.append(_sample_row(symbol, frame, index, split, "MAINWAVE"))
        for index in controls:
            rows.append(_sample_row(symbol, frame, index, split, "CONTROL"))
    return rows


def _rate_table(samples: pd.DataFrame, columns: dict[str, str], context: str) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        block = samples[samples["split"] == split]
        positive = block[block["label"] == "MAINWAVE"]
        control = block[block["label"] == "CONTROL"]
        for name, column in columns.items():
            pos_rate = float(positive[column].mean()) if len(positive) else np.nan
            ctl_rate = float(control[column].mean()) if len(control) else np.nan
            rows.append({
                "context": context, "split": split, "item": name,
                "mainwave_n": len(positive), "control_n": len(control),
                "mainwave_rate": pos_rate, "control_rate": ctl_rate,
                "lift": pos_rate / ctl_rate if ctl_rate and np.isfinite(ctl_rate) else np.nan,
                "rate_delta": pos_rate - ctl_rate,
            })
    return pd.DataFrame(rows)


def _phase_table(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for context, column in {
        "PRE_START": "phase_pre_start", "POST_PEAK": "phase_post_peak"
    }.items():
        for split in SPLITS:
            block = samples[samples["split"] == split]
            for phase in PE.PHASE_LABELS:
                positive = block[block["label"] == "MAINWAVE"][column].eq(phase)
                control = block[block["label"] == "CONTROL"][column].eq(phase)
                pos_rate = float(positive.mean()) if len(positive) else np.nan
                ctl_rate = float(control.mean()) if len(control) else np.nan
                rows.append({
                    "context": context, "split": split, "item": phase,
                    "mainwave_rate": pos_rate, "control_rate": ctl_rate,
                    "lift": pos_rate / ctl_rate if ctl_rate and np.isfinite(ctl_rate) else np.nan,
                    "rate_delta": pos_rate - ctl_rate,
                })
    return pd.DataFrame(rows)


def _status(valid: pd.Series, oos: pd.Series, min_rate: float,
            min_lift: float) -> str:
    if valid.empty or oos.empty:
        return "INSUFFICIENT"
    stable = (
        valid["mainwave_rate"] >= min_rate and oos["mainwave_rate"] >= min_rate
        and valid["lift"] >= min_lift and oos["lift"] >= min_lift
    )
    if stable:
        return "PASS"
    directional = valid["rate_delta"] > 0 and oos["rate_delta"] > 0
    return "WATCH" if directional else "REJECT"


def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 3) if np.isfinite(number) else None


def _validation(event_metrics: pd.DataFrame, phase_metrics: pd.DataFrame) -> dict:
    events = {}
    for event in PE.EVENT_COLUMNS:
        context = "SUPPORT_EVENT" if event in SUPPORT_WINDOWS else "POST_PEAK_RISK"
        item = event
        valid = event_metrics[(event_metrics["context"] == context)
                              & (event_metrics["split"] == "VALID_2025")
                              & (event_metrics["item"] == item)]
        oos = event_metrics[(event_metrics["context"] == context)
                            & (event_metrics["split"] == "OOS_2026H1")
                            & (event_metrics["item"] == item)]
        status = _status(valid.iloc[0] if len(valid) else pd.Series(dtype=float),
                         oos.iloc[0] if len(oos) else pd.Series(dtype=float),
                         0.01 if event in SUPPORT_WINDOWS else 0.03,
                         1.15 if event in SUPPORT_WINDOWS else 1.50)
        events[event] = {
            "status": status, "context": context,
            "valid_2025_lift": None if not len(valid) else _number(valid.iloc[0]["lift"]),
            "oos_2026_lift": None if not len(oos) else _number(oos.iloc[0]["lift"]),
        }
    phases = {}
    for phase in PE.PHASE_LABELS:
        context = "POST_PEAK" if phase in {"DISTRIBUTION", "MARKDOWN"} else "PRE_START"
        valid = phase_metrics[(phase_metrics["context"] == context)
                              & (phase_metrics["split"] == "VALID_2025")
                              & (phase_metrics["item"] == phase)]
        oos = phase_metrics[(phase_metrics["context"] == context)
                            & (phase_metrics["split"] == "OOS_2026H1")
                            & (phase_metrics["item"] == phase)]
        status = _status(valid.iloc[0] if len(valid) else pd.Series(dtype=float),
                         oos.iloc[0] if len(oos) else pd.Series(dtype=float), 0.03, 1.10)
        phases[phase] = {
            "status": status, "context": context,
            "valid_2025_lift": None if not len(valid) else _number(valid.iloc[0]["lift"]),
            "oos_2026_lift": None if not len(oos) else _number(oos.iloc[0]["lift"]),
        }
    efforts = {}
    effort_context = {
        "DEMAND_EXPANSION": "SEQUENCE_OR_EFFORT",
        "ABSORPTION": "SEQUENCE_OR_EFFORT",
        "NO_SUPPLY": "SEQUENCE_OR_EFFORT",
        "DISTRIBUTION_PRESSURE": "POST_PEAK_EFFORT",
    }
    for effort, context in effort_context.items():
        valid = event_metrics[(event_metrics["context"] == context)
                              & (event_metrics["split"] == "VALID_2025")
                              & (event_metrics["item"] == effort)]
        oos = event_metrics[(event_metrics["context"] == context)
                            & (event_metrics["split"] == "OOS_2026H1")
                            & (event_metrics["item"] == effort)]
        status = _status(valid.iloc[0] if len(valid) else pd.Series(dtype=float),
                         oos.iloc[0] if len(oos) else pd.Series(dtype=float),
                         0.05, 1.15)
        efforts[effort] = {
            "status": status, "context": context,
            "valid_2025_lift": None if not len(valid) else _number(valid.iloc[0]["lift"]),
            "oos_2026_lift": None if not len(oos) else _number(oos.iloc[0]["lift"]),
        }
    sequences = {}
    for sequence in ("SPRING_TO_TEST", "SOS_TO_LPS"):
        valid = event_metrics[(event_metrics["context"] == "SEQUENCE_OR_EFFORT")
                              & (event_metrics["split"] == "VALID_2025")
                              & (event_metrics["item"] == sequence)]
        oos = event_metrics[(event_metrics["context"] == "SEQUENCE_OR_EFFORT")
                            & (event_metrics["split"] == "OOS_2026H1")
                            & (event_metrics["item"] == sequence)]
        sequences[sequence] = {
            "status": _status(
                valid.iloc[0] if len(valid) else pd.Series(dtype=float),
                oos.iloc[0] if len(oos) else pd.Series(dtype=float), 0.01, 1.15),
            "valid_2025_lift": None if not len(valid) else _number(valid.iloc[0]["lift"]),
            "oos_2026_lift": None if not len(oos) else _number(oos.iloc[0]["lift"]),
        }
    return {"version": VERSION, "detector_version": PE.VERSION,
            "events": events, "phases": phases, "efforts": efforts,
            "sequences": sequences}


def _report(samples: pd.DataFrame, event_metrics: pd.DataFrame,
            phase_metrics: pd.DataFrame, validation: dict) -> str:
    counts = samples.groupby(["split", "label"]).size().unstack(fill_value=0)
    event_show = event_metrics.copy()
    for column in ["mainwave_rate", "control_rate", "rate_delta"]:
        event_show[column] = (event_show[column] * 100).round(1)
    event_show["lift"] = event_show["lift"].round(2)
    phase_show = phase_metrics.copy()
    for column in ["mainwave_rate", "control_rate", "rate_delta"]:
        phase_show[column] = (phase_show[column] * 100).round(1)
    phase_show["lift"] = phase_show["lift"].round(2)
    verdict = pd.DataFrame([
        {"type": "event", "item": key, **value}
        for key, value in validation["events"].items()
    ] + [
        {"type": "phase", "item": key, **value}
        for key, value in validation["phases"].items()
    ] + [
        {"type": "effort", "item": key, **value}
        for key, value in validation["efforts"].items()
    ] + [
        {"type": "sequence", "item": key, **value}
        for key, value in validation["sequences"].items()
    ])
    return "\n".join([
        "# 主升浪阶段与关键事件历史验证", "",
        f"- 研究版本：`{VERSION}`；检测器：`{PE.VERSION}`。",
        "- 时间隔离：2022-2024定义；2025冻结验证；2026H1 OOS。",
        "- 正样本：未来40日最大涨幅>=25%、最大水下>=-15%、成交额>=5000万、20日去重。",
        "- 对照：同股票同年度、流动性合格、未来40日最大涨幅<10%，并排除正样本前后40日。",
        "- 本研究验证发现能力与阶段区分，不修改生产状态机，不生成交易指令。", "",
        "## 样本", "", counts.to_markdown(), "",
        "## 冻结验证裁决", "", verdict.to_markdown(index=False), "",
        "## 事件明细（rate/delta为百分比）", "",
        event_show.to_markdown(index=False), "",
        "## 阶段明细（rate/delta为百分比）", "",
        phase_show.to_markdown(index=False), "",
        "## 使用边界", "",
        "- PASS仅允许进入观察报表；生产转移仍需独立前向验证。",
        "- WATCH保留显示但不得成为READY硬支柱；REJECT不接入候选解释。",
        "- 事件是公开量价代理证据，不证明存在单一机构或所谓主力操盘。",
    ])


def run(max_files: int = 0) -> int:
    files = sorted(glob.glob("data/*_daily.csv"))
    if max_files:
        files = files[:max_files]
    rows: list[dict] = []
    for number, path in enumerate(files, 1):
        rows.extend(_stock_samples(path))
        if number % 500 == 0:
            print(f"[phase_event_study] files={number}/{len(files)} samples={len(rows)}")
    samples = pd.DataFrame(rows)
    if samples.empty:
        raise RuntimeError("no historical samples produced")
    event_metrics = _rate_table(
        samples,
        {event: f"event_{event}" for event in SUPPORT_WINDOWS},
        "SUPPORT_EVENT",
    )
    risk = _rate_table(
        samples, {"UTAD": "post_peak_UTAD", "SOW": "post_peak_SOW"},
        "POST_PEAK_RISK",
    )
    sequence = _rate_table(samples, {
        "SPRING_TO_TEST": "sequence_spring_test",
        "SOS_TO_LPS": "sequence_sos_lps",
        "DEMAND_EXPANSION": "effort_pre_demand",
        "ABSORPTION": "effort_pre_absorption",
        "NO_SUPPLY": "effort_pre_no_supply",
        "DISTRIBUTION_PRESSURE": "effort_pre_distribution",
    }, "SEQUENCE_OR_EFFORT")
    post_peak_effort = _rate_table(
        samples, {"DISTRIBUTION_PRESSURE": "effort_post_distribution"},
        "POST_PEAK_EFFORT")
    event_metrics = pd.concat(
        [event_metrics, risk, sequence, post_peak_effort], ignore_index=True)
    phase_metrics = _phase_table(samples)
    validation = _validation(event_metrics, phase_metrics)
    OUT.mkdir(parents=True, exist_ok=True)
    samples.to_csv(OUT / "samples.csv", index=False, encoding="utf-8-sig")
    event_metrics.to_csv(OUT / "event_metrics.csv", index=False, encoding="utf-8-sig")
    phase_metrics.to_csv(OUT / "phase_metrics.csv", index=False, encoding="utf-8-sig")
    (OUT / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "report.md").write_text(
        _report(samples, event_metrics, phase_metrics, validation), encoding="utf-8")
    print(f"[phase_event_study] samples={len(samples)} -> {OUT / 'report.md'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()
    return run(args.max_files)


if __name__ == "__main__":
    raise SystemExit(main())
