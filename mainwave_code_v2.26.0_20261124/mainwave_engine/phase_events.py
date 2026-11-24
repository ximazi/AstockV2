# -*- coding: utf-8 -*-
"""Point-in-time market-phase and price-volume event observations.

This module is report-only. It does not change ARMED/READY/TRIGGERED transitions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "phase_events_observation_v2_20260830"
VALIDATION_PATH = Path("results/mainwave_phase_event_study/validation.json")
PACKAGED_VALIDATION_PATH = Path(__file__).with_name("calibration") / "phase_event_validation.json"

EVENT_LABELS = {
    "SPRING": "挖坑回收",
    "TEST": "缩量测试",
    "SOS": "强势突破",
    "LPS": "突破回踩承接",
    "UTAD": "高位假突破",
    "SOW": "放量破位",
}
PHASE_LABELS = {
    "ACCUMULATION": "吸筹",
    "REACCUMULATION": "再吸筹",
    "MARKUP": "拉升",
    "CLIMAX": "高潮转换",
    "DISTRIBUTION": "派发",
    "MARKDOWN": "下跌",
    "TRANSITION": "过渡/未确认",
}
EFFORT_LABELS = {
    "DEMAND_EXPANSION": "需求扩张",
    "ABSORPTION": "放量承接",
    "NO_SUPPLY": "缩量无供应",
    "DISTRIBUTION_PRESSURE": "放量受阻",
    "NEUTRAL": "中性",
}
EVENT_COLUMNS = list(EVENT_LABELS)
EVENT_TTL = {"LPS": 10, "UTAD": 10}


def _safe_div(left: pd.Series, right: pd.Series) -> pd.Series:
    return left / right.replace(0, np.nan)


def classify_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a point-in-time observation frame; no forward columns are used."""
    required = {"date", "open", "high", "low", "close", "amount"}
    if raw.empty or not required.issubset(raw.columns):
        return pd.DataFrame()
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    numeric = ["open", "high", "low", "close", "amount"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", *numeric]).sort_values("date").reset_index(drop=True)
    if len(frame) < 65:
        return pd.DataFrame()

    raw_close = frame["close"]
    raw_return = raw_close.pct_change()
    if "pct_chg" in frame.columns:
        reported_return = pd.to_numeric(frame["pct_chg"], errors="coerce") / 100
        reported_return = reported_return.fillna(raw_return)
    else:
        reported_return = raw_return
    reported_return = reported_return.fillna(0).clip(lower=-0.95, upper=3.0)
    return_index = 100 * np.exp(np.log1p(reported_return).cumsum())
    scale = _safe_div(return_index, raw_close)
    o = frame["open"] * scale
    h = frame["high"] * scale
    low = frame["low"] * scale
    close = return_index
    amount = frame["amount"]
    previous_close = close.shift(1)
    true_range = pd.concat([
        h - low, (h - previous_close).abs(), (low - previous_close).abs()
    ], axis=1).max(axis=1)
    day_range = (h - low).clip(lower=0)
    atr20 = true_range.shift(1).rolling(20, min_periods=15).mean()
    atr60 = true_range.shift(1).rolling(60, min_periods=40).mean()
    range_med20 = day_range.shift(1).rolling(20, min_periods=15).median()
    amount_med20 = amount.shift(1).rolling(20, min_periods=15).median()
    prev_high20 = h.shift(1).rolling(20, min_periods=15).max()
    prev_low20 = low.shift(1).rolling(20, min_periods=15).min()
    high60 = h.shift(1).rolling(60, min_periods=40).max()
    low60 = low.shift(1).rolling(60, min_periods=40).min()
    ma20 = close.rolling(20, min_periods=15).mean()
    ma60 = close.rolling(60, min_periods=40).mean()
    cloc = _safe_div(close - low, day_range).fillna(0.5).clip(0, 1)
    amount_ratio = _safe_div(amount, amount_med20)
    range_ratio = _safe_div(day_range, range_med20)
    ret1 = close.pct_change()
    ret5 = close.pct_change(5)
    ret20 = close.pct_change(20)
    ret60 = close.pct_change(60)
    range_pos60 = _safe_div(close - low60, high60 - low60).clip(0, 1)
    ma20_slope5 = ma20.pct_change(5)
    range_width20_atr = _safe_div(prev_high20 - prev_low20, atr20)

    spring = (
        (low < prev_low20 - 0.05 * atr20)
        & (low >= prev_low20 - 1.50 * atr20)
        & (close > prev_low20)
        & (cloc >= 0.65)
        & (amount_ratio <= 1.80)
        & (ret60 <= 0.35)
    )
    spring_level = prev_low20.where(spring).shift(1).ffill(limit=15)
    test = (
        spring_level.notna()
        & (low <= spring_level + 0.50 * atr20)
        & (close >= spring_level)
        & (amount_ratio <= 0.80)
        & (range_ratio <= 0.90)
        & (cloc >= 0.50)
    )
    sos = (
        (close > prev_high20 + 0.10 * atr20)
        & (cloc >= 0.70)
        & (amount_ratio >= 1.20)
        & (range_ratio >= 1.10)
        & (ret1 > 0)
    )
    sos_level = prev_high20.where(sos).shift(1).ffill(limit=15)
    lps = (
        sos_level.notna()
        & (low <= sos_level + 0.50 * atr20)
        & (close >= sos_level - 0.10 * atr20)
        & (amount_ratio <= 0.85)
        & (range_ratio <= 0.95)
        & (ret1 > -0.05)
    )
    high_context = (ret60 >= 0.20) | (range_pos60 >= 0.80)
    utad = (
        high_context
        & (h > prev_high20 + 0.10 * atr20)
        & (close < prev_high20)
        & (cloc <= 0.35)
        & (amount_ratio >= 1.30)
    )
    sow = (
        (close < prev_low20 - 0.10 * atr20)
        & (cloc <= 0.35)
        & (amount_ratio >= 1.20)
        & (range_ratio >= 1.10)
    )

    high_effort = amount_ratio >= 1.30
    distribution_pressure = (
        high_context & high_effort & (cloc <= 0.40)
        & ((ret1 <= 0.01) | (range_ratio >= 1.20))
    )
    demand_expansion = sos | (
        high_effort & (ret1 > 0.02) & (cloc >= 0.70) & (range_ratio >= 1.10)
    )
    absorption = (
        (range_pos60 <= 0.55) & high_effort & (cloc >= 0.55)
        & (ret1 >= -0.01) & ~demand_expansion
    )
    no_supply = (
        (ret1 <= 0) & (amount_ratio <= 0.70) & (range_ratio <= 0.80)
        & (cloc >= 0.45)
    )

    phase = pd.Series("TRANSITION", index=frame.index, dtype="object")
    base = (
        (range_width20_atr <= 6.0) & (ret20.abs() <= 0.12)
        & (range_pos60 <= 0.75)
    ) | spring | test
    reaccumulation = (
        (ret60 >= 0.15) & (close >= ma60) & (range_width20_atr <= 6.0)
        & (atr20 <= 0.90 * atr60) & ~utad
    )
    markup = (
        (close > ma20) & (ma20 > ma60) & (ma20_slope5 > 0)
        & (ret20 >= 0.05)
    )
    climax = (
        high_context & (range_pos60 >= 0.90) & high_effort
        & ((ret5 >= 0.18) | (_safe_div(close - ma20, atr20) >= 4.0))
    )
    distribution = utad | distribution_pressure
    markdown = sow | ((close < ma20) & (ma20 < ma60) & (ret20 <= -0.08))
    phase.loc[base] = "ACCUMULATION"
    phase.loc[reaccumulation] = "REACCUMULATION"
    phase.loc[markup] = "MARKUP"
    phase.loc[climax] = "CLIMAX"
    phase.loc[distribution] = "DISTRIBUTION"
    phase.loc[markdown] = "MARKDOWN"

    effort = pd.Series("NEUTRAL", index=frame.index, dtype="object")
    effort.loc[no_supply] = "NO_SUPPLY"
    effort.loc[absorption] = "ABSORPTION"
    effort.loc[demand_expansion] = "DEMAND_EXPANSION"
    effort.loc[distribution_pressure] = "DISTRIBUTION_PRESSURE"

    for name, values in {
        "SPRING": spring, "TEST": test, "SOS": sos, "LPS": lps,
        "UTAD": utad, "SOW": sow,
    }.items():
        frame[name] = values.fillna(False).astype(bool)
    frame["broad_phase"] = phase
    frame["effort_result"] = effort
    frame["atr20"] = atr20
    frame["amount_ratio"] = amount_ratio
    frame["range_ratio"] = range_ratio
    frame["close_location"] = cloc
    frame["ret5"] = ret5
    frame["ret20"] = ret20
    frame["ret60"] = ret60
    frame["range_position60"] = range_pos60
    frame["return_index"] = return_index
    frame["structure_floor"] = prev_low20
    frame["structure_ceiling"] = prev_high20
    return frame


def load_validation() -> dict:
    for path in (VALIDATION_PATH, PACKAGED_VALIDATION_PATH):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return {"events": {}, "phases": {}}


def _validation() -> dict:
    return load_validation()


def latest_snapshot(symbol: str, as_of: str, data_root: Path = Path("data")) -> dict:
    path = data_root / f"{symbol}_daily.csv"
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return {}
    frame = classify_frame(raw)
    if frame.empty:
        return {}
    as_of_ts = pd.Timestamp(as_of)
    frame = frame[frame["date"] <= as_of_ts]
    if frame.empty:
        return {}
    latest = frame.iloc[-1]
    recent = frame.tail(20)
    events = []
    for index, row in recent.iterrows():
        for event in EVENT_COLUMNS:
            if bool(row[event]):
                events.append((index, row["date"], event))
    validation = _validation()
    _, raw_event_date, raw_event = events[-1] if events else (-1, pd.NaT, "")
    approved = [
        item for item in events
        if validation.get("events", {}).get(item[2], {}).get("status") in {"PASS", "WATCH"}
        and (len(frame) - 1 - item[0]) <= EVENT_TTL.get(item[2], 0)
    ]
    _, event_date, event = approved[-1] if approved else (-1, pd.NaT, "")
    event_status = validation.get("events", {}).get(event, {}).get("status", "") \
        if event else ""
    phase = str(latest["broad_phase"])
    phase_status = validation.get("phases", {}).get(phase, {}).get("status", "UNVALIDATED")
    effort = str(latest["effort_result"])
    effort_status = validation.get("efforts", {}).get(
        effort, {}).get("status", "UNVALIDATED")
    sequence = ""
    if bool(recent["TEST"].any()) and bool(recent["SPRING"].any()):
        sequence = "SPRING_TO_TEST"
    if bool(recent["LPS"].any()) and bool(recent["SOS"].any()):
        sequence = "SOS_TO_LPS"
    if event in {"UTAD", "SOW"}:
        sequence = "RISK_TRANSITION"
    if event == "UTAD" or (
        effort == "DISTRIBUTION_PRESSURE" and effort_status == "PASS"
    ):
        observation_action = "RISK_REVIEW"
    elif phase == "CLIMAX" and phase_status in {"PASS", "WATCH"}:
        observation_action = "NO_CHASE_REVIEW"
    elif event == "LPS":
        observation_action = "LPS_CONFIRMATION_WATCH"
    elif effort == "DEMAND_EXPANSION" and effort_status == "PASS":
        observation_action = "DEMAND_CONFIRMATION_WATCH"
    elif phase == "MARKUP" and phase_status == "PASS":
        observation_action = "MARKUP_CONTEXT_ONLY"
    else:
        observation_action = "NO_VALIDATED_EVENT"
    return {
        "broad_phase": phase,
        "broad_phase_label": PHASE_LABELS.get(phase, ""),
        "phase_validation": phase_status,
        "latest_event": event,
        "latest_event_label": EVENT_LABELS.get(event, ""),
        "latest_event_date": "" if pd.isna(event_date) else str(event_date.date()),
        "event_validation": event_status,
        "latest_event_raw": raw_event,
        "latest_event_raw_date": "" if pd.isna(raw_event_date) else str(raw_event_date.date()),
        "event_sequence": sequence,
        "effort_result": effort,
        "effort_result_label": EFFORT_LABELS.get(effort, ""),
        "effort_validation": effort_status,
        "observation_action": observation_action,
    }
