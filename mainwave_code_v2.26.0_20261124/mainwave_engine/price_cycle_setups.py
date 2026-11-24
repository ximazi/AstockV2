"""Causal, observation-only setup memory and asymmetric price-speed evidence."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import breakout_events as BE

VERSION = "price_cycle_setups_v2_1_20260910"
# Research placeholders, in frozen ATR units / active sessions. No order authority.
PULLBACK_ATR = 0.8
PLATFORM_SESSIONS = 5
SETUP_TTL = 20
SPEED_LABELS = {
    "UNKNOWN": "速度数据不足", "UP_ACCELERATING": "上涨加速",
    "UP_RAPID": "快速上涨", "UP_STEADY": "上涨延续",
    "UP_DECELERATING": "上涨减速", "DOWN_FAST": "快速下跌",
    "DOWN_DECELERATING": "下跌减速，非底部确认",
    "DOWN_CONTINUING": "下跌延续", "QUIET": "低位移，方向待确认",
}
SETUP_LABELS = {
    "FIRST_PULLBACK": "本轮首次有效回踩", "LATER_PULLBACK": "本轮后续回踩",
    "PULLBACK_UNKNOWN": "回踩，首次身份未知", "BASE_BREAK": "趋势内平台整理",
}
STATUS_LABELS = {
    "NONE": "暂无有效形态", "FORMING": "形成中", "CONFIRMED": "当日确认，仅观察",
    "FAILED": "形态失效", "EXPIRED": "形态过期",
    "HISTORY_CONFIRMED": "历史已确认", "HISTORY_FAILED": "历史已失效",
    "HISTORY_EXPIRED": "历史已过期",
}
EVIDENCE_LABELS = {
    "NONE": "暂无新增供需证据", "DEMAND_RECOVERY": "价格与需求恢复",
    "HIGH_EFFORT_STALL": "高换手、价格受阻", "SELLING_SLOWING": "下跌降速，承接仍待验证",
    "MISSING": "供需数据不足",
}


def _median(values) -> float | None:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data) & (data > 0)]
    return float(np.median(data)) if len(data) else None


def speed_frame(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["return_index"].astype(float)
    atr = frame["atr20"].shift(10).where(lambda x: x > 0)
    # Both disjoint windows use the SAME already-known scale, not today's swollen ATR.
    speed = (close - close.shift(5)) / atr / 5
    prior = (close.shift(5) - close.shift(10)) / atr / 5
    acceleration = speed - prior
    code = pd.Series("QUIET", index=frame.index)
    code.loc[speed > 0.08] = "UP_STEADY"
    code.loc[(speed > 0.08) & (acceleration < -0.03)] = "UP_DECELERATING"
    code.loc[(speed > 0.08) & (acceleration > 0.03)] = "UP_ACCELERATING"
    code.loc[(speed > 0.60) & (acceleration > 0)] = "UP_RAPID"
    code.loc[speed < -0.08] = "DOWN_CONTINUING"
    code.loc[(speed < -0.08) & (acceleration > 0.03)] = "DOWN_DECELERATING"
    code.loc[(speed < -0.60) & (acceleration <= 0.03)] = "DOWN_FAST"
    code.loc[speed.isna() | acceleration.isna()] = "UNKNOWN"
    return pd.DataFrame({
        "speed_state": code, "speed_label": code.map(SPEED_LABELS),
        "speed_atr_per_session": speed, "speed_change": acceleration,
    })


def annotate(frame: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    """Attach child setups without changing ANY parent-cycle or production columns."""
    required = {"open", "high", "low", "close", "atr20", "return_index"}
    if timeline.empty or not required.issubset(frame.columns):
        return timeline
    frame = frame.reset_index(drop=True)
    timeline = timeline.reset_index(drop=True)
    if "breakout_version" not in timeline:
        timeline = BE.annotate(frame, timeline)
    speed = speed_frame(frame)
    scale = frame["return_index"] / frame["close"]
    highs = (frame["high"] * scale).to_numpy(float)
    lows = (frame["low"] * scale).to_numpy(float)
    closes = frame["return_index"].to_numpy(float)
    ema = frame["return_index"].ewm(span=20, adjust=False).mean().to_numpy(float)
    atr = frame["atr20"].to_numpy(float)
    turn = pd.to_numeric(frame.get("turnover", pd.Series(np.nan, index=frame.index)),
                         errors="coerce").where(lambda x: x > 0).to_numpy(float)
    active, last, cycle = None, None, ""
    ordinal, adjustments, peak_idx, leg_start = 0, 0, 0, 0
    first_known, start_observed = False, False
    output = []
    for i, row in enumerate(timeline.to_dict("records")):
        date, parent = row["date"], row["cycle_stage"]
        cid = row["cycle_id"]
        if cid != cycle:
            cycle, active, last = cid, None, None
            ordinal, adjustments = 0, 0
            peak_idx = leg_start = i
            start_observed = (row["cycle_reason"] != "TREND_ALREADY_VISIBLE" and i >= BE.LOOKBACK
                              and np.isfinite(atr[i - 1]) and atr[i - 1] > 0)
            first_known = False
        linked = (row["breakout_status"] == "ADVANCING" and bool(row["breakout_accepted_at"])
                  and row["breakout_advanced_at"] <= timeline.iloc[peak_idx]["date"])
        if not adjustments:
            first_known = bool(start_observed and linked)
        if closes[i] > closes[peak_idx]:
            peak_idx = i
        current_speed = speed.iloc[i].to_dict()
        evidence = "NONE"
        relative = row.get("turnover_relative20", np.nan)
        cloc = frame.iloc[i].get("close_location", 0.5)
        if not np.isfinite(turn[i]):
            evidence = "MISSING"
        elif relative >= 1.3 and cloc <= 0.4:
            evidence = "HIGH_EFFORT_STALL"
        elif closes[i] > ema[i] and cloc >= 0.65 and relative >= 1.1:
            evidence = "DEMAND_RECOVERY"
        elif current_speed["speed_state"] == "DOWN_DECELERATING":
            evidence = "SELLING_SLOWING"
        closed_today = False
        if active is not None:
            support = _median(turn[active["start_index"]:i + 1])
            active["setup_turnover_contraction"] = (
                support / active["impulse_turnover"] if support and active["impulse_turnover"] else None)
            active["setup_age"] = i - active["start_index"]
            outcome = ""
            chain_broken = (active.get("setup_breakout_id") and
                            (row["breakout_id"] != active["setup_breakout_id"]
                             or row["breakout_status"] in {"FAILED", "REJECTED", "EXPIRED"}))
            if parent in {"FAILED", "REBUILDING"} or chain_broken or closes[i] < active["setup_invalid_index"]:
                outcome = "FAILED"
            elif active["setup_age"] > SETUP_TTL:
                outcome = "EXPIRED"
            else:
                pullback_turn = _median(turn[active["start_index"]:i])
                demand = pullback_turn and np.isfinite(turn[i]) and turn[i] >= 1.1 * pullback_turn
                drying = active["setup_turnover_contraction"]
                if (closes[i] > active["setup_trigger_index"] and cloc >= 0.60
                        and closes[i] > closes[i - 1] and demand
                        and drying is not None and drying <= 1.0
                        and parent not in {"DISTRIBUTION_RISK", "OVERHEATED"}):
                    outcome = "CONFIRMED"
            if outcome:
                active.update(setup_status=outcome, setup_outcome=outcome, setup_ended_at=date)
                if outcome == "CONFIRMED":
                    active["setup_confirmed_at"] = date
                last, active, closed_today = active, None, True
                peak_idx = leg_start = i
        eligible = (cycle and parent in {"IGNITION", "MARKUP", "PULLBACK"}
                    and np.isfinite(atr[peak_idx]) and atr[peak_idx] > 0)
        if eligible and active is None and not closed_today:
            frozen_atr = atr[peak_idx]
            advanced = closes[peak_idx] - closes[leg_start] >= frozen_atr
            elapsed = i - peak_idx
            pullback = (elapsed >= 2 and closes[peak_idx] - closes[i] >= PULLBACK_ATR * frozen_atr
                        and closes[i] >= closes[i - 1] and lows[i] >= min(lows[i - 2:i]))
            lo = max(leg_start + 1, i - PLATFORM_SESSIONS + 1)
            platform = (i - lo + 1 >= PLATFORM_SESSIONS and elapsed >= PLATFORM_SESSIONS
                        and max(highs[lo:i + 1]) - min(lows[lo:i + 1]) <= 2.5 * frozen_atr
                        and abs(closes[i] - closes[lo]) <= 0.5 * frozen_atr and closes[i] >= ema[i])
            if advanced and (pullback or platform):
                ordinal += 1
                if pullback:
                    adjustments += 1
                    kind = ("FIRST_PULLBACK" if adjustments == 1 else "LATER_PULLBACK") if first_known else "PULLBACK_UNKNOWN"
                    lo = peak_idx + 1
                    trigger = max(highs[max(lo, i - 1):i + 1]) + 0.1 * frozen_atr
                else:
                    kind = "BASE_BREAK"
                    trigger = max(highs[lo:i + 1]) + 0.1 * frozen_atr
                active = {
                    "setup_id": f"{cycle}:s{ordinal}:{date}", "setup_kind": kind,
                    "setup_breakout_id": row["breakout_id"] if linked else "",
                    "setup_breakout_accepted_at": row["breakout_accepted_at"] if linked else "",
                    "setup_breakout_advanced_at": row["breakout_advanced_at"] if linked else "",
                    "setup_first_known": bool(first_known),
                    "setup_label": SETUP_LABELS[kind], "setup_status": "FORMING",
                    "setup_outcome": "", "setup_started_at": date,
                    "setup_adjustment_from": timeline.iloc[lo]["date"],
                    "setup_confirmed_at": "", "setup_ended_at": "",
                    "setup_trigger_index": float(trigger),
                    "setup_invalid_index": float(min(lows[lo:i + 1]) - 0.2 * frozen_atr),
                    "setup_frozen_atr": float(frozen_atr), "setup_age": 0,
                    "setup_turnover_contraction": None,
                    "start_index": lo,
                    "impulse_turnover": _median(turn[max(leg_start, peak_idx - 4):peak_idx + 1]),
                }
        record = dict(active or last or {})
        if not active and last and not closed_today:
            record["setup_status"] = "HISTORY_" + last["setup_outcome"]
        status = record.get("setup_status", "NONE")
        record.pop("start_index", None)
        record.pop("impulse_turnover", None)
        record.update({
            "setup_status": status, "setup_status_label": STATUS_LABELS[status],
            "setup_adjustment_number": adjustments,
            "setup_first_known": record.get("setup_first_known", bool(cycle and first_known)),
            "setup_version": VERSION, "setup_permission": "OBSERVATION_ONLY",
            "setup_next_condition": ("收复冻结确认位，换手收敛后需求恢复；仍须独立执行审批"
                                     if status == "FORMING" else "历史事件不自动生成新订单"),
            "speed_evidence": evidence, "speed_evidence_label": EVIDENCE_LABELS[evidence],
            **current_speed,
        })
        output.append(record)
    result = pd.concat([timeline, pd.DataFrame(output)], axis=1)
    for key in ("setup_id", "setup_kind", "setup_label", "setup_started_at", "setup_adjustment_from",
                "setup_confirmed_at", "setup_ended_at", "setup_outcome", "setup_breakout_id",
                "setup_breakout_accepted_at", "setup_breakout_advanced_at"):
        if key not in result:
            result[key] = ""
        result[key] = result[key].fillna("")
    for key in ("setup_trigger_index", "setup_invalid_index", "setup_frozen_atr",
                "setup_turnover_contraction", "setup_age"):
        if key not in result:
            result[key] = np.nan
    extras = sorted(c for c in result if c not in timeline.columns)
    return result[[*timeline.columns, *extras]]


def setup_history(timeline: pd.DataFrame) -> pd.DataFrame:
    if "setup_id" not in timeline:
        return pd.DataFrame()
    selected = timeline[timeline["setup_id"].ne("")].drop_duplicates("setup_id", keep="last")
    fields = [c for c in selected if c.startswith("setup_") or c in {"symbol", "cycle_id", "date"}]
    return selected[fields].reset_index(drop=True)


STAGE_COLORS = {
    "WATCH": "#8795a5", "PROBED": "#698fbb", "ABSORBING": "#698fbb",
    "COILED": "#5180bc", "ARMED": "#5180bc", "READY": "#5180bc",
    "IGNITION": "#16a6a0", "MARKUP": "#d65c67", "PULLBACK": "#d7b250",
    "OVERHEATED": "#e88a45", "DISTRIBUTION_RISK": "#a2668c",
    "FAILED": "#6a8497", "REBUILDING": "#729779",
}


def segments(timeline: pd.DataFrame) -> list[dict]:
    """Run-length stage spans; boundaries start on their actual first-known day."""
    spans = []
    for row in timeline.to_dict("records"):
        key = (row["cycle_id"], row["cycle_stage"])
        if spans and (spans[-1]["cycle_id"], spans[-1]["stage"]) == key:
            spans[-1]["end"] = row["date"]
            spans[-1]["sessions"] += 1
        else:
            spans.append({"start": row["date"], "end": row["date"], "sessions": 1,
                          "cycle_id": key[0], "stage": key[1], "label": row["cycle_stage_label"],
                          "color": STAGE_COLORS.get(key[1], "#8795a5"),
                          "reason": row["cycle_reason_label"]})
    return spans
