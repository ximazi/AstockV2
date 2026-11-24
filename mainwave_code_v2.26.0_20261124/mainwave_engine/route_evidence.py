"""Shared causal price evidence for daily discovery and historical experiments.

These are observable proxies, not claims about institutional ownership. Theme
propagation requires separate dated chain evidence; price alone cannot prove it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import breakout_events as BE
from . import phase_events as PE
from . import preparation_track as PT

VERSION = "route_evidence_v2_20260910"
ROUTE_LABELS = {"R1": "明确个股点火反查", "R2": "个股先行准备", "R3": "紧链传播承接"}
# Fixed hypotheses before the broad historical run, not stock-specific fitted thresholds.
POLICIES = ("attempt", "accepted", "accepted_demand", "accepted_prepared",
            "accepted_bias2", "accepted_bias3", "accepted_bias4", "prepared_watch")


def build(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw.date)
    raw = raw.sort_values("date").drop_duplicates("date", keep="last")
    suspended = raw.loc[pd.to_numeric(raw.get("tradestatus", pd.Series(index=raw.index, dtype=float)), errors="coerce").eq(0)
                        & pd.to_numeric(raw.amount, errors="coerce").eq(0), "date"].dt.strftime("%Y-%m-%d").tolist()
    raw = raw[pd.to_numeric(raw.amount, errors="coerce").gt(0)].reset_index(drop=True)
    frame = PE.classify_frame(raw)
    if frame.empty:
        return frame
    identity = pd.DataFrame({"symbol": symbol, "date": frame.date.dt.strftime("%Y-%m-%d")})
    events = BE.annotate(frame, identity)
    frame = pd.concat([frame, events.drop(columns=["symbol", "date"])], axis=1)
    close, atr = frame.return_index, frame.atr20
    scale = close / frame.close
    for key in ("open", "high", "low"):
        frame[f"index_{key}"] = frame[key] * scale
    frame["index_close"] = close
    ma20 = close.rolling(20).mean()
    frame["bias_atr"] = (close - ma20) / atr
    frame["atr_pct"] = atr / close * 100
    frame["liquid"] = frame.amount.shift(1).rolling(20).median().ge(3e7)
    st = pd.to_numeric(frame.get("isST", pd.Series(np.nan, index=frame.index)), errors="coerce")
    frame["st_known"] = st.notna()
    frame["eligible"] = frame.liquid & st.eq(0) & atr.gt(0) & (frame.index >= 120)

    probe = frame.effort_result.eq("DEMAND_EXPANSION")
    # Preparation must predate the current session. All anchors expire, not infinite ffill.
    probe_base = close.shift(1).where(probe).shift(1).ffill(limit=20)
    probe_close = close.where(probe).shift(1).ffill(limit=20)
    probe_atr = atr.where(probe).shift(1).ffill(limit=20)
    frame["probe_at"] = frame.date.where(probe).shift(1).ffill(limit=20).dt.strftime("%Y-%m-%d").fillna("")
    frame["retained"] = (close.ge(probe_base + 0.5 * (probe_close - probe_base))
                         & close.ge(probe_close - 2 * probe_atr))
    turn = pd.to_numeric(frame.get("turnover", pd.Series(np.nan, index=frame.index)), errors="coerce").where(lambda s: s > 0)
    frame["turnover_relative"] = turn / turn.shift(1).rolling(20, min_periods=15).median()
    frame["turnover_contract"] = turn.rolling(5).median() / turn.shift(5).rolling(20).median()
    frame["amount_contract"] = frame.amount.rolling(5).median() / frame.amount.shift(5).rolling(20).median()
    width = frame.index_high - frame.index_low
    frame["range_contract"] = width.rolling(5).median() / width.shift(5).rolling(20).median()
    frame["supply"] = (frame.turnover_contract.le(0.85) & frame.amount_contract.le(0.85)
                       & frame.range_contract.le(0.85) & close.ge(frame.structure_floor)
                       & ma20.ge(ma20.shift(5)))
    frame["prepared"] = frame.retained & frame.supply & frame.bias_atr.between(-1, 2)
    prior_prepared = frame.prepared.shift(1, fill_value=False).rolling(30).max().eq(1)
    frame["prepared_at"] = frame.date.where(frame.prepared).shift(1).ffill(limit=30).dt.strftime("%Y-%m-%d").fillna("")
    at_attempt = frame.breakout_event.eq("ATTEMPT")
    frame["prepared_before_attempt"] = prior_prepared.astype(float).where(at_attempt).ffill().eq(1)
    frame["attempt_prepared_at"] = frame.prepared_at.astype("string").where(at_attempt).ffill().fillna("")
    frame["demand"] = (probe.rolling(3).sum().ge(1) & close.gt(close.shift(3)))
    stall = (frame.amount_ratio.ge(1.3) & frame.close_location.le(0.4)
             & (close.diff() <= 0.2 * atr))
    frame["supply_warning"] = frame.UTAD | frame.SOW | stall.rolling(5).sum().ge(2)
    frame["attempt"] = at_attempt
    frame["accepted"] = frame.breakout_event.eq("ACCEPTED")
    frame["accepted_demand"] = frame.accepted & frame.demand & ~frame.supply_warning
    frame["accepted_prepared"] = frame.accepted_demand & frame.prepared_before_attempt
    for n in (2, 3, 4):
        frame[f"accepted_bias{n}"] = frame.accepted_demand & frame.bias_atr.le(n)
    frame["prepared_watch"] = frame.prepared & ~frame.prepared.shift(1, fill_value=False)
    result = pd.concat([frame, PT.annotate(frame, symbol)], axis=1)
    result.attrs["suspended_days"] = suspended
    return result


def assess(row: dict, policy: str = "accepted_demand") -> dict:
    """Assess the current observation; policy is frozen separately after research."""
    if policy not in POLICIES or policy in {"attempt", "prepared_watch"}:
        raise ValueError(f"Not a confirmation policy: {policy}")
    accepted = row.get("breakout_status") in {"ACCEPTED", "ADVANCING"}
    risks = []
    if row.get("supply_warning", False):
        risks.append("SUPPLY_PRESSURE")
    if row.get("bias_atr", 0) > 3:
        risks.append("EXTENDED_NOT_DISTRIBUTION")
    passed = bool(row.get(policy, False))
    status = ("DATA_WAIT" if not row.get("eligible", False) else
              "RISK_WATCH" if "SUPPLY_PRESSURE" in risks else
              "WAIT_PRICE" if "EXTENDED_NOT_DISTRIBUTION" in risks else
              "CONFIRMATION_REVIEW" if passed else
              "PREPARATION_WATCH" if row.get("prepared", False) else
              "TREND_WATCH" if accepted else "WAIT_ACCEPTANCE")
    return {"version": VERSION, "policy": policy, "status": status,
            "risks": risks, "confirmation_pass": passed,
            "preparation_source": "CAUSAL_PRICE_PROXY", "orders_enabled": False}
