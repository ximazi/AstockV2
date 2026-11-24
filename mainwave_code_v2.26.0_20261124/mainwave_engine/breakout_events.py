"""Causal price-only breakout evidence; never changes production cycle states."""
from __future__ import annotations

import numpy as np
import pandas as pd

VERSION = "breakout_events_v2_20260910"
# Research placeholders in active sessions / frozen return-index ATR units.
LOOKBACK = 20
ACCEPT_CLOSES = 2
ACCEPT_TTL = 5
BREAK_BUFFER_ATR = 0.1
FAIL_BUFFER_ATR = 0.5
ADVANCE_ATR = 1.0
LABELS = {
    "NONE": "尚无突破事件", "ATTEMPT": "突破尝试，等待站稳",
    "ACCEPTED": "突破已接受，等待推进", "ADVANCING": "突破后已有价格推进",
    "REJECTED": "突破被拒绝", "EXPIRED": "突破尝试过期", "FAILED": "突破结构失效",
}


def annotate(frame: pd.DataFrame, timeline: pd.DataFrame) -> pd.DataFrame:
    scale = frame.return_index.to_numpy(float) / frame.close.to_numpy(float)
    high = frame.high.to_numpy(float) * scale
    low = frame.low.to_numpy(float) * scale
    close = frame.return_index.to_numpy(float)
    atr = frame.atr20.to_numpy(float)
    ceiling = pd.Series(high).rolling(LOOKBACK, min_periods=LOOKBACK).max().shift(1).to_numpy()
    floor = pd.Series(low).rolling(LOOKBACK, min_periods=LOOKBACK).min().shift(1).to_numpy()
    records, active = [], None
    for i, row in enumerate(timeline.to_dict("records")):
        date, event = row["date"], ""
        terminal = {"REJECTED", "EXPIRED", "FAILED"}
        if active and active["status"] not in terminal:
            trend_broken = (active["status"] in {"ACCEPTED", "ADVANCING"}
                            and i > 0 and close[i] < floor[i] - BREAK_BUFFER_ATR * atr[i - 1])
            if close[i] < active["invalid"] or trend_broken:
                active["status"] = "REJECTED" if active["status"] == "ATTEMPT" else "FAILED"
                event = active["status"]
            elif active["status"] == "ATTEMPT":
                active["holds"] = active["holds"] + 1 if close[i] > active["boundary"] else 0
                if i - active["start_index"] > ACCEPT_TTL:
                    active["status"], event = "EXPIRED", "EXPIRED"
                elif active["holds"] >= ACCEPT_CLOSES:
                    active.update(status="ACCEPTED", accepted_at=date, accepted_close=float(close[i]))
                    event = "ACCEPTED"
            elif active["status"] == "ACCEPTED" and close[i] >= active["accepted_close"] + ADVANCE_ATR * active["atr"]:
                active.update(status="ADVANCING", advanced_at=date)
                event = "ADVANCING"
        can_start = active is None or active["status"] in terminal
        if (can_start and not event and i >= LOOKBACK and np.isfinite(atr[i - 1]) and atr[i - 1] > 0
                and close[i] > ceiling[i] + BREAK_BUFFER_ATR * atr[i - 1]
                and close[i - 1] <= ceiling[i]):
            active = {"id": f"{row.get('symbol', '')}:breakout:{date}",
                      "status": "ATTEMPT", "attempt_at": date, "accepted_at": "", "advanced_at": "",
                      "boundary": float(ceiling[i]), "invalid": float(ceiling[i] - FAIL_BUFFER_ATR * atr[i - 1]),
                      "atr": float(atr[i - 1]), "start_index": i, "holds": 1}
            event = "ATTEMPT"
        a = active or {}
        status = a.get("status", "NONE")
        records.append({
            "breakout_id": a.get("id", ""), "breakout_status": status,
            "breakout_status_label": LABELS[status], "breakout_event": event,
            "breakout_attempt_at": a.get("attempt_at", ""), "breakout_accepted_at": a.get("accepted_at", ""),
            "breakout_advanced_at": a.get("advanced_at", ""), "breakout_boundary_index": a.get("boundary", np.nan),
            "breakout_invalid_index": a.get("invalid", np.nan), "breakout_frozen_atr": a.get("atr", np.nan),
            "breakout_version": VERSION,
        })
    return pd.concat([timeline.reset_index(drop=True), pd.DataFrame(records)], axis=1)
