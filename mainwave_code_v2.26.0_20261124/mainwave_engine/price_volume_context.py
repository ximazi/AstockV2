"""Causal, read-only price/volume observations. Never grants trading permission."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "price_volume_shadow_v1_20260912"
RULES = {
    "baseline_sessions": 20, "min_baseline": 15, "liquidity_amount": 30_000_000,
    "contest_activity": 1.3, "contest_body_atr": .3, "contest_progress_atr": .3,
    "up_progress_atr": .3, "up_close_location": .7, "activity_low": .6,
    "activity_high": 1.1, "gap_atr_max": .75, "contest_window": 10,
    "contest_count": 2, "episode_ttl": 30, "mature_sessions": 6,
    "mature_progress_atr": 3., "local_activity_jump": 1.5,
}
LABELS = {"UNRESOLVED": "证据不足/无明确变化", "CONTESTED": "高量低推进分歧",
          "EFFICIENT_UP": "低量有效向上推进", "LOW_LIQUIDITY_MOVE": "流动性/价格限制异常",
          "SUPPLY_PRESSURE": "关键位置供给压力", "EFFICIENT_DOWN": "低量有效下跌"}
COLORS = {"UNRESOLVED": "#dce1e5", "CONTESTED": "#b98516", "EFFICIENT_UP": "#158477",
          "LOW_LIQUIDITY_MOVE": "#687583", "SUPPLY_PRESSURE": "#bb4560", "EFFICIENT_DOWN": "#4769b0"}
EVENTS = {"CONTEST_TO_UP": "分歧后有效推进", "UP_TO_CONTEST": "推进后再现分歧",
          "UP_RECONFIRMED": "分歧后价格再确认"}
NOTICE = "只读量价代理，参数待校准；不证明资金身份，不改变生命周期、排序或买卖许可。"


def _number(frame, key):
    return pd.to_numeric(frame.get(key, pd.Series(np.nan, index=frame.index)), errors="coerce")


def annotate(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Consume the existing classified/price-breakout frame, without mutating it."""
    if frame.empty:
        return pd.DataFrame()
    f = frame.reset_index(drop=True)
    close = _number(f, "return_index")
    scale = close / _number(f, "close").replace(0, np.nan)
    high, low, op = (_number(f, k) * scale for k in ("high", "low", "open"))
    atr = _number(f, "atr20").where(lambda v: v > 0)
    amount, volume, turnover = (_number(f, k).where(lambda v: v > 0) for k in ("amount", "volume", "turnover"))
    baseline = lambda v: v.shift(1).rolling(RULES["baseline_sessions"], min_periods=RULES["min_baseline"]).median()
    amount_base, volume_base, turnover_base = (baseline(v) for v in (amount, volume, turnover))
    va, ta = volume / volume_base, turnover / turnover_base
    activity = ta.combine_first(va)
    source = pd.Series(np.where(ta.notna(), "TURNOVER", np.where(va.notna(), "VOLUME", "MISSING")))
    cloc = (close - low) / (high - low).replace(0, np.nan)
    progress, body, gap = (close - close.shift()) / atr, (close - op) / atr, (op - close.shift()) / atr
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    metrics = pd.DataFrame({"date": pd.to_datetime(f.date).dt.strftime("%Y-%m-%d"),
        "progress_atr": progress, "body_atr": body, "gap_atr": gap, "close_location": cloc,
        "relative_turnover": ta, "relative_volume": va, "relative_amount": amount / amount_base,
        "activity": activity, "activity_source": source, "amount_baseline": amount_base,
        "volume_baseline": volume_base, "turnover_baseline": turnover_base,
        "local_activity_ratio": activity / activity.shift().rolling(5, min_periods=3).median(),
        "progress_3d_atr": (close - close.shift(3)) / atr,
        "path_efficiency_5d": (close - close.shift(5)).abs() / tr.rolling(5).sum().replace(0, np.nan),
        "upper_wick": (high - pd.concat([op, close], axis=1).max(axis=1)) / (high - low).replace(0, np.nan),
        "close_index": close, "high_index": high, "low_index": low, "atr_index": atr})
    # Share volume / turnover approximates float only for detecting denominator changes.
    float_proxy = volume / turnover
    float_jump = float_proxy.pct_change(fill_method=None).abs() > .1
    rows, episode = [], None
    date_indices = dict(zip(metrics.date, range(len(metrics))))
    terminal = {"FAILED", "REJECTED", "EXPIRED"}
    anchor_columns = [k for k in f if k.startswith("breakout_") or k in {"structure_floor", "structure_ceiling"}]
    for i, (m, r) in enumerate(zip(metrics.to_dict("records"), f[anchor_columns].to_dict("records"))):
        day = m["date"]
        parent = str(r.get("breakout_id") or "")
        status = str(r.get("breakout_status", "NONE"))
        boundary = r.get("breakout_boundary_index", np.nan)
        invalid = r.get("breakout_invalid_index", np.nan)
        gaps, reason, state, event = [], "NO_COMBINATION", "UNRESOLVED", ""
        if not np.isfinite(m["relative_turnover"]):
            gaps.append("TURNOVER_MISSING")
        if not np.isfinite(m["relative_volume"]):
            gaps.append("VOLUME_MISSING")
        quality_ok = all(np.isfinite(m[k]) for k in ("atr_index", "progress_atr", "activity", "amount_baseline"))
        quality_ok = quality_ok and all(np.isfinite(v) and v > 0 for v in (op.iloc[i], high.iloc[i], low.iloc[i], close.iloc[i]))
        quality_ok = quality_ok and high.iloc[i] >= max(op.iloc[i], close.iloc[i]) and low.iloc[i] <= min(op.iloc[i], close.iloc[i])
        if not quality_ok or not np.isfinite(amount.iloc[i]):
            reason = "MISSING_OR_INVALID_BASELINE"
        elif float_jump.iloc[i]:
            reason = "FLOAT_DENOMINATOR_CHANGE"
            quality_ok = False
            gaps.append(reason)
        elif (m["amount_baseline"] < RULES["liquidity_amount"] or amount.iloc[i] < RULES["liquidity_amount"]
              or high.iloc[i] == low.iloc[i] or abs(m["gap_atr"]) > RULES["gap_atr_max"]):
            state, reason = "LOW_LIQUIDITY_MOVE", "ILLIQUID_LOCKED_OR_GAP"
        elif m["activity"] >= RULES["contest_activity"] and abs(m["body_atr"]) <= RULES["contest_body_atr"] and abs(m["progress_atr"]) <= RULES["contest_progress_atr"]:
            state, reason = "CONTESTED", "ACTIVITY_WITHOUT_NET_PROGRESS"
        elif (RULES["activity_low"] <= m["activity"] <= RULES["activity_high"] and m["progress_atr"] >= RULES["up_progress_atr"]
              and m["body_atr"] > 0 and m["close_location"] >= RULES["up_close_location"]
              and status in {"ATTEMPT", "ACCEPTED", "ADVANCING"} and close.iloc[i] > boundary):
            state, reason = "EFFICIENT_UP", "ABOVE_FROZEN_LEVEL_WITH_LOW_ACTIVITY"
        elif m["activity"] <= RULES["activity_high"] and m["progress_atr"] <= -RULES["up_progress_atr"] and m["body_atr"] < 0 and m["close_location"] <= .3:
            state, reason = "EFFICIENT_DOWN", "LOW_ACTIVITY_DOWN_NOT_A_BOTTOM"
        attempt_at = str(r.get("breakout_attempt_at") or "")[:10]
        attempt_age = i - date_indices.get(attempt_at, i)
        mature = (status in {"ACCEPTED", "ADVANCING"} and attempt_age >= RULES["mature_sessions"]
                  and np.isfinite(boundary) and i > 0
                  and high.iloc[max(0, i - 20):i].max() >= boundary + RULES["mature_progress_atr"] * atr.iloc[i])
        near_level = np.isfinite(boundary) and abs(high.iloc[i] - boundary) <= atr.iloc[i]
        pressure = (quality_ok and state != "LOW_LIQUIDITY_MOVE" and (mature or near_level)
                    and m["activity"] >= RULES["contest_activity"]
                    and m["local_activity_ratio"] >= RULES["local_activity_jump"]
                    and (m["close_location"] <= .4 or m["upper_wick"] >= .45))
        if pressure:
            state, reason = "SUPPLY_PRESSURE", "LOCAL_ACTIVITY_JUMP_WEAK_RESULT"
        ended = ""
        if episode:
            if i - episode["start"] > RULES["episode_ttl"]:
                ended = "TTL_EXPIRED"
            elif not quality_ok or state == "LOW_LIQUIDITY_MOVE":
                ended = "DATA_OR_LIQUIDITY_BREAK"
            elif close.iloc[i] < episode["floor"] or (episode["parent"] and status in terminal):
                ended = "STRUCTURE_FAILED"
            elif parent != episode["parent"] and parent and status not in terminal:
                # A pre-breakout range can bind forward once, never across two breakouts.
                if not episode["parent"] and abs(boundary - episode["ceiling"]) <= .5 * atr.iloc[i]:
                    episode.update(parent=parent, floor=float(invalid), ceiling=float(boundary))
                else:
                    ended = "NEW_BREAKOUT"
        old = episode if ended else None
        if ended:
            episode = None
        if not ended and not episode and state in {"CONTESTED", "EFFICIENT_UP", "SUPPLY_PRESSURE"}:
            floor = invalid if status in {"ATTEMPT", "ACCEPTED", "ADVANCING"} else r.get("structure_floor", np.nan)
            ceiling = boundary if status in {"ATTEMPT", "ACCEPTED", "ADVANCING"} else r.get("structure_ceiling", np.nan)
            if np.isfinite(floor) and np.isfinite(ceiling):
                episode = {"id": f"{symbol}:pv:{day}", "start": i, "first_seen": day,
                           "parent": parent if status not in terminal and status != "NONE" else "",
                           "floor": float(floor), "ceiling": float(ceiling), "contests": [],
                           "up_at": "", "risk_high": None, "reclaims": 0, "last_event": "", "last_event_at": ""}
        if episode:
            accepted = status in {"ACCEPTED", "ADVANCING"} and parent == episode["parent"]
            prior_contests = [j for j in episode["contests"] if i - j <= RULES["contest_window"]]
            if episode["risk_high"] is not None:
                held = close.iloc[i] > episode["risk_high"] and i > 0 and low.iloc[i] >= low.iloc[i - 1]
                episode["reclaims"] = episode["reclaims"] + 1 if held else 0
            if state == "EFFICIENT_UP" and accepted:
                if episode["risk_high"] is not None and episode["reclaims"] >= 2:
                    event = "UP_RECONFIRMED"
                    episode.update(risk_high=None, reclaims=0)
                elif not episode["up_at"] and len(prior_contests) >= RULES["contest_count"]:
                    event = "CONTEST_TO_UP"
                episode["up_at"] = episode["up_at"] or day
            elif state in {"CONTESTED", "SUPPLY_PRESSURE"}:
                if episode["up_at"] and (mature or near_level) and episode["risk_high"] is None:
                    event = "UP_TO_CONTEST"
                    episode["risk_high"] = float(high.iloc[i])
                # Restrict preparation to the original frozen range, not unrelated high volume.
                if low.iloc[i] <= episode["ceiling"] + .5 * atr.iloc[i] and high.iloc[i] >= episode["floor"]:
                    episode["contests"].append(i)
            if event:
                episode.update(last_event=event, last_event_at=day)
        a = episode or old or {}
        rows.append({**m, "state": state, "label": LABELS[state], "reason": reason, "missing": "|".join(gaps),
            "event": event, "event_label": EVENTS.get(event, ""), "known_at": day, "known_after": "SESSION_CLOSE",
            "episode_id": a.get("id", ""), "parent_breakout_id": a.get("parent", ""),
            "first_seen": a.get("first_seen", ""), "age_sessions": i - a["start"] if a else None,
            "expires_after_sessions": RULES["episode_ttl"], "invalid_reason": ended, "episode_valid": bool(episode),
            "latest_event": a.get("last_event", ""), "latest_event_at": a.get("last_event_at", ""),
            "boundary_index": a.get("ceiling"), "invalid_index": a.get("floor"),
            "pressure_high_index": a.get("risk_high"), "mature_price_context": bool(mature),
            "alternative_demand_evidence": bool(event in {"CONTEST_TO_UP", "UP_RECONFIRMED"}),
            "baseline_end": str(metrics.iloc[i - 1].date) if i else "",
            "rule_version": VERSION, "orders_enabled": False})
    return pd.DataFrame(rows)


def report(frame: pd.DataFrame, symbol: str) -> dict:
    timeline = json.loads(annotate(frame, symbol).to_json(orient="records", force_ascii=False))
    return {"version": VERSION, "mode": "SHADOW_ONLY", "orders_enabled": False,
            "rules": RULES, "labels": LABELS, "colors": COLORS, "event_labels": EVENTS, "notice": NOTICE,
            "latest": timeline[-1] if timeline else {}, "timeline": timeline,
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def case_attachment(case: dict) -> dict:
    """Rebuild only from frozen raw bars; never read today's prices or rewrite the case."""
    from . import phase_events as PE, breakout_events as BE
    raw = pd.DataFrame(case.get("raw_history", []))
    required = {"date", "open", "high", "low", "close", "amount"}
    frame = pd.DataFrame()
    if required.issubset(raw.columns):
        raw["date"] = pd.to_datetime(raw.date)
        raw = raw[raw.date.le(case["summary"]["window_end"])].sort_values("date").drop_duplicates("date", keep="last")
        raw = raw[pd.to_numeric(raw.amount, errors="coerce") > 0].reset_index(drop=True)
        frame = PE.classify_frame(raw)
        if not frame.empty:
            anchors = BE.annotate(frame, pd.DataFrame({"date": frame.date.dt.strftime("%Y-%m-%d"), "symbol": case["summary"]["symbol"]}))
            frame = pd.concat([frame, anchors.drop(columns=["date", "symbol"])], axis=1)
    result = report(frame, case["summary"]["symbol"])
    result.update(source_case_hash=case["case_hash"], source="FROZEN_RAW_HISTORY_CURRENT_OBSERVER",
                  source_notice="当前观察规则对冻结行情的重建，非当时生产命中；原交易与人工复盘未改写。",
                  dependency_versions={"phase_events": PE.VERSION, "breakout_events": BE.VERSION},
                  dependency_sha256={Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                                     for m in (PE, BE)})
    return result
