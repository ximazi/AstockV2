"""Continuous, point-in-time stock cycles. Observation only, not an entry signal.

Replay uses daily price observations plus dated production evidence. A stock stays
tracked when it leaves the candidate list. Historical replay is not a live hit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import phase_events as PE
from . import price_cycle_setups as PCS
from . import breakout_events as BE

VERSION = "stock_cycle_v1_3_20260910"
OUT_ROOT = Path("results/mainwave_engine")
# Observation placeholders reuse existing 20-session structure / volume scales.
# They have NOT been calibrated as new buy or sell conditions.
REBUILD_SESSIONS = 20
LABELS = {
    "WATCH": "结构观察", "PROBED": "资金试探", "ABSORBING": "承接蓄势",
    "COILED": "结构收敛", "ARMED": "临界准备", "READY": "等待触发",
    "IGNITION": "启动观察", "MARKUP": "上涨延续", "PULLBACK": "启动后整理",
    "OVERHEATED": "过热观察", "DISTRIBUTION_RISK": "派发风险观察",
    "FAILED": "本轮结构失效", "REBUILDING": "失效后修复",
}
TURNOVER_LABELS = {
    "MISSING": "换手数据不足", "NORMAL": "换手无明显异常",
    "ACTIVE_DEMAND": "换手增强且价格走强", "ACTIVE_SUPPORT": "换手增强、承接待验证",
    "HIGH_DISAGREEMENT": "高位换手放大、价格受阻",
    "POST_START_DRYING": "启动后回调换手收缩", "QUIET_UNCONFIRMED": "低换手、方向未确认",
    "HIGH_ACTIVITY_UNCONFIRMED": "高换手，方向待确认",
}
EARLY_STAGES = {"PROBED", "ABSORBING", "COILED", "ARMED", "READY"}
POST_START = {"IGNITION", "MARKUP", "PULLBACK", "OVERHEATED", "DISTRIBUTION_RISK"}
REASONS = {
    "NO_NEW_EVIDENCE": "无新增证据，延续上期观察",
    "NEW_STRUCTURE_AFTER_FAILURE": "失效后形成新结构并突破，开启新一轮",
    "WAIT_NEW_STRUCTURE_BREAKOUT": "已进入修复，等待新结构突破",
    "PRIOR_CYCLE_FAILED": "原结构已失效，尚未满足重建条件",
    "TREND_BROKEN": "原有上涨趋势转为下跌结构",
    "CLOSE_BELOW_FROZEN_STRUCTURE_FLOOR": "收盘跌破本轮冻结结构底线",
    "PRICE_VOLUME_DISTRIBUTION_WARNING": "量价出现派发风险，需复核",
    "WAIT_DEMAND_RECLAIM_OF_RISK_LEVEL": "风险尚未解除，等待需求恢复并收复压力位",
    "PRICE_VOLUME_CLIMAX": "出现量价过热特征",
    "TREND_CONTINUES_AFTER_START": "本轮启动后上涨结构延续",
    "POST_START_PAUSE_NOT_NEW_ACCUMULATION": "已启动后的整理，不重置为初次吸筹",
    "PRODUCTION_TRIGGER": "当日生产引擎记录触发",
    "PRICE_BREAKOUT_OBSERVATION": "历史量价突破观察，非生产命中证明",
    "TREND_ALREADY_VISIBLE": "首次识别时已见上涨，不代表捕捉到起点",
    "PRODUCTION_STRUCTURE_INVALIDATED": "当日生产记录结构失效",
    "DATED_PRODUCTION_EVIDENCE": "根据当日生产证据更新准备阶段",
    "PRODUCTION_EVIDENCE_EXPIRED": "生产准备证据已退化或过期",
}


def _read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _finite(value) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def turnover_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Percent units; baselines exclude today. Never infer fund ownership."""
    result = frame.copy()
    turn = pd.to_numeric(result.get("turnover", pd.Series(np.nan, index=result.index)),
                         errors="coerce").where(lambda x: x > 0)
    result["turnover_pct"] = turn
    baseline = turn.shift(1).rolling(20, min_periods=15).median()
    result["turnover_relative20"] = turn / baseline
    # Each percentile compares today only with the preceding available history.
    result["turnover_percentile120"] = turn.rolling(121, min_periods=41).apply(
        lambda x: np.nan if not np.isfinite(x[-1]) else
        float(np.mean(x[:-1][np.isfinite(x[:-1])] < x[-1]) +
              0.5 * np.mean(x[:-1][np.isfinite(x[:-1])] == x[-1])) * 100, raw=True)
    result["turnover_contraction5"] = turn.rolling(5, min_periods=5).median() / (
        turn.shift(5).rolling(20, min_periods=15).median())
    ret = result["return_index"].pct_change()
    up = turn.where(ret > 0).rolling(20, min_periods=3).median()
    down = turn.where(ret < 0).rolling(20, min_periods=3).median()
    result["turnover_down_up20"] = down / up
    result["turnover_status"] = np.where(turn.notna() & baseline.notna(), "AVAILABLE", "MISSING")
    return result


def replay(symbol: str, classified: pd.DataFrame,
           trajectory: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pure prefix-stable replay; events from later dates cannot alter old states."""
    frame = turnover_frame(classified)
    price_events = None
    if {"open", "high", "low", "close", "atr20"}.issubset(frame.columns):
        identity = pd.DataFrame({"symbol": symbol,
                                 "date": pd.to_datetime(frame.date).dt.strftime("%Y-%m-%d")})
        price_events = BE.annotate(frame, identity).to_dict("records")
    evidence = {}
    if trajectory is not None and len(trajectory):
        trace = trajectory[trajectory["symbol"].eq(symbol)].copy()
        trace["date"] = pd.to_datetime(trace["date"]).dt.strftime("%Y-%m-%d")
        evidence = trace.drop_duplicates("date", keep="last").set_index("date").to_dict("index")
    stage, cycle, start, invalid, failed_index = "WATCH", 0, "", None, None
    cycle_id, stage_since, risk_high = "", "", None
    start_source, production_trigger_at = "", ""
    rows = []
    for index, row in enumerate(frame.to_dict("records")):
        date = str(pd.Timestamp(row["date"]).date())
        ev = evidence.get(date, {})
        fresh = str(ev.get("data_fresh", True)).lower() in {"true", "1", "1.0"}
        production = str(ev.get("state", "")) if fresh else ""
        capital = str(ev.get("capital_stage", "")) if fresh else ""
        phase, effort = row["broad_phase"], row["effort_result"]
        close, floor = float(row["return_index"]), _finite(row.get("structure_floor"))
        previous = stage
        reason = "NO_NEW_EVIDENCE"
        trigger = production == "TRIGGERED" or bool(row.get("SOS", False))
        trend_seen = phase == "MARKUP"
        raw_event = price_events[index] if price_events is not None else {}
        repair_visible = (failed_index is not None and index - failed_index > REBUILD_SESSIONS
                          and floor is not None and close > floor)
        if failed_index is not None:
            failed_date = str(pd.Timestamp(frame.iloc[failed_index]["date"]).date())
            new_attempt = (raw_event.get("breakout_attempt_at", "") > failed_date
                           and raw_event.get("breakout_status") in {"ATTEMPT", "ACCEPTED", "ADVANCING"})
            # Raw acceptance is independent of the failed parent; no retroactive restart.
            if production == "TRIGGERED" or (new_attempt and raw_event.get("breakout_event") == "ACCEPTED"):
                stage, reason = "IGNITION", "NEW_STRUCTURE_AFTER_FAILURE"
                failed_index = None
                start = ""
            else:
                stage = "REBUILDING" if repair_visible or new_attempt else "FAILED"
                reason = "WAIT_NEW_STRUCTURE_BREAKOUT" if repair_visible or new_attempt else "PRIOR_CYCLE_FAILED"
        elif stage in POST_START:
            if (invalid is not None and close < invalid) or phase == "MARKDOWN":
                stage = "FAILED"
                reason = "TREND_BROKEN" if phase == "MARKDOWN" else "CLOSE_BELOW_FROZEN_STRUCTURE_FLOOR"
                failed_index = index
            elif phase == "DISTRIBUTION" or bool(row.get("UTAD", False)):
                stage, reason = "DISTRIBUTION_RISK", "PRICE_VOLUME_DISTRIBUTION_WARNING"
                risk_high = max(risk_high or close, close,
                                _finite(row.get("structure_ceiling")) or close)
            elif stage == "DISTRIBUTION_RISK" and not (
                    effort == "DEMAND_EXPANSION" and close > (risk_high or close)):
                reason = "WAIT_DEMAND_RECLAIM_OF_RISK_LEVEL"
            elif phase == "CLIMAX":
                stage, reason = "OVERHEATED", "PRICE_VOLUME_CLIMAX"
            elif trend_seen or trigger:
                stage, reason = "MARKUP", "TREND_CONTINUES_AFTER_START"
            else:
                stage, reason = "PULLBACK", "POST_START_PAUSE_NOT_NEW_ACCUMULATION"
        elif trigger or trend_seen:
            stage = "IGNITION" if trigger else "MARKUP"
            reason = "PRODUCTION_TRIGGER" if production == "TRIGGERED" else (
                "PRICE_BREAKOUT_OBSERVATION" if trigger else "TREND_ALREADY_VISIBLE")
        elif production in {"FAILED", "INVALID"}:
            stage, reason = "FAILED", "PRODUCTION_STRUCTURE_INVALIDATED"
            failed_index = index
        elif production in {"ARMED", "READY"} or capital in EARLY_STAGES:
            proposed = production if production in {"ARMED", "READY"} else capital
            stage, reason = proposed, "DATED_PRODUCTION_EVIDENCE"
        elif production in {"DEGRADED", "OUT"}:
            stage, reason = "WATCH", "PRODUCTION_EVIDENCE_EXPIRED"

        if stage in POST_START and not start:
            cycle += 1
            cycle_id, start = f"{symbol}:{date}", date
            invalid = floor
            risk_high = None
            start_source = "PRODUCTION" if production == "TRIGGERED" else "PRICE_RECONSTRUCTION"
            production_trigger_at = ""
        if production == "TRIGGERED" and stage in POST_START and start and not production_trigger_at:
            production_trigger_at = date
        if stage != previous or not stage_since:
            stage_since = date

        turnover_context = "NORMAL"
        relative, contraction = _finite(row["turnover_relative20"]), _finite(row["turnover_contraction5"])
        if row["turnover_status"] == "MISSING":
            turnover_context = "MISSING"
        elif relative is not None and relative >= 1.3:
            turnover_context = "HIGH_ACTIVITY_UNCONFIRMED"
            if phase == "DISTRIBUTION" or stage == "DISTRIBUTION_RISK":
                turnover_context = "HIGH_DISAGREEMENT"
            elif effort == "DEMAND_EXPANSION":
                turnover_context = "ACTIVE_DEMAND"
            elif phase in {"ACCUMULATION", "REACCUMULATION"} and row["close_location"] >= 0.55:
                turnover_context = "ACTIVE_SUPPORT"
        elif contraction is not None and contraction <= 0.7:
            turnover_context = "POST_START_DRYING" if stage == "PULLBACK" else "QUIET_UNCONFIRMED"
        rows.append({
            "symbol": symbol, "date": date, "cycle_stage": stage,
            "cycle_stage_label": LABELS[stage], "cycle_id": cycle_id,
            "cycle_number": cycle, "cycle_started_at": start,
            "cycle_start_source": start_source, "production_trigger_at": production_trigger_at,
            "cycle_stage_since": stage_since, "cycle_previous_stage": previous,
            "cycle_reason": reason, "cycle_changed": stage != previous,
            "cycle_reason_label": REASONS[reason],
            "cycle_invalid_index": invalid, "cycle_validation": "UNVALIDATED_OBSERVATION",
            "daily_phase": phase, "production_state": production,
            "turnover_context": turnover_context,
            "turnover_context_label": TURNOVER_LABELS[turnover_context],
            **{key: row[key] for key in row if key.startswith("turnover_")},
            **{key: value for key, value in raw_event.items() if key.startswith("breakout_")},
        })
    return PCS.annotate(frame, pd.DataFrame(rows))


def analyze(symbol: str, as_of: str, data_root: Path = Path("data"),
            trajectory: pd.DataFrame | None = None) -> tuple[dict, pd.DataFrame]:
    raw = _read(data_root / f"{symbol}_daily.csv")
    if raw.empty:
        return {}, pd.DataFrame()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw[raw["date"] <= pd.Timestamp(as_of)].sort_values("date").drop_duplicates("date", keep="last")
    raw = raw[pd.to_numeric(raw["amount"], errors="coerce") > 0]
    classified = PE.classify_frame(raw)
    if classified.empty:
        return {}, pd.DataFrame()
    timeline = replay(symbol, classified, trajectory)
    snapshot = json.loads(timeline.tail(1).to_json(orient="records"))[0]
    snapshot.update({"as_of": as_of, "cycle_version": VERSION,
                     "cycle_data_status": "FRESH" if snapshot["date"] == as_of else "STALE",
                     "history_from": str(classified["date"].iloc[0].date())})
    if snapshot["cycle_invalid_index"] is not None:
        snapshot["cycle_invalid_price"] = (snapshot["cycle_invalid_index"] /
                                           classified["return_index"].iloc[-1] * classified["close"].iloc[-1])
    return snapshot, timeline


def write_daily(as_of: str, candidates: pd.DataFrame, output: Path,
                root: Path = OUT_ROOT, data_root: Path = Path("data")) -> pd.DataFrame:
    """Update current AND previously selected stocks, without a mutable future store."""
    prior_paths = sorted(path for path in root.glob("20??????/stock_lifecycle.csv")
                         if path.parent.name <= as_of.replace("-", ""))
    prior = _read(prior_paths[-1]) if prior_paths else pd.DataFrame()
    previous = prior.set_index("symbol").to_dict("index") if len(prior) else {}
    current = set(candidates.get("symbol", pd.Series(dtype=str)).dropna())
    symbols = sorted(current | set(previous))
    trajectory = _read(root / "trajectory_log.csv")
    snapshots, events, setups = [], [], []
    for index, symbol in enumerate(symbols, 1):
        if index == 1 or index % 25 == 0 or index == len(symbols):
            print(f"[stock_lifecycle] [{index}/{len(symbols)}] {symbol}", flush=True)
        snapshot, timeline = analyze(symbol, as_of, data_root, trajectory)
        if not snapshot:
            snapshot = {**previous.get(symbol, {}), "symbol": symbol, "as_of": as_of,
                        "cycle_data_status": "MISSING"}
        snapshot.update({"in_current_pool": symbol in current,
                         "tracking_since": previous.get(symbol, {}).get("tracking_since", as_of)})
        snapshots.append(snapshot)
        if len(timeline):
            changes = timeline[timeline["cycle_changed"]].copy()
            changes["record_source"] = "POINT_IN_TIME_RECONSTRUCTION"
            events.append(changes)
            setups.append(PCS.setup_history(timeline))
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(snapshots)
    frame.to_csv(output / "stock_lifecycle.csv", index=False, encoding="utf-8-sig")
    (pd.concat(setups, ignore_index=True) if setups else pd.DataFrame()).to_csv(
        output / "price_cycle_setups.csv", index=False, encoding="utf-8-sig")
    (pd.concat(events, ignore_index=True) if events else pd.DataFrame()).to_csv(
        output / "stock_lifecycle_events.csv", index=False, encoding="utf-8-sig")
    shown = [key for key in ["symbol", "in_current_pool", "cycle_stage_label", "cycle_started_at",
                            "cycle_stage_since", "cycle_reason_label", "turnover_pct",
                            "turnover_relative20", "turnover_context_label", "speed_label",
                            "setup_label", "setup_status_label", "cycle_data_status"]
             if key in frame]
    lines = [f"# 个股连续生命周期 / {as_of}", "", f"版本: {VERSION}", "",
             "每日延续跟踪，今日落选不删除历史。量价观察不是买卖指令，换手率不证明资金身份。",
             "历史量价重建不等于历史生产命中。连续阶段与换手解释的阈值均为占位待校准。", "",
             frame[shown].to_markdown(index=False) if len(frame) else "No tracked stocks."]
    (output / "stock_lifecycle.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[stock_lifecycle] tracked={len(frame)} current={len(current)} -> {output / 'stock_lifecycle.md'}")
    if candidates.empty or frame.empty:
        return candidates
    fields = [key for key in frame if key.startswith(("cycle_", "turnover_", "setup_", "speed_", "breakout_")) or key == "production_trigger_at"]
    return candidates.drop(columns=fields, errors="ignore").merge(
        frame[["symbol", *fields]], on="symbol", how="left", validate="one_to_one")
