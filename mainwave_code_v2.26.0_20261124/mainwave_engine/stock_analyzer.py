# -*- coding: utf-8 -*-
"""Single-stock full-cycle analysis backed by daily main-wave artifacts."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__ as ENGINE_VERSION
from . import candidate_value as CV
from . import phase_events as PE
from . import stock_lifecycle as SL
from . import price_cycle_setups as PCS


VERSION = "single_stock_lifecycle_analysis_v10_20261029"
OUT_ROOT = Path("results/mainwave_engine")
LIFECYCLE = [
    "PROBED", "ABSORBING", "COILED", "ARMED", "READY", "IGNITION",
    "ACCELERATION", "OVERHEATED", "DISTRIBUTION",
]
LIFECYCLE_LABELS = {
    "PROBED": "资金试探", "ABSORBING": "吸收筹码", "COILED": "结构收敛",
    "ARMED": "临界准备", "READY": "等待触发", "IGNITION": "启动确认",
    "ACCELERATION": "主升加速", "OVERHEATED": "过热", "DISTRIBUTION": "派发/破位",
}


def normalize_symbol(value: str) -> str:
    text = str(value or "").strip().lower().replace(".", "_").replace("-", "_")
    match = re.search(r"(\d{6})", text)
    if not match:
        raise ValueError("股票代码必须包含6位数字")
    code = match.group(1)
    if text.startswith(("sh_", "sz_", "bj_")):
        return text[:2] + "_" + code
    if code.startswith(("60", "68")):
        return "sh_" + code
    if code.startswith(("00", "30")):
        return "sz_" + code
    if code.startswith(("43", "83", "87", "92")):
        return "bj_" + code
    raise ValueError(f"无法根据代码 {code} 判断交易所")


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _number(value, default=None):
    try:
        number = float(value)
        return number if pd.notna(number) else default
    except (TypeError, ValueError):
        return default


def latest_report_date(on_or_before: str = "") -> str:
    cutoff = pd.Timestamp(on_or_before) if on_or_before else pd.Timestamp.max
    dates = []
    for path in OUT_ROOT.glob("20??????"):
        if not (path / "mainwave_weekly_candidates.csv").exists():
            continue
        try:
            date = pd.Timestamp(path.name)
        except ValueError:
            continue
        if date <= cutoff:
            dates.append(date)
    if not dates:
        raise FileNotFoundError("没有可用的 mainwave_engine 日报输出")
    return str(max(dates).date())


def _mapping(symbol: str, as_of: str = "") -> dict:
    payload = _read_json(Path("concept_mapping.json"), {})
    mapping = payload.get("mapping", {})
    if as_of:
        from .theme_semantics import apply_membership_evidence
        mapping, _ = apply_membership_evidence(mapping, as_of)
    return mapping.get(symbol, {})


def _row_for(frame: pd.DataFrame, symbol: str) -> dict:
    if frame.empty or "symbol" not in frame.columns:
        return {}
    found = frame[frame["symbol"].astype(str).eq(symbol)]
    return found.iloc[0].to_dict() if len(found) else {}


def _data_frame(symbol: str, as_of: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path("data") / f"{symbol}_daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少行情文件：{path}")
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw[raw["date"] <= pd.Timestamp(as_of)].sort_values("date").reset_index(drop=True)
    raw = raw.drop_duplicates("date", keep="last")
    raw = raw[pd.to_numeric(raw["amount"], errors="coerce") > 0].reset_index(drop=True)
    classified = PE.classify_frame(raw)
    if classified.empty:
        raise ValueError(f"{symbol} 有效行情不足65个交易日")
    return raw, classified


def _adjusted_chart(classified: pd.DataFrame, sessions: int = 180) -> list[dict]:
    frame = classified.copy()
    raw_close = pd.to_numeric(frame["close"], errors="coerce")
    anchor = float(raw_close.iloc[-1] / frame["return_index"].iloc[-1])
    scale = frame["return_index"] / raw_close * anchor
    frame["adj_open"] = frame["open"] * scale
    frame["adj_high"] = frame["high"] * scale
    frame["adj_low"] = frame["low"] * scale
    frame["adj_close"] = frame["return_index"] * anchor
    frame["ema20"] = frame["adj_close"].ewm(span=20, adjust=False).mean()
    frame["ema60"] = frame["adj_close"].ewm(span=60, adjust=False).mean()
    frame["ema10"] = frame["adj_close"].ewm(span=10, adjust=False).mean()
    # MACD(12,26,9) on the whole adjusted history, so the first displayed bars are not warm-up values
    frame["dif"] = frame["adj_close"].ewm(span=12, adjust=False).mean() - frame["adj_close"].ewm(span=26, adjust=False).mean()
    frame["dea"] = frame["dif"].ewm(span=9, adjust=False).mean()
    rows = []
    for row in frame.tail(sessions).to_dict("records"):
        rows.append({
            "date": str(pd.Timestamp(row["date"]).date()),
            "open": round(float(row["adj_open"]), 3),
            "high": round(float(row["adj_high"]), 3),
            "low": round(float(row["adj_low"]), 3),
            "close": round(float(row["adj_close"]), 3),
            "ema20": round(float(row["ema20"]), 3),
            "ema10": round(float(row["ema10"]), 3),
            "ema60": round(float(row["ema60"]), 3),
            "dif": round(float(row["dif"]), 4),
            "dea": round(float(row["dea"]), 4),
            "macd": round(float(2 * (row["dif"] - row["dea"])), 4),
            "amount": round(float(row.get("amount", 0) or 0) / 1e8, 2),
            "phase": str(row.get("broad_phase", "")),
        })
    return rows


def _event_history(classified: pd.DataFrame, sessions: int = 180) -> list[dict]:
    events = []
    validation = PE.load_validation().get("events", {})
    for row in classified.tail(sessions).to_dict("records"):
        for event in PE.EVENT_COLUMNS:
            if bool(row.get(event)):
                events.append({
                    "date": str(pd.Timestamp(row["date"]).date()),
                    "event": event,
                    "label": PE.EVENT_LABELS.get(event, event),
                    "validation": validation.get(event, {}).get("status", "UNVALIDATED"),
                    "price": round(float(row["close"]), 3),
                })
    return events[-30:]


def _stage_lifecycle(stage: dict, carrier: dict, snapshot: dict) -> str:
    for value in (carrier.get("shadow_stage"), carrier.get("production_state")):
        if str(value) in LIFECYCLE:
            return str(value)
    mapped = {
        "S0_base_watch": "PROBED", "S1_ignition_entry": "IGNITION",
        "S2_acceleration_caution": "ACCELERATION", "S3_overheated_no_chase": "OVERHEATED",
        "S4_breakdown_or_distribution": "DISTRIBUTION",
    }.get(str(stage.get("mainwave_stage", "")))
    if mapped:
        return mapped
    return {
        "ACCUMULATION": "PROBED", "REACCUMULATION": "COILED", "MARKUP": "IGNITION",
        "CLIMAX": "OVERHEATED", "DISTRIBUTION": "DISTRIBUTION",
        "MARKDOWN": "DISTRIBUTION", "TRANSITION": "PROBED",
    }.get(str(snapshot.get("broad_phase", "")), "PROBED")


def _fallback_candidate(symbol: str, as_of: str, mapping: dict, stage: dict,
                        carrier: dict, snapshot: dict, raw: pd.DataFrame,
                        lifecycle: str, market_state: str) -> dict:
    close = float(raw["close"].iloc[-1])
    previous_high = float(raw["high"].shift(1).tail(20).max())
    previous_low = float(raw["low"].shift(1).tail(10).min())
    trigger = _number(carrier.get("trigger_price"), previous_high)
    invalid = _number(carrier.get("invalid_price"), previous_low)
    r5 = (close / float(raw["close"].iloc[-6]) - 1) * 100 if len(raw) >= 6 else None
    r20 = (close / float(raw["close"].iloc[-21]) - 1) * 100 if len(raw) >= 21 else None
    ema20 = float(raw["close"].ewm(span=20, adjust=False).mean().iloc[-1])
    theme = str(stage.get("external_theme", "") or mapping.get("industry", "") or "未确认")
    action = "MARKET_RISK_WATCH_ONLY" if market_state != "GREEN" else "FORWARD_WATCH"
    return {
        "rank": None, "as_of": as_of, "symbol": symbol,
        "name": mapping.get("name", stage.get("name", "")), "close": close,
        "ema20_bias": (close / ema20 - 1) * 100, "r5": r5, "r20": r20,
        "belonging_theme": theme, "theme_source": "SUPPLEMENTAL_SINGLE_STOCK",
        "discovery_path": "NOT_IN_DAILY_POOL", "discovery_path_label": "未进入当日核心池",
        "stock_lifecycle": lifecycle, "stock_lifecycle_label": LIFECYCLE_LABELS[lifecycle],
        "theme_lifecycle": "", "theme_lifecycle_label": "未确认",
        "trigger_distance_pct": (trigger / close - 1) * 100 if close else None,
        "stop_distance_pct": (close / invalid - 1) * 100 if invalid else None,
        "trigger_price": trigger, "invalid_price": invalid, "action": action,
        "broad_phase": snapshot.get("broad_phase", ""),
        "broad_phase_label": snapshot.get("broad_phase_label", ""),
        "phase_validation": snapshot.get("phase_validation", ""),
        "latest_event": snapshot.get("latest_event", ""),
        "latest_event_label": snapshot.get("latest_event_label", ""),
        "latest_event_date": snapshot.get("latest_event_date", ""),
        "event_validation": snapshot.get("event_validation", ""),
        "effort_result": snapshot.get("effort_result", ""),
        "effort_result_label": snapshot.get("effort_result_label", ""),
        "effort_validation": snapshot.get("effort_validation", ""),
        "phase_observation_action": snapshot.get("observation_action", ""),
        "note": "单股补充分析；未通过当日核心池不等于结构失效",
    }


def _trajectory(symbol: str, as_of: str) -> list[dict]:
    frame = _read_csv(OUT_ROOT / "trajectory_log.csv")
    if frame.empty or not {"symbol", "date"}.issubset(frame.columns):
        return []
    frame = frame[
        frame["symbol"].astype(str).eq(symbol)
        & frame["date"].astype(str).le(as_of)
    ].sort_values("date").tail(90)
    fields = [
        "date", "state", "capital_stage", "route", "health", "structure_ready",
        "supply_ready", "supply_kind", "imp", "imp_age", "fresh", "fresh_age",
        "pit", "pit_age", "armed_core", "ready_core",
    ]
    return json.loads(frame[[c for c in fields if c in frame.columns]].to_json(
        orient="records", force_ascii=False)) if len(frame) else []


def _episodes(symbol: str) -> dict:
    capital = _read_json(OUT_ROOT / "capital_episode_store.json", {"active": {}, "closed": []})
    candidates = _read_json(OUT_ROOT / "candidate_episode_store.json", {"active": {}, "closed": []})
    capital_rows = list(capital.get("active", {}).values()) + list(capital.get("closed", []))
    candidate_rows = list(candidates.get("active", {}).values()) + list(candidates.get("closed", []))
    return {
        "capital": [row for row in capital_rows if row.get("symbol") == symbol][-10:],
        "candidate": [row for row in candidate_rows if row.get("symbol") == symbol][-10:],
    }


def _decision(value: dict, in_daily_pool: bool) -> tuple[str, str]:
    if not in_daily_pool:
        return "未进入当日核心池", "先解释未入池原因，再观察结构是否继续改善"
    tier = value.get("value_tier")
    permission = value.get("execution_permission")
    if tier == "D_AVOID":
        return "回避新开仓", str(value.get("risk_reason", "风险条件未解除"))
    if tier == "A_READY" and permission == "ALLOWED_IF_CONFIRMED":
        return "条件买入候选", "等待次日价格确认并严格使用失效价"
    if tier == "A_READY":
        return "结构优先，等待市场闸门", "个股已接近买点，但当前市场只允许观察"
    if tier == "B_SETUP":
        return "优质观察，等待触发", str(value.get("buy_reason", "结构尚未完成"))
    return "普通观察", str(value.get("buy_reason", "当前阶段不具备明确买点"))


def analyze(symbol_input: str, as_of: str = "", persist: bool = False,
            *, historical: bool = False, chart_sessions: int = 180) -> dict:
    symbol = normalize_symbol(symbol_input)
    report_date = str(pd.Timestamp(as_of).date()) if as_of else latest_report_date()
    report_dir = OUT_ROOT / report_date.replace("-", "")
    raw, classified = _data_frame(symbol, report_date)
    actual_date = str(pd.Timestamp(raw["date"].iloc[-1]).date())
    # Historical prices must work even without archived daily reports. Do not
    # import today's mutable episode store or calibration into an old decision.
    try:
        latest_date = latest_report_date()
    except FileNotFoundError:
        latest_date = ""
    historical = historical or not latest_date or (bool(as_of) and report_date < latest_date)
    report_available = (report_dir / "mainwave_weekly_candidates.csv").exists()
    mapping = _mapping(symbol, report_date)
    weekly = _read_csv(report_dir / "mainwave_weekly_candidates.csv")
    discovery = _read_csv(report_dir / "mainwave_discovery_candidates.csv")
    weekly_row = _row_for(discovery, symbol) or _row_for(weekly, symbol)
    stage = _row_for(_read_csv(report_dir / "stock_lifecycle.csv"), symbol)
    carrier = _row_for(_read_csv(report_dir / "carrier_qualification.csv"), symbol)
    snapshot = PE.latest_snapshot(symbol, report_date)
    lifecycle = str(weekly_row.get("stock_lifecycle", "")) or _stage_lifecycle(
        stage, carrier, snapshot
    )
    from . import market_context as MC
    from . import analysis_core as AC
    from . import route_evidence as RE
    from .route_scan import load_policy, assess_routes
    from . import theme_risk as TR
    market = MC.load(report_date)
    market_state = market["state"]
    candidate = weekly_row or _fallback_candidate(
        symbol, report_date, mapping, stage, carrier, snapshot, raw, lifecycle, market_state
    )
    cycle, cycle_history = SL.analyze(
        symbol, report_date, trajectory=pd.DataFrame() if historical else _read_csv(OUT_ROOT / "trajectory_log.csv"))
    candidate.update({key: val for key, val in cycle.items()
                      if key.startswith(("cycle_", "turnover_", "setup_", "speed_")) or key == "production_trigger_at"})
    value_frame = pd.DataFrame() if historical else CV.evaluate_candidates(
        report_date, pd.DataFrame([candidate]), market_state
    )
    value = value_frame.iloc[0].to_dict() if len(value_frame) else {}
    value_row = _row_for(_read_csv(report_dir / "candidate_value_analysis.csv"), symbol)
    if not historical and value_row and value_row.get("cycle_version") == SL.VERSION:
        value = value_row

    latest = classified.iloc[-1]
    close = float(raw["close"].iloc[-1])
    trigger = _number(carrier.get("trigger_price"), _number(candidate.get("trigger_price")))
    invalid = _number(carrier.get("invalid_price"), _number(candidate.get("invalid_price")))
    if trigger is None:
        trigger = float(raw["high"].shift(1).tail(20).max())
    if invalid is None:
        invalid = float(raw["low"].shift(1).tail(10).min())
    decision, decision_reason = _decision(value, bool(weekly_row))
    if historical:
        decision, decision_reason = "历史结构复盘", "现引擎按当时可见量价重建；非历史生产命中证明，不生成买卖许可"
        value = {"execution_permission": "HISTORICAL_ONLY", "execution_permission_label": "仅历史复盘",
                 "value_tier_label": "不评定当前买入价值", "buy_state_label": "非交易信号"}
    trajectory = _trajectory(symbol, report_date)
    bars = _adjusted_chart(classified, chart_sessions)
    visible_history = cycle_history[cycle_history["date"].ge(bars[0]["date"])].copy()
    anchor = close / float(latest["return_index"])
    for source, target in (("setup_trigger_index", "setup_trigger_price"),
                           ("setup_invalid_index", "setup_invalid_price"),
                           ("setup_frozen_atr", "setup_frozen_atr_price")):
        val = _number(cycle.get(source))
        cycle[target] = val * anchor if val is not None else None
    setup_rows = PCS.setup_history(visible_history)
    for key in ("trigger", "invalid"):
        if f"setup_{key}_index" in setup_rows:
            setup_rows[f"setup_{key}_price"] = setup_rows[f"setup_{key}_index"] * anchor
    result = {
        "meta": {
            "version": VERSION, "engine_version": ENGINE_VERSION,
            "requested_symbol": symbol_input, "symbol": symbol,
            "report_date": report_date, "price_date": actual_date,
            "mode": "HISTORICAL_RECONSTRUCTION" if historical else "DAILY_ANALYSIS",
            "report_available": report_available,
            "in_daily_pool": bool(weekly_row), "market_state": market_state,
        },
        "identity": {
            "symbol": symbol, "code": symbol[-6:],
            "name": mapping.get("name", stage.get("name", candidate.get("name", ""))),
            "industry": mapping.get("industry", stage.get("industry", "")),
            "concepts": list(dict.fromkeys([t for link in mapping.get("_membership_evidence", []) for t in link["themes"]]
                                           + mapping.get("concepts", [])))[:12],
            "membership_evidence": mapping.get("_membership_evidence", []),
        },
        "decision": {
            "headline": decision, "reason": decision_reason,
            "value_rank": value.get("value_rank"),
            "value_tier": value.get("value_tier", ""),
            "value_tier_label": value.get("value_tier_label", ""),
            "buy_state": value.get("buy_state", ""),
            "buy_state_label": value.get("buy_state_label", ""),
            "execution_permission": value.get("execution_permission", ""),
            "execution_permission_label": value.get("execution_permission_label", ""),
            "risk_class": value.get("risk_class", ""),
            "risk_reason": value.get("risk_reason", ""),
        },
        "price": {
            "close": round(close, 3),
            "r5_pct": round(float(latest.get("ret5", 0)) * 100, 2),
            "r20_pct": round(float(latest.get("ret20", 0)) * 100, 2),
            "r60_pct": round(float(latest.get("ret60", 0)) * 100, 2),
            "ema20_bias_pct": round(_number(candidate.get("ema20_bias"), 0), 2),
            "amount_ratio": round(_number(latest.get("amount_ratio"), 0), 2),
            "range_position60_pct": round(_number(latest.get("range_position60"), 0) * 100, 1),
            "trigger_price": round(trigger, 3),
            "trigger_distance_pct": round((trigger / close - 1) * 100, 2),
            "invalid_price": round(invalid, 3),
            "stop_distance_pct": round((close / invalid - 1) * 100, 2) if invalid else None,
        },
        "lifecycle": {
            "stages": [{"code": code, "label": label} for code, label in SL.LABELS.items()],
            "current": cycle.get("cycle_stage", lifecycle),
            "current_label": cycle.get("cycle_stage_label", LIFECYCLE_LABELS.get(lifecycle, lifecycle)),
            "daily_signal": lifecycle,
            "cycle_id": cycle.get("cycle_id", ""),
            "started_at": cycle.get("cycle_started_at", ""),
            "stage_since": cycle.get("cycle_stage_since", ""),
            "reason": cycle.get("cycle_reason", ""),
            "reason_label": cycle.get("cycle_reason_label", ""),
            "source": cycle.get("cycle_start_source", ""),
            "data_status": cycle.get("cycle_data_status", "MISSING"),
            "production_trigger_at": cycle.get("production_trigger_at", ""),
            "history": json.loads(cycle_history.to_json(orient="records")),
            "discovery_path": candidate.get("discovery_path", "NOT_IN_DAILY_POOL"),
            "discovery_path_label": candidate.get("discovery_path_label", "未进入当日核心池"),
            "broad_phase": snapshot.get("broad_phase", ""),
            "broad_phase_label": snapshot.get("broad_phase_label", ""),
            "phase_validation": snapshot.get("phase_validation", ""),
        },
        "turnover": {key: val for key, val in cycle.items() if key.startswith("turnover_")},
        "speed": {key: val for key, val in cycle.items() if key.startswith("speed_")},
        "setups": {
            "current": {key: val for key, val in cycle.items() if key.startswith("setup_")},
            "history": json.loads(setup_rows.to_json(orient="records", force_ascii=False)),
            "validation": "UNVALIDATED_OBSERVATION",
        },
        "breakout": {key: val for key, val in cycle.items() if key.startswith("breakout_")},
        "event": {
            "latest": snapshot.get("latest_event", ""),
            "latest_label": snapshot.get("latest_event_label", ""),
            "latest_date": snapshot.get("latest_event_date", ""),
            "validation": snapshot.get("event_validation", ""),
            "raw_latest": snapshot.get("latest_event_raw", ""),
            "effort": snapshot.get("effort_result", ""),
            "effort_label": snapshot.get("effort_result_label", ""),
            "effort_validation": snapshot.get("effort_validation", ""),
            "observation_action": snapshot.get("observation_action", ""),
            "history": _event_history(classified, chart_sessions),
        },
        "theme": {
            "belonging_theme": candidate.get("belonging_theme", mapping.get("industry", "")),
            "source": "当前分类，仅作导航，非历史题材证据" if historical else candidate.get("theme_source", ""),
            "lifecycle": candidate.get("theme_lifecycle", ""),
            "lifecycle_label": candidate.get("theme_lifecycle_label", ""),
        },
        "potential": {
            "context": value.get("potential_context", ""),
            "calibration_source": value.get("calibration_source", ""),
            "sample_n": int(_number(value.get("calibration_n"), 0)),
            "mfe20_p25_pct": _number(value.get("mfe20_p25_pct"), 0),
            "mfe20_median_pct": _number(value.get("mfe20_median_pct"), 0),
            "mfe20_p75_pct": _number(value.get("mfe20_p75_pct"), 0),
            "mae20_p25_pct": _number(value.get("mae20_p25_pct"), 0),
            "close_ret20_median_pct": _number(value.get("close_ret20_median_pct"), 0),
            "touch15_20_pct": _number(value.get("touch15_20_pct"), 0),
            "disclaimer": "历史MFE区间是最大上行波动，不是预测收益或保证收益",
        },
        "evidence": {
            "carrier": carrier,
            "stage_label": stage,
            "trajectory": trajectory,
            "episodes": {} if historical else _episodes(symbol),
        },
        "chart": {
            "price_basis": "以最新原始收盘价为锚的等比复权价格",
            "bars": bars,
            "stage_segments": PCS.segments(visible_history),
        },
    }
    result = json.loads(json.dumps(result, ensure_ascii=False, default=str).replace("NaN", "null"))
    if historical:
        result["potential"] = {"context": "历史结构案例，无收益预测", "sample_n": 0,
                               "disclaimer": "事后选取闭合周期及失败对照；不可据此推算发现率、胜率或潜在收益。"}
    if not historical:
        risk = TR.evaluate_many([result["theme"].get("belonging_theme", "")] + mapping.get("concepts", []), report_date)
        known_at = str(risk.get("theme_risk_known_at") or "")[:10]
        if not known_at or known_at <= report_date:
            result["theme"]["risk"] = risk
    route_frame = RE.build(raw, symbol)
    from . import price_volume_context as PVC
    result["price_volume_context"] = PVC.report(route_frame, symbol)
    result["price_volume_context"]["timeline"] = result["price_volume_context"]["timeline"][-chart_sessions:]
    if len(route_frame):
        from . import pullback_plan as PP
        timing = json.loads(PP.annotate(route_frame).tail(1).to_json(orient="records", force_ascii=False))[0]
        timing.update(validation="RESEARCH_ONLY", orders_enabled=False, rules=PP.RULES)
        result["entry_timing"] = timing
        from . import entry_policy as EP, exit_policy as XP
        from .theme_lifecycle import market_calendar
        calendar = market_calendar(actual_date)
        anchor = EP.reference(route_frame, calendar, symbol, result.get("setups", {}).get("current"))
        result["sell_observation"] = XP.observe(route_frame, calendar, symbol, anchor, XP.load_market(actual_date))
        result["buy_observation"] = EP.observe(route_frame, calendar, symbol, anchor=anchor,
                                               exit_observation=result["sell_observation"])
        observation = route_frame.iloc[-1].to_dict()
        policy = load_policy()
        route_ids = "|".join(r for r in str(candidate.get("route_ids", "")).split("|") if r in RE.ROUTE_LABELS)
        route_view = assess_routes(observation, policy["policy"], route_ids.split("|"))
        if "R3" in route_ids.split("|") and candidate.get("route_source") != "DATED_CHAIN_AND_PRICE" and not route_view["independent_price_evidence"]:
            route_view.update(route_requirements_met=False, confirmation_pass=False)
        route_view.update({"validation": policy["validation"],
                           "route_ids": route_ids, "eligible": bool(observation["eligible"]),
                           "probe_at": observation["probe_at"],
                           "prepared_at": observation["attempt_prepared_at"],
                           "attempt_at": observation["breakout_attempt_at"],
                           "accepted_at": observation["breakout_accepted_at"],
                           "reference_close": close,
                           "atr_price": _number(observation["atr20"] * close / observation["return_index"]),
                           "bias_atr": _number(observation["bias_atr"]),
                           "retained": bool(observation["retained"]), "supply": bool(observation["supply"]),
                           "demand": bool(observation["demand"]),
                           "prepared_before_attempt": bool(observation["prepared_before_attempt"])})
        route_view.update({key: observation[key] for key in RE.PT.FIELDS})
        # Dated CSVs alone do not certify publication time or a tight economic link.
        route_view["strict_chain_verified"] = False
        result["route_evidence"] = route_view
    core = AC.evaluate(result, market)
    from . import evidence_evolution as EE
    prior = None if historical else EE.load_previous(OUT_ROOT, symbol, report_date, AC.VERSION)
    core["evolution"] = EE.compare(core, prior)
    core["price_volume_context"] = {k: v for k, v in result["price_volume_context"].items() if k != "timeline"}
    result["analysis_core"] = core
    result["decision"] = {
        "headline": core["advice_label"], "reason": "；".join(core["reasons"][:2]),
        "value_tier": core["advice"], "value_tier_label": core["advice_label"],
        "buy_state": core["advice"], "buy_state_label": core["advice_label"],
        "execution_permission": "HISTORICAL_ONLY" if historical else "ANALYSIS_ONLY",
        "execution_permission_label": "仅历史复盘" if historical else "仅分析建议，不生成成交",
        "risk_class": "|".join(r["code"] for r in core["state_vector"]["risk_overlays"]) or "NO_OBSERVED_ALERT",
        "risk_reason": "；".join(r["label"] for r in core["state_vector"]["risk_overlays"]) or "未触发已实现的风险条件，不等于无风险",
    }
    if persist and not historical:
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / f"stock_analysis_{symbol}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def candidate_list(as_of: str = "") -> dict:
    report_date = latest_report_date(as_of)
    board_path = OUT_ROOT / report_date.replace("-", "") / "analysis_board.json"
    board = _read_json(board_path, {})
    from .analysis_core import VERSION as CORE_VERSION
    if board.get("version") == CORE_VERSION and board.get("as_of") == report_date:
        return {"report_date": report_date, "records": board["records"], "summary": board["summary"]}
    path = OUT_ROOT / report_date.replace("-", "") / "mainwave_weekly_candidates.csv"
    frame = _read_csv(path)
    frame["value_tier"] = "DATA_WAIT"
    frame["value_tier_label"] = "待生成核心分析"
    frame["value_rank"] = None
    if "cycle_stage_label" in frame:
        frame["stock_lifecycle_label"] = frame["cycle_stage_label"].fillna(frame["stock_lifecycle_label"])
    fields = [
        "value_rank", "symbol", "name", "value_tier", "value_tier_label",
        "buy_state_label", "execution_permission_label", "belonging_theme",
        "discovery_path_label", "stock_lifecycle_label", "mfe20_median_pct",
    ]
    records = json.loads(frame[[c for c in fields if c in frame.columns]].to_json(
        orient="records", force_ascii=False)) if len(frame) else []
    return {"report_date": report_date, "records": records}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--date", default="")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    result = analyze(args.symbol, args.date, persist=not args.no_save)
    print(json.dumps({
        "symbol": result["meta"]["symbol"], "date": result["meta"]["report_date"],
        "decision": result["decision"]["headline"],
        "lifecycle": result["lifecycle"]["current"],
        "value_tier": result["decision"]["value_tier"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
