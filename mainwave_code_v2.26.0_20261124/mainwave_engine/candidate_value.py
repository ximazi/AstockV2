# -*- coding: utf-8 -*-
"""Current-value classification for the unified main-wave candidate list."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import phase_events as PE


VERSION = "candidate_value_analysis_v1_20260830"
CALIBRATION_PATH = Path("results/mainwave_value_calibration/calibration.json")
PACKAGED_CALIBRATION_PATH = Path(__file__).with_name("calibration") / "value_calibration.json"

TIER_LABELS = {
    "A_READY": "A 可执行准备",
    "B_SETUP": "B 优质观察",
    "C_WATCH": "C 普通观察",
    "D_AVOID": "D 回避/等待修复",
}
BUY_STATE_LABELS = {
    "BUY_ZONE": "已到结构买入区",
    "WAIT_TRIGGER": "等待触发确认",
    "WAIT_PULLBACK": "突破后等待回踩",
    "WATCH_LATE": "阶段偏后，只观察",
    "AVOID_RISK": "风险否决",
}
EXECUTION_LABELS = {
    "ALLOWED_IF_CONFIRMED": "市场允许后可按条件执行",
    "MARKET_WATCH_ONLY": "市场闸门仅允许观察",
    "BLOCKED": "当前禁止新开仓",
}


def _number(value, default=None):
    try:
        number = float(value)
        return number if pd.notna(number) else default
    except (TypeError, ValueError):
        return default


def _load_calibration(path: Path = CALIBRATION_PATH) -> dict:
    candidates = [path]
    if path == CALIBRATION_PATH:
        candidates.append(PACKAGED_CALIBRATION_PATH)
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("event_detector") == PE.VERSION:
            return payload
    return {}


def _technical_context(symbol: str, as_of: str) -> dict:
    try:
        raw = pd.read_csv(Path("data") / f"{symbol}_daily.csv", encoding="utf-8-sig")
    except (OSError, pd.errors.EmptyDataError):
        return {}
    frame = PE.classify_frame(raw)
    if frame.empty:
        return {}
    frame = frame[frame["date"] <= pd.Timestamp(as_of)]
    if frame.empty:
        return {}
    latest = frame.iloc[-1]
    return {
        "range_position60": _number(latest.get("range_position60"), 0.5),
        "ret20": _number(latest.get("ret20"), 0.0),
    }


def potential_context(row: dict, technical: dict | None = None) -> str:
    technical = technical or {}
    event = str(row.get("latest_event", "") or "")
    effort = str(row.get("effort_result", "") or "")
    phase = str(row.get("broad_phase", "") or "")
    if event in {"LPS", "UTAD"}:
        return f"EVENT_{event}"
    if effort in {"DEMAND_EXPANSION", "DISTRIBUTION_PRESSURE"}:
        return f"EFFORT_{effort}"
    if phase in {"MARKUP", "CLIMAX", "DISTRIBUTION"}:
        return f"PHASE_{phase}"
    position_value = _number(technical.get("range_position60"), 0.5)
    ret20 = _number(technical.get("ret20"), _number(row.get("r20"), 0.0) / 100)
    position = "LOW" if position_value < 0.35 else ("MID" if position_value < 0.75 else "HIGH")
    momentum = "WEAK" if ret20 < 0 else (
        "FLAT" if ret20 < 0.05 else ("BUILDING" if ret20 < 0.15 else "EXTENDED")
    )
    return f"TECH_{position}_{momentum}"


def _risk_state(row: dict) -> tuple[str, str]:
    if row.get("cycle_data_status") in {"MISSING", "STALE"}:
        return "HARD_RISK", "连续跟踪行情缺失或陈旧，不能确认当前买点"
    production_hit = isinstance(row.get("production_trigger_at"), str) and bool(row["production_trigger_at"].strip())
    if production_hit and row.get("cycle_stage") in {"FAILED", "REBUILDING", "DISTRIBUTION_RISK"}:
        return "HARD_RISK", "连续生命周期存在失效或派发风险，等待结构修复"
    action = str(row.get("action", "") or "")
    lifecycle = str(row.get("stock_lifecycle", "") or "")
    phase = str(row.get("broad_phase", "") or "")
    observation = str(row.get("phase_observation_action", "") or "")
    theme_lifecycle = str(row.get("theme_lifecycle", "") or "")
    if action.startswith("BLOCKED_"):
        return "BLOCKED", f"规则禁止：{action}"
    if observation == "RISK_REVIEW" or phase == "DISTRIBUTION":
        return "HARD_RISK", "已验证的派发或高位假突破风险"
    if lifecycle in {"OVERHEATED", "DISTRIBUTION"}:
        return "HARD_RISK", f"个股生命周期已到 {lifecycle}"
    if phase == "CLIMAX" or observation == "NO_CHASE_REVIEW":
        return "LATE_RISK", "高峰转换阶段，不追高"
    if theme_lifecycle in {"CROWDED", "RETREAT"}:
        return "THEME_RISK", f"题材生命周期已到 {theme_lifecycle}"
    return "CLEAR", "当前没有结构性硬否决"


def _buy_state(row: dict, risk_state: str) -> tuple[str, str]:
    if risk_state in {"BLOCKED", "HARD_RISK"}:
        return "AVOID_RISK", "风险否决优先于历史潜力"
    lifecycle = str(row.get("stock_lifecycle", "") or "")
    production_hit = isinstance(row.get("production_trigger_at"), str) and bool(row["production_trigger_at"].strip())
    if production_hit and row.get("cycle_stage") in {"IGNITION", "MARKUP", "PULLBACK", "OVERHEATED"}:
        return "WATCH_LATE", "本轮已经启动，不能按当天蓄势标签重新认定早期买点"
    trigger = _number(row.get("trigger_distance_pct"))
    stop = _number(row.get("stop_distance_pct"))
    if risk_state in {"LATE_RISK", "THEME_RISK"} or lifecycle in {
        "IGNITION", "ACCELERATION", "OVERHEATED", "DISTRIBUTION"
    }:
        return "WATCH_LATE", "个股或题材已进入偏后阶段"
    if trigger is not None and trigger < 0:
        return "WAIT_PULLBACK", "价格已越过触发位，等待回踩支撑确认"
    if lifecycle in {"READY", "ARMED"} and trigger is not None and 0 <= trigger <= 3:
        if stop is None or stop <= 8:
            return "BUY_ZONE", "处于早期阶段，临近触发且失效距离受控"
    return "WAIT_TRIGGER", "结构已形成，但买入确认尚未完成"


def _tier(row: dict, risk_state: str, buy_state: str) -> str:
    if risk_state in {"BLOCKED", "HARD_RISK"}:
        return "D_AVOID"
    lifecycle = str(row.get("stock_lifecycle", "") or "")
    if buy_state == "BUY_ZONE" and lifecycle in {"READY", "ARMED"}:
        return "A_READY"
    if buy_state in {"WAIT_TRIGGER", "WAIT_PULLBACK"} and lifecycle in {
        "READY", "ARMED", "COILED", "ABSORBING"
    }:
        return "B_SETUP"
    return "C_WATCH"


def _execution_permission(row: dict, market_state: str, risk_state: str) -> str:
    if risk_state in {"BLOCKED", "HARD_RISK"}:
        return "BLOCKED"
    action = str(row.get("action", "") or "")
    if market_state in {"RED", "ORANGE", "UNKNOWN"} or action == "MARKET_RISK_WATCH_ONLY":
        return "MARKET_WATCH_ONLY"
    return "ALLOWED_IF_CONFIRMED"


def evaluate_candidates(as_of: str, weekly: pd.DataFrame, market_state: str,
                        calibration_path: Path = CALIBRATION_PATH) -> pd.DataFrame:
    calibration = _load_calibration(calibration_path)
    contexts = calibration.get("contexts", {})
    minimum_n = int(calibration.get("minimum_context_n", 200) or 200)
    baseline = contexts.get("BASELINE", {})
    rows = []
    for source in weekly.to_dict("records"):
        technical = _technical_context(str(source.get("symbol", "")), as_of)
        context = potential_context(source, technical)
        stats = contexts.get(context, {})
        calibration_source = context
        if not stats or int(stats.get("n", 0) or 0) < minimum_n:
            stats = baseline
            calibration_source = "BASELINE" if baseline else "UNAVAILABLE"
        risk_state, risk_reason = _risk_state(source)
        buy_state, buy_reason = _buy_state(source, risk_state)
        tier = _tier(source, risk_state, buy_state)
        permission = _execution_permission(source, market_state, risk_state)
        rows.append({
            **source,
            "value_tier": tier,
            "value_tier_label": TIER_LABELS[tier],
            "buy_state": buy_state,
            "buy_state_label": BUY_STATE_LABELS[buy_state],
            "execution_permission": permission,
            "execution_permission_label": EXECUTION_LABELS[permission],
            "risk_class": risk_state,
            "risk_reason": risk_reason,
            "buy_reason": buy_reason,
            "potential_context": context,
            "calibration_source": calibration_source,
            "calibration_n": int(stats.get("n", 0) or 0),
            "mfe20_p25_pct": _number(stats.get("mfe20_p25"), 0) * 100,
            "mfe20_median_pct": _number(stats.get("mfe20_median"), 0) * 100,
            "mfe20_p75_pct": _number(stats.get("mfe20_p75"), 0) * 100,
            "mae20_p25_pct": _number(stats.get("mae20_p25"), 0) * 100,
            "close_ret20_median_pct": _number(stats.get("close_ret20_median"), 0) * 100,
            "touch15_20_pct": _number(stats.get("touch15_20"), 0) * 100,
            "mfe40_median_pct": _number(stats.get("mfe40_median"), 0) * 100,
            "potential_available": bool(stats),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    tier_order = {"A_READY": 0, "B_SETUP": 1, "C_WATCH": 2, "D_AVOID": 3}
    buy_order = {"BUY_ZONE": 0, "WAIT_TRIGGER": 1, "WAIT_PULLBACK": 2,
                 "WATCH_LATE": 3, "AVOID_RISK": 4}
    path_order = {"LURK": 0, "PROPAGATION": 1, "IGNITION": 2}
    theme_source_order = {
        "PROPAGATION_CONFIRMED": 0, "WEEKLY_IGNITION": 1,
        "HOTSPOT_LIMIT_MEMBER": 1, "POINT_IN_TIME_MAPPING": 2,
        "ACTIVE_THEME": 2, "CURRENT_INDUSTRY_DISPLAY": 3,
        "CURRENT_CONCEPT_DISPLAY": 3, "STALE_POINT_MAPPING": 4,
        "STALE_AUX_RELATION": 5, "MISSING": 6,
    }
    result = result.assign(
        _tier=result["value_tier"].map(tier_order).fillna(9),
        _buy=result["buy_state"].map(buy_order).fillna(9),
        _path=result["discovery_path"].map(path_order).fillna(9),
        _theme=result["theme_source"].map(theme_source_order).fillna(6),
    ).sort_values(
        ["_tier", "_buy", "_path", "_theme", "mfe20_median_pct",
         "stop_distance_pct", "trigger_distance_pct", "symbol"],
        ascending=[True, True, True, True, False, True, True, True],
        na_position="last",
    ).drop(columns=["_tier", "_buy", "_path", "_theme"]).reset_index(drop=True)
    result.insert(0, "value_rank", range(1, len(result) + 1))
    for column in [
        "mfe20_p25_pct", "mfe20_median_pct", "mfe20_p75_pct", "mae20_p25_pct",
        "close_ret20_median_pct", "touch15_20_pct", "mfe40_median_pct",
    ]:
        result[column] = pd.to_numeric(result[column], errors="coerce").round(1)
    return result


def write_candidate_value(as_of: str, output: Path, weekly: pd.DataFrame,
                          market_state: str) -> tuple[pd.DataFrame, dict]:
    frame = evaluate_candidates(as_of, weekly, market_state)
    csv_path = output / "candidate_value_analysis.csv"
    json_path = output / "candidate_value_analysis.json"
    report_path = output / "candidate_value_analysis.md"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    records = json.loads(frame.to_json(orient="records", force_ascii=False)) if len(frame) else []
    actionable = frame[
        frame["execution_permission"].eq("ALLOWED_IF_CONFIRMED")
        & frame["value_tier"].eq("A_READY")
    ] if len(frame) else frame
    structural_top = frame[frame["value_tier"].isin(["A_READY", "B_SETUP"])].head(5) \
        if len(frame) else frame
    summary = {
        "version": VERSION,
        "as_of": as_of,
        "market_state": market_state,
        "total": len(frame),
        "a_ready": int(frame["value_tier"].eq("A_READY").sum()) if len(frame) else 0,
        "b_setup": int(frame["value_tier"].eq("B_SETUP").sum()) if len(frame) else 0,
        "actionable_now": len(actionable),
        "top_symbols": structural_top["symbol"].tolist() if len(structural_top) else [],
        "records": records,
        "report": str(report_path),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    shown = frame.copy()
    if len(shown):
        shown["股票"] = shown["name"].fillna("") + " " + shown["symbol"].fillna("")
        shown["历史20日上行空间"] = shown.apply(
            lambda row: f"{row['mfe20_p25_pct']:.1f}% / {row['mfe20_median_pct']:.1f}% / {row['mfe20_p75_pct']:.1f}%",
            axis=1,
        )
        table = shown.rename(columns={
            "value_rank": "价值排名", "value_tier_label": "价值层级",
            "buy_state_label": "当前买点", "execution_permission_label": "执行许可",
            "belonging_theme": "所属题材", "discovery_path_label": "发现路径",
            "stock_lifecycle_label": "个股阶段", "risk_reason": "主要风险",
        })[[
            "价值排名", "股票", "价值层级", "当前买点", "执行许可", "所属题材",
            "发现路径", "个股阶段", "历史20日上行空间", "主要风险",
        ]].to_markdown(index=False)
    else:
        table = "当前没有候选。"
    lines = [
        f"# 主升浪候选当前价值分析 {as_of}", "",
        f"- 版本：`{VERSION}`；市场闸门：`{market_state}`。",
        "- 价值排名采用分类优先级：硬风险 -> 买点状态 -> 发现路径/生命周期 -> 题材证据 -> 历史空间，不做加权平均。",
        "- 历史空间依次为未来20个交易日 MFE 的25%分位/中位/75%分位，是历史最大上行波动，不是可兑现收益承诺。",
        f"- A级 {summary['a_ready']} 只，B级 {summary['b_setup']} 只，当前市场许可下可执行 A级 {summary['actionable_now']} 只。",
        "", table, "",
        "## 使用边界", "",
        "候选价值层回答的是‘当前哪只更接近可执行机会’，不替代次日价格确认、市场风险闸门和个股失效价。",
        "历史潜力区间来自2022-2025固定口径校准，并经过2026上半年样本外检查；收盘持有收益与盘中MFE不是同一个概念。",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return frame, summary
