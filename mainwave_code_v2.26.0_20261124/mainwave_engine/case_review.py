"""Route provenance and structural paper examples, separate from live permission."""
from __future__ import annotations

import math

from . import analysis_core as AC

VERSION = "case_review_v1_20260909"
ROUTES = {
    "R1": {"label": "线路一 · 点火个股反查", "steps": [
        "明确目标股：该点火股就是分析对象，不从题材另选股票",
        "反查资金：核对目标点火前的冲击、保留和供给收缩",
        "结构确认：当前周期仍有效，突破、推进、回踩属于同一结构",
        "位置复核：再转强有承接，未远离结构，失效边界明确"]},
    "R2": {"label": "线路二 · 个股先行埋伏", "steps": [
        "先发现埋伏股：资金介入、效果保留、供给收缩，有带日期记录",
        "等待题材配合：题材满足条件，点火股可以不是目标股",
        "等待结构确认：同轮突破站稳、推进、缩量回踩",
        "位置复核：回踩再转强且不追高，缺题材或结构就等待"]},
    "R3": {"label": "线路三 · 题材传播承接", "steps": [
        "题材与领头先动：确认源头及点火时间，领头不等于目标股",
        "识别传播链：业务、价值或催化关联与传播证据，不只同概念",
        "寻找承接股：节奏稍慢、未明显扩张，有资金准备和供给收敛",
        "结构与位置：同轮突破、推进、回踩再转强，无合格载体就等待"]},
}
RULES = (
    "独立结构模拟，非整条线路回测或实盘成交。仅统计目标识别后、同一周期的首笔合格回踩交易。"
    "收盘确认后，下一活跃交易日开盘必须落在冻结确认价至"
    "min(确认价+0.5ATR,失效价+2ATR)内，否则该次取消。"
    "收盘跌破冻结失效价、突破结构失效、周期失效或出现派发风险后，下一活跃日开盘退出；"
    "一字K线不假定可成交，卖出顺延。窗口末不强制平仓。"
    "收益为复权单笔毛价差，不是账户收益；不含费用、滑点，未重放历史市场/题材许可，"
    "未模拟盘口和真实排队成交。阈值沿用现有结构观察口径，尚未校准。"
)


def route_context(case: dict) -> dict:
    """Review route is not evidence of the original production discovery route."""
    context = case.get("route_context", {})
    route_id = context.get("id", "R1")
    if route_id not in ROUTES:
        raise ValueError(f"Unknown case route: {route_id}")
    audit = case.get("discovery_audit", {})
    target = case.get("target_ignition_at", "")
    if not target and audit:
        target = next((r["date"] for r in case.get("milestones", [])
                       if r["label"] == "题材T0前后3交易日内首次涨停"), "")
    if not audit:
        target = target or case.get("start", "")
    gaps = []
    if not context.get("source"):
        gaps.append("无原始生产线路归属；按指定个股事后复核，不证明当时可发现")
    if route_id == "R2" and not context.get("discovered_at"):
        gaps.append("缺少点火前入池日期证据")
    if route_id in {"R2", "R3"} and not context.get("theme_confirmed_at"):
        gaps.append("缺少题材条件当时满足的日期证据")
    if route_id == "R3" and not (context.get("leader_symbol") and context.get("link_evidence")
                                 and context.get("propagation_at")):
        gaps.append("缺少领头股、具体传播关联或传播可知日期")
    capital = [r for r in audit.get("trace", []) if r["date"] < target
               and r.get("capital_stage") in {"PROBED", "ABSORBING", "COILED"}]
    if not capital:
        gaps.append("未核验目标点火前同周期资金准备，不能把低位或历史命中当成吸筹确认")
    else:
        gaps.append("旧资金代理记录尚未证明与目标点火属于同一周期且连续保留")
    eligible = target
    if route_id in {"R2", "R3"}:
        required = [context.get("theme_confirmed_at"), context.get("discovered_at")
                    if route_id == "R2" else context.get("propagation_at")]
        eligible = max(required) if all(required) and context.get("source") else ""
        if route_id == "R3" and not (context.get("leader_symbol") and context.get("link_evidence")):
            eligible = ""
    return {"id": route_id, **ROUTES[route_id], "assignment": "复核线路，不等于生产发现归属",
            "source": context.get("source", "RETROSPECTIVE_TARGET_REVIEW"),
            "target_symbol": case["symbol"], "leader_symbol": context.get("leader_symbol", ""),
            "theme": case.get("theme", ""), "link_evidence": context.get("link_evidence", ""),
            "target_ignition_at": target, "target_ignition_kind": "LIMIT_UP_IN_AUDIT_WINDOW" if audit else "STRUCTURAL_PROXY",
            "theme_ignition_at": case.get("t0", ""), "eligible_at": eligible,
            "capital_first_seen": capital[0]["date"] if capital else "",
            "evidence_gaps": gaps, "production_hit_proven": False}


def _finite(value):
    try:
        return float(value) if math.isfinite(float(value)) else None
    except (ValueError, TypeError):
        return None


def review(payload: dict, classified) -> dict:
    case = payload["case"]
    route = route_context(case)
    history = {r["date"]: r for r in payload["lifecycle"]["history"]}
    bars = payload["chart"]["bars"]
    quotes = {str(r["date"])[:10]: r for r in classified.to_dict("records")}
    dates = [b["date"] for b in bars]
    # Use unrounded return-index prices for signals/returns, chart prices only for display.
    anchor_row = quotes[dates[-1]]
    anchor = float(anchor_row["close"]) / float(anchor_row["return_index"])

    def price(date, field):
        q = quotes[date]
        return float(q[field]) * float(q["return_index"]) / float(q["close"])

    def event(kind, date, label, field="close", signal_at="", reason=""):
        indexed = price(date, field)
        return {"kind": kind, "date": date, "label": label, "signal_at": signal_at,
                "raw_price": float(quotes[date][field]), "chart_price": indexed * anchor,
                "index_price": indexed, "reason": reason}

    def tradable(date):
        q = quotes[date]
        return all((_finite(q.get(k)) or 0) > 0 for k in ("open", "high", "low", "close", "amount")) and q["high"] > q["low"]

    events = []
    for kind, date, label in (
        ("TARGET_IGNITION", route["target_ignition_at"], "目标股点火（审计窗口涨停）" if case.get("discovery_audit") else "目标股结构启动（点火代理）"),
        ("THEME_IGNITION", route["theme_ignition_at"], "题材点火（非目标股买点）"),
    ):
        if date in quotes and date in dates:
            events.append(event(kind, date, label))
    cycle = (case.get("cycle_id") or history.get(route["target_ignition_at"], {}).get("cycle_id")) if route["id"] == "R1" else ""
    trade = {"status": "NO_ENTRY", "entry": None, "exit": None, "gross_return_pct": None,
             "unrealized_return_pct": None, "mark_at": dates[-1], "rejections": [],
             "reason": "目标识别后，同周期未出现合格的回踩再转强确认"}
    pending = None
    exit_pending = None
    for date in dates:
        row = history.get(date, {})
        if exit_pending and tradable(date):
            trade["exit"] = event("SELL", date, "模拟卖出", "open", *exit_pending)
            events.append(trade["exit"])
            trade.update(status="CLOSED", reason=exit_pending[1])
            trade["gross_return_pct"] = (price(date, "open") / trade["entry"]["index_price"] - 1) * 100
            break
        if pending:
            signal, trigger, invalid, atr, signal_cycle = pending
            upper = min(trigger + AC.CHASE_ATR * atr, invalid + AC.MAX_RISK_ATR * atr)
            if tradable(date) and trigger <= price(date, "open") <= upper:
                trade["entry"] = event("BUY", date, "模拟买入", "open", signal, "同周期回踩收盘确认，次日开盘在冻结区间内")
                trade.update(status="OPEN", reason="窗口内尚未产生可成交的退出")
                events.append(trade["entry"])
                stop = invalid
                cycle = signal_cycle
                entry_breakout = history[signal]["setup_breakout_id"]
            else:
                trade["rejections"].append({"signal_at": signal, "date": date,
                    "reason": "下一活跃日开盘不在冻结区间" if tradable(date) else "一字K线或缺少可成交数据"})
            pending = None
        if trade["entry"]:
            reason = ""
            if price(date, "close") < stop:
                reason = "收盘跌破买入时冻结的失效位"
            elif row.get("cycle_id") != cycle or row.get("cycle_stage") in {"FAILED", "REBUILDING", "DISTRIBUTION_RISK"}:
                reason = "本轮结构失效、换轮或出现派发风险"
            elif row.get("breakout_id") != entry_breakout or row.get("breakout_status") == "FAILED":
                reason = "买入对应的突破结构失效或替换"
            if reason and not exit_pending:
                exit_pending = (date, reason)
            continue
        if row.get("setup_status") != "CONFIRMED" or row.get("setup_confirmed_at") != date:
            continue
        if not route["eligible_at"] or date < route["eligible_at"] or not row.get("cycle_id"):
            trade["rejections"].append({"signal_at": date, "date": date, "reason": "形态确认早于目标识别/题材满足，或缺少当时周期，不倒填买点"})
            continue
        if route["id"] == "R1" and (not cycle or row.get("cycle_id") != cycle):
            trade["rejections"].append({"signal_at": date, "date": date, "reason": "形态确认不属于目标点火所在周期"})
            continue
        trigger, invalid, atr = [_finite(row.get(k)) for k in
                                 ("setup_trigger_index", "setup_invalid_index", "setup_frozen_atr")]
        accepted, advanced, adjustment = [row.get(k, "") for k in
            ("setup_breakout_accepted_at", "setup_breakout_advanced_at", "setup_adjustment_from")]
        linked = (accepted and advanced and adjustment and accepted < advanced < adjustment
                  and row.get("setup_breakout_id") == row.get("breakout_id")
                  and row.get("breakout_status") == "ADVANCING")
        upper = min(trigger + AC.CHASE_ATR * atr, invalid + AC.MAX_RISK_ATR * atr) if all(v is not None for v in (trigger, invalid, atr)) else None
        if (not linked or row.get("cycle_stage") not in {"IGNITION", "MARKUP", "PULLBACK"}
                or upper is None or atr <= 0 or not 0 < invalid < trigger
                or not trigger <= price(date, "close") <= upper):
            reason = "同链结构、风险阶段或冻结边界未通过复核"
            if linked and upper is not None and atr > 0 and 0 < invalid < trigger:
                scale = float(quotes[date]["close"]) / float(quotes[date]["return_index"])
                reason = f"收盘 {float(quotes[date]['close']):.3f}；冻结可观察区 {trigger * scale:.3f}~{upper * scale:.3f}；阶段 {row.get('cycle_stage')}"
            trade["rejections"].append({"signal_at": date, "date": date, "reason": reason})
            continue
        pending = (date, trigger, invalid, atr, row["cycle_id"])
    if trade["entry"] and not trade["exit"]:
        trade["unrealized_return_pct"] = (price(dates[-1], "close") / trade["entry"]["index_price"] - 1) * 100
        if exit_pending:
            trade.update(status="EXIT_PENDING", reason=f"{exit_pending[0]}已出现退出信号，窗口内无后续可成交开盘")
    elif pending:
        trade.update(status="ENTRY_PENDING", reason="已收盘确认，窗口内没有下一活跃交易日开盘")
    elif not trade["entry"] and trade["rejections"]:
        trade["reason"] = "出现形态确认，但结构或成交位置复核未通过"
    if not route["eligible_at"]:
        trade["reason"] = "缺少线路起点或可知时间，不能模拟入场"
    return {"version": VERSION, "route": route, "events": sorted(events, key=lambda r: r["date"]),
            "trade": trade, "rules": RULES, "profitability_validated": False}


def attach(payload: dict, classified) -> None:
    result = review(payload, classified)
    payload["case_review"] = result
    payload["case"]["review_summary"] = {"route_id": result["route"]["id"],
        "route_label": result["route"]["label"], "status": result["trade"]["status"],
        "gross_return_pct": result["trade"]["gross_return_pct"],
        "unrealized_return_pct": result["trade"]["unrealized_return_pct"]}
