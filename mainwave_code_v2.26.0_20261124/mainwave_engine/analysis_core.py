"""Pure, shared analysis contract. No orders, portfolio mutations or profit scores."""
from __future__ import annotations

import math
from . import route_contracts as RC
from . import entry_policy as EP, exit_policy as XP

VERSION = "lifecycle_analysis_core_v7_20260911"
PHASE_LABELS = {
    "PREPARATION": "启动前准备", "ADVANCE": "上涨周期", "DECLINE": "结构破坏",
    "REPAIR": "失效后修复", "UNKNOWN": "阶段待确认",
}
ADVICE_LABELS = {
    "PRIORITY_REVIEW": "优先复核机会", "WAIT_CONFIRMATION": "等待回踩确认",
    "WAIT_PRICE": "等待合理价格与风险边界", "EARLY_WATCH": "提前跟踪资金准备",
    "TREND_WATCH": "跟踪趋势，等待新形态", "RISK_WATCH": "风险观察，暂缓新介入",
    "MARKET_WAIT": "结构保留，等待市场修复",
    "DATA_WAIT": "数据不足，暂缓判断", "HISTORICAL_ONLY": "历史结构复盘",
    "WATCH": "继续观察",
}
ADVICE_ORDER = {name: i for i, name in enumerate((
    "PRIORITY_REVIEW", "WAIT_CONFIRMATION", "WAIT_PRICE", "EARLY_WATCH", "TREND_WATCH",
    "MARKET_WAIT", "WATCH", "DATA_WAIT", "RISK_WATCH", "HISTORICAL_ONLY"))}
# Research placeholders. They constrain review priority, not execution permission.
CONFIRMATION_TTL = 2
CHASE_ATR = 0.5
MAX_RISK_ATR = 2.0


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def evaluate(data: dict, market: dict) -> dict:
    meta, cycle, price = data["meta"], data["lifecycle"], data["price"]
    setup = data.get("setups", {}).get("current", {})
    breakout = data.get("breakout", {})
    route = data.get("route_evidence", {})
    turnover, theme = data.get("turnover", {}), data.get("theme", {})
    contract = RC.entry_context(route, theme)
    route_ready = contract["requirements_met"]
    r2_theme_ready = "ACTIVE_THEME_MISSING" not in contract["missing"]
    stage = cycle.get("current", "UNKNOWN")
    early = stage in {"PROBED", "ABSORBING", "COILED", "ARMED", "READY"}
    advance = stage in {"IGNITION", "MARKUP", "PULLBACK", "OVERHEATED", "DISTRIBUTION_RISK"}
    phase = ("PREPARATION" if early else "ADVANCE" if advance else "DECLINE" if stage == "FAILED"
             else "REPAIR" if stage == "REBUILDING" else "UNKNOWN")
    risks = []
    if stage in {"OVERHEATED", "DISTRIBUTION_RISK", "FAILED"}:
        risks.append({"code": stage, "label": cycle.get("current_label", stage), "source": "stock_lifecycle"})
    if market.get("state", "UNKNOWN") in {"RED", "ORANGE"}:
        risks.append({"code": "MARKET_CAUTION", "label": "市场风险或情绪过热", "source": "market_context"})
    if "SUPPLY_PRESSURE" in route.get("risks", []):
        risks.append({"code": "SUPPLY_PRESSURE", "label": "多日供给压力或结构破位", "source": "stock_lifecycle"})
    theme_risk = theme.get("risk", {})
    if theme_risk.get("theme_risk_action") in {"BLOCK_NEW_ENTRY", "REQUIRE_EXTERNAL_CONFIRMATION"}:
        risks.append({"code": "THEME_CAUTION", "label": theme_risk.get("theme_risk_reason") or "题材风险待复核",
                      "source": "theme_risk"})

    active_pullback = setup.get("setup_status") in {"FORMING", "CONFIRMED"}
    direct = (route.get("confirmation_pass") is True and advance and not active_pullback
              and breakout.get("breakout_status") in {"ACCEPTED", "ADVANCING"}
              and route.get("accepted_at") == meta.get("price_date"))
    if direct:
        reference, direct_atr = number(route.get("reference_close")), number(route.get("atr_price"))
        if reference is not None and direct_atr is not None and direct_atr > 0:
            setup = {"setup_kind": "BREAKOUT_ACCEPTANCE", "setup_status": "CONFIRMED",
                     "setup_trigger_price": reference, "setup_invalid_price": reference - 2 * direct_atr,
                     "setup_frozen_atr_price": direct_atr, "setup_confirmed_at": route["accepted_at"]}
        else:
            direct = False
    kind, status = setup.get("setup_kind", ""), setup.get("setup_status", "NONE")
    known = setup.get("setup_first_known") is True
    strategy = ("S4" if direct else "S2" if kind == "FIRST_PULLBACK" and known else "S3" if kind else
                "S1" if early else "NONE")
    trigger, invalid = number(setup.get("setup_trigger_price")), number(setup.get("setup_invalid_price"))
    atr, close = number(setup.get("setup_frozen_atr_price")), number(price.get("close"))
    boundary = all(value is not None and value > 0 for value in (trigger, invalid, atr, close))
    boundary = bool(boundary and invalid < trigger)
    upper = min(trigger + CHASE_ATR * atr, invalid + MAX_RISK_ATR * atr) if boundary else None
    if direct and boundary:
        upper = trigger + CHASE_ATR * atr
        bias = number(route.get("bias_atr"))
        if bias is not None:
            upper = min(upper, trigger + (2 - bias) * atr)
    band_valid = bool(boundary and upper >= trigger)
    in_band = bool(band_valid and trigger <= close <= upper)
    confirmed_at = str(setup.get("setup_confirmed_at") or "")
    session_dates = [r["date"] for r in data.get("chart", {}).get("bars", [])
                     if r["date"] <= meta["price_date"]]
    age = sum(d > confirmed_at for d in session_dates) if confirmed_at in session_dates else None
    confirmed = status in {"CONFIRMED", "HISTORY_CONFIRMED"} and age is not None and 0 <= age <= CONFIRMATION_TTL
    fresh = (meta.get("price_date") == meta.get("report_date") and cycle.get("data_status") == "FRESH"
             and market.get("as_of") == meta.get("report_date") and market.get("data_status") == "FRESH")
    turn_known = (number(turnover.get("turnover_relative20")) or 0) > 0
    shape_active = advance and (status == "FORMING" or confirmed)
    if shape_active and boundary and close < invalid:
        risks.append({"code": "SETUP_BROKEN", "label": "已跌破冻结形态失效位", "source": "stock_lifecycle"})
    accepted_at = str(setup.get("setup_breakout_accepted_at") or "")
    advanced_at = str(setup.get("setup_breakout_advanced_at") or "")
    adjustment_at = str(setup.get("setup_adjustment_from") or "")
    chain_linked = bool(setup.get("setup_breakout_id") and accepted_at and advanced_at and adjustment_at
                        and accepted_at < advanced_at < adjustment_at
                        and setup["setup_breakout_id"] == breakout.get("breakout_id")
                        and breakout.get("breakout_status") == "ADVANCING")
    if direct:
        chain_linked = bool(breakout.get("breakout_id") and breakout.get("breakout_accepted_at") == confirmed_at)
    if strategy == "S2" and not chain_linked:
        strategy = "S3"
    if not shape_active:
        strategy = "S1" if early else "NONE"
    checks = [
        {"id": "data", "label": "个股与市场数据同日有效", "status": "PASS" if fresh else "UNKNOWN"},
        {"id": "parent", "label": "上涨周期内，非结构失效", "status": "PASS" if advance else "FAIL"},
        {"id": "setup", "label": "存在回踩或平台形态", "status": "PASS" if shape_active else "FAIL"},
        {"id": "breakout", "label": "突破接受、推进、回踩属于同一事件链", "status": "PASS" if chain_linked else "UNKNOWN"},
        {"id": "confirmation", "label": "量价确认仍在观察时效内", "status": "PASS" if confirmed else "WAIT"},
        {"id": "turnover", "label": "换手数据可用", "status": "PASS" if turn_known else "UNKNOWN"},
        {"id": "boundary", "label": "冻结确认位、失效位和ATR有效", "status": "PASS" if band_valid else "UNKNOWN" if not boundary else "FAIL"},
        {"id": "price", "label": "当前价格未远离结构", "status": "PASS" if in_band else "WAIT"},
        {"id": "risk", "label": "无阶段或市场风险警报", "status": "PASS" if not risks else "FAIL"},
    ]
    if route:
        checks.extend([
            {"id": "route_context", "label": "当前线路的题材与载体必要证据齐备", "status": "PASS" if route_ready else "WAIT"},
            {"id": "route_demand", "label": "需求延续证据", "status": "PASS" if route.get("demand") else "WAIT"},
            {"id": "route_retention", "label": "前序冲击效果保留（非通用买入门槛）", "status": "PASS" if route.get("retained") else "UNKNOWN"},
            {"id": "route_supply", "label": "供给收缩（准备观察）", "status": "PASS" if route.get("supply") else "WAIT"},
        ])
    if direct:
        checks[2].update(label="突破接受有效，仅复核50%首仓；余仓等待回撤转强", status="PASS")
        checks[3].update(label="原始突破与接受属于同一价格事件", status="PASS" if chain_linked else "FAIL")
    stock_risks = [r for r in risks if r["source"] == "stock_lifecycle"]
    if stock_risks:
        advice = "RISK_WATCH"
    elif advance and shape_active:
        if not boundary or not turn_known or not chain_linked:
            advice = "DATA_WAIT"
        elif not band_valid:
            advice = "WAIT_PRICE"
        elif not confirmed:
            advice = "WAIT_CONFIRMATION" if status == "FORMING" else "TREND_WATCH"
        elif not in_band:
            advice = "WAIT_PRICE"
        else:
            advice = "PRIORITY_REVIEW"
    elif early:
        advice = "EARLY_WATCH"
    elif advance:
        advice = "TREND_WATCH"
    else:
        advice = "WATCH"
    structural_advice = advice
    if meta.get("mode") == "HISTORICAL_RECONSTRUCTION":
        advice = "HISTORICAL_ONLY"
    elif not fresh:
        advice = "DATA_WAIT"
    elif route.get("eligible") is False:
        advice = "DATA_WAIT"
    elif any(r["source"] == "theme_risk" for r in risks):
        advice = "RISK_WATCH"
    elif structural_advice in {"PRIORITY_REVIEW", "WAIT_CONFIRMATION", "WAIT_PRICE", "EARLY_WATCH", "TREND_WATCH"} \
            and any(r["code"] == "MARKET_CAUTION" for r in risks):
        advice = "MARKET_WAIT"
    if advice in {"PRIORITY_REVIEW", "MARKET_WAIT"} and not route_ready:
        advice = "WATCH"

    local_role = ("DOWN_REPAIR" if phase in {"REPAIR", "DECLINE"} else "ORIGIN_BREAKOUT" if direct else
                  "FIRST_RETEST" if strategy == "S2" else "CONTINUATION" if advance else "BASE_PREPARATION" if early else "UNKNOWN")
    location = "IN_BAND" if in_band else "EXTENDED" if boundary and close > (upper or trigger) else "UNKNOWN"
    contract.update(selected_strategy=strategy, decision=advice, orders_enabled=False)

    reasons = [r["label"] for r in risks]
    if not r2_theme_ready:
        reasons.append("题材契约尚无活跃阶段确认，保留个股观察；路线标签不能替代证据")
    if "STRICT_CHAIN_UNVERIFIED" in contract["missing"]:
        reasons.append("传播线索尚未核验当时可用时间与紧链关系，不能当作严格传播机会")
    if advice == "MARKET_WAIT":
        reasons.append("个股结构：" + ADVICE_LABELS[structural_advice])
    elif advice not in {"PRIORITY_REVIEW", "EARLY_WATCH", "HISTORICAL_ONLY"}:
        reasons.extend(c["label"] + "：" + c["status"] for c in checks if c["status"] != "PASS")
    if not reasons:
        reasons = ["当前观察条件齐备，先复核结构与题材证据；不是自动买入许可" if advice == "PRIORITY_REVIEW"
                   else "保留早期发现，等待供给收敛和启动确认" if early else "按当时量价重建，非历史生产命中证明"]
    next_condition = ("等待新的回踩或平台，不沿用过期确认" if status.startswith("HISTORY_") and not confirmed
                      else "收复冻结确认位且换手承接恢复，同时保持失效位有效")
    if not kind:
        next_condition = ("保留已启动结构，等待推进后的回踩或平台，不把上涨延续当作新突破"
                          if advance else "等待资金准备转为有效突破，再记录回踩或整理过程")
    elif direct:
        next_condition = "下一交易日开盘复核50%首仓，超价或风险门槛不通过则不买；余仓仅在首仓后7个交易日内回撤并再转强后复核，无机会不补"
    elif shape_active and not chain_linked:
        next_condition = "突破接受与推进前因不完整，保留形态观察；不能据此认定首次有效回踩"
    elif boundary and not band_valid:
        next_condition = "冻结确认位与失效位跨度超过当前观察约束，等待新结构；风险跨度阈值仍待校准"
    if stage == "DISTRIBUTION_RISK":
        next_condition = "等待需求恢复并收复风险压力位，不能把缩量横盘直接认作重新吸筹"
    elif stage in {"FAILED", "REBUILDING"}:
        next_condition = "先形成有效修复，再建立新周期证据；旧形态不继续作为介入理由"
    elif stage == "OVERHEATED":
        next_condition = "等待偏离收敛和有效回踩，重新检查承接与派发风险"
    if not r2_theme_ready and not stock_risks:
        next_condition = "等待题材发酵或活跃阶段证据，再复核当前结构；准备观察不直接升级为介入"

    def dimension(code, label, evidence_status, observation, source, missing):
        return {"code": code, "label": label, "status": evidence_status,
                "observation": observation, "source": source, "as_of": meta["report_date"], "missing": missing}

    dimensions = [
        dimension("D1", "企业与经营", "UNKNOWN", "未纳入本版机会裁定，不把盈利当作资金准备的替代指标",
                  "未接入经营证据适配器", ["经营变化、兑现持续性与披露时间"]),
        dimension("D2", "产业与题材", "PARTIAL" if theme.get("belonging_theme") else "UNKNOWN",
                  {"theme": theme.get("belonging_theme"), "lifecycle": theme.get("lifecycle")},
                  theme.get("source", ""), ["独立催化与产业链传导证据"]),
        dimension("D3", "供需与换手", "OBSERVED" if turn_known else "UNKNOWN", turnover,
                  "stock_lifecycle / price_cycle_setups", ["量价代理不能证明真实主力持仓"]),
        dimension("D4", "价格与波动", "OBSERVED" if cycle.get("data_status") == "FRESH" else "PARTIAL",
                  {"phase": phase, "stage": stage, "setup": kind, "reason": cycle.get("reason_label"),
                   "speed": data.get("speed", {})},
                  "stock_lifecycle / breakout_events / price_cycle_setups",
                  ["事件链为因果量价代理，接受窗口和推进ATR待分层校准"]),
        dimension("D5", "预期与注意力", "UNKNOWN", "题材归属或股价上涨不等于预期差已获证实",
                  "尚未绑定到单股假设的注意力证据", ["市场原有预期、增量信息、独立注意力证据"]),
        dimension("D6", "市场环境", "OBSERVED" if market.get("data_status") == "FRESH" else "UNKNOWN",
                  {"structure": market.get("structure"), "temperature": market.get("temperature"),
                   "context": market.get("context"), "observations": market.get("observations"),
                   "method": market.get("method"), "sources": market.get("sources")},
                  "market_context.json", market.get("issues", [])),
    ]
    strategies = [
        {"code": "S1", "label": "启动前准备", "status": "OBSERVE" if early else "NOT_MATCHED"},
        {"code": "S2", "label": "首次突破回踩", "status": "STRUCTURAL_PROXY" if strategy == "S2" else "NOT_MATCHED"},
        {"code": "S3", "label": "趋势内后续整理", "status": "STRUCTURAL_PROXY" if strategy == "S3" else "NOT_MATCHED"},
        {"code": "S4", "label": "突破接受确认", "status": "HISTORICAL_PRICE_PROXY" if direct else "NOT_MATCHED"},
        {"code": "S5", "label": "区间均值回归", "status": "RESEARCH_ONLY"},
        {"code": "S6", "label": "超跌修复", "status": "RESEARCH_RELEVANT" if market.get("reversion_research_relevant") else "RESEARCH_ONLY"},
    ]
    return {
        "version": VERSION, "as_of": meta["report_date"], "symbol": meta["symbol"],
        "mode": "ANALYSIS_ONLY", "orders_enabled": False, "market": market,
        "observation_mode": meta.get("mode", "DAILY_ANALYSIS"),
        "state_vector": {"phase": phase, "phase_label": PHASE_LABELS[phase],
                         "stage": stage, "stage_label": cycle.get("current_label", stage),
                         "cycle_id": cycle.get("cycle_id", ""), "setup_id": setup.get("setup_id", ""),
                         "setup_kind": kind, "setup_status": status, "theme_phase": theme.get("lifecycle"),
                         "local_role": local_role, "location": location,
                         "risk_overlays": risks},
        "breakout": breakout, "chain_linked": chain_linked,
        "route_evidence": route,
        "entry_timing": data.get("entry_timing"),
        "buy_plan": EP.plan(data.get("buy_observation"), advice, risks, route_ready,
                            fresh and meta.get("mode") != "HISTORICAL_RECONSTRUCTION"),
        "sell_plan": XP.plan(data.get("sell_observation"), meta.get("price_date") == meta.get("report_date")
                              and cycle.get("data_status") == "FRESH" and meta.get("mode") != "HISTORICAL_RECONSTRUCTION"),
        "entry_contract": contract,
        "dimensions": dimensions, "strategies": strategies, "active_strategy": strategy,
        "mechanism": {"code": "M0", "label": "量价结构假设", "m1_m2_status": "证据不足，不推断事件重估或经营改善"},
        "checks": checks, "advice": advice, "advice_label": ADVICE_LABELS[advice],
        "structural_advice": structural_advice, "structural_advice_label": ADVICE_LABELS[structural_advice],
        "reasons": reasons, "next_condition": next_condition,
        "review_window": {"trigger": trigger if boundary and shape_active else None,
                          "lower": trigger if band_valid and shape_active and chain_linked else None,
                          "upper": upper if band_valid and shape_active and chain_linked else None,
                          "invalid": invalid if boundary and shape_active else None, "frozen_atr": atr,
                          "confirmed_at": confirmed_at, "confirmation_age_sessions": age,
                          "ttl_sessions": CONFIRMATION_TTL,
                          "price_in_band": bool(in_band and shape_active and chain_linked),
                          "active": bool(shape_active and chain_linked)},
        "validation": "UNVALIDATED_ANALYSIS_OVERLAY",
        "limitations": ["独立价格事件与连续生命周期分开；大阶段为分析投影，不是完整V3价格状态机",
                        "首仓50%与回撤转强补仓为用户选定的正式分析规则，不等于已验证盈利；所有线路保留各自风险与证据门槛",
                        "卖点采用用户批准的收盘确认与条件续持方案；量价异常只是风险代理，历史研究未证明稳定收益优势",
                        "S2/S3只表示现有形态的结构代理，不代表严格V3策略全部必要条件已验证",
                        "观察窗口阈值为研究占位；不估计潜在收益率，不生成实际或虚拟成交"],
    }
