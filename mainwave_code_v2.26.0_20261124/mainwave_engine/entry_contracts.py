"""Entry contracts: what must be known before a participation plan can exist, per way of entering.

Discovery route (R1/R2/R3/P5), return engine (E1 main wave, E2 overnight, E3 leader/sector swing) and entry
contract (B1–B5) are orthogonal. A contract lists the fields a plan needs and the data it needs; a draft with
a missing field is not READY and never becomes one by being displayed. Daily-bar drafts (B1, B3) are shadow:
they describe a next-session participation the frozen rule does not know, they carry no approval and no order
authority. Contracts that need intraday data (B4 role repair on minute bars, B5 overnight) stay RESEARCH."""
import math

VERSION = "entry_contracts_v1_20260923"
PERMISSIONS = ("RESEARCH", "SHADOW", "PAPER_APPROVED")
CONTRACTS = {
    "B1": {"name": "突破被接受", "engines": ("E1", "E3"), "permission": "SHADOW", "data": "日线收盘确认；下一交易时点模拟",
           "fields": ("accepted_at", "entry_max", "invalidation", "atr", "expires_at"),
           "rule": "突破接受日收盘后，下一交易日开盘不高于收盘+0.5ATR 才参与；失效=突破无效位；不宣称当日盘中成交。"},
    "B2": {"name": "首次回踩", "engines": ("E1",), "permission": "PAPER_APPROVED", "data": "日线（冻结规则现行）",
           "fields": ("review_lower", "review_upper", "invalid", "frozen_atr"),
           "rule": "现行 entry_50_reclaim_v2：回踩后收复确认位，次日开盘复核，首仓 50% 单标的预算。"},
    "B3": {"name": "整理后再启动", "engines": ("E1", "E3"), "permission": "SHADOW", "data": "日线：上升趋势中的整理与再推进",
           "fields": ("consolidation_high", "entry_max", "invalidation", "atr", "expires_at"),
           "rule": "启动后整理（PULLBACK/MARKUP 且带内），收盘再创整理区间新高且换手不萎缩；失效=整理区间低点。"},
    "B4": {"name": "分歧后修复", "engines": ("E3",), "permission": "RESEARCH", "data": "分钟/盘口：断板、回封、承接先后",
           "fields": ("role_confirmed", "repair_event", "sector_response", "entry_max", "invalidation"),
           "rule": "核心股断板或回撤后的修复；需要盘中先后顺序证据，日线不生成可成交成绩。"},
    "B5": {"name": "尾盘顺序触发", "engines": ("E2",), "permission": "RESEARCH", "data": "14:30 后快照与次日早盘退出窗口",
           "fields": ("snapshot_1430", "trigger_sequence", "next_open_exit"),
           "rule": "隔夜策略；缺 14:30 后时点数据时只观察，不得用全天收盘回填。"},
}
ENGINES = {"E1": "主升浪挖掘：有持续推进可能的结构", "E2": "尾盘隔夜：限定时间内的短期延续", "E3": "强势核心/板块波段：活跃方向中的核心延续、修复与独立转强"}
CHASE_ATR = 0.5


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def draft_b1(record, day, next_day, route_row=None):
    """Breakout accepted on today's close → next-session participation band. Missing invalidation keeps it a draft."""
    src = record.get("source_record") or {}
    re_ = (record.get("evidence") or {}).get("route_evidence") or {}
    accepted_at = src.get("accepted_at") or re_.get("accepted_at") or ""
    status = src.get("breakout_status") or ""
    if not accepted_at or accepted_at != day or status not in ("ACCEPTED", "ADVANCING"):
        return None
    close, atr = _num(src.get("close")), _num(re_.get("atr_price"))
    invalidation = _num(src.get("invalid"))
    if invalidation is None and route_row:
        invalidation = _num(route_row.get("invalid_price"))
    missing = [k for k, v in (("close", close), ("atr", atr), ("invalidation", invalidation)) if v is None]
    draft = {"contract": "B1", "engine": "E1", "permission": "SHADOW", "version": VERSION, "symbol": record["symbol"],
             "opportunity_id": record.get("opportunity_id"), "accepted_at": accepted_at, "decision_day": day, "effective_trade_date": next_day,
             "entry_min": close, "entry_max": round(close + CHASE_ATR * atr, 2) if close is not None and atr else None,
             "invalidation": invalidation, "atr": atr, "valid_from": f"{next_day}T09:30:00+08:00", "expires_at": f"{next_day}T15:00:00+08:00",
             "missing": missing, "ready": not missing and invalidation is not None and close is not None and invalidation < close,
             "note": "日线突破接受后的下一交易时点影子计划；不是盘中打板；无批准、无订单。"}
    if not draft["ready"] and not missing:
        draft["missing"] = ["invalidation_below_close"]
    return draft


def draft_b3(record, day, next_day):
    """Re-ignition after a post-start consolidation: in-band continuation stage with a live setup and absorbed close."""
    sv = (record.get("evidence") or {}).get("state_vector") or {}
    src = record.get("source_record") or {}
    re_ = (record.get("evidence") or {}).get("route_evidence") or {}
    f = record.get("features") or {}
    if sv.get("stage") not in ("MARKUP", "PULLBACK") or sv.get("setup_status") not in ("CONFIRMED", "FORMING"):
        return None
    bias = _num(src.get("route_bias_atr"))
    if bias is None or bias > 2.0 or bias < -1.0:
        return None
    close, atr = _num(src.get("close")), _num(re_.get("atr_price"))
    low = _num(src.get("review_lower"))
    invalidation = _num(src.get("invalid"))
    absorbed = (_num(f.get("close_location")) or 0) >= 0.6 and (_num(f.get("relative_turnover")) or 0) >= 1.0
    missing = [k for k, v in (("close", close), ("atr", atr), ("consolidation_low", low), ("invalidation", invalidation)) if v is None]
    if not absorbed:
        missing.append("re_ignition_close")
    return {"contract": "B3", "engine": "E1", "permission": "SHADOW", "version": VERSION, "symbol": record["symbol"],
            "opportunity_id": record.get("opportunity_id"), "decision_day": day, "effective_trade_date": next_day,
            "consolidation_high": _num(src.get("review_upper")), "consolidation_low": low,
            "entry_min": close, "entry_max": round(close + CHASE_ATR * atr, 2) if close is not None and atr else None,
            "invalidation": invalidation, "atr": atr, "valid_from": f"{next_day}T09:30:00+08:00", "expires_at": f"{next_day}T15:00:00+08:00",
            "missing": missing, "ready": not missing, "note": "整理后再启动的日线影子计划；无批准、无订单。"}


def failure_cost(draft):
    """Loss per share at the invalidation and the same in ATR units: position is a cost the allocation step prices,
    not a veto the contract applies."""
    entry, inv, atr = draft.get("entry_max"), draft.get("invalidation"), draft.get("atr")
    if entry is None or inv is None:
        return {"loss_per_share": None, "loss_atr": None}
    loss = round(entry - inv, 4)
    return {"loss_per_share": loss, "loss_atr": round(loss / atr, 2) if atr else None}


def drafts(record, day, next_day, route_row=None):
    """All daily-bar shadow drafts a record supports today; B2 is the frozen rule's own plan and is not drafted here."""
    out = []
    for d in (draft_b1(record, day, next_day, route_row), draft_b3(record, day, next_day)):
        if d:
            d.update(failure_cost(d))
            out.append(d)
    return out


def summarize(all_drafts):
    by = {}
    for d in all_drafts:
        entry = by.setdefault(d["contract"], {"drafts": 0, "ready": 0, "permission": CONTRACTS[d["contract"]]["permission"]})
        entry["drafts"] += 1
        entry["ready"] += bool(d["ready"])
    return by


def forward_check(drafts, bars, day):
    """Next-session check of the previous session's drafts against the actual bar of their effective day.

    Contract reading only, not a return: did the open fall inside the band (invalidation < open <= entry_max), gap
    above it (the contract says do not chase), or open at/below the invalidation; did the close hold the invalidation.
    A draft whose effective day has no bar yet stays PENDING."""
    rows = []
    for d in drafts or []:
        if d.get("effective_trade_date") != day:
            continue
        bar = bars.get(d["symbol"])
        base = {"symbol": d["symbol"], "contract": d["contract"], "decision_day": d.get("decision_day"), "effective_trade_date": day,
                "ready": bool(d.get("ready")), "entry_max": d.get("entry_max"), "invalidation": d.get("invalidation")}
        if not bar or d.get("entry_max") is None or d.get("invalidation") is None:
            rows.append({**base, "result": "PENDING" if not bar else "INCOMPLETE_DRAFT"})
            continue
        o, c = bar["open"], bar["close"]
        if o <= d["invalidation"]:
            result = "OPEN_AT_OR_BELOW_INVALIDATION"
        elif o > d["entry_max"]:
            result = "GAP_ABOVE_BAND_NO_CHASE"
        else:
            result = "OPEN_IN_BAND"
        rows.append({**base, "result": result, "open": o, "close": c, "close_held_invalidation": c > d["invalidation"],
                     "close_vs_open_pct": round((c / o - 1) * 100, 2) if o else None})
    counts = {}
    for r in rows:
        key = (r["contract"], r["result"])
        counts[f"{key[0]}:{key[1]}"] = counts.get(f"{key[0]}:{key[1]}", 0) + 1
    in_band = [r for r in rows if r["result"] == "OPEN_IN_BAND"]
    return {"version": VERSION, "day": day, "checked": len(rows), "counts": counts,
            "in_band_close_held": sum(r["close_held_invalidation"] for r in in_band), "in_band": len(in_band), "rows": rows,
            "note": "合同读法核对，不是收益：带内开盘才算可参与；高开出带按合同不追；收盘是否守住失效位"}
