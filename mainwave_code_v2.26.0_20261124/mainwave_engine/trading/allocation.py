"""Account protocols and the allocation step: which approved, valid plans may occupy the next unit of money.

Two accounts are named. AUTO100W is the existing mechanical paper book (capital 1,000,000, its own rule).
NEW100K_PAPER is the compounding account the master plan asks for, active from 2026-09-24 under protocol
new100k_protocol_v1_20260923 (the same numbers the NEW100K ledger is initialised with). An account whose risk
profile has any unset field is CONFIG_PENDING and nothing is allocated to it. Seats: at most 2 stock securities
and 1 ETF, counted by security across holdings, pending buy orders and new allocations; the same security through
several routes is one seat. No order is created here."""
VERSION = "allocation_v1_20260923"
CONFIG_PENDING = "CONFIG_PENDING"
ACCOUNTS = {
    "AUTO100W": {"account_id": "AUTO100W", "run_mode": "PAPER", "capital": 1000000, "status": "ACTIVE", "seats": {"stock": 3, "etf": 0},
                 "risk_profile": {"single_risk_pct": 1.0, "total_risk_pct": 3.0, "symbol_cap_pct": 20.0, "theme_cap_pct": 40.0, "gross_cap_pct": 80.0},
                 "note": "旧自动账：参与榜前 3、趋势回踩、机械规则；成绩与新账不合并"},
    "NEW100K_PAPER": {"account_id": "NEW100K_PAPER", "run_mode": "PAPER", "capital": 100000, "status": "ACTIVE", "seats": {"stock": 2, "etf": 1},
                      "effective_at": "2026-09-24", "approved_equity_ceiling": 100000,
                      "risk_profile": {"single_risk_pct": 1.0, "total_risk_pct": 3.0, "symbol_cap_pct": 35.0, "etf_cap_pct": 30.0,
                                       "theme_cap_pct": 50.0, "gross_cap_pct": 90.0, "max_drawdown_pct": 10.0},
                      "note": "10 万元复利虚拟账（协议 new100k_protocol_v1_20260923，与 books.NEW100K_ACCOUNT 一致）；1000 倍目标不进入任何参数"},
}


def risk_base(current_verified_equity, approved_equity_ceiling):
    """Fixed risk percentages apply to the smaller of verified equity and the approved ceiling: losses shrink the base
    at once, gains widen it only after the ceiling is raised by review."""
    if current_verified_equity is None:
        return None
    return min(current_verified_equity, approved_equity_ceiling) if approved_equity_ceiling else current_verified_equity


def _is_etf(symbol):
    return str(symbol).startswith(("sh_51", "sh_56", "sh_58", "sz_15", "sz_16"))


def plan_valid(plan, target_date):
    """Approved, has every required field, and its effective window covers the target session."""
    required = ("plan_id", "account_id", "symbol", "approval_state", "effective_trade_date", "entry_min", "entry_max", "invalidation", "quantity_cap", "planned_loss_budget")
    missing = [k for k in required if plan.get(k) in (None, "")]
    if missing:
        return False, "MISSING:" + ",".join(missing)
    if plan["approval_state"] != "APPROVED":
        return False, f"NOT_APPROVED:{plan['approval_state']}"
    if plan["effective_trade_date"] != target_date:
        return False, "EXPIRED_OR_NOT_YET_EFFECTIVE" if plan["effective_trade_date"] < target_date else "NOT_YET_EFFECTIVE"
    if not (plan["invalidation"] < plan["entry_min"] <= plan["entry_max"]):
        return False, "PRICE_BAND_INVALID"
    return True, ""


def allocate_ready_plans(plans, account, target_date, holdings=None, pending_orders=None, cash=None, verified_equity=None):
    """Pure decision: for each plan, allocate or name the block. Never creates or touches an order."""
    holdings = holdings or []
    pending = pending_orders or []
    account_id = account.get("account_id")
    result = {"version": VERSION, "account_id": account_id, "target_date": target_date, "allocated": [], "blocked": []}
    if account.get("status") != "ACTIVE" or any(v is None for v in (account.get("risk_profile") or {}).values()):
        for p in plans:
            result["blocked"].append({"plan_id": p.get("plan_id"), "symbol": p.get("symbol"), "reason": CONFIG_PENDING})
        result["note"] = "账户协议未确认：不分配任何计划"
        return result
    seats = account.get("seats") or {"stock": 2, "etf": 1}
    occupied_stock = {h["symbol"] for h in holdings if not _is_etf(h["symbol"])} | {o["symbol"] for o in pending if o.get("side", "BUY") == "BUY" and not _is_etf(o["symbol"])}
    occupied_etf = {h["symbol"] for h in holdings if _is_etf(h["symbol"])} | {o["symbol"] for o in pending if o.get("side", "BUY") == "BUY" and _is_etf(o["symbol"])}
    base = risk_base(verified_equity, account.get("approved_equity_ceiling"))
    profile = account["risk_profile"]
    reserved = sum(float(o.get("reserve", 0) or 0) for o in pending)
    available_cash = (float(cash) - reserved) if cash is not None else None
    used_risk = sum(float(h.get("risk", 0) or 0) for h in holdings) + sum(float(o.get("risk", 0) or 0) for o in pending)
    theme_used = {}
    for h in holdings:
        theme_used[h.get("theme")] = theme_used.get(h.get("theme"), 0.0) + float(h.get("cost", 0) or 0)
    for p in sorted(plans, key=lambda x: (x.get("priority", 10 ** 6), x.get("symbol", ""))):
        ok, reason = plan_valid(p, target_date)
        if not ok:
            result["blocked"].append({"plan_id": p.get("plan_id"), "symbol": p.get("symbol"), "reason": reason})
            continue
        if p["account_id"] != account_id:
            result["blocked"].append({"plan_id": p["plan_id"], "symbol": p["symbol"], "reason": f"ACCOUNT_MISMATCH:{p['account_id']}"})
            continue
        sym = p["symbol"]
        if sym in occupied_stock or sym in occupied_etf or any(a["symbol"] == sym for a in result["allocated"]):
            result["blocked"].append({"plan_id": p["plan_id"], "symbol": sym, "reason": "SECURITY_ALREADY_OCCUPIES_SEAT"})
            continue
        is_etf = _is_etf(sym)
        seat_used = len(occupied_etf) + sum(_is_etf(a["symbol"]) for a in result["allocated"]) if is_etf else len(occupied_stock) + sum(not _is_etf(a["symbol"]) for a in result["allocated"])
        if seat_used >= seats["etf" if is_etf else "stock"]:
            result["blocked"].append({"plan_id": p["plan_id"], "symbol": sym, "reason": "NO_SEAT_ETF" if is_etf else "NO_SEAT_STOCK"})
            continue
        if base is None or available_cash is None:
            result["blocked"].append({"plan_id": p["plan_id"], "symbol": sym, "reason": "EQUITY_OR_CASH_UNVERIFIED"})
            continue
        per_share_loss = p["entry_max"] - p["invalidation"]
        risk_room = min(base * profile["single_risk_pct"] / 100, base * profile["total_risk_pct"] / 100 - used_risk)
        cap_pct = profile["etf_cap_pct"] if is_etf else profile["symbol_cap_pct"]
        theme_room = base * profile["theme_cap_pct"] / 100 - theme_used.get(p.get("primary_theme"), 0.0)
        notional_room = min(available_cash, base * cap_pct / 100, theme_room if not is_etf else base * cap_pct / 100)
        qty = int(min(notional_room / p["entry_max"], risk_room / per_share_loss, p["quantity_cap"]) // 100 * 100) if per_share_loss > 0 else 0
        if qty < 100:
            result["blocked"].append({"plan_id": p["plan_id"], "symbol": sym, "reason": "CAPITAL_BLOCK",
                                      "detail": {"risk_room": round(risk_room, 2), "notional_room": round(notional_room, 2), "per_share_loss": round(per_share_loss, 4)}})
            continue
        planned_loss = qty * per_share_loss
        result["allocated"].append({"plan_id": p["plan_id"], "symbol": sym, "quantity": qty, "notional_max": round(qty * p["entry_max"], 2),
                                    "planned_loss": round(planned_loss, 2), "risk_base": round(base, 2), "theme": p.get("primary_theme")})
        available_cash -= qty * p["entry_max"]
        used_risk += planned_loss
        theme_used[p.get("primary_theme")] = theme_used.get(p.get("primary_theme"), 0.0) + qty * p["entry_max"]
    result["seats"] = {"stock_occupied_before": len(occupied_stock), "etf_occupied_before": len(occupied_etf), "limits": seats}
    return result
