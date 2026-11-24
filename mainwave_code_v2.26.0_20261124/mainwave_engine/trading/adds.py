"""Conditional adds anchored on the actual paper fill, never on the analysis signal."""
import json
import math
from . import store

VERSION = "conditional_add_v1_20260920"
# The coded scheme is kept faithfully: two pullback tranches inside seven sessions, split 3:4.
SCHEME = {"policy": "entry_50_reclaim_v2_20260911", "pullbacks": (.03, .06),
          "weights": (.5 * 3 / 7, .5 * 4 / 7), "window_sessions": 7,
          "contraction_max": .85, "close_location_min": .6, "amount_ratio_min": 1.}
STOP_REASONS = {
    "NO_INITIAL_FILL": "首仓未成交，不开启补仓流程",
    "WINDOW_EXPIRED": "自实际首仓起7个交易日窗口到期",
    "ALREADY_REDUCED": "已有减仓或退出，终止剩余补仓",
    "STRUCTURE_FAILED": "结构失效，终止剩余补仓",
    "EXIT_PENDING": "存在待执行退出，优先处理退出",
    "NO_PULLBACK": "窗口内没有有效回撤，保持首仓",
    "TRANCHE_DONE": "该档已买入，每档最多一次",
    "RULES_PENDING_REVIEW": "计划规则版本与当前代码不一致，原代码不可重建，不补仓",
}


def _lots(conn, plan_id):
    return [dict(r) for r in conn.execute(
        "SELECT * FROM paper_lots WHERE plan_id=? ORDER BY entry_day,id", (plan_id,))]


def _sessions_since(calendar, start, day):
    return len([d for d in calendar if start < d <= day])


def anchor(conn, plan_id):
    """The add clock starts at the first actual fill; unfilled or partial is stated plainly."""
    lots = _lots(conn, plan_id)
    order = conn.execute("SELECT quantity, filled, status FROM paper_orders WHERE id=?",
                         (plan_id + ":initial",)).fetchone()
    if not lots:
        return {"status": "NO_INITIAL_FILL", "filled": 0,
                "ordered": order["quantity"] if order else 0,
                "order_status": order["status"] if order else "MISSING"}
    filled = sum(l["quantity"] for l in lots if l["entry_day"] == lots[0]["entry_day"])
    ordered = order["quantity"] if order else filled
    return {"status": "ANCHORED", "entry_day": lots[0]["entry_day"],
            "price_cents": lots[0]["price_cents"], "filled": filled, "ordered": ordered,
            "partial": filled < ordered, "order_status": order["status"] if order else "UNKNOWN",
            "remaining_quantity": sum(l["remaining"] for l in lots)}


def budget(plan, account, cfg, exposures, conn):
    """Frozen plan budget, minus what every open order already reserves, under the same caps as the initial."""
    from . import paper
    # Always read the ledger as it is now: other plans' reservations and this plan's earlier tranche count.
    account = dict(conn.execute("SELECT * FROM paper_account WHERE id=1").fetchone())
    exposures = paper._exposure(conn)
    # The approved initial is exactly half of the symbol plan budget, so the whole plan is twice it.
    planned = plan.get("planned_budget_cents") or int(plan["quantity"] * plan["limit_cents"] * 2)
    spent = sum(l["quantity"] * l["price_cents"] + l["entry_fee_cents"] for l in _lots(conn, plan["id"]))
    reserved = sum(r[0] for r in conn.execute(
        "SELECT reserve_cents FROM paper_orders WHERE plan_id=? AND status IN ('OPEN','PARTIAL','UNKNOWN')",
        (plan["id"],)))
    reserved_all = sum(r[0] for r in conn.execute(
        "SELECT reserve_cents FROM paper_orders WHERE status IN ('OPEN','PARTIAL','UNKNOWN')"))
    capital = min(account["initial_cents"], account["cash_cents"] + sum(x.get("market_value", 0) for x in exposures))
    cost = lambda cond: sum(x["cost"] for x in exposures if cond(x))
    # The single-symbol cap is a cap on the symbol, so what the symbol already risks (held and pending) comes off it.
    symbol_risk = sum(x["risk"] for x in exposures if x["symbol"] == plan["symbol"])
    return {"planned_cents": planned, "spent_cents": spent, "reserved_cents": reserved,
            "remaining_cents": max(0, planned - spent - reserved),
            "cash_cents": max(0, account["cash_cents"] - reserved_all),
            "symbol_headroom_cents": max(0, int(capital * cfg["symbol_cap_pct"] / 100 - cost(lambda x: x["symbol"] == plan["symbol"]))),
            "theme_headroom_cents": max(0, int(capital * cfg["theme_cap_pct"] / 100 - cost(lambda x: x["theme"] == plan["theme"]))),
            "gross_headroom_cents": max(0, int(capital * cfg["gross_cap_pct"] / 100 - cost(lambda x: True))),
            "single_risk_cap_cents": int(capital * cfg["single_risk_pct"] / 100), "symbol_risk_cents": symbol_risk,
            "single_risk_headroom_cents": max(0, int(capital * cfg["single_risk_pct"] / 100 - symbol_risk)),
            "risk_headroom_cents": max(0, int(capital * cfg["total_risk_pct"] / 100) - sum(x["risk"] for x in exposures))}


def evaluate(conn, plan, frame, calendar, day, exposures, account, cfg, dispatch_status="CURRENT"):
    """Re-verify price, structure, sold state, cash and risk at the moment of the add."""
    from .. import entry_policy as EP
    result = {"version": VERSION, "plan_id": plan["id"], "symbol": plan["symbol"], "day": day,
              "scheme": {**SCHEME, "pullbacks": list(SCHEME["pullbacks"]),
                         "weights": list(SCHEME["weights"])},
              "tranches": [], "status": "NO_ACTION", "stop_reason": ""}
    if dispatch_status != "CURRENT":
        return {**result, "status": "STOPPED", "stop_reason": "RULES_PENDING_REVIEW",
                "dispatch_status": dispatch_status}
    base = anchor(conn, plan["id"])
    result["anchor"] = base
    if base["status"] != "ANCHORED":
        return {**result, "status": "STOPPED", "stop_reason": "NO_INITIAL_FILL"}
    age = _sessions_since(calendar, base["entry_day"], day)
    result["age_sessions"] = age

    sold = {json.loads(r[0])["reason"] for r in conn.execute(
        "SELECT payload FROM paper_orders WHERE plan_id=? AND side='SELL' AND filled>0", (plan["id"],))}
    pending_exit = conn.execute(
        "SELECT 1 FROM paper_orders WHERE plan_id=? AND side='SELL' AND status IN ('OPEN','PARTIAL','EXIT_BLOCKED')",
        (plan["id"],)).fetchone()
    if sold:
        return {**result, "status": "STOPPED", "stop_reason": "ALREADY_REDUCED", "sold_reasons": sorted(sold)}
    if pending_exit:
        return {**result, "status": "STOPPED", "stop_reason": "EXIT_PENDING"}
    # entry_policy expires at >= WINDOW; the wired version must not silently widen it.
    if age >= SCHEME["window_sessions"]:
        return {**result, "status": "STOPPED", "stop_reason": "WINDOW_EXPIRED"}

    days = frame.date.dt.strftime("%Y-%m-%d").tolist() if hasattr(frame.date, "dt") else list(frame.date)
    if base["entry_day"] not in days or day not in days:
        return {**result, "status": "DATA_WAIT", "stop_reason": "PRICE_HISTORY_GAP"}
    a = {c: frame[c].to_numpy() for c in frame.columns}
    a["ma20"] = frame.index_close.rolling(20).mean().to_numpy()
    entry = days.index(base["entry_day"])
    last = days.index(day)
    scale = float(a["close"][last] / a["index_close"][last])
    price = base["price_cents"] / 100 / scale
    atr = plan["atr_cents"] / 100 / scale
    structure = plan["structure_cents"] / 100 / scale
    hard = price - 2 * atr
    if a["index_close"][last] < structure:
        return {**result, "status": "STOPPED", "stop_reason": "STRUCTURE_FAILED"}

    done = {json.loads(r[0]).get("tranche") for r in conn.execute(
        "SELECT payload FROM paper_orders WHERE plan_id=? AND side='BUY' AND id LIKE ?",
        (plan["id"], plan["id"] + ":add%"))}
    tests = [None, None]
    money = budget(plan, account, cfg, exposures, conn)
    result["budget"] = money
    for j in range(entry, last + 1):
        if j > entry:
            _, strong, contracted = EP.recovery_evidence(a, j)
            for k, pullback in enumerate(SCHEME["pullbacks"]):
                test = tests[k]
                if j == last and test and test["contracted"] and strong and (k + 2) not in done:
                    ceiling = EP.order_ceiling(a["index_close"][j], a["ma20"][j], atr, price)
                    floor = max(structure, hard)
                    result["tranches"].append({
                        "tranche": k + 2, "weight": SCHEME["weights"][k],
                        "touch_date": test["day"], "confirm_date": days[j],
                        "min_price_cents": int(round(floor * scale * 100)),
                        "max_price_cents": int(round(ceiling * scale * 100)),
                        "execution": "NEXT_EXCHANGE_OPEN_ONLY",
                        "sizing_basis": "剩余计划预算按3:4分配，只用实际剩余数量与现金"})
                if a["index_low"][j] <= price * (1 - pullback) and price * (1 - pullback) > hard:
                    if test is None:
                        tests[k] = {"day": days[j], "contracted": contracted}
                    elif contracted:
                        test["contracted"] = True
    result["tests"] = [t for t in tests if t]
    result["completed_tranches"] = sorted(x for x in done if x)
    if not result["tranches"]:
        result["status"] = "NO_ACTION"
        result["stop_reason"] = "" if result["tests"] else "NO_PULLBACK"
    else:
        result["status"] = "ADD_CONFIRMED"
    return result


def queue(conn, plan, evaluation, account, cfg, at=None):
    """Place at most one add order per tranche; a lowered average never widens the hard floor."""
    from . import paper
    placed = []
    for tranche in evaluation.get("tranches", []):
        oid = f"{plan['id']}:add{tranche['tranche']}"
        if conn.execute("SELECT 1 FROM paper_orders WHERE id=?", (oid,)).fetchone():
            continue
        limit = tranche["max_price_cents"]
        # Re-read after every placement so the second tranche sees what the first one reserved.
        money = budget(plan, account, cfg, None, conn)
        share = int(money["planned_cents"] * tranche["weight"])
        cash = min(share, money["remaining_cents"], money["cash_cents"], money["symbol_headroom_cents"],
                   money["theme_headroom_cents"], money["gross_headroom_cents"])
        # The hard floor stays the one frozen on the initial fill.
        hard = plan["stop_cents"]
        risk_each = max(1, limit - hard)
        risk_budget = min(money["single_risk_headroom_cents"], money["risk_headroom_cents"])
        quantity = max(0, int(min(cash / limit, risk_budget / risk_each)) // cfg["lot_size"] * cfg["lot_size"])
        while quantity and (quantity * limit + paper.fee(quantity * limit, cfg, "BUY") > cash or
                            quantity * risk_each + paper.fee(quantity * limit, cfg, "BUY") > risk_budget):
            quantity -= cfg["lot_size"]
        if quantity < cfg["lot_size"]:
            paper.event(conn, "ADD_REJECTED", {"plan_id": plan["id"], "tranche": tranche["tranche"],
                                               "reason": "BUDGET_OR_RISK_BELOW_MINIMUM"})
            continue
        reserve = quantity * limit + paper.fee(quantity * limit, cfg, "BUY")
        conn.execute("INSERT INTO paper_orders VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (oid, plan["id"], "BUY", quantity, 0, "OPEN", reserve,
                      quantity * risk_each + paper.fee(quantity * limit, cfg, "BUY"),
                      at or store.now(),
                      store.encoded({"reason": "CONDITIONAL_ADD", "tranche": tranche["tranche"],
                                     "min_price_cents": tranche["min_price_cents"],
                                     "max_price_cents": limit, "policy": SCHEME["policy"],
                                     "anchor_entry_day": evaluation["anchor"]["entry_day"],
                                     "reference_day": tranche["confirm_date"],
                                     "model": paper.MODEL})))
        paper.event(conn, "ADD_APPROVED", {"plan_id": plan["id"], "order_id": oid,
                                           "tranche": tranche["tranche"], "quantity": quantity,
                                           "reserved_cents": reserve})
        placed.append({"order_id": oid, "tranche": tranche["tranche"], "quantity": quantity,
                       "reserve_cents": reserve})
    return placed
