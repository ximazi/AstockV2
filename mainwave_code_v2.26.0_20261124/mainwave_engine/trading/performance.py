"""Per-lot ledger performance. Price observation and paper net results are reported apart."""
import json
from . import store

VERSION = "paper_performance_v1_20260920"


def _lots_and_sells(conn, plan_id):
    lots = [dict(r) for r in conn.execute(
        "SELECT * FROM paper_lots WHERE plan_id=? ORDER BY entry_day,id", (plan_id,))]
    sells = [dict(r) for r in conn.execute(
        "SELECT f.* FROM paper_fills f JOIN paper_orders o ON o.id=f.order_id "
        "WHERE o.plan_id=? AND o.side='SELL' ORDER BY f.day,f.rowid", (plan_id,))]
    return lots, sells


def _realized(lots, sells):
    """FIFO match of sells against buy lots; the fee of each side is charged where it occurred."""
    queue = [{"remaining": l["quantity"], "price": l["price_cents"], "fee_per_share": l["entry_fee_cents"] / l["quantity"]}
             for l in lots]
    realized, matched = 0, []
    for sell in sells:
        left = sell["quantity"]
        for lot in queue:
            if left <= 0:
                break
            take = min(left, lot["remaining"])
            if take <= 0:
                continue
            cost = take * lot["price"] + take * lot["fee_per_share"]
            proceeds = take * sell["price_cents"]
            realized += proceeds - cost
            lot["remaining"] -= take
            left -= take
            matched.append({"day": sell["day"], "quantity": take, "cost_cents": int(cost),
                            "proceeds_cents": proceeds})
        realized -= sell["fee_cents"]
    return int(realized), matched


def plan_ledger(conn, plan, day):
    lots, sells = _lots_and_sells(conn, plan["id"])
    remaining = sum(l["remaining"] for l in lots)
    valuation = conn.execute("SELECT payload FROM paper_valuations WHERE plan_id=? AND as_of<=? ORDER BY as_of DESC,rowid DESC LIMIT 1",
                             (plan["id"], day)).fetchone()
    mark = json.loads(valuation[0]) if valuation else {}
    realized, matched = _realized(lots, sells)
    sellable = sum(l["remaining"] for l in lots if l["entry_day"] < day)
    cost_remaining = sum(l["remaining"] * l["price_cents"] + l["entry_fee_cents"] * l["remaining"] / l["quantity"]
                         for l in lots if l["quantity"])
    blocked = conn.execute("SELECT 1 FROM paper_orders WHERE plan_id=? AND status IN ('EXIT_BLOCKED','PARTIAL','UNKNOWN')",
                           (plan["id"],)).fetchone() is not None
    mark_cents = mark.get("mark_cents") if mark.get("status") not in (None, "DATA_WAIT") else None
    unrealized = int(remaining * mark_cents - cost_remaining) if mark_cents and remaining else None
    return {"plan_id": plan["id"], "symbol": plan["symbol"], "name": plan.get("name"), "theme": plan.get("theme"),
            "lots": [{"entry_day": l["entry_day"], "quantity": l["quantity"], "remaining": l["remaining"],
                      "price_cents": l["price_cents"], "fee_cents": l["entry_fee_cents"],
                      "cost_cents": l["quantity"] * l["price_cents"] + l["entry_fee_cents"]} for l in lots],
            "remaining": remaining, "sellable_today": sellable, "t_plus_1_locked": remaining - sellable,
            "cost_remaining_cents": int(cost_remaining), "realized_cents": realized, "matched_sells": matched,
            "mark_cents": mark_cents, "mark_day": mark.get("day"), "mark_status": mark.get("status", "NO_VALUATION"),
            "unrealized_cents": unrealized,
            "exposure_continues": bool(remaining) and (blocked or mark.get("status") in ("DATA_WAIT", "EXIT_REQUIRED", "REDUCE_REQUIRED")),
            "exposure_note": ("退出受阻、部分成交或数据等待期间仍计风险暴露，不提前结算为已退出"
                              if blocked or mark.get("status") == "DATA_WAIT" else "")}


def _daily_cash(initial, fills, cash_events=None):
    by_day = {}
    for f in fills:
        delta = (-f["quantity"] * f["price_cents"] if f["side"] == "BUY" else f["quantity"] * f["price_cents"]) - f["fee_cents"]
        by_day[f["day"]] = by_day.get(f["day"], 0) + delta
    for day, cents in (cash_events or {}).items():
        by_day[day] = by_day.get(day, 0) + cents
    cash, series = initial, {}
    for day in sorted(by_day):
        cash += by_day[day]
        series[day] = cash
    return series


def _history(conn):
    """Confirmed corporate actions keyed to their event day: share deltas per lot, cash per day."""
    from . import corporate_actions
    deltas, cash = {}, {}
    for action in corporate_actions.applied(conn):
        done = action["applied"]
        if done["kind"] == "SHARE_EVENT":
            for l in done["lots"]:
                deltas.setdefault(l["lot"], []).append((action["day"], l["remaining"][1] - l["remaining"][0]))
        elif done["kind"] == "CASH_EVENT":
            cash[action["day"]] = cash.get(action["day"], 0) + sum(l["cash_cents"] for l in done["lots"])
    return deltas, cash


def nav_series(conn, initial, plans):
    """Daily equity where every open lot has a same-day mark; otherwise the day is DATA_WAIT, not filled."""
    fills = [dict(r) for r in conn.execute("SELECT f.*,o.side FROM paper_fills f JOIN paper_orders o ON o.id=f.order_id ORDER BY f.day")]
    deltas, cash_events = _history(conn)
    cash_by_day = _daily_cash(initial, fills, cash_events)
    marks = {}
    for row in conn.execute("SELECT as_of, plan_id, payload FROM paper_valuations ORDER BY as_of,rowid"):
        marks.setdefault(row["as_of"], {})[row["plan_id"]] = json.loads(row["payload"])
    days = sorted(set(cash_by_day) | set(marks))
    series, cash, peak, worst_dd = [], initial, initial, 0.0
    for day in days:
        cash = cash_by_day.get(day, cash)
        holdings, complete = 0, True
        for plan in plans:
            lots = [dict(r) for r in conn.execute(
                "SELECT id, quantity, remaining, entry_day FROM paper_lots WHERE plan_id=? AND entry_day<=?", (plan["id"], day))]
            sold = sum(f["quantity"] for f in fills if f["side"] == "SELL" and f["day"] <= day
                       and conn.execute("SELECT plan_id FROM paper_orders WHERE id=?", (f["order_id"],)).fetchone()[0] == plan["id"])
            # Before a share event's day the lot is still in its old units; the event never rewrites earlier days.
            held = sum(l["quantity"] - sum(d for event_day, d in deltas.get(l["id"], []) if event_day > day)
                       for l in lots) - sold
            if held <= 0:
                continue
            mark = marks.get(day, {}).get(plan["id"], {})
            if mark.get("status") in (None, "DATA_WAIT") or not mark.get("mark_cents"):
                complete = False
                break
            holdings += held * mark["mark_cents"]
        if not complete:
            series.append({"day": day, "cash_cents": cash, "equity_cents": None, "status": "DATA_WAIT"})
            continue
        equity = cash + holdings
        peak = max(peak, equity)
        dd = (equity / peak - 1) if peak else 0.0
        worst_dd = min(worst_dd, dd)
        series.append({"day": day, "cash_cents": cash, "holdings_cents": holdings, "equity_cents": equity,
                       "drawdown": round(dd, 6), "status": "MARKED"})
    return series, round(worst_dd, 6)


def report(day, db=store.DB_PATH):
    from . import paper
    with paper.database(db) as conn:
        account = conn.execute("SELECT * FROM paper_account").fetchone()
        if not account:
            return {"version": VERSION, "status": "UNCONFIGURED", "as_of": day,
                    "note": "未配置模拟账户，只有价格观察成绩，没有账户净绩效。"}
        cfg = json.loads(account["config"])
        plans = [json.loads(r[0]) for r in conn.execute("SELECT payload FROM paper_plans ORDER BY created_at")]
        ledgers = [plan_ledger(conn, p, day) for p in plans]
        series, worst_dd = nav_series(conn, account["initial_cents"], plans)
        reserved = sum(r[0] for r in conn.execute("SELECT reserve_cents FROM paper_orders WHERE status IN ('OPEN','PARTIAL','UNKNOWN')"))
    initial = account["initial_cents"]
    open_ledgers = [l for l in ledgers if l["remaining"]]
    invested = sum(l["cost_remaining_cents"] for l in open_ledgers)
    marked = all(l["mark_cents"] for l in open_ledgers)
    equity = account["cash_cents"] + sum(l["remaining"] * l["mark_cents"] for l in open_ledgers) if marked else None
    realized = sum(l["realized_cents"] for l in ledgers)
    unrealized = sum(l["unrealized_cents"] or 0 for l in open_ledgers) if marked else None
    by_symbol, by_theme = {}, {}
    for l in open_ledgers:
        cost = l["cost_remaining_cents"]
        by_symbol[l["symbol"]] = by_symbol.get(l["symbol"], 0) + cost
        by_theme[l["theme"] or "未归属"] = by_theme.get(l["theme"] or "未归属", 0) + cost
    caps = {"symbol_cap_cents": initial * cfg["symbol_cap_pct"] / 100, "theme_cap_cents": initial * cfg["theme_cap_pct"] / 100,
            "gross_cap_cents": initial * cfg["gross_cap_pct"] / 100}
    breaches = ([f"{s} 个股敞口 {v} > 上限" for s, v in by_symbol.items() if v > caps["symbol_cap_cents"]]
                + [f"{t} 题材敞口 {v} > 上限" for t, v in by_theme.items() if v > caps["theme_cap_cents"]]
                + ([f"总敞口 {invested} > 上限"] if invested > caps["gross_cap_cents"] else []))
    return {"version": VERSION, "as_of": day, "status": "READY",
            "account": {"initial_cents": initial, "cash_cents": account["cash_cents"], "reserved_cents": reserved,
                        "equity_cents": equity, "invested_cents": invested,
                        "capital_usage": round(invested / initial, 4) if initial else None,
                        "realized_cents": realized, "unrealized_cents": unrealized,
                        "net_return": round((equity / initial - 1), 6) if equity else None,
                        "max_drawdown": worst_dd, "marked": marked},
            "plans": ledgers, "nav": series,
            "exposure": {"by_symbol": by_symbol, "by_theme": by_theme, "caps": caps, "breaches": breaches,
                         "continuing": [l["symbol"] for l in open_ledgers if l["exposure_continues"]],
                         "note": "同股多模式共用同一预算；题材敞口按计划登记的所属题材归口，一股多题材时以计划题材为准。"},
            "separation": {"paper_net": "本报告：账本原始成交、费用、按批成本与净值",
                           "price_observation": "见 ranking_validation.json：固定窗口价格观察，不是成交收益",
                           "note": "两套成绩分别输出，不合并。"}}
