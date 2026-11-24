"""Campaign profit tracker: a read-only projection of the paper ledger (strong_trend_core_v1 §02).

A campaign is one plan in the paper ledger: first actual fill to final exit. Everything here is computed
from fills, lots, valuations and confirmed corporate actions; nothing writes the ledger, so a rerun always
reproduces the same numbers. Money is integer cents.

  total net PnL  = cumulative sell gross - cumulative buy gross + remaining qty x mark + income - fees
  peak           = max(0, total net PnL at each FRESH end-of-day mark, with the quantity held that day)
  giveback       = max(0, peak - current); ratio = giveback / peak, None when there is no positive peak

Scenario figures ("exit at protection", "stress exit") are estimates, never "locked" profit.
"""
import json

from . import paper

VERSION = "profit_tracker_v1_20260923"
STRESS_GAP = 0.10          # declared stress: one main-board limit-down gap below the protection reference
GIVEBACK_ALERT = 0.30      # an alert only; exits stay with the campaign's frozen exit contract
NOTICE = ("利润追踪为账本只读投影：按保护参考价/压力价的数字是情景估算，不是锁定利润，也不保证成交；"
          "回吐只告警，不构成第二套卖出规则。")


def _fills(conn, plan_id):
    return [dict(r) for r in conn.execute(
        "SELECT f.id, f.day, f.quantity, f.price_cents, f.fee_cents, o.side FROM paper_fills f "
        "JOIN paper_orders o ON o.id=f.order_id WHERE o.plan_id=? ORDER BY f.day, f.rowid", (plan_id,))]


def _corporate(conn, plan_id):
    """Confirmed share deltas per lot and cash income per day for this plan only."""
    from . import corporate_actions
    deltas, income = {}, {}
    for action in corporate_actions.applied(conn):
        if action.get("plan_id") != plan_id:
            continue
        done = action["applied"]
        if done["kind"] == "SHARE_EVENT":
            for lot in done["lots"]:
                deltas.setdefault(lot["lot"], []).append((action["day"], lot["remaining"][1] - lot["remaining"][0]))
        elif done["kind"] == "CASH_EVENT":
            income[action["day"]] = income.get(action["day"], 0) + sum(l["cash_cents"] for l in done["lots"])
    return deltas, income


def state_on(fills, lots, deltas, income, day):
    """Cumulative flows and the quantity actually held at the end of `day` (never today's size on old days)."""
    upto = [f for f in fills if f["day"] <= day]
    buy = sum(f["quantity"] * f["price_cents"] for f in upto if f["side"] == "BUY")
    sell = sum(f["quantity"] * f["price_cents"] for f in upto if f["side"] == "SELL")
    fees = sum(f["fee_cents"] for f in upto)
    bought = sum(l["quantity"] + sum(d for event_day, d in deltas.get(l["id"], []) if event_day <= day)
                 for l in lots if l["entry_day"] <= day)
    sold = sum(f["quantity"] for f in upto if f["side"] == "SELL")
    cash_income = sum(v for d, v in income.items() if d <= day)
    return {"buy_gross": buy, "sell_gross": sell, "fees": fees, "qty": bought - sold, "income": cash_income}


def net_pnl(state, mark_cents):
    return state["sell_gross"] - state["buy_gross"] + state["qty"] * mark_cents + state["income"] - state["fees"]


def giveback(peak, current):
    loss = max(0, peak - current)
    return loss, (loss / peak if peak > 0 else None)


def fifo_split(lots, fills, mark_cents):
    """Display split only (FIFO, buy fee capitalised per share). The cash identity above is the reconciliation."""
    queue = [{"id": l["id"], "day": l["entry_day"], "left": l["quantity"], "qty": l["quantity"], "price": l["price_cents"],
              "fee_share": l["entry_fee_cents"] / l["quantity"], "realized": 0.0} for l in lots if l["quantity"]]
    realized = 0.0
    for sell in (f for f in fills if f["side"] == "SELL"):
        left = sell["quantity"]
        for lot in queue:
            take = min(left, lot["left"])
            if take <= 0:
                continue
            gain = take * (sell["price_cents"] - lot["price"] - lot["fee_share"]) - sell["fee_cents"] * take / sell["quantity"]
            lot["realized"] += gain
            realized += gain
            lot["left"] -= take
            left -= take
            if not left:
                break
    per_lot, unrealized = [], 0.0
    for lot in queue:
        open_gain = lot["left"] * (mark_cents - lot["price"] - lot["fee_share"]) if mark_cents is not None else None
        unrealized += open_gain or 0.0
        per_lot.append({"lot_id": lot["id"], "entry_day": lot["day"], "quantity": lot["qty"], "remaining": lot["left"],
                        "price_cents": lot["price"], "realized_cents": round(lot["realized"]),
                        "unrealized_cents": None if open_gain is None else round(open_gain)})
    return round(realized), (round(unrealized) if mark_cents is not None else None), per_lot


def campaign(conn, plan, day, cfg, protection_cents=None, mark_override=None):
    """One campaign as of `day`. `mark_override` = (cents, day, source) lets the position monitor mark from
    fresh bars when the closure has not written a valuation; it never changes the stored history."""
    pid = plan["id"]
    lots = [dict(r) for r in conn.execute("SELECT * FROM paper_lots WHERE plan_id=? ORDER BY entry_day, id", (pid,))]
    fills = _fills(conn, pid)
    deltas, income = _corporate(conn, pid)
    if not lots:
        return None
    marks = [(r["as_of"], json.loads(r["payload"])) for r in conn.execute(
        "SELECT as_of, payload FROM paper_valuations WHERE plan_id=? AND as_of<=? ORDER BY as_of, rowid", (pid, day))]
    first_fill = lots[0]["entry_day"]
    peak, peak_day, path = 0, None, []
    for as_of, v in marks:
        if as_of < first_fill or v.get("status") == "DATA_WAIT" or not v.get("mark_cents"):
            continue
        value = net_pnl(state_on(fills, lots, deltas, income, as_of), v["mark_cents"])
        path.append({"day": as_of, "net_pnl_cents": value})
        if value > peak:
            peak, peak_day = value, as_of
    now = state_on(fills, lots, deltas, income, day)
    closed = now["qty"] == 0 and any(f["side"] == "SELL" for f in fills)
    latest = next(((d, v) for d, v in reversed(marks) if v.get("status") != "DATA_WAIT" and v.get("mark_cents")), None)
    if closed:
        mark, mark_day, mark_source = 0, max(f["day"] for f in fills), "CLOSED"
    elif mark_override and (not latest or mark_override[1] > latest[0]):
        mark, mark_day, mark_source = mark_override
    elif latest:
        mark, mark_day, mark_source = latest[1]["mark_cents"], latest[0], "CLOSURE_VALUATION"
    else:
        mark, mark_day, mark_source = None, None, "NO_TRUSTED_MARK"
    result = {"campaign_id": pid, "symbol": plan["symbol"], "name": plan.get("name"), "book": plan.get("book"),
              "entry_day": first_fill, "strategy": plan.get("mode"), "entry_contract": plan.get("setup_kind"),
              "exit_version": plan.get("exit_version"), "trade_closed": closed,
              "qty": now["qty"], "cumulative_buy_gross_cents": now["buy_gross"],
              "cumulative_sell_gross_cents": now["sell_gross"], "occurred_fees_cents": now["fees"],
              "net_income_cents": now["income"], "mark_cents": mark, "mark_day": mark_day, "mark_source": mark_source,
              "net_pnl_cents": None, "realized_fifo_cents": None, "unrealized_fifo_cents": None, "lots": [],
              "observed_peak_cents": peak, "peak_day": peak_day, "giveback_cents": None, "giveback_ratio": None,
              "giveback_alert": False, "pnl_path": path, "reconciled": None}
    if mark is None:
        result["projection_status"] = "NO_TRUSTED_MARK"
        return result
    current = net_pnl(now, mark)
    if not closed and mark_day == day:
        peak = max(peak, current)
    loss, ratio = giveback(peak, current)
    realized, unrealized, per_lot = fifo_split(lots, fills, None if closed else mark)
    # Split reconciles to the cash identity except for income and corporate share events, which it does not model.
    split_total = realized + (unrealized or 0) + now["income"]
    result.update(net_pnl_cents=current, realized_fifo_cents=realized, unrealized_fifo_cents=unrealized, lots=per_lot,
                  observed_peak_cents=peak, giveback_cents=loss, giveback_ratio=None if ratio is None else round(ratio, 6),
                  giveback_alert=bool(ratio is not None and ratio >= GIVEBACK_ALERT),
                  reconciled=abs(split_total - current) <= len(fills) + 1 or bool(deltas))
    if closed:
        result["projection_status"] = "CLOSED_FROZEN"
        return result
    protect = protection_cents
    exit_cost = paper.fee(now["qty"] * (protect or mark), cfg, "SELL")
    stress_price = round(min(protect or mark, mark) * (1 - STRESS_GAP))
    stress_cost = paper.fee(now["qty"] * stress_price, cfg, "SELL")
    at_protection = current + now["qty"] * (protect - mark) - exit_cost if protect else None
    at_stress = current + now["qty"] * (stress_price - mark) - stress_cost
    result.update(projection_status="OPEN",
                  protection_reference_cents=protect,
                  estimated_exit_cost_cents=exit_cost,
                  net_pnl_if_exit_at_protection_cents=at_protection,
                  stress_exit_price_cents=stress_price,
                  net_pnl_if_stress_exit_cents=at_stress,
                  current_wealth_loss_in_stress_cents=current - at_stress,
                  principal_loss_in_stress_cents=max(0, -at_stress),
                  stress_basis=f"保护参考价与现价中较低者再下跌 {int(STRESS_GAP * 100)}%（一个主板跌停跳空）",
                  projection_warning="参考价/压力价不保证成交，非锁定利润")
    return result


def time_weighted_return(subperiods):
    """subperiods: [(start_after_flow_cents, end_before_next_flow_cents)]; empty input is not a zero return."""
    factor, count = 1.0, 0
    for start, end in subperiods:
        if start <= 0:
            raise ValueError("子区间起始净值必须为正")
        factor *= end / start
        count += 1
    if not count:
        raise ValueError("没有收益观测")
    return factor - 1


def unit_nav(series, initial_cents, flows=None):
    """Units are issued/redeemed at the prevailing unit value, so external cash never shows up as return.
    `flows` = {day: cents} external contributions (+) / withdrawals (-) booked before that day's mark."""
    flows, units, rows, peak, worst = flows or {}, initial_cents / 100.0, [], 1.0, 0.0
    last_unit = 1.0
    for row in series:
        flow = flows.get(row["day"], 0)
        if flow:
            units += flow / 100.0 / last_unit
        if row.get("equity_cents") is None:
            rows.append({"day": row["day"], "unit_nav": None, "status": "DATA_WAIT"})
            continue
        last_unit = row["equity_cents"] / 100.0 / units
        peak = max(peak, last_unit)
        worst = min(worst, last_unit / peak - 1)
        rows.append({"day": row["day"], "unit_nav": round(last_unit, 6), "drawdown": round(last_unit / peak - 1, 6),
                     "status": "MARKED"})
    return rows, round(worst, 6)


def account(conn, day, campaigns, nav_series, flows=None):
    raw = conn.execute("SELECT * FROM paper_account").fetchone()
    cfg = json.loads(raw["config"])
    initial = raw["initial_cents"]
    open_c = [c for c in campaigns if not c["trade_closed"]]
    marked = all(c["mark_cents"] is not None for c in open_c)
    equity = raw["cash_cents"] + sum(c["qty"] * c["mark_cents"] for c in open_c) if marked else None
    external = sum((flows or {}).values())
    units, worst = unit_nav(nav_series, initial, flows)
    ceiling = cfg.get("equity_ceiling_cents")
    return {"initial_cents": initial, "cash_cents": raw["cash_cents"], "equity_cents": equity,
            "net_external_contribution_cents": external,
            "money_pnl_cents": None if equity is None else equity - initial - external,
            "fees_cents": sum(c["occurred_fees_cents"] for c in campaigns),
            "realized_fifo_cents": sum(c["realized_fifo_cents"] or 0 for c in campaigns),
            "unrealized_fifo_cents": sum(c["unrealized_fifo_cents"] or 0 for c in open_c) if marked else None,
            "stress_wealth_loss_cents": sum(c.get("current_wealth_loss_in_stress_cents") or 0 for c in open_c),
            "stress_note": "各持仓压力损失直接相加，不用某只股票的浮盈抵消另一只的压力亏损",
            "unit_nav": units, "max_drawdown_unit_nav": worst,
            "risk_basis_cents": None if equity is None else (min(equity, ceiling) if ceiling else min(equity, initial)),
            "capital_policy": cfg.get("capital_policy", "INITIAL_CAP"),
            "flows_note": "当前账本无外部资金进出记录；单位净值已支持按份额申赎，充值不产生收益、不重置回撤。"}


def report(db, day):
    """Every campaign in one ledger plus the account layer."""
    from . import performance
    with paper.database(db) as conn:
        raw = conn.execute("SELECT * FROM paper_account").fetchone()
        if not raw:
            return {"version": VERSION, "status": "UNCONFIGURED", "as_of": day}
        cfg = json.loads(raw["config"])
        plans = [json.loads(r[0]) for r in conn.execute("SELECT payload FROM paper_plans ORDER BY created_at")]
        campaigns = []
        for plan in plans:
            latest = conn.execute("SELECT payload FROM paper_valuations WHERE plan_id=? AND as_of<=? ORDER BY as_of DESC, rowid DESC LIMIT 1",
                                  (plan["id"], day)).fetchone()
            protect = protection_of(plan, json.loads(latest[0]) if latest else {})
            c = campaign(conn, plan, day, cfg, protect)
            if c:
                campaigns.append(c)
        series, _ = performance.nav_series(conn, raw["initial_cents"], plans)
        acct = account(conn, day, campaigns, series)
    return {"version": VERSION, "status": "READY", "as_of": day, "notice": NOTICE, "account": acct, "campaigns": campaigns}


def protection_of(plan, valuation):
    """The protection currently in force: the frozen exit policy's floor from the last valuation, else the plan's hard stop.
    Valuation floors are stored in raw price units (see daily.monitor_paper)."""
    floors = [valuation.get(k) for k in ("soft_floor", "hard_floor") if valuation.get(k)]
    if floors:
        return round(max(floors) * 100)
    return plan.get("stop_cents")
