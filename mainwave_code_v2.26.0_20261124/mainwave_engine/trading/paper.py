"""Paper-only order gateway. Integer cash; fills, not signals, change holdings."""
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
import re
from . import store

VERSION = "paper_gateway_v1_20260919"
MODEL = "NEXT_OPEN_FIXED_CAP_PAPER_ONLY"
ACTIVE = ("OPEN", "PARTIAL", "UNKNOWN", "EXIT_BLOCKED")
NOTICE = "独立模拟账户，非真实成交。撮合假设为已提交订单后的交易日开盘、固定单次股数上限；涨跌停开盘不假定成交，不模拟真实排队。旧虚拟持仓不自动迁入。"


class Conflict(ValueError):
    pass


def cents(value):
    try:
        d = Decimal(str(value))
        if not d.is_finite():
            raise ValueError("金额必须有限")
        return int((d * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    except Exception as exc:
        raise ValueError("金额无效") from exc


def fee(notional, cfg, side):
    commission = max(cfg["min_fee_cents"], math.ceil(notional * cfg["fee_bps"] / 10000))
    return commission + (math.ceil(notional * cfg["sell_tax_bps"] / 10000) if side == "SELL" else 0)


@contextmanager
def database(db=store.DB_PATH):
    with store.connect(db) as conn:
        # Individual statements preserve the encompassing immediate transaction.
        for sql in (
            "CREATE TABLE IF NOT EXISTS paper_account (id INTEGER PRIMARY KEY CHECK(id=1), initial_cents INTEGER NOT NULL, cash_cents INTEGER NOT NULL, created_at TEXT NOT NULL, config TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS paper_plans (id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL REFERENCES snapshots(id), opportunity_id TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS paper_orders (id TEXT PRIMARY KEY, plan_id TEXT NOT NULL REFERENCES paper_plans(id), side TEXT NOT NULL, quantity INTEGER NOT NULL, filled INTEGER NOT NULL, status TEXT NOT NULL, reserve_cents INTEGER NOT NULL, risk_cents INTEGER NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS paper_fills (id TEXT PRIMARY KEY, order_id TEXT NOT NULL REFERENCES paper_orders(id), day TEXT NOT NULL, quantity INTEGER NOT NULL, price_cents INTEGER NOT NULL, fee_cents INTEGER NOT NULL)",
            "CREATE TABLE IF NOT EXISTS paper_lots (id TEXT PRIMARY KEY REFERENCES paper_fills(id), plan_id TEXT NOT NULL REFERENCES paper_plans(id), symbol TEXT NOT NULL, entry_day TEXT NOT NULL, quantity INTEGER NOT NULL, remaining INTEGER NOT NULL, price_cents INTEGER NOT NULL, entry_fee_cents INTEGER NOT NULL)",
            "CREATE TABLE IF NOT EXISTS paper_events (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS paper_valuations (id TEXT PRIMARY KEY, as_of TEXT NOT NULL, plan_id TEXT NOT NULL REFERENCES paper_plans(id), payload TEXT NOT NULL)",
        ):
            conn.execute(sql)
        yield conn


def event(conn, kind, body, key=None):
    identity = key or store.digest([kind, body])
    conn.execute("INSERT OR IGNORE INTO paper_events VALUES (?,?,?,?)", (identity, store.now(), kind, store.encoded(body)))


def initialize(body, db=store.DB_PATH, at=None):
    fields = {"capital", "fee_bps", "min_fee", "sell_tax_bps", "single_risk_pct", "total_risk_pct", "symbol_cap_pct", "theme_cap_pct", "gross_cap_pct", "max_fill_lots"}
    # Optional account protocol keys (a compounding account); absent keys keep the original behaviour exactly.
    optional = {"capital_policy", "equity_ceiling", "stock_seats"}
    if not isinstance(body, dict) or not fields <= set(body) <= fields | optional:
        raise ValueError("需明确初始本金、费率、单笔/总风险、个股/题材/总敞口及模拟单次成交手数")
    extra = {k: body[k] for k in optional if k in body}
    body = {k: v for k, v in body.items() if k in fields}
    if any(isinstance(v, bool) or not isinstance(v, (int, float, str)) for v in body.values()):
        raise ValueError("Account settings must be numeric")
    cfg = {k: float(v) for k, v in body.items()}
    policy = extra.get("capital_policy", "INITIAL_CAP")
    if policy not in ("INITIAL_CAP", "COMPOUND_CEILING"):
        raise ValueError("资本口径只能是 INITIAL_CAP 或 COMPOUND_CEILING")
    ceiling = float(extra.get("equity_ceiling", cfg["capital"]))
    seats = extra.get("stock_seats")
    if not math.isfinite(ceiling) or ceiling < cfg["capital"]:
        raise ValueError("扩容上限不得低于初始本金")
    if seats is not None and (isinstance(seats, bool) or not isinstance(seats, int) or not 1 <= seats <= 10):
        raise ValueError("股票席位需为1至10的整数")
    if not all(math.isfinite(v) for v in cfg.values()):
        raise ValueError("配置不能含空值或非有限数")
    if not 1000 <= cfg["capital"] <= 100000000 or not 0 < cfg["single_risk_pct"] <= cfg["total_risk_pct"] <= 20:
        raise ValueError("本金或风险范围无效")
    if not 0 < cfg["symbol_cap_pct"] <= cfg["theme_cap_pct"] <= cfg["gross_cap_pct"] <= 100:
        raise ValueError("敞口需满足 个股 <= 题材 <= 总敞口 <= 100%")
    if any(not 0 <= cfg[k] <= 100 for k in ("fee_bps", "sell_tax_bps", "min_fee")):
        raise ValueError("模拟费率/最低手续费超出允许范围")
    if not 1 <= cfg["max_fill_lots"] <= 1000 or cfg["max_fill_lots"] % 1:
        raise ValueError("模拟成交手数需为1至1000的整数")
    capital = cents(cfg.pop("capital"))
    cfg["min_fee_cents"] = cents(cfg.pop("min_fee"))
    cfg.update(model=MODEL, version=VERSION, first_fraction=.5, lot_size=100,
               additions_enabled=False, live_orders_enabled=False)
    if extra:
        cfg.update(capital_policy=policy, equity_ceiling_cents=cents(ceiling), stock_seats=seats)
    with database(db) as conn:
        if conn.execute("SELECT 1 FROM paper_account").fetchone():
            raise Conflict("模拟账户已初始化；不允许重置本金或覆盖历史账本")
        created = at or store.now()
        conn.execute("INSERT INTO paper_account VALUES (1,?,?,?,?)", (capital, capital, created, store.encoded(cfg)))
        event(conn, "ACCOUNT_INITIALIZED", {"capital_cents": capital, "config": cfg})
    return view(db)


def _account(conn):
    row = conn.execute("SELECT * FROM paper_account WHERE id=1").fetchone()
    if not row:
        raise Conflict("尚未配置模拟账户；目前只保存观察和排名")
    return dict(row), json.loads(row["config"])


def source_fingerprint():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (store.ROOT / "mainwave_engine/exit_policy.py", store.ROOT / "mainwave_engine/pullback_plan.py")}


def _exposure(conn):
    values = []
    for lot in conn.execute("SELECT l.*, p.payload FROM paper_lots l JOIN paper_plans p ON p.id=l.plan_id WHERE remaining>0"):
        p = json.loads(lot["payload"])
        valuation = conn.execute("SELECT payload FROM paper_valuations WHERE plan_id=? ORDER BY as_of DESC,rowid DESC LIMIT 1", (lot["plan_id"],)).fetchone()
        v = json.loads(valuation[0]) if valuation else {}
        mark = v.get("mark_cents", lot["price_cents"])
        values.append({"symbol": lot["symbol"], "theme": p["theme"],
                       "cost": lot["remaining"] * max(mark, lot["price_cents"]), "market_value": lot["remaining"] * mark,
                       "valuation_day": v.get("day"), "valuation_status": v.get("status"),
                       "risk": lot["remaining"] * max(0, mark - p["stop_cents"])})
    for o in conn.execute("SELECT o.*,p.payload AS plan FROM paper_orders o JOIN paper_plans p ON p.id=o.plan_id WHERE o.side='BUY' AND o.status IN ('OPEN','PARTIAL','UNKNOWN')"):
        p = json.loads(o["plan"])
        values.append({"symbol": p["symbol"], "theme": p["theme"], "cost": o["reserve_cents"], "risk": o["risk_cents"]})
    return values


def submit(identity, symbol, db=store.DB_PATH, at=None, book="MANUAL"):
    """Explicit approval from UI/CLI (MANUAL) or the frozen mechanical rule (AUTO); never both in one ledger."""
    if book not in {"MANUAL", "AUTO", "NEW100K"}:
        raise ValueError("账本只能是 MANUAL、AUTO 或 NEW100K")
    at = at or store.now()
    with database(db) as conn:
        account, cfg = _account(conn)
        snap = store.snapshot(conn, identity)
        current = store.latest(conn)
        if current["snapshot_id"] != identity:
            raise Conflict("快照已更新，请按最新证据重新复核")
        if snap["recorded_at"] > at or snap["as_of"] > at[:10] or (datetime.fromisoformat(at[:10]) - datetime.fromisoformat(snap["as_of"])).days > 7:
            raise Conflict("快照时间无效或过期")
        row = next((r for r in snap["records"] if r["symbol"] == symbol), None)
        if not row or not row["paper_plan_eligible"]:
            raise Conflict("当前机会未通过回踩、市场、数据与风险条件")
        scope = row.get("alpha_scope") or ranking_scope(row.get("route_ids", ""))
        if book in ("AUTO", "NEW100K") and scope == "R3_RESEARCH":
            raise Conflict("R3 仅为研究线索，不进入自动策略账")
        if not re.fullmatch(r"(?:sh_60|sz_00)\d{4}", symbol) or any(x in row["name"].upper() for x in ("ST", "退")):
            raise Conflict("首期模拟执行仅普通主板；其他板块继续观察")
        plan_id = store.digest([row["opportunity_id"], VERSION])[:24]
        if conn.execute("SELECT 1 FROM paper_plans WHERE id=?", (plan_id,)).fetchone():
            return {"plan_id": plan_id, "created": False}
        if any(e["symbol"] == symbol for e in _exposure(conn)):
            raise Conflict("该股票已有持仓或未决买单，不叠加另一模式预算")
        w = row["window"]
        low, limit, stop, atr = (cents(w.get(k)) for k in ("lower", "upper", "invalid", "frozen_atr"))
        if not 0 < stop < low <= limit or atr <= 0:
            raise ValueError("冻结结构边界或ATR无效")
        exposures = _exposure(conn)
        if any(e.get("market_value") is not None and (e.get("valuation_day") != snap["as_of"] or e.get("valuation_status") != "HOLD") for e in exposures):
            raise Conflict("已有持仓缺少同日有效估值，先复核持仓风险再开新仓")
        if conn.execute("SELECT 1 FROM paper_orders WHERE status='UNKNOWN' OR (side='SELL' AND status IN ('OPEN','PARTIAL','EXIT_BLOCKED'))").fetchone():
            raise Conflict("先处理未决订单和退出风险，再申请新仓")
        reserved = sum(r[0] for r in conn.execute("SELECT reserve_cents FROM paper_orders"))
        equity = account["cash_cents"] + sum(x.get("market_value", 0) for x in exposures)
        # INITIAL_CAP (original): gains never widen the base. COMPOUND_CEILING: losses shrink it at once, gains widen it
        # up to the approved ceiling, which only a reviewed protocol change raises.
        capital = (min(equity, cfg["equity_ceiling_cents"]) if cfg.get("capital_policy") == "COMPOUND_CEILING"
                   else min(account["initial_cents"], equity))
        seats = cfg.get("stock_seats")
        if seats and len({e["symbol"] for e in exposures}) >= seats:
            event(conn, "PLAN_REJECTED", {"symbol": symbol, "snapshot_id": identity, "reason": "NO_SEAT_STOCK"})
            return {"created": False, "reason": f"股票席位已满（{int(seats)} 只，含未决买单）"}
        total = sum(x["cost"] for x in exposures)
        theme = sum(x["cost"] for x in exposures if x["theme"] == row["theme"])
        budget = min(account["cash_cents"] - reserved, capital * cfg["symbol_cap_pct"] / 100 * cfg["first_fraction"],
                     capital * cfg["theme_cap_pct"] / 100 - theme, capital * cfg["gross_cap_pct"] / 100 - total)
        risk_budget = min(capital * cfg["single_risk_pct"] / 100,
                          capital * cfg["total_risk_pct"] / 100 - sum(x["risk"] for x in exposures))
        # Fixed hard floor may be below the structure; size using the worse boundary.
        hard = min(stop, low - 2 * atr)
        if hard <= 0:
            raise ValueError("风险边界无效")
        quantity = max(0, int(min(budget / limit, risk_budget / (limit - hard))) // 100 * 100)
        while quantity and (quantity * limit + fee(quantity * limit, cfg, "BUY") > budget or
                            quantity * (limit - hard) + fee(quantity * limit, cfg, "BUY") + fee(quantity * hard, cfg, "SELL") > risk_budget):
            quantity -= 100
        if quantity < 100:
            event(conn, "PLAN_REJECTED", {"symbol": symbol, "snapshot_id": identity, "reason": "BUDGET_BELOW_MINIMUM"})
            return {"created": False, "reason": "现金或风险预算不足最小申报量"}
        from .. import exit_policy as XP
        from . import rules as RL
        payload = {"id": plan_id, "snapshot_id": identity, "opportunity_id": row["opportunity_id"],
                   "symbol": symbol, "name": row["name"], "theme": row["theme"], "route_ids": row["route_ids"],
                   "book": book, "alpha_scope": scope, "strength_rank": row.get("strength_rank"),
                   "participation_rank": row.get("participation_rank"), "strength_score": row.get("strength_score"),
                   "mode": "TREND_PULLBACK", "setup_kind": row["setup_kind"], "entry_day": snap["as_of"],
                   "lower_cents": low, "limit_cents": limit, "stop_cents": hard, "structure_cents": stop,
                   "atr_cents": atr, "quantity": quantity, "version": VERSION, "created_at": at,
                   "knowledge_registry_id": snap["knowledge_registry_id"], "ranking_version": snap["version"],
                   "knowledge_ids": ["F06-K04", "F10-K04", "F02-K07"], "exit_version": XP.VERSION,
                   "exit_rules": XP.RULES, "exit_code_hashes": source_fingerprint(),
                   "first_fraction": .5, "additions_enabled": True, "model": MODEL,
                   "planned_budget_cents": quantity * limit * 2,
                   "rule_fingerprint": RL.fingerprint(),
                   "exit_execution": "收盘识别风险后下个可交易开盘执行；不是盘中止损成交；部分止盈也按下一开盘模拟。"}
        conn.execute("INSERT INTO paper_plans VALUES (?,?,?,?,?)", (plan_id, identity, row["opportunity_id"], at, store.encoded(payload)))
        oid = plan_id + ":initial"
        reserve = quantity * limit + fee(quantity * limit, cfg, "BUY")
        conn.execute("INSERT INTO paper_orders VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (oid, plan_id, "BUY", quantity, 0, "OPEN", reserve,
                      quantity * (limit - hard) + fee(quantity * limit, cfg, "BUY") + fee(quantity * hard, cfg, "SELL"),
                      at, store.encoded({"reason": "INITIAL_HALF_BUDGET", "model": MODEL})))
        event(conn, "ORDER_APPROVED", {"order_id": oid, "plan": payload, "reserved_cents": reserve})
        return {"plan_id": plan_id, "order_id": oid, "created": True}


def ranking_scope(route_ids):
    from . import ranking
    return ranking.alpha_scope(route_ids)


def _cancel(conn, oid, reason):
    conn.execute("UPDATE paper_orders SET status='CANCELLED',reserve_cents=0,risk_cents=0 WHERE id=?", (oid,))
    event(conn, "ORDER_CANCELLED", {"order_id": oid, "reason": reason})


def cancel(oid, db=store.DB_PATH):
    with database(db) as conn:
        o = conn.execute("SELECT * FROM paper_orders WHERE id=?", (oid,)).fetchone()
        if not o or o["side"] != "BUY" or o["status"] not in ACTIVE or o["status"] == "UNKNOWN":
            raise Conflict("只能取消已知状态的未完成模拟买单；退出任务不可在此取消")
        _cancel(conn, oid, "USER_CANCELLED")
    return view(db)


def order_band(order_payload, plan):
    """A conditional add carries its own confirmed band and reference day; the initial uses the plan window."""
    return (order_payload.get("min_price_cents", plan["lower_cents"]),
            order_payload.get("max_price_cents", plan["limit_cents"]),
            order_payload.get("reference_day", plan["entry_day"]))


def apply_fill(conn, oid, fill_id, day, quantity, price):
    """Internal paper adapter only. Callbacks are validated and deduplicated transactionally."""
    old = conn.execute("SELECT * FROM paper_fills WHERE id=?", (fill_id,)).fetchone()
    price = cents(price)
    if old:
        if (old["order_id"], old["day"], old["quantity"], old["price_cents"]) != (oid, day, quantity, price):
            raise Conflict("相同成交ID内容冲突")
        return False
    account, cfg = _account(conn)
    o = conn.execute("SELECT * FROM paper_orders WHERE id=?", (oid,)).fetchone()
    if not o or o["status"] not in ACTIVE or isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0 or quantity > o["quantity"] - o["filled"] or price <= 0:
        raise ValueError("成交数量或订单状态无效")
    if day < o["created_at"][:10]:
        raise ValueError("成交不能早于委托")
    plan = json.loads(conn.execute("SELECT payload FROM paper_plans WHERE id=?", (o["plan_id"],)).fetchone()[0])
    notional = quantity * price
    charge = fee(notional, cfg, o["side"])
    if o["side"] == "BUY":
        low, high, _ = order_band(json.loads(o["payload"]), plan)
        if quantity % 100 or price > high or price < low:
            raise ValueError("买入成交违反冻结价格/数量限制")
        other_reserve = sum(r[0] for r in conn.execute("SELECT reserve_cents FROM paper_orders WHERE id<>?", (oid,)))
        if notional + charge > account["cash_cents"] - other_reserve:
            raise Conflict("成交超过可用资金")
        delta = -notional - charge
    else:
        lots = list(conn.execute("SELECT * FROM paper_lots WHERE plan_id=? AND remaining>0 AND entry_day<? ORDER BY entry_day,id", (o["plan_id"], day)))
        if sum(r["remaining"] for r in lots) < quantity:
            raise Conflict("可售不足或T+1限制")
        delta = notional - charge
        remaining = quantity
        for lot in lots:
            used = min(remaining, lot["remaining"])
            conn.execute("UPDATE paper_lots SET remaining=remaining-? WHERE id=?", (used, lot["id"]))
            remaining -= used
            if not remaining:
                break
    conn.execute("INSERT INTO paper_fills VALUES (?,?,?,?,?,?)", (fill_id, oid, day, quantity, price, charge))
    if o["side"] == "BUY":
        conn.execute("INSERT INTO paper_lots VALUES (?,?,?,?,?,?,?,?)",
                     (fill_id, o["plan_id"], plan["symbol"], day, quantity, quantity, price, charge))
    filled = o["filled"] + quantity
    left = o["quantity"] - filled
    cap = order_band(json.loads(o["payload"]), plan)[1] if o["side"] == "BUY" else plan["limit_cents"]
    reserve = left * cap + fee(left * cap, cfg, "BUY") if left and o["side"] == "BUY" else 0
    risk = left * (cap - plan["stop_cents"]) + fee(left * cap, cfg, "BUY") + fee(left * plan["stop_cents"], cfg, "SELL") if reserve else 0
    if reserve > account["cash_cents"] + delta - (other_reserve if o["side"] == "BUY" else 0):
        raise Conflict("部分成交后的最低手续费导致剩余预留不足")
    conn.execute("UPDATE paper_account SET cash_cents=cash_cents+? WHERE id=1", (delta,))
    conn.execute("UPDATE paper_orders SET filled=?,status=?,reserve_cents=?,risk_cents=? WHERE id=?",
                 (filled, "FILLED" if left == 0 else "PARTIAL", reserve, risk, oid))
    event(conn, "FILL", {"fill_id": fill_id, "order_id": oid, "day": day, "quantity": quantity, "price_cents": price, "fee_cents": charge})
    return True


def queue_exit(conn, plan_id, day, reason, fraction=1.):
    existing = conn.execute("SELECT * FROM paper_orders WHERE plan_id=? AND side='SELL' AND status IN ('OPEN','PARTIAL','UNKNOWN','EXIT_BLOCKED')", (plan_id,)).fetchall()
    lots = list(conn.execute("SELECT remaining FROM paper_lots WHERE plan_id=?", (plan_id,)))
    total = sum(r[0] for r in lots)
    if existing:
        if fraction < 1 or any(x["status"] == "UNKNOWN" for x in existing) or sum(x["quantity"] - x["filled"] for x in existing) >= total:
            return
        for o in existing:
            _cancel(conn, o["id"], "ESCALATE_TO_FULL_EXIT")
    qty = total
    if fraction < 1:
        qty = int(qty * fraction) // 100 * 100
    if qty <= 0:
        return
    oid = f"{plan_id}:exit:{day}:{reason}"
    if conn.execute("SELECT 1 FROM paper_orders WHERE id=?", (oid,)).fetchone():
        return
    conn.execute("INSERT INTO paper_orders VALUES (?,?,?,?,?,?,?,?,?,?)",
                 (oid, plan_id, "SELL", qty, 0, "OPEN", 0, 0, day + "T15:00:00+08:00", store.encoded({"reason": reason})))
    event(conn, "EXIT_REQUIRED", {"order_id": oid, "reason": reason, "quantity": qty})


def process_open(conn, day, bars):
    _, cfg = _account(conn)
    for o in conn.execute("SELECT o.*,p.payload AS plan FROM paper_orders o JOIN paper_plans p ON p.id=o.plan_id WHERE o.status IN ('OPEN','PARTIAL','EXIT_BLOCKED') ORDER BY o.side DESC,o.created_at,o.id").fetchall():
        if o["created_at"] >= day + "T09:30:00+08:00":
            continue
        fill_id = store.digest([o["id"], day, MODEL])
        if conn.execute("SELECT 1 FROM paper_fills WHERE id=?", (fill_id,)).fetchone():
            continue
        p = json.loads(o["plan"])
        bar = bars.get(p["symbol"], {})
        if p["exit_code_hashes"] != source_fingerprint() and o["side"] == "BUY":
            _cancel(conn, o["id"], "FROZEN_RULE_MISMATCH")
            continue
        if bar.get("date") != day or not bar.get("raw_verified"):
            if o["side"] == "SELL":
                conn.execute("UPDATE paper_orders SET status='EXIT_BLOCKED' WHERE id=?", (o["id"],))
            event(conn, "DATA_WAIT", {"order_id": o["id"], "day": day, "reason": "NO_VERIFIED_BAR"})
            continue
        op, pre = cents(bar["open"]), cents(bar["preclose"])
        # First deployment is ordinary main-board paper only. Unknown security status blocks matching.
        blocked = not bar.get("ordinary_mainboard") or bar.get("corporate_action") or not pre > 0
        at_limit = op >= round(pre * 1.1) - 1 if o["side"] == "BUY" else op <= round(pre * .9) + 1
        if blocked or at_limit:
            if o["side"] == "BUY":
                _cancel(conn, o["id"], "SECURITY_OR_LIMIT_NOT_FILLABLE")
            else:
                conn.execute("UPDATE paper_orders SET status='EXIT_BLOCKED' WHERE id=?", (o["id"],))
                event(conn, "EXIT_BLOCKED", {"order_id": o["id"], "day": day, "reason": "SECURITY_OR_LIMIT"})
            continue
        low, high, reference = order_band(json.loads(o["payload"]), p)
        if o["side"] == "BUY" and (paper_day_gap(o["created_at"], day) > 7 or bar.get("previous_market_day") != reference):
            _cancel(conn, o["id"], "MISSED_NEXT_SESSION_OR_EXPIRED")
            continue
        if o["side"] == "BUY" and not low <= op <= high:
            _cancel(conn, o["id"], "OPEN_OUTSIDE_FROZEN_WINDOW")
            continue
        qty = min(o["quantity"] - o["filled"], int(cfg["max_fill_lots"]) * 100)
        if o["side"] == "SELL":
            sellable = conn.execute("SELECT COALESCE(SUM(remaining),0) FROM paper_lots WHERE plan_id=? AND entry_day<?", (o["plan_id"], day)).fetchone()[0]
            qty = min(qty, sellable)
        if qty:
            apply_fill(conn, o["id"], fill_id, day, qty, op / 100)
        if o["side"] == "BUY":
            left = conn.execute("SELECT status FROM paper_orders WHERE id=?", (o["id"],)).fetchone()[0]
            if left != "FILLED":
                _cancel(conn, o["id"], "PAPER_IOC_REMAINDER_CANCELLED")


def paper_day_gap(created, day):
    return (datetime.fromisoformat(day) - datetime.fromisoformat(created[:10])).days


def view(db=store.DB_PATH, full=False):
    with database(db) as conn:
        raw = conn.execute("SELECT * FROM paper_account").fetchone()
        result = {"configured": bool(raw), "notice": NOTICE, "version": VERSION, "live_orders_enabled": False,
                  "orders": [], "positions": [], "fills": [], "events": []}
        if not raw:
            return result
        result["account"] = dict(raw)
        result["account"]["config"] = json.loads(raw["config"])
        result["orders"] = [{**dict(o), "payload": json.loads(o["payload"])} for o in conn.execute("SELECT * FROM paper_orders ORDER BY created_at DESC,id")]
        result["fills"] = [dict(r) for r in conn.execute("SELECT f.*,o.side FROM paper_fills f JOIN paper_orders o ON o.id=f.order_id ORDER BY day DESC,f.rowid DESC")]
        for p in conn.execute("SELECT * FROM paper_plans ORDER BY created_at DESC"):
            lots = [dict(r) for r in conn.execute("SELECT * FROM paper_lots WHERE plan_id=?", (p["id"],))]
            if not lots:
                continue
            plan = json.loads(p["payload"])
            valuation = conn.execute("SELECT payload FROM paper_valuations WHERE plan_id=? ORDER BY as_of DESC,rowid DESC LIMIT 1", (p["id"],)).fetchone()
            result["positions"].append({"plan": plan, "lots": lots, "remaining": sum(r["remaining"] for r in lots),
                                        "valuation": json.loads(valuation[0]) if valuation else None})
        result["events"] = [{**dict(r), "payload": json.loads(r["payload"])} for r in conn.execute("SELECT * FROM paper_events ORDER BY rowid DESC" + ("" if full else " LIMIT 100"))]
        result["account"]["reserved_cents"] = sum(o["reserve_cents"] for o in result["orders"])
        result["account"]["available_cents"] = raw["cash_cents"] - result["account"]["reserved_cents"]
        expected_cash = raw["initial_cents"] + sum((-f["quantity"] * f["price_cents"] if f["side"] == "BUY" else f["quantity"] * f["price_cents"]) - f["fee_cents"] for f in result["fills"])
        bought = sum(f["quantity"] for f in result["fills"] if f["side"] == "BUY")
        sold = sum(f["quantity"] for f in result["fills"] if f["side"] == "SELL")
        quantities_ok = bought - sold == sum(p["remaining"] for p in result["positions"])
        result["reconciled"] = expected_cash == raw["cash_cents"] and result["account"]["available_cents"] >= 0 and quantities_ok
        active = [p for p in result["positions"] if p["remaining"]]
        known = all(p["valuation"] and p["valuation"].get("mark_cents") and p["valuation"]["status"] != "DATA_WAIT" for p in active)
        result["account"]["equity_cents"] = raw["cash_cents"] + sum(p["valuation"]["mark_cents"] * p["remaining"] for p in active) if known else None
        return result
