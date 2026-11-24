"""Corporate actions as explicit cash and share events; inferred from price, applied only when confirmed."""
import json
from . import store

VERSION = "corporate_actions_v1_20260920"
TOLERANCE = .001
# Above this factor the gap is treated as a share event; below it as cash. Both wait for confirmation.
SHARE_FACTOR = 1.05


def _table(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS corporate_actions (id TEXT PRIMARY KEY, plan_id TEXT NOT NULL, "
                 "day TEXT NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL)")


def detect(frame, since=""):
    """Days where the exchange preclose differs from the prior raw close beyond rounding."""
    events = []
    days = frame.date.astype(str).str[:10].tolist() if hasattr(frame.date, "str") else \
        frame.date.dt.strftime("%Y-%m-%d").tolist()
    close = frame.close.astype(float).tolist()
    preclose = frame.preclose.astype(float).tolist()
    for i in range(1, len(frame)):
        if days[i] <= since or not close[i - 1] or not preclose[i]:
            continue
        factor = close[i - 1] / preclose[i]
        if abs(factor - 1) > TOLERANCE:
            kind = "SHARE_EVENT" if factor >= SHARE_FACTOR else "CASH_EVENT"
            events.append({"day": days[i], "prior_close": close[i - 1], "preclose": preclose[i],
                           "factor": round(factor, 6), "kind": kind,
                           "basis": "PRICE_INFERRED_REQUIRES_MANUAL_CONFIRMATION"})
    return events


def record(conn, plan_id, event):
    """Register the inferred event; nothing in the ledger moves until it is confirmed."""
    _table(conn)
    identity = store.digest([plan_id, event["day"], event["factor"]])[:24]
    payload = {"id": identity, "version": VERSION, "plan_id": plan_id, **event, "status": "PENDING_CONFIRMATION",
               "applied": None}
    conn.execute("INSERT OR IGNORE INTO corporate_actions VALUES (?,?,?,?,?)",
                 (identity, plan_id, event["day"], "PENDING_CONFIRMATION", store.encoded(payload)))
    return payload


def confirm(identity, kind, db=store.DB_PATH, at=None, factor=None, cash_per_share_cents=None):
    """A human names the event and may state its exact terms; the ledger moves once, dated to the event day.

    The price-inferred factor is only a default. Lots are kept in current units afterwards; the recorded
    per-lot deltas and the event day let performance rebuild what was held before the event.
    """
    from . import paper
    if kind not in {"SHARE_EVENT", "CASH_EVENT", "NOT_AN_EVENT"}:
        raise ValueError("公司行动类型只能是 SHARE_EVENT / CASH_EVENT / NOT_AN_EVENT")
    if factor is not None and not float(factor) > 0:
        raise ValueError("股数比例必须为正数")
    with paper.database(db) as conn:
        _table(conn)
        row = conn.execute("SELECT payload FROM corporate_actions WHERE id=?", (identity,)).fetchone()
        if not row:
            raise ValueError("没有该公司行动记录")
        payload = json.loads(row[0])
        if payload["status"] != "PENDING_CONFIRMATION":
            return payload
        lots = [dict(r) for r in conn.execute(
            "SELECT * FROM paper_lots WHERE plan_id=? AND entry_day<? AND remaining>0", (payload["plan_id"], payload["day"]))]
        ratio = float(factor) if factor is not None else payload["factor"]
        per_share = (int(cash_per_share_cents) if cash_per_share_cents is not None
                     else int(round((payload["prior_close"] - payload["preclose"]) * 100)))
        applied = {"kind": kind, "at": at or store.now(), "effective_day": payload["day"],
                   "basis": "MANUAL_TERMS" if (factor is not None or cash_per_share_cents is not None) else "PRICE_INFERRED",
                   "factor": ratio if kind == "SHARE_EVENT" else None,
                   "cash_per_share_cents": per_share if kind == "CASH_EVENT" else None, "lots": []}
        for lot in lots:
            if kind == "SHARE_EVENT":
                new_qty = int(round(lot["remaining"] * ratio))
                new_price = int(round(lot["price_cents"] / ratio))
                conn.execute("UPDATE paper_lots SET remaining=?,quantity=quantity+?,price_cents=? WHERE id=?",
                             (new_qty, new_qty - lot["remaining"], new_price, lot["id"]))
                applied["lots"].append({"lot": lot["id"], "remaining": [lot["remaining"], new_qty],
                                        "price_cents": [lot["price_cents"], new_price],
                                        "cost_cents_before": lot["remaining"] * lot["price_cents"],
                                        "cost_cents_after": new_qty * new_price})
            elif kind == "CASH_EVENT":
                cash = lot["remaining"] * per_share
                conn.execute("UPDATE paper_account SET cash_cents=cash_cents+? WHERE id=1", (cash,))
                applied["lots"].append({"lot": lot["id"], "cash_cents": cash})
        applied["cash_cents"] = sum(l.get("cash_cents", 0) for l in applied["lots"])
        payload["status"] = "APPLIED" if kind != "NOT_AN_EVENT" else "DISMISSED"
        payload["applied"] = applied
        conn.execute("UPDATE corporate_actions SET status=?,payload=? WHERE id=?",
                     (payload["status"], store.encoded(payload), identity))
        paper.event(conn, "CORPORATE_ACTION_" + payload["status"], {"id": identity, "plan_id": payload["plan_id"],
                                                                  "day": payload["day"], "kind": kind})
    return payload


def relation(raw_fill_price_cents, factors):
    """How a ledger fill price maps onto an adjusted series after later events."""
    adjusted = raw_fill_price_cents
    for factor in factors:
        adjusted = adjusted / factor
    return {"raw_fill_price_cents": raw_fill_price_cents, "adjusted_equivalent_cents": int(round(adjusted)),
            "factors": list(factors), "note": "账本保存原始成交价；复权K线上的对应价按事件因子换算，二者不混用。"}


def pending_for(conn, plan_id):
    _table(conn)
    return [json.loads(r[0]) for r in conn.execute(
        "SELECT payload FROM corporate_actions WHERE plan_id=? AND status='PENDING_CONFIRMATION'", (plan_id,))]


def applied(conn):
    """Confirmed events are the only ones that move history: share deltas per lot, cash per event day."""
    _table(conn)
    return [json.loads(r[0]) for r in conn.execute(
        "SELECT payload FROM corporate_actions WHERE status='APPLIED' ORDER BY day")]


def pending(db=store.DB_PATH):
    with store.connect(db) as conn:
        _table(conn)
        return [json.loads(r[0]) for r in conn.execute(
            "SELECT payload FROM corporate_actions WHERE status='PENDING_CONFIRMATION' ORDER BY day")]
