"""Four-layer opportunity identity and cycle continuity. A new cycle inherits no memory."""
import json
from . import store

VERSION = "continuity_v1_20260920"
# Absence is not deletion: an unseen opportunity keeps its record until it expires.
UNSEEN_LIMIT = 5
OPEN_EVENTS = {"DISCOVERED", "REDISCOVERED", "WAITING", "CONFIRMED", "PRICE_EXCEEDED",
               "DATA_WAIT", "TEMPORARILY_UNSEEN"}
CLOSED_EVENTS = {"EXPIRED", "INVALIDATED", "CYCLE_ROTATED"}
LABELS = {
    "DISCOVERED": "首次发现", "REDISCOVERED": "新周期重新发现", "WAITING": "等待买点条件",
    "CONFIRMED": "条件确认可复核", "PRICE_EXCEEDED": "价格超出结构可接受范围",
    "DATA_WAIT": "数据待核", "TEMPORARILY_UNSEEN": "本次未扫描到",
    "EXPIRED": "超过未见期限，按过期处理", "INVALIDATED": "结构失效",
    "CYCLE_ROTATED": "旧周期结束，已开新周期",
}
ABSENCE = {"NOT_SCANNED": "本次未扫描到", "OFF_DISPLAY": "退出展示榜但仍有效",
           "STRUCTURE_FAILED": "结构失效"}


def _table(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS opportunity_continuity (
            opportunity_id TEXT NOT NULL, as_of TEXT NOT NULL, snapshot_id TEXT NOT NULL,
            event TEXT NOT NULL, payload TEXT NOT NULL,
            PRIMARY KEY (opportunity_id, as_of));
    """)


def identity(record):
    """Symbol, cycle, opportunity and local setup are four separate answers."""
    symbol = record["symbol"]
    cycle = record.get("cycle_id") or "unanchored"
    vector = (record.get("evidence") or {}).get("state_vector", {})
    return {"symbol": symbol, "cycle_id": cycle,
            "opportunity_id": record.get("opportunity_id") or store.digest([symbol, cycle])[:24],
            "setup_id": vector.get("setup_id") or record.get("setup_id") or "",
            "setup_kind": record.get("setup_kind") or vector.get("setup_kind", "UNKNOWN"),
            "anchored": cycle != "unanchored"}


def classify(record):
    """Today's event for a present opportunity, with the reason kept beside it."""
    state, blockers = record.get("state"), set(record.get("blockers") or [])
    bias = record.get("source_record", {}).get("route_bias_atr")
    try:
        exceeded = bias is not None and abs(float(bias)) > 2
    except (TypeError, ValueError):
        exceeded = False
    if state == "DATA_WAIT" or "DATA_WAIT" in blockers:
        return "DATA_WAIT", "当日行情或证据不完整"
    if state == "RISK_WATCH" or "STRUCTURE_RISK" in blockers:
        return "INVALIDATED", "结构风险或卖出结构告警"
    if exceeded:
        return "PRICE_EXCEEDED", f"距结构 {float(bias):.2f} ATR，超出可接受范围"
    if state == "PLAN_REVIEW":
        return "CONFIRMED", "条件齐备，可进入人工复核"
    return "WAITING", "；".join(sorted(blockers)) or "等待买点条件"


def _prior(conn, day):
    rows = conn.execute(
        "SELECT opportunity_id, as_of, event, payload FROM opportunity_continuity "
        "WHERE as_of<? ORDER BY as_of, rowid", (day,)).fetchall()
    latest = {}
    for row in rows:
        latest[row["opportunity_id"]] = {"as_of": row["as_of"], "event": row["event"],
                                         **json.loads(row["payload"])}
    return latest


def advance(payload, db=store.DB_PATH, sessions=None):
    """Emit one continuity event per opportunity per day; re-running a day is idempotent."""
    day, snapshot_id = payload["as_of"], payload.get("snapshot_id", "")
    with store.connect(db) as conn:
        _table(conn)
        prior = _prior(conn, day)
        events, rotations = [], []
        seen_cycle = {}
        for record in payload["records"]:
            ident = identity(record)
            oid = ident["opportunity_id"]
            seen_cycle[ident["symbol"]] = ident["cycle_id"]
            event, reason = classify(record)
            before = prior.get(oid)
            if before is None:
                # A prior closed cycle on the same symbol makes this a new round, not a continuation.
                earlier = [p for k, p in prior.items()
                           if p.get("symbol") == ident["symbol"] and p.get("cycle_id") != ident["cycle_id"]]
                event = "REDISCOVERED" if earlier else "DISCOVERED"
                reason = ("上一轮周期已结束，本轮为新机会，不继承旧周期记忆"
                          if earlier else "首次进入候选域")
                previous_opportunity = sorted(earlier, key=lambda p: p["as_of"])[-1]["opportunity_id"] if earlier else ""
            else:
                previous_opportunity = before.get("previous_opportunity_id", "")
            events.append({**ident, "as_of": day, "event": event, "reason": reason,
                           "previous_event": (before or {}).get("event", ""),
                           "previous_as_of": (before or {}).get("as_of", ""),
                           "previous_opportunity_id": previous_opportunity,
                           "state": record.get("state"), "blockers": record.get("blockers") or [],
                           "route_ids": record.get("route_ids", ""),
                           "strength_rank": record.get("strength_rank"),
                           "participation_rank": record.get("participation_rank"),
                           "displayed": bool(record.get("participation_rank") and record["participation_rank"] <= 15),
                           "absence": "", "last_seen": day, "snapshot_id": snapshot_id, "version": VERSION})
        present = {e["opportunity_id"] for e in events}
        order = sessions or []
        for oid, before in prior.items():
            if oid in present or before.get("event") in CLOSED_EVENTS:
                continue
            symbol, cycle = before.get("symbol", ""), before.get("cycle_id", "")
            if symbol in seen_cycle and seen_cycle[symbol] != cycle:
                events.append({**{k: before.get(k) for k in ("symbol", "cycle_id", "opportunity_id",
                                                             "setup_id", "setup_kind")},
                               "as_of": day, "event": "CYCLE_ROTATED",
                               "reason": "同一股票已开启新周期，旧机会不再延长以吸收后续涨幅",
                               "previous_event": before.get("event", ""), "previous_as_of": before.get("as_of", ""),
                               "previous_opportunity_id": before.get("previous_opportunity_id", ""),
                               "state": "CLOSED", "blockers": [], "route_ids": before.get("route_ids", ""),
                               "strength_rank": None, "participation_rank": None, "displayed": False,
                               "absence": "STRUCTURE_FAILED" if before.get("event") == "INVALIDATED" else "NOT_SCANNED",
                               "last_seen": before.get("last_seen", before.get("as_of", "")),
                               "snapshot_id": snapshot_id, "version": VERSION})
                rotations.append(oid)
                continue
            last_seen = before.get("last_seen") or before.get("as_of", "")
            gap = len([d for d in order if last_seen < d <= day]) if order else 1
            expired = gap >= UNSEEN_LIMIT
            events.append({**{k: before.get(k) for k in ("symbol", "cycle_id", "opportunity_id",
                                                         "setup_id", "setup_kind")},
                           "as_of": day, "event": "EXPIRED" if expired else "TEMPORARILY_UNSEEN",
                           "reason": (f"连续 {gap} 个交易日未再扫描到，按过期处理" if expired
                                      else f"本次未扫描到（第 {gap} 个交易日），记录保留"),
                           "previous_event": before.get("event", ""), "previous_as_of": before.get("as_of", ""),
                           "previous_opportunity_id": before.get("previous_opportunity_id", ""),
                           "state": "UNSEEN", "blockers": [], "route_ids": before.get("route_ids", ""),
                           "strength_rank": None, "participation_rank": None, "displayed": False,
                           "absence": "NOT_SCANNED", "last_seen": last_seen,
                           "snapshot_id": snapshot_id, "version": VERSION})
        for event in events:
            conn.execute("INSERT OR REPLACE INTO opportunity_continuity VALUES (?,?,?,?,?)",
                         (event["opportunity_id"], day, snapshot_id, event["event"], store.encoded(event)))
    counts = {}
    for event in events:
        counts[event["event"]] = counts.get(event["event"], 0) + 1
    return {"version": VERSION, "as_of": day, "snapshot_id": snapshot_id, "events": events,
            "summary": {"total": len(events), "present": len(present), "absent": len(events) - len(present),
                        "rotated": len(rotations), "by_event": counts},
            "labels": LABELS, "absence_labels": ABSENCE,
            "notice": "掉出展示榜不等于失效；未扫描到、退出展示与结构失效分别记录，记录不因排名下降被删除。"}


def timeline(opportunity_id, db=store.DB_PATH):
    """Every dated transition for one opportunity, plus its linked earlier round."""
    with store.connect(db) as conn:
        _table(conn)
        rows = [json.loads(r["payload"]) for r in conn.execute(
            "SELECT payload FROM opportunity_continuity WHERE opportunity_id=? ORDER BY as_of", (opportunity_id,))]
        linked = ""
        for row in rows:
            linked = row.get("previous_opportunity_id") or linked
        earlier = [json.loads(r["payload"]) for r in conn.execute(
            "SELECT payload FROM opportunity_continuity WHERE opportunity_id=? ORDER BY as_of", (linked,))] if linked else []
    return {"version": VERSION, "opportunity_id": opportunity_id, "events": rows,
            "previous_round": {"opportunity_id": linked, "events": earlier} if linked else None,
            "labels": LABELS}
