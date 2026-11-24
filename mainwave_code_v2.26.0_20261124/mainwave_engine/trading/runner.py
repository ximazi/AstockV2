"""Light runner: dedupe lock, checkpoints, resume and reconciliation. Risk before new ideas."""
import json
import os
import socket
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from . import store

VERSION = "light_runner_v1_20260920"
LOCK_MINUTES = 30
STAGES = ("PREMARKET", "INTRADAY_1430", "AFTER_CLOSE")
# Risk and expiring work is processed before any new candidate work.
ORDER = ("PENDING_EXITS", "POSITION_RISK", "EXPIRING_TASKS", "NEW_CANDIDATES")


def _table(conn):
    for sql in (
        "CREATE TABLE IF NOT EXISTS run_locks (name TEXT PRIMARY KEY, owner TEXT NOT NULL, "
        "acquired_at TEXT NOT NULL, heartbeat TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS run_checkpoints (id TEXT PRIMARY KEY, day TEXT NOT NULL, "
        "stage TEXT NOT NULL, step TEXT NOT NULL, status TEXT NOT NULL, updated_at TEXT NOT NULL, "
        "payload TEXT NOT NULL)",
    ):
        conn.execute(sql)


def owner():
    return f"{socket.gethostname()}:{os.getpid()}"


def acquire(name, db=store.DB_PATH, at=None, minutes=LOCK_MINUTES):
    """A repeated schedule must not create a second run; a dead run is reclaimed by timeout."""
    at = at or store.now()
    with store.connect(db) as conn:
        _table(conn)
        row = conn.execute("SELECT * FROM run_locks WHERE name=?", (name,)).fetchone()
        me = owner()
        if row:
            stale = datetime.fromisoformat(at) - datetime.fromisoformat(row["heartbeat"]) > timedelta(minutes=minutes)
            if row["owner"] == me:
                conn.execute("UPDATE run_locks SET heartbeat=? WHERE name=?", (at, name))
                return {"acquired": True, "reentrant": True, "owner": me}
            if not stale:
                return {"acquired": False, "owner": row["owner"], "since": row["acquired_at"],
                        "reason": "ANOTHER_RUN_IN_PROGRESS"}
            conn.execute("UPDATE run_locks SET owner=?,acquired_at=?,heartbeat=? WHERE name=?", (me, at, at, name))
            return {"acquired": True, "reclaimed": True, "owner": me, "previous": row["owner"]}
        conn.execute("INSERT INTO run_locks VALUES (?,?,?,?)", (name, me, at, at))
        return {"acquired": True, "owner": me}


def renew(name, db=store.DB_PATH, at=None):
    """Heartbeat from a long run, so a live holder is never mistaken for a dead one and reclaimed."""
    with store.connect(db) as conn:
        _table(conn)
        conn.execute("UPDATE run_locks SET heartbeat=? WHERE name=? AND owner=?", (at or store.now(), name, owner()))
    return {"renewed": True, "name": name}


def release(name, db=store.DB_PATH):
    with store.connect(db) as conn:
        _table(conn)
        conn.execute("DELETE FROM run_locks WHERE name=? AND owner=?", (name, owner()))
    return {"released": True, "name": name}


def checkpoint(day, stage, step, status, body=None, db=store.DB_PATH, at=None):
    identity = f"{day}:{stage}:{step}"
    with store.connect(db) as conn:
        _table(conn)
        conn.execute("INSERT INTO run_checkpoints VALUES (?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                     "status=excluded.status,updated_at=excluded.updated_at,payload=excluded.payload",
                     (identity, day, stage, step, status, at or store.now(), store.encoded(body or {})))
    return {"id": identity, "status": status}


def completed(day, stage, step, db=store.DB_PATH):
    with store.connect(db) as conn:
        _table(conn)
        row = conn.execute("SELECT status FROM run_checkpoints WHERE id=?",
                           (f"{day}:{stage}:{step}",)).fetchone()
    return bool(row) and row[0] == "DONE"


def freshness(day, data_root=None, output=None):
    """Quote, index and theme freshness are judged apart; partial staleness names its effect."""
    data_root = Path(data_root or store.ROOT / "data")
    output = Path(output or store.ROOT / "results/mainwave_engine" / day.replace("-", ""))
    sources = {}
    index = data_root / "000300_SH_index.csv"
    if index.exists():
        frame = pd.read_csv(index, usecols=["trade_date"])
        days = {str(x) if "-" in str(x) else f"{str(x)[:4]}-{str(x)[4:6]}-{str(x)[6:8]}" for x in frame.trade_date}
        sources["index"] = {"fresh": day in days, "impact": "缺指数则市场闸门与超额基准不可用"}
    else:
        sources["index"] = {"fresh": False, "impact": "缺指数则市场闸门与超额基准不可用"}
    universe = output / "analysis_universe.json"
    if universe.exists():
        payload = json.loads(universe.read_text(encoding="utf-8-sig"))
        sources["quotes"] = {"fresh": payload.get("as_of") == day,
                             "impact": "候选行情不是当日则排名与买点复核作废"}
        themed = sum(1 for r in payload.get("records", []) if r.get("belonging_theme"))
        sources["themes"] = {"fresh": bool(themed), "coverage": themed,
                             "impact": "题材缺失则同题材比较域收窄，按不足样本处理"}
    else:
        sources["quotes"] = {"fresh": False, "impact": "当日分析结果缺失"}
        sources["themes"] = {"fresh": False, "impact": "当日分析结果缺失"}
    stale = [k for k, v in sources.items() if not v["fresh"]]
    return {"version": VERSION, "as_of": day, "sources": sources, "stale": stale,
            "status": "FRESH" if not stale else "PARTIAL_STALE",
            "note": "部分过期不等于全部不可用；受影响能力已逐项列出。"}


def pending(day, db=store.DB_PATH):
    """The next task is always answerable, in risk-first order."""
    from . import paper
    tasks = []
    with paper.database(db) as conn:
        for row in conn.execute("SELECT o.id,o.side,o.status,p.payload FROM paper_orders o "
                                "JOIN paper_plans p ON p.id=o.plan_id "
                                "WHERE o.status IN ('OPEN','PARTIAL','EXIT_BLOCKED','UNKNOWN')"):
            plan = json.loads(row["payload"])
            kind = ("PENDING_EXITS" if row["side"] == "SELL" else "NEW_CANDIDATES")
            tasks.append({"group": kind, "order_id": row["id"], "symbol": plan["symbol"],
                          "side": row["side"], "status": row["status"],
                          "action": "退出受阻，待下个可交易开盘重试" if row["status"] == "EXIT_BLOCKED"
                          else "等待下个开盘撮合"})
        for row in conn.execute("SELECT plan_id,payload FROM paper_valuations WHERE as_of=?", (day,)):
            valuation = json.loads(row["payload"])
            if valuation.get("status") in {"EXIT_REQUIRED", "REDUCE_REQUIRED", "DATA_WAIT"}:
                tasks.append({"group": "POSITION_RISK", "plan_id": row["plan_id"],
                              "status": valuation["status"],
                              "action": valuation.get("reason") or "；".join(valuation.get("reasons", []))})
    rank = {name: n for n, name in enumerate(ORDER)}
    tasks.sort(key=lambda t: (rank.get(t["group"], 9), t.get("symbol", "")))
    return {"version": VERSION, "as_of": day, "order": list(ORDER), "tasks": tasks,
            "next": tasks[0] if tasks else None,
            "note": "先处理退出与持仓风险，再处理新增机会；重型研究不阻塞持仓检查。"}


def reconcile(day, db=store.DB_PATH):
    """After a restart, cash, lots and orders must still agree with the fills."""
    from . import paper
    with paper.database(db) as conn:
        account = conn.execute("SELECT * FROM paper_account").fetchone()
        if not account:
            return {"version": VERSION, "status": "UNCONFIGURED"}
        fills = [dict(r) for r in conn.execute("SELECT f.*,o.side FROM paper_fills f JOIN paper_orders o ON o.id=f.order_id")]
        reserved = sum(r[0] for r in conn.execute(
            "SELECT reserve_cents FROM paper_orders WHERE status IN ('OPEN','PARTIAL','UNKNOWN')"))
        lots = sum(r[0] for r in conn.execute("SELECT quantity FROM paper_lots"))
        sold = sum(r[0] for r in conn.execute("SELECT quantity FROM paper_lots")) - \
            sum(r[0] for r in conn.execute("SELECT remaining FROM paper_lots"))
        spent = sum(f["quantity"] * f["price_cents"] + f["fee_cents"] for f in fills if f["side"] == "BUY")
        got = sum(f["quantity"] * f["price_cents"] - f["fee_cents"] for f in fills if f["side"] == "SELL")
        # Confirmed cash events are ledger cash too; without them a dividend reads as a mismatch.
        from . import corporate_actions
        dividends = sum(sum(l.get("cash_cents", 0) for l in a["applied"]["lots"])
                        for a in corporate_actions.applied(conn) if a["applied"]["kind"] == "CASH_EVENT")
        expected = account["initial_cents"] - spent + got + dividends
        duplicates = len(fills) - len({f["id"] for f in fills})
        return {"version": VERSION, "as_of": day, "status": "CONSISTENT" if expected == account["cash_cents"] and not duplicates else "MISMATCH",
                "cash_cents": account["cash_cents"], "expected_cash_cents": expected, "corporate_cash_cents": dividends,
                "difference_cents": account["cash_cents"] - expected, "reserved_cents": reserved,
                "bought_lots": lots, "sold_quantity": sold, "duplicate_fills": duplicates,
                "note": "重复调度不重复记账；不一致时保留原始成交并交人工核对，不自动改写。"}


def status(day, db=store.DB_PATH, data_root=None):
    with store.connect(db) as conn:
        _table(conn)
        checkpoints = [dict(r) for r in conn.execute(
            "SELECT day,stage,step,status,updated_at FROM run_checkpoints WHERE day=? ORDER BY rowid", (day,))]
        locks = [dict(r) for r in conn.execute("SELECT * FROM run_locks")]
    return {"version": VERSION, "as_of": day, "stages": list(STAGES), "checkpoints": checkpoints,
            "locks": locks, "freshness": freshness(day, data_root), "pending": pending(day, db),
            "reconciliation": reconcile(day, db)}
