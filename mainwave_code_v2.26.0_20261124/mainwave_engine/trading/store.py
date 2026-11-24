"""Append-only evidence, with rebuildable current-opportunity views."""
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "results/trading_system/trading.sqlite3"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def encoded(value):
    return json.dumps(clean(value), ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(encoded(value).encode("utf-8")).hexdigest()


@contextmanager
def connect(db=DB_PATH):
    db = Path(db)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id TEXT PRIMARY KEY, as_of TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS opportunities (
            id TEXT PRIMARY KEY, symbol TEXT NOT NULL, first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL, snapshot_id TEXT NOT NULL REFERENCES snapshots(id), payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS opportunity_events (
            id TEXT PRIMARY KEY, opportunity_id TEXT NOT NULL, snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
            as_of TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS registry_versions (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY, opportunity_id TEXT NOT NULL, snapshot_id TEXT NOT NULL REFERENCES snapshots(id),
            created_at TEXT NOT NULL, payload TEXT NOT NULL);
    """)
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def latest(conn, as_of=None):
    row = conn.execute("SELECT payload FROM snapshots WHERE as_of<=? ORDER BY as_of DESC, rowid DESC LIMIT 1",
                       (as_of or "9999-12-31",)).fetchone()
    return json.loads(row[0]) if row else None


def snapshot(conn, identity):
    if not isinstance(identity, str) or not identity:
        raise ValueError("Invalid snapshot ID")
    row = conn.execute("SELECT payload FROM snapshots WHERE id=?", (identity,)).fetchone()
    if not row:
        raise ValueError("排名快照不存在")
    return json.loads(row[0])


def persist(payload, db=DB_PATH):
    identity = digest(payload)
    with connect(db) as conn:
        old = conn.execute("SELECT payload FROM snapshots WHERE id=?", (identity,)).fetchone()
        if old:
            return json.loads(old[0])
        result = {**payload, "snapshot_id": identity, "recorded_at": now()}
        conn.execute("INSERT INTO snapshots VALUES (?,?,?,?)", (identity, payload["as_of"], result["recorded_at"], encoded(result)))
        for row in payload["records"]:
            oid = row["opportunity_id"]
            prior = conn.execute("SELECT * FROM opportunities WHERE id=?", (oid,)).fetchone()
            event = {"state": row["state"], "strength_rank": row["strength_rank"],
                     "participation_rank": row["participation_rank"], "reasons": row["blockers"],
                     "previous_state": json.loads(prior["payload"])["state"] if prior and prior["last_seen"] <= payload["as_of"] else None}
            eid = digest([identity, oid])
            conn.execute("INSERT INTO opportunity_events VALUES (?,?,?,?,?)",
                         (eid, oid, identity, payload["as_of"], encoded(event)))
            if not prior or payload["as_of"] >= prior["last_seen"]:
                first = prior["first_seen"] if prior else payload["as_of"]
                conn.execute("INSERT INTO opportunities VALUES (?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                             "last_seen=excluded.last_seen,snapshot_id=excluded.snapshot_id,payload=excluded.payload",
                             (oid, row["symbol"], first, payload["as_of"], identity, encoded(row)))
        return result


def history(opportunity_id, db=DB_PATH):
    with connect(db) as conn:
        events = [dict(r) for r in conn.execute(
            "SELECT * FROM opportunity_events WHERE opportunity_id=? ORDER BY as_of,rowid", (opportunity_id,))]
        for r in events:
            r["payload"] = json.loads(r["payload"])
        reviews = [json.loads(r[0]) for r in conn.execute(
            "SELECT payload FROM assessments WHERE opportunity_id=? ORDER BY id", (opportunity_id,))]
        return {"events": events, "assessments": reviews}


def assess(identity, opportunity_id, body, db=DB_PATH):
    if not isinstance(body, dict) or set(body) != {"verdict", "note"}:
        raise ValueError("复盘需要 verdict 与 note")
    if not isinstance(body["verdict"], str) or body["verdict"] not in {"REVIEWED", "MISSED", "FALSE_POSITIVE", "WAIT", "CONFIRMED"}:
        raise ValueError("复盘状态无效")
    if not isinstance(body["note"], str) or not 1 <= len(body["note"].strip()) <= 8000:
        raise ValueError("复盘说明需为1至8000字")
    with connect(db) as conn:
        snap = snapshot(conn, identity)
        if not any(r["opportunity_id"] == opportunity_id for r in snap["records"]):
            raise ValueError("机会不属于该快照")
        result = {**body, "snapshot_id": identity, "opportunity_id": opportunity_id,
                  "created_at": now(), "author": "HUMAN", "changes_rules": False}
        conn.execute("INSERT INTO assessments(opportunity_id,snapshot_id,created_at,payload) VALUES (?,?,?,?)",
                     (opportunity_id, identity, result["created_at"], encoded(result)))
        return result
