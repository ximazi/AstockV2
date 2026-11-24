"""Schema versions, reversible migrations, verifiable backups and hash-kept archives."""
import gzip
import hashlib
import json
import sqlite3
from pathlib import Path
from . import store

VERSION = "schema_ops_v1_20260920"
# Tables added by batches A-D; the baseline tables are created by store.connect itself.
BATCH_TABLES = {
    "opportunity_continuity": "CREATE TABLE IF NOT EXISTS opportunity_continuity (opportunity_id TEXT NOT NULL, as_of TEXT NOT NULL, snapshot_id TEXT NOT NULL, event TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY (opportunity_id, as_of))",
    "rule_versions": "CREATE TABLE IF NOT EXISTS rule_versions (id TEXT PRIMARY KEY, kind TEXT NOT NULL, version TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL)",
    "run_locks": "CREATE TABLE IF NOT EXISTS run_locks (name TEXT PRIMARY KEY, owner TEXT NOT NULL, acquired_at TEXT NOT NULL, heartbeat TEXT NOT NULL)",
    "run_checkpoints": "CREATE TABLE IF NOT EXISTS run_checkpoints (id TEXT PRIMARY KEY, day TEXT NOT NULL, stage TEXT NOT NULL, step TEXT NOT NULL, status TEXT NOT NULL, updated_at TEXT NOT NULL, payload TEXT NOT NULL)",
    "ai_assessments": "CREATE TABLE IF NOT EXISTS ai_assessments (id TEXT PRIMARY KEY, opportunity_id TEXT NOT NULL, as_of TEXT NOT NULL, evidence_hash TEXT NOT NULL, model TEXT NOT NULL, prompt_version TEXT NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL)",
    "experiments": "CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL)",
    "corporate_actions": "CREATE TABLE IF NOT EXISTS corporate_actions (id TEXT PRIMARY KEY, plan_id TEXT NOT NULL, day TEXT NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL)",
    "archive_index": "CREATE TABLE IF NOT EXISTS archive_index (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT NOT NULL, first_day TEXT NOT NULL, last_day TEXT NOT NULL, rows INTEGER NOT NULL, pruned INTEGER NOT NULL)",
}
# Each step is reversible; `down` first exports the tables it drops so no record is ever lost.
MIGRATIONS = [
    {"version": 1, "name": "baseline", "creates": ["archive_index"], "note": "store.connect 建表；归档索引随基线，永不回退删除"},
    {"version": 2, "name": "batch_a_c", "creates": ["opportunity_continuity", "rule_versions", "run_locks",
                                                    "run_checkpoints", "ai_assessments"]},
    {"version": 3, "name": "batch_d", "creates": ["experiments", "corporate_actions"]},
]
CURRENT = MIGRATIONS[-1]["version"]
COUNTED = ("snapshots", "opportunities", "opportunity_events", "assessments", "registry_versions",
           "paper_plans", "paper_orders", "paper_fills", "paper_lots", "paper_events", "paper_valuations",
           "opportunity_continuity", "rule_versions", "ai_assessments", "experiments", "corporate_actions")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def version(conn):
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def _set_version(conn, value):
    conn.execute(f"PRAGMA user_version={int(value)}")


def _dump_tables(conn, tables, dest):
    """Export whole tables to gzip'd JSON lines with a per-row hash; the archive keeps the failed samples too."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with gzip.open(dest, "wt", encoding="utf-8") as handle:
        for table in tables:
            if not conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                continue
            for row in conn.execute(f"SELECT * FROM {table}"):
                record = {"table": table, "row": dict(row)}
                record["row_sha256"] = store.digest(record["row"])
                handle.write(store.encoded(record) + "\n")
                rows += 1
    return rows, _sha(dest)


def upgrade(db=store.DB_PATH, target=CURRENT):
    """Apply forward steps in order; already-applied steps are skipped by user_version."""
    applied = []
    with store.connect(db) as conn:
        current = version(conn)
        for step in MIGRATIONS:
            if step["version"] <= current or step["version"] > target:
                continue
            for table in step["creates"]:
                conn.execute(BATCH_TABLES[table])
            _set_version(conn, step["version"])
            applied.append(step["version"])
        final = version(conn)
    return {"version": VERSION, "from": current, "to": final, "applied": applied}


def downgrade(db=store.DB_PATH, target=CURRENT - 1, archive_dir=None):
    """Reverse steps above `target`, exporting every dropped table first so nothing is silently cleared."""
    archive_dir = Path(archive_dir or Path(db).parent / "archive")
    exported, reversed_steps = [], []
    with store.connect(db) as conn:
        current = version(conn)
        for step in sorted(MIGRATIONS, key=lambda s: -s["version"]):
            if step["version"] > current or step["version"] <= target or step["version"] == 1:
                continue
            if step["creates"]:
                dest = archive_dir / f"downgrade_v{step['version']}_{store.now()[:10]}.jsonl.gz"
                rows, sha = _dump_tables(conn, step["creates"], dest)
                _index(conn, dest, sha, "", "", rows)
                exported.append({"path": str(dest), "rows": rows, "sha256": sha, "tables": step["creates"]})
                for table in step["creates"]:
                    conn.execute(f"DROP TABLE IF EXISTS {table}")
            _set_version(conn, step["version"] - 1)
            reversed_steps.append(step["version"])
        final = version(conn)
    return {"version": VERSION, "from": current, "to": final, "reversed": reversed_steps,
            "exported": exported, "note": "回退前已导出被删表并登记哈希；失败样本不因回退消失。"}


def _index(conn, path, sha, first_day, last_day, rows):
    conn.execute("CREATE TABLE IF NOT EXISTS archive_index (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT NOT NULL, first_day TEXT NOT NULL, last_day TEXT NOT NULL, rows INTEGER NOT NULL, pruned INTEGER NOT NULL)")
    conn.execute("INSERT OR IGNORE INTO archive_index VALUES (?,?,?,?,?,?,?,0)",
                 (sha, store.now(), str(path), sha, first_day, last_day, rows))


def counts(conn):
    out = {}
    for table in COUNTED:
        try:
            out[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except sqlite3.OperationalError:
            out[table] = None
    return out


def backup(db=store.DB_PATH, dest=None):
    """Online copy through the sqlite backup API plus a manifest the restore check can verify against."""
    db = Path(db)
    dest = Path(dest or db.parent / "backups" / f"trading_{store.now().replace(':', '').replace('-', '')[:15]}.sqlite3")
    dest.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(db)
    target = sqlite3.connect(dest)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    with store.connect(db) as conn:
        manifest = {"version": VERSION, "created_at": store.now(), "source": str(db), "backup": str(dest),
                    "user_version": version(conn), "counts": counts(conn)}
    manifest["sha256"] = _sha(dest)
    manifest_path = dest.with_suffix(".manifest.json")
    manifest_path.write_text(store.encoded(manifest), encoding="utf-8")
    return {**manifest, "manifest": str(manifest_path)}


def verify_restore(backup_path, manifest_path=None):
    """Open the backup on its own and prove cash, lots, pending work and reviews are what the manifest says."""
    from . import runner
    backup_path = Path(backup_path)
    manifest_path = Path(manifest_path or backup_path.with_suffix(".manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    problems = []
    if _sha(backup_path) != manifest["sha256"]:
        problems.append("备份文件哈希与清单不符")
    with store.connect(backup_path) as conn:
        if version(conn) != manifest["user_version"]:
            problems.append("schema 版本与清单不符")
        for table, expected in manifest["counts"].items():
            got = counts(conn).get(table)
            if expected is not None and got != expected:
                problems.append(f"{table} 行数 {got} ≠ 清单 {expected}")
    reconciliation = runner.reconcile("", backup_path)
    if reconciliation.get("status") == "MISMATCH":
        problems.append("还原后现金与成交不一致")
    pending = runner.pending("", backup_path)
    return {"version": VERSION, "backup": str(backup_path), "status": "CONSISTENT" if not problems else "MISMATCH",
            "problems": problems, "reconciliation": reconciliation, "pending_tasks": len(pending["tasks"]),
            "reviews": manifest["counts"].get("assessments"),
            "note": "还原校验只读取备份文件本身，不改动生产库。"}


def archive(db=store.DB_PATH, before_day="", dest_dir=None):
    """Compress old snapshots and their events with hashes; the live rows stay, failed samples included."""
    dest_dir = Path(dest_dir or Path(db).parent / "archive")
    with store.connect(db) as conn:
        snaps = conn.execute("SELECT id, as_of FROM snapshots WHERE as_of<? ORDER BY as_of", (before_day,)).fetchall()
        if not snaps:
            return {"version": VERSION, "status": "NOTHING_TO_ARCHIVE", "before_day": before_day}
        ids = [r["id"] for r in snaps]
        dest = dest_dir / f"snapshots_{snaps[0]['as_of']}_{snaps[-1]['as_of']}.jsonl.gz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        rows = 0
        with gzip.open(dest, "wt", encoding="utf-8") as handle:
            marks = ",".join("?" * len(ids))
            for table, key in (("snapshots", "id"), ("opportunity_events", "snapshot_id"), ("assessments", "snapshot_id")):
                for row in conn.execute(f"SELECT * FROM {table} WHERE {key} IN ({marks})", ids):
                    record = {"table": table, "row": dict(row)}
                    record["row_sha256"] = store.digest(record["row"])
                    handle.write(store.encoded(record) + "\n")
                    rows += 1
        sha = _sha(dest)
        _index(conn, dest, sha, snaps[0]["as_of"], snaps[-1]["as_of"], rows)
    return {"version": VERSION, "status": "ARCHIVED", "path": str(dest), "sha256": sha, "rows": rows,
            "first_day": snaps[0]["as_of"], "last_day": snaps[-1]["as_of"], "pruned": False,
            "note": "归档是带哈希的副本；原记录未删除，失败样本仍可查。"}


def status(db=store.DB_PATH):
    with store.connect(db) as conn:
        current = version(conn)
        index = [dict(r) for r in conn.execute("SELECT * FROM archive_index ORDER BY created_at")] \
            if conn.execute("SELECT name FROM sqlite_master WHERE name='archive_index'").fetchone() else []
        table_counts = counts(conn)
    backups = sorted(Path(db).parent.glob("backups/*.manifest.json"))
    return {"version": VERSION, "schema_version": current, "latest_schema": CURRENT,
            "pending_migrations": [m["version"] for m in MIGRATIONS if m["version"] > current],
            "counts": table_counts, "archives": index,
            "backups": [json.loads(p.read_text(encoding="utf-8-sig")) for p in backups[-5:]]}
