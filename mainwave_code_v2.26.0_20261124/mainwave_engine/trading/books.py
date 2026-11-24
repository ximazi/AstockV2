"""Three books kept apart: SIGNAL (what the ranking said), AUTO (what a frozen mechanical rule trades),
MANUAL (what a person approved). A paper profit can only be attributed when the books never mix."""
import json
from pathlib import Path
from . import store, ranking

VERSION = "three_books_v1_20260921"
BOOKS = {
    "SIGNAL": "信号账：排序与发现的机械记录，回答发现能力有没有Alpha",
    "AUTO": "自动策略账：按冻结机械规则模拟执行，无人工挑选，回答完整机械策略是否赚钱",
    "MANUAL": "人工账：人在 PLAN_REVIEW 中挑选后模拟执行，回答人工判断有没有增益",
}
# The business definition of the trade set, frozen; a different rule is a different book version.
AUTO_RULE = {"version": "auto_top3_participation_v1_20260921", "top_n": 3, "rank": "participation_rank",
             "require": "paper_plan_eligible", "exclude_alpha_scope": ("R3_RESEARCH",),
             "definition": "TradeSet_t = {原参与榜名次 <= 3 且 PLAN_REVIEW 且 非R3-only}；被排除的名次不补位"}
ACCOUNT_KEYS = ("fee_bps", "sell_tax_bps", "single_risk_pct", "total_risk_pct", "symbol_cap_pct",
                "theme_cap_pct", "gross_cap_pct", "max_fill_lots")
# The auto book's account is fixed in code; once the book has started it cannot be reset.
AUTO_ACCOUNT = dict(capital=1000000, fee_bps=3, min_fee=5, sell_tax_bps=5, single_risk_pct=1, total_risk_pct=3,
                    symbol_cap_pct=20, theme_cap_pct=40, gross_cap_pct=80, max_fill_lots=3)
AUTO_DB = "results/trading_system/auto_book.sqlite3"
# The compounding book (NEW100K): same frozen trade set and entry rule as AUTO, its own ledger, 2 stock seats,
# risk base min(equity, approved ceiling), new plans paused while drawdown from the equity peak exceeds the limit.
# Parameters are the recommended protocol adopted 2026-09-23; changing any of them is a new protocol version.
NEW100K_ACCOUNT = dict(capital=100000, fee_bps=3, min_fee=5, sell_tax_bps=5, single_risk_pct=1, total_risk_pct=3,
                       symbol_cap_pct=35, theme_cap_pct=50, gross_cap_pct=90, max_fill_lots=100,
                       capital_policy="COMPOUND_CEILING", equity_ceiling=100000, stock_seats=2)
NEW100K_PROTOCOL = {"version": "new100k_protocol_v1_20260923", "account_id": "NEW100K_PAPER", "effective_at": "2026-09-24",
                    "etf_seats": 1, "etf_cap_pct": 30, "drawdown_pause_pct": 10.0,
                    "ceiling_review": "净值较上限高出 20% 且期间最大回撤未触及暂停线时，复核后上限上调至当前净值；亏损不需复核，风险基数即时随净值缩小",
                    "note": "1000 倍目标不进入任何参数；进度落后不提高风险比例；ETF 位在有独立批准的 ETF 计划前留空"}
NEW100K_DB = "results/trading_system/new100k_book.sqlite3"


def _table(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS signal_book (day TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL, "
                 "payload TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS signal_book_revisions (id TEXT PRIMARY KEY, day TEXT NOT NULL, "
                 "snapshot_id TEXT NOT NULL, recorded_at TEXT NOT NULL, payload TEXT NOT NULL)")


def _scope(record):
    return record.get("alpha_scope") or ranking.alpha_scope(record.get("route_ids", ""))


def trade_set(payload, rule=AUTO_RULE):
    """The mechanical selection from the snapshot alone; an empty set is a valid answer, not a gap.

    The rank bound applies to the original board: an excluded first place leaves two names, never three.
    """
    rows = [r for r in payload["records"] if r.get(rule["require"]) and r.get(rule["rank"]) is not None
            and r[rule["rank"]] <= rule["top_n"] and _scope(r) not in rule["exclude_alpha_scope"]]
    rows.sort(key=lambda r: (r[rule["rank"]], r["symbol"]))
    return [{"symbol": r["symbol"], "name": r.get("name"), "opportunity_id": r["opportunity_id"],
             "participation_rank": r.get("participation_rank"), "strength_rank": r.get("strength_rank"),
             "strength_score": r.get("strength_score"), "route_ids": r.get("route_ids", ""),
             "alpha_scope": _scope(r), "theme": r.get("theme")} for r in rows]


def record_signal(day, snap, db=store.DB_PATH):
    """SIGNAL book: the day's first publication is the frozen one and is what every reader gets back.

    A rerun with a different snapshot is kept as a revision beside it, never in its place, so the
    signal book, the daily report and the auto book cannot drift apart on the same day.
    """
    eligible = [r for r in snap["records"] if r.get("paper_plan_eligible")]
    entry = {"version": VERSION, "book": "SIGNAL", "day": day, "snapshot_id": snap["snapshot_id"], "rule": AUTO_RULE,
             "basis": snap.get("observation_basis"), "trade_set": trade_set(snap),
             "eligible": len(eligible),
             "excluded_r3_research": sum(1 for r in eligible if _scope(r) == "R3_RESEARCH")}
    with store.connect(db) as conn:
        _table(conn)
        row = conn.execute("SELECT payload FROM signal_book WHERE day=?", (day,)).fetchone()
        if not row:
            conn.execute("INSERT INTO signal_book VALUES (?,?,?)", (day, snap["snapshot_id"], store.encoded(entry)))
            return {**entry, "rerun": None}
        frozen = json.loads(row[0])
        same = frozen["snapshot_id"] == entry["snapshot_id"] and frozen["trade_set"] == entry["trade_set"]
        if not same:
            conn.execute("INSERT OR IGNORE INTO signal_book_revisions VALUES (?,?,?,?,?)",
                         (f"{day}:{entry['snapshot_id']}", day, entry["snapshot_id"], store.now(), store.encoded(entry)))
    return {**frozen, "rerun": {"status": "SAME_AS_FROZEN" if same else "DIFFERENT_SNAPSHOT_KEPT_AS_REVISION",
                                "snapshot_id": entry["snapshot_id"],
                                "trade_set": [r["symbol"] for r in entry["trade_set"]],
                                "note": "首发记录冻结；重跑结果另存修订，执行与验证都以首发为准"}}


def signal_history(db=store.DB_PATH):
    with store.connect(db) as conn:
        _table(conn)
        return [json.loads(r[0]) for r in conn.execute("SELECT payload FROM signal_book ORDER BY day")]


def _ledger_evidence(conn, expected=None):
    """What the ledger actually did, read back from it: effective config, orders vs fills, add refusals."""
    expected = {k: v for k, v in (expected or AUTO_ACCOUNT).items() if k in ("capital", "min_fee", *ACCOUNT_KEYS)}
    from collections import Counter
    row = conn.execute("SELECT initial_cents, config FROM paper_account").fetchone()
    cfg = json.loads(row["config"])
    effective = {"capital": row["initial_cents"] / 100, "min_fee": cfg["min_fee_cents"] / 100,
                 **{k: cfg[k] for k in ACCOUNT_KEYS}}
    matches = all(float(effective[k]) == float(expected[k]) for k in expected)
    orders = [{"symbol": json.loads(o["plan"])["symbol"], "order_id": o["id"], "ordered": o["quantity"],
               "filled": o["filled"], "status": o["status"]}
              for o in conn.execute("SELECT o.*, p.payload AS plan FROM paper_orders o JOIN paper_plans p ON p.id=o.plan_id "
                                    "WHERE o.side='BUY' ORDER BY o.created_at")]
    cancelled = {json.loads(r[0])["order_id"]: json.loads(r[0])["reason"]
                 for r in conn.execute("SELECT payload FROM paper_events WHERE kind='ORDER_CANCELLED'")}
    for o in orders:
        o["unfilled_reason"] = cancelled.get(o["order_id"], "")
    add_refusals = Counter(json.loads(r[0]).get("reason", "") for r in conn.execute(
        "SELECT payload FROM paper_events WHERE kind='ADD_REJECTED'"))
    counts = {"plans": conn.execute("SELECT COUNT(*) FROM paper_plans").fetchone()[0],
              "fills": conn.execute("SELECT COUNT(*) FROM paper_fills").fetchone()[0],
              "open_positions": conn.execute("SELECT COUNT(DISTINCT plan_id) FROM paper_lots WHERE remaining>0").fetchone()[0]}
    return {"effective_account": effective, "account_matches_code": matches,
            "config_fingerprint": store.digest([effective, AUTO_RULE]),
            "executions": orders, "add_refusals": dict(add_refusals), **counts}


def _ensure_snapshot(auto_db, main_db, snapshot_id):
    """The frozen signal's snapshot must be on the auto ledger; a crashed first run may have left it out."""
    with store.connect(auto_db) as conn:
        if conn.execute("SELECT 1 FROM snapshots WHERE id=?", (snapshot_id,)).fetchone():
            return
    with store.connect(main_db) as conn:
        stored = store.snapshot(conn, snapshot_id)
    original = {k: v for k, v in stored.items() if k not in ("snapshot_id", "recorded_at")}
    store.persist(original, auto_db)


def run_auto(day, payload, snap, root=store.ROOT, data_root=None, at=None, live=True, signal=None, main_db=None):
    """AUTO book: existing positions first (fills, exits, adds), then the frozen trade set; no human input.

    Only a same-day after-close run touches the ledger. A reconstruction reads the ledger and records what
    the rule would have chosen; it matches nothing, values nothing and places nothing.
    """
    from . import paper, daily, migrations, performance, runner
    root = Path(root)
    db = root / AUTO_DB
    migrations.upgrade(db)
    with paper.database(db) as conn:
        configured = conn.execute("SELECT 1 FROM paper_account").fetchone() is not None
    if not configured:
        paper.initialize(AUTO_ACCOUNT, db, at=at)
    # The day's first publication decides; a same-day rerun executes the frozen set, not its own.
    frozen = signal or {"snapshot_id": snap["snapshot_id"], "trade_set": trade_set(snap)}
    chosen = frozen["trade_set"]
    base = {"version": VERSION, "book": "AUTO", "day": day, "db": str(db), "rule": AUTO_RULE, "live": live,
            "account": AUTO_ACCOUNT, "snapshot_id": frozen["snapshot_id"],
            "signal_source": "FROZEN_SIGNAL_BOOK" if signal else "SNAPSHOT",
            "note": "自动账与人工账各自独立数据库；两账成绩不合并、不互相抵扣。"}
    if not live:
        with paper.database(db) as conn:
            evidence = _ledger_evidence(conn)
        return {**base, "ledger_touched": False,
                "monitor": {"status": "SKIPPED_NOT_LIVE", "issues": []},
                "placed": [{**row, "status": "NOT_LIVE_DAY_NO_ORDER", "detail": "非当日收盘后运行：不撮合、不估值、不下单"}
                           for row in chosen],
                **evidence, "performance": {"status": "NOT_EVALUATED_NOT_LIVE"},
                "reconciliation": runner.reconcile(day, db)["status"]}
    if frozen["snapshot_id"] == snap["snapshot_id"]:
        # Same payload, same snapshot id: the auto book judges exactly what the main ledger recorded.
        store.persist(payload, db)
    elif main_db is not None:
        _ensure_snapshot(db, main_db, frozen["snapshot_id"])
    # The approval clock is read after the snapshot is on record; an earlier clock reads as a stale snapshot.
    at = at or store.now()
    monitor = daily.monitor_paper(day, db, data_root)
    placed = []
    for row in chosen:
        try:
            result = paper.submit(frozen["snapshot_id"], row["symbol"], db, at=at, book="AUTO")
            placed.append({**row, "status": "ORDER_APPROVED" if result.get("created") else "NOT_CREATED",
                           "detail": result.get("reason", "") or result.get("order_id", "")})
        except (paper.Conflict, ValueError) as exc:
            placed.append({**row, "status": "REFUSED", "detail": str(exc)})
    with paper.database(db) as conn:
        evidence = _ledger_evidence(conn)
    perf = performance.report(day, db)
    return {**base, "ledger_touched": True,
            "monitor": {"status": monitor["status"], "issues": monitor.get("issues", [])},
            "placed": placed, **evidence,
            "performance": {"status": perf["status"],
                            **({k: perf["account"][k] for k in ("equity_cents", "net_return", "max_drawdown", "marked")}
                               if perf.get("account") else {})},
            "reconciliation": runner.reconcile(day, db)["status"]}


def current_drawdown(perf):
    """Drawdown of the last marked equity from its running peak, in percent; None while nothing is marked."""
    marked = [p["equity_cents"] for p in (perf or {}).get("nav") or [] if p.get("equity_cents")]
    if not marked:
        return None
    return round((1 - marked[-1] / max(marked)) * 100, 3)


def run_new100k(day, payload, snap, root=store.ROOT, data_root=None, at=None, live=True, signal=None, main_db=None):
    """NEW100K book: the frozen trade set (same as AUTO) under the compounding protocol. Only same-day live runs
    place orders; before the effective date, on a reconstruction, or past the drawdown pause, nothing is placed."""
    from . import paper, daily, migrations, performance, runner
    root = Path(root)
    db = root / NEW100K_DB
    frozen = signal or {"snapshot_id": snap["snapshot_id"], "trade_set": trade_set(snap)}
    chosen = frozen["trade_set"]
    base = {"version": VERSION, "book": "NEW100K", "day": day, "db": str(db), "protocol": NEW100K_PROTOCOL,
            "account": NEW100K_ACCOUNT, "snapshot_id": frozen["snapshot_id"], "live": live,
            "note": "复利账：与旧 100 万自动账、人工账各自独立；成绩不合并"}
    if day < NEW100K_PROTOCOL["effective_at"]:
        return {**base, "status": "NOT_YET_EFFECTIVE", "ledger_touched": False,
                "placed": [{**row, "status": "NOT_YET_EFFECTIVE"} for row in chosen]}
    migrations.upgrade(db)
    with paper.database(db) as conn:
        configured = conn.execute("SELECT 1 FROM paper_account").fetchone() is not None
    if not configured:
        paper.initialize(NEW100K_ACCOUNT, db, at=at)
    if not live:
        with paper.database(db) as conn:
            evidence = _ledger_evidence(conn, NEW100K_ACCOUNT)
        return {**base, "status": "NOT_LIVE", "ledger_touched": False,
                "placed": [{**row, "status": "NOT_LIVE_DAY_NO_ORDER"} for row in chosen], **evidence}
    if frozen["snapshot_id"] == snap["snapshot_id"]:
        store.persist(payload, db)
    elif main_db is not None:
        _ensure_snapshot(db, main_db, frozen["snapshot_id"])
    at = at or store.now()
    monitor = daily.monitor_paper(day, db, data_root)
    drawdown = current_drawdown(performance.report(day, db))
    paused = drawdown is not None and drawdown >= NEW100K_PROTOCOL["drawdown_pause_pct"]
    placed = []
    for row in chosen:
        if paused:
            placed.append({**row, "status": "DRAWDOWN_PAUSE",
                           "detail": f"净值回撤 {drawdown}% 达到 {NEW100K_PROTOCOL['drawdown_pause_pct']}%：暂停新开仓，待复核"})
            continue
        try:
            result = paper.submit(frozen["snapshot_id"], row["symbol"], db, at=at, book="NEW100K")
            placed.append({**row, "status": "ORDER_APPROVED" if result.get("created") else "NOT_CREATED",
                           "detail": result.get("reason", "") or result.get("order_id", "")})
        except (paper.Conflict, ValueError) as exc:
            placed.append({**row, "status": "REFUSED", "detail": str(exc)})
    with paper.database(db) as conn:
        evidence = _ledger_evidence(conn, NEW100K_ACCOUNT)
    perf = performance.report(day, db)
    return {**base, "status": "LIVE", "ledger_touched": True, "drawdown_pct": current_drawdown(perf), "paused": paused,
            "monitor": {"status": monitor["status"], "issues": monitor.get("issues", [])}, "placed": placed, **evidence,
            "performance": {"status": perf["status"],
                            **({k: perf["account"][k] for k in ("equity_cents", "net_return", "max_drawdown", "marked")}
                               if perf.get("account") else {})},
            "reconciliation": runner.reconcile(day, db)["status"]}
