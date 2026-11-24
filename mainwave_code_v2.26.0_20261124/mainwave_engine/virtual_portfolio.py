"""Persistent paper positions with entry-anchored, advice-only daily risk review."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sqlite3
from uuid import uuid4

import pandas as pd

from . import exit_policy as XP, route_evidence as RE, stock_analyzer as SA
from . import stock_lifecycle as SL
from .pipeline import _write
from .theme_lifecycle import market_calendar

VERSION = "virtual_portfolio_v1"
DB_PATH = Path(__file__).resolve().parents[1] / "results/mainwave_engine/virtual_portfolio/positions.sqlite3"
NOTICE = "手工虚拟持仓，按指定行情日原始收盘价记账；不是实盘成交。未设股数，不计算账户金额；复权价差不含费用，不等同现金收益。风险建议不自动减仓或卖出。"
LABELS = {"HOLD": "继续持有观察", "SOFT_CONFIRMING": "收盘保护确认中", "REDUCE": "建议分批减仓",
          "SELL": "建议卖出剩余", "DATA_WAIT": "数据待核验", "NEW": "已加入，待日常评估"}


STRATEGIES = {"S4_LAD": "S4 回踩再启动 · 分批止盈（S4离场研究_20260925.md）"}
LAD_VERSION = "s4_lad_v3_20260929"  # v2 graded denial exit (B2h) + S4 event identity; v3 sells the rest on the CSI 1000 trend exit (G1)
# Rules each stored policy version replays with; a position is always reviewed under the version it was entered with.
LAD_POLICIES = {"s4_lad_v1_20260925": {"b2h": False, "g1": False},
                "s4_lad_v2_20260926": {"b2h": True, "g1": False},
                "s4_lad_v3_20260929": {"b2h": True, "g1": True}}
# Migration approved by the user on 2026-09-29 ("S4 分批止盈补上 G1 红灯时卖出剩余", applied to the open S4 positions):
# older versions take the G1 exit only on G1 sessions from this date on; earlier G1 sessions are not replayed into them.
G1_MIGRATION = "2026-09-29"


class Conflict(ValueError):
    pass


def connect(db):
    db = Path(db)
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS positions (
            id TEXT PRIMARY KEY, symbol TEXT NOT NULL, entry_date TEXT NOT NULL,
            status TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE UNIQUE INDEX IF NOT EXISTS one_open_symbol ON positions(symbol) WHERE status='OPEN';
        CREATE TABLE IF NOT EXISTS observations (
            position_id TEXT NOT NULL, as_of TEXT NOT NULL, payload TEXT NOT NULL,
            PRIMARY KEY(position_id, as_of));
    """)
    return conn


def checked_date(value):
    day = date.fromisoformat(value)
    if day.isoformat() != value or day > datetime.now(timezone(timedelta(hours=8))).date():
        raise ValueError("需要已经结束的行情日期，格式 YYYY-MM-DD")
    now = datetime.now(timezone(timedelta(hours=8)))
    if day == now.date() and now.hour < 15:
        raise ValueError("今日尚未收盘，请使用已收盘交易日")
    return value


def load_frame(symbol, as_of):
    raw = pd.read_csv(Path("data") / f"{symbol}_daily.csv", encoding="utf-8-sig")
    raw["date"] = pd.to_datetime(raw.date)
    raw = raw[raw.date.le(as_of)].sort_values("date").drop_duplicates("date", keep="last")
    return RE.build(raw, symbol)


def close_quote(symbol, as_of):
    checked_date(as_of)
    f = load_frame(symbol, as_of)
    if f.empty or f.date.iloc[-1].strftime("%Y-%m-%d") != as_of or as_of not in market_calendar(as_of):
        raise ValueError("该日没有有效收盘行情或交易日历，不能用旧价格记账")
    r = f.iloc[-1]
    if not all(math.isfinite(float(r[k])) and float(r[k]) > 0 for k in ("close", "atr20", "index_close")):
        raise ValueError("收盘价或初始波动数据无效")
    return f, r


def add(symbol, as_of, db=DB_PATH, strategy=None, meta=None):
    if strategy is not None and strategy not in STRATEGIES:
        raise ValueError(f"未知策略：{strategy}")
    symbol = SA.normalize_symbol(symbol)
    # Fast retry path also preserves the original price/date after later daily updates.
    with closing(connect(db)) as conn:
        old = conn.execute("SELECT payload FROM positions WHERE symbol=? AND status='OPEN'", (symbol,)).fetchone()
        if old:
            return {"position": json.loads(old[0]), "created": False}
    _, quote = close_quote(symbol, as_of)
    board = SA._read_json(SA.OUT_ROOT / as_of.replace("-", "") / "analysis_universe.json", {})
    context = next((r for r in board.get("records", []) if r["symbol"] == symbol), {})
    identity = SA._mapping(symbol)
    row = {"id": uuid4().hex, "symbol": symbol, "name": context.get("name") or identity.get("name") or symbol,
           "theme": context.get("belonging_theme", ""), "route_ids": context.get("route_ids", ""),
           "entry_date": as_of, "entry_price": float(quote.close),
           "atr_fraction": float(quote.atr20 / quote.index_close), "status": "OPEN",
           "source": "MANUAL_CLOSE", "created_at": datetime.now(timezone.utc).isoformat(),
           "version": VERSION, "exit_policy_version": LAD_VERSION if strategy == "S4_LAD" else XP.VERSION}
    if strategy:
        row["strategy"] = strategy
    row.update({k: v for k, v in (meta or {}).items() if k.startswith("s4_")})
    if 2 * row["atr_fraction"] >= 1:
        raise ValueError("初始风险底线无效，不能建立虚拟持仓")
    with closing(connect(db)) as conn, conn:
        conn.execute("INSERT OR IGNORE INTO positions VALUES (?, ?, ?, ?, ?)",
                     (row["id"], symbol, as_of, "OPEN", json.dumps(row, ensure_ascii=False)))
        stored = json.loads(conn.execute("SELECT payload FROM positions WHERE symbol=? AND status='OPEN'", (symbol,)).fetchone()[0])
    return {"position": stored, "created": stored["id"] == row["id"]}


def positions(db=DB_PATH):
    if not Path(db).exists():
        return []
    with closing(connect(db)) as conn:
        return [json.loads(r[0]) for r in conn.execute("SELECT payload FROM positions ORDER BY entry_date DESC, symbol, id")]


def load_g1(as_of):
    """Sessions with the CSI 1000 trend exit (G1) on: close < MA20 and MA20 below its level five sessions earlier."""
    path = Path(__file__).resolve().parents[1] / "data/000852_SH_index.csv"
    if not path.exists():
        return frozenset()
    m = XP.market_frame(pd.read_csv(path), as_of)
    return frozenset(m.index[m.market_weak.to_numpy()].strftime("%Y-%m-%d"))


def evaluate(position, frame, as_of, calendar, market, g1=frozenset()):
    """Rebuild advice from the actual paper close entry, never a signal's hypothetical open."""
    base = {"as_of": as_of, "version": VERSION, "policy_version": XP.VERSION,
            "status": "DATA_WAIT", "reason": "行情缺失或不连续，不能判为持有安全", "events": []}
    f = frame[frame.date.le(as_of)].copy().reset_index(drop=True)
    if f.empty:
        return base
    days = f.date.dt.strftime("%Y-%m-%d").tolist()
    base.update(price_date=days[-1], close=float(f.close.iloc[-1]))
    entry_day = position["entry_date"]
    if entry_day not in days:
        return base
    entry = days.index(entry_day)
    if not math.isclose(float(f.close.iloc[entry]), position["entry_price"], rel_tol=1e-6):
        return {**base, "reason": "入场日原始行情已修订，请核对原始成本；不自动覆盖台账"}
    present = set(days) | set(frame.attrs.get("suspended_days", []))
    needed = [d for d in calendar if entry_day <= d <= as_of]
    if not needed or needed[-1] != as_of or any(d not in present for d in needed):
        return base
    relevant = f.iloc[entry:][["index_open", "index_high", "index_low", "index_close", "close"]]
    if relevant.isna().any().any() or not relevant.map(lambda x: math.isfinite(x) and x > 0).all().all():
        return base
    if position.get("strategy") == "S4_LAD":
        return lad_review(f, entry, days, as_of, base, position, g1)
    f = XP.enrich(f, market)
    a = {c: f[c].to_numpy() for c in f.columns}
    price = float(f.index_close.iloc[entry])
    atr = price * position["atr_fraction"]
    hard = soft = price - 2 * atr
    peak, pending, reduction, breaches, required = price, "", False, 0, 1
    events, seen = [], set()
    initial_warning, _ = XP.warning(a, entry, entry, price, atr, price)
    if initial_warning == "EXIT_STRUCTURE":
        pending = initial_warning
        events.append({"date": entry_day, "reason": pending, "label": XP.REASONS[pending]})
        seen.add(pending)
    # Close entry: the entry day's earlier low/high cannot trigger a stop or profit target.
    for j in range(entry + 1, len(f)):
        reasons = []
        if a["index_low"][j] <= hard:
            pending = "INITIAL_STOP"
        review = XP.close_review(a, j, entry, price, atr, peak, soft_stop=soft, breaches=breaches,
                                 required=required, pending_exit=pending, pending_reduce=reduction)
        soft, breaches, required = (review[k] for k in ("soft_stop", "breaches", "required"))
        pending, reduction = review["pending_exit"], review["pending_reduce"]
        if j - entry + 1 >= 60:
            pending = pending or "HORIZON"
        if pending:
            reasons.append(pending)
        elif reduction:
            reasons.append("RISK_REDUCTION")
        for multiple in (3, 6):
            if a["index_high"][j] >= price + multiple * atr:
                reasons.append(f"TARGET_{multiple}ATR")
                reduction = True
        for reason in reasons:
            if reason not in seen:
                events.append({"date": days[j], "reason": reason,
                               "label": XP.REASONS.get(reason, f"达到{reason.removeprefix('TARGET_')}分批保护位")})
                seen.add(reason)
        peak = max(peak, float(a["index_high"][j]))
    current = f.iloc[-1]
    scale = float(current.close / current.index_close)
    suspended = days[-1] != as_of
    status = "SELL" if pending else "REDUCE" if reduction else "SOFT_CONFIRMING" if breaches else "HOLD"
    reason = (XP.REASONS.get(pending) if pending else "分批保护信号已出现，尚未登记处置" if reduction
              else f"连续收盘失守 {breaches}/{required} 次" if breaches else "尚未触发持仓退出规则")
    return {**base, "status": "DATA_WAIT" if suspended else status,
            "reason": "已知停牌，等待可交易行情；此前风险仍需处置" if suspended else reason,
            "pending_action": status if suspended else "", "events": events,
            "return_pct": float((current.index_close / price - 1) * 100),
            "raw_return_pct": float((current.close / position["entry_price"] - 1) * 100),
            "hard_floor": hard * scale, "soft_floor": soft * scale, "soft_breaches": breaches,
            "soft_required": required, "held_sessions": len(f) - entry,
            "strong": bool(current.exit_strong), "market_known": bool(current.market_known),
            "pending_reason": pending, "alert_since": events[0]["date"] if events else ""}


def lad_review(f, entry, days, as_of, base, position=None, g1=frozenset()):
    """S4 staged profit-taking (LAD), advice only; mirrors research/exit_study_s4.py on the paper close entry:
    1/3 at +10% and 1/3 at +20% (intraday touch); the rest leaves at the next open after two closes < EMA20 x 0.97,
    or, once a close has been >= +10%, a heavy bearish candle or long upper shadow on >= 1.5x volume; -10% intraday
    stop; 60 sessions. Also reports the return a strict LAD execution would have made (adjusted prices, no fees).
    v2 graded denial (research/accept_failure.py B2h): within 3 sessions a close below the signal day's open sells
    half; a close below the pullback low (lowest low of the 10 sessions before the signal), or two closes below EMA20
    within 5 sessions, sells the rest. The signal day is the entry day (the paper entry is that day's close).
    v3: a session in g1 (the CSI 1000 trend exit, S4研究总结_20260925.md §4) sells the rest at the next open.
    The position's stored exit_policy_version selects the rules (LAD_POLICIES); older versions take G1 only from
    G1_MIGRATION. Stored s4_pullback_low / s4_signal_open (raw prices frozen at the pick) are used when present,
    converted to the adjusted series by the entry day's ratio; otherwise they are rebuilt from the frame.
    Every partial sale is capped at what is still held (research/accept_failure.py sell(): q = min(q, pos))."""
    position = position or {}
    version = position.get("exit_policy_version") or LAD_VERSION
    if version not in LAD_POLICIES:
        return {**base, "status": "DATA_WAIT", "reason": f"未知的分批止盈版本 {version}，不按现行规则替代回放"}
    policy = LAD_POLICIES[version]
    g1_from = "" if policy["g1"] else G1_MIGRATION
    o, h, l, c = (f[k].to_numpy(float) for k in ("index_open", "index_high", "index_low", "index_close"))
    ema20 = f.index_close.ewm(span=20, adjust=False).mean().to_numpy()
    vol = (f["volume"] if "volume" in f else f["amount"]).astype(float)
    vr = vol / vol.shift(1).rolling(20).mean()
    e = c[entry]
    held, realized, events = 1.0, 0.0, []
    pending, exited, below, armed, sold = "", None, 0, False, 0
    to_adj = float(f.index_close.iloc[entry] / f.close.iloc[entry])  # raw price -> adjusted series on the entry day
    stored = position.get("s4_pullback_low") is not None and position.get("s4_signal_open") is not None
    pb_low = float(position["s4_pullback_low"]) * to_adj if stored else l[max(0, entry - 10):entry].min()
    sig_open = float(position["s4_signal_open"]) * to_adj if stored else o[entry]
    half, warned, under = False, False, 0
    for j in range(entry + 1, len(f)):
        if pending:
            realized += held * o[j] / e
            held, exited = 0.0, (days[j], pending)
            break
        if half:
            realized += held / 2 * o[j] / e
            held, half = held / 2, False
        if l[j] <= e * 0.90:
            realized += held * min(o[j], e * 0.90) / e
            held, exited = 0.0, (days[j], "止损 −10%")
            events.append({"date": days[j], "reason": "LAD_STOP", "label": "盘中触及 −10% 止损，卖出剩余"})
            break
        for n, level in ((1, 1.10), (2, 1.20)):
            if sold < n and h[j] >= e * level:
                sold = n
                q = min(1 / 3, held)
                realized += q * max(o[j], e * level) / e
                held -= q
                events.append({"date": days[j], "reason": f"LAD_TP{n}",
                               "label": f"达到 +{round((level - 1) * 100)}%，按规则卖出 1/3" + ("（即剩余全部）" if held <= 1e-9 else "")})
        if held <= 1e-9:
            held, exited = 0.0, (days[j], "分批止盈后已无剩余")
            break
        armed = armed or c[j] >= e * 1.10
        below = below + 1 if c[j] < ema20[j] * 0.97 else 0
        under = under + 1 if j - entry <= 5 and c[j] < ema20[j] else 0
        if policy["b2h"] and not warned and j - entry <= 3 and c[j] < sig_open:
            warned = half = True
            events.append({"date": days[j], "reason": "LAD_WARN", "label": "恢复阳线被吞没（收盘低于信号日开盘），下一交易日开盘卖一半"})
        body = (o[j] - c[j]) / c[j - 1]
        rng = h[j] - l[j]
        heavy = (c[j] / c[j - 1] - 1 <= -0.05 or body >= 0.04) and vr.iloc[j] >= 1.5
        shadow = rng > 0 and (h[j] - max(o[j], c[j])) / rng >= 0.5 and rng / c[j - 1] >= 0.05 and vr.iloc[j] >= 1.5
        if days[j] in g1 and days[j] >= g1_from:
            pending = "大盘趋势清仓（G1：中证1000 收盘跌破 MA20 且 MA20 下拐）"
        elif policy["b2h"] and c[j] < pb_low:
            pending = "收盘跌破回踩低点（回踩事件被否定）"
        elif policy["b2h"] and under >= 2:
            pending = "买入后 5 日内两次收于 EMA20 下（恢复被否定）"
        elif below >= 2:
            pending = "两日收盘跌破 EMA20×0.97"
        elif armed and (heavy or shadow):
            pending = "放量大阴线" if heavy else "放量长上影"
        elif j - entry >= 60:
            pending = "持有满 60 个交易日"
        if pending:
            events.append({"date": days[j], "reason": "LAD_EXIT", "label": f"{pending}，下一交易日开盘卖出剩余"})
    scale = float(f.close.iloc[-1] / f.index_close.iloc[-1])
    mark = realized + held * c[-1] / e
    latest = [x for x in events if x["date"] == days[-1]]
    status = ("SELL" if exited or pending else
              "REDUCE" if any(x["reason"].startswith(("LAD_TP", "LAD_WARN")) for x in latest) else "HOLD")
    done = f"已按规则卖出 {1 - held:.0%}" if held < 1 else "尚未减仓"
    reason = (f"{exited[0]} 按规则卖出剩余（{exited[1]}）" if exited else f"{pending}：下一交易日开盘卖出剩余" if pending
              else "；".join(x["label"] for x in latest) if status == "REDUCE" else f"持有；{done}")
    suspended = days[-1] != as_of
    return {**base, "status": "DATA_WAIT" if suspended else status, "reason": reason, "events": events,
            "strategy": "S4_LAD", "policy_version": version,
            "g1_rule": "入场版本自带" if policy["g1"] else f"迁移：{G1_MIGRATION} 起适用（用户 2026-09-29 批准）",
            "anchor_source": "记入时冻结" if stored else "按行情重建",
            "return_pct": float((c[-1] / e - 1) * 100), "raw_return_pct": float((c[-1] / e - 1) * 100),
            "lad_return_pct": float((mark - 1) * 100), "lad_remaining": round(held, 4),
            "hard_floor": e * 0.90 * scale, "soft_floor": ema20[-1] * 0.97 * scale,
            "tp1": e * 1.10 * scale, "tp2": e * 1.20 * scale,
            "pullback_low": pb_low * scale, "signal_open": sig_open * scale,
            "held_sessions": len(f) - entry, "pending_reason": pending, "alert_since": events[0]["date"] if events else "",
            "pending_action": status if suspended else ""}


def scan_position(position, as_of):
    frame = load_frame(position["symbol"], as_of)
    result = evaluate(position, frame, as_of, market_calendar(as_of), XP.load_market(as_of), load_g1(as_of))
    if result["status"] != "DATA_WAIT":
        try:
            cycle, _ = SL.analyze(position["symbol"], as_of, trajectory=pd.DataFrame())
            result["lifecycle"] = cycle.get("cycle_stage_label", "未形成")
            result["cycle_id"] = cycle.get("cycle_id", "")
        except (OSError, ValueError, KeyError) as exc:
            result["lifecycle"] = f"生命周期待核验：{exc}"
    return result


def run(as_of, db=DB_PATH, output_root=None):
    checked_date(as_of)
    records = []
    for position in positions(db):
        if position["entry_date"] > as_of or (position["status"] == "CLOSED" and position["exit_date"] <= as_of):
            continue
        try:
            observation = scan_position(position, as_of)
        except (OSError, ValueError, KeyError) as exc:
            observation = {"as_of": as_of, "version": VERSION, "status": "DATA_WAIT", "reason": str(exc), "events": []}
        with closing(connect(db)) as conn, conn:
            conn.execute("INSERT OR REPLACE INTO observations VALUES (?, ?, ?)",
                         (position["id"], as_of, json.dumps(observation, ensure_ascii=False, allow_nan=False)))
        records.append({**position, "observation": observation})
        print(f"[virtual_portfolio] {position['symbol']} {LABELS[observation['status']]}: {observation['reason']}", flush=True)
    counts = {key: sum(r["observation"]["status"] == key for r in records) for key in LABELS}
    payload = {"as_of": as_of, "version": VERSION, "notice": NOTICE, "records": records,
               "summary": {"total": len(records), **counts, "status": "PARTIAL" if counts["DATA_WAIT"] else "SUCCESS"}}
    output = (SA.OUT_ROOT if output_root is None else Path(output_root)) / as_of.replace("-", "")
    _write(output / "virtual_positions.json", payload)
    lines = [f"# 虚拟持仓监控 {as_of}", "", NOTICE, "", f"持仓 {len(records)} | 卖出 {counts['SELL']} | 减仓 {counts['REDUCE']} | 待核验 {counts['DATA_WAIT']}", ""]
    for r in records:
        o = r["observation"]
        lines += [f"## {r['name']} {r['symbol']}", f"- 加入：{r['entry_date']} 收盘 {r['entry_price']:.3f}",
                  f"- 评估：{LABELS[o['status']]}；{o['reason']}",
                  f"- 生命周期：{o.get('lifecycle', '待核验')}；行情日期：{o.get('price_date', '未知')}",
                  f"- 复权价差：{o.get('return_pct', '--')}%；硬底线：{o.get('hard_floor', '--')}；收盘保护：{o.get('soft_floor', '--')}"]
        if o.get("strategy") == "S4_LAD":
            lines.append(f"- S4 分批止盈：剩余 {o['lad_remaining']:.0%}；止盈位 {o['tp1']:.2f} / {o['tp2']:.2f}；"
                         f"按规则执行的模拟收益 {o['lad_return_pct']:.1f}%")
        lines.append("")
    (output / "virtual_positions.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[virtual_portfolio] {as_of} total={len(records)} sell={counts['SELL']} reduce={counts['REDUCE']} data_wait={counts['DATA_WAIT']} -> {output / 'virtual_positions.md'}", flush=True)
    return payload


def listing(db=DB_PATH):
    rows = positions(db)
    if rows:
        with closing(connect(db)) as conn:
            for row in rows:
                found = conn.execute("SELECT payload FROM observations WHERE position_id=? ORDER BY as_of DESC LIMIT 1", (row["id"],)).fetchone()
                row["observation"] = json.loads(found[0]) if found else {
                    "status": "NEW", "reason": "等待日常持仓扫描", "events": [],
                    "price_date": row["entry_date"], "close": row["entry_price"], "return_pct": 0.,
                    "hard_floor": row["entry_price"] * (1 - 2 * row["atr_fraction"])}
                if row["observation"]["status"] == "DATA_WAIT":
                    for prior in conn.execute("SELECT payload FROM observations WHERE position_id=? ORDER BY as_of DESC", (row["id"],)):
                        alert = json.loads(prior[0])
                        if alert["status"] in {"SELL", "REDUCE"}:
                            row["last_actionable_observation"] = alert
                            break
    try:
        report_date = SA.latest_report_date()
    except FileNotFoundError:
        report_date = ""
    return {"positions": rows, "notice": NOTICE, "labels": LABELS, "version": VERSION, "report_date": report_date}


def close_position(position_id, as_of, db=DB_PATH):
    with closing(connect(db)) as conn:
        found = conn.execute("SELECT payload FROM positions WHERE id=?", (position_id,)).fetchone()
    if not found:
        raise ValueError("虚拟持仓不存在")
    position = json.loads(found[0])
    if position["status"] == "CLOSED":
        return position
    if as_of <= position["entry_date"]:
        raise ValueError("按A股T+1约束，加入当日不能模拟卖出")
    f, quote = close_quote(position["symbol"], as_of)
    from . import data as D
    limit = D.limit_pct(position["symbol"])
    if limit == 10 and quote.get("isST") == 1:
        limit = 5
    lower = math.floor(float(quote.preclose) * (1 - limit / 100) * 100 + .5) / 100
    if float(quote.high) <= lower + .001:
        raise ValueError("当日一字跌停，不能假设卖出成交")
    with closing(connect(db)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        latest = conn.execute("SELECT MAX(as_of) FROM observations WHERE position_id=?", (position_id,)).fetchone()[0]
        if latest and as_of < latest:
            raise Conflict("卖出日期早于已保存的持仓评估，请使用最新行情日")
        old = json.loads(conn.execute("SELECT payload FROM positions WHERE id=?", (position_id,)).fetchone()[0])
        if old["status"] == "CLOSED":
            return old
        entry = f[f.date.eq(position["entry_date"])]
        if entry.empty or not math.isclose(float(entry.close.iloc[0]), position["entry_price"], rel_tol=1e-6):
            raise ValueError("入场行情缺失或已修订，请核验后再卖出")
        position.update(status="CLOSED", exit_date=as_of, exit_price=float(quote.close),
                        closed_at=datetime.now(timezone.utc).isoformat(),
                        return_pct=float((quote.index_close / entry.index_close.iloc[0] - 1) * 100))
        conn.execute("UPDATE positions SET status='CLOSED', payload=? WHERE id=?", (json.dumps(position, ensure_ascii=False), position_id))
    return position


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    run(parser.parse_args().date)
