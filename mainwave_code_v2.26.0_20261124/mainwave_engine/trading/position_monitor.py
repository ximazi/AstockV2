"""Position monitor, independent of candidate generation (strong_trend_core_v1 §04, S06/S13).

Runs on the ledgers and local daily bars only: a failed scan, AI timeout or news error never stops it.
It is a read-only projection. Exits are still queued by each book's frozen exit policy in the after-close
closure (single writer per ledger); this module states what that contract requires, flags what the
closure has not yet processed, and never writes quantities or cash.

Five orthogonal states per position (trade / trend / data / execution / profit) plus one action, in the
design's priority order. Missing data is never reported as normal.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from . import etf_dip, paper, profit_tracker, ranking, store

VERSION = "position_monitor_v1_20260923"
OUT = store.ROOT / "results/trading_system/position_monitor"
EXIT_OPEN = ("OPEN", "PARTIAL", "EXIT_BLOCKED", "UNKNOWN")
ACTION_WORDS = {"EXIT_REQUIRED": "必须退出", "REDUCE_REQUIRED": "按原合同减仓", "MANUAL_REVIEW": "人工核查",
                "HOLD_REVIEW": "持有并复核", "HOLD": "按原合同持有", "ADD_CANDIDATE": "加仓候选"}
MARKET_INDEX = "000852_SH"  # CSI 1000
MARKET_RULE = "趋势清仓 G1：中证1000 收盘 < MA20 且 MA20 低于 5 个交易日前（大跌预警与分级减仓_20260924.md）"
REBOUND_RULE = ("大跌反弹篮子 P2：中证1000 距 60 日最高收盘 −15%/−20%/−25% 各投 1/3，买当日距自身 60 日高跌幅最大的 1/5，"
                "取约 20 只等权，次日收盘买入，每档持有 20 个交易日，不设个股止损；与趋势持仓分账（大跌后抄底研究_20260925.md）")
WARNING_INDEX = {"000300_SH": "沪深300", "000905_SH": "中证500", "000852_SH": "中证1000", "399006_SZ": "创业板指"}
WARNING_WINDOW = 6
WARNING_RULE = ("短期回调预警（research/index_warning_signals.py，2024 至今与 2016-2023 两段一致）：指数 MA20 刚拐头向下"
                "（低于 5 日前）的前 6 个交易日，之后 10-20 日平均偏弱；MA20 下行已超过 6 天的不再预警（历史上随后多为反弹）。"
                "预警约一半是误报，只提示，不是卖出信号。MA60 斜率转负不作预警（出现时下跌多已发生）。")
REBOUND_LEVELS = (-0.15, -0.20, -0.25)
REBOUND_WINDOW = 60  # sessions after a leg's first -15% day in which the deeper tranches may still fill
TREND_FROM_VALUATION = {"HOLD": "INTACT", "EXIT_REQUIRED": "BROKEN", "REDUCE_REQUIRED": "DEGRADED",
                        "RULES_REVIEW_HOLD": "UNKNOWN", "DATA_WAIT": "UNKNOWN"}


def books():
    from . import books as B
    return {"MANUAL": store.DB_PATH, "AUTO": store.ROOT / B.AUTO_DB, "NEW100K": store.ROOT / B.NEW100K_DB}


def bar_state(symbol, day, data_root):
    path = Path(data_root) / f"{symbol}_daily.csv"
    if not path.exists():
        return {"data_state": "MISSING", "reason": "本地无日线文件"}
    try:
        f = ranking.prices(path, day)
    except (OSError, ValueError, KeyError) as exc:
        return {"data_state": "MISSING", "reason": f"日线读取失败：{exc}"[:120]}
    if f.empty:
        return {"data_state": "MISSING", "reason": "无截至当日的日线"}
    r = f.iloc[-1]
    if r["date"] != day:
        return {"data_state": "STALE", "last_day": r["date"], "close": float(r["close"]), "reason": f"最新日线为 {r['date']}"}
    return {"data_state": "FRESH", "last_day": day, "close": float(r["close"]), "low": float(r["low"]),
            "suspended": ranking.number(r.get("tradestatus")) == 0}


def _index_frame(day, data_root):
    f = pd.read_csv(Path(data_root) / f"{MARKET_INDEX}_index.csv")
    f["date"] = pd.to_datetime(f["trade_date"].astype(str)).dt.strftime("%Y-%m-%d")
    return f[f["date"] <= day].drop_duplicates("date").sort_values("date").reset_index(drop=True)


def market_risk(day, data_root):
    """Market trend-exit alert. Report only: it never changes a position's action, an order or a quantity."""
    base = {"index": "中证1000", "rule": MARKET_RULE, "report_only": True}
    try:
        f = _index_frame(day, data_root)
    except (OSError, ValueError, KeyError) as exc:
        return {**base, "level": "UNKNOWN", "text": f"中证1000 日线不可用（{str(exc)[:80]}），大盘风险未知，不按正常处理"}
    if len(f) < 25:
        return {**base, "level": "UNKNOWN", "text": "中证1000 历史不足 25 个交易日，大盘风险未知"}
    c = f["close"].astype(float).reset_index(drop=True)
    ma20 = c.rolling(20).mean()
    close, m, m5 = float(c.iloc[-1]), float(ma20.iloc[-1]), float(ma20.iloc[-6])
    facts = {"as_of": f["date"].iloc[-1], "close": round(close, 2), "ma20": round(m, 2), "ma20_5d_ago": round(m5, 2),
             "close_vs_ma20": round(close / m - 1, 4), "drawdown_20d": round(close / float(c.iloc[-20:].max()) - 1, 4)}
    below, turning = close < m, m < m5
    # tomorrow's MA20 if the index merely closes flat, against the level it will then be compared with
    flat_tomorrow = (float(c.iloc[-19:].sum()) + close) / 20
    m4 = float(ma20.iloc[-5])
    facts["ma20_if_flat_tomorrow"] = round(flat_tomorrow, 2)
    if facts["as_of"] != day:
        return {**base, **facts, "level": "STALE", "text": f"中证1000 最新日线为 {facts['as_of']}，今日大盘风险未确认，不按正常处理"}
    if below and turning:
        return {**base, **facts, "level": "TREND_EXIT",
                "text": f"大盘趋势破坏：中证1000 收盘 {close:.2f} 低于 MA20 {m:.2f}，且 MA20 在下拐（5 日前 {m5:.2f}）。"
                        "按趋势清仓规则应清仓、停止新买（仅提示，不自动卖出）。解除：收盘回到 MA20 上方或 MA20 不再下拐。"}
    if below and flat_tomorrow < m4:
        return {**base, **facts, "level": "CRITICAL",
                "text": f"大盘趋势临界：中证1000 收盘 {close:.2f} 低于 MA20 {m:.2f}，明日平收 MA20 即下拐（临界 {flat_tomorrow:.2f} 对 {m4:.2f}），"
                        "趋势清仓一步之遥。不建议新开仓（仅提示）。解除：收盘回到 MA20 上方。"}
    state = "收盘在 MA20 下方，MA20 尚未下拐" if below else "MA20 在下拐，收盘仍在 MA20 上方" if turning else "收盘在 MA20 上方，MA20 未下拐"
    return {**base, **facts, "level": "NORMAL", "text": f"大盘趋势未破坏：{state}（收盘 {close:.2f}，MA20 {m:.2f}）"}


def beaten_basket(day, data_root, pick=20):
    """Most beaten fifth of non-ST A-shares by close vs own 60-session high (compounded pct_chg), largest amount first."""
    rows = []
    for path in sorted(Path(data_root).glob("s[hz]_*_daily.csv")):
        if path.name[:5] not in ("sh_60", "sz_00", "sz_30", "sh_68"):
            continue
        try:
            f = pd.read_csv(path, usecols=["date", "pct_chg", "tradestatus", "isST", "amount"])
        except (OSError, ValueError):
            continue
        f["date"] = f["date"].astype(str).str[:10]
        f = f[(f["date"] <= day) & (f["tradestatus"] == 1) & f["pct_chg"].notna()].drop_duplicates("date").tail(60)
        if len(f) < 40 or f["date"].iloc[-1] != day or ranking.number(f["isST"].iloc[-1]) == 1:
            continue
        c = (1 + f["pct_chg"] / 100).cumprod()
        rows.append({"symbol": path.name[:9], "dd60": float(c.iloc[-1] / c.max() - 1), "amount": float(f["amount"].iloc[-1] or 0)})
    if not rows:
        return {"universe": 0, "q1_size": 0, "picks": [], "pick_rule": "无可用个股日线"}
    x = pd.DataFrame(rows)
    q1 = x[x["dd60"] <= x["dd60"].quantile(0.2)].sort_values("amount", ascending=False)
    from .. import strong_trend
    names = strong_trend.names(store.ROOT)
    return {"universe": len(x), "q1_size": len(q1), "q1_dd60_max": round(float(q1["dd60"].max()), 4),
            "pick_rule": "Q1 内按当日成交额取前 20（随机 20 只已验证接近整篮；按成交额挑选未验证）",
            "picks": [{"symbol": r.symbol, "name": names.get(r.symbol, ""), "dd60": round(r.dd60, 4), "amount_yi": round(r.amount / 1e8, 2)}
                      for r in q1.head(pick).itertuples()]}


def market_warning(day, data_root):
    """Early pullback warning on four broad indices. Report only: never changes a position, a plan or an order."""
    rows = []
    for code, name in WARNING_INDEX.items():
        try:
            f = pd.read_csv(Path(data_root) / f"{code}_index.csv")
        except (OSError, ValueError):
            rows.append({"index": name, "level": "UNKNOWN", "text": "本地无日线"})
            continue
        f["date"] = pd.to_datetime(f["trade_date"].astype(str)).dt.strftime("%Y-%m-%d")
        f = f[f["date"] <= day].drop_duplicates("date").sort_values("date").reset_index(drop=True)
        if len(f) < 66 or f["date"].iloc[-1] != day:
            rows.append({"index": name, "level": "STALE", "text": f"最新日线 {f['date'].iloc[-1] if len(f) else '无'}，今日未确认"})
            continue
        c = f["close"].astype(float)
        ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
        k = len(f) - 1
        while k > 5 and ma20.iloc[k] < ma20.iloc[k - 5]:
            k -= 1
        age = len(f) - 1 - k                   # sessions MA20 has been below its level 5 sessions earlier
        since = f["date"].iloc[k + 1] if age else ""
        below60 = "，且收盘已在 MA60 下" if c.iloc[-1] < ma60.iloc[-1] else ""
        level = "WARNING" if 0 < age <= WARNING_WINDOW else "NORMAL"
        text = (f"MA20 自 {since} 拐头向下（第 {age} 天）{below60}：短期回调风险，不追高、新仓谨慎" if level == "WARNING"
                else f"MA20 已下行 {age} 天，预警期已过" if age else "MA20 未下行")
        rows.append({"index": name, "level": level, "close": round(float(c.iloc[-1]), 2), "ma20": round(float(ma20.iloc[-1]), 2),
                     "ma60": round(float(ma60.iloc[-1]), 2), "ma20_falling_since": since, "ma20_falling_days": age, "text": text})
    known = [r["level"] for r in rows if r["level"] in ("WARNING", "NORMAL")]
    level = "WARNING" if "WARNING" in known else "NORMAL" if known else "UNKNOWN"
    return {"rule": WARNING_RULE, "report_only": True, "level": level, "indices": rows}


def rebound_basket(day, data_root):
    """Crash rebound basket (P2). Report only: it never creates a plan, an order or a quantity."""
    base = {"index": "中证1000", "rule": REBOUND_RULE, "report_only": True}
    try:
        f = _index_frame(day, data_root)
    except (OSError, ValueError, KeyError) as exc:
        return {**base, "level": "UNKNOWN", "text": f"中证1000 日线不可用（{str(exc)[:80]}），反弹篮子状态未知"}
    if len(f) < 60:
        return {**base, "level": "UNKNOWN", "text": "中证1000 历史不足 60 个交易日，反弹篮子状态未知"}
    c = f["close"].astype(float)
    dd = (c / c.rolling(60).max() - 1).to_numpy()
    dates, n = f["date"].tolist(), len(f)
    facts = {"as_of": dates[-1], "close": round(float(c.iloc[-1]), 2), "drawdown_60d": round(float(dd[-1]), 4)}
    if dates[-1] != day:
        return {**base, **facts, "level": "STALE", "text": f"中证1000 最新日线为 {dates[-1]}，今日反弹篮子状态未确认"}
    hit = dd <= REBOUND_LEVELS[0]
    # a leg starts on a -15% day with no -15% day in the 60 sessions before it
    starts = [k for k in range(n) if hit[k] and not hit[max(0, k - REBOUND_WINDOW):k].any()]
    start = starts[-1] if starts and n - 1 - starts[-1] < REBOUND_WINDOW else None
    if start is None:
        return {**base, **facts, "level": "INACTIVE",
                "text": f"大跌反弹篮子未触发：中证1000 距 60 日最高收盘 {dd[-1]:.1%}（第一档在 −15%）"}
    tranches = []
    for i, lvl in enumerate(REBOUND_LEVELS, 1):
        k = next((k for k in range(start, n) if dd[k] <= lvl), None)
        hold_end = None if k is None else dates[k + 21] if k + 21 < n else "买入后第 20 个交易日"
        tranches.append({"tranche": i, "level": lvl, "signal_day": dates[k] if k is not None else None, "hold_until": hold_end})
    today = [t for t in tranches if t["signal_day"] == day]
    done = "；".join(f"第{t['tranche']}档({t['level']:.0%}) " + (f"{t['signal_day']} 触发，持有至 {t['hold_until']}" if t["signal_day"] else "未到")
                    for t in tranches)
    window = f"本轮起点 {dates[start]}，再过 {REBOUND_WINDOW - (n - 1 - start)} 个交易日未到的档位作废"
    out = {**base, **facts, "leg_start": dates[start], "tranches": tranches}
    if today:
        t = today[-1]
        return {**out, "level": "TRIGGERED_TODAY", "basket": beaten_basket(day, data_root),
                "text": f"大跌反弹篮子第 {t['tranche']} 档今日触发：中证1000 距 60 日高 {dd[-1]:.1%}。"
                        f"明日收盘买入 1/3，持有 20 个交易日（仅提示，不下单）。{done}。{window}。"}
    return {**out, "level": "ACTIVE", "text": f"大跌反弹篮子进行中：距 60 日高 {dd[-1]:.1%}。{done}。{window}。"}


def _valuations(conn, pid, day):
    rows = [(r["as_of"], json.loads(r["payload"])) for r in conn.execute(
        "SELECT as_of, payload FROM paper_valuations WHERE plan_id=? AND as_of<=? ORDER BY as_of, rowid", (pid, day))]
    today = next((v for d, v in reversed(rows) if d == day), None)
    before = next((v for d, v in reversed(rows) if d < day), None)
    return today, before


def assess(conn, plan, day, cfg, data_root):
    from . import corporate_actions
    pid = plan["id"]
    lots = [dict(r) for r in conn.execute("SELECT * FROM paper_lots WHERE plan_id=? ORDER BY entry_day, id", (pid,))]
    held = sum(l["remaining"] for l in lots)
    if not held:
        return None
    today, before = _valuations(conn, pid, day)
    bar = bar_state(plan["symbol"], day, data_root)
    if corporate_actions.pending_for(conn, pid):
        bar = {**bar, "data_state": "CORPORATE_ACTION_PENDING", "reason": "公司行动待确认；股数/价格/保护线未对齐前不做风险授权"}
    # Protection already in force today comes from an earlier close; today's close only sets the next one.
    effective = profit_tracker.protection_of(plan, before or {})
    upcoming = profit_tracker.protection_of(plan, today) if today and today.get("status") != "DATA_WAIT" else None
    hard = plan.get("stop_cents")
    exits = [dict(r) for r in conn.execute(
        "SELECT * FROM paper_orders WHERE plan_id=? AND side='SELL' AND status IN ('OPEN','PARTIAL','EXIT_BLOCKED','UNKNOWN')", (pid,))]
    pending_exit = sum(o["quantity"] - o["filled"] for o in exits)
    sellable = sum(l["remaining"] for l in lots if l["entry_day"] < day)
    statuses = {o["status"] for o in exits}
    trade_state = ("EXIT_BLOCKED" if statuses & {"EXIT_BLOCKED", "UNKNOWN"} else "EXIT_PARTIAL" if "PARTIAL" in statuses
                   else "EXIT_PENDING" if exits else "OPEN")
    # Trend: the closure's frozen exit policy is authoritative; the monitor only adds the protection checks.
    trend, trend_basis = "UNKNOWN", "今日收盘闭环尚未对此持仓估值"
    if today:
        trend = TREND_FROM_VALUATION.get(today.get("status"), "UNKNOWN")
        trend_basis = f"收盘闭环（{today.get('policy_version') or plan.get('exit_version')}）：{today.get('status')} {today.get('reason') or ''}".strip()
    breach = None
    if bar["data_state"] == "FRESH":
        if hard and round(bar["low"] * 100) <= hard:
            breach = f"日内最低 {bar['low']:.2f} 触及初始硬止损 {hard / 100:.2f}"
        elif effective and round(bar["close"] * 100) < effective:
            breach = f"收盘 {bar['close']:.2f} 跌破此前已生效保护 {effective / 100:.2f}"
        if breach:
            trend, trend_basis = "BROKEN", breach + ("（收盘闭环未处理，待闭环按冻结合同登记退出）" if not today else "")
        elif not today:
            trend, trend_basis = "INTACT", "仅核对了已生效保护线与硬止损；冻结退出合同的完整判断待收盘闭环"
    mark = None
    if bar["data_state"] == "FRESH":
        mark = (round(bar["close"] * 100), day, "MONITOR_DAILY_CLOSE")
    c = profit_tracker.campaign(conn, plan, day, cfg, upcoming or effective, mark_override=mark)
    ratio = c.get("giveback_ratio")
    pnl = c.get("net_pnl_cents")
    profit_state = ("UNKNOWN" if pnl is None else
                    "GIVEBACK_FROM_PEAK" if c.get("giveback_alert") else
                    "NEW_PEAK" if c["observed_peak_cents"] and pnl >= c["observed_peak_cents"] else
                    "NEAR_BREAKEVEN" if abs(pnl) <= (c.get("estimated_exit_cost_cents") or 0) else
                    "NET_PROFIT" if pnl > 0 else "NET_LOSS")
    # Action priority (design §04.4): hard risk / pending exit > trend exit > reduce > data > review > hold.
    blocks = []
    if trade_state != "OPEN":
        action = "EXIT_REQUIRED"
        reason = "已有退出委托未完成" + ("，当前受阻，敞口仍计入风险" if trade_state == "EXIT_BLOCKED" else "")
    elif trend == "BROKEN":
        action, reason = "EXIT_REQUIRED", trend_basis
    elif today and today.get("status") == "REDUCE_REQUIRED":
        action, reason = "REDUCE_REQUIRED", f"原退出合同要求减仓：{today.get('reason')}"
    elif bar["data_state"] != "FRESH":
        action, reason = "MANUAL_REVIEW", f"数据状态 {bar['data_state']}：{bar.get('reason', '')}；保留最后可信值，停止依赖它的新风险授权"
    elif trend in ("DEGRADED", "UNKNOWN") or c.get("giveback_alert"):
        action = "HOLD_REVIEW"
        reason = "；".join(x for x in (trend_basis if trend != "INTACT" else "",
                                       f"利润回吐 {ratio:.0%}（仅告警，不自动卖出）" if c.get("giveback_alert") else "") if x)
    else:
        action, reason = "HOLD", "原交易依据未被价格证伪，按原合同持有"
    if action in ("EXIT_REQUIRED", "REDUCE_REQUIRED"):
        blocks.append("存在退出/减仓要求")
    if bar["data_state"] != "FRESH":
        blocks.append("数据未就绪")
    if trend != "INTACT":
        blocks.append(f"趋势状态 {trend}")
    if bar.get("suspended"):
        blocks.append("停牌")
    next_event = ("下一可交易开盘按退出委托撮合；跌停/停牌则记为退出受阻并继续跟踪" if action == "EXIT_REQUIRED" else
                  f"收盘跌破 {(upcoming or effective or 0) / 100:.2f} 即触发退出" if (upcoming or effective) else "等待保护线建立")
    return {"campaign_id": pid, "symbol": plan["symbol"], "name": plan.get("name"), "theme": plan.get("theme"),
            "trade_state": trade_state, "trend_state": trend, "data_state": bar["data_state"],
            "execution_state": {"held": held, "sellable": sellable, "t_plus_1_locked": held - sellable,
                                "pending_exit_qty": pending_exit, "suspended": bool(bar.get("suspended")),
                                "exit_orders": [{"id": o["id"], "status": o["status"], "left": o["quantity"] - o["filled"]} for o in exits]},
            "profit_state": profit_state, "action": action, "action_word": ACTION_WORDS[action], "action_reason": reason,
            "add_blocked_by": blocks,
            "six_questions": {
                "why_bought": {"entry_day": c["entry_day"], "mode": plan.get("mode"), "setup": plan.get("setup_kind"),
                               "routes": plan.get("route_ids"), "still_valid": trend_basis},
                "holding": {"held": held, "sellable": sellable, "locked_t1": held - sellable},
                "protection": {"initial_hard_cents": hard, "effective_today_cents": effective,
                               "next_session_cents": upcoming,
                               "next_effective": "下一交易日" if upcoming else None},
                "risk": {k: c.get(k) for k in ("net_pnl_if_exit_at_protection_cents", "net_pnl_if_stress_exit_cents",
                                               "current_wealth_loss_in_stress_cents", "principal_loss_in_stress_cents",
                                               "stress_basis")},
                "add": "禁止：" + "、".join(blocks) if blocks else "未被禁止；是否加仓只由该计划冻结的加仓合同在收盘闭环评估",
                "action": {"action": action, "next_event": next_event}},
            "profit": c, "bar": bar}


def run(day, data_root=None, ledgers=None):
    day = pd.Timestamp(day).date().isoformat()
    data_root = Path(data_root or store.ROOT / "data")
    result = {"version": VERSION, "as_of": day, "computed_at": store.now(), "live_orders_enabled": False,
              "notice": "独立持仓监控：只读账本与本地日线，不依赖当日选股流水线；不写数量与现金。" + profit_tracker.NOTICE,
              "market_risk": market_risk(day, data_root), "market_warning": market_warning(day, data_root),
              "rebound_basket": rebound_basket(day, data_root), "etf_dip": etf_dip.alert(day, data_root),
              "books": {}, "alerts": []}
    for book, db in (ledgers or books()).items():
        if not Path(db).exists():
            result["books"][book] = {"status": "NO_LEDGER"}
            continue
        try:
            with paper.database(db) as conn:
                raw = conn.execute("SELECT * FROM paper_account").fetchone()
                if not raw:
                    result["books"][book] = {"status": "UNCONFIGURED"}
                    continue
                cfg = json.loads(raw["config"])
                plans = [json.loads(r[0]) for r in conn.execute("SELECT payload FROM paper_plans ORDER BY created_at")]
                positions = [p for p in (assess(conn, plan, day, cfg, data_root) for plan in plans) if p]
                pending_buys = [{"order_id": o["id"], "symbol": json.loads(o["plan"])["symbol"], "status": o["status"],
                                 "quantity": o["quantity"] - o["filled"], "reserve_cents": o["reserve_cents"]}
                                for o in conn.execute("SELECT o.*, p.payload AS plan FROM paper_orders o JOIN paper_plans p ON p.id=o.plan_id "
                                                      "WHERE o.side='BUY' AND o.status IN ('OPEN','PARTIAL','UNKNOWN')")]
            ledger = paper.view(db)
            profits = profit_tracker.report(db, day)
            result["books"][book] = {"status": "READY", "reconciled": ledger["reconciled"], "positions": positions,
                                     "pending_buys": pending_buys, "account": profits["account"],
                                     "closed_campaigns": [c for c in profits["campaigns"] if c["trade_closed"]]}
            if not ledger["reconciled"]:
                result["alerts"].append({"level": "P0", "book": book, "text": "账实不符：现金或股数与成交不守恒，停止新增风险"})
            for p in positions:
                level = ("P0" if p["trade_state"] == "EXIT_BLOCKED" else "P1" if p["action"] in ("EXIT_REQUIRED", "REDUCE_REQUIRED", "MANUAL_REVIEW")
                         else "P2" if p["action"] == "HOLD_REVIEW" else None)
                if level:
                    result["alerts"].append({"level": level, "book": book, "symbol": p["symbol"], "name": p["name"],
                                             "action": p["action_word"], "text": p["action_reason"]})
        except Exception as exc:  # one broken ledger never hides the others
            result["books"][book] = {"status": "FAILED", "error": str(exc)[:300]}
            result["alerts"].append({"level": "P0", "book": book, "text": f"账本读取失败：{str(exc)[:120]}"})
    result["alerts"].sort(key=lambda a: a["level"])
    result["status"] = "ALERT" if any(a["level"] == "P0" for a in result["alerts"]) else "READY"
    return result


def yuan(cents):
    return "—" if cents is None else f"{cents / 100:,.2f}"


def render(result):
    lines = [f"# 持仓行动 {result['as_of']}", ""]
    m = result.get("market_risk") or {}
    if m.get("level") == "TREND_EXIT":
        lines += ['<div style="background:#c00000;color:#fff;padding:14px 18px;font-size:20px;font-weight:bold;border:4px solid #7a0000">',
                  f"🔴🔴🔴 大盘风险：趋势清仓 🔴🔴🔴<br>{m['text']}<br>规则：{m['rule']}</div>", ""]
    elif m.get("level") == "CRITICAL":
        lines += [f'<div style="background:#b36b00;color:#fff;padding:12px 16px;font-size:18px;font-weight:bold;border:3px solid #7a4600">🟠 {m["text"]}</div>', ""]
    elif m.get("level") in ("STALE", "UNKNOWN"):
        lines += [f'<div style="background:#b36b00;color:#fff;padding:10px 14px;font-weight:bold">🟠 大盘风险未确认：{m.get("text")}</div>', ""]
    elif m:
        lines += [f"🟢 {m['text']}", ""]
    w = result.get("market_warning") or {}
    if w.get("level") == "WARNING":
        items = "；".join(f"{r['index']}：{r['text']}" for r in w["indices"] if r["level"] == "WARNING")
        lines += [f'<div style="background:#b35900;color:#fff;padding:10px 14px;font-weight:bold">🟠 短期回调预警：{items}</div>',
                  f"<small>{w['rule']}</small>", ""]
    rb = result.get("rebound_basket") or {}
    if rb.get("level") == "TRIGGERED_TODAY":
        b = rb["basket"]
        picks = "、".join(f"{x['name']}{x['symbol'][3:]}({x['dd60']:.0%})" for x in b["picks"]) or "无"
        lines += ['<div style="background:#1f3a8a;color:#fff;padding:14px 18px;font-size:18px;font-weight:bold;border:4px solid #0b1f5c">',
                  f"🟦 大跌反弹篮子触发 🟦<br>{rb['text']}<br>候选（Q1 共 {b['q1_size']} 只，{b['pick_rule']}）：{picks}<br>"
                  f"规则：{rb['rule']}</div>", ""]
    elif rb.get("level") == "ACTIVE":
        lines += [f'<div style="background:#1f3a8a;color:#fff;padding:10px 14px;font-weight:bold">🟦 {rb["text"]}</div>', ""]
    elif rb.get("level") in ("STALE", "UNKNOWN"):
        lines += [f"🟠 {rb.get('text')}", ""]
    elif rb:
        lines += [f"⚪ {rb['text']}", ""]
    dip = result.get("etf_dip") or {}
    if dip.get("level") in ("TRIGGERED_TODAY", "HOLDING"):
        new = "；".join(f"{r['index']}（{r['group']}，{r['signal']}，抄底仓的 {r['size_of_sleeve']:.1%}{'，ETF ' + r['etf'] if r['etf'] else ''}）"
                       for r in dip["new"])
        hold = "；".join(f"{r['index']} {r['signal']}（{r['entry_day']} 开盘 {r['entry']:.2f} 买入，占 {r['size_of_sleeve']:.1%}，"
                        f"还剩 {r['sessions_left']} 个交易日后卖出）" for r in dip["holding"])
        lines += ['<div style="background:#0f766e;color:#fff;padding:12px 16px;font-weight:bold">',
                  (f"🟩 宽基 ETF 抄底信号：明日开盘按规则买入 {new}<br>" if dip["new"] else "")
                  + (f"🟩 抄底仓按规则持有：{hold}（合计占抄底仓 {dip['exposure']:.1%}）" if dip["holding"] else "") + "</div>",
                  f"<small>{dip['rule']}</small>", ""]
    elif dip:
        lines += [f"⚪ 宽基 ETF 抄底：{dip.get('text')}", ""]
    lines += [result["notice"], ""]
    if result["alerts"]:
        lines += ["## 待处理", ""] + [f"- **{a['level']} {a['book']}** {a.get('name') or ''}{a.get('symbol') or ''} {a.get('action') or ''}：{a['text']}"
                                   for a in result["alerts"]] + [""]
    for book, b in result["books"].items():
        lines += [f"## {book}", ""]
        if b.get("status") != "READY":
            lines += [f"状态：{b.get('status')} {b.get('error', '')}", ""]
            continue
        a = b["account"]
        lines.append(f"净资产 {yuan(a['equity_cents'])}；现金 {yuan(a['cash_cents'])}；账户利润 {yuan(a['money_pnl_cents'])}；"
                     f"单位净值回撤 {a['max_drawdown_unit_nav']:.2%}；压力情景合计损失 {yuan(a['stress_wealth_loss_cents'])}；对账 {'通过' if b['reconciled'] else '失败'}。")
        if b["pending_buys"]:
            lines.append("未决买单（占席位与资金，未成交不记仓）：" + "、".join(f"{o['symbol']} {o['quantity']}股" for o in b["pending_buys"]))
        if not b["positions"]:
            lines += ["", "无持仓。", ""]
            continue
        lines += ["", "|股票|动作|持有/可售|净盈亏|峰值/回吐|今日保护→下一保护|按保护退出估算|压力退出|原因|",
                  "|---|---|---|---:|---|---|---:|---:|---|"]
        for p in b["positions"]:
            c, q = p["profit"], p["six_questions"]
            ratio = "—" if c.get("giveback_ratio") is None else f"{c['giveback_ratio']:.0%}"
            lines.append(f"|{p['name']} {p['symbol']}|{p['action_word']}|{q['holding']['held']}/{q['holding']['sellable']}|"
                         f"{yuan(c.get('net_pnl_cents'))}|{yuan(c['observed_peak_cents'])}/{ratio}|"
                         f"{yuan(q['protection']['effective_today_cents'])}→{yuan(q['protection']['next_session_cents'])}|"
                         f"{yuan(c.get('net_pnl_if_exit_at_protection_cents'))}|{yuan(c.get('net_pnl_if_stress_exit_cents'))}|{p['action_reason']}|")
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    result = run(args.date)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / result["as_of"].replace("-", "")
    stem.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    stem.with_suffix(".md").write_text(render(result), encoding="utf-8")
    m = result["market_risk"]
    mark = "!!!!!!!!!! " if m["level"] == "TREND_EXIT" else ">>>>>>>>>> " if m["level"] == "CRITICAL" else ""
    print(f"[position_monitor] {mark}大盘风险 {m['level']}：{m['text']}", flush=True)
    w = result["market_warning"]
    print(f"[position_monitor] 短期回调预警 {w['level']}：" + "；".join(f"{r['index']} {r['level']}" for r in w["indices"]), flush=True)
    dip = result["etf_dip"]
    print(f"[position_monitor] {'>>>>>>>>>> ' if dip['level'] == 'TRIGGERED_TODAY' else ''}ETF 抄底 {dip['level']}：{dip['text']}", flush=True)
    rb = result["rebound_basket"]
    print(f"[position_monitor] {'>>>>>>>>>> ' if rb['level'] == 'TRIGGERED_TODAY' else ''}反弹篮子 {rb['level']}：{rb['text']}", flush=True)
    counts = {b: len(v.get("positions", [])) for b, v in result["books"].items()}
    print(f"[position_monitor] {result['as_of']} status={result['status']} positions={counts} alerts={len(result['alerts'])} -> {stem}.json", flush=True)
    return 2 if result["status"] == "ALERT" else 0


if __name__ == "__main__":
    raise SystemExit(main())
