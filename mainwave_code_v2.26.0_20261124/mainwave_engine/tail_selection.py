"""Separate entry tactics fed by the preparation pool and existing leader/board observations."""
from collections import Counter

from . import intraday_quotes as Q, potential_pool as P

VERSION = "tail_selection_v1_20260919"
NOTICE = "独立试行策略：点火、EMA10回踩下影、点火后回踩、龙头回踩、1进2观察。条件满足仅供人工复核，不自动下单；未验证收益。"


def build(report, pool, previous_day):
    at, day = report["observed_at"], report["as_of"]
    pool_ok = (pool.get("version") == P.VERSION and pool.get("as_of") == previous_day < day
               and pool.get("known_at") and Q.stamp(pool["known_at"]) <= Q.stamp(at))
    members = {r["symbol"]: r for r in pool.get("records", []) if r["state"] == "POTENTIAL"} if pool_ok else {}
    observation = report.get("tail_observation", {})
    leaders = {r["symbol"]: r for r in observation.get("leaders", [])}
    boards = {r["symbol"]: r for r in observation.get("boards", [])}
    holdings = {r["symbol"]: r for r in report.get("holdings", [])}
    records = []
    universe = {r["symbol"]: r for r in report["records"]}
    for symbol in set(leaders) | set(boards):
        universe.setdefault(symbol, {"symbol": symbol, "name": boards.get(symbol, leaders.get(symbol, {})).get("name", "")})
    for row in universe.values():
        symbol, q = row["symbol"], row.get("quote") or {}
        seed = row.get("preparation", {})
        usable = (seed.get("as_of") == previous_day and seed.get("status") == "READY"
                  and q.get("quality") == "DAILY_SNAPSHOT" and row.get("metrics", {}).get("status") == "PROVISIONAL")
        if usable:
            age = (Q.stamp(at) - Q.stamp(q["observed_at"])).total_seconds()
            usable = 0 <= age <= 120 and Q.stamp(q["observed_at"]).date().isoformat() == day
        setup = P.entry_checks(seed, q) if usable else P.entry_checks({}, {})
        in_pool = symbol in members
        if not (in_pool or symbol in leaders or symbol in boards or setup["ignition"] or setup["pullback"]):
            continue
        strategies, reasons, blockers = [], [], []
        if in_pool:
            reasons.append(f"{previous_day}已在潜在池；首次入池{members[symbol]['first_seen']}")
        if setup["ignition"]:
            strategies.append("点火启动")
            reasons.append("突破前20日高点、强收位置及尾盘累计量能满足试行条件")
        if setup["pullback"]:
            strategies.append("EMA10回踩下影")
            reasons.append("上升趋势中回测前收盘EMA10参考带，出现下影并收回均线")
        if setup["combined"]:
            strategies.append("点火后回踩")
            reasons.append(f"{setup['ignition_date']}已有点火迹象且事件低点未破")
        leader = leaders.get(symbol, {})
        if leader:
            strategies.append(("龙头候选回踩" if setup["pullback"] else "日线领头候选") if leader["state"] == "DAILY_LEADER" else "题材前排观察")
            reasons.append("角色为候选，未以涨幅排名证明分钟带动性")
        board = boards.get(symbol, {})
        if board:
            strategies.append("1进2晋级观察" if board.get("group") == "TODAY_PROMOTION" else "首板次日预备")
            reasons.extend(board.get("evidence", []))
        bonus = P.RULES["pool_bonus"] if in_pool and setup["ignition"] else 0
        if bonus:
            reasons.append(f"点火前已入池，关注加分+{bonus}，不抵消任何风险")
        price_pattern = setup["ignition"] or setup["pullback"]
        if not usable:
            blockers.append("缺少新鲜累计OHLC/量能或前收盘结构基线，分钟单根K线不可替代")
        if not price_pattern:
            blockers.append("尚未形成点火或EMA10回踩下影买点")
        risks = list(setup["risk_flags"])
        risks.extend(row.get("risk_flags", []))
        if row.get("status") == "RISK_WATCH":
            risks.append("原三路径价格/结构风险警报")
        holding = holdings.get(symbol)
        if holding:
            blockers.append("已有虚拟持仓，仅在持仓决策区评估，不列入新增参与名单")
            if holding.get("status") in {"SELL_REVIEW", "REDUCE_REVIEW"} or holding.get("pending_action") in {"SELL_REVIEW", "REDUCE_REVIEW"}:
                risks.append("已有持仓退出或减仓警报，禁止新增参与")
        if usable and not seed.get("liquid", False):
            risks.append("前收盘流动性门槛未通过")
        minute = row.get("minute_context", {})
        minute_ok = (minute.get("status") == "COMPLETE" and minute.get("late_pattern") in {"ABOVE_VWAP", "PULLBACK_ABOVE_VWAP"})
        if not minute_ok:
            blockers.append("等待完整分钟轨迹与VWAP承接")
        if setup["pullback"] and not minute.get("reclaim_confirmed", False):
            blockers.append("等待分钟级回落后再转强顺序确认")
        theme = row.get("live_theme") or {}
        if theme.get("state") not in {"HOT", "ACTIVE"}:
            blockers.append("实时题材缺失、偏弱或分化")
        if row.get("relative_index_pp") is None or row["relative_index_pp"] < 0:
            blockers.append("未确认强于对应指数")
        if not report["market"]["new_review_allowed"]:
            blockers.append("市场或尾盘参与窗口未开放")
        if board.get("state") in {"SECOND_PRICE", "FIRST_TOUCH", "TOUCH_UNCLASSIFIED"}:
            blockers.append("处于涨停附近，未核验排队成交，不开放模拟参与复核")
        board_missing = board.get("state") == "DATA_WAIT"
        if board_missing:
            blockers.append("1进2关键数据缺失，等待首板记录、精确涨停价与行情核验")
        blockers.extend(risks)
        status = ("RISK_WATCH" if risks else "DATA_WAIT" if not usable or board_missing else
                  "WAIT_CONFIRMATION" if not price_pattern or not minute_ok or (setup["pullback"] and not minute.get("reclaim_confirmed")) else
                  "MARKET_WAIT" if not report["market"]["new_review_allowed"] else
                  "WAIT_CONFIRMATION" if blockers else "SETUP_REVIEW")
        # Evidence families share a score; a simultaneous ignition/pullback is not double-counted.
        score = (members.get(symbol, {}).get("score", 0) + bonus + (2 if price_pattern else 0)
                 + int(setup["combined"]) + int(leader.get("state") == "DAILY_LEADER" and setup["pullback"]))
        records.append({"symbol": symbol, "name": row.get("name", ""), "theme": row.get("theme", ""),
                        "route_ids": row.get("route_ids", ""), "strategies": strategies or ["潜在池跟踪"],
                        "status": status, "price": q.get("close"), "ema10": setup["ema10"],
                        "lower_wick_ratio": setup["lower_wick_ratio"], "pool_member": in_pool, "pool_bonus": bonus,
                        "score": score, "reasons": reasons, "blockers": blockers, "setup": setup,
                        "board_state": board.get("state"), "leadership_confirmed": False,
                        "paper_review_eligible": status == "SETUP_REVIEW", "orders_enabled": False})
    order = {"SETUP_REVIEW": 0, "MARKET_WAIT": 1, "WAIT_CONFIRMATION": 2, "DATA_WAIT": 3, "RISK_WATCH": 4}
    records.sort(key=lambda r: (order[r["status"]], -r["score"], r["symbol"]))
    focus, reviews, themes = [], [], Counter()
    for row in records:
        if row["status"] in {"RISK_WATCH", "DATA_WAIT"} or themes[row["theme"]] >= 2:
            continue
        if len(focus) < 5:
            focus.append(row["symbol"])
            themes[row["theme"]] += 1
            if row["paper_review_eligible"] and len(reviews) < 3:
                reviews.append(row["symbol"])
    return {"version": VERSION, "as_of": day, "mode": "INTRADAY_PROVISIONAL", "notice": NOTICE,
            "pool_as_of": pool.get("as_of"), "pool_usable": bool(pool_ok), "orders_enabled": False,
            "records": records, "focus_symbols": focus, "paper_review_symbols": reviews,
            "summary": {"focus": len(focus), "reviews": len(reviews), "candidates": len(records),
                        "states": dict(Counter(r["status"] for r in records))}}


def expire(selection):
    selection.update(mode="STALE", focus_symbols=[], paper_review_symbols=[])
    selection["summary"].update(focus=0, reviews=0)
    for row in selection["records"]:
        row.update(status="DATA_WAIT", paper_review_eligible=False)
        row["blockers"].append("盘中快照已过期，仅保留历史观察")


def report_lines(selection):
    lines = ["", "## 5. 潜在池与独立尾盘买点", NOTICE, f"- {selection['summary']}",
             "", "|股票|策略|状态|池内/加分|依据与缺项|", "|---|---|---|---|---|"]
    for r in selection["records"][:15]:
        lines.append(f"|{r['symbol']} {r['name']}|{' / '.join(r['strategies'])}|{r['status']}|{r['pool_member']}/+{r['pool_bonus']}|{'；'.join(r['reasons'] + r['blockers'])}|")
    return lines
