"""Join dated preparation and first-board evidence without creating a trade plan."""
import csv

from .review_clock import instant, visible


VERSION = "first_board_watch_v1_20261120"
LIMIT_UP_VERSION = "p5_limit_up_v1_20261120"
LOOKBACK = 3
MAX_FOCUS = 3


def read_pool(path, day):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {r["symbol"]: r for r in csv.DictReader(stream)
                if r.get("as_of") == day and r.get("detector") == "P5_union"}


def load_pools(root, days, cutoff):
    until = instant(cutoff)
    if not until:
        return {}
    pools = {}
    for i, day in enumerate(days):
        path = root / day.replace("-", "") / "preparation_watch.csv"
        deadline = instant(days[i + 1] + "T09:14:59.999999+08:00") if i + 1 < len(days) else until
        if visible(path, day, min(until, deadline).isoformat()):
            pools[day] = read_pool(path, day)
    return pools


def _held(row):
    return row.get("above_ma20_now") == "True" and row.get("reclaim_failed") == "False"


def limit_up_matches(day, prior_days, pools, boards, heights, rows):
    """Full intersection; theme, chart quality and trade eligibility cannot hide a hit."""
    matches = []
    disputed = set((boards.get("reconciliation", {}).get("U") or {}).get("local_only", []))
    for symbol in sorted(set(boards["U"])):
        before = [d for d in prior_days if symbol in pools.get(d, {})]
        today = pools.get(day, {}).get(symbol)
        if not before and today is None:
            continue
        dates = before + ([day] if today is not None else [])
        evidence = []
        for d in dates:
            p = pools[d][symbol]
            state = "HELD_MA20" if _held(p) else "RECLAIM_FAILED" if p.get("reclaim_failed") == "True" else "WATCH_ONLY"
            evidence.append({"date": d, "state": state})
        frame = rows[symbol]["frame"]
        bar = frame.iloc[-1]
        height = heights.get(symbol)
        first = bool(before) and height == 1
        matches.append({"symbol": symbol, "name": pools[dates[-1]][symbol].get("name") or symbol,
                        "height": height, "priority": "BOARD_DATA_DISPUTED" if symbol in disputed else
                        "PRIOR_P5_FIRST_BOARD" if first else
                        "PRIOR_P5_OTHER_BOARD" if before else "SAME_DAY_P5_ONLY",
                        "evidence": evidence, "first_board_close": float(bar.close), "first_board_low": float(bar.low),
                        "one_price": bool(bar.high == bar.low), "board_disputed": symbol in disputed})
    matches.sort(key=lambda r: (r["priority"] != "PRIOR_P5_FIRST_BOARD", r["priority"], r["symbol"]))
    missing = [d for d in [*prior_days, day] if d not in pools]
    return {"version": LIMIT_UP_VERSION, "as_of": day, "status": "PARTIAL" if missing else "READY",
            "orders_enabled": False, "closing_limit_up_count": len(set(boards["U"])), "unverified_dates": missing,
            "focus_count": sum(r["priority"] == "PRIOR_P5_FIRST_BOARD" for r in matches), "rows": matches}


def limit_up_table_rows(watch, feedback):
    labels = {"PRIOR_P5_FIRST_BOARD": "P5首板重点", "PRIOR_P5_OTHER_BOARD": "P5涨停核对",
              "SAME_DAY_P5_ONLY": "P5同日重合", "BOARD_DATA_DISPUTED": "P5封板待核"}
    for r in watch.get("rows", []):
        dates = "、".join(e["date"] + ("(收复失败)" if e["state"] == "RECLAIM_FAILED" else "") for e in r["evidence"])
        note = ("下一交易日优先复核承接；不追高、不下单。" if r["priority"] == "PRIOR_P5_FIRST_BOARD" else
                "仅核对，不计为首板提前命中，也不是接力许可。")
        if r["one_price"] or r["board_disputed"]:
            note += (" 一字或单价：无可验证成交。" if r["one_price"] else "") + (
                " 收盘涨停参考名单不一致，待核。" if r["board_disputed"] else "")
        height = f"{r['height']}板" if r["height"] else "板数待核"
        yield [labels[r["priority"]], f"{r['name']}（{r['symbol']}）", "--", f"P5记录：{dates}",
               "--", "--", f"收盘 {r['first_board_close']:.2f}；{height}", note]
    for row in table_rows({"picks": []}, feedback):
        yield ["P5首板次日复核", *row[1:]]


def build(day, target, prior_days, pools, core, funnel):
    stocks = {r["symbol"]: r for r in core["rows"]}
    rows, peers = [], {}
    for r in funnel["rows"]:
        if r["grade"] == "CORROBORATED" and r["chain"]:
            peers.setdefault(r["chain"], set()).add(r["symbol"])
    for r in funnel["rows"]:
        s, theme = r["symbol"], r["chain"]
        current = pools.get(day, {}).get(s, {})
        stock = stocks.get(s, {})
        peer_symbols = sorted(peers.get(theme, set()) - {s})
        prior = [d for d in prior_days if _held(pools.get(d, {}).get(s, {}))]
        high, low = stock.get("high"), stock.get("low")
        if not (r["height"] == 1 and r["grade"] == "CORROBORATED" and peer_symbols
                and r.get("engine_state") not in {"RISK_WATCH", "DATA_WAIT"}
                and prior and _held(current) and high is not None and low is not None and high > low):
            continue
        try:
            bias = float(current["bias_atr"])
        except (ValueError, KeyError):
            continue
        if not -1 <= bias <= 2:
            continue
        rows.append({"symbol": s, "name": r["name"], "theme": theme, "peers": peer_symbols,
                     "prior_reclaim": prior[-1], "failed_reclaims": [d for d in prior_days if d > prior[-1]
                          and pools.get(d, {}).get(s, {}).get("reclaim_failed") == "True"],
                     "bias_atr": bias, "first_board_close": stock["close"], "first_board_low": low,
                     "route_state": r.get("engine_state") or "NOT_IN_CANDIDATES"})
    rows.sort(key=lambda r: (-len(r["peers"]), r["bias_atr"], r["symbol"]))
    picks, themes = [], set()
    for row in rows:
        if row["theme"] not in themes and len(picks) < MAX_FOCUS:
            picks.append(row)
            themes.add(row["theme"])
    missing = [d for d in [*prior_days, day] if d not in pools]
    attributed = bool(funnel.get("attribution_available_at"))
    return {"version": VERSION, "as_of": day, "review_on": target, "orders_enabled": False,
            "picks": picks, "qualified_count": len(rows), "unverified_dates": missing,
            "status": "PARTIAL" if missing or not attributed else "READY"}


def followup(previous, facts, rows):
    watch = (previous.get("funnel") or {}).get("first_board_watch") or {}
    day = facts.get("as_of")
    if watch.get("version") != VERSION or watch.get("review_on") != day or watch.get("as_of") != facts.get("previous_day"):
        return []
    return _review_next_day(watch["picks"], day, rows)


def limit_up_followup(watch, day, rows):
    if watch.get("version") != LIMIT_UP_VERSION:
        return []
    return _review_next_day([r for r in watch["rows"] if r["priority"] == "PRIOR_P5_FIRST_BOARD"], day, rows)


def _review_next_day(picks, day, rows):
    feedback = []
    for pick in picks:
        row = rows.get(pick["symbol"]) or {}
        frame = row.get("frame")
        item = {"symbol": pick["symbol"], "name": pick["name"], "theme": pick.get("theme") or "--",
                "state": "DATA_WAIT", "expired": True, "orders_enabled": False}
        if frame is not None and not frame.empty:
            bar = frame.iloc[-1]
            if str(bar.date) == day and bool(bar.valid) and abs(bar.preclose - pick["first_board_close"]) < .011:
                if bar.high == bar.low:
                    state = "ONE_PRICE_NO_CONFIRMED_FILL"
                elif bar.low < pick["first_board_low"]:
                    state = "STRUCTURE_LOST"
                elif bar.low > pick["first_board_close"]:
                    state = "NO_PULLBACK_WINDOW"
                elif bar.close < max(bar.open, pick["first_board_close"]):
                    state = "WEAK_CLOSE"
                else:
                    state = "HOLDING_LEVELS"
                item.update(state=state, open=float(bar.open), close=float(bar.close))
        feedback.append(item)
    return feedback


def table_rows(watch, feedback):
    labels = {"ONE_PRICE_NO_CONFIRMED_FILL": "一字或单价，无可验证成交",
              "STRUCTURE_LOST": "跌破首板低点", "NO_PULLBACK_WINDOW": "始终高于首板收盘，无回踩窗口",
              "WEAK_CLOSE": "收盘弱于开盘或首板收盘", "HOLDING_LEVELS": "日线关键价保留，仍未核实分钟承接",
              "DATA_WAIT": "缺有效当日价或前收口径不一致"}
    for r in watch["picks"]:
        yield ["首板短线观察", f"{r['name']}（{r['symbol']}）", r["theme"],
               f"前次修复 {r['prior_reclaim']}；同题材封板 {'、'.join(r['peers'])}", "--", "--",
               f"原系统 {r['route_state']}；仅跟踪 {watch['review_on']}",
               f"首板低点 {r['first_board_low']:.2f}，收盘 {r['first_board_close']:.2f}。"
               "不追高开或一字；等回踩承接再转强。失守或冲高回吐则撤观察；不生成交易计划。"]
    for r in feedback:
        yield ["首板次日复核", f"{r['name']}（{r['symbol']}）", r["theme"], labels[r["state"]],
               "--", "--", "本轮观察已到期", "不是成交结果，不滚动补记机会。"]
