"""Price-only observations alongside, never in place of, full entry confirmation."""
from collections import Counter
from statistics import median

from . import intraday_quotes as Q, intraday_decisions as ID, potential_pool as P, tail_observation as TO

VERSION = "intraday_price_v1_20260921"
NOTICE = "价格级观察：当日参照价已核验；不证明资金流入、放量承接、回踩完成或可成交，不开放买入。"


def reference_prices(sources, day, at):
    source = sources.get("limits", {})
    if not TO.source_usable(source, day, at):
        return {}
    rows = source.get("records", [])
    counts = Counter(r.get("ts_code") for r in rows)
    return {r["ts_code"]: r for r in rows if counts[r.get("ts_code")] == 1
            and str(r.get("trade_date")) == day.replace("-", "")
            and (Q.number(r.get("pre_close")) or 0) > 0}


def build(report, references, sources, positions, prior_positions, membership):
    rows = report["records"]
    for row in rows:
        q = row.get("quote") or {}
        ref = references.get(Q.ts_code(row["symbol"]), {})
        preclose = Q.number(q.get("pre_close")) or Q.number(ref.get("pre_close"))
        row["price_change_pct"] = (q["close"] / preclose - 1) * 100 if q and preclose and preclose > 0 else None
    valid = {r["symbol"]: r for r in rows if r["price_change_pct"] is not None}
    gains = [r["price_change_pct"] for r in valid.values()]
    coverage = len(valid) / len(rows) if rows else 0
    up = sum(v > 0 for v in gains)
    middle = median(gains) if gains else None
    broad = report.get("scope") == "market" and coverage >= .9
    market = {"status": "PRICE_OBSERVATION" if broad else "SAMPLE_ONLY" if gains else "DATA_WAIT",
              "covered": len(valid), "requested": len(rows), "coverage": coverage,
              "up": up, "down": sum(v < 0 for v in gains), "flat": sum(v == 0 for v in gains),
              "advance_ratio": up / len(gains) if gains else None, "median_pct": middle,
              "reading": ("多数上涨" if up / len(gains) > .5 else "多数下跌" if sum(v < 0 for v in gains) / len(gains) > .5 else "涨跌分化") if broad else "仅样本观察，不代表全市场",
              "prior_context": report["market"].get("previous_close_context", {}).get("context"),
              "indices_complete": all(r.get("observed_at") for r in report["market"].get("indices", [])) and bool(report["market"].get("indices")),
              "new_review_allowed": False}
    heat = ID.theme_heat(rows, membership or {}, report.get("scope", "market"), price_only=True)
    by_theme = {}
    for theme in heat["records"]:
        if theme["state"] != "DATA_WAIT":
            for symbol in theme["member_symbols"]:
                by_theme.setdefault(symbol, theme)

    holdings = []
    for position in positions:
        s = position["symbol"]
        row = valid.get(s, {})
        q, m = row.get("quote") or {}, row.get("metrics", {})
        prior = prior_positions.get(position["id"], {})
        comparable = m.get("status") in {"PROVISIONAL", "PRICE_REFERENCE_VERIFIED"}
        scale = m.get("reference_scale", 1)
        hard = Q.number(prior.get("hard_floor"))
        soft = Q.number(prior.get("soft_floor")) or hard
        hard, soft = (hard * scale if hard and comparable else None), (soft * scale if soft and comparable else None)
        price = q.get("close")
        reading = ("原退出警报仍待处理" if prior.get("status") in {"SELL", "REDUCE"} else
                   "现价跌破硬底线，优先人工复核" if price and hard and price <= hard else
                   "现价跌破收盘保护位，等待收盘确认" if price and soft and price <= soft else
                   "现价尚在边界上方；不排除盘中曾跌破" if price and hard else "价格或风险边界待核验")
        holdings.append({"symbol": s, "name": position.get("name", ""), "price": price,
                         "pct_chg": row.get("price_change_pct"), "hard_floor": hard, "soft_floor": soft,
                         "hard_distance_pct": (price / hard - 1) * 100 if price and hard else None,
                         "ema20_bias_pct": m.get("ema20_bias_pct") if comparable else None,
                         "reading": reading, "t1_locked": position["entry_date"] >= report["as_of"],
                         "orders_enabled": False})

    held = {p["symbol"] for p in positions}
    candidates = []
    for s, row in valid.items():
        m, seed, q = row["metrics"], row.get("preparation", {}), row["quote"]
        if s in held or m.get("status") not in {"PROVISIONAL", "PRICE_REFERENCE_VERIFIED"}:
            continue
        if seed.get("status") != "READY" or seed.get("as_of") != m["previous_date"]:
            continue
        if row.get("prior_advice") in {"DATA_WAIT", "HISTORICAL_ONLY"}:
            continue
        scale, price = m["reference_scale"], q["close"]
        atr = (Q.number(seed.get("atr20")) or 0) * scale
        ema10 = m.get("ema10")
        distance = (price - ema10) / atr if atr > 0 and ema10 else None
        lower, upper, invalid = (Q.number(row.get(k)) for k in ("review_lower", "review_upper", "invalid"))
        within = bool(lower and upper and lower <= price <= upper)
        risk = (row.get("risk_flags") or row.get("status") == "RISK_WATCH" or seed.get("pressure")
                or seed.get("risk_flags") or (invalid and price <= invalid)
                or (upper and price > upper) or (seed.get("low20") and price < seed["low20"] * scale))
        # Reuse the existing entry-distance veto; no new return-optimised ranking weights.
        if (risk or not seed.get("liquid") or not seed.get("ema20_rising") or price < m["ema20"]
                or distance is None or distance < 0 or distance > P.RULES["ignition_distance_atr"]):
            continue
        if not (row.get("route_ids") or seed.get("evidence") or m.get("above_prior20_high")):
            continue
        theme = by_theme.get(s, {})
        candidates.append({"symbol": s, "name": row["name"], "theme": theme.get("name", row.get("theme", "")),
                           "route_ids": row.get("route_ids", ""), "lifecycle": row.get("prior_lifecycle", ""),
                           "price": price, "pct_chg": row["price_change_pct"], "ema10_distance_atr": distance,
                           "ema20_bias_pct": m["ema20_bias_pct"], "review_lower": lower, "review_upper": upper,
                           "invalid": invalid, "within_range": within, "breakout": m.get("above_prior20_high", False),
                           "theme_relative_pp": theme.get("relative_market_pp"),
                           "reading": "进入既定观察区间，承接待核" if within else "突破前20日高点，量能待核" if m.get("above_prior20_high") else "均线上方，等待买点确认",
                           "prior_rank": row.get("prior_rank"), "paper_review_eligible": False, "orders_enabled": False})
    candidates.sort(key=lambda r: (not r["within_range"], not r["breakout"], r["ema10_distance_atr"],
                                   -(r["theme_relative_pp"] or 0), r["prior_rank"] or 100000, r["symbol"]))
    focus, counts = [], Counter()
    for row in candidates:
        if counts[row["theme"]] < 2:
            focus.append(row)
            counts[row["theme"]] += 1
        if len(focus) == 5:
            break

    boards = sources.get("boards", {})
    previous = sources.get("previous_day", "")
    board_ok = bool(previous and previous < report["as_of"] and TO.source_usable(boards, previous, report["observed_at"]))
    prior_boards = {r["ts_code"]: r for r in boards.get("records", [])} if board_ok else {}
    board_rows = []
    for code, item in prior_boards.items():
        row = valid.get(Q.symbol_of(code), {})
        q = row.get("quote") or {}
        limit = Q.number(references.get(code, {}).get("up_limit"))
        board_rows.append({"symbol": Q.symbol_of(code), "name": item.get("name", ""), "price": q.get("close"),
                           "pct_chg": row.get("price_change_pct"), "up_limit": limit,
                           "at_limit": bool(q and limit and abs(q["close"] - limit) < .005),
                           "first_board": Q.number(item.get("limit_times")) == 1})
    board_gains = [r["pct_chg"] for r in board_rows if r["pct_chg"] is not None]
    cohort = {"status": "PRICE_OBSERVATION" if board_ok else "DATA_WAIT", "previous_day": previous,
              "mother_pool_complete": board_ok and boards.get("status") in {"AVAILABLE", "EMPTY"},
              "total": len(board_rows) if board_ok else None, "covered": len(board_gains),
              "mean_pct": sum(board_gains) / len(board_gains) if board_gains else None,
              "median_pct": median(board_gains) if board_gains else None,
              "positive_ratio": sum(v > 0 for v in board_gains) / len(board_gains) if board_gains else None,
              "at_limit": sum(r["at_limit"] for r in board_rows), "records": board_rows}
    return {"version": VERSION, "mode": "PRICE_OBSERVATION", "observed_at": report["observed_at"], "notice": NOTICE,
            "market": market, "themes": heat, "holdings": holdings, "candidates": focus, "candidate_count": len(candidates),
            "previous_boards": cohort, "orders_enabled": False,
            "missing": ["日内完整OHLC、累计量额与换手", "下影线、缩量回踩、回封及承接顺序", "实时指数缺失时不能做指数风险确认"],
            "ranking_method": "观察区间优先、突破线索、EMA10的ATR距离、题材价格相对强度、原排名；排除已有风险和过度偏离。未作收益优化。"}


def report_lines(price):
    def fmt(value, suffix=""):
        return "--" if value is None else f"{value:.2f}{suffix}"
    m, cohort = price["market"], price["previous_boards"]
    lines = ["", "## 价格级观察（不替代完整买卖确认）", price["notice"],
             f"- 价格参照覆盖 {m['covered']}/{m['requested']}；{m['reading']}；涨/跌/平 {m['up']}/{m['down']}/{m['flat']}；中位涨幅 {fmt(m['median_pct'], '%')}。",
             f"- 前收盘环境 {m['prior_context'] or '未知'}；实时指数 {'已取得' if m['indices_complete'] else '缺失，不能用个股宽度冒充指数'}。",
             f"- 昨日涨停母池：覆盖 {cohort['covered']}/{cohort['total'] if cohort['total'] is not None else '--'}；均值 {fmt(cohort['mean_pct'], '%')}；中位数 {fmt(cohort['median_pct'], '%')}；现价处于涨停价 {cohort['at_limit']}只，非封板/晋级确认。",
             "", "|题材|价格状态|有效/成员|中位涨幅|相对市场|", "|---|---|---|---|---|"]
    if price.get("mode") == "STALE":
        lines.insert(3, "本节是已过期的历史快照，不代表当前行情。")
    for t in price["themes"]["records"][:10]:
        lines.append(f"|{t['name']}|{ID.label(t['state'])}|{t['fresh_members']}/{t['members']}|{fmt(t['median_pct'], '%')}|{fmt(t['relative_market_pp'], 'pp')}|")
    lines += ["", "|持仓|现价|涨跌幅|硬底线/收盘保护|价格观察|", "|---|---|---|---|---|"]
    for h in price["holdings"]:
        lines.append(f"|{h['name']} {h['symbol']}|{fmt(h['price'])}|{fmt(h['pct_chg'], '%')}|{fmt(h['hard_floor'])}/{fmt(h['soft_floor'])}|{h['reading']}{'；T+1当日不可卖' if h['t1_locked'] else ''}|")
    lines += ["", "|价格观察候选|题材/路径|现价|EMA20偏离|观察区间|依据|", "|---|---|---|---|---|---|"]
    for r in price["candidates"]:
        lines.append(f"|{r['name']} {r['symbol']}|{r['theme']} / {r['route_ids']}|{fmt(r['price'])}|{fmt(r['ema20_bias_pct'], '%')}|{fmt(r['review_lower'])}~{fmt(r['review_upper'])}|{r['reading']}|")
    lines += ["", price["ranking_method"], "待确认：" + "；".join(price["missing"])]
    return lines
