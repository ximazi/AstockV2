"""Explainable intraday market, holding and candidate decisions (no orders)."""
from collections import Counter, defaultdict
from statistics import median

from . import intraday_quotes as Q
from .theme_semantics import is_generic_concept
from .tail_observation import LABELS as OBSERVATION_LABELS

INDEX_NAMES = {"000001.SH": "上证指数", "399001.SZ": "深证成指", "399006.SZ": "创业板指", "000300.SH": "沪深300"}
LABELS = {
    "UNKNOWN": "数据不足", "RISK": "风险升高", "RESTRICTED": "收盘环境限制", "REVIEW_ONLY": "允许条件复核",
    "HOT": "共振活跃", "ACTIVE": "一般活跃", "SPLIT": "分化", "WEAK": "偏弱", "DATA_WAIT": "等待数据",
    "PROVISIONAL_REVIEW": "模拟参与复核", "WAIT_CONFIRMATION": "等待承接确认", "WAIT_PRICE": "等待合适位置",
    "NEW_PRICE_DISCOVERY": "新突破线索", "MARKET_WAIT": "市场限制，保留观察", "WATCH": "继续观察",
    "RISK_WATCH": "风险观察", "SELL_REVIEW": "退出复核", "REDUCE_REVIEW": "分批保护复核",
    "WAIT_CLOSE_CONFIRMATION": "等待收盘确认", "HOLD_OBSERVATION": "继续持有观察", "T1_LOCKED": "当日买入不可卖",
    "PRICE_HOT": "价格强势", "PRICE_ACTIVE": "价格活跃", "PRICE_SPLIT": "价格分化", "PRICE_WEAK": "价格偏弱",
}


def label(value):
    return LABELS.get(value, OBSERVATION_LABELS.get(value, value))


def theme_heat(records, membership, scope, price_only=False):
    mapping, meta = membership.get("mapping", {}), membership.get("meta", {})
    current = {r["symbol"]: r for r in records}
    valid = {s: r for s, r in current.items() if r.get("quote") and
             (r.get("price_change_pct") is not None if price_only else r["quote"].get("quality") == "DAILY_SNAPSHOT")}
    changes = {s: r["price_change_pct"] if price_only else (r["quote"]["close"] / r["quote"]["pre_close"] - 1) * 100 for s, r in valid.items()}
    market_median = median(changes.values()) if changes else None
    groups = defaultdict(set)
    for symbol, item in mapping.items():
        if symbol in current:
            for theme in item.get("concepts", []):
                if theme and not is_generic_concept(theme):
                    groups[theme].add(symbol)
    dated = meta.get("status", "MISSING") not in {"MISSING", "INDUSTRY_ONLY"} and not meta.get("status", "").startswith("STALE")
    if price_only:
        dated = (dated and bool(meta.get("as_of")) and bool(valid)
                 and meta["as_of"] < min(r["quote"]["observed_at"][:10] for r in valid.values())
                 and len(valid) / len(current) >= .9)
    rows = []
    for name, members in groups.items():
        usable = sorted(members & valid.keys())
        count, coverage = len(usable), len(usable) / len(members)
        gains = [changes[s] for s in usable]
        middle = median(gains) if gains else None
        up = sum(v > 0 for v in gains) / count if count else None
        above = sum(valid[s]["quote"].get("vwap") is not None and valid[s]["quote"]["close"] >= valid[s]["quote"]["vwap"] for s in usable) / count if count and not price_only else None
        amounts = [valid[s]["quote"].get("amount") or 0 for s in usable]
        amount = sum(amounts) if not price_only else None
        concentration = max(amounts) / amount if amount else None
        relative = middle - market_median if middle is not None and market_median is not None else None
        known = dated and scope == "market" and count >= 5 and coverage >= .8
        state = "DATA_WAIT"
        if known:
            state = ("WEAK" if middle < 0 and up < .4 else
                     "SPLIT" if (concentration is not None and concentration >= .5) or middle <= 0 < max(gains) else
                     "HOT" if middle >= 1 and relative >= .5 and up >= .6 and (price_only or above >= .6) else "ACTIVE")
            if price_only:
                state = "PRICE_" + state
        leaders = sorted(usable, key=lambda s: (-changes[s], s))[:3]
        rows.append({"name": name, "state": state, "members": len(members), "fresh_members": count,
                     "coverage": coverage, "median_pct": middle, "relative_market_pp": relative,
                     "advance_ratio": up, "above_vwap_ratio": above, "amount_yi": amount / 1e8 if amount is not None else None,
                     "largest_amount_share": concentration, "member_symbols": sorted(members),
                     "leaders": [{"symbol": s, "name": valid[s]["name"], "pct_chg": changes[s]} for s in leaders]})
    order = {"HOT": 0, "ACTIVE": 1, "SPLIT": 2, "WEAK": 3, "DATA_WAIT": 4}
    rows.sort(key=lambda r: (order[r["state"].removeprefix("PRICE_")], -(r["relative_market_pp"] or 0), -(r["advance_ratio"] or 0), r["name"]))
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return {"status": "AVAILABLE" if any(r["state"] != "DATA_WAIT" for r in rows) else "DATA_WAIT",
            "source": "参照价核验后的价格强弱，不要求量能确认" if price_only else "实时成分股价格热度，非新闻热搜或资金净流入", "membership": meta, "records": rows,
            "market_median_pct": market_median,
            "threshold_status": "试行价格观察阈值，未验证胜率：至少5个有效成员/80%覆盖；中位涨幅1%、相对市场0.5个百分点、上涨占比60%。" if price_only else "试行观察阈值，未验证胜率：至少5个有效成员/80%覆盖；中位涨幅1%、相对市场0.5个百分点、上涨及VWAP上方占比60%。",
            "notice": "题材可重叠，成交额不可跨题材相加；使用前收盘已知分类，未归类的新题材不会被凭空推断。池内扫描不发布全市场热门题材。"}


def candidate_context(row, own, market):
    q, m = row.get("quote") or {}, row.get("metrics", {})
    valid = [t for t in own if t["state"] != "DATA_WAIT"]
    best = valid[0] if valid else None
    row["live_theme"] = {k: best[k] for k in ("name", "state", "rank", "median_pct", "relative_market_pp", "coverage")} if best else None
    row["theme"] = best["name"] if best else row.get("theme", "")
    row["weak_themes"] = [t["name"] for t in valid if t["state"] == "WEAK"]
    price = q.get("close")
    benchmark = "399006.SZ" if row["symbol"].startswith("sz_30") else "399001.SZ" if row["symbol"].startswith("sz_") else "000001.SH"
    bench_change = market["index_changes_pct"].get(benchmark)
    pct = m.get("pct_chg") if m.get("status") == "PROVISIONAL" else None
    row["relative_index_pp"] = pct - bench_change if pct is not None and bench_change is not None else None
    row["benchmark"] = benchmark
    row["relative_theme_pp"] = pct - best["median_pct"] if pct is not None and best else None
    reasons, blockers = [], []
    if best:
        reasons.append(f"{best['name']}：{label(best['state'])}，成员中位涨幅{best['median_pct']:+.2f}%")
    else:
        blockers.append("缺少可核验的实时题材强度")
    if row["relative_index_pp"] is not None:
        reasons.append(f"相对{INDEX_NAMES[benchmark]} {row['relative_index_pp']:+.2f}个百分点")
    if m.get("status") != "PROVISIONAL":
        blockers.append("价格参照已核验，全天量价数据仍不完整" if m.get("status") == "PRICE_REFERENCE_VERIFIED" else "行情或价格参考基准不完整")
    if row["prior_advice"] != "PRIORITY_REVIEW" or not row["route_ids"]:
        blockers.append("尚未获得正式三路径结构确认")
    lower, upper, invalid = (row.get(k) for k in ("review_lower", "review_upper", "invalid"))
    if not (price and lower and upper and lower <= price <= upper):
        blockers.append("现价不在已确认介入区间")
    if not price or not invalid or price <= invalid:
        blockers.append("失效边界缺失或已被跌破")
    minute = row["minute_context"]
    if minute.get("status") != "COMPLETE" or minute.get("late_pattern") not in {"ABOVE_VWAP", "PULLBACK_ABOVE_VWAP"}:
        blockers.append("完整分钟数据未确认尾盘承接")
    if row["relative_index_pp"] is None or row["relative_index_pp"] < 0:
        blockers.append("未确认强于对应宽基指数")
    if best and all(t["state"] in {"WEAK", "SPLIT"} for t in valid):
        blockers.append("关联题材偏弱或分化")
    if not market["new_review_allowed"]:
        blockers.append("市场或交易时间窗口未放开")
    if row["status"] != "PROVISIONAL_REVIEW":
        blockers.append(f"当前状态：{label(row['status'])}")
    row.update(reasons=reasons, blockers=blockers, paper_review_eligible=not blockers,
               risk_distance_pct=(1 - invalid / price) * 100 if price and invalid else None)


def holding_context(holding, row, market):
    holding["market_state"] = market["state"]
    holding["live_theme"] = row.get("live_theme") if row else None
    holding["context_alerts"] = []
    if market["state"] in {"RISK", "RESTRICTED", "UNKNOWN"}:
        holding["context_alerts"].append(f"市场：{label(market['state'])}，不作新增仓位建议")
    if row and row.get("weak_themes"):
        holding["context_alerts"].append("关联题材偏弱：" + "、".join(row["weak_themes"][:3]))
    weak_price = bool(row and (row.get("quote") or {}).get("vwap") and row["quote"]["close"] < row["quote"]["vwap"])
    weak_theme = bool(row and row.get("live_theme") and row["live_theme"]["state"] in {"WEAK", "SPLIT"})
    action = holding.get("pending_action", holding["status"])
    # Context can tighten an observation, never cancel a latched exit or replace a missing hard floor.
    if action == "HOLD_OBSERVATION" and weak_price and (market["state"] == "RISK" or weak_theme):
        action = "REDUCE_REVIEW"
        holding["reason"] = "跌到日内VWAP下方，且市场风险或关联题材同步走弱，复核分批保护；并非已成交"
        if holding["status"] != "T1_LOCKED":
            holding["status"] = action
        holding["pending_action"] = action
    holding["add_allowed"] = False


def enrich(report, membership):
    heat = theme_heat(report["records"], membership, report.get("scope", "market"))
    report["themes"] = heat
    rows, market = report["records"], report["market"]
    by_symbol = {r["symbol"]: r for r in rows}
    by_member = defaultdict(list)
    for theme in heat["records"]:
        for symbol in theme["member_symbols"]:
            by_member[symbol].append(theme)
    for row in rows:
        candidate_context(row, by_member[row["symbol"]], market)
    for holding in report["holdings"]:
        holding_context(holding, by_symbol.get(holding["symbol"]), market)
    order = {s: i for i, s in enumerate(("PROVISIONAL_REVIEW", "WAIT_CONFIRMATION", "WAIT_PRICE", "NEW_PRICE_DISCOVERY", "MARKET_WAIT", "WATCH", "RISK_WATCH", "DATA_WAIT"))}
    def key(row):
        theme = row["live_theme"] or {}
        return (not row["paper_review_eligible"], order[row["status"]], theme.get("state") != "HOT",
                -(row["relative_index_pp"] or 0), abs(row["metrics"].get("ema20_bias_pct") or 0), row["prior_rank"] or 100000, row["symbol"])
    rows.sort(key=key)
    for rank, row in enumerate(rows, 1):
        row["intraday_rank"] = rank
    held = {h["symbol"] for h in report["holdings"]}
    eligible = [r for r in rows if r["symbol"] not in held and r["paper_review_eligible"]]
    # Overlapping concepts do not earn repeated bonuses. Limit the focus, not the audited universe.
    focus, counts = [], Counter()
    for row in rows:
        if row["symbol"] in held or row["status"] in {"DATA_WAIT", "RISK_WATCH", "WATCH"}:
            continue
        if not row["new_price_breakout"] and not (row["route_ids"] and row["prior_advice"] in {"PRIORITY_REVIEW", "WAIT_CONFIRMATION", "WAIT_PRICE", "EARLY_WATCH", "TREND_WATCH"}):
            continue
        theme = (row["live_theme"] or {}).get("name", "未归类")
        if counts[theme] >= 2:
            continue
        focus.append(row["symbol"])
        counts[theme] += 1
        if len(focus) == 5:
            break
    report["candidates"] = {"focus_symbols": focus, "paper_review_symbols": [r["symbol"] for r in eligible if r["symbol"] in focus][:3],
                            "eligible_count": len(eligible), "automatic_orders": False,
                            "notice": "重点观察不等于买入。模拟参与复核不自动记账；虚拟持仓仍由人工在收盘后以正式收盘价添加。"}
    report["ranking_method"] = "先结构/价格/数据门槛，再比较单一最强关联题材、相对指数强度、EMA偏离；非预期收益或胜率预测。三路径不因盘中热点而改写。"
    if heat["status"] == "DATA_WAIT" and report["summary"]["status"] == "PROVISIONAL":
        report["summary"]["status"] = "PARTIAL"
    report["summary"].update(hot_themes=sum(t["state"] == "HOT" for t in heat["records"]),
                             paper_reviews=len(report["candidates"]["paper_review_symbols"]), focus=len(focus),
                             holding_actions=dict(Counter(h["status"] for h in report["holdings"])))
    return report


def report_lines(report):
    market, heat = report["market"], report.get("themes", {})
    def fmt(value, suffix=""):
        return "--" if value is None else f"{value:.2f}{suffix}"
    lines = ["", "## 1. 行情总体与风险", f"- {label(market['state'])}；新增条件复核：{'开放' if market['new_review_allowed'] else '关闭'}。",
             f"- 行情覆盖 {market['snapshot_coverage']:.1%}；上涨占比 {fmt((market.get('advance_ratio') or 0) * 100, '%') if market.get('advance_ratio') is not None else '--'}。",
             "", "|指数|当前涨跌|距日内高点|行情时间|", "|---|---|---|---|"]
    for row in market.get("indices", []):
        lines.append(f"|{row['name']}|{fmt(row.get('pct_chg'), '%')}|{fmt(row.get('drawdown_pct'), '%')}|{row.get('observed_at') or '缺失'}|")
    lines += ["", "### 实时题材价格热度", heat.get("source", "待计算"), "", "|题材|状态|有效/成员|中位涨幅|强于市场|上涨占比|", "|---|---|---|---|---|---|"]
    for t in heat.get("records", [])[:10]:
        lines.append(f"|{t['name']}|{label(t['state'])}|{t['fresh_members']}/{t['members']}|{fmt(t['median_pct'], '%')}|{fmt(t['relative_market_pp'], 'pp')}|{fmt(t['advance_ratio'] * 100 if t['advance_ratio'] is not None else None, '%')}|")
    lines += ["", heat.get("notice", ""), "", "## 2. 已有持仓决策", "", "|股票|判断|现价|硬底线|收盘保护位|依据与背景|", "|---|---|---|---|---|---|"]
    for h in report["holdings"]:
        lines.append(f"|{h['symbol']} {h['name']}|{label(h['status'])}|{fmt(h.get('latest_price'))}|{fmt(h.get('hard_floor'))}|{fmt(h.get('soft_floor'))}|{h['reason']}；{'；'.join(h.get('context_alerts', []))}|")
    if not report["holdings"]:
        lines.append("|无虚拟持仓|--|--|--|--|--|")
    lines += ["", "## 3. 潜力标的优先级", report.get("candidates", {}).get("notice", ""), "", "|股票|路径/前收盘阶段|实时题材|当前判断|现价/介入区间|解释/尚缺条件|", "|---|---|---|---|---|---|"]
    focus = set(report.get("candidates", {}).get("focus_symbols", []))
    for r in report["records"]:
        if r["symbol"] not in focus:
            continue
        lines.append(f"|{r['symbol']} {r['name']}|{r['route_ids'] or '未分类'} / {r['prior_lifecycle']}|{r['theme']}|{label(r['status'])}|{fmt((r['quote'] or {}).get('close'))} / {fmt(r.get('review_lower'))}~{fmt(r.get('review_upper'))}|{'；'.join(r['reasons'] + r['blockers'])}|")
    if not focus:
        lines.append("|本轮无满足观察条件的标的|--|--|等待|--|不强凑数量|")
    papers = report.get("candidates", {}).get("paper_review_symbols", [])
    lines += ["", "## 本轮总结", f"- 市场：{label(market['state'])}；数据状态：{report['summary']['status']}。",
              f"- 持仓：{len(report['holdings'])}只；重点观察：{len(focus)}只；模拟参与复核：{len(papers)}只。",
              "- 模拟参与复核名单：" + ("、".join(papers) or "无，继续等待"), "- 所有判断均为暂定观察，不是成交指令。"]
    return lines
