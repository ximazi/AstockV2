"""Read-only leader and first-to-second-board observations, never entry signals."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import pandas as pd

from . import intraday_quotes as Q
from .pipeline import _write

VERSION = "tail_observation_v1_20260918"
SOURCE_FIELDS = {
    "limit_list_d": "trade_date,ts_code,name,close,pct_chg,turnover_ratio,first_time,last_time,open_times,limit_times,fd_amount",
    "stk_limit": "trade_date,ts_code,pre_close,up_limit,down_limit",
}
LABELS = {
    "DAILY_LEADER": "前收盘信息领头候选", "PRICE_FRONT": "盘中涨幅前排，龙头待核",
    "FIRST_TOUCH": "今日首板触板候选", "FIRST_OPEN": "今日首板触板后回落",
    "TOUCH_UNCLASSIFIED": "触板，首板身份待核", "SECOND_WAIT": "二板晋级待确认",
    "SECOND_PRICE": "当前处于二板价，封板待核", "SECOND_OPEN": "触及二板后回落",
    "DATA_WAIT": "等待数据", "AVAILABLE": "资料可用", "EMPTY": "当日无记录",
    "PARTIAL": "资料不完整", "UNAVAILABLE": "资料不可用", "STALE": "历史观察，已过期",
    "ARCHIVED_FIRST": "历史首板档案，非今日晋级判断",
}
NOTICE = "独立影子观察，不开放买入或打板成交。当前处于涨停价不等于封板成功；收盘终态和排队成交均需另行核实。"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}


def source_usable(source, day, at):
    if source.get("as_of") != day or source.get("status") not in {"AVAILABLE", "EMPTY", "PARTIAL"}:
        return False
    try:
        return Q.stamp(source["retrieved_at"]) <= Q.stamp(at)
    except (KeyError, ValueError):
        return False


def fetch_source(client, api, day, root, at=None, limit_type="U"):
    """Cache dated inputs before acquiring time-sensitive quotes; no current daily board list.

    Each limit type has its own cache file, so U, D and Z never overwrite one another.
    """
    fixed_at = at is not None
    at = Q.stamp(at or Q.now_china())
    day = pd.Timestamp(day).date().isoformat()
    if day > at.date().isoformat() or (api == "limit_list_d" and day == at.date().isoformat() and at.hour < 15):
        raise ValueError("首板日档只能使用已收盘日期，不能将当日日榜终值当实时输入")
    suffix = f"_{limit_type}" if api == "limit_list_d" and limit_type != "U" else ""
    path = Path(root) / "reference" / f"{api}{suffix}_{day.replace('-', '')}.json"
    cached = read_json(path)
    if cached.get("version") == VERSION and cached.get("as_of") == day:
        age = (at - Q.stamp(cached["retrieved_at"])).total_seconds()
        if age >= 0 and (age < 300 or (cached.get("status") == "AVAILABLE" and day < at.date().isoformat())
                         or (api == "stk_limit" and cached.get("status") == "AVAILABLE")):
            return cached
    result = {"api": api, "as_of": day, "version": VERSION, "retrieved_at": at.isoformat(),
              "status": "UNAVAILABLE", "records": [], "issues": [], "limit_type": limit_type if api == "limit_list_d" else ""}
    key_field = ("limit_times" if limit_type == "U" else "open_times") if api == "limit_list_d" else "up_limit"
    try:
        params = {"trade_date": day.replace("-", "")}
        if api == "limit_list_d":
            params["limit_type"] = limit_type
        frame = client.query(api, fields=SOURCE_FIELDS[api], **params)
        required = {"trade_date", "ts_code", key_field}
        if not required.issubset(frame.columns):
            raise ValueError("返回缺少日期、代码或关键字段；不按零候选处理")
        if (not frame.trade_date.astype(str).eq(day.replace("-", "")).all()
                or frame.ts_code.duplicated().any()):
            raise ValueError("返回日期不符或股票重复，拒绝误用旧日记录")
        values = frame[key_field].map(Q.number)
        if api == "limit_list_d" and limit_type != "U":
            if not frame.empty and (values.isna().any() or values.lt(0).any() or values.mod(1).ne(0).any()):
                raise ValueError("开板次数非法")
        elif values.isna().any() or values.le(0).any() or (api == "limit_list_d" and values.mod(1).ne(0).any()):
            raise ValueError("连板数或涨停价格非法")
        if not frame.ts_code.astype(str).str.fullmatch(r"\d{6}\.(SH|SZ|BJ)").all():
            raise ValueError("股票代码格式非法")
        result["records"] = json.loads(frame.to_json(orient="records"))
        cap = 2500 if api == "limit_list_d" else 5800
        result["status"] = "EMPTY" if frame.empty else "PARTIAL" if len(frame) >= cap else "AVAILABLE"
        if result["status"] == "PARTIAL":
            result["issues"].append("返回达到接口条数上限；母池可能截断，不能按缺席判定非涨停")
    except (Q.ApiError, ValueError) as exc:
        result["issues"].append(str(exc))
    result["retrieved_at"] = (at if fixed_at else Q.stamp(Q.now_china())).isoformat()
    _write(path, result)
    return result


def leader_source(daily_root, day, at):
    folder = Path(daily_root) / day.replace("-", "")
    status = read_json(folder / "mainwave_run_status.json")
    result = {"as_of": day, "status": "UNAVAILABLE", "records": []}
    if (not (folder / "_SUCCESS").exists() or status.get("as_of") != day or status.get("status") != "SUCCESS"
            or not status.get("finished_at") or Q.stamp(status["finished_at"]) > Q.stamp(at)):
        return result
    try:
        frame = pd.read_csv(folder / "leader_state_daily.csv").fillna("")
        if not {"as_of", "leader", "theme_label", "leader_state"}.issubset(frame.columns) or not frame.as_of.eq(day).all():
            return result
        result.update(status="AVAILABLE", records=json.loads(frame.to_json(orient="records")), retrieved_at=status["finished_at"])
    except (OSError, ValueError):
        pass
    return result


def _quote(row, at):
    q = row.get("quote") or {}
    try:
        age = (Q.stamp(at) - Q.stamp(q["observed_at"])).total_seconds()
        if (q.get("quality") == "DAILY_SNAPSHOT" and 0 <= age <= 120
                and Q.stamp(q["observed_at"]).date() == Q.stamp(at).date()):
            return q
    except (KeyError, ValueError):
        pass
    return {}


def _common(row, at):
    q, m = _quote(row, at), row.get("metrics", {})
    return {"symbol": row["symbol"], "name": row.get("name", ""), "route_ids": row.get("route_ids", ""),
            "lifecycle": row.get("prior_lifecycle", ""), "lifecycle_as_of": row.get("lifecycle_as_of", ""),
            "theme": row.get("theme", ""), "price": q.get("close"), "quote_at": q.get("observed_at"),
            "relative_index_pp": row.get("relative_index_pp"), "ema20_bias_pct": m.get("ema20_bias_pct"),
            "turnover_pct_estimate": m.get("turnover_pct_estimate"), "orders_enabled": False,
            "risk_watch": row.get("status") == "RISK_WATCH", "evidence": [], "missing": []}


def build(report, sources):
    at, day = report["observed_at"], report["as_of"]
    previous = sources.get("previous_day", "")
    current = {r["symbol"]: r for r in report["records"]}
    prior, limits, leaders = (sources.get(k, {}) for k in ("boards", "limits", "leaders"))
    prior_ok = previous < day and source_usable(prior, previous, at)
    prior_complete = prior_ok and prior.get("status") in {"AVAILABLE", "EMPTY"}
    up_prices = {Q.symbol_of(r["ts_code"]): Q.number(r["up_limit"]) for r in limits.get("records", [])} if source_usable(limits, day, at) else {}
    # Positive records from a partial mother pool are usable; absence from it proves nothing.
    yesterday = {Q.symbol_of(r["ts_code"]): r for r in prior.get("records", [])} if prior_ok else {}
    firsts = {s: r for s, r in yesterday.items() if Q.number(r.get("limit_times")) == 1}
    boards = []
    for symbol in sorted(set(current) | set(firsts)):
        row = current.get(symbol, {"symbol": symbol})
        q, up = _quote(row, at), up_prices.get(symbol)
        price_ok = bool(q and up and up > q["pre_close"] and q["high"] <= up + .005)
        touched = price_ok and q["high"] >= up - .005
        at_limit = price_ok and abs(q["close"] - up) < .005
        if symbol not in firsts and (not touched or symbol in yesterday):
            continue
        item = _common(row, at)
        item.update(strategy="FIRST_TO_SECOND", first_board_date=previous if symbol in firsts else None,
                    candidate_date=day, up_limit=up, prior_board=firsts.get(symbol), final_confirmed=False)
        if symbol in firsts:
            item["name"] = item["name"] or firsts[symbol].get("name", "")
            state = "DATA_WAIT" if not price_ok else "SECOND_PRICE" if at_limit else "SECOND_OPEN" if touched else "SECOND_WAIT"
            item["group"] = "TODAY_PROMOTION"
            item["evidence"].append(f"{previous}日榜连板数为1；今日收盘晋级结果尚未确认")
        else:
            state = "TOUCH_UNCLASSIFIED" if not prior_complete else "FIRST_TOUCH" if at_limit else "FIRST_OPEN"
            item["group"] = "NEXT_SESSION_PREP"
            item["evidence"].append("今日累计最高价触及精确涨停价；不据此推断触板时间或封单")
        item["state"] = state
        if not price_ok:
            item["missing"].append("新鲜累计快照或当日精确涨停价缺失/不一致")
        if not prior_complete:
            item["missing"].append("昨日涨停母池不完整，首板资格及晋级分母不可完全核实")
        if item["risk_watch"]:
            item["missing"].append("当前价格/结构有风险警报，不因触板而取消")
        item["missing"].append("未验证封单、秒级开封与排队成交；仅作观察")
        boards.append(item)
    order = {"SECOND_PRICE": 0, "SECOND_WAIT": 1, "SECOND_OPEN": 2, "FIRST_TOUCH": 0, "FIRST_OPEN": 2, "TOUCH_UNCLASSIFIED": 3, "DATA_WAIT": 4}
    boards.sort(key=lambda r: (r["risk_watch"], order[r["state"]], -(r["relative_index_pp"] or 0), r["symbol"]))

    seeds = defaultdict(list)
    if previous < day and source_usable(leaders, previous, at):
        for item in leaders.get("records", []):
            if re.fullmatch(r"(sh|sz)_\d{6}", item.get("leader", "")):
                seeds[item["leader"]].append(item)
    front = defaultdict(list)
    for theme in report.get("themes", {}).get("records", []):
        if theme["state"] in {"HOT", "ACTIVE"}:
            for item in theme["leaders"]:
                front[item["symbol"]].append(theme["name"])
    roles = []
    for symbol in sorted(set(seeds) | set(front)):
        row = current.get(symbol, {"symbol": symbol})
        item = _common(row, at)
        item.update(strategy="LEADER_RECLAIM", state="DAILY_LEADER" if symbol in seeds else "PRICE_FRONT",
                    daily_evidence=seeds[symbol], intraday_front_themes=front[symbol],
                    leadership_confirmed=False, minute_pattern=row.get("minute_context", {}).get("late_pattern", "WAIT_MINUTES"))
        for seed in seeds[symbol]:
            item["evidence"].append(f"{previous} {seed['theme_label']}：{seed['leader_state']}；日线事件先后关联，非分钟带动证明")
        if front[symbol]:
            item["evidence"].append("实时题材涨幅前排：" + "、".join(front[symbol]))
        q = _quote(row, at)
        if q:
            item["evidence"].append("现价在日内VWAP上方" if q.get("vwap") and q["close"] >= q["vwap"] else "未确认现价守住日内VWAP")
        else:
            item["missing"].append("缺少新鲜累计快照；只保留前收盘角色")
        if item["risk_watch"]:
            item["missing"].append("价格/结构风险观察，领头身份不能抵消")
        item["missing"] += ["分钟带动与共同市场因素尚未区分", "分歧检验、回踩再转强顺序未确认", "人气与盘口持续性未核实"]
        roles.append(item)
    roles.sort(key=lambda r: (r["risk_watch"], r["price"] is None, r["state"] != "DAILY_LEADER", -(r["relative_index_pp"] or 0), r["symbol"]))
    promotion = [r for r in boards if r["group"] == "TODAY_PROMOTION"]
    return {"version": VERSION, "as_of": day, "previous_day": previous, "observed_at": at,
            "mode": "SHADOW_ONLY", "notice": NOTICE, "orders_enabled": False,
            "source_status": {k: v.get("status", "UNAVAILABLE") if source_usable(v, previous if k != "limits" else day, at) else "UNAVAILABLE"
                              for k, v in (("boards", prior), ("limits", limits), ("leaders", leaders))},
            "issues": [issue for v in (prior, limits) for issue in v.get("issues", [])],
            "leaders": roles, "boards": boards,
            "summary": {"leader_candidates": len(roles), "prior_first_boards": len(firsts) if prior_complete else None,
                        "known_first_boards": len(firsts), "today_promotion": len(promotion),
                        "promotion_states": dict(Counter(r["state"] for r in promotion)),
                        "next_session_prep": sum(r["group"] == "NEXT_SESSION_PREP" for r in boards)},
            "ranking_method": "分组观察排序：风险提示、晋级观察状态、相对指数；非收益或龙头概率排名。"}


def offline(root, daily_root, day, at):
    """Show dated preparation without inventing a current intraday report or fetching data."""
    paths = sorted((Path(root) / "reference").glob("limit_list_d_????????.json"), reverse=True)
    source = next((read_json(p) for p in paths if p.stem[-8:] < day.replace("-", "")), {})
    previous = source.get("as_of", "")
    sources = {"previous_day": previous, "boards": source,
               "leaders": leader_source(daily_root, previous, at) if previous else {}}
    baseline = read_json(Path(daily_root) / previous.replace('-', '') / 'analysis_universe.json') if previous else {}
    rows = []
    if baseline.get('as_of') == previous and sources['leaders'].get('status') == 'AVAILABLE':
        rows = [{"symbol": r['symbol'], "name": r.get('name', ''), "route_ids": r.get('route_ids', ''),
                 "theme": r.get('belonging_theme', ''), "prior_lifecycle": r.get('stock_lifecycle_label', ''),
                 "lifecycle_as_of": previous} for r in baseline.get('records', [])]
    result = build({"as_of": day, "observed_at": Q.stamp(at).isoformat(), "records": rows}, sources)
    result["mode"] = "REFERENCE_ONLY"
    for row in result["boards"]:
        row.update(state="ARCHIVED_FIRST", group="ARCHIVE", candidate_date=None)
        row["evidence"] = [f"{previous}已确认首板；尚无所选日期的实时晋级判断"]
    result["summary"].update(today_promotion=0, promotion_states={})
    return result


def report_lines(observation):
    lines = ["", "## 4. 龙头与首板晋级观察（不参与原买入排序）", observation["notice"],
             f"- 前收盘日：{observation['previous_day']}；来源状态：{observation['source_status']}。",
             "", "|股票|分组/角色|状态|依据|尚缺证据|", "|---|---|---|---|---|"]
    for row in observation["leaders"][:10] + observation["boards"]:
        lines.append(f"|{row['symbol']} {row['name']}|{row['strategy']}|{LABELS.get(row['state'], row['state'])}|{'；'.join(row['evidence'])}|{'；'.join(row['missing'])}|")
    lines += ["", f"- 观察汇总：{observation['summary']}；自动交易：关闭。"]
    return lines
