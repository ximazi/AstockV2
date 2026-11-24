"""Read-only EOD bridge: frozen preparation -> attributed limit-up -> next-session response."""
import csv
import json
from collections import defaultdict
from pathlib import Path

from .eod_review import read_json, file_sha
from .review_clock import instant, visible
from . import hotspot_funnel as HF


VERSION = "hotspot_relay_shadow_v2_20261118"
FOLLOW_SESSIONS = 8  # Observation only, aligns with the existing R3 anchor window.
# Only explicit synonyms, not broad overlapping tags such as "芯片概念".
ALIASES = HF.ALIASES


def _frozen(folder, day, cutoff=None):
    for path in sorted((folder / "fupan").glob("v[0-9]*/hotspot_core.json"), reverse=True):
        meta = read_json(path.with_name("report.json")).get("meta", {})
        target = meta.get("target_date")
        boundary = instant(f"{target}T09:15:00+08:00") if target and target > day else None
        rendered = instant(meta.get("rendered_at"))
        if (meta.get("forward_frozen") is True and meta.get("trade_date") == day and rendered and boundary
                and day <= rendered.date().isoformat() and rendered < boundary
                and (cutoff is None or (instant(cutoff) and rendered <= instant(cutoff)))):
            core = read_json(path)
            if core.get("as_of") == day:
                return core, path
    return {}, None


def _carriers(folder, day, targets=(), cutoff=None):
    path = folder / "carrier_qualification.csv"
    if not visible(path, day, cutoff):
        return None, {}
    with path.open(encoding="utf-8-sig", newline="") as src:
        pool, stages = {}, {}
        dated = False
        for r in csv.DictReader(src):
            dated |= r.get("as_of") == day
            if r.get("as_of") != day or r.get("data_fresh") != "True":
                continue
            if r.get("hard_gate_pass") == "True" and r.get("carrier_qualification") == "QUALIFIED_LURK":
                pool[r["symbol"]] = r["name"]
            if r["symbol"] in targets:
                stages[r["symbol"]] = r.get("carrier_qualification")
        return (pool if dated else None), stages


def _groups(rows, members, day, cutoff=None):
    groups = defaultdict(set)
    for r in rows:
        a = r.get("attribution") or {}
        if (r["symbol"] in members and a.get("status") == "CORROBORATED" and a.get("theme")
                and (a.get("trade_date") == day or (a.get("known_at") or "")[:10] == day)
                and (not cutoff or (instant(a.get("known_at")) and instant(a["known_at"]) <= instant(cutoff)))):
            groups[ALIASES.get(a["theme"], a["theme"])].add(r["symbol"])
    return groups


def _followup(folder, day, prior, previous, yesterday, boards, cutoff=None):
    old = previous.get("relay", {}).get("followup", {})
    seeds = {s: r["last_board"] for s, r in old.items() if r.get("sessions_since_board", 0) < FOLLOW_SESSIONS
             and r.get("tracking_state") != "STRUCTURE_FAILED"}
    seeds.update(dict.fromkeys(yesterday, prior))
    seeds.update(dict.fromkeys(boards, day))
    path = folder / "route_candidates.csv"
    found = {s: {"last_board": last, "routes": [], "tracking_state": "NO_FRESH_ROUTE",
                 "sessions_since_board": 0 if last == day else 1 if last == prior else old[s].get("sessions_since_board", 0) + 1}
             for s, last in seeds.items()}
    if not visible(path, day, cutoff):
        for item in found.values():
            item["tracking_state"] = "DATA_WAIT"
        return found, sorted(seeds), "DATA_WAIT"
    with path.open(encoding="utf-8-sig", newline="") as src:
        for r in csv.DictReader(src):
            s = r.get("symbol")
            if s not in seeds or r.get("as_of") != day or r.get("data_status") != "FRESH":
                continue
            item = found[s]
            item["tracking_state"] = "ROUTE_TRACKED"
            item["routes"].append({"route": r["route_id"], "state": r["route_status"],
                                   "reason": r["route_status_label"], "structure": r["breakout_status"]})
    for item in found.values():
        if item["routes"] and all(r["structure"] == "FAILED" for r in item["routes"]):
            item["tracking_state"] = "STRUCTURE_FAILED"
    return found, sorted(s for s, item in found.items() if not item["routes"]), "DATED_ROUTE_TABLE"


def build(day, root, facts, core, admission, cutoff=None):
    root = Path(root)
    prior = facts.get("previous_day") or ""
    folder = root / prior.replace("-", "")
    prior_cutoff = day + "T09:14:59.999999+08:00"
    previous, path = _frozen(folder, prior, prior_cutoff) if prior and prior < day else ({}, None)
    pool = facts.get("yesterday_pool", {})
    comparable = bool(previous) and pool.get("basis") == "FROZEN_PREVIOUS_REVIEW"
    old = _groups(previous.get("rows", []), set(pool.get("members", [])), prior, prior_cutoff if cutoff else None) if comparable else {}
    current_u = set(facts["boards"]["U"])
    now = _groups(core["rows"], current_u, day, cutoff)
    perf = {r["symbol"]: r["close_vs_pre"] for r in pool.get("performance", [])
            if r.get("status") == "OK" and r.get("close_vs_pre") is not None}
    prior_watch = (previous.get("relay") or {}).get("qualified_watch")
    archived_watch, stages = _carriers(folder, prior, current_u, prior_cutoff) if prior and prior < day else (None, {})
    if prior_watch is None:
        prior_watch = archived_watch
    watch, _ = _carriers(root / day.replace("-", ""), day, cutoff=cutoff)
    stocks = {r["symbol"]: r for r in core["rows"]}
    eligible = {r["symbol"]: r.get("eligible") for r in admission.get("rows", [])}
    names = {r["symbol"]: r["name"] for r in previous.get("rows", []) + core["rows"]}
    chains = []
    for theme in sorted(now.keys() | old.keys()):
        a, b = old.get(theme, set()), now.get(theme, set())
        outcomes = {s: perf[s] for s in sorted(a) if s in perf}
        up = sum(v > 0 for v in outcomes.values())
        down = sum(v < 0 for v in outcomes.values())
        state = "同日多股点火" if len(b) >= 2 else "单股线索" if b else "暂无当日归因"
        if a:
            if len(outcomes) != len(a):
                state = "旧核心反馈不完整"
            elif down > len(a) / 2:
                state = "新旧分歧" if b else "旧核心负反馈"
            elif b - set(pool["members"]) and len(b) >= 2 and up > len(a) / 2:
                state = "接力扩散观察"
            elif b and b <= a:
                state = "仅原班封板"
            else:
                state = "延续待证"
        chains.append({"theme": theme, "state": state, "boards": sorted(b),
                       "new_board_members": sorted(b - set(pool["members"])) if comparable else None,
                       "previous_close_change_pct": outcomes, "previous_count": len(a) if comparable else None,
                       "positive": up, "negative": down})
    chains.sort(key=lambda r: (-len(r["boards"]), r["theme"]))
    hits = [{
        "symbol": s, "name": names.get(s, s), "prior_qualified": s in prior_watch if prior_watch is not None else None,
        "prior_carrier_stage": stages.get(s), "entry_review_eligible": eligible.get(s),
        "lifecycle": stocks.get(s, {}).get("lifecycle"), "ema20_bias_pct": stocks.get(s, {}).get("ema20_bias_pct"),
        "engine_state": stocks.get(s, {}).get("engine_state"),
        "attribution": stocks.get(s, {}).get("attribution") or None,
    } for s in sorted(current_u)]
    tracking, not_seen, tracking_basis = _followup(root / day.replace("-", ""), day, prior, previous,
        pool.get("members", []) if pool.get("basis") == "FROZEN_PREVIOUS_REVIEW" else [], current_u, cutoff)
    return {"version": VERSION, "as_of": day, "previous_day": prior, "prior_source": str(path) if path else None,
            "prior_themes_comparable": comparable, "market_allowed": admission.get("market_allowed"),
            "eligible_count": admission.get("eligible_count"), "orders_enabled": False, "names": names,
            "qualified_watch": watch, "prior_watch_count": len(prior_watch) if prior_watch is not None else None,
            "watch_hits_today": None if prior_watch is None else sorted(current_u.intersection(prior_watch)),
            "attributed": sum(len(v) for v in now.values()), "limit_count": len(current_u), "hits": hits, "chains": chains,
            "followup": tracking, "not_in_fresh_routes": not_seen, "followup_basis": tracking_basis,
            "prior_role_review": HF.forward_check(previous, facts)}


def lines(relay):
    names = relay["names"]
    for r in relay["chains"]:
        old = r["previous_count"]
        feedback = "旧池不可比" if old is None else f"昨日已核实{old}只，收涨{r['positive']}、收跌{r['negative']}"
        yield [r["theme"], r["state"], "、".join(names.get(s, s) for s in r["boards"]) or "无已核实封板", feedback]


def replay(day, root):
    folder = root / day.replace("-", "")
    core, source = _frozen(folder, day)
    if not source:
        raise ValueError(f"{day}: 没有开盘前发布的冻结复盘")
    frozen = read_json(source.with_name("report.json"))
    cutoff = frozen["meta"]["rendered_at"]
    fact_ref = next(r for r in frozen["sources"] if r["id"] == "M:" + day)
    facts_path = Path(fact_ref["source_path"])
    if not fact_ref.get("snapshot_sha256") or file_sha(facts_path) != fact_ref["snapshot_sha256"]:
        raise ValueError(f"{day}: 冻结事实包校验失败")
    facts = read_json(facts_path)
    if facts.get("as_of") != day:
        raise ValueError(f"{day}: 事实日期不符")
    admission = read_json(source.with_name("entry_review.json"))
    relay = build(day, root, facts, core, admission, cutoff)
    # A frozen report's explicit reference bounds availability of its legacy version.
    references = {r["attribution"]["source"] for r in core["rows"] if (r.get("attribution") or {}).get("source")}
    attribution = {}
    if len(references) == 1:
        version = Path(references.pop())
        if version.parent.resolve() == (folder / "limit_theme").resolve():
            artifact = version / "report.json"
            archived = read_json(artifact)
            if archived.get("facts_source", {}).get("sha256") == fact_ref["snapshot_sha256"]:
                attribution = {**archived, "report_path": str(version), "sha256": file_sha(artifact), "available_at": cutoff}
    closure_ref = next((r for r in frozen["sources"] if r["id"] == "C:closure"), {})
    records = {}
    if closure_ref.get("snapshot_sha256") and file_sha(closure_ref["source_path"]) == closure_ref["snapshot_sha256"]:
        closure = read_json(closure_ref["source_path"])
        if closure.get("as_of") == day:
            records = {r["symbol"]: r for r in closure.get("records", [])}
    funnel = HF.build(day, cutoff, facts, core, attribution, admission, relay, records)
    out = folder / "hotspot_funnel_replay_v1"
    out.mkdir(exist_ok=True)
    (out / "relay.json").write_text(json.dumps(relay, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "hotspot_funnel.json").write_text(json.dumps(funnel, ensure_ascii=False, indent=1), encoding="utf-8")
    report = [f"# {day} 热点接力观察（历史回放，不是当日发出的信号）", "",
              f"涨停 {relay['limit_count']}，当日交叉归因 {relay['attributed']}；其余不是负例。",
              f"昨日合格埋伏池 {relay['prior_watch_count']}；今日封板交集 {relay['watch_hits_today']}。仅代表早于本日，不保证早于首板。",
              "", "| 紧链 | 状态 | 已核实当日封板 | 昨日封板反馈 |", "|---|---|---|---|"]
    report += ["| " + " | ".join(row) + " |" for row in lines(relay)]
    report += ["", "## 逐股漏斗", "", f"全量对账：{funnel['universe']}；正式参与榜：{funnel['participation_top']}。",
               "历史版本引用的最迟可证实时间：" + cutoff + "。无当时发布回执的路线文件不作前向证据。",
               "", "| 股票 | 题材 | 角色和证据 | 原引擎 | 未参与原因 |", "|---|---|---|---|---|"]
    report += ["| " + " | ".join(str(r[i]) for i in (1, 2, 3, 6, 7)) + " |" for r in HF.table_rows(funnel)]
    report += ["", f"正式市场门：{relay['market_allowed']}；原系统可复核：{relay['eligible_count']}只。",
               "", "| 当日封板 | 昨晚载体资格 | 今日归因（未核实则留空） | 当前生命周期 | EMA20偏离% |",
               "|---|---|---|---|---:|"]
    for r in relay["hits"]:
        report.append(f"| {r['name']} {r['symbol']} | {r['prior_carrier_stage'] or '无可用记录'} | "
                      f"{(r['attribution'] or {}).get('theme', '--')} | {r['lifecycle'] or '--'} | {r['ema20_bias_pct']} |")
    report += ["", "| 路线仍在续踪 | 最近封板 | 当前路线与条件 |", "|---|---|---|"]
    for s, item in sorted(relay["followup"].items()):
        why = "、".join(r["route"] + "：" + r["reason"] for r in item["routes"]) or "资料不足，保留待核"
        report.append(f"| {relay['names'].get(s, s)} {s} | {item['last_board']} | {why} |")
    report += ["", "标签与涨幅不是主力身份或交易许可。次日看旧核心收盘、独立新成员、爆量破位；不以未取得归因推定退潮。"]
    (out / "review.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(out / "review.md")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    replay(parser.parse_args().date, Path("results/mainwave_engine"))
