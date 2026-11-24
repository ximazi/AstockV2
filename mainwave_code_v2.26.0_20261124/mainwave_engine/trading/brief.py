"""The daily deliverable: Top3/5, per-stock step diagnosis, cross-day change, outcome comparison.

Every section carries three separate stamps — software implemented, historically validated,
forward validated — derived from evidence on disk. Having output never counts as proof.
"""
import json
from pathlib import Path
from . import store

VERSION = "daily_brief_v1_20260920"
BANNER = "有输出不等于筛选能力已被证明。每一节分别标注：软件已实现 / 历史已验证 / 前向已验证，三者互不推出。"
STATUS = {
    "software": {"IMPLEMENTED": "软件已实现", "MISSING_INPUT": "缺输入，未生成"},
    "historical": {"NOT_DONE": "历史验证未做", "PARTIAL": "历史验证部分完成", "DONE": "历史验证完成"},
    "forward": {"NOT_STARTED": "前向验证未开始", "RECORDING": "前向记录中（样本未成熟）", "MATURE": "前向样本已成熟"},
}
ARTEFACTS = {"board": "ranking_explained.json", "cards": "route_diagnostics.json",
             "chain": "opportunity_continuity.json", "review": "ranking_validation.json",
             "audit": "validation_audit.json", "books": "books_daily.json"}
BLOCKERS = {"DATA_WAIT": "数据待核", "STRUCTURE_RISK": "结构风险", "MARKET_WAIT": "市场闸门",
            "WAIT_PULLBACK_CONFIRMATION": "回踩未确认", "TRACKING_ONLY": "仅续踪", "BENCHMARK_MISSING": "基准缺失"}
CHANGE = {"OWN_MOVE": "自身变化", "COHORT_STRENGTHENED": "比较域走强（分升名降）",
          "COHORT_WEAKENED": "比较域走弱（分降名升）", "NO_PRIOR_ROUND": "无前一轮"}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig")) if Path(path).exists() else None


def _day_folders(root):
    return sorted(p for p in (Path(root) / "results/mainwave_engine").glob("20??????") if p.is_dir())


def _days_with(root, name):
    return [p.name for p in _day_folders(root) if (p / name).exists()]


def _stamp(software, historical, forward, note=""):
    return {"software": software, "historical": historical[0], "historical_note": historical[1],
            "forward": forward[0], "forward_note": forward[1],
            "labels": {"software": STATUS["software"][software], "historical": STATUS["historical"][historical[0]],
                       "forward": STATUS["forward"][forward[0]]}, "note": note}


LIVE_BASIS = "FORWARD_RECORDED_AFTER_CLOSE"


def _forward_from_review(review, days):
    """Only rows recorded live after the close count as forward; reconstructions are reported apart."""
    if not review:
        return ("NOT_STARTED", "无验证产物")
    by_basis = review.get("summary", {}).get("by_basis")
    if by_basis is None:
        return ("NOT_STARTED", "验证产物未区分回建与前向来源，不作为前向记录")
    live = by_basis.get(LIVE_BASIS, {})
    rebuilt = sum(v.get("rows", 0) for k, v in by_basis.items() if k != LIVE_BASIS)
    tail = f"；回建 {rebuilt} 行另计，不入前向" if rebuilt else ""
    mature, rows, live_days = live.get("mature_rows", 0), live.get("rows", 0), live.get("independent_days", 0)
    if mature:
        return ("MATURE", f"前向成熟 {mature} 行 / 前向记录 {live_days} 日{tail}")
    if rows:
        return ("RECORDING", f"前向记录 {live_days} 日 {rows} 行，成熟 0 行；未成熟不记零{tail}")
    return ("NOT_STARTED", f"尚无前向记录{tail}")


def _historical_from_protocols(audit):
    protocols = [p for p in ((audit or {}).get("protocols") or {}).get("protocols", []) if p.get("frozen_days")]
    if not protocols:
        return ("NOT_DONE", "无冻结研究协议")
    names = "；".join(f"{p['protocol']} 冻结 {p['frozen_days']} 日" for p in protocols)
    return ("PARTIAL", f"{names}；结论仍在噪声内，未达到 DONE")


def _top(board, key, cards_by_symbol, chain_by_opp, limit):
    rows = []
    for row in (board or {}).get(key, [])[:limit]:
        explained = (board or {}).get("explained", {}).get(row["opportunity_id"], {})
        card = cards_by_symbol.get(row["symbol"], {})
        event = chain_by_opp.get(row["opportunity_id"], {})
        rows.append({
            "symbol": row["symbol"], "name": row.get("name"), "theme": row.get("theme"),
            "route_ids": row.get("route_ids", ""), "state": row.get("state"),
            "strength_rank": row.get("strength_rank"), "participation_rank": row.get("participation_rank"),
            "strength_score": row.get("strength_score"),
            "blockers": [BLOCKERS.get(b, b) for b in row.get("blockers") or []],
            "verdict": explained.get("verdict"), "verdict_reading": explained.get("verdict_reading"),
            "rank_change": row.get("rank_change"),
            "change_attribution": (explained.get("change") or {}).get("attribution"),
            "change_reading": (explained.get("change") or {}).get("reading"),
            "diagnosis": [{"route": c["route"], "verdict": c["verdict"],
                           "stopped_at": c.get("first_blocking_step") or "", "stopped_status": c.get("stopped_status", ""),
                           "next_condition": c.get("next_condition", "")} for c in card.get("cards", [])],
            "continuity_event": event.get("event"), "continuity_reason": event.get("reason"),
        })
    return rows


def _outcomes(review, audit, cards_report):
    horizon = (review or {}).get("horizons", [5])[0]
    comparisons = [c for c in (review or {}).get("comparisons", []) if c["horizon"] == horizon]
    latest = max((c["signal_day"] for c in comparisons), default="")
    # The newest signal day is usually immature; the newest matured one is shown beside it, never hidden by it.
    mature_day = max((c["signal_day"] for c in comparisons if c["mature_n"]), default="")
    wanted = ("strength_top3", "strength_top5", "participation_top3", "participation_top5",
              "legacy_top3", "legacy_top5", "momentum_top3", "momentum_top5", "all_candidates",
              "route_R1", "route_R2", "r3_research")
    cohorts = [{"group": c["group"], "signal_day": c["signal_day"], "basis": c.get("basis", "UNKNOWN_BASIS"),
                "alpha_scope": c.get("alpha_scope", "PRICE_ALPHA"),
                "theme_day_clusters": (c.get("clusters") or {}).get("theme_days"),
                "selected": c["selected_n"], "mature": c["mature_n"],
                "independent_opportunities": c.get("independent_opportunities"),
                "mean": c.get("mean"), "positive_fraction": c.get("positive_fraction"), "worst": c.get("worst"),
                "complete": c.get("complete", False),
                "reading": "已成熟，可读" if c["mature_n"] and c.get("complete") else
                           "部分成熟，不作定论" if c["mature_n"] else "未成熟，不记零"}
               for c in comparisons if c["signal_day"] in {latest, mature_day} and c["group"] in wanted]
    missed = (audit or {}).get("missed") or {}
    process = [{"route": r["route"], "label": r["label"], "total": r["total"], "verdicts": r["verdicts"],
                "top_stops": dict(sorted(r.get("first_failures", {}).items(), key=lambda kv: -kv[1])[:3]),
                "cases": {k: [c["symbol"] for c in v] for k, v in r.get("cases", {}).items()}}
               for r in (cards_report or {}).get("routes", [])]
    return {"horizon": horizon, "signal_day": latest, "mature_signal_day": mature_day, "cohorts": cohorts,
            "missed": {"status": missed.get("status"), "rule": (missed.get("rule") or {}).get("definition"),
                       "signal_day": missed.get("signal_day"),
                       "matured_signal_days": missed.get("matured_signal_days", []),
                       "summary": missed.get("summary"), "cases": (missed.get("cases") or [])[:5],
                       "notice": missed.get("notice")},
            "process": process,
            "process_note": "过程侧的通过/未通过/证据不足，不是结果侧的成功/失败；结果只在窗口成熟后才有。"}


def build(day, root=store.ROOT, limit=5):
    """Assemble from the day's frozen artefacts; a missing input is named, never blanked."""
    root = Path(root)
    folder = root / "results/mainwave_engine" / day.replace("-", "")
    inputs = {key: _read(folder / name) for key, name in ARTEFACTS.items()}
    missing = [ARTEFACTS[k] for k, v in inputs.items() if v is None]
    board, cards_report, chain, review, audit, ledgers = (inputs[k] for k in ("board", "cards", "chain", "review", "audit", "books"))
    cards_by_symbol = {c["symbol"]: c for c in (cards_report or {}).get("cards", [])}
    chain_by_opp = {e["opportunity_id"]: e for e in (chain or {}).get("events", [])}

    strength = _top(board, "strength_top", cards_by_symbol, chain_by_opp, limit)
    participation = _top(board, "participation_top", cards_by_symbol, chain_by_opp, limit)

    earlier = [p for p in _day_folders(root) if p.name < day.replace("-", "") and (p / ARTEFACTS["board"]).exists()]
    previous = _read(earlier[-1] / ARTEFACTS["board"]) if earlier else None
    today_set = {r["symbol"] for r in strength}
    prev_set = {r["symbol"] for r in (previous or {}).get("strength_top", [])[:limit]}
    # Rank history can predate the continuity chain; say so instead of letting "+29" sit next to "首次".
    chain_gap = any(r["rank_change"] is not None and r["continuity_event"] == "DISCOVERED" for r in strength)
    cross_day = {
        "chain_note": ("连续性记录起点晚于排序记录：名次变化来自排序历史，今日事件按连续性首次记录，两者时间基线不同"
                       if chain_gap else ""),
        "previous_day": (previous or {}).get("as_of", ""),
        "entered_top": sorted(today_set - prev_set) if previous else [],
        "left_top": sorted(prev_set - today_set) if previous else [],
        "stayed": sorted(today_set & prev_set) if previous else [],
        "comparison_basis": "与最近一个有排序解释产物的交易日比较" if previous else "无前日对照",
        "continuity": (chain or {}).get("summary", {}),
        "rows": [{"symbol": r["symbol"], "name": r["name"], "rank_change": r["rank_change"],
                  "attribution": CHANGE.get(r["change_attribution"], r["change_attribution"]),
                  "reading": r["change_reading"], "event": r["continuity_event"], "reason": r["continuity_reason"]}
                 for r in strength],
    }

    review_days = len(_days_with(root, ARTEFACTS["review"]))
    forward_review = _forward_from_review(review, review_days)
    historical = _historical_from_protocols(audit)
    audit_report = (root / "research/theme_ignition/audit_results/AUDIT_REPORT.md").exists()
    sections = {
        "top": {"strength": strength, "participation": participation,
                "stamp": _stamp("IMPLEMENTED" if board else "MISSING_INPUT", historical, forward_review,
                                "榜单是可复核顺序，不是买入清单；分数未校准。")},
        "diagnosis": {"per_stock": {r["symbol"]: r["diagnosis"] for r in strength + participation},
                      "stamp": _stamp("IMPLEMENTED" if cards_report else "MISSING_INPUT",
                                      ("PARTIAL", "三题材命中审计已做（research/theme_ignition/audit_results）") if audit_report
                                      else ("NOT_DONE", "无历史命中审计"),
                                      ("RECORDING", f"记录 {len(_days_with(root, ARTEFACTS['cards']))} 日") if cards_report
                                      else ("NOT_STARTED", ""),
                                      "环节结论描述筛选过程；证据缺失记 UNKNOWN，不按未通过处理。")},
        "cross_day": {**cross_day,
                      "stamp": _stamp("IMPLEMENTED" if (board and chain) else "MISSING_INPUT",
                                      ("NOT_DONE", "名次变化归因未做历史研究"),
                                      ("RECORDING", f"连续性记录 {len(_days_with(root, ARTEFACTS['chain']))} 日") if chain
                                      else ("NOT_STARTED", ""),
                                      "分升名降归因于比较域变化，与自身改善分开。")},
        "outcomes": {**_outcomes(review, audit, cards_report),
                     "stamp": _stamp("IMPLEMENTED" if (review and audit) else "MISSING_INPUT", historical, forward_review,
                                     "固定市场交易日窗口；窗口内停牌记 DATA_GAP；漏选按预登记后验定义审计且不回灌排名。")},
    }
    auto = (ledgers or {}).get("auto") or {}
    sections["books"] = {
        "signal": (ledgers or {}).get("signal") or {}, "auto": auto, "manual": (ledgers or {}).get("manual") or {},
        "separation": "信号账回答发现有无Alpha，自动账回答机械策略是否赚钱，人工账回答人工有无增益；三者不合并。",
        "stamp": _stamp("IMPLEMENTED" if ledgers else "MISSING_INPUT", ("NOT_DONE", "三本账只有前向记录，没有历史"),
                        ("RECORDING", f"自动账 {auto.get('plans', 0)} 个计划 / {auto.get('fills', 0)} 笔成交") if auto.get("plans")
                        else ("NOT_STARTED", "自动账尚无计划"),
                        "自动账按冻结机械规则下单，人工账由人挑选；各自样本成熟前不比较。")}
    proven = all(s["stamp"]["historical"] == "DONE" and s["stamp"]["forward"] == "MATURE" for s in sections.values())
    return {"version": VERSION, "as_of": day, "banner": BANNER, "status_legend": STATUS,
            "screening_ability_proven": proven,
            "proof_reason": ("历史验证与前向验证均已完成" if proven else
                             "至少一节的历史验证未完成或前向样本未成熟；有输出不等于已证明"),
            "missing_inputs": missing, "sections": sections,
            "snapshot_id": (board or {}).get("snapshot_id", "")}


def _row(cells):
    return "| " + " | ".join("" if c is None else str(c) for c in cells) + " |"


def render(brief):
    lines = [f"# 每日交付 {brief['as_of']}", "", f"> {brief['banner']}", "",
             f"- 筛选能力已证明：**{'是' if brief['screening_ability_proven'] else '否'}** —— {brief['proof_reason']}"]
    if brief["missing_inputs"]:
        lines.append(f"- 缺输入：{'、'.join(brief['missing_inputs'])}（对应节标 MISSING_INPUT，不用旧结果顶替）")
    for key, title in (("top", "一、Top3/5"), ("diagnosis", "二、每只逐环节诊断"),
                       ("cross_day", "三、跨日变化"), ("outcomes", "四、成功/失败/漏选对照"),
                       ("books", "五、三本账（不合并）")):
        s = brief["sections"][key]["stamp"]
        lines += ["", f"## {title}", "",
                  f"- 状态：{s['labels']['software']} ｜ {s['labels']['historical']}（{s['historical_note']}）｜ {s['labels']['forward']}（{s['forward_note']}）",
                  f"- 口径：{s['note']}", ""]
        if key == "top":
            for name, rows in (("强势榜", brief["sections"]["top"]["strength"]), ("参与榜", brief["sections"]["top"]["participation"])):
                lines += [f"### {name}", "", _row(["强势#", "参与#", "股票", "题材", "路径", "状态", "阻断", "结论", "名次变化"]),
                          _row(["---"] * 9)]
                for r in rows:
                    lines.append(_row([r["strength_rank"], r["participation_rank"], f"{r['name']} {r['symbol']}", r["theme"],
                                       r["route_ids"], r["state"], "；".join(r["blockers"]) or "无", r["verdict_reading"],
                                       "首次" if r["rank_change"] is None else f"{r['rank_change']:+d}"]))
                lines.append("")
        elif key == "diagnosis":
            lines += [_row(["股票", "路径", "结论", "停在哪一步", "下一条件"]), _row(["---"] * 5)]
            for symbol, cards in brief["sections"]["diagnosis"]["per_stock"].items():
                if not cards:
                    lines.append(_row([symbol, "无声明路径", "按结构观察", "--", "--"]))
                for c in cards:
                    lines.append(_row([symbol, c["route"], c["verdict"], f"{c['stopped_at']}:{c['stopped_status']}" if c["stopped_at"] else "--", c["next_condition"]]))
        elif key == "cross_day":
            x = brief["sections"]["cross_day"]
            if x.get("chain_note"):
                lines.append(f"- 注意：{x['chain_note']}")
            lines += [f"- 对照日：{x['previous_day'] or '无'}（{x['comparison_basis']}）",
                      f"- 进入 Top：{'、'.join(x['entered_top']) or '无'}；离开 Top：{'、'.join(x['left_top']) or '无'}；留任：{len(x['stayed'])}",
                      f"- 连续性事件：{x['continuity'].get('by_event', {})}", "",
                      _row(["股票", "名次变化", "归因", "解释", "今日事件"]), _row(["---"] * 5)]

            for r in x["rows"]:
                lines.append(_row([f"{r['name']} {r['symbol']}", "首次" if r["rank_change"] is None else f"{r['rank_change']:+d}",
                                   r["attribution"], r["reading"], f"{r['event']}：{r['reason']}" if r["event"] else "--"]))
        elif key == "books":
            b = brief["sections"]["books"]
            sig, au, ma = b["signal"], b["auto"], b["manual"]
            chosen = "、".join("{}#{}".format(r["symbol"], r["participation_rank"]) for r in sig.get("trade_set") or []) or "空集"
            placed = "、".join("{}:{}".format(p["symbol"], p["status"]) for p in au.get("placed") or []) or "无下单"
            lines += [f"- {b['separation']}",
                      f"- 信号账：交易集 {chosen}（可复核 {sig.get('eligible', 0)}，R3-only 排除 {sig.get('excluded_r3_research', 0)}）",
                      f"- 自动账：{'当日实跑' if au.get('live') else '非当日不下单'}；计划 {au.get('plans', 0)}，持仓 {au.get('open_positions', 0)}，"
                      f"成交 {au.get('fills', 0)}，对账 {au.get('reconciliation', '--')}；本次 {placed}",
                      f"- 人工账：{ma.get('status', '--')}；绩效 {ma.get('performance', '--')}"]
        else:
            o = brief["sections"]["outcomes"]
            lines += [f"- 结果侧（{o['horizon']} 日窗口，最新信号日 {o['signal_day'] or '无'}，最新已成熟信号日 {o.get('mature_signal_day') or '无'}）：", "",
                      _row(["信号日", "来源", "对照组", "选定", "已成熟", "独立机会", "簇(题材×日)", "均值%", "上涨比例", "最差%", "读法"]), _row(["---"] * 11)]
            for c in o["cohorts"]:
                lines.append(_row([c["signal_day"], "前向" if c.get("basis") == LIVE_BASIS else "回建",
                                   c["group"] + ("（仅研究）" if c.get("alpha_scope") == "R3_RESEARCH" else ""),
                                   c["selected"], c["mature"], c["independent_opportunities"], c.get("theme_day_clusters"),
                                   None if c["mean"] is None else round(c["mean"], 2),
                                   None if c["positive_fraction"] is None else f"{c['positive_fraction']:.0%}",
                                   None if c["worst"] is None else round(c["worst"], 2), c["reading"]]))
            m = o["missed"]
            lines += ["", f"- 漏选审计：{m['status']}；定义：{m['rule'] or '--'}；已成熟信号日 {'、'.join(m.get('matured_signal_days') or []) or '无'}；{m['notice'] or ''}"]
            if m["summary"]:
                lines.append(f"  摘要：{m['summary']}")
            lines += ["", f"- 过程侧：{o['process_note']}", ""]
            for p in o["process"]:
                lines.append(f"  - {p['route']} {p['label']}：{p['total']} 只，结论 {p['verdicts']}，最常停在 {p['top_stops']}")
    return "\n".join(lines) + "\n"
