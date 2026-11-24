"""Bounded AI evidence packet (review.ai.input.v2), the analyzer prompt, and the merge of an analyzer's answer.

The packet carries facts, tables, every required stock, gaps and the mode contracts — never prior claims,
recommendations or raw prose. The analyzer's JSON (see assets/prompt_analyzer_v2.md) is merged back into
the report as interpretations; it can never add a price, a plan or a status upgrade."""
import hashlib
import json

from . import ASSETS, AI_SCHEMA_VERSION
from .contract import validate

CONSTRAINTS = [
    "只按指定时点分析，不用后续行情。历史演示不是前向样本。",
    "先锁定同目标/期限/预算，不用容量替代短线机会；也不反向变成涨幅越大越买。",
    "关键资料缺失不等于模式失败：记录补证任务，对替代标的同标准。",
    "同股需逐模式检查；研究子模式不得自动获实盘批准。",
    "不能删指定股、风险或反证；不能虚构价格、仓位、胜率或资金身份。",
    "返回源ID支持的解释/反证/未知/有限窗口请求；事实数字不改写。",
    "本包只有部分股票，不宣称完成全市场最优标的筛选。",
]


def prepare(report):
    errors = validate(report)
    if errors:
        raise ValueError("\n".join(errors))
    meta = report["meta"]
    # Deliberately omit claims, recommendations, raw source prose and any earlier AI ranking.
    sources = [{k: s[k] for k in ("id", "origin", "source_path", "locator", "snapshot_sha256", "available_at")}
               for s in report["sources"] if s["origin"] != "USER_MATERIAL"]
    stocks = [{"symbol": s["symbol"], "name": s["name"], "pinned": s["pinned"], "facts": s["metrics"], "fact_source_ids": s["support_ids"],
               "local_role": s["role"], "local_priority": f"{s['opportunity_label']}：{s['opportunity_reason']}",
               "prior_plan_id": s["prior_plan_id"], "plan": s["plan"], "data_missing": s["missing"], "current_status": s["status"],
               "model_to_test": meta["allowed_modes"], "role_to_verify": None,
               "note": "local_role/local_priority 是本地机械提名与打分（连板数、题材、换手等事实的加总），不是已验证判断；模型逐模式检查。"} for s in report["stocks"]]
    return {"schema_version": AI_SCHEMA_VERSION, "scope": meta["scope"], "scope_is_full_market_stock_screen": False,
            "time_boundary": {k: meta[k] for k in ("trade_date", "target_date", "market_as_of", "knowledge_as_of", "forward_frozen", "mode")},
            "task": {k: meta[k] for k in ("objective", "horizon", "allowed_modes", "rule_version", "mapping_version")},
            "market_metrics": report["metrics"], "history": report["history"],
            "local_tables": [{"section": s["id"], "columns": s["table"]["columns"], "rows": s["table"]["rows"],
                              "source_ids": s["table"]["evidence_ids"], "note": s["table"]["note"]}
                             for s in report["sections"] if s["id"] in ("01", "02", "03", "04", "05", "06")],
            "stocks": stocks, "required_symbols": meta["required_symbols"], "quality_issues": report["issues"],
            "source_catalog": sources,
            "mode_requirements": [{"mode_id": m["id"], "scope": m["scope"], "minimum": m["minimum"], "constraints": m["rule"]} for m in report["modes"]],
            "bounded_data_requests": report["requests"], "constraints": CONSTRAINTS}


def stats(report, packet):
    raw = json.dumps(packet, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {"display_input_bytes": len(json.dumps(report, ensure_ascii=False).encode("utf-8")), "ai_packet_bytes": len(raw),
            "ai_packet_sha256": hashlib.sha256(raw).hexdigest(), "required_symbols": report["meta"]["required_symbols"],
            "covered_required": all(any(s["symbol"] == sym for s in packet["stocks"]) for sym in report["meta"]["required_symbols"]),
            "stock_cards": len(packet["stocks"]), "source_index_entries": len(packet["source_catalog"]),
            "full_market_stock_screen": False, "token_count": None,
            "token_note": "未接目标模型计数器；字节不是 tokens。",
            "compression_note": "从展示包提取范围限定事实，不是同等信息无损压缩；不发送既有 AI 结论。"}


def prompt_text(packet):
    """System prompt (the v2 analyzer prompt) followed by the packet; paste into any model, save the JSON answer."""
    system = (ASSETS / "prompt_analyzer_v2.md").read_text(encoding="utf-8")
    return (system + "\n\n---\n\n## 输入事实包（review.ai.input.v2）\n\n```json\n"
            + json.dumps(packet, ensure_ascii=False, indent=1) + "\n```\n\n请只输出上述 JSON 结构，不输出 HTML。\n")


def merge_analysis(report, analysis):
    """Apply an analyzer's answer: interpretations, comparisons, requests, conflicts. Facts, prices and statuses stay."""
    known = {s["id"] for s in report["sources"]}
    dropped = []

    def keep(ids):
        good = [i for i in ids or [] if i in known]
        dropped.extend(i for i in ids or [] if i not in known)
        return good
    claims = {c["id"]: c for c in report["claims"]}
    by_section = {s["id"]: s["claim_id"] for s in report["sections"]}
    for a in analysis.get("section_assessments", []):
        cid = by_section.get(str(a.get("section_id", "")).zfill(2))
        if not cid or cid not in claims:
            continue
        c = claims[cid]
        c["type"] = "INTERPRETATION"
        for key, src in (("conclusion", "conclusion"), ("interpretation", "interpretation"), ("comparison", "comparison"), ("next_check", "next_check")):
            if a.get(src):
                c[key] = "AI：" + str(a[src])
        c["evidence_ids"] = sorted(set(c["evidence_ids"]) | set(keep(a.get("support_ids"))))
        extra = [f"反证 {i}" for i in keep(a.get("contrary_ids"))] + [str(u) for u in a.get("unknowns") or []]
        c["contrary"] = list(dict.fromkeys(c["contrary"] + extra))
    stocks = {s["symbol"]: s for s in report["stocks"]}
    for a in analysis.get("stock_assessments", []):
        s = stocks.get(a.get("symbol"))
        if not s:
            continue
        if a.get("opportunity"):
            s["opportunity_reason"] = str(a["opportunity"])
        if a.get("strongest_counterevidence"):
            s["contrary"] = str(a["strongest_counterevidence"])
        if a.get("next_check"):
            s["next_check"] = str(a["next_check"])
        modes = a.get("mode_assessments") or []
        if modes:
            s["notes"] = (s["notes"] + " ｜ " if s["notes"] else "") + "AI 逐模式：" + "；".join(
                f"{m.get('mode')}={m.get('result')}（{m.get('reason', '')}）" for m in modes)
        s["support_ids"] = sorted(set(s["support_ids"]) | set(keep([i for m in modes for i in (m.get("support_ids") or [])])))
    for i, pair in enumerate(analysis.get("pairwise_comparisons") or [], 1):
        symbols = [x for x in pair.get("symbols", []) if x in stocks]
        if len(symbols) < 2:
            continue
        missing = [str(m) for m in pair.get("missing") or []]
        report["comparisons"].append({"id": f"AI_CMP{i:02d}", "title": str(pair.get("title") or "同目标比较"), "symbols": symbols,
                                      "objective": str(pair.get("objective") or report["meta"]["objective"]),
                                      "matched": bool(pair.get("matched")) and not missing,
                                      "conclusion": str(pair.get("conclusion") or ""), "support_ids": keep(pair.get("support_ids")), "missing": missing})
    for i, req in enumerate(analysis.get("bounded_data_requests") or [], 1):
        report["requests"].append({"id": f"AI_REQ{i:02d}", "priority": req.get("priority") if req.get("priority") in ("P0", "P1", "P2") else "P1",
                                   "symbols": [x for x in req.get("symbols", []) if isinstance(x, str)],
                                   "purpose": str(req.get("claim_to_test") or req.get("purpose") or ""),
                                   "fields": [str(f) for f in req.get("fields") or []], "max_rows": int(req.get("max_rows") or 120),
                                   "cutoff": str(req.get("as_of") or report["meta"]["knowledge_as_of"])})
    for i, conflict in enumerate(analysis.get("unresolved_conflicts") or [], 1):
        report["issues"].append({"id": f"AI_Q{i:02d}", "severity": "WARN", "title": "AI 提出的未解决冲突",
                                 "detail": str(conflict), "impact": "待人工核对", "evidence_ids": []})
    missing_symbols = [str(x) for x in (analysis.get("coverage") or {}).get("missing_symbols") or [] if isinstance(x, str)]
    if missing_symbols:
        report["issues"].append({"id": "AI_COVERAGE", "severity": "WARN", "title": "AI 未覆盖的股票", "detail": "、".join(missing_symbols),
                                 "impact": "这些股票只有本地规则读法", "evidence_ids": []})
    if analysis.get("objective_confirmation"):
        c06 = claims.get(by_section.get("06"))
        if c06:
            c06["comparison"] = (c06["comparison"] + " AI 目标确认：" + str(analysis["objective_confirmation"])).strip()
    report["issues"] = [q for q in report["issues"] if q["id"] != "Q_NO_AI"]
    report["audit"]["note"] = (report["audit"]["note"] + " AI 解读已合并（解释层，不改事实、价格与状态）。").strip()
    return {"dropped_unknown_source_ids": sorted(set(dropped))}
