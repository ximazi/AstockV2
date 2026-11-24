"""The review.report.v2 contract: JSON schema plus the semantic checks the template's renderer enforced.

Ported from the v2 package's render_report.validate so the rules stay identical: seven steps in order,
every reference resolves, required symbols never dropped, PLAN_READY only with a complete approved plan
inside a forward-frozen report, no history after the review day, no order authority."""
import json
import math
from datetime import datetime

from . import ASSETS


def load_schema():
    return json.loads((ASSETS / "report.schema.json").read_text(encoding="utf-8"))


def dt(value):
    if not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result if result.tzinfo else None
    except ValueError:
        return None


def validate(data):
    from jsonschema import Draft202012Validator
    problems = [f"{'/'.join(map(str, e.absolute_path)) or '/'}: {e.message}" for e in Draft202012Validator(load_schema()).iter_errors(data)]
    if problems:
        return problems
    ids = {}
    for key in ("stocks", "sources", "claims"):
        values = [r["id"] for r in data[key]]
        ids[key] = set(values)
        if len(values) != len(ids[key]):
            problems.append(f"{key}: 重复id")
    meta = data["meta"]
    knowledge = dt(meta["knowledge_as_of"])
    if [s["id"] for s in data["sections"]] != ["01", "02", "03", "04", "05", "06", "07"]:
        problems.append("七步顺序不能变更")
    for sec in data["sections"]:
        if sec["claim_id"] not in ids["claims"]:
            problems.append("未知claim_id " + sec["claim_id"])
        n = len(sec["table"]["columns"])
        if any(len(r) != n for r in sec["table"]["rows"]):
            problems.append("表格列数不一致: " + sec["id"])
    for cid in data["summary"]["claim_ids"]:
        if cid not in ids["claims"]:
            problems.append("摘要引用不存在的结论: " + cid)
    for fid in data["summary"]["focus_ids"]:
        if fid not in ids["stocks"]:
            problems.append("重点卡不存在: " + fid)
    for sym in meta["required_symbols"]:
        if not any(s["symbol"] == sym for s in data["stocks"]):
            problems.append("漏掉必保留股票: " + sym)

    def walk(x):
        if isinstance(x, list):
            for v in x:
                walk(v)
        if not isinstance(x, dict):
            return
        for k, v in x.items():
            if k in ("evidence_ids", "support_ids", "source_ids"):
                for sid in v:
                    if sid not in ids["sources"]:
                        problems.append("证据不存在: " + sid)
            elif k == "evidence_id" and v not in ids["sources"]:
                problems.append("证据不存在: " + v)
            else:
                walk(v)
    walk(data)
    source_by = {s["id"]: s for s in data["sources"]}
    for s in data["stocks"]:
        if s["mode"] not in meta["allowed_modes"]:
            problems.append("候选模式不在允许范围: " + s["id"])
        p = s["plan"]
        if p:
            for sid in p["source_ids"]:
                av = dt(source_by.get(sid, {}).get("available_at"))
                if not av or not knowledge or av > knowledge:
                    problems.append("计划证据可知时间未知/越界: " + sid)
        if s["status"] not in ("PLAN_READY", "SIGNAL_CONFIRMED"):
            continue
        if meta["mode"] in ("HISTORICAL_DEMO", "TEMPLATE"):
            problems.append("历史演示/模板不能声称已批准或触发")
        if not p:
            problems.append(s["id"] + "预案齐备但缺plan")
            continue
        for k in ("id", "strategy_version", "signal_condition", "exit_rule", "valid_from", "expires_at"):
            if not p[k].strip():
                problems.append(s["id"] + "缺计划字段" + k)
        for k in ("entry_min", "entry_max", "invalidation_price", "max_loss_cash"):
            if not math.isfinite(p[k]) or p[k] <= 0:
                problems.append(s["id"] + "数值必须为正: " + k)
        if not p["invalidation_price"] < p["entry_min"] <= p["entry_max"]:
            problems.append(s["id"] + "做多价格范围不合法")
        if p["approval_status"] != "APPROVED" or p["budget_status"] != "PASS" or p["hard_blocks"]:
            problems.append(s["id"] + "审批、预算或阻断未通过")
        if s["missing"] or not s["model_version"]:
            problems.append(s["id"] + "仍有必需缺失或无模式版本")
        if not p["source_ids"]:
            problems.append(s["id"] + "无计划依据")
        end, start = dt(p["expires_at"]), dt(p["valid_from"])
        if not end or not start or not knowledge or end <= knowledge or start >= end:
            problems.append(s["id"] + "时效无效")
        if s["status"] == "SIGNAL_CONFIRMED":
            quote, market = dt(p["quote_as_of"]), dt(meta["market_as_of"])
            if meta["mode"] != "INTRADAY_UPDATE" or p["trigger_state"] != "CONFIRMED" or p["quote_fresh"] is not True or meta["quote_status"] != "LIVE_SNAPSHOT":
                problems.append(s["id"] + "非实时已确认信号")
            if not quote or not market or not knowledge or quote > knowledge or quote > market or (start and start > knowledge):
                problems.append(s["id"] + "实时触发/报价时点不合法")
    for pair in data["comparisons"]:
        if pair["matched"] and pair["missing"]:
            problems.append("比较声称齐备但仍有缺失: " + pair["id"])
        for sym in pair["symbols"]:
            if not any(s["symbol"] == sym for s in data["stocks"]):
                problems.append("比较对象不在股票表: " + sym)
    if meta["forward_frozen"]:
        frozen = dt(meta["frozen_at"])
        if meta["mode"] == "HISTORICAL_DEMO":
            problems.append("重建不能标为前向冻结")
        if not frozen or not knowledge or frozen < knowledge:
            problems.append("冻结时间无效或早于可知截止")
        for s in data["sources"]:
            if s["origin"] in ("USER_MATERIAL", "DESIGN"):
                continue
            av = dt(s["available_at"])
            if not av or not knowledge or av > knowledge:
                problems.append("前向证据可知时刻未知/越界: " + s["id"])
    if meta["mode"] != "TEMPLATE":
        if not knowledge or not dt(meta["market_as_of"]):
            problems.append("必须提供带时区的市场与知识截止")
        elif dt(meta["market_as_of"]) > knowledge:
            problems.append("市场截止不能晚于知识截止")
    for h in data["history"]:
        if meta["trade_date"] and h["date"] > meta["trade_date"]:
            problems.append("历史图表包含复盘日之后数据")
    return sorted(set(problems))
