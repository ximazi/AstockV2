"""Per-step route diagnosis. Absent evidence is UNKNOWN, never a silent pass or fail."""
from collections import Counter
from . import store

VERSION = "route_diagnostics_v1_20260920"
# Upstream checks speak four states; WAIT ("not yet") must not collapse into FAIL or UNKNOWN.
PASS, FAIL, WAIT, UNKNOWN = "PASS", "FAIL", "WAIT", "UNKNOWN"
STATUS_LABELS = {"PASS": "通过", "FAIL": "未通过", "WAIT": "条件未到", "UNKNOWN": "证据缺失"}
NECESSARY, SCORING, WAITING = "NECESSARY", "SCORING", "WAITING"

ROUTE_LABELS = {"R1": "明确点火个股反查", "R2": "个股先行准备", "R3": "紧链扩散"}
ROUTE_QUESTIONS = {
    "R1": "这是低中位启动、延续加速、旧周期修复，还是高位供给风险？点火前是否已经准备？",
    "R2": "为何低位整理值得观察？资金活动是否留下可保留的价格效果？缩量是收敛还是缺乏需求？",
    "R3": "是否存在当时可知的关联与传播？慢涨是节奏稍晚还是始终弱势？",
}


def _flag(value):
    """Tri-state: a missing field is unknown, not false."""
    if value is None or value == "":
        return UNKNOWN
    return PASS if bool(value) else FAIL


def _has(value):
    return value not in (None, "", [], {})


def _bias(context):
    value = context.get("bias_atr")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _step(sid, label, kind, status, value, blocker="", nxt=""):
    return {"id": sid, "label": label, "kind": kind, "status": status, "value": value,
            "blocker": blocker if status == FAIL else "", "next_condition": nxt,
            "rule_version": VERSION}


def _r1(c):
    checks = c["checks"]
    bias = _bias(c)
    ignition = c.get("ignition_at") or c.get("attempt_at")
    return [
        _step("prior_extension", "先前涨幅与伸展", SCORING,
              PASS if bias is not None and bias <= 2 else FAIL if bias is not None else UNKNOWN,
              {"bias_atr": bias, "location": c.get("location"), "phase": c.get("phase_label")},
              "价格已远离结构，属于高位供给风险区", "回到结构 2ATR 以内再复核"),
        _step("pre_ignition_preparation", "点火前是否已准备", NECESSARY,
              _flag(c.get("prepared_before_attempt")) if _has(c.get("prepared_before_attempt"))
              else (PASS if _has(c.get("prepared_at")) else UNKNOWN),
              {"prepared_at": c.get("prepared_at"), "preparation_state": c.get("preparation_state"),
               "preparation_age": c.get("preparation_age")},
              "点火前没有可核对的准备证据", "等待下一轮准备形成"),
        _step("retention", "点火前量价保留", NECESSARY, _flag(c.get("retained")),
              {"retained": c.get("retained"), "probe_at": c.get("probe_at")},
              "冲击后价格效果未保留", "等待新的冲击并观察保留"),
        _step("breakout_accepted", "突破与接受属同一事件", NECESSARY,
              checks.get("breakout", UNKNOWN),
              {"ignition_at": ignition, "accepted_at": c.get("accepted_at"),
               "breakout_status": c.get("breakout_status")},
              "突破未被接受或与原事件不一致", "等待收盘接受确认"),
        _step("advance", "点火后推进", SCORING,
              PASS if c.get("breakout_status") in {"ADVANCING"} else
              UNKNOWN if not _has(c.get("breakout_status")) else FAIL,
              {"breakout_status": c.get("breakout_status")},
              "接受后未形成推进", "观察推进或转入回踩复核"),
        _step("pullback", "回踩与确认时效", WAITING, checks.get("confirmation", UNKNOWN),
              {"setup_kind": c.get("setup_kind"), "review_window_active": c.get("window_active")},
              "确认已超出观察时效", "等待新的回踩确认"),
        _step("supply_warning", "放量滞涨与失败反抽", NECESSARY,
              FAIL if c.get("supply") else PASS if c.get("supply") is False else UNKNOWN,
              {"supply_warning": c.get("supply"), "risks": c.get("risks")},
              "出现放量滞涨或失败反抽", "结构修复后重新评估"),
    ]


def _r2(c):
    checks = c["checks"]
    return [
        _step("impulse", "冲击出现", NECESSARY,
              PASS if _has(c.get("probe_at")) else UNKNOWN,
              {"probe_at": c.get("probe_at"), "preparation_origin": c.get("preparation_origin")},
              "没有可核对的资金冲击", "等待放量冲击出现"),
        _step("impulse_retained", "冲击后价格效果保留", NECESSARY, _flag(c.get("retained")),
              {"retained": c.get("retained")}, "冲击后回吐，未留下价格效果", "等待下一次冲击并观察保留"),
        _step("pullback_supply", "回撤期供给", NECESSARY, checks.get("route_supply", UNKNOWN),
              {"supply_warning": c.get("supply"), "activity_contraction": c.get("activity_contraction")},
              "回撤期供给压力未减", "等待回撤缩量"),
        _step("contraction_meaning", "缩量是收敛还是缺乏需求", SCORING,
              FAIL if c.get("contraction_disagreement") else
              PASS if c.get("volatility_contraction") or c.get("activity_contraction") else UNKNOWN,
              {"volatility_contraction": c.get("volatility_contraction"),
               "activity_contraction": c.get("activity_contraction"),
               "contraction_disagreement": c.get("contraction_disagreement")},
              "缩量与波动收敛背离，偏向缺乏需求", "等待需求侧出现"),
        _step("demand", "反复承接", NECESSARY, _flag(c.get("demand")),
              {"demand": c.get("demand")}, "没有可重复观察的承接", "等待承接再次出现"),
        _step("structure_room", "结构范围与压力空间", SCORING, checks.get("boundary", UNKNOWN),
              {"bias_atr": _bias(c), "atr_price": c.get("atr_price"),
               "reference_close": c.get("reference_close")},
              "结构边界或波动数据不足", "等待结构边界重新成立"),
        _step("preparation_alive", "准备是否过期", WAITING,
              _flag(c.get("preparation_valid")) if _has(c.get("preparation_valid")) else UNKNOWN,
              {"preparation_state": c.get("preparation_state"), "preparation_age": c.get("preparation_age"),
               "preparation_first_seen": c.get("preparation_first_seen")},
              "准备已过期", "等待新的准备形成"),
    ]


def _r3(c):
    checks = c["checks"]
    return [
        _step("membership", "题材成员与关系来源", NECESSARY,
              PASS if _has(c.get("theme")) and c.get("theme") != "未归属" else UNKNOWN,
              {"theme": c.get("theme"), "preparation_source": c.get("preparation_source")},
              "没有可核对的题材归属", "等待题材成员关系确认"),
        _step("membership_dated", "关系生效时间当时可知", NECESSARY,
              _flag(c.get("strict_chain_verified")) if _has(c.get("strict_chain_verified")) else UNKNOWN,
              {"strict_chain_verified": c.get("strict_chain_verified"),
               "chain_linked": c.get("chain_linked")},
              "链条关系未通过当时可知校验", "补齐带日期的成员关系"),
        _step("leader_first", "领头先启动", NECESSARY, checks.get("route_context", UNKNOWN),
              {"route_status": c.get("route_status"), "theme_phase": c.get("theme_phase")},
              "题材未处于可传播阶段", "等待领头启动"),
        _step("carrier_order", "载体响应顺序", SCORING,
              PASS if _has(c.get("attempt_at")) and _has(c.get("probe_at")) else UNKNOWN,
              {"probe_at": c.get("probe_at"), "attempt_at": c.get("attempt_at"),
               "accepted_at": c.get("accepted_at")},
              "响应顺序无法重建", "补齐事件日期"),
        _step("relative_strength", "相对强度", SCORING, checks.get("route_demand", UNKNOWN),
              {"demand": c.get("demand"), "retained": c.get("retained")},
              "相对强度不足，属于始终偏弱而非节奏稍晚", "观察相对强度转正"),
        _step("structure_ready", "结构准备", NECESSARY, checks.get("route_retention", UNKNOWN),
              {"preparation_state": c.get("preparation_state"), "retained": c.get("retained")},
              "结构准备不足", "等待结构准备完成"),
        _step("chase_counter", "退潮与追高反证", NECESSARY,
              FAIL if (_bias(c) or 0) > 2 or c.get("supply") else
              PASS if _bias(c) is not None else UNKNOWN,
              {"bias_atr": _bias(c), "supply_warning": c.get("supply"), "risks": c.get("risks")},
              "已进入追高或退潮区", "等待回到结构内"),
    ]


BUILDERS = {"R1": _r1, "R2": _r2, "R3": _r3}


def _context(record):
    evidence = record.get("evidence") or {}
    source = record.get("source_record") or {}
    route = evidence.get("route_evidence") or {}
    vector = evidence.get("state_vector") or {}
    checks = {c.get("id"): c.get("status", UNKNOWN) for c in (evidence.get("checks") or [])}
    window = evidence.get("review_window") or {}
    return {**route, "checks": checks, "theme": record.get("theme"),
            "setup_kind": record.get("setup_kind"), "window_active": window.get("active"),
            "ignition_at": source.get("ignition_at"), "breakout_status": source.get("breakout_status"),
            "route_status": source.get("route_status"), "chain_linked": source.get("chain_linked"),
            "location": source.get("location"), "phase_label": source.get("phase_label"),
            "preparation_state": source.get("preparation_state") or route.get("preparation_state"),
            "theme_phase": vector.get("theme_phase")}


def card(record):
    """One diagnosis card per declared route; overlaps keep every route's own answer."""
    context = _context(record)
    routes = [r for r in str(record.get("route_ids", "")).split("|") if r in BUILDERS]
    cards = []
    for route in routes:
        steps = BUILDERS[route](context)
        necessary = [s for s in steps if s["kind"] == NECESSARY]
        blocking = [s for s in necessary if s["status"] == FAIL]
        unknown = [s for s in necessary if s["status"] == UNKNOWN]
        pending = [s for s in steps if s["status"] == WAIT]
        verdict = ("FAILED" if blocking else "INSUFFICIENT_DATA" if unknown else
                   "READY" if all(s["status"] == PASS for s in necessary) and not pending else "WAITING")
        stopped = (blocking or unknown or pending or [])
        first_failure = stopped[0]["id"] if stopped else ""
        cards.append({"route": route, "label": ROUTE_LABELS[route], "question": ROUTE_QUESTIONS[route],
                      "verdict": verdict, "first_blocking_step": first_failure, "steps": steps,
                      "stopped_status": stopped[0]["status"] if stopped else PASS,
                      "next_condition": (stopped or [{"next_condition": record.get("next_condition", "")}])[0].get("next_condition", ""),
                      "rule_version": VERSION})
    return {"symbol": record["symbol"], "name": record.get("name", record["symbol"]),
            "opportunity_id": record.get("opportunity_id"), "route_ids": record.get("route_ids", ""),
            "cards": cards, "declared_routes": routes,
            "note": "无声明路线的候选按结构观察处理，不给出路线诊断。" if not routes else ""}


def report(payload, limit=3):
    """Per-step denominators plus named cases; a route with zero ready targets is a valid result."""
    cards = [card(r) for r in payload["records"]]
    routes = {}
    for item in cards:
        for entry in item["cards"]:
            bucket = routes.setdefault(entry["route"], {
                "route": entry["route"], "label": entry["label"], "question": entry["question"],
                "total": 0, "verdicts": Counter(), "steps": {}, "cases": {}})
            bucket["total"] += 1
            bucket["verdicts"][entry["verdict"]] += 1
            for step in entry["steps"]:
                row = bucket["steps"].setdefault(step["id"], {
                    "id": step["id"], "label": step["label"], "kind": step["kind"],
                    "denominator": 0, "counts": Counter()})
                row["denominator"] += 1
                row["counts"][step["status"]] += 1
            if entry["first_blocking_step"]:
                bucket.setdefault("first_failures", Counter())[
                    f'{entry["first_blocking_step"]}:{entry["stopped_status"]}'] += 1
            cases = bucket["cases"].setdefault(entry["verdict"], [])
            if len(cases) < limit:
                cases.append({"symbol": item["symbol"], "name": item["name"],
                              "opportunity_id": item["opportunity_id"],
                              "first_blocking_step": entry["first_blocking_step"],
                              "next_condition": entry["next_condition"]})
    for bucket in routes.values():
        bucket["verdicts"] = dict(bucket["verdicts"])
        bucket["first_failures"] = dict(bucket.get("first_failures", {}))
        for row in bucket["steps"].values():
            row["counts"] = dict(row["counts"])
            row["pass_rate"] = round(row["counts"].get(PASS, 0) / row["denominator"], 4) if row["denominator"] else None
        bucket["steps"] = list(bucket["steps"].values())
    undeclared = [c for c in cards if not c["declared_routes"]]
    return {"version": VERSION, "as_of": payload["as_of"], "snapshot_id": payload.get("snapshot_id", ""),
            "routes": list(routes.values()), "cards": cards, "status_labels": STATUS_LABELS,
            "summary": {"analyzed": len(cards), "with_route": len(cards) - len(undeclared),
                        "without_route": len(undeclared),
                        "by_route": {k: v["total"] for k, v in routes.items()}},
            "notice": "逐环节分母按声明该路线的候选计算；证据缺失记 UNKNOWN，不按未通过处理。"
                      "本报告描述筛选过程，不含收益结论。"}
