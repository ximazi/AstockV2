"""Mode contracts. Discovery, route, mode, local setup and permission stay five separate answers."""
from . import store

VERSION = "mode_contract_v1_20260920"

TREND_PULLBACK = "TREND_PULLBACK"
STRENGTH_CONFIRM = "STRENGTH_CONFIRM"
OVERNIGHT = "OVERNIGHT"

MODE_LABELS = {TREND_PULLBACK: "趋势回踩", STRENGTH_CONFIRM: "强势确认", OVERNIGHT: "尾盘隔夜"}
# Only the trend mode has a frozen executable contract in this batch.
EXECUTABLE = {TREND_PULLBACK}

SUB_SCENARIOS = {
    "BREAKOUT_TEST": {
        "label": "突破接受后测试", "mode": TREND_PULLBACK,
        "setup_kinds": ("BREAKOUT_ACCEPTANCE",),
        "confirm": "突破被收盘接受后回测不破接受位，缩量测试后再转强；确认日次日开盘复核。",
        "invalidate": "收盘跌破突破接受位，或接受事件被替换为新的突破编号。",
    },
    "FIRST_PULLBACK": {
        "label": "推进后首次回踩", "mode": TREND_PULLBACK,
        "setup_kinds": ("FIRST_PULLBACK",),
        "confirm": "推进后第一次回撤到 3%/6% 档，先缩量测试、后恢复转强且站上MA20。",
        "invalidate": "收盘跌破冻结失效位，或回撤跌破首仓价-2ATR0 的硬底线。",
    },
    "EMA10_RECLAIM": {
        "label": "EMA10下影再转强", "mode": TREND_PULLBACK,
        "setup_kinds": ("EMA10_RECLAIM", "LATER_PULLBACK"),
        "confirm": "下探EMA10留下下影后收盘收复EMA10，且量能不弱于前一日。",
        "invalidate": "连续两日收盘低于EMA10，或跌破冻结失效位。",
        "status": "MAPPED_NOT_ENABLED",
    },
}
OBSERVATION_ONLY = {
    STRENGTH_CONFIRM: {"label": "强势确认", "requires": "封板、开板与回封需分钟或盘口数据",
                       "status": "OBSERVATION_ONLY_DATA_INSUFFICIENT"},
    OVERNIGHT: {"label": "尾盘隔夜", "requires": "须绑定14:30后快照与次日退出窗口",
                "status": "OBSERVATION_ONLY_NEEDS_INTRADAY_SNAPSHOT"},
}
CONTRACT_SUMMARY = {"executable_modes": sorted(EXECUTABLE),
                    "sub_scenarios": {k: v["label"] for k, v in SUB_SCENARIOS.items()},
                    "observation_only": sorted(OBSERVATION_ONLY)}

# Hard risk and an already effective exit outrank any new mode signal.
PRIORITY = ("ACTIVE_EXIT", "HARD_RISK", "PENDING_REDUCE", "EXISTING_PLAN", "NEW_MODE_SIGNAL")


def classify(record):
    """The five fields, answered separately; an unknown local setup never inherits permission."""
    evidence = record.get("evidence") or {}
    vector = evidence.get("state_vector") or {}
    source = record.get("source_record") or {}
    kind = record.get("setup_kind") or vector.get("setup_kind") or "UNKNOWN"
    scenario = next((name for name, spec in SUB_SCENARIOS.items() if kind in spec["setup_kinds"]), "")
    return {
        "discovery_source": source.get("discovery_path_label") or source.get("preparation_origin") or "UNKNOWN",
        "route_ids": record.get("route_ids", ""),
        "trade_mode": TREND_PULLBACK if scenario else "UNASSIGNED",
        "local_setup": kind,
        "sub_scenario": scenario,
        "sub_scenario_label": SUB_SCENARIOS.get(scenario, {}).get("label", ""),
        "participation_permission": ("PLAN_REVIEW" if record.get("paper_plan_eligible")
                                     else record.get("state", "OBSERVE")),
        "executable": bool(scenario) and SUB_SCENARIOS.get(scenario, {}).get("status") != "MAPPED_NOT_ENABLED",
        "contract_version": VERSION,
    }


def evaluations(record):
    """One opportunity may carry several mode verdicts; they never merge into one permission."""
    fields = classify(record)
    rows = []
    for name, spec in SUB_SCENARIOS.items():
        active = fields["sub_scenario"] == name
        enabled = spec.get("status") != "MAPPED_NOT_ENABLED"
        rows.append({"mode": spec["mode"], "sub_scenario": name, "label": spec["label"],
                     "state": (fields["participation_permission"] if active and enabled else
                               "MAPPED_NOT_ENABLED" if active else "NOT_THIS_SETUP"),
                     "confirm": spec["confirm"], "invalidate": spec["invalidate"],
                     "executable": active and enabled})
    for mode, spec in OBSERVATION_ONLY.items():
        rows.append({"mode": mode, "sub_scenario": "", "label": spec["label"], "state": spec["status"],
                     "confirm": spec["requires"], "invalidate": "", "executable": False})
    return {"version": VERSION, "fields": fields, "evaluations": rows,
            "budget": "同股多模式共用同一资金与风险约束，不叠加不可见预算。",
            "priority": list(PRIORITY),
            "note": "某一模式信号成立不生成另一模式的计划；未知局部形态保持可见但不继承执行许可。"}


def conflict(plan_state):
    """Which concern wins when several apply at once."""
    for name in PRIORITY:
        if plan_state.get(name.lower()):
            return {"winner": name, "reason": {
                "ACTIVE_EXIT": "已生效退出优先，不被新模式接管",
                "HARD_RISK": "硬风险优先于任何新信号",
                "PENDING_REDUCE": "待减仓先于新增仓",
                "EXISTING_PLAN": "同股已有计划，新模式不另开预算",
                "NEW_MODE_SIGNAL": "无冲突，按新模式评估",
            }[name], "version": VERSION}
    return {"winner": "NONE", "reason": "无待处理项", "version": VERSION}
