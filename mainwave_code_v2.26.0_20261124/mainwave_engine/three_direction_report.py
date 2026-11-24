# -*- coding: utf-8 -*-
"""Unified stock-first, ignition-confirmation and propagation daily report."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from . import __version__ as ENGINE_VERSION
from . import candidate_value as CV
from . import carrier_quality as CQ
from . import phase_events as PE
from . import r2_stage_watch as R2
from . import stock_lifecycle as SL
from . import theme_risk as TR
from . import theme_semantics as TS


VERSION = "three_direction_report_v1_9_20260910"
WEEKLY_TABLE_VERSION = "weekly_candidate_table_v2_2_20260905"
OUT_ROOT = Path("results/mainwave_engine")
DYNAMIC_STORE = OUT_ROOT / "dynamic_theme_state_store.json"
DYNAMIC_HISTORY = OUT_ROOT / "dynamic_theme_lifecycle_history.csv"
MAX_MARKET_CANDIDATES = 15
MAX_THEME_CANDIDATES = 3

DISCOVERY_PATH_LABELS = {
    "LURK": "埋伏股",
    "IGNITION": "点火股",
    "PROPAGATION": "扩散股",
}
STOCK_LIFECYCLE_LABELS = {
    "PROBED": "资金试探",
    "ABSORBING": "吸筹",
    "COILED": "收敛蓄势",
    "ARMED": "临界准备",
    "READY": "待触发",
    "IGNITION": "点火启动",
    "ACCELERATION": "主升加速",
    "OVERHEATED": "过热",
    "DISTRIBUTION": "派发/破位",
}
THEME_LIFECYCLE_LABELS = {
    "SEED": "萌芽",
    "FERMENTING": "发酵",
    "IGNITED": "点火",
    "DIFFUSING": "扩散",
    "CROWDED": "拥挤",
    "RETREAT": "退潮",
    "DORMANT": "休眠",
    "ACTIVE": "活跃观察",
    "PREHEAT": "预热",
    "FORWARD": "前瞻观察",
}
ACTION_LABELS = {
    "FORWARD_WATCH": "前向观察",
    "MARKET_RISK_WATCH_ONLY": "市场风险观察",
    "BLOCKED_BY_THEME_RISK": "题材风险阻断",
    "WAIT_CARRIER": "等待扩散载体",
    "PROPAGATION_CONFIRMED": "扩散载体确认",
}
VALIDATION_LABELS = {
    "PASS": "已验证",
    "WATCH": "观察",
    "REJECT": "未通过",
    "UNVALIDATED": "未验证",
}
OBSERVATION_ACTION_LABELS = {
    "RISK_REVIEW": "主动风险复核",
    "NO_CHASE_REVIEW": "高潮期不追",
    "LPS_CONFIRMATION_WATCH": "LPS承接观察",
    "DEMAND_CONFIRMATION_WATCH": "需求扩张观察",
    "MARKUP_CONTEXT_ONLY": "已在拉升，仅作阶段标记",
    "NO_VALIDATED_EVENT": "无已验证新事件",
}


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _display_mapping(as_of: str) -> dict:
    """Use the current map for report labels only; scoring keeps point-in-time maps."""
    payload = _read_json(Path("concept_mapping.json"), {})
    mapping = payload.get("mapping", {})
    if isinstance(mapping, dict) and mapping:
        return TS.apply_membership_evidence(mapping, as_of)[0]
    historical, _ = TS.load_mapping_as_of(as_of)
    return historical


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def next_decision_date(as_of: str) -> str:
    return str((pd.Timestamp(as_of) + pd.offsets.BDay(1)).date())


def _hotspot_context(as_of: str) -> tuple[dict, str]:
    payload = _read_json(Path("hotspot_daily.json"), {})
    if str(payload.get("date", "")) != as_of.replace("-", ""):
        return {}, "STALE_OR_MISSING"
    return payload.get("concepts", {}), "FRESH"


def _market_risk_context(as_of: str) -> dict:
    from .market_context import load
    return {**load(as_of), "source": "market_context.json"}


def _symbol_theme_matches(as_of: str) -> tuple[dict[str, list[dict]], dict]:
    concepts, hotspot_status = _hotspot_context(as_of)
    mapping, mapping_meta = TS.load_mapping_as_of(as_of)
    aux_by_symbol, _, aux_meta = TS.load_aux_relationships()
    direct: dict[str, set[str]] = {}
    for theme, item in concepts.items():
        for symbol in item.get("stocks", []):
            direct.setdefault(symbol, set()).add(theme)
    rows: dict[str, list[dict]] = {}
    for symbol in set(mapping) | set(aux_by_symbol) | set(direct):
        point = set(mapping.get(symbol, {}).get("concepts", []))
        auxiliary = aux_by_symbol.get(symbol, set())
        matches = []
        for theme, item in concepts.items():
            if symbol in direct and theme in direct[symbol]:
                source = "HOTSPOT_LIMIT_MEMBER"
            elif theme in point:
                if theme in mapping.get(symbol, {}).get("_dynamic_concepts", []):
                    source = "DYNAMIC_HOT_MAPPING"
                else:
                    source = ("POINT_IN_TIME_MAPPING"
                              if mapping_meta.get("age_days", 999) <= 30
                              else "STALE_POINT_MAPPING")
            elif theme in auxiliary:
                source = "STALE_AUX_RELATION"
            else:
                continue
            matches.append({
                "theme": theme, "source": source,
                "heat": float(item.get("heat_score", 0) or 0),
                "limit_count": int(item.get("limit_count", 0) or 0),
                "lifecycle": str(item.get("lifecycle", "")),
                "generic": TS.is_generic_concept(theme),
            })
        source_order = {"HOTSPOT_LIMIT_MEMBER": 0, "DYNAMIC_HOT_MAPPING": 1,
                        "POINT_IN_TIME_MAPPING": 2, "STALE_POINT_MAPPING": 3,
                        "STALE_AUX_RELATION": 4}
        matches.sort(key=lambda row: (
            row["generic"], source_order[row["source"]],
            -row["limit_count"], -row["heat"], row["theme"]
        ))
        if matches:
            rows[symbol] = matches
    return rows, {
        "hotspot_status": hotspot_status, "mapping": mapping_meta,
        "semantic_auxiliary": aux_meta,
    }


def _related_themes(matches: list[dict]) -> str:
    specific = [str(row.get("theme", "")) for row in matches if not row.get("generic")]
    generic = [str(row.get("theme", "")) for row in matches if row.get("generic")]
    return "|".join(dict.fromkeys([*specific, *generic]))


def build_direction1(as_of: str, decision_date: str, output: Path,
                     theme_matches: dict[str, list[dict]], market_state: str = "GREEN") -> pd.DataFrame:
    frame = _read_csv(output / "carrier_qualification.csv")
    if not len(frame):
        return pd.DataFrame()
    frame = frame[frame["carrier_qualification"] == "QUALIFIED_LURK"].copy()
    rows = []
    for row in frame.to_dict("records"):
        matches = theme_matches.get(row["symbol"], [])
        best = matches[0] if matches else {}
        risk = TR.evaluate(best.get("theme", ""), decision_date)
        if risk["theme_risk_action"] == "BLOCK_NEW_ENTRY":
            structural_action = "BLOCKED_BY_THEME_RISK"
        elif risk["theme_risk_action"] == "REQUIRE_EXTERNAL_CONFIRMATION":
            structural_action = "WAIT_EXTERNAL_CONFIRMATION"
        elif best.get("source") in {
            "HOTSPOT_LIMIT_MEMBER", "DYNAMIC_HOT_MAPPING", "POINT_IN_TIME_MAPPING",
        }:
            structural_action = "THEME_CONFIRMED_WATCH"
        elif best:
            structural_action = "FORWARD_WATCH_THEME_UNVERIFIED"
        else:
            structural_action = "FORWARD_WATCH"
        if market_state == "RED":
            action = "BLOCKED_BY_MARKET_RISK"
        elif structural_action == "BLOCKED_BY_THEME_RISK":
            action = structural_action
        elif market_state in {"ORANGE", "UNKNOWN"}:
            action = "MARKET_RISK_WATCH_ONLY"
        else:
            action = structural_action
        rows.append({
            "as_of": as_of, "decision_date": decision_date,
            "symbol": row["symbol"], "name": row.get("name", ""),
            "discovery_route": "STOCK_FIRST",
            "theme": best.get("theme", ""), "theme_match_source": best.get("source", ""),
            "related_themes": _related_themes(matches),
            "theme_lifecycle": best.get("lifecycle", ""),
            "theme_heat": best.get("heat"), "theme_limit_count": best.get("limit_count"),
            "shadow_stage": row.get("shadow_stage", ""),
            "retention_state": row.get("retention_state", ""),
            "route": row.get("route", ""), "carrier_tier": row.get("carrier_tier", ""),
            "r5": row.get("r5"), "r20": row.get("r20"),
            "ma20_bias": row.get("ma20_bias"),
            "trigger_distance_pct": row.get("trigger_distance_pct"),
            "stop_distance_pct": row.get("stop_distance_pct"),
            "structural_action": structural_action, "action": action, **risk,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    action_order = {"THEME_CONFIRMED_WATCH": 0, "FORWARD_WATCH": 1,
                    "FORWARD_WATCH_THEME_UNVERIFIED": 2,
                    "WAIT_EXTERNAL_CONFIRMATION": 3, "MARKET_RISK_WATCH_ONLY": 4,
                    "BLOCKED_BY_THEME_RISK": 5, "BLOCKED_BY_MARKET_RISK": 6}
    stage_order = {"READY": 0, "ARMED": 1, "COILED": 2, "ABSORBING": 3}
    result = result.assign(
        _action=result["structural_action"].map(action_order).fillna(9),
        _stage=result["shadow_stage"].map(stage_order).fillna(9),
        _route=result["route"].map(CQ.ROUTE_ORDER).fillna(99),
    ).sort_values(
        ["_stage", "_action", "trigger_distance_pct", "stop_distance_pct", "_route", "symbol"],
        na_position="last",
    ).drop(columns=["_action", "_stage", "_route"])
    result["rank"] = range(1, len(result) + 1)
    return result


def build_direction2(as_of: str, decision_date: str, output: Path,
                     market_state: str = "GREEN") -> pd.DataFrame:
    source = _read_csv(output / "weekly_reverse_trace" / "pre_ignition_carriers.csv")
    if not len(source):
        return pd.DataFrame()
    source = source[source["role"].isin(["IGNITION_CARRIER", "EVE_QUALIFIED_LURK"])].copy()
    rows = []
    for row in source.to_dict("records"):
        risk = TR.evaluate(str(row.get("theme", "")), decision_date)
        current = row.get("current_qualification") == "QUALIFIED_LURK"
        if not current:
            structural_action = "HISTORICAL_EVIDENCE_ONLY"
        elif risk["theme_risk_action"] == "BLOCK_NEW_ENTRY":
            structural_action = "BLOCKED_BY_THEME_RISK"
        elif risk["theme_risk_action"] == "REQUIRE_EXTERNAL_CONFIRMATION":
            structural_action = "WAIT_EXTERNAL_CONFIRMATION"
        else:
            structural_action = "IGNITION_CONFIRMED_WATCH"
        if not current:
            action = structural_action
        elif market_state == "RED":
            action = "BLOCKED_BY_MARKET_RISK"
        elif structural_action == "BLOCKED_BY_THEME_RISK":
            action = structural_action
        elif market_state in {"ORANGE", "UNKNOWN"}:
            action = "MARKET_RISK_WATCH_ONLY"
        else:
            action = structural_action
        rows.append({
            "as_of": as_of, "decision_date": decision_date,
            "raw_theme": row.get("raw_theme", row.get("theme", "")),
            "theme": row.get("theme", ""), "t0": row.get("t0", ""),
            "symbol": row.get("symbol", ""), "name": row.get("name", ""),
            "role": row.get("role", ""), "first_probe": row.get("first_probe", ""),
            "probe_lead_sessions": row.get("probe_lead_sessions"),
            "first_qualified": row.get("first_qualified", ""),
            "qualified_lead_sessions": row.get("qualified_lead_sessions"),
            "pre_qualification": row.get("pre_qualification", ""),
            "current_qualification": row.get("current_qualification", ""),
            "bottleneck": row.get("bottleneck", ""),
            "structural_action": structural_action, "action": action, **risk,
        })
    result = pd.DataFrame(rows)
    action_order = {"IGNITION_CONFIRMED_WATCH": 0, "WAIT_EXTERNAL_CONFIRMATION": 1,
                    "BLOCKED_BY_THEME_RISK": 2, "MARKET_RISK_WATCH_ONLY": 3,
                    "BLOCKED_BY_MARKET_RISK": 4, "HISTORICAL_EVIDENCE_ONLY": 5}
    role_order = {"EVE_QUALIFIED_LURK": 0, "IGNITION_CARRIER": 1}
    result = result.assign(
        _action=result["structural_action"].map(action_order).fillna(9),
        _role=result["role"].map(role_order).fillna(9),
        _pre=(result["pre_qualification"] != "QUALIFIED_LURK").astype(int),
    ).sort_values(
        ["_action", "theme", "_pre", "_role", "qualified_lead_sessions", "symbol"],
        ascending=[True, True, True, True, False, True], na_position="last",
    )
    result = result.drop(columns=["_action", "_role", "_pre"])
    result["rank"] = range(1, len(result) + 1)
    return result


def _event_lifecycle(follow_through: str) -> str:
    return {"EXPANDED": "DIFFUSING", "HELD": "IGNITED", "PENDING": "IGNITED",
            "FADED": "RETREAT"}.get(str(follow_through), "IGNITED")


def _theme_action(lifecycle: str, current_carriers: int, risk: dict,
                  market_state: str, registered: bool = False) -> str:
    if market_state == "RED":
        return "BLOCKED_BY_MARKET_RISK"
    if risk["theme_risk_action"] == "BLOCK_NEW_ENTRY":
        return "BLOCKED_BY_THEME_RISK"
    if market_state in {"ORANGE", "UNKNOWN"}:
        return "MARKET_RISK_WATCH_ONLY"
    if risk["theme_risk_action"] == "REQUIRE_EXTERNAL_CONFIRMATION":
        return "WAIT_EXTERNAL_CONFIRMATION"
    if lifecycle == "RETREAT":
        return "WAIT_NEW_CYCLE"
    if current_carriers:
        return "PROPAGATION_CONFIRMED"
    return "WAIT_PROPAGATION_OR_CARRIER" if registered else "WAIT_CARRIER"


def _dynamic_memory(as_of: str, decision_date: str, current_rows: list[dict],
                    output: Path, market_state: str) -> list[dict]:
    previous = _read_json(DYNAMIC_STORE, {"_meta": {"as_of": ""}, "events": {}})
    previous_as_of = str(previous.get("_meta", {}).get("as_of", ""))
    if previous_as_of and as_of < previous_as_of:
        return current_rows
    events = dict(previous.get("events", {}))
    current_ids = set()
    for row in current_rows:
        event_id = f"{row['raw_theme']}|{row['t0']}"
        current_ids.add(event_id)
        events[event_id] = {**row, "event_id": event_id, "last_observed_as_of": as_of}

    hotspot, _ = _hotspot_context(as_of)
    quality = _read_csv(output / "carrier_qualification.csv")
    qualified = set(quality.loc[
        quality.get("carrier_qualification", pd.Series(dtype=str)) == "QUALIFIED_LURK", "symbol"
    ]) if len(quality) else set()
    active_rows = []
    retained_events = {}
    for event_id, stored in events.items():
        age = max(0, len(pd.bdate_range(stored.get("t0", as_of), as_of)) - 1)
        if age > 30:
            continue
        if event_id in current_ids:
            row = stored
        elif age <= 15:
            item = hotspot.get(stored.get("raw_theme", ""), hotspot.get(stored.get("theme", ""), {}))
            limit_count = int(item.get("limit_count", 0) or 0)
            heat = float(item.get("heat_score", 0) or 0)
            threshold = int(stored.get("ignition_threshold", 3) or 3)
            prior_lifecycle = stored.get("lifecycle", "IGNITED")
            if limit_count >= max(2, threshold - 1):
                lifecycle = "DIFFUSING"
            elif heat >= 25 or (age <= 3 and prior_lifecycle != "RETREAT"):
                lifecycle = "IGNITED"
            else:
                lifecycle = "RETREAT"
            members = [value for value in str(stored.get("member_symbols", "")).split("|") if value]
            current = sorted(set(members) & qualified)
            risk = TR.evaluate(str(stored.get("theme", "")), decision_date)
            structural_action = _theme_action(lifecycle, len(current), risk, "GREEN")
            row = {
                **stored, "as_of": as_of, "decision_date": decision_date,
                "source": "DYNAMIC_CHAIN_MEMORY", "lifecycle": lifecycle,
                "follow_through": "CARRIED_FORWARD", "limit_count_current": limit_count,
                "current_qualified_carriers": len(current),
                "current_carrier_symbols": "|".join(current[:3]),
                "structural_action": structural_action,
                "action": _theme_action(lifecycle, len(current), risk, market_state), **risk,
            }
            events[event_id] = {**row, "event_id": event_id}
        else:
            row = None
        retained_events[event_id] = events[event_id]
        if row is not None:
            active_rows.append(row)

    _atomic_json(DYNAMIC_STORE, {
        "_meta": {"as_of": as_of, "version": VERSION}, "events": retained_events,
    })
    history_rows = [{
        "as_of": as_of, "event_id": row.get("event_id", ""),
        "raw_theme": row.get("raw_theme", ""), "theme": row.get("theme", ""),
        "t0": row.get("t0", ""), "lifecycle": row.get("lifecycle", ""),
        "information_leader": row.get("information_leader", ""),
        "current_qualified_carriers": row.get("current_qualified_carriers", 0),
        "action": row.get("action", ""), "source": row.get("source", ""),
    } for row in active_rows]
    history = pd.DataFrame(history_rows)
    if DYNAMIC_HISTORY.exists():
        old = _read_csv(DYNAMIC_HISTORY)
        if len(old):
            old = old[old["as_of"] != as_of]
            history = pd.concat([old, history], ignore_index=True)
    if len(history):
        history.sort_values(["as_of", "event_id"]).to_csv(
            DYNAMIC_HISTORY, index=False, encoding="utf-8-sig"
        )
    return active_rows


def build_direction3(as_of: str, decision_date: str, output: Path,
                     market_state: str = "GREEN") -> pd.DataFrame:
    events = _read_csv(output / "weekly_reverse_trace" / "weekly_theme_events.csv")
    carriers = _read_csv(output / "weekly_reverse_trace" / "pre_ignition_carriers.csv")
    mapping, _ = TS.load_mapping_as_of(as_of)
    point_members = TS.members_by_concept(mapping)
    quality = _read_csv(output / "carrier_qualification.csv")
    qualified = set(quality.loc[
        quality.get("carrier_qualification", pd.Series(dtype=str)) == "QUALIFIED_LURK",
        "symbol",
    ]) if len(quality) else set()
    rows = []
    if len(events):
        monitored = events[events["event_type"].isin([
            "NEW_IGNITION_30D", "REIGNITION", "EXPANSION",
        ])]
        for event in monitored.to_dict("records"):
            raw_theme = event.get("theme", "")
            theme = event.get("resolved_theme", raw_theme)
            group = carriers[
                (carriers.get("raw_theme", carriers.get("theme")) == raw_theme)
                & (carriers["t0"] == event["date"])
            ] if len(carriers) else carriers
            ignition = group[group["role"] == "IGNITION_CARRIER"] if len(group) else group
            if len(ignition):
                leader_pick = ignition.assign(
                    _pre=(ignition["pre_qualification"] != "QUALIFIED_LURK").astype(int)
                ).sort_values(["_pre", "event_day_pct", "symbol"],
                              ascending=[True, False, True], na_position="last").iloc[0]
                leader = leader_pick.get("symbol", "")
            else:
                leader = ""
            if len(group):
                current_symbols = sorted(set(group.loc[
                    group["current_qualification"] == "QUALIFIED_LURK", "symbol"
                ]))
                member_symbols = sorted(set(group["symbol"]))
            else:
                direct_members = {
                    value for value in str(event.get("limit_stocks", "")).split("|") if value
                }
                member_symbols = sorted(
                    point_members.get(str(raw_theme), set())
                    | point_members.get(str(theme), set())
                    | direct_members
                )
                current_symbols = sorted(set(member_symbols) & qualified)
            lifecycle = _event_lifecycle(event.get("follow_through", ""))
            risk = TR.evaluate(str(theme), decision_date)
            source = ("WEEKLY_EXPANSION_CHAIN"
                      if event.get("event_type") == "EXPANSION"
                      else "WEEKLY_DYNAMIC_CHAIN")
            structural_action = _theme_action(lifecycle, len(current_symbols), risk, "GREEN")
            action = _theme_action(lifecycle, len(current_symbols), risk, market_state)
            if source == "WEEKLY_EXPANSION_CHAIN":
                if structural_action in {"PROPAGATION_CONFIRMED", "WAIT_CARRIER"}:
                    structural_action = "EXPANSION_MONITOR_NO_CHASE"
                if action in {"PROPAGATION_CONFIRMED", "WAIT_CARRIER"}:
                    action = "EXPANSION_MONITOR_NO_CHASE"
            rows.append({
                "as_of": as_of, "decision_date": decision_date,
                "raw_theme": raw_theme, "theme": theme, "source": source,
                "semantic_status": event.get("semantic_status", "RAW_THEME"),
                "lifecycle": lifecycle, "t0": event.get("date", ""),
                "event_type": event.get("event_type", ""),
                "follow_through": event.get("follow_through", ""),
                "information_leader": leader,
                "limit_count_t0": event.get("limit_count"),
                "post_week_max": event.get("post_week_max"),
                "current_qualified_carriers": len(current_symbols),
                "current_carrier_symbols": "|".join(current_symbols[:3]),
                "member_symbols": "|".join(member_symbols),
                "ignition_threshold": event.get("ignition_threshold", 3),
                "structural_action": structural_action, "action": action, **risk,
            })
    rows = _dynamic_memory(as_of, decision_date, rows, output, market_state)
    registered = _read_csv(output / "leader_state_daily.csv")
    episodes = _read_csv(output / "theme_episodes.csv")
    if len(registered):
        episode_by_id = {row["theme_id"]: row for row in episodes.to_dict("records")} \
            if len(episodes) else {}
        for row in registered.to_dict("records"):
            episode = episode_by_id.get(row.get("theme_id"), {})
            lifecycle = row.get("theme_lifecycle", episode.get("theme_lifecycle", ""))
            risk = TR.evaluate(str(row.get("theme_label", "")), decision_date)
            confirmed = (int(row.get("qualified_carriers", 0) or 0)
                         if row.get("leader_state") == "LEADER_CONFIRMED" else 0)
            structural_action = _theme_action(
                lifecycle, confirmed, risk, "GREEN", registered=True
            )
            action = _theme_action(lifecycle, confirmed, risk, market_state, registered=True)
            rows.append({
                "as_of": as_of, "decision_date": decision_date,
                "raw_theme": row.get("theme_label", ""), "theme": row.get("theme_label", ""),
                "source": "REGISTERED_CHAIN", "semantic_status": "REGISTERED_TIGHT_CHAIN",
                "lifecycle": lifecycle, "t0": "", "event_type": "",
                "follow_through": "", "information_leader": row.get("leader", ""),
                "limit_count_t0": None, "post_week_max": None,
                "current_qualified_carriers": row.get("qualified_carriers", 0),
                "current_carrier_symbols": "", "structural_action": structural_action,
                "action": action, **risk,
            })
    horizon = _read_csv(output / "theme_horizon.csv")
    if len(horizon):
        for row in horizon.to_dict("records"):
            label = str(row.get("theme_label", ""))
            risk = TR.evaluate(label, decision_date)
            priority = str(row.get("priority_band", "CALENDAR_WATCH"))
            carrier_value = row.get("carrier_count", 0)
            carrier_count = int(carrier_value) if pd.notna(carrier_value) else 0
            if risk["theme_risk_action"] == "BLOCK_NEW_ENTRY":
                structural_action = "BLOCKED_BY_THEME_RISK"
            elif risk["theme_risk_action"] == "REQUIRE_EXTERNAL_CONFIRMATION":
                structural_action = "WAIT_EXTERNAL_CONFIRMATION"
            else:
                structural_action = {
                    "NOW_FOCUS": "HORIZON_NOW_FOCUS",
                    "BUILD_WATCH": "HORIZON_BUILD_WATCH",
                    "CALENDAR_WATCH": "CALENDAR_ONLY_WAIT",
                    "REPAIR_ONLY": "REPAIR_REQUIRED",
                }.get(priority, "CALENDAR_ONLY_WAIT")
            if market_state == "RED":
                action = "BLOCKED_BY_MARKET_RISK"
            elif structural_action == "BLOCKED_BY_THEME_RISK":
                action = structural_action
            elif market_state in {"ORANGE", "UNKNOWN"}:
                action = "MARKET_RISK_WATCH_ONLY"
            else:
                action = structural_action
            rows.append({
                "as_of": as_of, "decision_date": decision_date,
                "raw_theme": label, "theme": label, "source": "THEME_HORIZON",
                "semantic_status": "CALENDAR_RESEARCH_CONTEXT",
                "lifecycle": row.get("calendar_state", ""), "t0": "",
                "event_type": "", "follow_through": "",
                "information_leader": "", "limit_count_t0": None,
                "post_week_max": None,
                "current_qualified_carriers": carrier_count,
                "current_carrier_symbols": "", "structural_action": structural_action,
                "action": action,
                "horizon_priority": priority,
                "research_priority": row.get("research_priority", ""),
                "external_catalyst_state": row.get("external_catalyst_state", ""),
                "catalyst": row.get("catalyst", ""), **risk,
            })
    result = pd.DataFrame(rows)
    if not len(result):
        return result
    action_order = {"PROPAGATION_CONFIRMED": 0, "WAIT_CARRIER": 1,
                    "HORIZON_NOW_FOCUS": 2, "HORIZON_BUILD_WATCH": 3,
                    "WAIT_PROPAGATION_OR_CARRIER": 4, "CALENDAR_ONLY_WAIT": 5,
                    "WAIT_EXTERNAL_CONFIRMATION": 6, "BLOCKED_BY_THEME_RISK": 7,
                    "EXPANSION_MONITOR_NO_CHASE": 8, "WAIT_NEW_CYCLE": 9,
                    "REPAIR_REQUIRED": 10, "MARKET_RISK_WATCH_ONLY": 11,
                    "BLOCKED_BY_MARKET_RISK": 12}
    lifecycle_order = {"DIFFUSING": 0, "IGNITED": 1, "FERMENTING": 2,
                       "SEED": 3, "CROWDED": 4, "RETREAT": 5, "DORMANT": 6}
    research_order = {"HIGH": 0, "MEDIUM": 1, "BASE": 2, "REPAIR": 3}
    return result.assign(
        _action=result["structural_action"].map(action_order).fillna(9),
        _life=result["lifecycle"].map(lifecycle_order).fillna(9),
        _research=result.get("research_priority", pd.Series(index=result.index, dtype=str))
        .map(research_order).fillna(9),
    ).sort_values(["_action", "_research", "_life", "theme"]).drop(
        columns=["_action", "_research", "_life"]
    )


def _as_float(value, default=None):
    try:
        number = float(value)
        return number if pd.notna(number) else default
    except (TypeError, ValueError):
        return default


def _stock_metrics(symbol: str, as_of: str, fallback: dict | None = None) -> dict:
    fallback = fallback or {}
    path = Path("data") / f"{symbol}_daily.csv"
    frame = _read_csv(path)
    if len(frame) and {"date", "close"}.issubset(frame.columns):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame = frame[(frame["date"] <= pd.Timestamp(as_of)) & frame["close"].notna()]
        frame = frame.sort_values("date")
    if len(frame):
        close = frame["close"]
        latest = float(close.iloc[-1])
        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        return {
            "close": latest,
            "ema20_bias": (latest / ema20 - 1) * 100 if ema20 else None,
            "r5": (latest / float(close.iloc[-6]) - 1) * 100 if len(close) >= 6 else None,
            "r20": (latest / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else None,
        }
    return {
        "close": _as_float(fallback.get("close")),
        "ema20_bias": None,
        "r5": _as_float(fallback.get("r5", fallback.get("rise_5d"))),
        "r20": _as_float(fallback.get("r20", fallback.get("rise_20d"))),
    }


def _phase_context(symbol: str, as_of: str) -> dict:
    snapshot = PE.latest_snapshot(symbol, as_of)
    return {
        "broad_phase": snapshot.get("broad_phase", ""),
        "broad_phase_label": snapshot.get("broad_phase_label", ""),
        "phase_validation": snapshot.get("phase_validation", ""),
        "latest_event": snapshot.get("latest_event", ""),
        "latest_event_label": snapshot.get("latest_event_label", ""),
        "latest_event_date": snapshot.get("latest_event_date", ""),
        "event_validation": snapshot.get("event_validation", ""),
        "effort_result": snapshot.get("effort_result", ""),
        "effort_result_label": snapshot.get("effort_result_label", ""),
        "effort_validation": snapshot.get("effort_validation", ""),
        "phase_observation_action": snapshot.get("observation_action", ""),
    }


def _stage_context(as_of: str) -> dict[str, dict]:
    path = OUT_ROOT / as_of.replace("-", "") / "stock_lifecycle.csv"
    frame = _read_csv(path)
    return {str(row["symbol"]): row for row in frame.to_dict("records")} \
        if len(frame) and "symbol" in frame.columns else {}


def _canonical_stock_lifecycle(stage: dict, metrics: dict,
                               discovery_path: str, shadow_stage: str = "") -> str:
    if shadow_stage in STOCK_LIFECYCLE_LABELS:
        return shadow_stage
    stage_code = str(stage.get("mainwave_stage", ""))
    mapped = {
        "S0_base_watch": "PROBED",
        "S1_ignition_entry": "IGNITION",
        "S2_acceleration_caution": "ACCELERATION",
        "S3_overheated_no_chase": "OVERHEATED",
        "S4_breakdown_or_distribution": "DISTRIBUTION",
    }.get(stage_code)
    if mapped:
        return mapped
    if discovery_path not in {"IGNITION", "PROPAGATION"}:
        return "PROBED"
    r5 = _as_float(metrics.get("r5"), 0.0)
    r20 = _as_float(metrics.get("r20"), 0.0)
    ema_bias = _as_float(metrics.get("ema20_bias"), 0.0)
    if r20 <= -12 or ema_bias <= -8:
        return "DISTRIBUTION"
    if r20 >= 30 or ema_bias >= 25 or r5 >= 22:
        return "OVERHEATED"
    if r20 >= 12 or ema_bias >= 12 or r5 >= 10:
        return "ACCELERATION"
    return "IGNITION"


def _belonging_theme(symbol: str, explicit_theme: str, theme_source: str,
                     mapping: dict, stage: dict) -> tuple[str, str]:
    if explicit_theme:
        return explicit_theme, theme_source or "ACTIVE_THEME"
    external = str(stage.get("external_theme", "") or "")
    if external:
        return external, "STAGE_EXTERNAL_THEME"
    mapped = mapping.get(symbol, {})
    industry = str(mapped.get("industry", "") or stage.get("industry", "") or "")
    if industry:
        return industry, "CURRENT_INDUSTRY_DISPLAY"
    concepts = mapped.get("concepts", [])
    if concepts:
        return str(concepts[0]), "CURRENT_CONCEPT_DISPLAY"
    return "待确认", "MISSING"


def _weekly_note(discovery_path: str, theme_lifecycle: str, theme_source: str,
                 trigger_distance, action: str) -> str:
    if discovery_path == "IGNITION":
        if theme_lifecycle == "RETREAT":
            return "点火后题材退潮，当前无合格载体"
        if theme_lifecycle == "DIFFUSING":
            return "点火领头股；题材扩散不等于该股仍是买点"
        return "当周点火领头股，等待持续性确认"
    if discovery_path == "PROPAGATION":
        return "点火后紧链传播确认的合格载体"
    if theme_source in {"STALE_POINT_MAPPING", "STALE_AUX_RELATION"}:
        return "题材关系陈旧，待重新验证"
    distance = _as_float(trigger_distance)
    if distance is not None and distance < 0:
        return "已越过触发位，等待回踩确认"
    if theme_source in {
        "CURRENT_INDUSTRY_DISPLAY", "CURRENT_CONCEPT_DISPLAY", "MISSING"
    }:
        return "仅个股资金结构，尚未形成题材交汇"
    if action == "MARKET_RISK_WATCH_ONLY":
        return "结构成立，市场风险仅允许观察"
    return "等待个股触发与题材确认"


def build_weekly_candidates(as_of: str, direction1: pd.DataFrame,
                            direction3: pd.DataFrame) -> pd.DataFrame:
    mapping = _display_mapping(as_of)
    stages = _stage_context(as_of)
    rows: list[dict] = []
    used: set[str] = set()

    if len(direction3):
        ignition = direction3[
            direction3.get("source", pd.Series(index=direction3.index, dtype=str)).eq(
                "WEEKLY_DYNAMIC_CHAIN"
            ) & direction3.get(
                "information_leader", pd.Series(index=direction3.index, dtype=str)
            ).fillna("").ne("")
        ]
        for item in ignition.to_dict("records"):
            symbol = str(item.get("information_leader", ""))
            if not symbol or symbol in used:
                continue
            stage = stages.get(symbol, {})
            mapped = mapping.get(symbol, {})
            metrics = _stock_metrics(symbol, as_of, stage)
            phase = _phase_context(symbol, as_of)
            lifecycle = _canonical_stock_lifecycle(stage, metrics, "IGNITION")
            theme = str(item.get("theme", "") or item.get("raw_theme", ""))
            theme_lifecycle = str(item.get("lifecycle", ""))
            action = str(item.get("action", ""))
            rows.append({
                "as_of": as_of, "symbol": symbol,
                "name": stage.get("name", mapped.get("name", "")),
                "close": metrics["close"], "ema20_bias": metrics["ema20_bias"],
                "r5": metrics["r5"], "r20": metrics["r20"],
                "belonging_theme": theme, "theme_source": "WEEKLY_IGNITION",
                "related_themes": theme,
                "discovery_path": "IGNITION",
                "discovery_path_label": DISCOVERY_PATH_LABELS["IGNITION"],
                "stock_lifecycle": lifecycle,
                "stock_lifecycle_label": STOCK_LIFECYCLE_LABELS[lifecycle],
                "theme_lifecycle": theme_lifecycle,
                "theme_lifecycle_label": THEME_LIFECYCLE_LABELS.get(
                    theme_lifecycle, theme_lifecycle or "未确认"
                ),
                "trigger_distance_pct": None, "stop_distance_pct": None,
                "action": action,
                **phase,
                "note": _weekly_note(
                    "IGNITION", theme_lifecycle, "WEEKLY_IGNITION", None, action
                ),
            })
            used.add(symbol)

        confirmed = direction3[
            direction3.get(
                "structural_action", pd.Series(index=direction3.index, dtype=str)
            ).eq("PROPAGATION_CONFIRMED")
            & direction3.get(
                "event_type", pd.Series(index=direction3.index, dtype=str)
            ).fillna("").ne("EXPANSION")
        ]
        for item in confirmed.to_dict("records"):
            leader = str(item.get("information_leader", ""))
            for symbol in str(item.get("current_carrier_symbols", "")).split("|"):
                symbol = symbol.strip()
                if not symbol or symbol == leader or symbol in used:
                    continue
                stage = stages.get(symbol, {})
                mapped = mapping.get(symbol, {})
                metrics = _stock_metrics(symbol, as_of, stage)
                phase = _phase_context(symbol, as_of)
                lifecycle = _canonical_stock_lifecycle(stage, metrics, "PROPAGATION")
                theme_lifecycle = str(item.get("lifecycle", ""))
                action = str(item.get("action", ""))
                rows.append({
                    "as_of": as_of, "symbol": symbol,
                    "name": stage.get("name", mapped.get("name", "")),
                    "close": metrics["close"], "ema20_bias": metrics["ema20_bias"],
                    "r5": metrics["r5"], "r20": metrics["r20"],
                    "belonging_theme": item.get("theme", ""),
                    "theme_source": "PROPAGATION_CONFIRMED",
                    "related_themes": item.get("theme", ""),
                    "discovery_path": "PROPAGATION",
                    "discovery_path_label": DISCOVERY_PATH_LABELS["PROPAGATION"],
                    "stock_lifecycle": lifecycle,
                    "stock_lifecycle_label": STOCK_LIFECYCLE_LABELS[lifecycle],
                    "theme_lifecycle": theme_lifecycle,
                    "theme_lifecycle_label": THEME_LIFECYCLE_LABELS.get(
                        theme_lifecycle, theme_lifecycle or "未确认"
                    ),
                    "trigger_distance_pct": None, "stop_distance_pct": None,
                    "action": action,
                    **phase,
                    "note": _weekly_note(
                        "PROPAGATION", theme_lifecycle, "PROPAGATION_CONFIRMED", None, action
                    ),
                })
                used.add(symbol)

    if len(direction1):
        for item in direction1.to_dict("records"):
            symbol = str(item.get("symbol", ""))
            if not symbol or symbol in used:
                continue
            stage = stages.get(symbol, {})
            metrics = _stock_metrics(symbol, as_of, item)
            phase = _phase_context(symbol, as_of)
            lifecycle = _canonical_stock_lifecycle(
                stage, metrics, "LURK", str(item.get("shadow_stage", ""))
            )
            theme, theme_source = _belonging_theme(
                symbol, str(item.get("theme", "") or ""),
                str(item.get("theme_match_source", "") or ""), mapping, stage
            )
            theme_lifecycle = str(item.get("theme_lifecycle", "") or "")
            action = str(item.get("action", ""))
            rows.append({
                "as_of": as_of, "symbol": symbol,
                "name": item.get("name", mapping.get(symbol, {}).get("name", "")),
                "close": metrics["close"], "ema20_bias": metrics["ema20_bias"],
                "r5": metrics["r5"], "r20": metrics["r20"],
                "belonging_theme": theme, "theme_source": theme_source,
                "related_themes": item.get("related_themes", theme),
                "discovery_path": "LURK",
                "discovery_path_label": DISCOVERY_PATH_LABELS["LURK"],
                "stock_lifecycle": lifecycle,
                "stock_lifecycle_label": STOCK_LIFECYCLE_LABELS[lifecycle],
                "theme_lifecycle": theme_lifecycle,
                "theme_lifecycle_label": THEME_LIFECYCLE_LABELS.get(
                    theme_lifecycle, theme_lifecycle or "未交汇"
                ),
                "trigger_distance_pct": _as_float(item.get("trigger_distance_pct")),
                "stop_distance_pct": _as_float(item.get("stop_distance_pct")),
                "action": action,
                **phase,
                "note": _weekly_note(
                    "LURK", theme_lifecycle, theme_source,
                    item.get("trigger_distance_pct"), action
                ),
            })
            used.add(symbol)

    result = pd.DataFrame(rows, columns=list(rows[0]) if rows else ["as_of", "symbol", "name", "discovery_path"])
    if len(result):
        result.insert(0, "rank", range(1, len(result) + 1))
        for column in ["close", "ema20_bias", "r5", "r20",
                       "trigger_distance_pct", "stop_distance_pct"]:
            result[column] = pd.to_numeric(result[column], errors="coerce").round(2)
    return result


def write_weekly_candidates(as_of: str, output: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(output / "mainwave_weekly_candidates.csv", index=False, encoding="utf-8-sig")
    counts = frame.get("discovery_path", pd.Series(dtype=str)).value_counts()
    start = str((pd.Timestamp(as_of) - pd.offsets.BDay(4)).date())
    display = frame.copy()
    if len(display):
        display["股票"] = display["name"].fillna("") + " " + display["symbol"].fillna("")
        display["发现路径"] = display["discovery_path_label"] + "(" + display["discovery_path"] + ")"
        display["主升浪阶段"] = (
            display["stock_lifecycle_label"] + "(" + display["stock_lifecycle"] + ")"
        )
        display["题材阶段"] = display["theme_lifecycle_label"]
        display["当日量价形态"] = display.apply(
            lambda row: (
                f"{row.get('broad_phase_label', '')}"
                f"[{VALIDATION_LABELS.get(row.get('phase_validation', ''), '未验证')}]"
            ) if row.get("broad_phase_label") else "未确认",
            axis=1,
        )
        display["关键事件"] = display.apply(
            lambda row: (
                f"{row.get('latest_event_label')}@{row.get('latest_event_date')}"
                f"[{VALIDATION_LABELS.get(row.get('event_validation', ''), '未验证')}]"
            ) if row.get("latest_event") else "无已验证事件",
            axis=1,
        )
        display["量价关系"] = display.apply(
            lambda row: (
                f"{row.get('effort_result_label')}"
                f"[{VALIDATION_LABELS.get(row.get('effort_validation', ''), '未验证')}]"
            ) if row.get("effort_validation") in {"PASS", "WATCH"} else "无已验证关系",
            axis=1,
        )
        display["事件观察"] = display["phase_observation_action"].map(
            OBSERVATION_ACTION_LABELS).fillna("无已验证新事件")
        display["当前动作"] = display["action"].map(ACTION_LABELS).fillna(display["action"])
        display["关联题材"] = display["related_themes"].fillna("").map(
            lambda value: "|".join(str(value).split("|")[:5])
        )
        display = display.rename(columns={
            "close": "最新价", "ema20_bias": "EMA20偏离%", "r5": "5日涨幅%",
            "r20": "20日涨幅%", "belonging_theme": "所属题材",
            "trigger_distance_pct": "距触发%", "note": "备注",
        })
        display["连续生命周期"] = display.get("cycle_stage_label", "未重建")
        display["本轮启动日"] = display.get("cycle_started_at", "")
        display["阶段变更原因"] = display.get("cycle_reason_label", "")
        display["换手率%"] = display.get("turnover_pct")
        display["相对换手x"] = display.get("turnover_relative20")
        display["换手解释"] = display.get("turnover_context_label", "未分析")
        display["价格速度"] = display.get("speed_label", "未分析")
        display["回踩或整理"] = display.get("setup_label", "")
        display["子形态状态"] = display.get("setup_status_label", "")
        for column in ["换手率%", "相对换手x"]:
            display[column] = pd.to_numeric(display[column], errors="coerce").round(2)
        shown = display[[
            "rank", "股票", "最新价", "EMA20偏离%", "5日涨幅%", "20日涨幅%",
            "所属题材", "关联题材", "发现路径", "连续生命周期", "本轮启动日", "阶段变更原因",
            "当日量价形态", "换手率%", "相对换手x", "换手解释", "价格速度", "回踩或整理", "子形态状态", "关键事件", "量价关系",
            "题材阶段", "距触发%", "事件观察", "当前动作", "备注",
        ]]
        table = shown.to_markdown(index=False)
    else:
        table = "本周没有通过候选口径的股票。"
    lines = [
        f"# 本周主升浪候选表  {start} 至 {as_of}", "",
        f"- 版本：`{WEEKLY_TABLE_VERSION}` / engine `{ENGINE_VERSION}`。",
        "- 发现路径回答股票为何进入表：埋伏股、点火股或扩散股。",
        "- 连续生命周期保留本轮启动、整理和失效历史；当日量价形态不是生命周期重置。",
        "- 全部已跟踪标的（含今日落选）见 stock_lifecycle.md；逐轮变更见 stock_lifecycle_events.csv。",
        "- 回踩/平台子记录见 price_cycle_setups.csv；价格速度、子形态不改写生产权限，均为待校准观察。",
        f"- 本周：埋伏股 {int(counts.get('LURK', 0))} 只，点火股 "
        f"{int(counts.get('IGNITION', 0))} 只，扩散股 {int(counts.get('PROPAGATION', 0))} 只。",
        "", table, "",
        "说明：题材处于DIFFUSING不代表该股票就是扩散股；只有点火后紧链传播确认的合格载体才标记为扩散股。",
        "所属题材使用当前映射补全，仅用于报表展示，不参与历史点火、扩散或排序判定。",
        "当日量价形态与关键事件沿用历史验证口径；连续生命周期和换手解释为待校准观察，不新增买入许可。",
        "量价重建启动日不等于当时生产命中；只有有日期的生产TRIGGERED证据用于防止重新标为早期买点。",
    ]
    (output / "mainwave_weekly_candidates.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _table(frame: pd.DataFrame, columns: list[str], empty: str) -> str:
    if not len(frame):
        return empty
    existing = [column for column in columns if column in frame.columns]
    return frame[existing].to_markdown(index=False)


def run(as_of: str, decision_date: str = "") -> int:
    output = OUT_ROOT / as_of.replace("-", "")
    decision_date = decision_date or next_decision_date(as_of)
    theme_matches, data_meta = _symbol_theme_matches(as_of)
    market_risk = _market_risk_context(as_of)
    direction1 = build_direction1(as_of, decision_date, output, theme_matches,
                                  market_risk["state"])
    direction2 = build_direction2(as_of, decision_date, output, market_risk["state"])
    direction3 = build_direction3(as_of, decision_date, output, market_risk["state"])
    direction1.to_csv(output / "direction1_forward_lurks.csv", index=False, encoding="utf-8-sig")
    direction2.to_csv(output / "direction2_ignition_confirmation.csv", index=False, encoding="utf-8-sig")
    direction3.to_csv(output / "direction3_theme_propagation.csv", index=False, encoding="utf-8-sig")
    discovery = build_weekly_candidates(as_of, direction1, direction3)
    discovery["route_ids"] = discovery.discovery_path.map({"IGNITION": "R1", "LURK": "R2", "PROPAGATION": "R3"}).fillna("")
    discovery["route_source"] = "QUALIFICATION_CONTEXT"
    routes = _read_csv(output / "route_candidates.csv")
    route_summary = _read_json(output / "route_scan_summary.json", {})
    if route_summary.get("as_of") == as_of and len(routes):
        # New price-first observations must not disappear behind legacy qualification gates.
        original = discovery.set_index("symbol")
        observed = routes.set_index("symbol")
        for symbol in observed.index.intersection(original.index):
            if observed.at[symbol, "theme_source"] == "IDENTITY_ONLY_NOT_CATALYST":
                for field in ("belonging_theme", "related_themes", "theme_source", "theme_lifecycle", "theme_lifecycle_label"):
                    if pd.notna(original.at[symbol, field]) and original.at[symbol, field] != "":
                        observed.at[symbol, field] = original.at[symbol, field]
        order = list(observed.index) + [s for s in original.index if s not in observed.index]
        discovery = observed.combine_first(original).reindex(order).reset_index()
        for field in ("theme_lifecycle_label", "phase_observation_action", "latest_event", "latest_event_label",
                      "latest_event_date", "event_validation", "phase_validation", "broad_phase_label",
                      "effort_validation", "effort_result_label", "related_themes", "note"):
            discovery[field] = discovery.get(field, pd.Series(index=discovery.index, dtype=str)).fillna("")
        discovery["rank"] = range(1, len(discovery) + 1)
    discovery.to_csv(output / "mainwave_discovery_candidates.csv", index=False, encoding="utf-8-sig")
    # Lifecycle tracking sees every route candidate; display limits are applied only afterwards.
    discovery = SL.write_daily(as_of, discovery, output, root=OUT_ROOT)
    weekly = discovery.head(MAX_MARKET_CANDIDATES).copy()
    write_weekly_candidates(as_of, output, weekly)
    value_frame, value_summary = CV.write_candidate_value(
        as_of, output, weekly, market_risk["state"]
    )
    slow_carriers = _read_csv(output / "slow_carrier_watch.csv")
    slow_summary = _read_json(output / "slow_carrier_summary.json", {})
    mean_reversion = _read_csv(output / "mean_reversion_shadow.csv")
    mean_summary = _read_json(output / "mean_reversion_shadow_summary.json", {})

    d1_actionable = int(direction1["action"].isin(
        ["THEME_CONFIRMED_WATCH", "FORWARD_WATCH",
         "FORWARD_WATCH_THEME_UNVERIFIED"]).sum()) if len(direction1) else 0
    d1_structural = int(direction1["structural_action"].isin(
        ["THEME_CONFIRMED_WATCH", "FORWARD_WATCH",
         "FORWARD_WATCH_THEME_UNVERIFIED"]).sum()) if len(direction1) else 0
    d2_current = int((direction2["current_qualification"] == "QUALIFIED_LURK").sum()) \
        if len(direction2) else 0
    d3_confirmed = int((direction3["action"] == "PROPAGATION_CONFIRMED").sum()) \
        if len(direction3) else 0
    d3_structural_confirmed = int(
        (direction3["structural_action"] == "PROPAGATION_CONFIRMED").sum()
    ) if len(direction3) else 0
    weekly_counts = weekly.get("discovery_path", pd.Series(dtype=str)).value_counts()
    weekly_events = weekly.get("latest_event", pd.Series(dtype=str)).fillna("")
    weekly_observations = weekly.get(
        "phase_observation_action", pd.Series(dtype=str)).fillna("")
    summary = {
        "as_of": as_of, "decision_date": decision_date, "version": VERSION,
        "engine_version": ENGINE_VERSION, "data": data_meta, "market_risk": market_risk,
        "routes": route_summary.get("routes", {}), "route_scan": route_summary,
        "discovery_candidates": {"total_before_display": len(discovery), "display_limit": MAX_MARKET_CANDIDATES,
                                 "source": "mainwave_discovery_candidates.csv"},
        "direction1": {"total": len(direction1), "actionable_watch": d1_actionable,
                       "structural_watch": d1_structural},
        "direction2": {"total": len(direction2), "current_confirmed": d2_current,
                       "historical_only": int((direction2["action"] == "HISTORICAL_EVIDENCE_ONLY").sum())
                       if len(direction2) else 0},
        "direction3": {"total": len(direction3), "propagation_confirmed": d3_confirmed,
                       "structural_propagation_confirmed": d3_structural_confirmed,
                       "expansion_monitor": int(direction3.get(
                           "event_type", pd.Series(index=direction3.index, dtype=str)
                       ).eq("EXPANSION").sum())
                       if len(direction3) else 0,
                       "wait_carrier": int(direction3["structural_action"].isin(
                           ["WAIT_CARRIER", "WAIT_PROPAGATION_OR_CARRIER"]).sum())
                       if len(direction3) else 0,
                       "horizon_watch": int((direction3["source"] == "THEME_HORIZON").sum())
                       if len(direction3) else 0},
        "weekly_candidates": {
            "total": len(weekly),
            "lurk": int(weekly_counts.get("LURK", 0)),
            "ignition": int(weekly_counts.get("IGNITION", 0)),
            "propagation": int(weekly_counts.get("PROPAGATION", 0)),
            "validated_events": int(weekly_events.ne("").sum()),
            "lps": int(weekly_events.eq("LPS").sum()),
            "utad": int(weekly_events.eq("UTAD").sum()),
            "demand_confirmation": int(
                weekly_observations.eq("DEMAND_CONFIRMATION_WATCH").sum()),
            "risk_review": int(weekly_observations.eq("RISK_REVIEW").sum()),
            "report": str(output / "mainwave_weekly_candidates.md"),
        },
        "candidate_value": {
            key: value_summary[key] for key in [
                "version", "total", "a_ready", "b_setup", "actionable_now",
                "top_symbols", "report",
            ]
        },
        "slow_carrier": {
            "version": slow_summary.get("version", "MISSING"),
            "calibration_status": slow_summary.get("calibration_status", "UNKNOWN"),
            "focus_total": int(slow_summary.get("focus_total", 0)),
            "ready": int(slow_summary.get("ready", 0)),
            "first_response": int(slow_summary.get("first_response", 0)),
            "valid_lag": int(slow_summary.get("valid_lag", 0)),
            "top_symbols": slow_summary.get("top_symbols", []),
            "report": str(output / "slow_carrier_watch.md"),
        },
        "mean_reversion_shadow": {
            "version": mean_summary.get("version", "MISSING"),
            "market_context": mean_summary.get("market_context", "UNKNOWN"),
            "signals": int(mean_summary.get("signals", 0)),
            "watches": int(mean_summary.get("watches", 0)),
            "signal_symbols": mean_summary.get("signal_symbols", []),
            "report": str(output / "mean_reversion_shadow.md"),
        },
        "order_generation": False,
    }
    (output / "mainwave_three_direction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    direction3_display = direction3.head(MAX_MARKET_CANDIDATES)
    direction2_display = (direction2.groupby("theme", sort=False).head(MAX_THEME_CANDIDATES).head(MAX_MARKET_CANDIDATES)
                          if len(direction2) else direction2)
    expansion_watch = direction3[
        direction3.get("event_type", pd.Series(index=direction3.index, dtype=str)).eq(
            "EXPANSION"
        )
    ].head(MAX_MARKET_CANDIDATES)
    lines = [
        f"# 主升浪三方向闭环日报  {as_of}", "",
        "## 当前三路径入口", "",
        "R1：明确个股点火反查；R2：个股先行准备；R3：紧链传播承接。全域扫描与逐关证据见 route_scan.md。",
        f"- 本次三路径覆盖：{route_summary.get('routes', {})}；同股允许多路线，股票不重复。",
        _table(routes.head(MAX_MARKET_CANDIDATES), ["rank", "symbol", "name", "route_ids", "belonging_theme",
                "route_status_label", "probe_at", "breakout_attempt_at", "breakout_accepted_at", "route_risks"], "本次无新三路径记录。"), "",
        f"- 引擎 `{ENGINE_VERSION}`；报告 `{VERSION}`；下一决策日 `{decision_date}`。",
        "- 本报告只做机会发现和风险分流，不生成订单；最终动作仍受市场总风险和持仓系统约束。",
        f"- 数据：热点 `{data_meta['hotspot_status']}`；映射 `{data_meta['mapping'].get('status', '')}`；辅助紧链 `{data_meta['semantic_auxiliary'].get('status', '')}`。", "",
        f"- 全市场风险总闸：`{market_risk['state']}`；RED 阻断、ORANGE/UNKNOWN 只观察、GREEN 才进入后续分流。", "",
        "## 线路二补充：现有资金资格证据", "",
        f"完整资格池 {len(direction1)} 只；此处展示最多 {MAX_MARKET_CANDIDATES} 只，全部候选进入后续分析。", "",
        _table(direction1.head(MAX_MARKET_CANDIDATES), ["rank", "symbol", "name", "theme", "theme_match_source",
                            "shadow_stage", "retention_state", "route", "r5", "r20",
                            "trigger_distance_pct", "stop_distance_pct", "structural_action", "action",
                            "theme_risk_action"], "当前没有通过全部硬门槛的潜伏标的。"), "",
        "## 补充审计：题材事件侧的历史反查", "",
        f"展示每题材最多 {MAX_THEME_CANDIDATES} 只；当前仍合格 {d2_current} 只。历史命中不等于今天还能买。", "",
        _table(direction2_display, ["rank", "raw_theme", "theme", "t0", "symbol", "name", "role",
                            "probe_lead_sessions", "qualified_lead_sessions", "pre_qualification",
                            "current_qualification", "structural_action", "action", "theme_risk_action"],
               "本周没有可反查的新点火或再点火事件。"), "",
        "## 线路三补充：领头股、紧链与注意力", "",
        f"传播确认 {d3_confirmed} 条；没有合格载体时明确输出等待，不用题材热度替代个股条件。", "",
        _table(direction3_display, ["raw_theme", "theme", "source", "lifecycle", "t0",
                            "information_leader", "current_qualified_carriers",
                            "current_carrier_symbols", "horizon_priority",
                            "research_priority", "external_catalyst_state",
                            "structural_action", "action", "theme_risk_action"],
               "当前没有可分析的动态或登记紧链。"), "",
        "### 扩张/退潮监控（不构成追涨候选）", "",
        _table(expansion_watch, ["raw_theme", "theme", "event_type", "lifecycle", "t0",
                                 "limit_count_t0", "post_week_max",
                                 "current_qualified_carriers", "current_carrier_symbols",
                                 "structural_action", "action"],
               "当前没有扩张事件需要监控。"), "",
        "## 最终结论", "",
        f"- 方向一：结构合格观察 {d1_structural} 只；经市场总闸后当前允许观察 {d1_actionable} 只。",
        f"- 方向二：{d2_current} 只同时满足历史点火关联与当前载体资格。",
        f"- 方向三：结构传播确认 {d3_structural_confirmed} 条；经市场总闸后有效确认 {d3_confirmed} 条。",
        "- 风险优先级：市场RED > 题材阻断 > 市场ORANGE/UNKNOWN > 外盘确认 > 生命周期 > 个股资格；任何均值或热度都不能覆盖失败硬门槛。",
    ]
    weekly_section = [
        "## 本周主升浪候选总表", "",
        f"三路径去重发现池共 {len(discovery)} 只，本表为兼容展示；最终同口径分析与排序见 analysis_board.md。", "",
        f"发现路径：埋伏股 {int(weekly_counts.get('LURK', 0))} 只，点火股 "
        f"{int(weekly_counts.get('IGNITION', 0))} 只，扩散股 "
        f"{int(weekly_counts.get('PROPAGATION', 0))} 只。", "",
        _table(weekly, ["rank", "symbol", "name", "belonging_theme",
                        "discovery_path_label", "stock_lifecycle_label",
                        "theme_lifecycle_label", "ema20_bias", "r5", "r20",
                        "trigger_distance_pct", "action", "note"],
               "本周没有通过统一候选口径的股票。"), "",
    ]
    value_section = [
        "## 候选当前价值排名", "",
        f"A级准备 {value_summary['a_ready']} 只，B级观察 {value_summary['b_setup']} 只，"
        f"当前市场许可下可执行A级 {value_summary['actionable_now']} 只。", "",
        _table(value_frame, ["value_rank", "symbol", "name", "value_tier_label",
                             "buy_state_label", "execution_permission_label",
                             "belonging_theme", "discovery_path_label",
                             "stock_lifecycle_label", "mfe20_p25_pct",
                             "mfe20_median_pct", "mfe20_p75_pct", "risk_reason"],
               "当前没有候选价值结果。"), "",
    ]
    mean_visible = mean_reversion[
        mean_reversion.get("action", pd.Series(index=mean_reversion.index, dtype=str)).isin(
            ["SHADOW_SIGNAL_NEXT_OPEN", "SHADOW_REPAIR_SIGNAL_NEXT_OPEN",
             "SHADOW_WATCH_REPAIR", "SHADOW_WATCH_PANIC_NO_ENTRY",
             "SHADOW_NEAR_TRIGGER", "OVERSOLD_STALE"]
        )
    ].head(MAX_MARKET_CANDIDATES)
    overlay_section = [
        "## Slow-carrier and mean-reversion overlays", "",
        f"Slow carrier: focus={slow_summary.get('focus_total', 0)}, "
        f"ready={slow_summary.get('ready', 0)}, "
        f"first-response={slow_summary.get('first_response', 0)}. "
        "Lifecycle evidence is report-only.", "",
        _table(slow_carriers, ["rank_global", "theme_label", "leader", "symbol", "name",
                               "slow_stage", "window_bucket", "carrier_tier",
                               "response_evidence", "action"],
               "No slow carrier passed the current lifecycle gates."), "",
        f"Mean reversion: context={mean_summary.get('market_context', 'UNKNOWN')}, "
        f"signals={mean_summary.get('signals', 0)}, watches={mean_summary.get('watches', 0)}. "
        "This branch is independent from main-wave discovery and creates no orders.", "",
        _table(mean_visible, ["sector_label", "scope", "symbol", "name",
                              "drawdown60_pct", "ema20_bias_pct", "threshold_pct",
                              "latest_cross_date", "cross_age_sessions", "action"],
               "No active mean-reversion watch item."), "",
    ]
    first_section = next(
        (index for index, line in enumerate(lines) if line.startswith("## ")), len(lines)
    )
    stage_watch = R2.run(as_of, root=OUT_ROOT)
    stage_names = {r["lane"]: f"{r['name']}（{r['symbol']}）" for r in stage_watch["picks"]}
    stage_section = ["## R2 分阶段观察", "",
                     f"突破前准备：{stage_names.get('PREP', '空缺')}；趋势延续：{stage_names.get('TREND', '空缺')}。"
                     f"状态：{stage_watch['status']}。仅观察，[原突破、首次回踩与续踪记录](r2_stage_watch.md)。", ""]
    lines[first_section:first_section] = stage_section + overlay_section + value_section + weekly_section
    (output / "mainwave_three_direction_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(f"[three_direction] D1={len(direction1)}/{d1_actionable} "
          f"D2={len(direction2)}/{d2_current} D3={len(direction3)}/{d3_confirmed}")
    print(f"[three_direction] weekly LURK={int(weekly_counts.get('LURK', 0))} "
          f"IGNITION={int(weekly_counts.get('IGNITION', 0))} "
          f"PROPAGATION={int(weekly_counts.get('PROPAGATION', 0))}")
    print(f"[three_direction] value A={value_summary['a_ready']} "
          f"B={value_summary['b_setup']} actionable={value_summary['actionable_now']} "
          f"top={','.join(value_summary['top_symbols']) or 'NONE'}")
    print(f"[three_direction] slow-carrier focus={slow_summary.get('focus_total', 0)} "
          f"ready={slow_summary.get('ready', 0)}; mean-reversion "
          f"signals={mean_summary.get('signals', 0)} watches={mean_summary.get('watches', 0)}")
    print(f"[three_direction] -> {output / 'candidate_value_analysis.md'}")
    print(f"[three_direction] -> {output / 'mainwave_weekly_candidates.md'}")
    print(f"[three_direction] -> {output / 'mainwave_three_direction_report.md'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--decision-date", default="")
    args = parser.parse_args()
    date = args.date if "-" in args.date else f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}"
    decision_date = args.decision_date
    if decision_date and "-" not in decision_date:
        decision_date = f"{decision_date[:4]}-{decision_date[4:6]}-{decision_date[6:]}"
    return run(date, decision_date)


if __name__ == "__main__":
    raise SystemExit(main())
