# -*- coding: utf-8 -*-
"""P1 observation closure: episodes, leaders, propagation, attention, focus."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__ as ENGINE_VERSION
from . import artifacts as ART
from . import capital_episode as CE
from . import carrier_quality as CQ
from . import candidate_episode as CAND
from . import data as D
from . import decision_bridge as BRIDGE
from . import theme_lifecycle as TL
from .daily_run import market_calendar
from .theme_registry import themes_as_of


OUT_ROOT = Path("results/mainwave_engine")
STATE = OUT_ROOT / "p1_state_store.json"
JOURNAL = OUT_ROOT / "p1_candidate_journal.csv"
OUTCOMES = CAND.OUTCOMES
VERSION = "p1_closure_v1_5_20260903"
MATURE_STAGES = {"ABSORBING", "COILED", "ARMED", "READY"}
BLOCKED_PRODUCTION = {"TRIGGERED", "FAILED", "INVALID"}
ACTIVE_THEME_STAGES = {"FERMENTING", "IGNITED", "DIFFUSING"}


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _stock_metrics(symbol: str, as_of: str, production_record: dict | None = None) -> dict:
    return CQ.stock_metrics(symbol, as_of, production_record)


def _internal_attention(snapshot: dict, timeline: pd.DataFrame) -> dict:
    stage = snapshot["stage"]
    level = {
        "DORMANT": "LOW", "SEED": "MEDIUM", "FERMENTING": "MEDIUM",
        "IGNITED": "HIGH", "DIFFUSING": "HIGH", "CROWDED": "CROWDED",
        "RETREAT": "LOW",
    }[stage]
    slope = "FLAT"
    if len(timeline) >= 6:
        now = timeline.iloc[-1]
        prior = timeline.iloc[-6]
        pulse_delta = float(now["pulse_members"] - prior["pulse_members"])
        breadth_delta = float(now["breadth"] - prior["breadth"])
        if pulse_delta >= 1 or breadth_delta >= 0.10:
            slope = "RISING"
        elif pulse_delta <= -1 or breadth_delta <= -0.10:
            slope = "FALLING"
    if stage == "RETREAT":
        slope = "FALLING"
    return {"internal_attention_level": level, "internal_attention_slope": slope}


def _alias_match(name: str, aliases: list[str]) -> bool:
    left = name.casefold()
    return any(alias.casefold() in left or left in alias.casefold() for alias in aliases if alias)


def _external_attention(theme: dict, as_of: str) -> dict:
    daily_path = Path("hotspot_daily.json")
    history_path = Path("hotspot_history.json")
    daily = _read_json(daily_path, {})
    compact = as_of.replace("-", "")
    aliases = theme.get("attention_aliases", [theme["label"]])
    if str(daily.get("date", "")) != compact:
        return {
            "external_attention_level": "UNKNOWN",
            "external_attention_slope": "UNKNOWN",
            "external_matches": "",
            "external_heat": None,
            "external_limit_count": None,
            "external_data_status": "STALE_OR_MISSING",
        }
    matches = [(name, item) for name, item in daily.get("concepts", {}).items()
               if _alias_match(str(name), aliases)]
    if not matches:
        return {
            "external_attention_level": "LOW", "external_attention_slope": "FLAT",
            "external_matches": "", "external_heat": 0.0, "external_limit_count": 0,
            "external_data_status": "FRESH_NO_MATCH",
        }
    heat = max(float(item.get("heat_score", 0) or 0) for _, item in matches)
    limit_count = max(int(item.get("limit_count", 0) or 0) for _, item in matches)
    level = "CROWDED" if limit_count >= 10 else "HIGH" if heat >= 80 else "MEDIUM" if heat >= 25 else "LOW"

    history = _read_json(history_path, {})
    current_total = sum(int(item.get("limit_count", 0) or 0) for _, item in matches)
    prior_total = 0
    for name, _ in matches:
        points = sorted((date, int(count)) for date, count in history.get(name, {}).items()
                        if date < compact)
        if points:
            prior_total += points[-1][1]
    slope = "RISING" if current_total > prior_total else "FALLING" if current_total < prior_total else "FLAT"
    return {
        "external_attention_level": level, "external_attention_slope": slope,
        "external_matches": "|".join(name for name, _ in matches),
        "external_heat": heat, "external_limit_count": limit_count,
        "external_data_status": "FRESH",
    }


def _theme_attention(theme: dict, snapshot: dict, timeline: pd.DataFrame, as_of: str) -> dict:
    internal = _internal_attention(snapshot, timeline)
    external = _external_attention(theme, as_of)
    int_level = internal["internal_attention_level"]
    int_slope = internal["internal_attention_slope"]
    ext_level = external["external_attention_level"]
    ext_slope = external["external_attention_slope"]
    crowded = snapshot["stage"] == "CROWDED" or ext_level == "CROWDED"
    internal_up = int_slope == "RISING" or int_level in {"HIGH", "CROWDED"}
    external_up = ext_slope == "RISING" or ext_level in {"HIGH", "CROWDED"}
    if ext_level == "UNKNOWN":
        relation = "INTERNAL_ONLY"
    elif crowded:
        relation = "CROWDED"
    elif internal_up and external_up:
        relation = "CONFIRMED"
    elif internal_up and not external_up:
        relation = "INTERNAL_LEADING"
    elif external_up and not internal_up:
        relation = "EXTERNAL_ONLY"
    elif {int_slope, ext_slope} == {"RISING", "FALLING"}:
        relation = "DIVERGENT"
    else:
        relation = "ALIGNED_NEUTRAL"
    if relation in {"INTERNAL_ONLY", "INTERNAL_LEADING"}:
        combined_level, combined_slope = int_level, int_slope
    elif relation == "DIVERGENT":
        order = {"UNKNOWN": -1, "LOW": 0, "MEDIUM": 1, "HIGH": 2, "CROWDED": 3}
        combined_level = max((int_level, ext_level), key=lambda value: order[value])
        combined_slope = "CONFLICT"
    else:
        combined_level, combined_slope = ext_level, ext_slope
    return {
        **internal, **external,
        "attention_level": combined_level,
        "attention_slope": combined_slope,
        "attention_relation": relation,
        "attention_conflict": relation == "DIVERGENT",
        "crowding_state": "CROWDED" if crowded else "NOT_CROWDED",
    }


def _episode_start(episode: dict) -> str:
    return episode.get("evidence", {}).get("event", {}).get("first_seen", episode.get("start_date", ""))


def _maturity(stage: str) -> int:
    return {"READY": 5, "ARMED": 4, "COILED": 3, "ABSORBING": 2, "PROBED": 1}.get(stage, 0)


def _theme_rows(theme: dict, snapshot: dict, attention: dict, episodes: dict,
                stock_store: dict, as_of: str, previous_leader: str,
                stock_metrics: dict | None = None, previous_episode_id: str = "",
                previous_cycle=None,
                carrier_quality: dict | None = None) -> tuple[list[dict], dict, list[dict]]:
    proposed_start = (
        snapshot.get("stage_since") if snapshot["stage"] in {"SEED", "FERMENTING"}
        else snapshot.get("last_ignition")
    ) or snapshot.get("stage_since") or theme["active_from"]
    proposed_episode_id = f"{theme['id']}:C{snapshot['cycle']}:{proposed_start}"
    same_cycle = bool(previous_episode_id) and (
        str(previous_cycle) == str(snapshot["cycle"])
        or previous_episode_id == proposed_episode_id
    )
    theme_episode_id = previous_episode_id if same_cycle else proposed_episode_id
    if previous_episode_id and not same_cycle:
        previous_leader = ""

    metrics = stock_metrics or {
        symbol: _stock_metrics(symbol, as_of, stock_store.get(symbol, {}))
        for symbol in theme["members"]
    }
    valid_r20 = [item["r20"] for item in metrics.values() if np.isfinite(item["r20"])]
    chain_median = float(np.median(valid_r20)) if valid_r20 else 0.0
    valid_drawdown = [item.get("drawdown20") for item in metrics.values()
                      if np.isfinite(item.get("drawdown20", np.nan))]
    drawdown_median = float(np.median(valid_drawdown)) if valid_drawdown else 0.0
    valid_repair = [item.get("repair5") for item in metrics.values()
                    if np.isfinite(item.get("repair5", np.nan))]
    repair_median = float(np.median(valid_repair)) if valid_repair else 0.0
    eligible = []
    for symbol, name in theme["members"].items():
        episode = episodes.get(symbol)
        if not episode or episode.get("shadow_stage") not in MATURE_STAGES:
            continue
        production = episode.get("production_state", stock_store.get(symbol, {}).get("state", "OUT"))
        if production in BLOCKED_PRODUCTION or episode.get("retention_state") == "EXITED":
            continue
        item = metrics[symbol]
        rel20 = item["r20"] - chain_median if np.isfinite(item["r20"]) else -999.0
        relative_drawdown = (item.get("drawdown20") - drawdown_median
                             if np.isfinite(item.get("drawdown20", np.nan)) else -999.0)
        relative_repair = (item.get("repair5") - repair_median
                           if np.isfinite(item.get("repair5", np.nan)) else -999.0)
        route = str(stock_store.get(symbol, {}).get("route") or "UNCLASSIFIED")
        quality = ((carrier_quality or {}).get(symbol)
                   or CQ.evaluate(episode, item, route, production))
        eligible.append({
            "symbol": symbol, "name": name, "episode": episode, "metrics": item,
            "production_state": production, "relative_r20": rel20,
            "relative_drawdown20": relative_drawdown,
            "relative_repair5": relative_repair,
            "route": route, "quality": quality,
            "event_start": _episode_start(episode),
        })

    leader = None
    leader_state = "NONE"
    response_rows = []
    leader_replaced = False
    if eligible:
        leader_pool = [item for item in eligible if item["relative_r20"] >= 0]
    else:
        leader_pool = []
    if leader_pool:
        leader_pool.sort(key=lambda item: (
            0 if item["episode"]["retention_state"] == "RETAINED" else 1,
            -item["relative_drawdown20"],
            -item["relative_repair5"],
            -item["relative_r20"],
            CQ.ROUTE_ORDER.get(item["route"], 99),
            item["event_start"] or "9999-99-99",
            -_maturity(item["episode"]["shadow_stage"]),
            item["symbol"],
        ))
        leader = leader_pool[0]
        for item in eligible:
            if item["symbol"] == leader["symbol"]:
                continue
            responded = bool(item["event_start"] and leader["event_start"]
                             and item["event_start"] > leader["event_start"])
            response_rows.append({
                "theme_id": theme["id"], "leader": leader["symbol"],
                "follower": item["symbol"], "leader_event_start": leader["event_start"],
                "follower_event_start": item["event_start"], "responded_after_leader": responded,
            })
        response_count = sum(row["responded_after_leader"] for row in response_rows)
        leader_state = "LEADER_CONFIRMED" if response_count >= 2 else "LEADER_SEED"
        leader_replaced = bool(previous_leader and previous_leader != leader["symbol"])
    else:
        response_count = 0

    propagation = {
        "as_of": as_of, "theme_id": theme["id"], "theme_label": theme["label"],
        "theme_episode_id": theme_episode_id, "theme_lifecycle": snapshot["stage"],
        "cycle": snapshot["cycle"], "leader": leader["symbol"] if leader else "",
        "leader_event_start": leader["event_start"] if leader else "",
        "leader_state": leader_state, "previous_leader": previous_leader,
        "leader_replaced": leader_replaced, "theme_signal_ex_leader": response_count,
        "eligible_carriers": len(eligible),
        "qualified_carriers": sum(item["quality"]["hard_gate_pass"] for item in eligible),
    }

    candidates = []
    if leader and attention["crowding_state"] != "CROWDED" and snapshot["stage"] != "RETREAT":
        leader_qualified = leader["quality"]["hard_gate_pass"]
        if (leader_state == "LEADER_CONFIRMED"
                and leader["episode"]["retention_state"] == "RETAINED"
                and snapshot["stage"] in ACTIVE_THEME_STAGES):
            intersection = "A"
        elif (leader_state == "LEADER_SEED"
                and leader_qualified
                and leader["episode"]["retention_state"] == "RETAINED"
                and snapshot["stage"] in {"DORMANT", "SEED", "FERMENTING", "IGNITED"}):
            intersection = "B"
        else:
            intersection = ""
        if intersection and leader_qualified:
            candidates.append(_candidate_row(
                leader, theme, propagation, attention, intersection, "information_leader", response_count))

        if intersection == "A":
            followers = [item for item in eligible if item["symbol"] != leader["symbol"]
                         and item["quality"]["hard_gate_pass"]]
            followers.sort(key=lambda item: (
                -item["relative_drawdown20"],
                -item["relative_repair5"],
                -item["relative_r20"],
                -_maturity(item["episode"]["shadow_stage"]),
                item["metrics"].get("trigger_distance_pct", 999),
                item["metrics"].get("stop_distance_pct", 999),
                CQ.ROUTE_ORDER.get(item["route"], 99),
                item["symbol"]))
            if followers:
                candidates.append(_candidate_row(
                    followers[0], theme, propagation, attention, "A", "accumulation_carrier", response_count))
            remaining = [item for item in followers
                         if item["symbol"] not in {row["symbol"] for row in candidates}]
            remaining.sort(key=lambda item: (
                -item["relative_drawdown20"], -item["relative_repair5"],
                -item["relative_r20"], CQ.ROUTE_ORDER.get(item["route"], 99),
                item["symbol"]))
            if remaining:
                candidates.append(_candidate_row(
                    remaining[0], theme, propagation, attention, "A", "elasticity_carrier", response_count))
    return candidates[:3], propagation, response_rows


def _candidate_row(item: dict, theme: dict, propagation: dict, attention: dict,
                   intersection: str, role: str, response_count: int) -> dict:
    episode = item["episode"]
    why = (
        f"{intersection}类/{role}; {episode['shadow_stage']}; "
        f"carrier={item['quality']['carrier_tier']}; retention={episode['retention_state']}; "
        f"ex-leader responses={response_count}; "
        f"attention={attention['attention_level']}/{attention['attention_slope']}"
    )
    invalidation = (
        "production TRIGGERED/FAILED/INVALID; retention EXITED; current theme episode RETREAT; "
        "attention CROWDED; capital episode expired; leader replaced; carrier hard gate failed"
    )
    return {
        "as_of": propagation["as_of"], "theme_id": theme["id"],
        "theme_label": theme["label"], "theme_episode_id": propagation["theme_episode_id"],
        "theme_lifecycle": propagation["theme_lifecycle"],
        "attention_level": attention["attention_level"],
        "attention_slope": attention["attention_slope"],
        "attention_relation": attention.get("attention_relation", "UNKNOWN"),
        "crowding_state": attention["crowding_state"],
        "symbol": item["symbol"], "name": item["name"],
        "episode_id": episode["episode_id"], "shadow_stage": episode["shadow_stage"],
        "retention_state": episode["retention_state"],
        "leader_state": propagation["leader_state"] if role == "information_leader" else "FOLLOWER",
        "intersection_class": intersection, "role": role,
        "production_state": item["production_state"],
        "first_seen_date": item["event_start"], "valid_until": episode["valid_until"],
        "valid_for_sessions": episode.get("valid_for_sessions", ""),
        "relative_r20": round(item["relative_r20"], 2),
        "relative_drawdown20": round(item["relative_drawdown20"], 2),
        "relative_repair5": round(item["relative_repair5"], 2),
        "route": item["route"],
        "carrier_qualification": item["quality"]["carrier_qualification"],
        "carrier_tier": item["quality"]["carrier_tier"],
        "trigger_distance_pct": item["metrics"].get("trigger_distance_pct"),
        "stop_distance_pct": item["metrics"].get("stop_distance_pct"),
        "close": round(float(item["metrics"]["close"]), 2) if np.isfinite(item["metrics"]["close"]) else None,
        "why_now": why, "invalidation": invalidation,
    }


def _rank_candidates(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=[
            "as_of", "theme_id", "theme_label", "theme_episode_id", "theme_lifecycle",
            "attention_level", "attention_slope", "attention_relation", "crowding_state",
            "symbol", "name",
            "episode_id", "shadow_stage", "retention_state", "leader_state",
            "intersection_class", "role", "production_state", "first_seen_date",
            "valid_until", "valid_for_sessions", "relative_r20", "relative_drawdown20",
            "relative_repair5", "route", "carrier_qualification", "carrier_tier",
            "trigger_distance_pct", "stop_distance_pct", "close", "why_now", "invalidation",
            "rank_in_theme", "rank_global"])
    frame = pd.DataFrame(rows)
    frame["_class"] = frame["intersection_class"].map({"A": 0, "B": 1})
    frame["_role"] = frame["role"].map({
        "information_leader": 0, "accumulation_carrier": 1, "elasticity_carrier": 2})
    frame["_retention"] = frame["retention_state"].map({"RETAINED": 0, "UNCERTAIN": 1})
    frame["_maturity"] = frame["shadow_stage"].map(
        {"READY": 0, "ARMED": 1, "COILED": 2, "ABSORBING": 3}).fillna(9)
    frame["_route"] = frame["route"].map(CQ.ROUTE_ORDER).fillna(99)
    frame = frame.sort_values(
        ["_class", "relative_drawdown20", "relative_repair5", "relative_r20",
         "_role", "_retention", "_maturity", "trigger_distance_pct", "stop_distance_pct",
         "_route", "symbol"],
        ascending=[True, False, False, False, True, True, True, True, True, True, True])
    frame["rank_in_theme"] = frame.groupby("theme_id").cumcount() + 1
    frame = frame[frame["rank_in_theme"] <= 3].head(15).copy()
    frame["rank_global"] = range(1, len(frame) + 1)
    return frame.drop(columns=["_class", "_role", "_retention", "_maturity", "_route"])


def _sha256(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _append_candidate_journal(focus: pd.DataFrame) -> pd.DataFrame:
    """Upsert the immutable daily discovery journal."""
    journal_columns = [
        "as_of", "theme_id", "theme_label", "theme_episode_id", "symbol", "name",
        "episode_id", "intersection_class", "role", "shadow_stage", "retention_state",
        "carrier_qualification", "carrier_tier", "relative_drawdown20", "relative_repair5",
        "relative_r20", "trigger_distance_pct", "stop_distance_pct",
        "leader_state", "production_state", "first_seen_date", "valid_until", "why_now",
        "invalidation", "candidate_version",
    ]
    today = focus.copy()
    if len(today):
        today["candidate_version"] = VERSION
        today = today[[column for column in journal_columns if column in today.columns]]
    else:
        today = pd.DataFrame(columns=journal_columns)
    if JOURNAL.exists():
        journal = pd.read_csv(JOURNAL, encoding="utf-8-sig")
        if len(today):
            keys = set(zip(today["as_of"], today["theme_id"], today["symbol"], today["episode_id"]))
            keep = [key not in keys for key in zip(
                journal["as_of"], journal["theme_id"], journal["symbol"], journal["episode_id"])]
            journal = pd.concat([journal.loc[keep], today], ignore_index=True)
    else:
        journal = today
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    journal.sort_values(["as_of", "theme_id", "symbol"]).to_csv(
        JOURNAL, index=False, encoding="utf-8-sig")
    return journal


def run(as_of: str) -> int:
    out = OUT_ROOT / as_of.replace("-", "")
    trajectory_path = out / "trajectory.csv"
    if not trajectory_path.exists():
        raise RuntimeError(f"missing daily trajectory: {trajectory_path}; run daily_run first")
    snapshot_sources = [
        trajectory_path, Path("theme_registry.json"), Path("concept_mapping.json"), Path("concept_membership_evidence.json"),
        Path("hotspot_daily.json"), Path("hotspot_history.json"), out / "market_context.json",
        OUT_ROOT / "state_store.json", OUT_ROOT / "theme_state_store.json",
        CE.STORE, CAND.STORE, STATE,
    ]
    snapshot_dir, snapshot_manifest = ART.snapshot_inputs(
        snapshot_sources, out, {"as_of": as_of, "engine_version": ENGINE_VERSION,
                                "p1_version": VERSION}
    )
    trajectory = pd.read_csv(trajectory_path, encoding="utf-8-sig").to_dict("records")
    calendar = market_calendar(as_of)
    episode_store, closed_today = CE.update_store(trajectory, as_of, calendar)
    capital_frame = CE.write_daily(
        episode_store, closed_today, as_of, out / "capital_episodes.csv")

    stock_store = _read_json(OUT_ROOT / "state_store.json", {}).get("symbols", {})
    carrier_frame, carrier_quality, carrier_metrics = CQ.build_frame(
        as_of, episode_store["active"], stock_store, D.name_map()
    )
    carrier_summary = CQ.write_report(carrier_frame, as_of, out)
    previous_state = _read_json(STATE, {"_meta": {"as_of": ""}, "themes": {}})
    previous_as_of = previous_state.get("_meta", {}).get("as_of", "")
    if previous_as_of and as_of < previous_as_of:
        raise RuntimeError(f"P1 state is at {previous_as_of}; refuse historical write {as_of}")

    candidates = []
    theme_rows = []
    attention_rows = []
    propagation_rows = []
    response_rows = []
    new_state = {}
    for theme in themes_as_of(as_of):
        analyzed = TL.analyze_theme(theme, as_of)
        snapshot = analyzed["snapshot"]
        attention = _theme_attention(theme, snapshot, analyzed["timeline"], as_of)
        previous_theme = previous_state.get("themes", {}).get(theme["id"], {})
        previous_leader = previous_theme.get("leader", "")
        theme_metrics = {
            symbol: carrier_metrics.get(symbol)
                    or _stock_metrics(symbol, as_of, stock_store.get(symbol, {}))
            for symbol in theme["members"]
        }
        selected, propagation, responses = _theme_rows(
            theme, snapshot, attention, episode_store["active"], stock_store, as_of,
            previous_leader, previous_episode_id=previous_theme.get("theme_episode_id", ""),
            previous_cycle=previous_theme.get("cycle"), stock_metrics=theme_metrics,
            carrier_quality=carrier_quality)
        candidates.extend(selected)
        propagation_rows.append(propagation)
        response_rows.extend(responses)
        attention_rows.append({"as_of": as_of, "theme_id": theme["id"],
                               "theme_label": theme["label"], **attention})
        theme_episode_id = propagation["theme_episode_id"]
        wait_reason = ""
        if snapshot["stage"] == "CROWDED" or attention["crowding_state"] == "CROWDED":
            wait_reason = "CROWDED_NO_CHASE"
        elif snapshot["stage"] == "RETREAT":
            wait_reason = "CURRENT_EPISODE_RETREAT_WAIT_NEW_CYCLE"
        elif not selected and snapshot["stage"] in ACTIVE_THEME_STAGES:
            wait_reason = "C_NO_QUALIFIED_CARRIER"
        elif not selected:
            wait_reason = "WAIT_PROPAGATION_OR_CARRIER"
        theme_rows.append({
            "as_of": as_of, "theme_id": theme["id"], "theme_label": theme["label"],
            "theme_episode_id": theme_episode_id, "theme_lifecycle": snapshot["stage"],
            "cycle": snapshot["cycle"], "attention_level": attention["attention_level"],
            "attention_slope": attention["attention_slope"],
            "attention_relation": attention["attention_relation"],
            "crowding_state": attention["crowding_state"],
            "leader": propagation["leader"], "leader_state": propagation["leader_state"],
            "theme_signal_ex_leader": propagation["theme_signal_ex_leader"],
            "selected_count": len(selected), "wait_reason": wait_reason,
            "valid_until": "until lifecycle transition, crowding, or carrier invalidation",
        })
        new_state[theme["id"]] = {
            "theme_episode_id": theme_episode_id, "leader": propagation["leader"],
            "leader_state": propagation["leader_state"], "cycle": snapshot["cycle"],
            "as_of": as_of,
        }

    potential = pd.DataFrame(candidates)
    focus = _rank_candidates(candidates)
    themes_frame = pd.DataFrame(theme_rows)
    attention_frame = pd.DataFrame(attention_rows)
    propagation_frame = pd.DataFrame(propagation_rows)
    responses_frame = pd.DataFrame(response_rows)
    focus.to_csv(out / "intersection_candidates.csv", index=False, encoding="utf-8-sig")
    themes_frame.to_csv(out / "theme_episodes.csv", index=False, encoding="utf-8-sig")
    attention_frame.to_csv(out / "attention_lifecycle.csv", index=False, encoding="utf-8-sig")
    propagation_frame.to_csv(out / "leader_state_daily.csv", index=False, encoding="utf-8-sig")
    responses_frame.to_csv(out / "theme_propagation.csv", index=False, encoding="utf-8-sig")
    _append_candidate_journal(focus)
    _, outcomes, funnel = CAND.update(
        focus, potential, themes_frame, episode_store, as_of, out
    )
    _, bridge_summary = BRIDGE.build(as_of, focus, out)

    _atomic_json(STATE, {"_meta": {"as_of": as_of, "version": VERSION}, "themes": new_state})
    counts = focus["intersection_class"].value_counts().to_dict() if len(focus) else {}
    summary = {
        "as_of": as_of, "version": VERSION, "engine_version": ENGINE_VERSION,
        "capital_episode_version": CE.VERSION, "capital_active": len(episode_store["active"]),
        "capital_closed_today": len(closed_today), "themes": len(theme_rows),
        "carrier_quality_version": CQ.VERSION,
        "carrier_qualified": carrier_summary["qualified_lurk"],
        "carrier_building": carrier_summary["building"],
        "carrier_extended": carrier_summary["extended_too_late"],
        "carrier_distribution_risk": carrier_summary["distribution_risk"],
        "focus_total": len(focus), "class_A": int(counts.get("A", 0)),
        "class_B": int(counts.get("B", 0)),
        "class_C_wait": int(sum(row["wait_reason"] == "C_NO_QUALIFIED_CARRIER" for row in theme_rows)),
        "candidate_outcomes": outcomes["outcome"].value_counts().to_dict() if len(outcomes) else {},
        "candidate_funnel": funnel,
        "l0_bridge": bridge_summary,
        "production_writeback": False,
    }
    (out / "p1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    focus_cols = ["rank_global", "intersection_class", "theme_label", "symbol", "name", "role",
                  "carrier_tier", "shadow_stage", "retention_state", "leader_state", "theme_lifecycle",
                  "attention_level", "attention_slope", "attention_relation",
                  "relative_drawdown20", "relative_repair5", "relative_r20",
                  "trigger_distance_pct", "stop_distance_pct", "production_state", "why_now"]
    theme_cols = ["theme_label", "theme_lifecycle", "cycle", "attention_level",
                  "attention_slope", "attention_relation", "crowding_state", "leader", "leader_state",
                  "theme_signal_ex_leader", "selected_count", "wait_reason"]
    lines = [
        f"# 主升浪P1发现闭环日报  {as_of}", "",
        f"- 版本：`{VERSION}` / engine `{ENGINE_VERSION}` / capital `{CE.VERSION}`",
        f"- 观察层：资金episode {len(episode_store['active'])}个；题材 {len(theme_rows)}个；"
        f"焦点 {len(focus)}只（A={counts.get('A', 0)}，B={counts.get('B', 0)}）",
        f"- Candidate episodes: active={funnel['active']} / closed={funnel['closed']} / "
        f"B-to-A={funnel['b_to_a']} / triggered={funnel['triggered']}",
        f"- Carrier hard gates: qualified={carrier_summary['qualified_lurk']} / "
        f"building={carrier_summary['building']} / extended={carrier_summary['extended_too_late']} / "
        f"distribution-risk={carrier_summary['distribution_risk']}; weighted score is disabled.",
        f"- L0 bridge: {bridge_summary['l0']['state']}; permissions are observation-only; orders=0.",
        "- 边界：不修改ARMED/READY/TRIGGERED，不生成正式买入指令；外部注意力缺失记UNKNOWN。", "",
        "## A类：题材传播确认 × 载体已准备", "",
        (focus[focus["intersection_class"] == "A"][focus_cols].to_markdown(index=False)
         if len(focus) and (focus["intersection_class"] == "A").any() else "（无）"), "",
        "## B类：个股先行 × 传播待确认", "",
        (focus[focus["intersection_class"] == "B"][focus_cols].to_markdown(index=False)
         if len(focus) and (focus["intersection_class"] == "B").any() else "（无）"), "",
        "## 题材状态与等待原因", "",
        themes_frame[theme_cols].to_markdown(index=False) if len(themes_frame) else "（无）", "",
        "## 阅读纪律", "",
        "- UNCERTAIN保留在episode观察层；A/B焦点都必须RETAINED。",
        "- CROWDED阻止追逐当前周期；RETREAT结束当前episode，但不永久排除下一轮发酵。",
        "- theme_signal_ex_leader只统计剔除领头股后的响应，避免循环确认。",
    ]
    (out / "p1_closure_report.md").write_text("\n".join(lines), encoding="utf-8")

    inputs = snapshot_sources
    manifest = {
        "as_of": as_of, "engine_version": ENGINE_VERSION, "p1_version": VERSION,
        "capital_episode_version": CE.VERSION,
        "candidate_episode_version": CAND.VERSION,
        "decision_bridge_version": BRIDGE.VERSION,
        "input_snapshot": {"path": str(snapshot_dir), **snapshot_manifest},
        "state_store_as_of": _read_json(OUT_ROOT / "state_store.json", {}).get("_meta", {}).get("engine_as_of", ""),
        "inputs": {str(path): _sha256(path) for path in inputs},
        "outputs": {
            name: _sha256(out / name) for name in (
                "capital_episodes.csv", "theme_episodes.csv", "attention_lifecycle.csv",
                "leader_state_daily.csv", "theme_propagation.csv",
                "intersection_candidates.csv", "candidate_episodes.csv", "candidate_funnel.json",
                "p1_decision_bridge.csv", "p1_decision_bridge_summary.json",
                "carrier_qualification.csv", "carrier_qualification_summary.json",
                "carrier_qualification.md", "p1_closure_report.md")
        },
        "cumulative_outputs": {
            str(JOURNAL): _sha256(JOURNAL), str(OUTCOMES): _sha256(OUTCOMES),
        },
    }
    (out / "p1_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[p1] capital_active={summary['capital_active']} themes={summary['themes']} "
          f"focus={summary['focus_total']} A={summary['class_A']} B={summary['class_B']}")
    print(f"[p1] carriers qualified={summary['carrier_qualified']} "
          f"building={summary['carrier_building']} extended={summary['carrier_extended']}")
    print(f"[p1] -> {out / 'p1_closure_report.md'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    date = args.date if "-" in args.date else f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}"
    return run(date)


if __name__ == "__main__":
    raise SystemExit(main())
