# -*- coding: utf-8 -*-
"""Hard-gate qualification for capital-behaviour carriers.

Qualification is deliberately categorical. A strong value in one dimension
cannot compensate for missing retention, supply, execution, or safety evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .review_clock import publish


VERSION = "carrier_quality_v1_2_20260903"
MATURE_STAGES = {"ABSORBING", "COILED", "ARMED", "READY"}
BROKEN_STATES = {"FAILED", "INVALID"}
TOO_LATE_STATES = {"TRIGGERED"}
MAX_TRIGGER_DISTANCE_PCT = 8.0
MAX_STOP_DISTANCE_PCT = 8.0
MAX_ABOVE_TRIGGER_PCT = 2.0
EXTENSION_LIMITS = {"r5": 15.0, "r20": 30.0, "ma20_bias": 12.0}

STATUS_ORDER = {
    "QUALIFIED_LURK": 0,
    "BUILDING": 1,
    "RETENTION_UNPROVEN": 2,
    "SUPPLY_UNPROVEN": 3,
    "PROBED_ONLY": 4,
    "EXECUTION_BLOCKED": 5,
    "EXTENDED_TOO_LATE": 6,
    "DISTRIBUTION_RISK": 7,
    "TRIGGERED_TOO_LATE": 8,
    "EXITED_OR_BROKEN": 9,
    "DATA_INCOMPLETE": 10,
    "NO_VALID_PROBE": 11,
    "NO_CAPITAL_EPISODE": 12,
}
STAGE_ORDER = {"READY": 0, "ARMED": 1, "COILED": 2, "ABSORBING": 3, "PROBED": 4}
ROUTE_ORDER = {"HYBRID": 0, "HOT": 1, "INST": 2, "UNCLASSIFIED": 3, "": 4}


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _event_seen(episode: dict) -> bool:
    event = episode.get("evidence", {}).get("event", {})
    return bool(event.get("first_seen") or episode.get("start_date"))


def _supply_seen(episode: dict) -> bool:
    supply = episode.get("evidence", {}).get("supply", {})
    return bool(supply.get("last_seen") and not supply.get("invalidated_at"))


def _distribution_risk(frame: pd.DataFrame, symbol: str) -> bool:
    """Reuse the production distribution semantics on the latest three bars."""
    if len(frame) < 25:
        return False
    high = frame["high"].to_numpy(float)
    low = frame["low"].to_numpy(float)
    close = frame["close"].to_numpy(float)
    amount = frame["amount"].to_numpy(float)
    pct = frame["pct_chg"].to_numpy(float)
    limit_pct = D.limit_pct(symbol)
    for index in range(max(20, len(frame) - 3), len(frame)):
        prior_amount = np.nanmedian(amount[index - 20:index])
        prior_high = np.nanmax(high[index - 20:index])
        price_range = max(high[index] - low[index], 1e-9)
        touched = high[index] >= prior_high or pct[index] >= limit_pct * 0.85
        climax = np.isfinite(prior_amount) and prior_amount > 0 and amount[index] >= 2.5 * prior_amount
        weak_close = (close[index] - low[index]) / price_range <= 0.35
        upper_shadow = (high[index] - close[index]) / price_range >= 0.40
        if touched and climax and weak_close and upper_shadow:
            return True
    return False


def metrics_from_frame(symbol: str, frame: pd.DataFrame | None,
                       production_record: dict | None = None) -> dict:
    """Calculate position and resilience evidence from an as-of price frame."""
    record = production_record or {}
    if frame is None or frame.empty:
        return {
            "data_fresh": False, "close": np.nan, "r5": np.nan, "r20": np.nan,
            "ma20_bias": np.nan, "amount_ratio": np.nan, "drawdown20": np.nan,
            "repair5": np.nan, "trigger_distance_pct": np.nan,
            "stop_distance_pct": np.nan, "execution_state": "DATA_MISSING",
            "extension_state": "UNKNOWN", "distribution_risk": False,
        }

    close = frame["close"].to_numpy(float)
    high = frame["high"].to_numpy(float)
    low = frame["low"].to_numpy(float)
    amount = frame["amount"].to_numpy(float)
    pct = frame["pct_chg"].to_numpy(float)
    ma20 = float(pd.Series(close).rolling(20).mean().iloc[-1])
    trigger = record.get("trigger_price")
    if not _finite(trigger):
        resistance = record.get("stored_resistance")
        atr = record.get("resistance_atr")
        if _finite(resistance) and _finite(atr):
            trigger = float(resistance) + 0.2 * float(atr)
    invalid = record.get("invalid_price")

    trigger_distance = ((float(trigger) / close[-1] - 1) * 100
                        if _finite(trigger) else np.nan)
    stop_distance = ((1 - float(invalid) / close[-1]) * 100
                     if _finite(invalid) else np.nan)
    if not (_finite(trigger_distance) and _finite(stop_distance)):
        execution_state = "POSITION_UNKNOWN"
    elif trigger_distance < -MAX_ABOVE_TRIGGER_PCT:
        execution_state = "ABOVE_TRIGGER"
    elif trigger_distance > MAX_TRIGGER_DISTANCE_PCT:
        execution_state = "TRIGGER_TOO_FAR"
    elif stop_distance <= 0:
        execution_state = "INVALID_LEVEL_BROKEN"
    elif stop_distance > MAX_STOP_DISTANCE_PCT:
        execution_state = "STOP_TOO_WIDE"
    else:
        execution_state = "EXECUTABLE"

    r5 = (close[-1] / close[-6] - 1) * 100
    r20 = (close[-1] / close[-21] - 1) * 100
    ma20_bias = (close[-1] / ma20 - 1) * 100
    drawdown20 = (close[-1] / np.nanmax(close[-20:]) - 1) * 100
    repair5 = (close[-1] / np.nanmin(low[-6:]) - 1) * 100
    recent_ranges = (high[-5:] - low[-5:]) / np.maximum(close[-5:], 1e-9)
    base_ranges = (high[-20:] - low[-20:]) / np.maximum(close[-20:], 1e-9)
    range_contraction = (float(np.nanmedian(recent_ranges) / np.nanmedian(base_ranges))
                         if np.nanmedian(base_ranges) > 0 else np.nan)
    down_amount = amount[-10:][pct[-10:] < 0]
    up_amount = amount[-10:][pct[-10:] > 0]
    pullback_volume_ratio = (
        float(np.nanmedian(down_amount) / np.nanmedian(up_amount))
        if len(down_amount) and len(up_amount) and np.nanmedian(up_amount) > 0 else np.nan
    )
    extended = (
        r5 > EXTENSION_LIMITS["r5"]
        or r20 > EXTENSION_LIMITS["r20"]
        or ma20_bias > EXTENSION_LIMITS["ma20_bias"]
    )
    return {
        "data_fresh": True,
        "close": round(float(close[-1]), 3),
        "r5": round(float(r5), 2),
        "r20": round(float(r20), 2),
        "ma20_bias": round(float(ma20_bias), 2),
        "amount_ratio": round(float(amount[-5:].mean() / amount[-20:].mean()), 2)
        if amount[-20:].mean() > 0 else np.nan,
        "drawdown20": round(float(drawdown20), 2),
        "repair5": round(float(repair5), 2),
        "range_contraction": round(range_contraction, 3) if _finite(range_contraction) else np.nan,
        "pullback_volume_ratio": round(pullback_volume_ratio, 3)
        if _finite(pullback_volume_ratio) else np.nan,
        "trigger_price": round(float(trigger), 3) if _finite(trigger) else np.nan,
        "invalid_price": round(float(invalid), 3) if _finite(invalid) else np.nan,
        "trigger_distance_pct": round(float(trigger_distance), 2)
        if _finite(trigger_distance) else np.nan,
        "stop_distance_pct": round(float(stop_distance), 2)
        if _finite(stop_distance) else np.nan,
        "execution_state": execution_state,
        "extension_state": "EXTENDED" if extended else "NOT_EXTENDED",
        "distribution_risk": _distribution_risk(frame, symbol),
    }


def stock_metrics(symbol: str, as_of: str, production_record: dict | None = None) -> dict:
    """Load one point-in-time price frame and calculate carrier evidence."""
    frame = D.load_tail(symbol, 80, as_of=as_of)
    if frame is None or str(frame["date"].iloc[-1]) != as_of:
        frame = None
    return metrics_from_frame(symbol, frame, production_record)


def evaluate(episode: dict | None, metrics: dict | None = None, route: str = "",
             production_state: str = "") -> dict:
    """Apply non-compensating gates and return one categorical decision."""
    episode = episode or {}
    metrics = metrics or {}
    stage = str(episode.get("shadow_stage", ""))
    production = str(production_state or episode.get("production_state", ""))
    retention = str(episode.get("retention_state", ""))
    capital_gate = _event_seen(episode)
    retention_gate = retention == "RETAINED"
    supply_gate = _supply_seen(episode)
    maturity_gate = stage in MATURE_STAGES
    data_gate = bool(metrics.get("data_fresh"))
    execution_gate = metrics.get("execution_state") == "EXECUTABLE"
    extension_gate = metrics.get("extension_state") != "EXTENDED"
    distribution_gate = not bool(metrics.get("distribution_risk"))
    production_gate = production not in BROKEN_STATES | TOO_LATE_STATES

    failures = []
    for name, passed in (
        ("CAPITAL_NOT_PROVEN", capital_gate),
        ("RETENTION_NOT_PROVEN", retention_gate),
        ("SUPPLY_NOT_PROVEN", supply_gate),
        ("STAGE_TOO_EARLY", maturity_gate),
        ("DATA_NOT_FRESH", data_gate),
        ("POSITION_NOT_EXECUTABLE", execution_gate),
        ("PRICE_ALREADY_EXTENDED", extension_gate),
        ("DISTRIBUTION_OR_WEAK_CLOSE", distribution_gate),
        ("PRODUCTION_STATE_BLOCKED", production_gate),
    ):
        if not passed:
            failures.append(name)

    if not episode:
        status = "NO_CAPITAL_EPISODE"
    elif not capital_gate:
        status = "NO_VALID_PROBE"
    elif production in BROKEN_STATES or retention == "EXITED":
        status = "EXITED_OR_BROKEN"
    elif production in TOO_LATE_STATES:
        status = "TRIGGERED_TOO_LATE"
    elif not retention_gate:
        status = "RETENTION_UNPROVEN"
    elif not supply_gate:
        status = "SUPPLY_UNPROVEN"
    elif not maturity_gate:
        status = "PROBED_ONLY"
    elif not data_gate:
        status = "DATA_INCOMPLETE"
    elif not distribution_gate:
        status = "DISTRIBUTION_RISK"
    elif not extension_gate:
        status = "EXTENDED_TOO_LATE"
    elif not execution_gate:
        status = ("BUILDING" if stage == "ABSORBING"
                  and metrics.get("execution_state") in {"POSITION_UNKNOWN", "TRIGGER_TOO_FAR"}
                  else "EXECUTION_BLOCKED")
    else:
        status = "QUALIFIED_LURK"

    if status == "QUALIFIED_LURK" and stage == "ABSORBING":
        tier = "EARLY_LURK"
    elif status == "QUALIFIED_LURK" and stage == "COILED":
        tier = "CORE_LURK"
    elif status == "QUALIFIED_LURK":
        tier = "TRIGGER_NEAR"
    else:
        tier = "NOT_QUALIFIED"
    route_style = {
        "HOT": "HOT_ATTENTION_CARRIER",
        "INST": "INST_EVIDENCE_CARRIER",
        "HYBRID": "HYBRID_PERSISTENCE_CARRIER",
    }.get(route, "UNCLASSIFIED_CARRIER")
    return {
        "carrier_qualification": status,
        "hard_gate_pass": status == "QUALIFIED_LURK",
        "carrier_tier": tier,
        "route_style": route_style,
        "gate_failures": "|".join(failures),
        "capital_gate": capital_gate,
        "retention_gate": retention_gate,
        "supply_gate": supply_gate,
        "maturity_gate": maturity_gate,
        "execution_gate": execution_gate,
        "extension_gate": extension_gate,
        "distribution_gate": distribution_gate,
    }


def build_frame(as_of: str, episodes: dict, production_store: dict,
                names: dict[str, str] | None = None) -> tuple[pd.DataFrame, dict, dict]:
    """Build the daily qualification table and reusable evidence maps."""
    names = names or {}
    rows = []
    quality_by_symbol = {}
    metrics_by_symbol = {}
    for symbol, episode in episodes.items():
        record = production_store.get(symbol, {})
        needs_position = episode.get("shadow_stage") in MATURE_STAGES
        metrics = stock_metrics(symbol, as_of, record) if needs_position else {}
        route = str(record.get("route") or "UNCLASSIFIED")
        quality = evaluate(episode, metrics, route, record.get("state", ""))
        metrics_by_symbol[symbol] = metrics
        quality_by_symbol[symbol] = quality
        rows.append({
            "as_of": as_of, "symbol": symbol, "name": names.get(symbol, ""),
            "industry": D.industry_map().get(symbol, ""),
            "concepts": "|".join(D.concepts_map(as_of).get(symbol, [])),
            "route": route, "production_state": record.get("state", episode.get("production_state", "")),
            "shadow_stage": episode.get("shadow_stage", ""),
            "retention_state": episode.get("retention_state", ""),
            "episode_id": episode.get("episode_id", ""),
            "event_first_seen": episode.get("evidence", {}).get("event", {}).get("first_seen", ""),
            "supply_first_seen": episode.get("evidence", {}).get("supply", {}).get("first_seen", ""),
            "supply_last_seen": episode.get("evidence", {}).get("supply", {}).get("last_seen", ""),
            "structure_first_seen": episode.get("evidence", {}).get("structure", {}).get("first_seen", ""),
            "structure_last_seen": episode.get("evidence", {}).get("structure", {}).get("last_seen", ""),
            **metrics, **quality,
        })
    frame = pd.DataFrame(rows)
    if len(frame):
        for column in ("trigger_distance_pct", "stop_distance_pct"):
            if column not in frame:
                frame[column] = np.nan
        frame = frame.assign(
            _status=frame["carrier_qualification"].map(STATUS_ORDER).fillna(99),
            _stage=frame["shadow_stage"].map(STAGE_ORDER).fillna(99),
            _route=frame["route"].map(ROUTE_ORDER).fillna(99),
        ).sort_values(
            ["_status", "_stage", "trigger_distance_pct", "stop_distance_pct", "_route", "symbol"],
            na_position="last",
        ).drop(columns=["_status", "_stage", "_route"])
    return frame, quality_by_symbol, metrics_by_symbol


def write_report(frame: pd.DataFrame, as_of: str, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "carrier_qualification.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    publish(csv_path, as_of)
    counts = frame["carrier_qualification"].value_counts().to_dict() if len(frame) else {}
    summary = {
        "as_of": as_of, "version": VERSION, "total_episodes": len(frame),
        "qualified_lurk": int(counts.get("QUALIFIED_LURK", 0)),
        "building": int(counts.get("BUILDING", 0)),
        "extended_too_late": int(counts.get("EXTENDED_TOO_LATE", 0)),
        "distribution_risk": int(counts.get("DISTRIBUTION_RISK", 0)),
        "status_counts": {key: int(value) for key, value in counts.items()},
        "weighted_score_used": False,
    }
    (output / "carrier_qualification_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    columns = [
        "carrier_qualification", "carrier_tier", "symbol", "name", "industry", "concepts", "route",
        "shadow_stage", "retention_state", "close", "r5", "r20", "ma20_bias",
        "drawdown20", "repair5", "trigger_distance_pct", "stop_distance_pct",
        "execution_state", "gate_failures",
    ]
    visible = frame[frame["carrier_qualification"].isin([
        "QUALIFIED_LURK", "BUILDING", "EXECUTION_BLOCKED", "EXTENDED_TOO_LATE",
        "DISTRIBUTION_RISK",
    ])].head(80) if len(frame) else frame
    lines = [
        f"# Carrier Qualification  {as_of}", "",
        f"- Version: `{VERSION}`; hard gates are used, weighted averaging is not used.",
        "- Qualification requires: prior capital probe AND retained evidence AND shrinking supply "
        "AND executable position AND no extension/distribution/broken-state evidence.",
        "- QUALIFIED_LURK is carrier eligibility, not a buy signal. Final targets still require theme, "
        "leader/propagation and attention intersection in the P1 report.",
        "- Ranking starts only after qualification; route style changes interpretation, never repairs a failed gate.", "",
        f"- Qualified={summary['qualified_lurk']} / Building={summary['building']} / "
        f"Extended={summary['extended_too_late']} / Distribution risk={summary['distribution_risk']}", "",
        visible[[column for column in columns if column in visible.columns]].to_markdown(index=False)
        if len(visible) else "No carrier evidence.",
    ]
    (output / "carrier_qualification.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
