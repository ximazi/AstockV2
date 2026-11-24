"""Point-in-time theme proposal and explicit registration workflow."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from . import capital_episode as CE
from . import carrier_quality as CQ
from . import theme_horizon as HORIZON
from . import theme_semantics as TS
from .theme_registry import load_registry, themes_as_of


VERSION = "theme_proposal_v1_4_20260829"
STORE = Path("results/mainwave_engine/theme_proposal_store.json")
REGISTRY = Path("theme_registry.json")
STAGE_ORDER = {"READY": 0, "ARMED": 1, "COILED": 2, "ABSORBING": 3, "PROBED": 4}


def _read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _optional(value, default=None):
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return default
    return value


def _atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _proposal_id(name: str) -> str:
    return f"theme_{hashlib.sha1(name.encode('utf-8')).hexdigest()[:10]}"


def _mapping_status(generated: str, as_of: str) -> str:
    try:
        age = (pd.Timestamp(as_of) - pd.Timestamp(generated[:10])).days
    except Exception:
        return "MISSING"
    if age < 0:
        return "FUTURE_UNAVAILABLE"
    return "FRESH" if age <= 30 else "STALE"


def _registered(name: str, as_of: str) -> str:
    folded = name.casefold()
    for theme in themes_as_of(as_of):
        aliases = theme.get("attention_aliases", []) + [theme["label"]]
        if any(alias.casefold() in folded or folded in alias.casefold() for alias in aliases):
            return theme["id"]
    return ""


def _calendar_context(name: str, as_of: str, proposal_state: str,
                      config: dict) -> dict:
    if as_of < str(config.get("effective_from", "9999-12-31")):
        return {
            "calendar_theme_id": "", "calendar_theme_label": "", "calendar_state": "NONE",
            "external_catalyst_state": "NOT_YET_KNOWN", "catalyst_review_on": "",
            "research_priority": "HIGH" if proposal_state == "PROPOSED" else "BASE",
        }
    matches = []
    for theme in config.get("themes", []):
        if not HORIZON._match(name, theme.get("aliases", [])):
            continue
        calendar = HORIZON.calendar_state(
            as_of, theme["window_start"], theme["window_end"], theme.get("preheat_days", 0)
        )
        external = HORIZON.external_catalyst_state(theme, as_of)
        if external == "CONFIRMED_STRONG" and calendar in {"ACTIVE", "PREHEAT"}:
            research = "HIGH"
        elif proposal_state == "PROPOSED":
            research = "HIGH"
        elif calendar in {"ACTIVE", "PREHEAT"}:
            research = "MEDIUM"
        else:
            research = "BASE"
        matches.append({
            "calendar_theme_id": theme["id"],
            "calendar_theme_label": theme["label"],
            "calendar_state": calendar,
            "external_catalyst_state": external,
            "catalyst_review_on": theme.get("external_catalyst", {}).get("review_on", ""),
            "research_priority": research,
        })
    if not matches:
        return {
            "calendar_theme_id": "", "calendar_theme_label": "", "calendar_state": "NONE",
            "external_catalyst_state": "NONE", "catalyst_review_on": "",
            "research_priority": "HIGH" if proposal_state == "PROPOSED" else "BASE",
        }
    order = {"HIGH": 0, "MEDIUM": 1, "BASE": 2}
    return sorted(matches, key=lambda row: (order[row["research_priority"]], row["calendar_theme_id"]))[0]


def generate(as_of: str) -> tuple[pd.DataFrame, dict]:
    compact = as_of.replace("-", "")
    output = Path("results/mainwave_engine") / compact
    hotspot = _read(Path("hotspot_daily.json"), {})
    if str(hotspot.get("date", "")) != compact:
        concepts = {}
        data_status = "STALE_OR_MISSING"
    else:
        concepts = hotspot.get("concepts", {})
        data_status = "FRESH"
    mapping, mapping_meta = TS.load_mapping_as_of(as_of)
    mapping_generated = str(mapping_meta.get("generated", ""))[:10]
    mapping_status = _mapping_status(mapping_generated, as_of)
    capital = CE.load_store()
    quality_path = output / "carrier_qualification.csv"
    quality_frame = (pd.read_csv(quality_path, encoding="utf-8-sig")
                     if quality_path.exists() else pd.DataFrame())
    quality_by_symbol = {
        row["symbol"]: row for row in quality_frame.to_dict("records")
    } if len(quality_frame) else {}
    calendar_config = _read(HORIZON.CONFIG, {})
    rows = []
    for name, item in concepts.items():
        heat = float(item.get("heat_score", 0) or 0)
        limit_count = int(item.get("limit_count", 0) or 0)
        hotspot_members = set(item.get("stocks", []))
        member_rows = []
        for symbol, episode in capital["active"].items():
            concepts_for_stock = mapping.get(symbol, {}).get("concepts", [])
            if symbol not in hotspot_members and (
                    mapping_status != "FRESH" or name not in concepts_for_stock):
                continue
            quality = quality_by_symbol.get(symbol, {})
            member_rows.append({
                "symbol": symbol, "name": mapping.get(symbol, {}).get("name", ""),
                "shadow_stage": episode.get("shadow_stage", ""),
                "retention_state": episode.get("retention_state", ""),
                "capital_episode_id": episode.get("episode_id", ""),
                "carrier_qualification": _optional(
                    quality.get("carrier_qualification"), "DATA_INCOMPLETE"),
                "carrier_tier": _optional(quality.get("carrier_tier"), "NOT_QUALIFIED"),
                "route": _optional(quality.get("route"), "UNCLASSIFIED"),
                "trigger_distance_pct": _optional(quality.get("trigger_distance_pct")),
                "stop_distance_pct": _optional(quality.get("stop_distance_pct")),
                "r5": _optional(quality.get("r5")),
                "r20": _optional(quality.get("r20")),
                "ma20_bias": _optional(quality.get("ma20_bias")),
                "gate_failures": _optional(
                    quality.get("gate_failures"), "QUALIFICATION_OUTPUT_MISSING"),
                "source": "hotspot_limit_member" if symbol in hotspot_members else "concept_mapping",
            })
        member_rows.sort(key=lambda row: (
            CQ.STATUS_ORDER.get(row["carrier_qualification"], 99),
            STAGE_ORDER.get(row["shadow_stage"], 9),
            row["trigger_distance_pct"] if pd.notna(row["trigger_distance_pct"]) else 999,
            row["stop_distance_pct"] if pd.notna(row["stop_distance_pct"]) else 999,
            CQ.ROUTE_ORDER.get(row["route"], 99),
            row["symbol"],
        ))
        qualified = [row for row in member_rows
                     if row["carrier_qualification"] == "QUALIFIED_LURK"]
        registered_id = _registered(name, as_of)
        hotspot_active = heat >= 25 and limit_count >= 2
        if registered_id:
            state = "REGISTERED"
        elif hotspot_active and qualified:
            state = "PROPOSED"
        elif hotspot_active:
            state = "WAIT_CARRIER"
        else:
            state = "OBSERVE"
        calendar_context = _calendar_context(name, as_of, state, calendar_config)
        rows.append({
            "as_of": as_of, "proposal_id": _proposal_id(name), "theme_label": name,
            "proposal_state": state, "registered_theme_id": registered_id,
            "heat_score": heat, "limit_count": limit_count,
            "hotspot_lifecycle": item.get("lifecycle", ""),
            "retained_mature_carriers": len(qualified),
            "qualified_lurk_carriers": len(qualified),
            "suggested_members": "|".join(row["symbol"] for row in qualified[:12]),
            "member_evidence": json.dumps(member_rows[:20], ensure_ascii=False),
            "source": ("hotspot_daily+capital_episode"
                       if mapping_status != "FRESH"
                       else "hotspot_daily+capital_episode+concept_mapping"),
            "active_from_if_approved": as_of, "data_status": data_status,
            "mapping_generated": mapping_generated, "mapping_status": mapping_status,
            **calendar_context,
        })
    frame = pd.DataFrame(rows)
    if len(frame):
        order = {"PROPOSED": 0, "WAIT_CARRIER": 1, "REGISTERED": 2, "OBSERVE": 3}
        research_order = {"HIGH": 0, "MEDIUM": 1, "BASE": 2}
        frame = frame.assign(_order=frame["proposal_state"].map(order)).sort_values(
            ["research_priority", "_order", "heat_score", "limit_count"],
            key=lambda column: column.map(research_order) if column.name == "research_priority" else column,
            ascending=[True, True, False, False],
        ).drop(columns="_order")
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "theme_proposals.csv", index=False, encoding="utf-8-sig")

    previous = _read(STORE, {"_meta": {}, "proposals": {}})
    registry = previous.get("proposals", {})
    for row in frame.to_dict("records"):
        current = registry.get(row["proposal_id"], {})
        registry[row["proposal_id"]] = {
            **current, **row,
            "first_seen": current.get("first_seen", as_of), "last_seen": as_of,
        }
    _atomic(STORE, {"_meta": {"as_of": as_of, "version": VERSION}, "proposals": registry})
    summary = {
        "as_of": as_of, "version": VERSION, "data_status": data_status,
        "mapping_generated": mapping_generated, "mapping_status": mapping_status,
        "total": len(frame),
        "proposed": int((frame.get("proposal_state") == "PROPOSED").sum()) if len(frame) else 0,
        "wait_carrier": int((frame.get("proposal_state") == "WAIT_CARRIER").sum()) if len(frame) else 0,
        "registered": int((frame.get("proposal_state") == "REGISTERED").sum()) if len(frame) else 0,
        "auto_registered": 0,
    }
    (output / "theme_proposal_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    columns = ["research_priority", "proposal_state", "theme_label", "calendar_theme_label",
               "calendar_state", "external_catalyst_state", "heat_score", "limit_count",
               "qualified_lurk_carriers", "suggested_members", "proposal_id"]
    report_frame = frame[frame["proposal_state"] != "OBSERVE"].head(30) if len(frame) else frame
    lines = [
        f"# Theme Proposals  {as_of}", "",
        "- Proposals never alter the registry automatically.",
        "- PROPOSED requires a fresh active hotspot and at least one QUALIFIED_LURK carrier.",
        "- Carrier qualification uses hard gates; a high score cannot compensate for failed retention, supply, execution, or safety evidence.", "",
        report_frame[columns].to_markdown(index=False) if len(report_frame) else "No actionable proposals.",
    ]
    (output / "theme_proposals.md").write_text("\n".join(lines), encoding="utf-8")
    return frame, summary


def approve(as_of: str, proposal_id: str) -> dict:
    compact = as_of.replace("-", "")
    proposal_path = Path("results/mainwave_engine") / compact / "theme_proposals.csv"
    if not proposal_path.exists():
        raise RuntimeError(f"missing proposal file: {proposal_path}")
    frame = pd.read_csv(proposal_path, encoding="utf-8-sig")
    selected = frame[frame["proposal_id"] == proposal_id]
    if len(selected) != 1:
        raise RuntimeError(f"proposal not found or ambiguous: {proposal_id}")
    row = selected.iloc[0].to_dict()
    if row["proposal_state"] != "PROPOSED":
        raise RuntimeError(f"only PROPOSED may be approved, got {row['proposal_state']}")
    symbols = [value for value in str(row.get("suggested_members", "")).split("|") if value]
    if not symbols:
        raise RuntimeError("proposal has no QUALIFIED_LURK members")
    mapping, _ = TS.load_mapping_as_of(as_of)
    registry = load_registry(REGISTRY)
    entry = {
        "id": proposal_id, "label": row["theme_label"],
        "attention_aliases": [row["theme_label"]], "active_from": as_of,
        "source": f"explicit approval of {proposal_id}",
        "notes": "Point-in-time approval from theme proposal workflow; no historical backfill claim.",
        "historical_cycle_anchors": [],
        "members": {symbol: mapping.get(symbol, {}).get("name", symbol) for symbol in symbols},
        "member_evidence": {
            symbol: {"relationship": "hotspot_concept_membership", "source": row["source"],
                     "effective_from": as_of} for symbol in symbols
        },
    }
    registry["themes"].append(entry)
    _atomic(REGISTRY, registry)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--approve", default="")
    args = parser.parse_args()
    date = args.date if "-" in args.date else f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}"
    if args.approve:
        entry = approve(date, args.approve)
        print(f"[theme_proposal] registered {entry['id']} active_from={entry['active_from']}")
    else:
        frame, summary = generate(date)
        print(f"[theme_proposal] proposed={summary['proposed']} wait={summary['wait_carrier']} total={len(frame)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
