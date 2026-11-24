"""Categorical theme-calendar overlay for forward mainwave observation."""
from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import pandas as pd

from . import recent_review as REVIEW


VERSION = "theme_horizon_v1_2_20260829"
CONFIG = Path("mainwave_theme_calendar.json")
OUT_ROOT = Path("results/mainwave_engine")
THEME_STATE = OUT_ROOT / "theme_state_store.json"

PRIORITY_ORDER = {
    "NOW_FOCUS": 0,
    "BUILD_WATCH": 1,
    "CALENDAR_WATCH": 2,
    "REPAIR_ONLY": 3,
    "CLOSED": 4,
    "UNAVAILABLE": 5,
}


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def calendar_state(as_of: str, start: str, end: str, preheat_days: int) -> str:
    today = pd.Timestamp(as_of)
    first = pd.Timestamp(start)
    last = pd.Timestamp(end)
    if today > last:
        return "EXPIRED"
    if first <= today <= last:
        return "ACTIVE"
    if first - timedelta(days=int(preheat_days)) <= today < first:
        return "PREHEAT"
    return "FORWARD"


def decide(calendar: str, market: str, carrier: str, repair: bool = False) -> tuple[str, str]:
    if repair:
        return "REPAIR_ONLY", "REPAIR_ONLY"
    if calendar == "EXPIRED":
        return "EXPIRED", "CLOSED"
    if carrier in {"NEAR_CANDIDATE", "BUILDING"}:
        if calendar == "ACTIVE":
            return "ACTIVE_WINDOW_WITH_CARRIER", "NOW_FOCUS"
        if calendar == "PREHEAT":
            return "PREHEAT_WITH_CARRIER", "NOW_FOCUS"
        return "EARLY_CARRIER_BEFORE_WINDOW", "BUILD_WATCH"
    if carrier == "EXTENDED_WAIT_PULLBACK":
        return f"{calendar}_EXTENDED_WAIT_PULLBACK", "BUILD_WATCH"
    if market in {"PROPOSED", "WAIT_CARRIER", "REGISTERED_ACTIVE"}:
        return f"{calendar}_WAIT_CARRIER", "BUILD_WATCH"
    return f"{calendar}_CALENDAR_ONLY", "CALENDAR_WATCH"


def external_catalyst_state(theme: dict, as_of: str) -> str:
    catalyst = theme.get("external_catalyst", {})
    if not catalyst:
        return "NONE"
    if as_of < str(catalyst.get("as_of", "9999-12-31")):
        return "NOT_YET_KNOWN"
    if as_of > str(catalyst.get("valid_until", "0000-01-01")):
        return "EXPIRED"
    if as_of >= str(catalyst.get("review_on", "9999-12-31")):
        return "STALE_REVIEW_REQUIRED"
    return str(catalyst.get("state", "NONE"))


def research_priority(priority_band: str, catalyst_state: str, calendar: str) -> str:
    if priority_band == "REPAIR_ONLY":
        return "REPAIR"
    if catalyst_state == "CONFIRMED_STRONG" and calendar in {"ACTIVE", "PREHEAT"}:
        return "HIGH"
    if priority_band == "NOW_FOCUS":
        return "HIGH"
    if priority_band == "BUILD_WATCH":
        return "MEDIUM"
    return "BASE"


def _match(value: str, aliases: list[str]) -> bool:
    folded = str(value).casefold()
    return any(alias.casefold() in folded or folded in alias.casefold() for alias in aliases)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()


def _proposal_evidence(theme: dict, proposals: pd.DataFrame, followup: pd.DataFrame,
                       as_of: str) -> dict:
    aliases = theme.get("aliases", [])
    if proposals.empty or "theme_label" not in proposals:
        return {"market_state": "NO_MARKET_EVIDENCE", "carrier_state": "NONE",
                "carrier_count": 0, "carriers": "", "matched_themes": ""}
    matched = proposals[proposals["theme_label"].map(lambda value: _match(value, aliases))]
    if matched.empty:
        return {"market_state": "NO_MARKET_EVIDENCE", "carrier_state": "NONE",
                "carrier_count": 0, "carriers": "", "matched_themes": ""}

    states = set(matched["proposal_state"].astype(str))
    if "PROPOSED" in states:
        market_state = "PROPOSED"
    elif "WAIT_CARRIER" in states:
        market_state = "WAIT_CARRIER"
    elif "REGISTERED" in states:
        market_state = "REGISTERED_ACTIVE"
    else:
        market_state = "OBSERVE"

    matched_labels = set(matched["theme_label"].astype(str))
    carrier_rows = followup[followup["theme"].astype(str).isin(matched_labels)] if len(followup) else pd.DataFrame()
    follow_states = set(carrier_rows.get("follow_state", pd.Series(dtype=str)).astype(str))
    if "NEAR_CANDIDATE" in follow_states:
        carrier_state = "NEAR_CANDIDATE"
    elif "BUILDING" in follow_states:
        carrier_state = "BUILDING"
    elif "EXTENDED_WAIT_PULLBACK" in follow_states:
        carrier_state = "EXTENDED_WAIT_PULLBACK"
    else:
        carrier_state = "NONE"

    carriers: list[str] = []
    if len(carrier_rows):
        order = {"NEAR_CANDIDATE": 0, "BUILDING": 1, "EXTENDED_WAIT_PULLBACK": 2,
                 "EXTENDED_PROBE": 3, "PROBED_NEEDS_RETENTION": 4}
        display = carrier_rows.assign(
            _order=carrier_rows["follow_state"].map(order).fillna(9)
        ).sort_values(["_order", "ma20_bias"]).head(3)
        carriers.extend(
            f"{row.symbol} {row['name']} {row.follow_state}"
            for _, row in display.iterrows()
        )
    if not carriers:
        for _, row in matched.iterrows():
            try:
                evidence = json.loads(row.get("member_evidence", "[]"))
            except Exception:
                evidence = []
            for item in evidence:
                marker = f"{item.get('shadow_stage', '')}/{item.get('retention_state', '')}".strip("/")
                metrics = REVIEW._metrics(item.get("symbol", ""), as_of)
                r5 = float(metrics.get("r5", 0) or 0)
                r20 = float(metrics.get("r20", 0) or 0)
                bias = float(metrics.get("ma20_bias", 0) or 0)
                if r5 > 15 or r20 > 30 or bias > 12:
                    marker = f"EXTENDED_PROBE(r20={r20:.1f}%,bias={bias:.1f}%)"
                carriers.append(f"{item.get('symbol', '')} {item.get('name', '')} {marker}".strip())
                if len(carriers) == 3:
                    break
            if len(carriers) == 3:
                break
    return {
        "market_state": market_state,
        "carrier_state": carrier_state,
        "carrier_count": int(matched.get("retained_mature_carriers", pd.Series(dtype=float)).fillna(0).sum()),
        "carriers": " | ".join(carriers),
        "matched_themes": "|".join(sorted(matched_labels)),
    }


def _repair_evidence(theme: dict, state_store: dict) -> dict:
    chain_states = state_store.get("themes", {})
    evidence = []
    for theme_id in theme.get("registered_theme_ids", []):
        row = chain_states.get(theme_id, {})
        if not row:
            continue
        drawdown = row.get("drawdown_from_120d_peak_pct", "")
        evidence.append(f"{theme_id}:{row.get('stage', 'UNKNOWN')}/dd={drawdown}%/C{row.get('cycle', '')}")
    return {
        "market_state": "DAMAGED_PRIOR_LEADERS",
        "carrier_state": "REPAIR_CONFIRMATION_REQUIRED",
        "carrier_count": 0,
        "carriers": " | ".join(evidence),
        "matched_themes": "|".join(theme.get("registered_theme_ids", [])),
    }


def _month_map(config: dict) -> dict[str, list[str]]:
    result = {}
    for month in range(9, 13):
        month_start = pd.Timestamp(2026, month, 1)
        month_end = month_start + pd.offsets.MonthEnd(0)
        labels = []
        for theme in config.get("themes", []):
            if pd.Timestamp(theme["window_start"]) <= month_end and pd.Timestamp(theme["window_end"]) >= month_start:
                labels.append(theme["label"])
        result[f"2026-{month:02d}"] = labels
    return result


def build(as_of: str) -> tuple[pd.DataFrame, dict]:
    config = _read_json(CONFIG, {})
    output = OUT_ROOT / as_of.replace("-", "")
    output.mkdir(parents=True, exist_ok=True)
    effective_from = str(config.get("effective_from", "9999-12-31"))
    if config.get("schema_version") != 1 or as_of < effective_from:
        frame = pd.DataFrame()
        summary = {
            "as_of": as_of, "version": VERSION, "status": "CONFIG_NOT_POINT_IN_TIME",
            "config_version": config.get("version", ""), "effective_from": effective_from,
            "counts": {}, "focus_labels": [], "month_map": _month_map(config),
        }
        frame.to_csv(output / "theme_horizon.csv", index=False, encoding="utf-8-sig")
        (output / "theme_horizon_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output / "theme_horizon.md").write_text(
            f"# Theme Horizon  {as_of}\n\nConfig is unavailable before {effective_from}.\n",
            encoding="utf-8",
        )
        return frame, summary

    proposals = _read_csv(output / "theme_proposals.csv")
    followup = _read_csv(output / "proposal_carrier_followup.csv")
    state_store = _read_json(THEME_STATE, {})
    rows = []
    for theme in config.get("themes", []):
        window = calendar_state(
            as_of, theme["window_start"], theme["window_end"], theme.get("preheat_days", 0)
        )
        repair = theme.get("mode") == "REPAIR"
        evidence = _repair_evidence(theme, state_store) if repair else _proposal_evidence(
            theme, proposals, followup, as_of
        )
        decision_state, priority = decide(
            window, evidence["market_state"], evidence["carrier_state"], repair
        )
        catalyst_state = external_catalyst_state(theme, as_of)
        rows.append({
            "as_of": as_of,
            "theme_id": theme["id"],
            "theme_label": theme["label"],
            "categories": "|".join(theme.get("categories", [])),
            "window_start": theme["window_start"],
            "window_end": theme["window_end"],
            "calendar_state": window,
            **evidence,
            "decision_state": decision_state,
            "priority_band": priority,
            "external_catalyst_state": catalyst_state,
            "research_priority": research_priority(priority, catalyst_state, window),
            "catalyst": theme.get("catalyst", ""),
            "transmission_focus": theme.get("transmission_focus", ""),
            "confirmation": theme.get("confirmation", ""),
            "invalidation": theme.get("invalidation", ""),
            "sources": "|".join(theme.get("sources", [])),
        })
    frame = pd.DataFrame(rows)
    frame = frame.assign(
        _order=frame["priority_band"].map(PRIORITY_ORDER).fillna(9)
    ).sort_values(["_order", "window_start", "theme_id"]).drop(columns="_order")
    frame.to_csv(output / "theme_horizon.csv", index=False, encoding="utf-8-sig")

    counts = {str(key): int(value) for key, value in frame["priority_band"].value_counts().items()}
    focus = frame[frame["priority_band"] == "NOW_FOCUS"]["theme_label"].tolist()
    high_research = frame[frame["research_priority"] == "HIGH"]["theme_label"].tolist()
    summary = {
        "as_of": as_of, "version": VERSION, "status": "SUCCESS",
        "config_version": config.get("version", ""), "effective_from": effective_from,
        "authored_on": config.get("authored_on", ""), "counts": counts,
        "focus_labels": focus, "high_research_labels": high_research,
        "month_map": _month_map(config),
        "order_generation": False, "automatic_registration": False,
        "application_scope": {
            "theme_proposal_annotation_and_sorting": True,
            "high_research_probe_review": True,
            "p1_candidate_requires_carrier_hard_gates": True,
            "execution_permission_unchanged": True
        },
    }
    (output / "theme_horizon_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    display = ["research_priority", "priority_band", "theme_label", "calendar_state",
               "external_catalyst_state", "market_state", "carrier_state",
               "decision_state", "carriers"]
    lines = [
        f"# Theme Horizon  {as_of}", "",
        f"- Version: `{VERSION}`; config `{config.get('version', '')}` (effective {effective_from}).",
        "- Calendar, market attention, carrier maturity and repair state are separate categorical facts.",
        "- This report neither registers themes nor creates orders. Calendar-only themes must wait for a carrier.", "",
        "## Current classification", "",
        frame[display].to_markdown(index=False) if len(frame) else "None.", "",
        "## September to December map", "",
    ]
    for month, labels in summary["month_map"].items():
        lines.append(f"- {month}: {', '.join(labels)}")
    lines.extend(["", "## Confirmation and invalidation", ""])
    for _, row in frame.iterrows():
        lines.extend([
            f"### {row['theme_label']}", "",
            f"- Catalyst: {row['catalyst']}",
            f"- Confirm: {row['confirmation']}",
            f"- Invalidate: {row['invalidation']}", "",
        ])
    (output / "theme_horizon.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[theme_horizon] focus={len(focus)} themes={len(frame)} -> {output/'theme_horizon.md'}")
    return frame, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    date = args.date if "-" in args.date else f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}"
    build(date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
