"""As-of follow-up review for proposed and registered theme carriers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import capital_episode as CE
from . import data as D
from .theme_registry import themes_as_of


VERSION = "recent_review_v1_2_20260829"
ORDER = {
    "NEAR_CANDIDATE": 0, "BUILDING": 1, "POSITION_BLOCKED": 2,
    "EXTENDED_WAIT_PULLBACK": 3, "EXTENDED_PROBE": 4,
    "PROBED_NEEDS_RETENTION": 5, "REJECTED_RISK": 6,
    "NO_CAPITAL_EPISODE": 7, "THEME_INACTIVE": 8,
}


def _read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _metrics(symbol: str, as_of: str) -> dict:
    frame = D.load_tail(symbol, 80, as_of=as_of)
    if frame is None or str(frame["date"].iloc[-1]) != as_of:
        return {"data_fresh": False}
    close = frame["close"].to_numpy(float)
    amount = frame["amount"].to_numpy(float)
    ma20 = pd.Series(close).rolling(20).mean().iloc[-1]
    return {
        "data_fresh": True, "close": round(close[-1], 2),
        "r3": round((close[-1] / close[-4] - 1) * 100, 2) if len(close) >= 4 else np.nan,
        "r5": round((close[-1] / close[-6] - 1) * 100, 2) if len(close) >= 6 else np.nan,
        "r20": round((close[-1] / close[-21] - 1) * 100, 2) if len(close) >= 21 else np.nan,
        "ma20_bias": round((close[-1] / ma20 - 1) * 100, 2),
        "amount_ratio": round(amount[-5:].mean() / amount[-20:].mean(), 2)
        if len(amount) >= 20 and amount[-20:].mean() > 0 else np.nan,
    }


def _follow_state(episode: dict, metrics: dict, theme_state: str = "",
                  quality: dict | None = None) -> str:
    if theme_state in {"CROWDED", "RETREAT"}:
        return "THEME_INACTIVE"
    qualification = (quality or {}).get("carrier_qualification", "")
    mapped = {
        "QUALIFIED_LURK": "NEAR_CANDIDATE",
        "BUILDING": "BUILDING",
        "EXECUTION_BLOCKED": "POSITION_BLOCKED",
        "EXTENDED_TOO_LATE": "EXTENDED_WAIT_PULLBACK",
        "DISTRIBUTION_RISK": "REJECTED_RISK",
        "EXITED_OR_BROKEN": "REJECTED_RISK",
        "TRIGGERED_TOO_LATE": "REJECTED_RISK",
        "RETENTION_UNPROVEN": "PROBED_NEEDS_RETENTION",
        "SUPPLY_UNPROVEN": "PROBED_NEEDS_RETENTION",
        "PROBED_ONLY": "PROBED_NEEDS_RETENTION",
    }
    if qualification in mapped:
        return mapped[qualification]
    stage = episode.get("shadow_stage", "")
    retained = episode.get("retention_state") == "RETAINED"
    r5 = float(metrics.get("r5", np.nan))
    r20 = float(metrics.get("r20", np.nan))
    bias = float(metrics.get("ma20_bias", np.nan))
    extended = (
        (np.isfinite(r5) and r5 > 15)
        or (np.isfinite(r20) and r20 > 30)
        or (np.isfinite(bias) and bias > 12)
    )
    if retained and stage in {"COILED", "ARMED", "READY"} and not extended:
        return "NEAR_CANDIDATE"
    if retained and stage == "ABSORBING" and not extended:
        return "BUILDING"
    if retained and stage in {"ABSORBING", "COILED", "ARMED", "READY"}:
        return "EXTENDED_WAIT_PULLBACK"
    if stage == "PROBED":
        if extended:
            return "EXTENDED_PROBE"
        return "PROBED_NEEDS_RETENTION"
    return "NO_CAPITAL_EPISODE"


def build(as_of: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    compact = as_of.replace("-", "")
    output = Path("results/mainwave_engine") / compact
    mapping = _read(Path("concept_mapping.json"), {}).get("mapping", {})
    capital = CE.load_store()["active"]
    quality_path = output / "carrier_qualification.csv"
    quality_frame = (pd.read_csv(quality_path, encoding="utf-8-sig")
                     if quality_path.exists() else pd.DataFrame())
    quality_by_symbol = {
        row["symbol"]: row for row in quality_frame.to_dict("records")
    } if len(quality_frame) else {}
    proposals_path = output / "theme_proposals.csv"
    proposals = pd.read_csv(proposals_path, encoding="utf-8-sig") if proposals_path.exists() else pd.DataFrame()
    lifecycle_path = output / "theme_episodes.csv"
    lifecycle = pd.read_csv(lifecycle_path, encoding="utf-8-sig") if lifecycle_path.exists() else pd.DataFrame()
    lifecycle_by_label = {
        row["theme_label"]: row for row in lifecycle.to_dict("records")
    } if len(lifecycle) else {}
    proposal_rows = []
    if len(proposals):
        priority = proposals.get("research_priority", pd.Series("BASE", index=proposals.index))
        selected = proposals[
            (proposals["proposal_state"] == "PROPOSED") | (priority == "HIGH")
        ]
        for proposal in selected.to_dict("records"):
            symbols = [value for value in str(proposal.get("suggested_members", "")).split("|")
                       if value and value != "nan"]
            if not symbols:
                try:
                    evidence = json.loads(proposal.get("member_evidence", "[]"))
                except Exception:
                    evidence = []
                symbols = [item.get("symbol", "") for item in evidence if item.get("symbol")]
            for symbol in dict.fromkeys(symbols):
                if not symbol or symbol == "nan":
                    continue
                metrics = _metrics(symbol, as_of)
                episode = capital.get(symbol, {})
                quality = quality_by_symbol.get(symbol, {})
                proposal_rows.append({
                    "as_of": as_of, "source_class": "THEME_PROPOSAL",
                    "theme": proposal["theme_label"], "symbol": symbol,
                    "name": mapping.get(symbol, {}).get("name", symbol),
                    "theme_state": proposal.get("proposal_state", ""),
                    "research_priority": proposal.get("research_priority", "BASE"),
                    "calendar_theme": proposal.get("calendar_theme_label", ""),
                    "external_catalyst_state": proposal.get("external_catalyst_state", "NONE"),
                    **metrics, "shadow_stage": episode.get("shadow_stage", ""),
                    "retention_state": episode.get("retention_state", ""),
                    "production_state": episode.get("production_state", ""),
                    "remaining_sessions": episode.get("valid_for_sessions", ""),
                    "carrier_qualification": quality.get("carrier_qualification", "DATA_INCOMPLETE"),
                    "carrier_tier": quality.get("carrier_tier", "NOT_QUALIFIED"),
                    "gate_failures": quality.get("gate_failures", "QUALIFICATION_OUTPUT_MISSING"),
                    "trigger_distance_pct": quality.get("trigger_distance_pct"),
                    "stop_distance_pct": quality.get("stop_distance_pct"),
                    "follow_state": _follow_state(episode, metrics, quality=quality),
                    "invalidation": "retention lost; structure failed; proposal rejected or expired",
                })
    proposal_frame = pd.DataFrame(proposal_rows)
    if len(proposal_frame):
        proposal_frame = proposal_frame.assign(
            _order=proposal_frame["follow_state"].map(ORDER)
        ).sort_values(["_order", "ma20_bias", "r20"]).drop(columns="_order")

    chain_rows = []
    for theme in themes_as_of(as_of):
        theme_context = lifecycle_by_label.get(theme["label"], {})
        theme_state = theme_context.get("theme_lifecycle", "UNOBSERVED")
        for symbol, name in theme["members"].items():
            metrics = _metrics(symbol, as_of)
            episode = capital.get(symbol, {})
            quality = quality_by_symbol.get(symbol, {})
            chain_rows.append({
                "as_of": as_of, "source_class": "REGISTERED_CHAIN", "theme": theme["label"],
                "symbol": symbol, "name": name, "theme_state": theme_state, **metrics,
                "research_priority": "REGISTERED_CHAIN",
                "shadow_stage": episode.get("shadow_stage", ""),
                "retention_state": episode.get("retention_state", ""),
                "production_state": episode.get("production_state", ""),
                "remaining_sessions": episode.get("valid_for_sessions", ""),
                "carrier_qualification": quality.get("carrier_qualification", "DATA_INCOMPLETE"),
                "carrier_tier": quality.get("carrier_tier", "NOT_QUALIFIED"),
                "gate_failures": quality.get("gate_failures", "QUALIFICATION_OUTPUT_MISSING"),
                "trigger_distance_pct": quality.get("trigger_distance_pct"),
                "stop_distance_pct": quality.get("stop_distance_pct"),
                "follow_state": _follow_state(episode, metrics, theme_state, quality),
                "invalidation": "retention lost; chain RETREAT/CROWDED; structure failed",
            })
    chain_frame = pd.DataFrame(chain_rows)
    if len(chain_frame):
        chain_frame = chain_frame.assign(
            _order=chain_frame["follow_state"].map(ORDER)
        ).sort_values(["theme", "_order", "ma20_bias", "r20"]).drop(columns="_order")

    proposal_frame.to_csv(output / "proposal_carrier_followup.csv", index=False, encoding="utf-8-sig")
    chain_frame.to_csv(output / "registered_chain_followup.csv", index=False, encoding="utf-8-sig")
    display = ["research_priority", "theme", "theme_state", "symbol", "name", "follow_state",
               "carrier_qualification", "carrier_tier", "shadow_stage", "retention_state",
               "close", "r3", "r5", "r20", "ma20_bias", "trigger_distance_pct",
               "stop_distance_pct", "gate_failures"]
    lines = [
        f"# Recent Mainwave Follow-up  {as_of}", "",
        f"- Version: `{VERSION}`. As-of metrics only; no future data and no order generation.",
        "- Priority is categorical, not a weighted score.", "",
        "## High-priority proposal and research carriers", "",
        proposal_frame[display].to_markdown(index=False) if len(proposal_frame) else "None.", "",
        "## Registered-chain carriers closest to retention", "",
        chain_frame[~chain_frame["follow_state"].isin({"NO_CAPITAL_EPISODE", "THEME_INACTIVE"})][display].to_markdown(index=False)
        if len(chain_frame) and (~chain_frame["follow_state"].isin({"NO_CAPITAL_EPISODE", "THEME_INACTIVE"})).any() else "None.",
    ]
    (output / "recent_followup.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[recent_review] proposals={len(proposal_frame)} registered={len(chain_frame)} -> {output/'recent_followup.md'}")
    return proposal_frame, chain_frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    date = args.date if "-" in args.date else f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}"
    build(date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
