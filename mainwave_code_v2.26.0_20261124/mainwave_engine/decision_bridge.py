"""Report-only discovery review using the independent market context."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import market_context as MC


VERSION = "discovery_review_bridge_v2_20260907"
COLUMNS = [
    "as_of", "decision_id", "symbol", "name", "theme_id", "theme_episode_id",
    "capital_episode_id", "intersection_class", "role", "discovery_recorded",
    "l0_state", "l0_allow_execution", "l0_allow_review", "l0_cap", "permission",
    "trade_action", "permission_reason",
]


def _l0(as_of: str) -> dict:
    market = MC.load(as_of)
    return {"state": market["state"], "allow": market["trend_review_allowed"],
            "cap": None, "panic": int(market["temperature"] == "PANIC"),
            "source": "market_context.json", "mode": "ANALYSIS_ONLY"}


def build(as_of: str, focus: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    l0 = _l0(as_of)
    rows = []
    for row in focus.to_dict("records"):
        if not l0["allow"]:
            permission = "BLOCKED_BY_L0"
            reason = f"L0={l0['state']} panic={l0['panic']}"
        elif row.get("intersection_class") == "A":
            permission = "P2_SHADOW_ELIGIBLE"
            reason = "A confirmed and market permits review; analysis only, no order authority"
        else:
            permission = "WATCH_ONLY"
            reason = "B remains early discovery; preserve the candidate and await structural confirmation"
        rows.append({
            "as_of": as_of,
            "decision_id": f"{row['theme_episode_id']}|{row['symbol']}|{row['episode_id']}",
            "symbol": row["symbol"], "name": row.get("name", ""),
            "theme_id": row["theme_id"], "theme_episode_id": row["theme_episode_id"],
            "capital_episode_id": row["episode_id"],
            "intersection_class": row.get("intersection_class", ""), "role": row.get("role", ""),
            "discovery_recorded": True, "l0_state": l0["state"],
            "l0_allow_execution": False, "l0_allow_review": l0["allow"], "l0_cap": l0["cap"],
            "permission": permission, "trade_action": "NO_ORDER_OBSERVATION_ONLY",
            "permission_reason": reason,
        })
    frame = pd.DataFrame(rows, columns=COLUMNS)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "p1_decision_bridge.csv", index=False, encoding="utf-8-sig")
    counts = frame["permission"].value_counts().to_dict() if len(frame) else {}
    summary = {
        "as_of": as_of, "version": VERSION, "discoveries": len(frame),
        "l0": l0, "permissions": {key: int(value) for key, value in counts.items()},
        "orders_created": 0, "production_writeback": False,
    }
    (output_dir / "p1_decision_bridge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        f"# Discovery Review Context  {as_of}", "",
        f"- Market: `{l0['state']}` / allow_review={l0['allow']}",
        "- Discovery records are preserved. Legacy L0 column names are compatibility fields, not execution authority.",
        "- Canonical stock analysis: analysis_board.md. This bridge creates no orders.", "",
        frame.to_markdown(index=False) if len(frame) else "No P1 focus candidates.",
    ]
    (output_dir / "p1_decision_bridge.md").write_text("\n".join(lines), encoding="utf-8")
    return frame, summary
