"""Read-only archive of reviewed research selections, separate from daily signals."""
from __future__ import annotations

import json
from pathlib import Path


REGISTRY = Path(__file__).with_name("selection_board.json")
GROUPS = {
    "CORE_WATCH": "核心观察",
    "CONDITIONAL_WATCH": "条件观察",
    "WATCH": "研究观察",
    "RESERVE": "潜伏预备",
}


def selection_list(path: Path = REGISTRY) -> dict:
    registry = json.loads(path.read_text(encoding="utf-8"))
    records = {}
    batches = []
    # Review order sets display priority; earlier snapshots never disappear.
    for batch_index, batch in enumerate(registry["batches"]):
        info = {key: batch[key] for key in ("id", "title", "as_of", "recorded_at", "report")}
        batches.append({**info, "count": len(batch["records"])})
        for position, entry in enumerate(batch["records"]):
            symbol = entry["symbol"]
            selection = {**info, **entry, "group_label": GROUPS[entry["group"]]}
            history = records.get(symbol, {}).get("selections", [])
            records[symbol] = {
                **entry, "group_label": selection["group_label"],
                "as_of": batch["as_of"], "recorded_at": batch["recorded_at"],
                "selections": [*history, selection],
                "_order": (list(GROUPS).index(entry["group"]), -batch_index, position),
            }
    rows = sorted(records.values(), key=lambda row: row["_order"])
    for row in rows:
        row.pop("_order")
    return {
        "version": registry["version"], "updated_at": registry["updated_at"],
        "scope": registry["scope"], "permission": "RESEARCH_ONLY_NO_ORDER",
        "batches": batches, "records": rows, "count": len(rows),
    }
