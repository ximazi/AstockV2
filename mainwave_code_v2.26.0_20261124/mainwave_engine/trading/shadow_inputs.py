"""Inputs the shadow layer reads from other artefacts (previous review facts, route scan prices, the paper book,
the pinned list) and the two artefacts it assembles from them: entry-contract drafts and the discovery register.
Everything read here is a frozen file of an earlier step; nothing is recomputed from bars."""
import json
from pathlib import Path

import pandas as pd

from datetime import datetime, timedelta

from .. import discovery_events, entry_contracts


def next_session(sessions, day):
    """Next market session after `day` from the calendar; weekday walk when the calendar ends."""
    later = [d for d in sessions if d > day]
    if later:
        return later[0]
    d = datetime.fromisoformat(day) + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d.date().isoformat()


def read(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}


def previous_review_facts(root, prior_day):
    """The previous session's eod_review facts (themes, sentiment, loss effects), dated by that session."""
    if not prior_day:
        return {}
    pointer = read(Path(root) / "results/mainwave_engine" / prior_day / "eod_review/latest.json")
    if not pointer.get("path"):
        return {}
    facts = read(Path(pointer["path"]) / "facts.json")
    if not facts:
        return {}
    themes = dict(facts.get("themes") or {})
    themes["as_of"] = facts.get("as_of")
    return {"as_of": facts.get("as_of"), "themes": themes, "sentiment": facts.get("sentiment") or {}, "loss_effects": facts.get("loss_effects") or {}}


def contract_drafts(snap, day, sessions, output):
    """Daily-bar shadow drafts (B1, B3) for every record; the route scan's invalid price fills the B1 gap when known."""
    route_rows = {}
    path = Path(output) / "route_candidates.csv"
    if path.exists():
        frame = pd.read_csv(path, encoding="utf-8-sig")
        keep = [c for c in ("symbol", "invalid_price", "boundary_price", "frozen_atr_price") if c in frame.columns]
        route_rows = {r["symbol"]: r for r in frame[keep].to_dict("records")} if "symbol" in keep else {}
    target = next_session(sessions or [], day)
    drafts = []
    for record in snap.get("records", []):
        drafts.extend(entry_contracts.drafts(record, day, target, route_rows.get(record["symbol"])))
    return {"version": entry_contracts.VERSION, "as_of": day, "effective_trade_date": target, "contracts": entry_contracts.CONTRACTS,
            "summary": entry_contracts.summarize(drafts), "drafts": drafts,
            "note": "影子草案：描述冻结规则不知道的下一交易时点参与方式；无批准、无订单、不进账本"}


def discovery_register(root, day, snap, shadow, watch, auto, sessions):
    """Union of every source the system saw today, with yesterday's register for continuity."""
    root = Path(root)
    tracking = read(root / "fupan_tracking.json")
    prior_day = ""
    folder = root / "results/mainwave_engine"
    for p in sorted(folder.glob("2026????"), reverse=True):
        if p.name < day.replace("-", "") and (p / "discovery_events.json").exists():
            prior_day = p.name
            break
    previous = read(folder / prior_day / "discovery_events.json") if prior_day else {}
    target = next_session(sessions or [], day)
    plans = [{"symbol": p["symbol"], "name": p.get("name"), "account_id": "AUTO100W", "plan_id": p.get("detail"),
              "approved": p.get("status") == "ORDER_APPROVED", "expired": False, "open_position": False}
             for p in (auto or {}).get("placed", []) if p.get("symbol")]
    return discovery_events.build(day, snap.get("records", []), (shadow or {}).get("rows"), watch, tracking.get("pinned"), plans, previous)
