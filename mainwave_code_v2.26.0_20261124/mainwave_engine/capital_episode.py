# -*- coding: utf-8 -*-
"""Observation-only capital episodes with independent evidence clocks."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


VERSION = "capital_episode_p1_v1_20260829"
STORE = Path("results/mainwave_engine/capital_episode_store.json")
HISTORY = Path("results/mainwave_engine/capital_episode_history.csv")

# Provisional observation windows. They are separate clocks, never a weighted score.
EVIDENCE_TTL = {"event": 60, "supply": 20, "structure": 10}
PRODUCTION_MATURE = {"ARMED", "READY", "TRIGGERED"}
PRODUCTION_FAILED = {"FAILED", "INVALID"}


def _truth(value) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _sessions_since(date: str, as_of: str, calendar: list[str]) -> int:
    if not date or date not in calendar or as_of not in calendar:
        return 10**6
    return calendar.index(as_of) - calendar.index(date)


def _date_by_age(as_of: str, age, calendar: list[str]) -> str:
    try:
        offset = max(0, int(float(age)))
    except (TypeError, ValueError):
        return as_of
    if as_of not in calendar:
        return as_of
    return calendar[max(0, calendar.index(as_of) - offset)]


def _event_date(row: dict, as_of: str, calendar: list[str]) -> str:
    dates = []
    for flag, age in (("imp", "imp_age"), ("fresh", "fresh_age"), ("pit", "pit_age")):
        if _truth(row.get(flag)):
            dates.append(_date_by_age(as_of, row.get(age), calendar))
    explicit = str(row.get("max_evidence_date") or "")
    if explicit and explicit <= as_of:
        dates.append(explicit)
    return min(dates) if dates else ""


def _touch(ledger: dict, kind: str, observed_at: str, confirmed_at: str) -> None:
    current = ledger.get(kind, {})
    current["first_seen"] = min(
        [x for x in (current.get("first_seen"), observed_at) if x], default=observed_at)
    current["last_seen"] = confirmed_at
    current["ttl_sessions"] = EVIDENCE_TTL[kind]
    current["invalidated_at"] = ""
    current["invalidation_reason"] = ""
    ledger[kind] = current


def _active(episode: dict, kind: str, as_of: str, calendar: list[str]) -> bool:
    item = episode.get("evidence", {}).get(kind, {})
    if item.get("invalidated_at"):
        return False
    anchor = item.get("first_seen") if kind == "event" else item.get("last_seen")
    return bool(anchor) and _sessions_since(anchor, as_of, calendar) <= EVIDENCE_TTL[kind]


def _remaining(episode: dict, kind: str, as_of: str, calendar: list[str]) -> int:
    item = episode.get("evidence", {}).get(kind, {})
    anchor = item.get("first_seen") if kind == "event" else item.get("last_seen")
    if not anchor:
        return 0
    return max(0, EVIDENCE_TTL[kind] - _sessions_since(anchor, as_of, calendar))


def _new_episode(symbol: str, start_date: str, sequence: int) -> dict:
    return {
        "episode_id": f"{symbol}:{start_date}:{sequence}",
        "symbol": symbol,
        "start_date": start_date,
        "last_positive_date": start_date,
        "last_negative_date": "",
        "shadow_stage": "PROBED",
        "retention_state": "UNCERTAIN",
        "retention_basis": "price_volume_proxy",
        "structure_state": "WAITING",
        "production_state": "OUT",
        "valid_for_sessions": EVIDENCE_TTL["event"],
        "valid_until": f"T+{EVIDENCE_TTL['event']} sessions unless invalidated",
        "failure_reason": "",
        "closed_at": "",
        "evidence": {},
    }


def update_episode(episode: dict | None, row: dict, as_of: str,
                   calendar: list[str], sequence: int = 1) -> tuple[dict | None, dict | None]:
    """Update one stock. Return ``(active, newly_closed)``."""
    production = str(row.get("state") or "OUT")
    data_fresh = _truth(row.get("data_fresh", True))
    observed_event = _event_date(row, as_of, calendar) if data_fresh else ""
    if episode is None and not observed_event:
        return None, None
    if episode is None:
        episode = _new_episode(str(row["symbol"]), observed_event, sequence)
    else:
        episode = json.loads(json.dumps(episode))

    ledger = episode.setdefault("evidence", {})
    if observed_event:
        _touch(ledger, "event", observed_event, as_of)
        episode["last_positive_date"] = as_of
    if data_fresh and _truth(row.get("supply_ready")):
        _touch(ledger, "supply", as_of, as_of)
        episode["last_positive_date"] = as_of
    if data_fresh and _truth(row.get("structure_ready")):
        _touch(ledger, "structure", as_of, as_of)
        episode["last_positive_date"] = as_of

    event_active = _active(episode, "event", as_of, calendar)
    supply_active = _active(episode, "supply", as_of, calendar)
    structure_active = _active(episode, "structure", as_of, calendar)
    health = _truth(row.get("health")) if data_fresh else None

    episode["production_state"] = production
    episode["structure_state"] = "READY" if structure_active else "WAITING"
    if production in PRODUCTION_MATURE:
        episode["shadow_stage"] = production
    elif structure_active and supply_active:
        episode["shadow_stage"] = "COILED"
    elif event_active and supply_active:
        episode["shadow_stage"] = "ABSORBING"
    elif event_active:
        episode["shadow_stage"] = "PROBED"
    else:
        episode["shadow_stage"] = "NONE"

    if supply_active and health is not False:
        episode["retention_state"] = "RETAINED"
    else:
        episode["retention_state"] = "UNCERTAIN"

    remaining = max(
        _remaining(episode, "event", as_of, calendar),
        _remaining(episode, "supply", as_of, calendar),
        _remaining(episode, "structure", as_of, calendar),
    )
    episode["valid_for_sessions"] = remaining
    episode["valid_until"] = f"T+{remaining} sessions unless invalidated"

    close_reason = ""
    if production == "TRIGGERED":
        close_reason = "TRIGGERED"
    elif production in PRODUCTION_FAILED:
        close_reason = "STRUCTURE_FAILED"
    elif not (event_active or supply_active or structure_active):
        close_reason = "EXPIRED"
    if close_reason:
        episode["retention_state"] = "EXITED" if close_reason != "TRIGGERED" else episode["retention_state"]
        episode["failure_reason"] = close_reason
        episode["closed_at"] = as_of
        episode["last_negative_date"] = as_of
        return None, episode
    return episode, None


def load_store(path: Path = STORE) -> dict:
    if not path.exists():
        return {"_meta": {"as_of": "", "version": VERSION}, "active": {}, "closed": []}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw.get("active"), dict) or not isinstance(raw.get("closed"), list):
        raise RuntimeError(f"invalid capital episode store: {path}")
    return raw


def _save_store(store: dict, path: Path = STORE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def update_store(rows: list[dict], as_of: str, calendar: list[str],
                 path: Path = STORE) -> tuple[dict, list[dict]]:
    store = load_store(path)
    previous = store.get("_meta", {}).get("as_of", "")
    if previous and as_of < previous:
        raise RuntimeError(f"capital episode store is at {previous}; refuse historical write {as_of}")

    active = dict(store["active"])
    unique_closed = {}
    for item in store["closed"]:
        key = (item.get("symbol"), item.get("start_date"), item.get("closed_at"),
               item.get("failure_reason"))
        current = unique_closed.get(key)
        if current is None or item.get("episode_id", "") < current.get("episode_id", ""):
            unique_closed[key] = item
    closed = list(unique_closed.values())
    seen = set()
    closed_today = [item for item in closed if item.get("closed_at") == as_of]
    closed_today_symbols = {item["symbol"] for item in closed_today}
    closed_ids = {item["episode_id"] for item in closed}
    sequence_by_symbol = {}
    for item in list(active.values()) + closed:
        symbol = item["symbol"]
        sequence_by_symbol[symbol] = max(
            sequence_by_symbol.get(symbol, 0), int(item["episode_id"].rsplit(":", 1)[-1]))

    for row in rows:
        symbol = str(row["symbol"])
        seen.add(symbol)
        current = active.get(symbol)
        if current is None and symbol in closed_today_symbols:
            continue
        sequence = sequence_by_symbol.get(symbol, 0) + (0 if current else 1)
        updated, ended = update_episode(current, row, as_of, calendar, sequence)
        if updated:
            active[symbol] = updated
        else:
            active.pop(symbol, None)
        if ended and ended["episode_id"] not in closed_ids:
            closed.append(ended)
            closed_today.append(ended)
            closed_ids.add(ended["episode_id"])

    for symbol in sorted(set(active) - seen):
        current = active[symbol]
        stale_row = {"symbol": symbol, "state": current.get("production_state", "OUT"),
                     "data_fresh": False}
        updated, ended = update_episode(current, stale_row, as_of, calendar)
        if updated:
            active[symbol] = updated
        else:
            active.pop(symbol, None)
        if ended and ended["episode_id"] not in closed_ids:
            closed.append(ended)
            closed_today.append(ended)
            closed_ids.add(ended["episode_id"])

    store = {
        "_meta": {"as_of": as_of, "version": VERSION,
                  "retention_semantics": "price_volume_proxy_not_proof_of_institutional_funds"},
        "active": active,
        "closed": closed,
    }
    _save_store(store, path)
    return store, closed_today


def write_daily(store: dict, closed_today: list[dict], as_of: str, output: Path,
                history: Path = HISTORY) -> pd.DataFrame:
    rows = []
    for item in store["active"].values():
        rows.append({**item, "status": "ACTIVE", "as_of": as_of})
    for item in closed_today:
        rows.append({**item, "status": "CLOSED", "as_of": as_of})
    flat = []
    for item in rows:
        flat.append({k: v for k, v in item.items() if k != "evidence"} | {
            "event_first_seen": item.get("evidence", {}).get("event", {}).get("first_seen", ""),
            "event_last_seen": item.get("evidence", {}).get("event", {}).get("last_seen", ""),
            "supply_last_seen": item.get("evidence", {}).get("supply", {}).get("last_seen", ""),
            "structure_last_seen": item.get("evidence", {}).get("structure", {}).get("last_seen", ""),
        })
    frame = pd.DataFrame(flat)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, encoding="utf-8-sig")
    if not frame.empty:
        if history.exists():
            old = pd.read_csv(history, encoding="utf-8-sig")
            old = old[old["as_of"] != as_of]
            frame = pd.concat([old, frame], ignore_index=True)
        frame.sort_values(["as_of", "symbol"]).to_csv(history, index=False, encoding="utf-8-sig")
    return pd.DataFrame(flat)
