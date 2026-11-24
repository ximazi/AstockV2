"""Persistent non-return lifecycle for P1 focus candidates."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd


VERSION = "candidate_episode_v1_20260829"
OUT_ROOT = Path("results/mainwave_engine")
STORE = OUT_ROOT / "candidate_episode_store.json"
EVENTS = OUT_ROOT / "candidate_episode_events.csv"
OUTCOMES = OUT_ROOT / "p1_candidate_outcomes.csv"

TERMINAL = {
    "EARLY_HIT", "STRUCTURE_FAILED", "EXPIRED", "RETENTION_LOST",
    "LEADER_REPLACED", "THEME_FAILED", "CROWDED_TOO_FAST", "DATA_FAILURE",
}
CANDIDATE_COLUMNS = [
    "candidate_id", "candidate_base", "sequence", "theme_id", "theme_label",
    "theme_episode_id", "symbol", "name", "capital_episode_id", "first_seen_date",
    "last_seen_date", "first_class", "current_class", "first_role", "current_role",
    "current_state", "confirmed_at", "ready_at", "triggered_at", "production_state",
    "retention_state", "remaining_sessions", "rank_global", "rank_in_theme",
    "terminal_reason", "closed_at",
]
OUTCOME_COLUMNS = [
    "candidate_id", "theme_id", "theme_episode_id", "symbol", "capital_episode_id",
    "first_candidate_date", "last_candidate_date", "intersection_first", "confirmed_at",
    "ready_at", "outcome", "outcome_at", "current_state",
]


def _load(path: Path = STORE) -> dict:
    if not path.exists():
        return {"_meta": {"as_of": "", "version": VERSION}, "active": {}, "closed": []}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value.get("active"), dict) or not isinstance(value.get("closed"), list):
        raise RuntimeError(f"invalid candidate episode store: {path}")
    return value


def _save(payload: dict, path: Path = STORE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _base(row: dict) -> str:
    return f"{row['theme_episode_id']}|{row['symbol']}|{row['episode_id']}"


def _state(row: dict, previous: str = "") -> str:
    production = str(row.get("production_state", ""))
    if production == "READY" or previous == "READY":
        return "READY"
    if str(row.get("intersection_class")) == "A" or previous in {"A_CONFIRMED", "READY"}:
        return "A_CONFIRMED"
    return "B_DISCOVERED"


def _event(candidate: dict, as_of: str, old: str, new: str, reason: str) -> dict:
    return {
        "candidate_id": candidate["candidate_id"], "candidate_base": candidate["candidate_base"],
        "event_at": as_of, "from_state": old or "NONE", "to_state": new,
        "reason": reason, "theme_id": candidate["theme_id"],
        "theme_episode_id": candidate["theme_episode_id"], "symbol": candidate["symbol"],
        "capital_episode_id": candidate["capital_episode_id"],
    }


def _new(row: dict, as_of: str, sequence: int) -> dict:
    base = _base(row)
    state = _state(row)
    return {
        "candidate_id": f"{base}|{sequence}", "candidate_base": base,
        "sequence": sequence, "theme_id": row["theme_id"],
        "theme_label": row.get("theme_label", ""),
        "theme_episode_id": row["theme_episode_id"], "symbol": row["symbol"],
        "name": row.get("name", ""), "capital_episode_id": row["episode_id"],
        "first_seen_date": as_of, "last_seen_date": as_of,
        "first_class": row.get("intersection_class", ""),
        "current_class": row.get("intersection_class", ""),
        "first_role": row.get("role", ""), "current_role": row.get("role", ""),
        "current_state": state,
        "confirmed_at": as_of if row.get("intersection_class") == "A" else "",
        "ready_at": as_of if state == "READY" else "", "triggered_at": "",
        "production_state": row.get("production_state", ""),
        "retention_state": row.get("retention_state", ""),
        "remaining_sessions": row.get("valid_for_sessions", ""),
        "rank_global": row.get("rank_global", ""), "rank_in_theme": row.get("rank_in_theme", ""),
        "terminal_reason": "", "closed_at": "",
    }


def _close(candidate: dict, as_of: str, reason: str) -> None:
    candidate["current_state"] = reason
    candidate["terminal_reason"] = reason
    candidate["closed_at"] = as_of
    if reason == "EARLY_HIT":
        candidate["triggered_at"] = as_of


def _append_events(rows: list[dict], path: Path = EVENTS) -> None:
    columns = [
        "candidate_id", "candidate_base", "event_at", "from_state", "to_state", "reason",
        "theme_id", "theme_episode_id", "symbol", "capital_episode_id",
    ]
    new = pd.DataFrame(rows, columns=columns)
    old = pd.read_csv(path, encoding="utf-8-sig") if path.exists() else pd.DataFrame(columns=columns)
    combined = pd.concat([old, new], ignore_index=True)
    if len(combined):
        combined = combined.drop_duplicates(
            ["candidate_id", "event_at", "from_state", "to_state"], keep="last"
        ).sort_values(["event_at", "candidate_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False, encoding="utf-8-sig")


def _terminal_reason(candidate: dict, theme: dict, active_capital: dict,
                     closed_capital: dict) -> str:
    capital_id = candidate["capital_episode_id"]
    if capital_id in closed_capital:
        terminal = str(closed_capital[capital_id].get("failure_reason", ""))
        if terminal == "TRIGGERED":
            return "EARLY_HIT"
        if terminal == "STRUCTURE_FAILED":
            return "STRUCTURE_FAILED"
        return "EXPIRED"
    capital = active_capital.get(capital_id)
    if capital is None:
        return "DATA_FAILURE"
    if capital.get("retention_state") != "RETAINED":
        return "RETENTION_LOST"
    if not theme or theme.get("theme_episode_id") != candidate["theme_episode_id"]:
        return "THEME_FAILED"
    if theme.get("crowding_state") == "CROWDED":
        return "CROWDED_TOO_FAST"
    if theme.get("theme_lifecycle") == "RETREAT":
        return "THEME_FAILED"
    if (candidate.get("first_role") == "information_leader"
            and theme.get("leader") not in {"", candidate["symbol"], None}):
        return "LEADER_REPLACED"
    return ""


def update(focus: pd.DataFrame, potential: pd.DataFrame, themes: pd.DataFrame,
           capital_store: dict, as_of: str, output_dir: Path,
           store_path: Path = STORE, event_path: Path = EVENTS,
           outcomes_path: Path = OUTCOMES) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    store = _load(store_path)
    previous = str(store.get("_meta", {}).get("as_of", ""))
    if previous and as_of < previous:
        raise RuntimeError(f"candidate episode store is at {previous}; refuse historical write {as_of}")

    active = dict(store["active"])
    closed = list(store["closed"])
    focus_rows = {_base(row): row for row in focus.to_dict("records")}
    potential_bases = {_base(row) for row in potential.to_dict("records")}
    theme_now = themes.set_index("theme_id").to_dict("index") if len(themes) else {}
    active_capital = {item["episode_id"]: item for item in capital_store["active"].values()}
    closed_capital = {item["episode_id"]: item for item in capital_store["closed"]}
    sequence = Counter()
    for item in list(active.values()) + closed:
        sequence[item["candidate_base"]] = max(sequence[item["candidate_base"]], int(item["sequence"]))
    closed_today_bases = {
        item["candidate_base"] for item in closed if item.get("closed_at") == as_of
    }
    events = []

    for base, row in focus_rows.items():
        candidate = active.get(base)
        if candidate is None:
            if base in closed_today_bases:
                continue
            candidate = _new(row, as_of, sequence[base] + 1)
            active[base] = candidate
            events.append(_event(candidate, as_of, "", candidate["current_state"], "entered P1 focus"))
        else:
            old = candidate["current_state"]
            new = _state(row, old)
            candidate.update({
                "last_seen_date": as_of, "current_class": row.get("intersection_class", ""),
                "current_role": row.get("role", ""),
                "production_state": row.get("production_state", ""),
                "retention_state": row.get("retention_state", ""),
                "remaining_sessions": row.get("valid_for_sessions", ""),
                "rank_global": row.get("rank_global", ""),
                "rank_in_theme": row.get("rank_in_theme", ""),
            })
            if row.get("intersection_class") == "A" and not candidate.get("confirmed_at"):
                candidate["confirmed_at"] = as_of
            if new != old:
                candidate["current_state"] = new
                if new == "READY" and not candidate.get("ready_at"):
                    candidate["ready_at"] = as_of
                events.append(_event(candidate, as_of, old, new, "P1/production state advanced"))

    for base in sorted(set(active) - set(focus_rows)):
        candidate = active[base]
        reason = _terminal_reason(
            candidate, theme_now.get(candidate["theme_id"], {}), active_capital, closed_capital
        )
        old = candidate["current_state"]
        if reason:
            _close(candidate, as_of, reason)
            events.append(_event(candidate, as_of, old, reason, reason))
            closed.append(candidate)
            active.pop(base, None)
            continue
        new = "OUT_OF_FOCUS_RANK" if base in potential_bases else "SIGNAL_PAUSED"
        if new != old:
            candidate["current_state"] = new
            events.append(_event(candidate, as_of, old, new, "not in today's final focus"))

    store = {
        "_meta": {"as_of": as_of, "version": VERSION},
        "active": active, "closed": closed,
    }
    _save(store, store_path)
    _append_events(events, event_path)

    daily = list(active.values()) + [item for item in closed if item.get("closed_at") == as_of]
    daily_frame = pd.DataFrame(daily, columns=CANDIDATE_COLUMNS)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_frame.to_csv(output_dir / "candidate_episodes.csv", index=False, encoding="utf-8-sig")

    all_items = list(active.values()) + closed
    outcomes = pd.DataFrame([{
        "candidate_id": item["candidate_id"], "theme_id": item["theme_id"],
        "theme_episode_id": item["theme_episode_id"], "symbol": item["symbol"],
        "capital_episode_id": item["capital_episode_id"],
        "first_candidate_date": item["first_seen_date"],
        "last_candidate_date": item["last_seen_date"], "intersection_first": item["first_class"],
        "confirmed_at": item.get("confirmed_at", ""), "ready_at": item.get("ready_at", ""),
        "outcome": item.get("terminal_reason") or "OPEN", "outcome_at": item.get("closed_at", ""),
        "current_state": item["current_state"],
    } for item in all_items], columns=OUTCOME_COLUMNS)
    outcomes.to_csv(outcomes_path, index=False, encoding="utf-8-sig")

    funnel = {
        "as_of": as_of, "version": VERSION, "total_episodes": len(all_items),
        "active": len(active), "closed": len(closed),
        "b_started": sum(item.get("first_class") == "B" for item in all_items),
        "b_to_a": sum(item.get("first_class") == "B" and bool(item.get("confirmed_at")) for item in all_items),
        "a_confirmed": sum(bool(item.get("confirmed_at")) for item in all_items),
        "ready": sum(bool(item.get("ready_at")) for item in all_items),
        "triggered": sum(item.get("terminal_reason") == "EARLY_HIT" for item in all_items),
        "terminal_reasons": dict(Counter(
            item.get("terminal_reason") for item in closed if item.get("terminal_reason")
        )),
    }
    (output_dir / "candidate_funnel.json").write_text(
        json.dumps(funnel, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return daily_frame, outcomes, funnel
