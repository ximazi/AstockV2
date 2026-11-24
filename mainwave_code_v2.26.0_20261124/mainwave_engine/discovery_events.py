"""Discovery event register: every security the system saw today, from every source, with a state and a reason.

Sources: the engine candidates (closure records), the preparation watch (P5), user-pinned tracking, the paper
books' plans and positions, and yesterday's register (for first_seen / last_seen and re-activation). A name
that is in none of today's sources but was active yesterday is DEACTIVATED, not dropped. Counts are taken on
the full union before any display cut. States are observation states: DISCOVERED → EVIDENCE_PENDING →
WATCHING → SETUP_FORMING → SETUP_CONFIRMED → PLAN_APPROVED_WAIT_TRIGGER → POSITION_OPEN, with EXPIRED /
INVALIDATED / DATA_WAIT; none of them is a permission."""
import json
from collections import Counter
from pathlib import Path

VERSION = "discovery_events_v1_20260923"
STATES = ("DISCOVERED", "EVIDENCE_PENDING", "WATCHING", "SETUP_FORMING", "SETUP_CONFIRMED", "PLAN_APPROVED_WAIT_TRIGGER",
          "POSITION_OPEN", "DATA_WAIT", "INVALIDATED", "EXPIRED", "DEACTIVATED")


def _state_from_candidate(record, shadow_row=None):
    state = record.get("state")
    sv = (record.get("evidence") or {}).get("state_vector") or {}
    if state == "DATA_WAIT":
        return "DATA_WAIT"
    if sv.get("stage") == "FAILED" or (shadow_row or {}).get("stage_bucket") == "FAILED":
        return "INVALIDATED"
    if state == "PLAN_REVIEW":
        return "SETUP_CONFIRMED"
    if sv.get("setup_status") in ("CONFIRMED",):
        return "SETUP_CONFIRMED"
    if sv.get("setup_status") in ("FORMING",):
        return "SETUP_FORMING"
    return "WATCHING"


def build(day, records, shadow_rows=None, preparation_watch=None, pinned=None, plans=None, previous=None):
    """One entry per security; sources and readings attached; previous register carried for continuity."""
    shadow_by = {r["symbol"]: r for r in (shadow_rows or [])}
    prev = {e["symbol"]: e for e in (previous or {}).get("events", [])} if previous and previous.get("as_of", "") < day else {}
    events = {}

    def touch(symbol, source, **fields):
        e = events.setdefault(symbol, {"symbol": symbol, "sources": [], "first_seen": prev.get(symbol, {}).get("first_seen") or day,
                                       "last_seen": day, "reactivated": symbol in prev and prev[symbol].get("state") == "DEACTIVATED",
                                       "previous_state": prev.get(symbol, {}).get("state"), "name": fields.get("name"), "state": "DISCOVERED",
                                       "not_reviewed_reason": "", "missing": []})
        if source not in e["sources"]:
            e["sources"].append(source)
        for k, v in fields.items():
            if v is not None and (k not in e or e[k] in (None, "", [])):
                e[k] = v
        return e

    for r in records:
        sh = shadow_by.get(r["symbol"])
        e = touch(r["symbol"], "ENGINE_CANDIDATE", name=r.get("name"), theme=r.get("theme"), route_ids=r.get("route_ids", ""),
                  engine_state=r.get("state"), tracking_only=bool(r.get("tracking_only")), shadow_rank=(sh or {}).get("shadow_rank"), reading=(sh or {}).get("reading"))
        e["state"] = _state_from_candidate(r, sh)
        if e["state"] == "DATA_WAIT":
            e["missing"].append("当日特征或基准")
    for w in preparation_watch or []:
        sym = w.get("symbol")
        if not sym:
            continue
        e = touch(sym, "P5_WATCH", name=w.get("name"), theme=w.get("industry"), bias_atr=w.get("bias_atr"), excess20=w.get("excess20"))
        if "ENGINE_CANDIDATE" not in e["sources"]:
            e["state"] = "EVIDENCE_PENDING"
            e["missing"] = list(dict.fromkeys(e["missing"] + ["引擎证据（结构、路径、生命周期）", "冻结触发价与失效价"]))
            e["not_reviewed_reason"] = "域外观察提名：待取证，不进候选状态"
    for p in pinned or []:
        e = touch(p["symbol"], "USER_PINNED", name=p.get("name"))
        if len(e["sources"]) == 1:
            e["state"] = "EVIDENCE_PENDING"
            e["missing"] = list(dict.fromkeys(e["missing"] + ["原计划、触发价与失效价", "引擎证据"]))
            e["not_reviewed_reason"] = "指定跟踪股不在任何发现源：缺证，必保留"
    for pl in plans or []:
        e = touch(pl["symbol"], "PAPER_BOOK", name=pl.get("name"), account_id=pl.get("account_id"), plan_id=pl.get("plan_id"))
        e["state"] = "POSITION_OPEN" if pl.get("open_position") else ("PLAN_APPROVED_WAIT_TRIGGER" if pl.get("approved") and not pl.get("expired") else "EXPIRED")
    for sym, old in prev.items():
        if sym not in events and old.get("state") not in ("DEACTIVATED", "INVALIDATED", "EXPIRED"):
            events[sym] = {**old, "state": "DEACTIVATED", "last_seen": old.get("last_seen"), "deactivated_at": day, "sources": [], "previous_state": old.get("state")}
    rows = sorted(events.values(), key=lambda e: ((e.get("shadow_rank") is None), e.get("shadow_rank") or 10 ** 6, e["symbol"]))
    candidates = {r["symbol"] for r in records}
    watch = {w["symbol"] for w in (preparation_watch or []) if w.get("symbol")}
    summary = {"total": len(rows), "active": sum(e["state"] != "DEACTIVATED" for e in rows),
               "by_state": dict(Counter(e["state"] for e in rows)), "by_source_count": dict(Counter(len(e["sources"]) for e in rows)),
               "candidates": len(candidates), "watch": len(watch), "overlap": len(candidates & watch), "outside": len(watch - candidates),
               "union": len(candidates | watch), "pinned": len(pinned or []), "new_today": sum(e["first_seen"] == day for e in rows),
               "reactivated": sum(bool(e.get("reactivated")) for e in rows), "deactivated": sum(e["state"] == "DEACTIVATED" for e in rows)}
    return {"version": VERSION, "as_of": day, "summary": summary, "events": rows,
            "note": "观察状态不是许可；域外 P5 名单与指定股在此登记，缺证写 missing，不因名额消失"}


def write(path, register):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(register, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return path
