# -*- coding: utf-8 -*-
"""Point-in-time theme membership and leader-driven semantic resolution."""
from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path


VERSION = "theme_semantics_v1_3_20261029"
MAPPING = Path("concept_mapping.json")
MAPPING_HISTORY = Path("concept_mapping_history")
DYNAMIC_MAPPING_HISTORY = Path("concept_dynamic_mapping")
RELATION_DB = Path("stock_relation.db")
MEMBERSHIP_EVIDENCE = Path("concept_membership_evidence.json")

GENERIC_CONCEPT_PARTS = {
    "融资融券", "机构重仓", "沪股通", "深股通", "MSCI", "富时罗素", "标准普尔",
    "中证", "上证", "创业板综", "微盘股", "AB股", "AH股", "QFII", "证金持股",
    "预增", "预减", "社保重仓", "基金重仓", "转债标的", "国企改革", "央国企改革",
}


def _read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _generated_date(payload: dict) -> str:
    return str(payload.get("generated", ""))[:10]


def apply_membership_evidence(mapping, as_of, known_at=None):
    """Dated, sourced attribution corrections; never infer a theme from an industry keyword."""
    result = dict(mapping)
    meta = {"path": str(MEMBERSHIP_EVIDENCE), "applied": [], "errors": [], "version": VERSION}
    cutoff = datetime.fromisoformat(f"{as_of}T23:59:59+08:00")
    if known_at:
        cutoff = min(cutoff, datetime.fromisoformat(known_at))
    try:
        payload = json.loads(MEMBERSHIP_EVIDENCE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        meta["errors"].append(f"membership evidence unavailable: {type(exc).__name__}")
        return result, meta
    for item in payload.get("links", []):
        try:
            known = datetime.fromisoformat(item["known_at"])
            if known.tzinfo is None or item["published_on"] > known.date().isoformat():
                raise ValueError("invalid publication/knowledge boundary")
            if known > cutoff or not item["effective_from"] <= as_of <= item.get("expires_after", "9999-12-31"):
                continue
            symbol, themes = item["symbol"], item["themes"]
            if symbol not in mapping:
                continue
            if (not isinstance(themes, list) or not themes
                    or any(not isinstance(t, str) or not t.strip() for t in themes)
                    or item.get("relation_kind") not in {"MARKET_NARRATIVE", "PRODUCT_APPLICATION", "CONFIRMED_BUSINESS"}
                    or not isinstance(item.get("source_urls"), list) or not item["source_urls"]
                    or any(not isinstance(u, str) or not u.startswith("https://") for u in item["source_urls"])
                    or not item.get("evidence")):
                raise ValueError("missing verified membership evidence")
            current = dict(result[symbol])
            current["concepts"] = list(dict.fromkeys(current.get("concepts", []) + themes))
            current["_membership_evidence"] = current.get("_membership_evidence", []) + [item]
            result[symbol] = current
            meta["applied"].append({"symbol": symbol, "themes": themes, "known_at": item["known_at"]})
        except (KeyError, TypeError, ValueError) as exc:
            meta["errors"].append(f"invalid membership link: {type(exc).__name__}")
    return result, meta


def load_mapping_as_of(as_of: str) -> tuple[dict, dict]:
    """Return the newest mapping known on or before as_of."""
    candidates: list[tuple[str, int, Path, dict]] = []
    paths = [MAPPING]
    if MAPPING_HISTORY.exists():
        paths.extend(sorted(MAPPING_HISTORY.glob("*.json")))
    for path in paths:
        payload = _read(path, {})
        generated = _generated_date(payload)
        if generated and generated <= as_of and isinstance(payload.get("mapping"), dict):
            full_stamp = str(payload.get("generated", generated))
            candidates.append((full_stamp, int(path == MAPPING), path, payload))
    if not candidates:
        return {}, {
            "status": "MISSING", "generated": "", "path": "", "version": VERSION,
        }
    full_stamp, _, path, payload = max(candidates, key=lambda item: (item[0], item[1]))
    generated = full_stamp[:10]
    mapping = {
        symbol: {**item, "concepts": list(item.get("concepts", []))}
        for symbol, item in payload.get("mapping", {}).items()
    }
    dynamic_files = 0
    dynamic_concepts = set()
    if DYNAMIC_MAPPING_HISTORY.exists():
        for overlay_path in sorted(DYNAMIC_MAPPING_HISTORY.glob("*.json")):
            overlay = _read(overlay_path, {})
            overlay_generated = _generated_date(overlay)
            if not (generated <= overlay_generated <= as_of):
                continue
            dynamic_files += 1
            dynamic_concepts.update(overlay.get("captured_concepts", []))
            for symbol, overlay_item in overlay.get("mapping", {}).items():
                current = mapping.setdefault(symbol, {
                    "name": overlay_item.get("name", ""), "industry": "", "concepts": [],
                })
                dynamic_for_symbol = current.setdefault("_dynamic_concepts", [])
                for concept in overlay_item.get("concepts", []):
                    if concept not in current["concepts"]:
                        current["concepts"].append(concept)
                    if concept not in dynamic_for_symbol:
                        dynamic_for_symbol.append(concept)
    mapping, membership = apply_membership_evidence(mapping, as_of)
    counts = [len(item.get("concepts", [])) for item in mapping.values()]
    multi = sum(value > 1 for value in counts)
    base_status = "RICH" if multi >= max(100, int(len(counts) * 0.10)) else "INDUSTRY_ONLY"
    age_days = (date.fromisoformat(as_of) - date.fromisoformat(generated)).days
    status = f"STALE_{base_status}" if age_days > 30 else base_status
    if dynamic_files:
        status += "_DYNAMIC"
    return mapping, {
        "status": status, "generated": generated, "path": str(path),
        "age_days": age_days, "stocks": len(mapping), "multi_concept_stocks": multi,
        "dynamic_overlay_files": dynamic_files,
        "dynamic_overlay_concepts": len(dynamic_concepts),
        "membership_evidence": membership,
        "version": VERSION,
    }


def normalize_symbol(code: str) -> str:
    text = str(code).strip()
    if text.startswith(("sh_", "sz_", "bj_")):
        return text
    if "." in text:
        number, exchange = text.split(".", 1)
        return f"{exchange.lower()}_{number}"
    if text.startswith(("6", "5")):
        return f"sh_{text}"
    if text.startswith(("0", "3")):
        return f"sz_{text}"
    if text.startswith(("4", "8", "9")):
        return f"bj_{text}"
    return text


def load_aux_relationships() -> tuple[dict[str, set[str]], dict[str, set[str]], dict]:
    by_symbol: dict[str, set[str]] = defaultdict(set)
    by_concept: dict[str, set[str]] = defaultdict(set)
    meta = {"status": "MISSING", "updated": "", "path": str(RELATION_DB)}
    if not RELATION_DB.exists():
        return by_symbol, by_concept, meta
    try:
        connection = sqlite3.connect(str(RELATION_DB))
        for code, concept in connection.execute(
                "select stock_code, concept_name from stock_concept_map"):
            symbol = normalize_symbol(code)
            name = str(concept).strip()
            if symbol and name:
                by_symbol[symbol].add(name)
                by_concept[name].add(symbol)
        row = connection.execute(
            "select update_time from last_update where module_name = ?",
            ("ConceptLinker",),
        ).fetchone()
        connection.close()
        updated = str(row[0]) if row else ""
        meta = {
            "status": "STALE_AUX" if updated else "UNKNOWN_AGE_AUX",
            "updated": updated, "path": str(RELATION_DB), "concepts": len(by_concept),
        }
    except (sqlite3.Error, OSError):
        meta["status"] = "ERROR"
    return by_symbol, by_concept, meta


def members_by_concept(mapping: dict) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for symbol, item in mapping.items():
        for concept in item.get("concepts", []):
            if concept:
                result[str(concept)].add(symbol)
    return result


def is_generic_concept(concept: str) -> bool:
    return any(part in str(concept) for part in GENERIC_CONCEPT_PARTS)


def _specific(concept: str, raw_theme: str) -> bool:
    if not concept or concept == raw_theme:
        return False
    return not is_generic_concept(concept)


def resolve_event_theme(raw_theme: str, leaders: list[str], mapping: dict,
                        aux_by_symbol: dict[str, set[str]]) -> dict:
    """Relabel a broad event only when every leader shares a specific concept."""
    normalized = sorted({normalize_symbol(value) for value in leaders if value})
    if len(normalized) < 3:
        return {
            "raw_theme": raw_theme, "resolved_theme": raw_theme,
            "semantic_status": "RAW_THEME", "semantic_confidence": "LOW",
            "semantic_evidence": "fewer_than_3_leaders",
        }
    concept_sets = []
    for symbol in normalized:
        current = set(mapping.get(symbol, {}).get("concepts", []))
        current.update(aux_by_symbol.get(symbol, set()))
        concept_sets.append(current)
    shared = set.intersection(*concept_sets) if concept_sets else set()
    membership = Counter(
        concept for concepts in aux_by_symbol.values() for concept in concepts
    )
    specific = sorted((name for name in shared if _specific(name, raw_theme)),
                      key=lambda value: (membership.get(value, 10**9), len(value), value))
    if not specific:
        return {
            "raw_theme": raw_theme, "resolved_theme": raw_theme,
            "semantic_status": "RAW_THEME", "semantic_confidence": "MEDIUM",
            "semantic_evidence": "no_unanimous_specific_concept",
        }
    resolved = specific[0]
    return {
        "raw_theme": raw_theme, "resolved_theme": resolved,
        "semantic_status": "LEADER_UNANIMOUS_RELABEL",
        "semantic_confidence": "HIGH",
        "semantic_evidence": (f"{len(normalized)}/{len(normalized)} ignition leaders share "
                              f"{resolved}; auxiliary_members={membership.get(resolved, '')}"),
    }


def resolved_members(resolution: dict, leaders: list[str], point_members: dict[str, set[str]],
                     aux_members: dict[str, set[str]]) -> tuple[list[str], str]:
    raw = resolution.get("raw_theme", resolution.get("theme", ""))
    resolved = resolution["resolved_theme"]
    if resolved != raw and aux_members.get(resolved):
        members = set(aux_members[resolved])
        source = "STALE_AUX_RELATION+IGNITION_LEADERS"
    else:
        members = set(point_members.get(raw, set()))
        source = "POINT_IN_TIME_MAPPING+IGNITION_LEADERS"
    members.update(normalize_symbol(value) for value in leaders if value)
    return sorted(members), source
