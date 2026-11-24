# -*- coding: utf-8 -*-
"""Auditable theme-specific macro risk overlays."""
from __future__ import annotations

import json
from pathlib import Path


VERSION = "theme_risk_v1_1_20261027"
CONFIG = Path("theme_risk_overrides.json")
RISK_ORDER = {"ALLOW": 0, "REQUIRE_EXTERNAL_CONFIRMATION": 1, "BLOCK_NEW_ENTRY": 2}


def _read() -> dict:
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"overrides": [], "error": type(exc).__name__}


def _alias_match(theme: str, alias: str) -> bool:
    left = theme.casefold()
    right = alias.casefold()
    return left == right if len(right) == 1 else right in left or left in right


def evaluate(theme: str, as_of: str, known_at: str | None = None) -> dict:
    matches = []
    config = _read()
    if config.get("error"):
        return {"theme_risk_action": "REQUIRE_EXTERNAL_CONFIRMATION",
                "theme_risk_reason": "题材风险配置不可读，不能按无风险处理", "theme_risk_version": VERSION}
    folded = str(theme).casefold()
    if not folded:
        return {
            "theme_risk_action": "ALLOW", "theme_risk_reason": "",
            "theme_risk_source": "", "theme_risk_effective_from": "",
            "theme_risk_review_on": "", "theme_risk_known_at": "",
            "theme_risk_source_url": "", "theme_risk_version": VERSION,
        }
    for item in config.get("overrides", []):
        effective = str(item.get("effective_from", "9999-12-31"))
        expires = str(item.get("expires_after", "9999-12-31"))
        cutoff = known_at or f"{as_of}T23:59:59+08:00"
        if not effective <= as_of <= expires or str(item.get("known_at", "9999")) > cutoff:
            continue
        aliases = [str(value) for value in item.get("aliases", [])]
        excludes = [str(value) for value in item.get("exclude_aliases", [])]
        if any(_alias_match(folded, value) for value in excludes):
            continue
        if any(_alias_match(folded, value) for value in aliases):
            matches.append(item)
    if not matches:
        return {
            "theme_risk_action": "ALLOW", "theme_risk_reason": "",
            "theme_risk_source": "", "theme_risk_effective_from": "",
            "theme_risk_review_on": "", "theme_risk_known_at": "",
            "theme_risk_source_url": "", "theme_risk_version": VERSION,
        }
    selected = max(matches, key=lambda item: RISK_ORDER.get(item.get("action", "ALLOW"), 0))
    return {
        "theme_risk_action": selected.get("action", "ALLOW"),
        "theme_risk_reason": selected.get("reason", ""),
        "theme_risk_source": selected.get("source", ""),
        "theme_risk_effective_from": selected.get("effective_from", ""),
        "theme_risk_review_on": selected.get("review_on", ""),
        "theme_risk_known_at": selected.get("known_at", ""),
        "theme_risk_source_url": "|".join(selected.get("source_urls", [])),
        "theme_risk_version": VERSION,
    }


def evaluate_many(themes, as_of, known_at=None):
    """Risk applies to every mapped theme, not just the selected industry label."""
    results = [evaluate(str(theme), as_of, known_at) for theme in dict.fromkeys(themes) if theme]
    return max(results, key=lambda r: RISK_ORDER[r["theme_risk_action"]]) if results else evaluate("", as_of, known_at)
