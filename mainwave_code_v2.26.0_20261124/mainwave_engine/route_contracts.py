"""Explicit entry evidence, never permission inferred from route-label order."""
from __future__ import annotations

VERSION = "route_contracts_v1_20260910"
ACTIVE_THEMES = {"FERMENTING", "IGNITED", "DIFFUSING", "EMERGING", "ACTIVE"}


def route_set(value) -> list[str]:
    return sorted(set(value.split("|") if isinstance(value, str) else value or []) & {"R1", "R2", "R3"})


def price_confirmation(row: dict) -> bool:
    """Require the original event and its dated acceptance, not an R1 label."""
    at, accepted = str(row.get("date", ""))[:10], str(row.get("breakout_accepted_at", ""))[:10]
    return bool(row.get("eligible") and row.get("breakout_id") and at and accepted and accepted <= at
                and row.get("breakout_status") in {"ACCEPTED", "ADVANCING"}
                and row.get("demand") and not row.get("supply_warning")
                and row.get("bias_atr") is not None and row["bias_atr"] <= 2)


def entry_context(route: dict, theme: dict) -> dict:
    routes = route_set(route.get("route_ids"))
    independent = route.get("independent_price_evidence") is True
    theme_ready = theme.get("lifecycle") in ACTIVE_THEMES
    base = route.get("route_requirements_met", True) is not False
    if independent:
        contract, ready = "PRICE_INDEPENDENT", base
    elif "R2" in routes:
        # A secondary propagation clue cannot veto an otherwise valid R2 contract.
        contract, ready = "THEME_REQUIRED", base and theme_ready
    elif "R3" in routes:
        contract = "PROPAGATION_REQUIRED"
        ready = base and theme_ready and route.get("strict_chain_verified") is True
    else:
        contract, ready = "STRUCTURE_ONLY", base
    missing = []
    if not base:
        missing.append("ROUTE_PREPARATION_INCOMPLETE")
    if contract in {"THEME_REQUIRED", "PROPAGATION_REQUIRED"} and not theme_ready:
        missing.append("ACTIVE_THEME_MISSING")
    if contract == "PROPAGATION_REQUIRED" and route.get("strict_chain_verified") is not True:
        missing.append("STRICT_CHAIN_UNVERIFIED")
    return {"version": VERSION, "route_set": routes, "contract": contract,
            "requirements_met": bool(ready), "theme_required": contract in {"THEME_REQUIRED", "PROPAGATION_REQUIRED"},
            "independent_price_evidence": independent, "missing": missing}
