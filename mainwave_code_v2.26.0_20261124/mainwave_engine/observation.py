# -*- coding: utf-8 -*-
"""Observation-only capital stages and idempotent feature trajectory logging."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


CAPITAL_STAGE_VERSION = "v0_placeholder_20260828"
TRAJECTORY = Path("results/mainwave_engine/trajectory_log.csv")


def derive_capital_stage(evidence: dict, production_state: str) -> str:
    """Derive a provisional label without affecting production transitions."""
    if production_state in {"ARMED", "READY", "TRIGGERED"}:
        return production_state
    if not evidence.get("data_fresh", True):
        return "UNKNOWN"
    event = bool(evidence.get("imp") or evidence.get("fresh") or evidence.get("pit"))
    structure = bool(evidence.get("structure_ready"))
    supply = bool(evidence.get("supply_ready"))
    health = bool(evidence.get("health"))
    if structure and supply and not health:
        return "COILED"
    if event and supply and not structure:
        return "ABSORBING"
    if event and not (structure and supply):
        return "PROBED"
    return "NONE"


def trajectory_row(run_date: str, symbol: str, state: str, evidence: dict,
                   data_fresh: bool, prefilter_ch: str = "") -> dict:
    """Build the canonical daily trajectory row used by live and read-only runs."""
    raw = {
        "data_fresh": data_fresh,
        "health": evidence.get("health_ok"),
        "structure_ready": evidence.get("structure_ready"),
        "supply_ready": evidence.get("supply_ready"),
        "imp": evidence.get("imp"),
        "fresh": evidence.get("fresh"),
        "pit": evidence.get("pit"),
    }
    return {
        "date": run_date, "symbol": symbol, "state": state,
        "prefilter_ch": prefilter_ch, "data_fresh": data_fresh,
        "health": bool(evidence.get("health_ok")) if data_fresh else "",
        "structure_ready": bool(evidence.get("structure_ready")) if data_fresh else "",
        "supply_ready": bool(evidence.get("supply_ready")) if data_fresh else "",
        "supply_kind": evidence.get("supply_kind", "") if data_fresh else "",
        "imp": bool(evidence.get("imp")) if data_fresh else "",
        "imp_age": evidence.get("imp_age", "") if data_fresh else "",
        "fresh": bool(evidence.get("fresh")) if data_fresh else "",
        "fresh_age": evidence.get("fresh_age", "") if data_fresh else "",
        "pit": bool(evidence.get("pit")) if data_fresh else "",
        "pit_age": evidence.get("pit_age", "") if data_fresh else "",
        "route": evidence.get("route", ""),
        "armed_core": bool(evidence.get("armed_core")) if data_fresh else "",
        "ready_core": bool(evidence.get("ready_core")) if data_fresh else "",
        "max_evidence_date": evidence.get("max_evidence_date", "") if data_fresh else "",
        "capital_stage": derive_capital_stage(raw, state),
        "capital_stage_version": CAPITAL_STAGE_VERSION,
    }


def write_trajectory(rows: list[dict], daily_path: Path, log_path: Path = TRAJECTORY) -> None:
    """Write one daily snapshot and upsert it into the cumulative CSV."""
    new = pd.DataFrame(rows)
    if new.empty:
        raise RuntimeError("trajectory rows are empty")
    if new.duplicated(["date", "symbol"]).any():
        raise RuntimeError("trajectory contains duplicate date/symbol rows")
    daily_path.parent.mkdir(parents=True, exist_ok=True)
    new.sort_values("symbol").to_csv(daily_path, index=False, encoding="utf-8-sig")

    if log_path.exists():
        old = pd.read_csv(log_path, encoding="utf-8-sig")
        keys = set(zip(new["date"], new["symbol"]))
        keep = [key not in keys for key in zip(old["date"], old["symbol"])]
        new = pd.concat([old.loc[keep], new], ignore_index=True)
    new = new.sort_values(["date", "symbol"])
    tmp = log_path.with_suffix(".tmp")
    new.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, log_path)
