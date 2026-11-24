# -*- coding: utf-8 -*-
"""Report-only slow-carrier lifecycle for tight-chain propagation.

The overlay starts when an information leader is first observable. It never
writes the production stock state machine and never creates orders.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__ as ENGINE_VERSION
from . import data as D
from .theme_registry import themes_as_of


VERSION = "slow_carrier_shadow_v1_20260903"
CALIBRATION_STATUS = "UNVALIDATED_SHADOW"
OUT_ROOT = Path("results/mainwave_engine")
ACTIVE_THEME_STAGES = {"SEED", "FERMENTING", "IGNITED", "DIFFUSING"}
TERMINAL_THEME_STAGES = {"CROWDED", "RETREAT"}
BLOCKED_PRODUCTION = {"TRIGGERED", "FAILED", "INVALID"}
WATCH_STATES = {"VALID_LAG", "FIRST_RESPONSE", "READY"}
MAX_PER_THEME = 3
MAX_GLOBAL = 15
WINDOW_SESSIONS = 8

STATE_ORDER = {
    "READY": 0,
    "FIRST_RESPONSE": 1,
    "VALID_LAG": 2,
    "PREPARED": 3,
    "EXPIRED": 4,
    "FAILED": 5,
    "NOT_PREPARED": 6,
    "DATA_INCOMPLETE": 7,
}
TIER_ORDER = {"TRIGGER_NEAR": 0, "CORE_LURK": 1, "EARLY_LURK": 2,
              "NOT_QUALIFIED": 3}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _truth(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes"}


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _latest(values: list[str]) -> str:
    valid = [str(value) for value in values if str(value)[:10].count("-") == 2]
    return max(valid) if valid else ""


def _theme_anchor(theme_row: dict, leader_quality: dict, as_of: str) -> str:
    episode_date = str(theme_row.get("theme_episode_id", "")).rsplit(":", 1)[-1]
    anchor = _latest([episode_date, leader_quality.get("event_first_seen", "")])
    if _truth(theme_row.get("leader_replaced", False)):
        return as_of
    return anchor or as_of


def _price_features(symbol: str, as_of: str) -> pd.DataFrame:
    frame = D.load_tail(symbol, 180, as_of=as_of)
    if frame is None or frame.empty or str(frame.iloc[-1]["date"]) != as_of:
        return pd.DataFrame()
    result = frame.copy().sort_values("date").reset_index(drop=True)
    result["date"] = result["date"].astype(str)
    close = result["close"].astype(float)
    high = result["high"].astype(float)
    low = result["low"].astype(float)
    amount = result["amount"].astype(float)
    previous = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - previous).abs(),
        (low - previous).abs(),
    ], axis=1).max(axis=1)
    atr20 = true_range.rolling(20, min_periods=15).mean()
    result["atr20_pct"] = atr20 / close * 100
    result["ema5"] = close.ewm(span=5, adjust=False, min_periods=5).mean()
    result["ema20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    result["ema20_bias_atr"] = (close / result["ema20"] - 1) * 100 / result["atr20_pct"]
    result["drawdown20_atr"] = (
        (close / high.rolling(20, min_periods=20).max() - 1) * 100
        / result["atr20_pct"]
    )
    result["ret1_pct"] = close.pct_change() * 100
    result["ret3_pct"] = close.pct_change(3) * 100
    amount_base = amount.shift(1).rolling(20, min_periods=10).median()
    result["amount_ratio1"] = amount / amount_base
    price_range = (high - low).replace(0, np.nan)
    result["close_location"] = ((close - low) / price_range).clip(0, 1)
    result["above_prev_high"] = close > high.shift(1)
    result["above_ema5"] = close > result["ema5"]
    return result


def _attach_progress(frame: pd.DataFrame, anchor: str) -> pd.DataFrame:
    result = frame.copy()
    prior = result[result["date"] < anchor]
    if len(prior):
        base = float(prior.iloc[-1]["close"])
    else:
        current = result[result["date"] >= anchor]
        if current.empty:
            return pd.DataFrame()
        base = float(current.iloc[0]["close"])
    result["since_anchor_pct"] = (result["close"].astype(float) / base - 1) * 100
    return result[result["date"] >= anchor].copy()


def _prepared(quality: dict) -> bool:
    return all(_truth(quality.get(column)) for column in (
        "capital_gate", "retention_gate", "supply_gate", "maturity_gate",
        "data_fresh", "extension_gate", "distribution_gate",
    )) and str(quality.get("production_state", "")) not in BLOCKED_PRODUCTION


def _valid_lag(row: pd.Series) -> bool:
    required = [
        row.get("leader_since_anchor_pct"), row.get("leader_atr20_pct"),
        row.get("since_anchor_pct"), row.get("atr20_pct"),
        row.get("chain_progress_percentile"), row.get("ema20_bias_atr"),
        row.get("drawdown20_atr"),
    ]
    if not all(_finite(value) for value in required):
        return False
    atr = max(float(row["atr20_pct"]), 0.01)
    leader_atr = max(float(row["leader_atr20_pct"]), 0.01)
    leader_move_atr = float(row["leader_since_anchor_pct"]) / leader_atr
    lag_gap_atr = (
        float(row["leader_since_anchor_pct"]) - float(row["since_anchor_pct"])
    ) / atr
    percentile = float(row["chain_progress_percentile"])
    return (
        leader_move_atr >= 0.75
        and lag_gap_atr >= 0.75
        and 0.15 <= percentile <= 0.70
        and float(row["ema20_bias_atr"]) >= -1.50
        and float(row["drawdown20_atr"]) >= -3.00
    )


def _response_evidence(row: pd.Series) -> tuple[bool, int, str]:
    atr = max(float(row.get("atr20_pct", np.nan)), 0.01)
    ret1_atr = float(row.get("ret1_pct", np.nan)) / atr
    relative_r3_atr = (
        float(row.get("ret3_pct", np.nan)) - float(row.get("chain_r3_median", np.nan))
    ) / atr
    price = _finite(ret1_atr) and ret1_atr >= 0.35 and float(
        row.get("close_location", 0) or 0) >= 0.65
    structure = _truth(row.get("above_prev_high")) or _truth(row.get("above_ema5"))
    volume = _finite(row.get("amount_ratio1")) and float(row["amount_ratio1"]) >= 1.15
    relative_strength = _finite(relative_r3_atr) and relative_r3_atr >= 0.35
    evidence = {
        "PRICE": price,
        "STRUCTURE": structure,
        "VOLUME": volume,
        "RELATIVE_STRENGTH": relative_strength,
    }
    passed = price and structure and (volume or relative_strength)
    return passed, sum(evidence.values()), "|".join(
        name for name, present in evidence.items() if present
    )


def classify_stage(*, prepared: bool, data_fresh: bool, theme_lifecycle: str,
                   crowded: bool, anchor_age: int, current_valid_lag: bool,
                   first_response_date: str, as_of: str, hard_gate_pass: bool,
                   failed: bool) -> str:
    """Classify one candidate without allowing scores to repair a failed gate."""
    if not data_fresh:
        return "DATA_INCOMPLETE"
    if failed:
        return "FAILED"
    if not prepared:
        return "NOT_PREPARED"
    if crowded or theme_lifecycle in TERMINAL_THEME_STAGES or anchor_age > WINDOW_SESSIONS:
        return "EXPIRED"
    if first_response_date:
        if first_response_date == as_of:
            return "FIRST_RESPONSE"
        return "READY" if hard_gate_pass else "FIRST_RESPONSE"
    if current_valid_lag:
        return "VALID_LAG"
    return "PREPARED"


def _window_bucket(age: int) -> str:
    if age <= 2:
        return "D0_2"
    if age <= 4:
        return "D3_4"
    if age <= 8:
        return "D5_8"
    return "EXPIRED"


def _action(stage: str, theme_lifecycle: str, attention_slope: str) -> str:
    if stage == "READY":
        if theme_lifecycle not in ACTIVE_THEME_STAGES:
            return "WAIT_THEME_WINDOW"
        if attention_slope == "FALLING":
            return "WAIT_ATTENTION_REPAIR"
        return "SHADOW_READY_REVIEW"
    if stage == "FIRST_RESPONSE":
        return "SHADOW_FIRST_RESPONSE_WAIT_NEXT_SESSION"
    if stage == "VALID_LAG":
        return "WATCH_FIRST_RESPONSE"
    return "NO_ACTION"


def _cap_focus(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ranked = frame[frame["slow_stage"].isin(WATCH_STATES)].copy()
    if ranked.empty:
        return ranked
    ranked["_state"] = ranked["slow_stage"].map(STATE_ORDER).fillna(99)
    ranked["_tier"] = ranked["carrier_tier"].map(TIER_ORDER).fillna(99)
    ranked = ranked.sort_values([
        "_state", "_tier", "response_evidence_count", "lag_gap_atr", "symbol"
    ], ascending=[True, True, False, True, True])
    ranked["rank_in_theme"] = ranked.groupby("theme_id").cumcount() + 1
    ranked = ranked[ranked["rank_in_theme"] <= MAX_PER_THEME].head(MAX_GLOBAL).copy()
    ranked["rank_global"] = range(1, len(ranked) + 1)
    return ranked.drop(columns=["_state", "_tier"])


def build(as_of: str, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    quality = _read_csv(output / "carrier_qualification.csv")
    leaders = _read_csv(output / "leader_state_daily.csv")
    attention = _read_csv(output / "attention_lifecycle.csv")
    if leaders.empty:
        raise RuntimeError("leader_state_daily.csv is missing or empty; run p1_daily first")
    quality_by_symbol = {
        str(row["symbol"]): row for row in quality.to_dict("records")
    } if len(quality) else {}
    attention_by_theme = {
        str(row["theme_id"]): row for row in attention.to_dict("records")
    } if len(attention) else {}
    registry = {theme["id"]: theme for theme in themes_as_of(as_of)}
    cache: dict[str, pd.DataFrame] = {}
    rows = []

    for theme_row in leaders.to_dict("records"):
        theme_id = str(theme_row.get("theme_id", ""))
        theme = registry.get(theme_id)
        leader = str(theme_row.get("leader", ""))
        if not theme or not leader:
            continue
        leader_quality = quality_by_symbol.get(leader, {})
        anchor = _theme_anchor(theme_row, leader_quality, as_of)
        member_frames = {}
        for symbol in theme["members"]:
            if symbol not in cache:
                cache[symbol] = _price_features(symbol, as_of)
            progressed = _attach_progress(cache[symbol], anchor) if len(cache[symbol]) else pd.DataFrame()
            if len(progressed):
                member_frames[symbol] = progressed
        if leader not in member_frames:
            continue
        panel = pd.concat([
            frame.assign(symbol=symbol) for symbol, frame in member_frames.items()
        ], ignore_index=True)
        panel["chain_progress_percentile"] = panel.groupby("date")[
            "since_anchor_pct"
        ].rank(pct=True, method="average")
        panel["chain_r3_median"] = panel.groupby("date")["ret3_pct"].transform("median")
        leader_panel = member_frames[leader].set_index("date")
        panel["leader_since_anchor_pct"] = panel["date"].map(
            leader_panel["since_anchor_pct"]
        )
        panel["leader_atr20_pct"] = panel["date"].map(leader_panel["atr20_pct"])
        session_dates = sorted(panel["date"].unique())
        session_age = {date: index for index, date in enumerate(session_dates)}
        anchor_age = session_age.get(as_of, len(session_dates) - 1)
        attention_row = attention_by_theme.get(theme_id, {})

        for symbol, name in theme["members"].items():
            if symbol == leader:
                continue
            candidate = panel[panel["symbol"] == symbol].sort_values("date").copy()
            carrier = quality_by_symbol.get(symbol, {})
            data_fresh = bool(len(candidate) and candidate.iloc[-1]["date"] == as_of)
            if not data_fresh:
                rows.append({
                    "as_of": as_of, "theme_id": theme_id,
                    "theme_label": theme_row.get("theme_label", theme.get("label", theme_id)),
                    "theme_lifecycle": theme_row.get("theme_lifecycle", ""),
                    "leader": leader, "symbol": symbol, "name": name,
                    "slow_stage": "DATA_INCOMPLETE", "action": "NO_ACTION",
                    "anchor_date": anchor, "anchor_age_sessions": anchor_age,
                    "calibration_status": CALIBRATION_STATUS,
                    "rank_basis": "hard gates -> lifecycle -> first response; no weighted score",
                })
                continue

            candidate["valid_lag"] = candidate.apply(_valid_lag, axis=1)
            prepared_since = _latest([
                anchor,
                carrier.get("event_first_seen", ""),
                carrier.get("supply_first_seen", ""),
                carrier.get("structure_first_seen", "")
                if str(carrier.get("shadow_stage", "")) in {"COILED", "ARMED", "READY"}
                else "",
            ]) or anchor
            first_response = ""
            current_response_count = 0
            current_response_basis = ""
            previous_valid = False
            for _, day in candidate.iterrows():
                current_valid = _truth(day["valid_lag"])
                response, count, basis = _response_evidence(day)
                if day["date"] == as_of:
                    current_response_count = count
                    current_response_basis = basis
                if (not first_response and day["date"] >= prepared_since
                        and session_age.get(day["date"], WINDOW_SESSIONS + 1) <= WINDOW_SESSIONS
                        and response and (previous_valid or current_valid)):
                    first_response = str(day["date"])
                previous_valid = current_valid

            current = candidate.iloc[-1]
            prepared = _prepared(carrier)
            qualification = str(carrier.get("carrier_qualification", "NO_CAPITAL_EPISODE"))
            failed = (
                str(carrier.get("production_state", "")) in BLOCKED_PRODUCTION
                or str(carrier.get("retention_state", "")) == "EXITED"
                or qualification in {"DISTRIBUTION_RISK", "EXTENDED_TOO_LATE",
                                     "TRIGGERED_TOO_LATE", "EXITED_OR_BROKEN"}
            )
            crowded = str(attention_row.get("crowding_state", "")) == "CROWDED"
            stage = classify_stage(
                prepared=prepared, data_fresh=data_fresh,
                theme_lifecycle=str(theme_row.get("theme_lifecycle", "")),
                crowded=crowded, anchor_age=anchor_age,
                current_valid_lag=_truth(current["valid_lag"]),
                first_response_date=first_response, as_of=as_of,
                hard_gate_pass=_truth(carrier.get("hard_gate_pass")), failed=failed,
            )
            atr = max(float(current.get("atr20_pct", np.nan)), 0.01)
            lag_gap_atr = (
                float(current.get("leader_since_anchor_pct", np.nan))
                - float(current.get("since_anchor_pct", np.nan))
            ) / atr
            rows.append({
                "as_of": as_of,
                "theme_id": theme_id,
                "theme_label": theme_row.get("theme_label", theme.get("label", theme_id)),
                "theme_lifecycle": theme_row.get("theme_lifecycle", ""),
                "attention_level": attention_row.get("attention_level", "UNKNOWN"),
                "attention_slope": attention_row.get("attention_slope", "UNKNOWN"),
                "leader": leader,
                "leader_state": theme_row.get("leader_state", ""),
                "symbol": symbol,
                "name": name,
                "slow_stage": stage,
                "window_bucket": _window_bucket(anchor_age),
                "action": _action(
                    stage, str(theme_row.get("theme_lifecycle", "")),
                    str(attention_row.get("attention_slope", "UNKNOWN")),
                ),
                "anchor_date": anchor,
                "anchor_age_sessions": anchor_age,
                "prepared_since": prepared_since if prepared else "",
                "first_response_date": first_response,
                "carrier_qualification": qualification,
                "carrier_tier": carrier.get("carrier_tier", "NOT_QUALIFIED"),
                "shadow_stage": carrier.get("shadow_stage", ""),
                "retention_state": carrier.get("retention_state", ""),
                "execution_state": carrier.get("execution_state", ""),
                "gate_failures": carrier.get("gate_failures", ""),
                "leader_since_anchor_pct": round(float(current["leader_since_anchor_pct"]), 2),
                "since_anchor_pct": round(float(current["since_anchor_pct"]), 2),
                "lag_gap_atr": round(float(lag_gap_atr), 2),
                "chain_progress_percentile": round(float(current["chain_progress_percentile"]), 3),
                "ema20_bias_atr": round(float(current["ema20_bias_atr"]), 2),
                "drawdown20_atr": round(float(current["drawdown20_atr"]), 2),
                "amount_ratio1": round(float(current["amount_ratio1"]), 2)
                if _finite(current["amount_ratio1"]) else np.nan,
                "response_evidence_count": current_response_count,
                "response_evidence": current_response_basis,
                "calibration_status": CALIBRATION_STATUS,
                "rank_basis": "hard gates -> lifecycle -> first response; no weighted score",
            })

    diagnostics = pd.DataFrame(rows)
    focus = _cap_focus(diagnostics)
    counts = diagnostics.get("slow_stage", pd.Series(dtype=str)).value_counts().to_dict()
    summary = {
        "as_of": as_of,
        "version": VERSION,
        "engine_version": ENGINE_VERSION,
        "calibration_status": CALIBRATION_STATUS,
        "production_writeback": False,
        "order_generation": False,
        "window_sessions": WINDOW_SESSIONS,
        "diagnostic_total": len(diagnostics),
        "focus_total": len(focus),
        "ready": int(counts.get("READY", 0)),
        "first_response": int(counts.get("FIRST_RESPONSE", 0)),
        "valid_lag": int(counts.get("VALID_LAG", 0)),
        "state_counts": {str(key): int(value) for key, value in counts.items()},
        "top_symbols": focus.get("symbol", pd.Series(dtype=str)).head(5).tolist(),
    }
    return diagnostics, focus, summary


def run(as_of: str) -> int:
    output = OUT_ROOT / as_of.replace("-", "")
    diagnostics, focus, summary = build(as_of, output)
    diagnostics.to_csv(output / "slow_carrier_diagnostics.csv", index=False,
                       encoding="utf-8-sig")
    focus.to_csv(output / "slow_carrier_watch.csv", index=False, encoding="utf-8-sig")
    (output / "slow_carrier_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    columns = [
        "rank_global", "theme_label", "leader", "symbol", "name", "slow_stage",
        "window_bucket", "theme_lifecycle", "attention_slope", "carrier_tier",
        "lag_gap_atr", "chain_progress_percentile", "response_evidence", "action",
    ]
    lines = [
        f"# Slow-carrier shadow watch  {as_of}", "",
        f"- Version: `{VERSION}` / engine `{ENGINE_VERSION}`.",
        f"- Calibration: `{CALIBRATION_STATUS}`; thresholds collect forward evidence and "
        "must not be promoted before a frozen historical/OOS review.",
        "- Lifecycle: PREPARED -> VALID_LAG -> FIRST_RESPONSE -> READY; "
        "EXPIRED/FAILED are terminal for the current theme window.",
        "- A valid laggard must be a registered tight-chain member with retained capital, "
        "shrinking supply and a mature structure. Low price alone never qualifies.",
        "- Price, structure and either volume or relative-strength response are required. "
        "All volatility comparisons are ATR-normalized and chain-relative.",
        f"- Candidate cap: {MAX_PER_THEME} per theme / {MAX_GLOBAL} market-wide. "
        "This overlay is report-only and creates no orders.", "",
        focus[[column for column in columns if column in focus.columns]].to_markdown(index=False)
        if len(focus) else "No slow-carrier candidate passed the current lifecycle gates.", "",
        "## State counts", "",
        " | ".join(f"{key}={value}" for key, value in summary["state_counts"].items())
        if summary["state_counts"] else "No diagnostics.",
    ]
    (output / "slow_carrier_watch.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[slow_carrier] focus={summary['focus_total']} ready={summary['ready']} "
          f"response={summary['first_response']} lag={summary['valid_lag']}")
    print(f"[slow_carrier] -> {output / 'slow_carrier_watch.md'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    date = args.date if "-" in args.date else f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}"
    return run(date)


if __name__ == "__main__":
    raise SystemExit(main())
