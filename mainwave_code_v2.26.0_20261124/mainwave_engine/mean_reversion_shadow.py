# -*- coding: utf-8 -*-
"""Weak-market mean-reversion overlay using frozen, report-only thresholds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import __version__ as ENGINE_VERSION
from research import mean_reversion_threshold_study as STUDY


VERSION = "mean_reversion_shadow_v1_20260903"
OUT_ROOT = Path("results/mainwave_engine")
CONFIG = Path("mainwave_engine/calibration/mean_reversion_shadow_v1.json")
MIN_STOCK_AMOUNT = 50_000_000
SIGNAL_ACTIONS = {"SHADOW_SIGNAL_NEXT_OPEN", "SHADOW_REPAIR_SIGNAL_NEXT_OPEN"}
WATCH_ACTIONS = {"SHADOW_WATCH_REPAIR", "SHADOW_WATCH_PANIC_NO_ENTRY", "SHADOW_NEAR_TRIGGER"}


def classify_action(*, enabled: bool, config_effective: bool = True,
                    data_fresh: bool, market_context: str,
                    market_temperature: str, active_contexts: set[str], oversold: bool,
                    near_trigger: bool, cross_age: int | None, mode: str, ttl: int,
                    confirm_date: str, as_of: str) -> str:
    """Keep oversold detection separate from entry permission."""
    if not config_effective:
        return "CONFIG_NOT_EFFECTIVE_YET"
    if not enabled:
        return "NO_VALIDATED_THRESHOLD"
    if not data_fresh:
        return "DATA_STALE"
    if market_context not in active_contexts:
        return "INACTIVE_MARKET_CONTEXT"
    if market_temperature == "PANIC":
        return "SHADOW_WATCH_PANIC_NO_ENTRY" if oversold or near_trigger else "NO_SIGNAL"
    if cross_age is None:
        return "SHADOW_NEAR_TRIGGER" if near_trigger else "NO_SIGNAL"
    if cross_age > ttl:
        return "OVERSOLD_STALE" if oversold else "SIGNAL_EXPIRED"
    if mode == "NEXT_OPEN":
        return "SHADOW_SIGNAL_NEXT_OPEN" if cross_age == 0 else "SIGNAL_EXPIRED"
    if mode == "REPAIR_CONFIRM":
        if confirm_date == as_of:
            return "SHADOW_REPAIR_SIGNAL_NEXT_OPEN"
        if confirm_date:
            return "SIGNAL_EXPIRED"
        return "SHADOW_WATCH_REPAIR"
    return "NO_SIGNAL"


def _load_config() -> dict:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("sectors"), dict):
        raise RuntimeError(f"invalid mean-reversion shadow config: {CONFIG}")
    if payload.get("production_enabled") is not False:
        raise RuntimeError("mean-reversion shadow must remain report-only")
    return payload


def _frames(as_of: pd.Timestamp, definition: dict, market: pd.DataFrame) -> dict[str, pd.DataFrame]:
    frames = {}
    for symbol in definition["members"]:
        path = Path("data") / f"{symbol}_daily.csv"
        if not path.exists():
            continue
        frame = STUDY.load_stock(path)
        frame = frame[frame["date"] <= as_of].reset_index(drop=True)
        if len(frame):
            frames[symbol] = STUDY.attach_market(frame, market)
    return frames


def _cross_state(frame: pd.DataFrame, threshold: float, scope: str) -> tuple[int | None, str]:
    amount_floor = MIN_STOCK_AMOUNT if scope == "STOCK" else 0
    eligible = frame["drawdown60_pct"].notna() & frame["amount_med20"].ge(amount_floor)
    crossings = STUDY.crossing_indices(frame["drawdown60_pct"], threshold, eligible)
    if not crossings:
        return None, ""
    latest = crossings[-1]
    age = len(frame) - 1 - latest
    return int(age), frame.at[latest, "date"].date().isoformat()


def _confirm_date(frame: pd.DataFrame, cross_age: int | None, mode: str) -> str:
    if cross_age is None or mode != "REPAIR_CONFIRM":
        return ""
    anchor = len(frame) - 1 - cross_age
    _, confirm = STUDY.find_entry_index(frame, anchor, mode)
    if confirm is None:
        return ""
    return frame.at[confirm, "date"].date().isoformat()


def build(as_of: str) -> tuple[pd.DataFrame, dict]:
    config = _load_config()
    as_of_ts = pd.Timestamp(as_of)
    market = STUDY.build_market_frame()
    current_market = market[market["date"] <= as_of_ts]
    if current_market.empty:
        raise RuntimeError(f"market context unavailable at {as_of}")
    market_row = current_market.iloc[-1]
    market_date = market_row["date"].date().isoformat()
    market_data_fresh = market_date == as_of
    active_contexts = set(config["active_market_contexts"])
    config_effective = as_of >= str(config.get("effective_from", config["frozen_at"]))
    rows = []

    for sector, definition in config["sectors"].items():
        stock_frames = _frames(as_of_ts, definition, market)
        items = []
        if len(stock_frames) >= 3:
            basket = STUDY.build_basket(stock_frames, minimum_members=3)
            items.append(("BASKET", sector, definition["label"], STUDY.attach_market(basket, market)))
        items.extend(
            ("STOCK", symbol, definition["members"][symbol], frame)
            for symbol, frame in stock_frames.items()
        )
        enabled = bool(definition.get("enabled"))
        threshold = definition.get("threshold_pct")
        mode = str(definition.get("entry_mode", "NONE"))
        ttl = int(definition.get("signal_ttl_sessions", 0))
        for scope, symbol, name, frame in items:
            current = frame.iloc[-1]
            current_date = current["date"].date().isoformat()
            data_fresh = current_date == as_of and market_data_fresh
            oversold = bool(enabled and threshold is not None
                            and float(current["drawdown60_pct"]) <= float(threshold))
            near_trigger = bool(enabled and threshold is not None
                                and float(current["drawdown60_pct"]) <= float(threshold) + 3)
            if enabled and threshold is not None:
                cross_age, cross_date = _cross_state(frame, float(threshold), scope)
                confirm_date = _confirm_date(frame, cross_age, mode)
            else:
                cross_age, cross_date, confirm_date = None, "", ""
            action = classify_action(
                enabled=enabled, data_fresh=data_fresh,
                config_effective=config_effective,
                market_context=str(current.get("market_context", "")),
                market_temperature=str(current.get("temperature", "")),
                active_contexts=active_contexts, oversold=oversold,
                near_trigger=near_trigger, cross_age=cross_age, mode=mode, ttl=ttl,
                confirm_date=confirm_date, as_of=as_of,
            )
            rows.append({
                "as_of": as_of,
                "sector": sector,
                "sector_label": definition["label"],
                "scope": scope,
                "symbol": symbol,
                "name": name,
                "close": round(float(current["raw_close"]), 3),
                "drawdown20_pct": round(float(current["drawdown20_pct"]), 2),
                "drawdown60_pct": round(float(current["drawdown60_pct"]), 2),
                "ema20_bias_pct": round(float(current["ema20_bias_pct"]), 2),
                "atr_deviation": round(float(current["atr_deviation"]), 2),
                "ret5_pct": round(float(current["ret5_pct"]), 2),
                "ret20_pct": round(float(current["ret20_pct"]), 2),
                "market_structure": current.get("structure", ""),
                "market_temperature": current.get("temperature", ""),
                "market_context": current.get("market_context", ""),
                "threshold_pct": threshold,
                "entry_mode": mode,
                "evidence_class": definition.get("evidence_class", ""),
                "latest_cross_date": cross_date,
                "cross_age_sessions": cross_age,
                "confirm_date": confirm_date,
                "oversold_now": oversold,
                "near_trigger": near_trigger,
                "action": action,
                "production_enabled": False,
            })

    frame = pd.DataFrame(rows)
    actions = frame.get("action", pd.Series(dtype=str)).value_counts().to_dict()
    summary = {
        "as_of": as_of,
        "version": VERSION,
        "config_version": config["version"],
        "engine_version": ENGINE_VERSION,
        "market_date": market_date,
        "market_data_fresh": market_data_fresh,
        "config_effective": config_effective,
        "market_structure": str(market_row["structure"]),
        "market_temperature": str(market_row["temperature"]),
        "market_context": str(market_row["market_context"]),
        "active_context": (
            market_data_fresh and str(market_row["market_context"]) in active_contexts
        ),
        "production_writeback": False,
        "order_generation": False,
        "signals": int(frame["action"].isin(SIGNAL_ACTIONS).sum()) if len(frame) else 0,
        "watches": int(frame["action"].isin(WATCH_ACTIONS).sum()) if len(frame) else 0,
        "action_counts": {str(key): int(value) for key, value in actions.items()},
        "signal_symbols": frame.loc[
            frame["action"].isin(SIGNAL_ACTIONS) & frame["scope"].eq("STOCK"), "symbol"
        ].head(5).tolist() if len(frame) else [],
    }
    return frame, summary


def run(as_of: str) -> int:
    output = OUT_ROOT / as_of.replace("-", "")
    output.mkdir(parents=True, exist_ok=True)
    frame, summary = build(as_of)
    frame.to_csv(output / "mean_reversion_shadow.csv", index=False, encoding="utf-8-sig")
    (output / "mean_reversion_shadow_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    visible = frame[
        frame["action"].isin(SIGNAL_ACTIONS | WATCH_ACTIONS | {"OVERSOLD_STALE"})
        | frame["scope"].eq("BASKET")
    ].copy()
    columns = [
        "sector_label", "scope", "symbol", "name", "drawdown60_pct",
        "ema20_bias_pct", "atr_deviation", "market_context", "threshold_pct",
        "evidence_class", "latest_cross_date", "cross_age_sessions", "action",
    ]
    lines = [
        f"# Mean-reversion shadow watch  {as_of}", "",
        f"- Version: `{VERSION}` / config `{summary['config_version']}`.",
        f"- Market: {summary['market_structure']} / {summary['market_temperature']} / "
        f"{summary['market_context']}; date={summary['market_date']}; "
        f"fresh={summary['market_data_fresh']}; "
        f"weak-market overlay active={summary['active_context']}.",
        "- This is independent from the main-wave lifecycle. A deep drawdown is not capital "
        "retention, theme propagation, or a buy signal.",
        "- Thresholds are frozen prospective shadow parameters. PANIC permits observation only; "
        "stale crossings cannot be promoted.",
        "- Storage has no validated threshold and must remain NO_VALIDATED_THRESHOLD.",
        "- Report-only: no production state writeback and no orders.", "",
        visible[[column for column in columns if column in visible.columns]].head(30).to_markdown(index=False)
        if len(visible) else "No current mean-reversion watch item.",
    ]
    (output / "mean_reversion_shadow.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[mean_reversion] market={summary['market_context']} "
          f"signals={summary['signals']} watches={summary['watches']}")
    print(f"[mean_reversion] -> {output / 'mean_reversion_shadow.md'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    date = args.date if "-" in args.date else f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}"
    return run(date)


if __name__ == "__main__":
    raise SystemExit(main())
