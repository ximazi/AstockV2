"""Causal first EMA pullback research scan. No production state or order writes."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mainwave_engine import phase_events as PE

VERSION = "first_pullback_research_v1_20260906"
TERMINAL = {"CONFIRMED", "FAILED", "EXPIRED", "RECOVERED_WITHOUT_CONFIRMATION"}
# Frozen research placeholders, not calibrated probabilities or trading thresholds.
RULES = {
    "base_sessions": 20, "base_below_ema20_days": 5, "base_evidence_days": 3,
    "base_width_atr_max": 8.0, "base_ret20_abs_max": 0.12,
    "launch_amount_ratio_min": 1.2, "launch_body_atr_min": 1.0,
    "pullback_atr_min": 0.8, "pullback_depth_atr_max": 3.0,
    "pullback_sessions_min": 2, "pullback_sessions_max": 20,
    "supply_ratio_max": 0.85, "demand_ratio_min": 1.2,
    "ema_touch_atr": 0.3, "confirmation_close_location_min": 0.65,
    "confirmation_recent_sessions": 3, "extension_atr_max": 2.0,
}


def prepare(raw: pd.DataFrame, as_of: str) -> pd.DataFrame:
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw[raw.date.le(pd.Timestamp(as_of))].sort_values("date").drop_duplicates("date", keep="last")
    raw = raw[pd.to_numeric(raw["amount"], errors="coerce").gt(0)]
    if "tradestatus" in raw:
        raw = raw[pd.to_numeric(raw.tradestatus, errors="coerce").eq(1)]
    frame = PE.classify_frame(raw)
    if frame.empty:
        return frame
    scale = frame.return_index / frame.close
    frame["h"] = frame.high * scale
    frame["l"] = frame.low * scale
    frame["c"] = frame.return_index
    frame["ema10"] = frame.c.ewm(span=10, adjust=False).mean()
    frame["ema20"] = frame.c.ewm(span=20, adjust=False).mean()
    base_evidence = (frame.broad_phase.isin(["ACCUMULATION", "REACCUMULATION"])
                     | frame.effort_result.isin(["ABSORPTION", "NO_SUPPLY"]))
    frame["base_evidence_days"] = base_evidence.shift(1).rolling(20).sum()
    frame["below_days"] = frame.c.le(frame.ema20).shift(1).rolling(20).sum()
    width = (frame.structure_ceiling - frame.structure_floor) / frame.atr20
    frame["launch"] = (
        frame.base_evidence_days.ge(RULES["base_evidence_days"])
        & frame.below_days.ge(RULES["base_below_ema20_days"])
        & width.le(RULES["base_width_atr_max"])
        & frame.ret20.shift(1).abs().le(RULES["base_ret20_abs_max"])
        & frame.c.gt(frame.ema10) & frame.c.gt(frame.ema20)
        & frame.c.gt(frame.h.shift(1).rolling(5).max() + 0.1 * frame.atr20)
        & (frame.c - frame.c.shift(1)).ge(RULES["launch_body_atr_min"] * frame.atr20)
        & frame.amount_ratio.ge(RULES["launch_amount_ratio_min"])
        & frame.close_location.ge(RULES["confirmation_close_location_min"])
    )
    frame["turn"] = pd.to_numeric(frame.get("turnover", np.nan), errors="coerce")
    return frame


def episodes(frame: pd.DataFrame) -> list[dict]:
    """Consume the first correction even if it fails; restart only after a new base."""
    records, active = [], None
    c, h, low = (frame[k].to_numpy(float) for k in ("c", "h", "l"))
    ema10, ema20, turn = (frame[k].to_numpy(float) for k in ("ema10", "ema20", "turn"))
    dates = frame.date.dt.strftime("%Y-%m-%d").tolist()
    for i, row in enumerate(frame.to_dict("records")):
        if i < 65:
            continue
        # A later correction cannot acquire 'first' identity without 20 new base days.
        new_base = active is None or (active["status"] in TERMINAL
                                     and i - active["ended_index"] > RULES["base_sessions"])
        if row["launch"] and new_base:
            active = {"launch_index": i, "launch_date": dates[i], "peak_index": i,
                      "base_evidence_days": int(row["base_evidence_days"]),
                      "base_floor": float(row["structure_floor"]), "atr": float(row["atr20"]),
                      "status": "ADVANCING", "pullback_index": None,
                      "confirmed_date": "", "ended_index": i}
            records.append(active)
        if active is None or active["status"] in TERMINAL:
            continue
        a, peak, atr = active["launch_index"], active["peak_index"], active["atr"]
        if c[i] < active["base_floor"]:
            active.update(status="FAILED", ended_index=i, failure="BASE_BROKEN")
            continue
        if active["status"] == "ADVANCING":
            if c[i] > c[peak]:
                active["peak_index"] = peak = i
            if i - a > 40:
                active.update(status="EXPIRED", ended_index=i)
            elif (i - peak >= RULES["pullback_sessions_min"]
                  and c[peak] - c[i] >= RULES["pullback_atr_min"] * atr):
                active.update(status="FORMING", pullback_index=peak + 1,
                              pullback_date=dates[peak + 1])
        if active["status"] != "FORMING":
            continue
        start = active["pullback_index"]
        if i - start + 1 > RULES["pullback_sessions_max"]:
            active.update(status="EXPIRED", ended_index=i)
            continue
        depth = (c[peak] - min(low[start:i + 1])) / atr
        if depth > RULES["pullback_depth_atr_max"]:
            active.update(status="FAILED", ended_index=i, failure="PULLBACK_TOO_DEEP")
            continue
        if i <= start:
            continue
        impulse = turn[a:peak + 1]
        supply = turn[start:i]
        valid_turn = np.isfinite(impulse).all() and np.isfinite(supply).all() and (supply > 0).all()
        contraction = float(np.median(supply) / np.median(impulse)) if valid_turn and np.median(impulse) > 0 else np.nan
        demand = turn[i] / np.median(supply) if valid_turn else np.nan
        touching = bool(np.any(low[start:i + 1] <= ema10[start:i + 1] + RULES["ema_touch_atr"] * atr))
        pivot = float(max(h[max(start, i - 2):i]) + 0.05 * atr)
        invalid = float(min(low[start:i + 1]) - 0.1 * atr)
        active.update(supply_ratio=contraction, demand_ratio=float(demand), touched_ema=touching,
                      trigger=pivot, invalid=invalid, depth_atr=float(depth),
                      pullback_sessions=i - start + 1,
                      dry_days=int(np.sum(supply < np.median(impulse))) if valid_turn else 0,
                      supply_days=len(supply), latest_date=dates[i])
        price_confirmed = (touching and c[i] > max(pivot, ema10[i], ema20[i])
                           and ema20[i] >= ema20[i - 1]
                           and row["close_location"] >= RULES["confirmation_close_location_min"])
        if (price_confirmed and contraction <= RULES["supply_ratio_max"]
                and active["dry_days"] >= len(supply) / 2
                and demand >= RULES["demand_ratio_min"]):
            active.update(status="CONFIRMED", confirmed_date=dates[i], ended_index=i,
                          confirmed_close=float(c[i]))
        elif c[i] > c[peak]:
            # The correction is over in PRICE even when demand did not qualify.
            # A later volume burst must not be relabelled as its first recovery.
            active.update(status="RECOVERED_WITHOUT_CONFIRMATION", ended_index=i)
    return records


def summarize(frame: pd.DataFrame, events: list[dict]) -> dict:
    if not events:
        return {"scan_status": "NO_BASE_LAUNCH"}
    e = events[-1]
    row = frame.iloc[-1]
    age = len(frame) - 1 - e["ended_index"]
    status = e["status"]
    if status == "CONFIRMED" and age > RULES["confirmation_recent_sessions"] - 1:
        status = "HISTORICAL_CONFIRMED"
    risk = []
    if status == "CONFIRMED":
        after = frame.iloc[e["ended_index"] + 1:]
        if after.c.lt(e["invalid"]).any():
            risk.append("POST_CONFIRMATION_BROKEN")
        if row.c < max(row.ema10, row.ema20, e["trigger"]):
            risk.append("RECLAIM_LOST")
    if row.broad_phase in {"DISTRIBUTION", "CLIMAX", "MARKDOWN"}:
        risk.append(str(row.broad_phase))
    extension = (row.c - row.ema20) / e["atr"]
    if extension > RULES["extension_atr_max"]:
        risk.append("EXTENDED_ABOVE_EMA20")
    anchor = row.close / row.c
    return {"scan_status": status, "first_scope": "LOCAL_BASE_FIRST_NOT_GLOBAL_CYCLE_PROOF",
            **{k: v for k, v in e.items() if not k.endswith("index")},
            "risk_flags": "|".join(risk), "close": row.close,
            "ema10": row.ema10 * anchor, "ema20": row.ema20 * anchor,
            "ema20_bias_pct": (row.c / row.ema20 - 1) * 100,
            "ret5_pct": row.ret5 * 100, "ret20_pct": row.ret20 * 100,
            "turnover_pct": row.turn, "extension_atr": extension,
            "trigger_price": e.get("trigger", np.nan) * anchor,
            "invalid_price": e.get("invalid", np.nan) * anchor,
            "risk_distance_pct": (1 - e.get("invalid", np.nan) / row.c) * 100,
            "confirmed_age": age if e["status"] == "CONFIRMED" else None,
            "broad_phase": row.broad_phase, "effort_result": row.effort_result,
            "price_date": row.date.strftime("%Y-%m-%d")}


def run(as_of: str, data_root: Path, output: Path) -> pd.DataFrame:
    started = time.perf_counter()
    mapping_payload = json.loads(Path("concept_mapping.json").read_text(encoding="utf-8"))
    mapping = mapping_payload.get("mapping", {})
    report = Path("results/mainwave_engine") / as_of.replace("-", "")
    market_file = report / "mainwave_three_direction_summary.json"
    market = json.loads(market_file.read_text(encoding="utf-8")).get("market_risk", {}).get("state", "UNKNOWN") if market_file.exists() else "UNKNOWN"
    paths = [p for p in sorted(data_root.glob("*_daily.csv"))
             if re.fullmatch(r"(?:sh_(?:60|68)\d{4}|sz_(?:00|30)\d{4}|bj_\d{6})_daily", p.stem)]
    results, audit, history = [], [], []
    history_from = (pd.Timestamp(as_of) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    for n, path in enumerate(paths, 1):
        symbol = path.stem.removesuffix("_daily")
        try:
            frame = prepare(pd.read_csv(path), as_of)
            if frame.empty:
                audit.append({"symbol": symbol, "status": "SHORT_HISTORY"})
                continue
            if frame.date.iloc[-1] != pd.Timestamp(as_of):
                audit.append({"symbol": symbol, "status": "STALE", "price_date": str(frame.date.iloc[-1].date())})
                continue
            if "isST" in frame and pd.to_numeric(frame.isST.iloc[-1], errors="coerce") == 1:
                audit.append({"symbol": symbol, "status": "ST_EXCLUDED"})
                continue
            events = episodes(frame)
            summary = summarize(frame, events)
            audit.append({"symbol": symbol, "status": summary["scan_status"]})
            if events:
                identity = mapping.get(symbol, {})
                summary.update(symbol=symbol, name=identity.get("name", ""),
                               industry=identity.get("industry", ""),
                               themes="|".join(identity.get("concepts", [])),
                               market_state=market, execution_permission="RESEARCH_ONLY_NO_ORDER")
                results.append(summary)
                history.extend({"symbol": symbol, **e} for e in events if e["launch_date"] >= history_from)
        except (ValueError, KeyError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            audit.append({"symbol": symbol, "status": "ERROR", "error": str(exc)})
        if n % 500 == 0:
            print(f"[first_pullback] {n}/{len(paths)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    all_rows = pd.DataFrame(results)
    all_rows.to_csv(output / "all_episodes_latest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(output / "universe_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(output / "episode_history.csv", index=False, encoding="utf-8-sig")
    focus = all_rows[all_rows.scan_status.isin(["CONFIRMED", "FORMING"])].copy() if len(all_rows) else all_rows
    if len(focus):
        focus["priority"] = np.select([
            focus.scan_status.eq("CONFIRMED") & focus.risk_flags.eq(""),
            focus.scan_status.eq("FORMING") & focus.risk_flags.eq("")
            & focus.supply_ratio.le(RULES["supply_ratio_max"]) & focus.touched_ema.eq(True),
        ], [1, 2], default=3)
        focus = focus.sort_values(["priority", "risk_distance_pct", "symbol"])
    focus.to_csv(output / "first_pullback_candidates.csv", index=False, encoding="utf-8-sig")
    counts = pd.DataFrame(audit).status.value_counts().to_dict()
    manifest = {"as_of": as_of, "version": VERSION, "rules": RULES,
                "validation": "UNVALIDATED_RESEARCH", "universe_files": len(paths),
                "status_counts": counts, "market_state": market,
                "mapping_as_of": mapping_payload.get("generated"), "theme_role": "CURRENT_DESCRIPTIVE_ONLY",
                "first_scope": "LOCAL_BASE_FIRST_NOT_GLOBAL_CYCLE_PROOF",
                "seconds": round(time.perf_counter() - started, 2),
                "production_modified": False, "orders_generated": 0}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return focus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    as_of = str(pd.Timestamp(args.date).date())
    output = args.output or Path("results/first_pullback_research") / as_of.replace("-", "")
    run(as_of, args.data_root, output)


if __name__ == "__main__":
    main()
