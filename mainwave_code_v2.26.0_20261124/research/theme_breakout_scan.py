"""Broad price-breakout research with existing dated theme context. No orders."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mainwave_engine.theme_semantics import load_mapping_as_of
from research.first_pullback_scan import prepare

VERSION = "theme_breakout_research_v1_20260906"
RULES = {"break_buffer_atr": 0.1, "amount_ratio_min": 1.2,
         "close_location_min": 0.6, "recent_sessions": 5,
         "near_break_atr": 0.5, "extension_warning_atr": 3.0}
GENERIC = {"融资融券", "沪股通", "深股通", "国企改革", "央企国企改革", "一带一路",
           "西部大开发", "海峡两岸", "福建自贸区", "共同富裕示范区", "参股券商",
           "股权转让(并购重组)", "乡村振兴"}


def snapshot(frame: pd.DataFrame) -> dict:
    x = frame.iloc[-1]
    c, h, atr = frame.c, frame.h, frame.atr20
    level20 = h.shift(1).rolling(20).max()
    level60 = h.shift(1).rolling(60).max()
    volume_ok = frame.amount_ratio.ge(RULES["amount_ratio_min"])
    close_ok = frame.close_location.ge(RULES["close_location_min"])
    breaks = c.gt(level20 + RULES["break_buffer_atr"] * atr) & volume_ok & close_ok
    indices = np.flatnonzero(breaks.to_numpy())
    indices = indices[indices >= len(frame) - RULES["recent_sessions"]]
    route, date, level, retained = "NO_SETUP", "", np.nan, False
    lost_after = False
    if len(indices):
        i = int(indices[-1])
        level = float(level20.iloc[i])
        date = frame.date.iloc[i].strftime("%Y-%m-%d")
        retained = bool(x.c >= max(level, x.ema20))
        lost_after = bool(c.iloc[i + 1:].lt(level).any())
        route = "BREAKOUT_TODAY" if i == len(frame) - 1 else "RECENT_BREAKOUT"
        if not retained:
            route = "BREAKOUT_LOST"
    elif x.c > level20.iloc[-1]:
        route, level = "PRICE_ONLY_BREAKOUT", float(level20.iloc[-1])
    elif (x.c > max(x.ema10, x.ema20) and frame.c.iloc[-2] <= max(frame.ema10.iloc[-2], frame.ema20.iloc[-2])
          and volume_ok.iloc[-1] and close_ok.iloc[-1]):
        route, level = "EMA_RECLAIM_ONLY", float(level20.iloc[-1])
    elif (0 <= (level20.iloc[-1] - x.c) / x.atr20 <= RULES["near_break_atr"]
          and x.c >= max(x.ema10, x.ema20) and x.ema20 >= frame.ema20.iloc[-6]
          and frame.h.tail(5).max() - frame.l.tail(5).min() <= 3 * x.atr20):
        route, level = "NEAR_BREAKOUT", float(level20.iloc[-1])
    anchor = x.close / x.c
    extension = (x.c - x.ema20) / x.atr20
    flags = []
    if x.broad_phase in {"CLIMAX", "DISTRIBUTION", "MARKDOWN"}:
        flags.append(x.broad_phase)
    if extension > RULES["extension_warning_atr"]:
        flags.append("EXTENDED_ABOVE_EMA20")
    if lost_after:
        flags.append("BREAK_LEVEL_LOST_AFTER_EVENT")
    if x.amount < 3e7:
        flags.append("LOW_LIQUIDITY")
    return {"price_date": x.date.strftime("%Y-%m-%d"), "route": route,
            "breakout_date": date, "breakout_level": level * anchor,
            "breakout_retained": retained, "close": x.close,
            "ema10": x.ema10 * anchor, "ema20": x.ema20 * anchor,
            "ema60": c.ewm(span=60, adjust=False).mean().iloc[-1] * anchor,
            "ema20_bias_pct": (x.c / x.ema20 - 1) * 100,
            "ret5_pct": x.ret5 * 100, "ret20_pct": x.ret20 * 100,
            "turnover_pct": x.turn, "amount_ratio": x.amount_ratio,
            "amount_yi": x.amount / 1e8, "close_location": x.close_location,
            "high20": level20.iloc[-1] * anchor, "high60": level60.iloc[-1] * anchor,
            "above_high60": bool(x.c > level60.iloc[-1]),
            "distance_to_high20_pct": (level20.iloc[-1] / x.c - 1) * 100,
            "extension_atr": extension, "risk_flags": "|".join(flags),
            "broad_phase": x.broad_phase, "effort_result": x.effort_result,
            "reference_low5": frame.l.tail(5).min() * anchor,
            "execution_permission": "RESEARCH_ONLY_NO_ORDER"}


def theme_context(identity: dict, calendar: list[dict], horizon: dict, proposals: dict,
                  propagation: dict | None = None) -> dict:
    concepts = set(identity.get("concepts", []))
    terms = concepts | {identity.get("industry", "")}
    terms.discard("")
    matches = [t for t in calendar if any(alias in term for alias in t["aliases"] for term in terms)]
    hot = [name for name in concepts if name not in GENERIC and name in proposals
           and float(proposals[name].get("heat_score", 0)) >= 25]
    hot.sort(key=lambda name: (-float(proposals[name]["heat_score"]), name))
    spreading = sorted(concepts & set(propagation or {}))
    return {"calendar_themes": "|".join(t["label"] for t in matches),
            "calendar_context": "|".join(t["label"] + ":" + str(horizon.get(t["id"], {}).get("priority_band", "UNKNOWN")) for t in matches),
            "hot_themes": "|".join(hot),
            "hot_theme_states": "|".join(name + ":" + str(proposals[name].get("hotspot_lifecycle", "")) for name in hot),
            "propagation_themes": "|".join(spreading),
            "propagation_context": "|".join(name + ":" + propagation[name] for name in spreading),
            "all_concepts": "|".join(sorted(concepts)),
            "theme_evidence": "CONCEPT_OR_INDUSTRY_MATCH_NEEDS_BUSINESS_REVIEW"}


def run(as_of: str, output: Path):
    started = time.perf_counter()
    mapping, mapping_meta = load_mapping_as_of(as_of)
    calendar_payload = json.loads(Path("mainwave_theme_calendar.json").read_text(encoding="utf-8"))
    report = Path("results/mainwave_engine") / as_of.replace("-", "")
    horizon = pd.read_csv(report / "theme_horizon.csv").set_index("theme_id").to_dict("index")
    proposals = pd.read_csv(report / "theme_proposals.csv").set_index("theme_label").to_dict("index")
    propagation_frame = pd.read_csv(report / "direction3_theme_propagation.csv").fillna("")
    propagation = {}
    for theme, group in propagation_frame.groupby("theme"):
        if theme in GENERIC:
            continue
        active = group[group.lifecycle.isin(["IGNITED", "DIFFUSING"])]
        if len(active):
            propagation[theme] = ";".join(sorted(set(
                str(r.t0) + ":" + r.lifecycle + ":" + r.follow_through for r in active.itertuples())))
    market = json.loads((report / "mainwave_three_direction_summary.json").read_text(encoding="utf-8"))["market_risk"]["state"]
    # Current calendar is not a historical membership archive. Refuse to backdate it.
    if as_of < max(calendar_payload.get("effective_from", ""), calendar_payload.get("authored_on", "")):
        raise ValueError("Theme calendar was not known on the requested date")
    paths = [p for p in sorted(Path("data").glob("*_daily.csv"))
             if re.fullmatch(r"(?:sh_(?:60|68)\d{4}|sz_(?:00|30)\d{4}|bj_\d{6})_daily", p.stem)]
    rows, audit = [], []
    for n, path in enumerate(paths, 1):
        symbol = path.stem.removesuffix("_daily")
        try:
            f = prepare(pd.read_csv(path), as_of)
            if f.empty:
                state = "SHORT_HISTORY"
            elif f.date.iloc[-1] != pd.Timestamp(as_of):
                state = "STALE"
            elif "isST" in f and pd.to_numeric(f.isST.iloc[-1], errors="coerce") == 1:
                state = "ST_EXCLUDED"
            else:
                identity = mapping.get(symbol, {})
                row = {"symbol": symbol, "name": identity.get("name", ""),
                       "industry": identity.get("industry", ""), **snapshot(f),
                       **theme_context(identity, calendar_payload["themes"], horizon, proposals, propagation),
                       "market_state": market}
                state = row["route"]
                rows.append(row)
            audit.append({"symbol": symbol, "status": state})
        except (KeyError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            audit.append({"symbol": symbol, "status": "ERROR", "error": str(exc)})
        if n % 1000 == 0:
            print(f"[theme_breakout] {n}/{len(paths)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "all_stock_snapshot.csv", index=False, encoding="utf-8-sig")
    frame[frame.route.ne("NO_SETUP")].to_csv(output / "all_paths.csv", index=False, encoding="utf-8-sig")
    themed = frame[frame.route.ne("NO_SETUP") & (frame.calendar_themes.ne("") | frame.hot_themes.ne("")
                                                | frame.propagation_themes.ne(""))]
    themed.to_csv(output / "theme_paths.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(output / "universe_audit.csv", index=False, encoding="utf-8-sig")
    manifest = {"as_of": as_of, "version": VERSION, "rules": RULES,
                "universe": len(paths), "exchanges": pd.Series([p.name[:2] for p in paths]).value_counts().to_dict(),
                "audit_counts": pd.DataFrame(audit).status.value_counts().to_dict(),
                "themed_counts": themed.route.value_counts().to_dict(),
                "market_state": market, "mapping_meta": mapping_meta,
                "calendar_version": calendar_payload["version"],
                "validation": "UNVALIDATED_RESEARCH", "production_modified": False,
                "orders_generated": 0, "seconds": round(time.perf_counter() - started, 2)}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    day = str(pd.Timestamp(args.date).date())
    run(day, Path("results/theme_breakout_research") / day.replace("-", ""))
