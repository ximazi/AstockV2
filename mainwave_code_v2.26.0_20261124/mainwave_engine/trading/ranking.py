"""Frozen, explainable ranking baseline, separate from executable permissions."""
from collections import Counter
import math
import pandas as pd
from . import store

VERSION = "strength_leader_baseline_v1_20260919"
RULES = {"strength": {"relative_momentum": .5, "drawdown": .25, "efficiency": .25},
         "min_peer_count": 3, "history_sessions": 61, "evaluation_horizons": [5, 10, 20],
         "knowledge_ids": ["F06-K02", "F06-K04", "F04-K01", "F04-K04", "F07-K01"],
         "validation": "FROZEN_UNCALIBRATED_BASELINE", "auto_tuning": False}
MODE_LABELS = {"TREND_PULLBACK": "趋势回踩", "STRENGTH_CONFIRM": "强势确认", "OVERNIGHT": "尾盘隔夜"}


def number(value):
    try:
        val = float(value)
        return val if math.isfinite(val) else None
    except (ValueError, TypeError):
        return None


def alpha_scope(route_ids):
    """R3 is a research lead, not a priced alpha: an R3-only candidate never enters strategy statistics."""
    routes = {r for r in str(route_ids or "").split("|") if r}
    return "R3_RESEARCH" if routes and routes <= {"R3"} else "PRICE_ALPHA"


def prices(path, day):
    f = pd.read_csv(path)
    f["date"] = pd.to_datetime(f["date"], errors="raise").dt.strftime("%Y-%m-%d")
    return f[f.date.le(day)].sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def features(frame, day):
    f = frame[frame.date.le(day)].tail(80).copy()
    if len(f) < RULES["history_sessions"] or f.date.iloc[-1] != day:
        return {"status": "DATA_WAIT"}
    pct = pd.to_numeric(f.pct_chg, errors="coerce")
    close = pd.to_numeric(f.close, errors="coerce")
    volume = number(f.volume.iloc[-1])
    if not all(math.isfinite(x) for x in pct) or pct.le(-100).any() or not all(math.isfinite(x) for x in close) or close.le(0).any() or volume is None or volume <= 0:
        return {"status": "DATA_WAIT"}
    index = (1 + pct / 100).cumprod()
    latest = f.iloc[-1]
    width = number(latest.high - latest.low)
    movement = index.diff().abs().tail(10).sum()
    turnover = pd.to_numeric(f.turnover, errors="coerce") if "turnover" in f else pd.Series(dtype=float)
    base = turnover.iloc[-21:-1].median() if len(turnover) else None
    return store.clean({"status": "READY", "close": float(close.iloc[-1]),
        "return5": float((index.iloc[-1] / index.iloc[-6] - 1) * 100),
        "return20": float((index.iloc[-1] / index.iloc[-21] - 1) * 100),
        "drawdown20": float((index.iloc[-1] / index.tail(20).max() - 1) * 100),
        "efficiency10": float((index.iloc[-1] - index.iloc[-11]) / movement) if movement else 0.,
        "close_location": float((latest.close - latest.low) / width) if width and width > 0 else None,
        "turnover": number(turnover.iloc[-1]) if len(turnover) else None,
        "relative_turnover": number(turnover.iloc[-1] / base) if base and base > 0 else None,
        "price_basis": "pct_chg compounded return index; execution uses raw price"})


def build(universe, evidence, metrics, benchmark, previous=None, leaders=None):
    day = universe["as_of"]
    if evidence.get("as_of") != day or universe.get("summary", {}).get("status") != "SUCCESS":
        raise ValueError("候选或证据日期/状态不完整")
    previous = previous or {}
    # Compare by opportunity, not by symbol: a new cycle must not inherit the old cycle's rank memory.
    before = {r["opportunity_id"]: r for r in previous.get("records", [])} if previous.get("as_of", day) < day and previous.get("version") == VERSION else {}
    records = []
    for item in universe["records"]:
        symbol = item["symbol"]
        core = evidence.get("records", {}).get(symbol, {})
        vector = core.get("state_vector", {})
        f = metrics.get(symbol, {"status": "DATA_WAIT"})
        theme = item.get("belonging_theme") or "未归属"
        cycle = item.get("cycle_id") or item.get("preparation_id") or "unanchored"
        oid = store.digest([symbol, cycle])[:24]
        risk = item.get("value_tier") == "RISK_WATCH" or item.get("sell_structure_alert") or any(
            r.get("code") != "MARKET_CAUTION" for r in vector.get("risk_overlays", []))
        window = core.get("review_window", {})
        mode_ready = vector.get("setup_kind") == "FIRST_PULLBACK"
        # Unknown kinds remain visible and cannot inherit breakout execution permission.
        blockers = []
        if f["status"] != "READY" or item.get("price_date") != day or not core:
            blockers.append("DATA_WAIT")
        if risk:
            blockers.append("STRUCTURE_RISK")
        if not universe.get("market", {}).get("trend_review_allowed") or universe.get("market", {}).get("data_status") != "FRESH":
            blockers.append("MARKET_WAIT")
        if not mode_ready or not window.get("active") or not item.get("initial_review"):
            blockers.append("WAIT_PULLBACK_CONFIRMATION")
        if item.get("tracking_only"):
            blockers.append("TRACKING_ONLY")
        state = ("DATA_WAIT" if "DATA_WAIT" in blockers else "RISK_WATCH" if risk else
                 "MARKET_WAIT" if "MARKET_WAIT" in blockers else "WAIT_SETUP" if blockers else "PLAN_REVIEW")
        row = {"opportunity_id": oid, "symbol": symbol, "name": item.get("name", symbol), "theme": theme,
               "route_ids": item.get("route_ids", ""), "alpha_scope": alpha_scope(item.get("route_ids", "")),
               "cycle_id": cycle, "setup_kind": vector.get("setup_kind", "UNKNOWN"),
               "mode": "TREND_PULLBACK", "modes": {"TREND_PULLBACK": state, "STRENGTH_CONFIRM": "OBSERVATION_ONLY",
                                                     "OVERNIGHT": "OBSERVATION_ONLY"},
               "state": state, "blockers": blockers, "features": f, "strength_score": None, "momentum_score": None,
               "role": "UNCONFIRMED", "leadership_confirmed": False, "role_evidence": (leaders or {}).get(symbol, []),
               "legacy_rank": item.get("value_rank"), "tracking_only": bool(item.get("tracking_only")),
               "price_date": item.get("price_date"), "window": window, "next_condition": core.get("next_condition", "等待数据"),
               "evidence": core, "source_record": item, "setup_id": vector.get("setup_id", ""),
               "previous_rank": before.get(oid, {}).get("strength_rank"),
               "previous_opportunity_seen": before.get(oid, {}).get("price_date"),
               "rank_change": None, "ranking_version": VERSION, "paper_plan_eligible": state == "PLAN_REVIEW",
               "live_orders_enabled": False}
        records.append(row)
    valid = [r for r in records if r["features"]["status"] == "READY"]
    for row in valid:
        f = row["features"]
        peers = [r for r in valid if r["theme"] == row["theme"] and r["symbol"] != row["symbol"]]
        # Exclude the target's own contribution; this is candidate-cohort strength, not a theme index.
        f["peer_count"] = len(peers)
        for n in (5, 20):
            market = number(benchmark.get(f"return{n}"))
            peer = float(pd.Series([p["features"][f"return{n}"] for p in peers]).median()) if len(peers) >= RULES["min_peer_count"] and row["theme"] != "未归属" else None
            f[f"market_excess{n}"] = f[f"return{n}"] - market if market is not None else None
            f[f"peer_excess{n}"] = f[f"return{n}"] - peer if peer is not None else None
        f["relative_momentum"] = ((f["market_excess5"] + f["market_excess20"]) / 2
                                  if f["market_excess5"] is not None and f["market_excess20"] is not None else None)
        if f["relative_momentum"] is None:
            row["blockers"].append("BENCHMARK_MISSING")
            row["paper_plan_eligible"] = False
            row["state"] = row["modes"]["TREND_PULLBACK"] = "DATA_WAIT"
    eligible = [r for r in valid if r["features"]["relative_momentum"] is not None]
    # Compare within board volatility groups; small groups are disclosed, never silently called calibrated.
    groups = {}
    for r in eligible:
        board = "STAR" if r["symbol"].startswith("sh_68") else "CHINEXT" if r["symbol"].startswith("sz_30") else "MAIN"
        groups.setdefault(board, []).append(r)
    for board, rows in groups.items():
        frame = pd.DataFrame([r["features"] for r in rows])
        ranks = frame[["relative_momentum", "drawdown20", "efficiency10"]].rank(pct=True, method="average")
        for i, r in enumerate(rows):
            # The percentiles that made the score are saved with it, so an explanation reads them, not a re-derivation.
            r["score_percentiles"] = {k: round(float(ranks[k].iloc[i]), 6) for k in ("relative_momentum", "drawdown20", "efficiency10")}
            r["strength_score"] = round(100 * (ranks.relative_momentum.iloc[i] * .5 + ranks.drawdown20.iloc[i] * .25 + ranks.efficiency10.iloc[i] * .25), 2)
            r["momentum_score"] = round(100 * ranks.relative_momentum.iloc[i], 2)
            r["comparison_group"] = {"board": board, "count": len(rows), "limited": len(rows) < 20}
    records.sort(key=lambda r: (r["strength_score"] is None, -(r["strength_score"] or 0), r["symbol"]))
    for n, r in enumerate(sorted(eligible, key=lambda r: (-(r["momentum_score"] or 0), r["symbol"])), 1):
        r["momentum_rank"] = n
    theme_counts = Counter()
    for n, r in enumerate(records, 1):
        r["strength_rank"] = n if r["strength_score"] is not None else None
        if r["strength_rank"] is not None:
            theme_counts[r["theme"]] += 1
            r["theme_strength_rank"] = theme_counts[r["theme"]]
            if r["role_evidence"]:
                r["role"] = "DAILY_LEADER_CANDIDATE"
            elif theme_counts[r["theme"]] <= 3:
                r["role"] = "RELATIVE_STRENGTH_FRONT"
            if r["previous_rank"] is not None:
                r["rank_change"] = r["previous_rank"] - n
        r["rank_reason"] = {"components": {k: r["features"].get(k) for k in ("relative_momentum", "drawdown20", "efficiency10")},
                            "current_blockers": r["blockers"], "previous_date": previous.get("as_of"),
                            "cohort_changed": set(before) != {x["opportunity_id"] for x in records}}
    order = {"PLAN_REVIEW": 0, "MARKET_WAIT": 1, "WAIT_SETUP": 2, "RISK_WATCH": 3, "DATA_WAIT": 4}
    participation = sorted(records, key=lambda r: (order[r["state"]], r["tracking_only"], r["strength_score"] is None,
        abs(number(r["source_record"].get("route_bias_atr")) or 0) > 2,
        -(r["strength_score"] or 0), r["symbol"]))
    for n, row in enumerate(participation, 1):
        row["participation_rank"] = n
    return {"as_of": day, "version": VERSION, "rules": RULES, "market": universe.get("market", {}),
            "observation_basis": "DATED_SOURCE_RECONSTRUCTION_NOT_PROOF_OF_LIVE_DISCOVERY",
            "scope": "已分析候选与续踪合并域，不是全市场龙头普查；日级角色线索不证明因果带动。",
            "summary": {"total": len(records), "rankable": len(eligible), "states": dict(Counter(r["state"] for r in records)),
                        "paper_plan_eligible": sum(r["paper_plan_eligible"] for r in records)},
            "records": records, "live_orders_enabled": False}
