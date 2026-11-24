"""Route 1 entry/exit attribution and route 2 causal preparation funnel.

Research only. Fixed hypotheses, 2023-24 selection with label purging, then
2025/2026 chronological checks. No historical theme reconstruction or orders.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import re
from time import perf_counter

import numpy as np
import pandas as pd

from mainwave_engine import data as D
from mainwave_engine import route_evidence as RE
from mainwave_engine import pullback_plan as PP
from research import route_clock as CLOCK


VERSION = "route_followup_v2_20260910"
ENTRIES = ("next_open", "gap_guard", "advance", "pullback")
EXITS = ("hold20", "legacy20", "rolling20", "activated20", "stage36_20",
         "stage24_20", "activated60", "stage36_60", "structure36_20")
FILTERS = ("base", "trend", "not_extended60", "relative_strength", "trend_rs")


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    c = frame.return_index
    ma60 = c.rolling(60).mean()
    frame["ma20"] = c.rolling(20).mean()
    frame["trend"] = c.gt(ma60) & ma60.gt(ma60.shift(5))
    frame["not_extended60"] = frame.ret60.le(.40)
    frame["low10"] = frame.index_low.rolling(10).min()
    frame["drawdown60"] = (c / frame.index_high.rolling(60).max() - 1) * 100
    contractions = frame[["turnover_contract", "amount_contract", "range_contract"]].le(.85).sum(axis=1)
    frame["two_of_three"] = (frame.retained & contractions.ge(2) & c.ge(frame.structure_floor)
                             & frame.ma20.ge(frame.ma20.shift(5)) & frame.bias_atr.between(-1, 2))
    return frame


def arrays(frame: pd.DataFrame) -> dict:
    out = {c: frame[c].to_numpy() for c in frame.columns}
    out["day"] = frame.date.dt.strftime("%Y-%m-%d").to_numpy()
    out["suspended_days"] = frame.attrs.get("suspended_days", [])
    return out


def entry_signal(a: dict, i: int, mode: str) -> tuple[int | None, str]:
    if mode in {"next_open", "gap_guard"}:
        return i, "SIGNAL"
    n, cid = len(a["day"]), a["breakout_id"][i]
    atr, boundary = a["atr20"][i], a["breakout_boundary_index"][i]
    peak, advanced, pullback = a["index_close"][i], False, False
    for j in range(i + 1, min(n, i + 21)):
        if a["breakout_id"][j] != cid or a["breakout_status"][j] not in {"ACCEPTED", "ADVANCING"}:
            return None, "INVALID_BEFORE_ENTRY"
        if not a["eligible"][j]:
            return None, "INELIGIBLE_BEFORE_ENTRY"
        if mode == "advance" and a["breakout_status"][j] == "ADVANCING" and not a["supply_warning"][j]:
            return j, "ADVANCE"
        if (mode == "pullback" and pullback and a["index_close"][j] > a["index_high"][j - 1]
                and a["close_location"][j] >= .6 and a["amount_ratio"][j] >= 1
                and not a["supply_warning"][j]):
            return j, "RECLAIM"
        # A pullback must follow a previously observed advance, never the same bar.
        if (advanced and a["index_low"][j] <= peak - .5 * atr
                and a["index_low"][j] <= max(boundary, a["ma20"][j]) + atr
                and a["index_close"][j] >= boundary - .5 * atr
                and a["amount_ratio"][j] <= .85 and a["close_location"][j] >= .45):
            pullback = True
        advanced = advanced or a["index_close"][j] >= a["index_close"][i] + atr
        peak = max(peak, a["index_high"][j])
    return None, ("WAIT_CENSORED" if i + 20 >= n else
                  "RAN_WITHOUT_PULLBACK" if mode == "pullback" and advanced and not pullback else "WINDOW_EXPIRED")


def simulate(a: dict, signal: int, symbol: str, mode: str, gap_guard: bool = False,
             *, trace: list | None = None, calendar: tuple | None = None, entry_guard: bool = False,
             exit_policy: str | None = None, conservative_limits: dict | None = None) -> dict:
    if exit_policy not in {None, "TREND", "RISK", "WIDE_RISK", "CLOSE_RISK", "CLOSE_CORE",
                           "REVIEW_ONLY", "EXTEND_RISK", "EXTEND_CLOSE"}:
        raise ValueError("Unknown exit policy")
    horizon = 60 if mode.endswith("60") else 20
    entry, end = signal + 1, signal + horizon
    if entry >= len(a["day"]):
        return {"status": "CENSORED"}
    calendar = CLOCK.sessions() if calendar is None else calendar
    gap = CLOCK.classify_gap(a["day"][signal], a["day"][entry], calendar, a.get("suspended_days", []))
    if gap["status"] != "NEXT_EXCHANGE_SESSION":
        return {"status": "ENTRY_" + gap["status"], "gap": gap}
    if a["isST"][entry] != 0:
        return {"status": "ENTRY_UNAVAILABLE"}
    limit = D.limit_pct(symbol) / 100
    upper = (conservative_limits or {}).get((symbol,a["day"][entry]), (None,None))[0]
    if upper is None:
        upper = math.floor(a["preclose"][entry] * (1 + limit) * 100 + .5) / 100
    if a["open"][entry] >= upper - .001:
        return {"status": "ENTRY_LIMIT_UP"}
    price, atr = a["index_open"][entry], a["atr20"][signal]
    if not (math.isfinite(price) and math.isfinite(atr) and price > 0 and atr > 0):
        return {"status": "INVALID_PRICE_OR_ATR"}
    if gap_guard and (price - a["index_close"][signal]) > atr:
        return {"status": "ENTRY_GAP"}
    if entry_guard:
        rejected = PP.opening_check(a, signal, price)
        if rejected:
            return {"status": rejected}
    phased = mode == "phase36_20"
    protect_profit = phased or mode == "profit36_20"
    structure = a["plan_structure"][signal] if phased and "plan_structure" in a else None
    if phased and (structure is None or not math.isfinite(structure)):
        return {"status": "INVALID_STRUCTURE_BOUNDARY"}
    initial_stop = price - 2 * atr
    if exit_policy == "WIDE_RISK" and a["exit_strong"][signal]:
        initial_stop = price - 3 * atr
    if mode == "structure36_20":
        if not math.isfinite(a["breakout_invalid_index"][signal]):
            return {"status": "INVALID_STRUCTURE_BOUNDARY"}
        initial_stop = max(initial_stop, a["breakout_invalid_index"][signal])
        if price <= initial_stop:
            return {"status": "ENTRY_BELOW_STRUCTURE"}
    stop, remaining, cash, pending = initial_stop, 1., 0., False
    pending_exit = ""
    pending_reduce, reduced = False, False
    close_trail = exit_policy in {"CLOSE_RISK", "CLOSE_CORE", "EXTEND_CLOSE"}
    soft_stop, soft_breaches, soft_required = initial_stop, 0, 1
    peak, trough, partials = price, price, 0
    targets = ([(price + 3 * atr, .5), (price + 6 * atr, .25)] if mode.startswith(("stage36", "structure36", "phase36", "profit36")) else
               [(price + 2 * atr, .5), (price + 4 * atr, .25)] if mode.startswith("stage24") else [])
    activated = False
    if exit_policy == "CLOSE_CORE" and a["exit_strong"][signal]:
        targets = [(price + 3 * atr, .25), (price + 6 * atr, .25)]
    last = len(a["day"]) if conservative_limits is not None else min(end + 1, len(a["day"]))
    for j in range(entry, last):
        gap = CLOCK.classify_gap(a["day"][j - 1], a["day"][j], calendar, a.get("suspended_days", []))
        if gap["status"] in {"UNEXPLAINED_SESSION_GAP", "CALENDAR_UNVERIFIED"}:
            return {"status": "HOLDING_DATA_UNVERIFIED", "last_known_date": a["day"][j - 1],
                    "remaining": remaining, "stop_index": stop, "gap": gap}
        op, hi, lo, cl = (a[k][j] for k in ("index_open", "index_high", "index_low", "index_close"))
        day_log = {"date": a["day"][j], "index_open": float(op), "index_high": float(hi),
                   "index_low": float(lo), "index_close": float(cl), "stop_before": float(stop),
                   "initial_stop": float(initial_stop), "atr0": float(atr), "entry_index": float(price),
                   "activated_before": bool(activated), "sellable": j > entry, "pending_before": bool(pending),
                   "calendar_status": gap["status"], "remaining_before": remaining, "fills": []}
        prior_peak = peak
        peak, trough = max(peak, hi), min(trough, lo)
        lower = (conservative_limits or {}).get((symbol,a["day"][j]), (None,None))[1]
        if lower is None:
            lower = math.floor(a["preclose"][j] * (1 - limit) * 100 + .5) / 100
        locked = a["high"][j] <= lower + .001
        if conservative_limits is not None:
            locked = locked or a["open"][j] <= lower + .001
        reason, fill = "", np.nan
        if j > entry and not locked and pending_reduce and not pending and not pending_exit and op > stop:
            weight = remaining / 2
            day_log["fills"].append({"reason": "RISK_REDUCTION", "weight": weight,
                                     "price_index": float(op), "execution": "NEXT_OPEN"})
            cash += weight * op / price
            remaining -= weight
            pending_reduce, reduced = False, True
        if j == entry:
            pending = mode != "hold20" and lo <= stop
        elif pending and not locked:
            reason, fill = "DEFERRED_STOP", op
        elif pending_exit and not locked:
            reason, fill = pending_exit, op
        elif mode != "hold20" and lo <= stop:
            if locked:
                pending = True
            else:
                reason, fill = "INITIAL_STOP" if stop == initial_stop else "TRAIL_STOP", min(op, stop)
        if reason:
            event = {"reason": reason, "weight": remaining, "price_index": float(fill)}
            if exit_policy and reason == pending_exit:
                event["execution"] = "NEXT_OPEN"
            day_log["fills"].append(event)
            cash += remaining * fill / price
            remaining = 0
        elif j > entry and not locked:
            for target, weight in list(targets):
                if exit_policy and remaining <= 1e-10:
                    break
                if hi >= target:
                    original_weight = weight
                    if exit_policy:
                        weight = min(weight, remaining)
                    day_log["fills"].append({"reason": "PARTIAL_TARGET", "weight": weight, "price_index": float(max(op, target))})
                    cash += weight * max(op, target) / price
                    remaining -= weight
                    targets.remove((target, original_weight))
                    partials += 1
                    if exit_policy and remaining <= 1e-10:
                        remaining, reason, fill = 0., "TARGET_COMPLETED", max(op, target)
        if remaining and j == end:
            if locked:
                day_log.update(locked=True, remaining_after=remaining, stop_next=float(stop), pending_after=pending)
                if trace is not None:
                    trace.append(day_log)
                if conservative_limits is None:
                    return {"status": "EXIT_UNRESOLVED", "remaining": remaining, "stop_index": stop}
                pending_exit = "HORIZON_DEFERRED"
                if trace is not None:
                    day_log["pending_after"] = True
                continue
            day_log["fills"].append({"reason": "HORIZON", "weight": remaining, "price_index": float(cl)})
            cash += remaining * cl / price
            remaining, reason, fill = 0., "HORIZON", cl
        if not remaining:
            day_log.update(locked=bool(locked), remaining_after=remaining, stop_next=float(stop), pending_after=bool(pending))
            if trace is not None:
                trace.append(day_log)
            return {"status": "CLOSED", "net": (cash - 1) * 100 - .30,
                    "gross": (cash - 1) * 100, "cost_pct": .30,
                    "gross_R": (cash - 1) * price / (price - initial_stop),
                    "net_R": ((cash - 1) - .003) * price / (price - initial_stop),
                    "entry_date": a["day"][entry], "exit_date": a["day"][j],
                    "entry_price": float(a["open"][entry]), "exit_index": float(fill),
                    "initial_risk_pct": (price - initial_stop) / price * 100,
                    "held": j - entry + 1, "reason": reason, "partials": partials,
                    "mfe": (peak / price - 1) * 100, "mae": (trough / price - 1) * 100}
        # Trailing rules become executable only on the next day, including activation.
        activated = activated or peak >= price + 2 * atr
        if close_trail and activated and exit_policy != "EXTEND_CLOSE":
            if cl < soft_stop:
                if soft_breaches == 0:
                    soft_required = 2 if any(a["exit_strong"][max(entry, j - 2):j]) else 1
                soft_breaches += 1
                if soft_breaches >= soft_required:
                    pending_exit = pending_exit or "SOFT_CLOSE_EXIT"
            else:
                soft_breaches = 0
        if j > entry and mode == "legacy20":
            stop = max(stop, float(np.min(a["index_low"][max(entry, j - 9):j + 1])))
        elif mode == "rolling20" or (mode != "hold20" and activated and mode != "legacy20"):
            trail = "low20" if exit_policy not in {None, "REVIEW_ONLY"} and a["exit_strong"][j] else "low10"
            if close_trail:
                if exit_policy != "EXTEND_CLOSE":
                    soft_stop = max(soft_stop, float(a[trail][j]))
            else:
                stop = max(stop, float(a[trail][j]))
        if phased and cl < structure:
            pending_exit = "STRUCTURE_CLOSE_EXIT"
        elif protect_profit and PP.profit_warning(a, j, entry, price, atr, prior_peak):
            pending_exit = pending_exit or "MATURE_WEAK_CLOSE_EXIT"
        if exit_policy:
            from research.adaptive_exit_policy import warning
            action, evidence = warning(a, j, entry, price, atr, prior_peak) if exit_policy not in {"TREND", "REVIEW_ONLY"} else ("", {})
            if exit_policy == "EXTEND_CLOSE":
                from mainwave_engine.exit_policy import close_review
                review = close_review(a, j, entry, price, atr, prior_peak, soft_stop=soft_stop,
                                      breaches=soft_breaches, required=soft_required, pending_exit=pending_exit,
                                      pending_reduce=pending_reduce, reduced=reduced)
                soft_stop, soft_breaches, soft_required = (review[k] for k in ("soft_stop", "breaches", "required"))
                pending_exit, pending_reduce = review["pending_exit"], review["pending_reduce"]
                action, evidence = review["action"], review["evidence"]
                if review["extension"] is not None:
                    day_log["extension_review"] = {"extend": review["extension"], "decided_at_close": a["day"][j]}
            elif action == "EXIT_STRUCTURE":
                pending_exit = pending_exit or action
                pending_reduce = False
            elif action and not reduced and not pending_exit:
                pending_reduce = True
            if j == signal + 20 and exit_policy in {"REVIEW_ONLY", "EXTEND_RISK"}:
                extend = (exit_policy != "REVIEW_ONLY" and a["exit_strong"][j]
                          and cl >= price + 2 * atr and not pending_exit and not pending_reduce)
                if not extend:
                    pending_exit = pending_exit or "TIME_REVIEW_EXIT"
                day_log["extension_review"] = {"extend": bool(extend), "decided_at_close": a["day"][j]}
            day_log.update(exit_policy=exit_policy, strong_close=bool(a["exit_strong"][j]),
                           market_known=bool(a["market_known"][j]), close_action=action, exit_evidence=evidence,
                           pending_exit_next=pending_exit, pending_reduce_next=pending_reduce)
            if close_trail:
                day_log.update(soft_stop_next=float(soft_stop), soft_breaches=soft_breaches,
                               soft_required=soft_required, hard_stop_fixed=True)
        if protect_profit:
            day_log.update(pending_exit_next=pending_exit, profit_phase="MATURE" if prior_peak >= price + 3 * atr else "INITIAL")
        day_log.update(locked=bool(locked), remaining_after=remaining, stop_next=float(stop),
                       activated_after=bool(activated), pending_after=bool(pending))
        if trace is not None:
            trace.append(day_log)
    return {"status": "CENSORED"}


def preparation_outcome(a: dict, i: int) -> dict:
    n, end = len(a["day"]), i + 20
    result = {"mature": end < n, "attempt": False, "accepted": False, "advanced": False,
              "already_accepted": a["breakout_status"][i] in {"ACCEPTED", "ADVANCING"},
              "attempt_delay": None, "accept_delay": None, "break_before_attempt": False}
    first_id = None
    for j in range(i + 1, min(n, end + 1)):
        if first_id is None and a["index_close"][j] < a["structure_floor"][i] - .5 * a["atr20"][i]:
            result["break_before_attempt"] = True
            break
        if first_id is None and a["breakout_event"][j] == "ATTEMPT":
            first_id = a["breakout_id"][j]
            result.update(attempt=True, attempt_delay=j - i)
        if first_id and a["breakout_id"][j] == first_id:
            if a["breakout_event"][j] == "ACCEPTED":
                result.update(accepted=True, accept_delay=j - i)
            if a["breakout_event"][j] == "ADVANCING":
                result["advanced"] = True
            if a["breakout_status"][j] in {"REJECTED", "FAILED", "EXPIRED"}:
                break
    if end < n:
        result.update(r20=(a["index_close"][end] / a["index_close"][i] - 1) * 100,
                      mfe20=(np.max(a["index_high"][i + 1:end + 1]) / a["index_close"][i] - 1) * 100,
                      outcome_at=a["day"][end])
    return result


def worker(args: tuple[str, str, str]) -> dict:
    filename, start, end = args
    path = Path(filename)
    symbol = path.stem.removesuffix("_daily")
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw = raw[pd.to_datetime(raw.date).le(end)]
    frame = RE.build(raw, symbol)
    if frame.empty:
        return {"symbol": symbol, "status": "SHORT_HISTORY", "trades": [], "prep": [], "events": [],
                "coverage": [], "variants": []}
    frame = enrich(frame)
    a = arrays(frame)
    trades, prep, events, coverage, variants = [], [], [], [], []
    valid = frame.eligible & frame.date.ge(start)
    fields = ("trend", "not_extended60", "ret20", "ret60", "bias_atr", "amount_ratio", "drawdown60",
              "prepared_before_attempt", "probe_at", "breakout_status", "turnover_contract",
              "amount_contract", "range_contract", "supply_warning")
    for i in frame.index[valid & (frame.accepted_bias2 | frame.prepared_watch)]:
        rec = {"symbol": symbol, "date": a["day"][i], "year": int(a["day"][i][:4]),
               **{key: a[key][i].item() if isinstance(a[key][i], np.generic) else a[key][i] for key in fields}}
        if frame.prepared_watch.iloc[i]:
            prep.append({**rec, **preparation_outcome(a, i)})
        if not frame.accepted_bias2.iloc[i]:
            continue
        ev = {**rec, "breakout_id": a["breakout_id"][i]}
        if i + 20 < len(frame):
            ev.update(r20=(a["index_close"][i + 20] / a["index_close"][i] - 1) * 100,
                      mfe20=(np.max(a["index_high"][i + 1:i + 21]) / a["index_close"][i] - 1) * 100,
                      outcome_at=a["day"][i + 20])
        events.append(ev)
        for entry in ENTRIES:
            trigger, reason = entry_signal(a, i, entry)
            for exit_mode in EXITS:
                base = {"symbol": symbol, "date": a["day"][i], "year": int(a["day"][i][:4]),
                        "entry": entry, "exit": exit_mode, "entry_reason": reason,
                        "evaluation_at": a["day"][i + (80 if exit_mode.endswith("60") else 40)]
                            if i + (80 if exit_mode.endswith("60") else 40) < len(frame) else None}
                if trigger is None:
                    trades.append({**base, "status": reason})
                    continue
                base.update(trigger_date=a["day"][trigger], delay=trigger - i)
                trades.append({**base, **simulate(a, trigger, symbol, exit_mode, entry == "gap_guard")})
    for name, mask in [("strict_early", frame.prepared), ("two_of_three_early", frame.two_of_three)]:
        early = mask & frame.eligible & ~frame.breakout_status.isin(["ACCEPTED", "ADVANCING"])
        # First observation per demand-anchor, not every flip back into a daily threshold.
        seen = set()
        for i in frame.index[valid & early]:
            anchor = a["probe_at"][i]
            if anchor in seen:
                continue
            seen.add(anchor)
            variants.append({"symbol": symbol, "date": a["day"][i], "year": int(a["day"][i][:4]),
                             "variant": name, "probe_at": anchor, **preparation_outcome(a, i)})
        prior = early.shift(1, fill_value=False).rolling(30).max().eq(1)
        at_attempt = prior.astype(float).where(frame.attempt).ffill().eq(1)
        for i in frame.index[valid & frame.accepted]:
            rec = {"symbol": symbol, "date": a["day"][i], "year": int(a["day"][i][:4]),
                   "variant": name, "covered_before_attempt": bool(at_attempt.iloc[i]),
                   "demand_confirmed": bool(frame.accepted_demand.iloc[i])}
            if i + 20 < len(frame):
                rec.update(r20=(a["index_close"][i + 20] / a["index_close"][i] - 1) * 100,
                           outcome_at=a["day"][i + 20])
            coverage.append(rec)
    return {"symbol": symbol, "status": "OK", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "trades": trades, "prep": prep, "events": events, "coverage": coverage, "variants": variants}


def periods(frame: pd.DataFrame, outcome: str):
    train = frame[frame.year.le(2024) & frame[outcome].notna() & frame[outcome].le("2024-12-31")]
    return [("TRAIN_2023_24", train), ("CHECK_2025", frame[frame.year.eq(2025)]),
            ("CHECK_2026", frame[frame.year.eq(2026)])]


def trade_stats(g: pd.DataFrame) -> dict:
    c = g[g.status.eq("CLOSED")]
    net = c.net.dropna()
    gains, losses = net[net > 0].sum(), -net[net < 0].sum()
    unresolved = int(g.status.isin(["CENSORED", "WAIT_CENSORED", "EXIT_UNRESOLVED", "HOLDING_DATA_UNVERIFIED"]).sum())
    return {"signals": len(g), "closed": len(c), "profit": int(net.gt(0).sum()),
            "win_pct": net.gt(0).mean() * 100, "mean": net.mean(), "median": net.median(),
            "p10": net.quantile(.1), "profit_factor": gains / losses if losses else np.nan,
            "held_median": c.held.median(), "early_stop_pct": ((c.held <= 3) & c.reason.str.contains("STOP")).mean() * 100,
            "partials_pct": c.partials.gt(0).mean() * 100,
            "closed_net_sum_per_signal": net.sum() / len(g) if len(g) and not unresolved else np.nan,
            "unresolved": unresolved}


def summarize(output: Path) -> None:
    ev = pd.read_csv(output / "r1_events.csv.gz")
    trades = pd.read_csv(output / "r1_trades.csv.gz")
    prep = pd.read_csv(output / "r2_funnel.csv.gz")
    benchmark = pd.read_csv("data/000300_SH_index.csv")
    benchmark["date"] = pd.to_datetime(benchmark.trade_date.astype(str)).dt.strftime("%Y-%m-%d")
    benchmark = benchmark.sort_values("date").drop_duplicates("date", keep="last")
    benchmark["market_ret20"] = benchmark.close.pct_change(20)
    ev = ev.merge(benchmark[["date", "market_ret20"]], on="date", how="left", validate="many_to_one")
    ev["relative_strength"] = ev.ret20.gt(ev.market_ret20)
    ev["trend_rs"] = ev.trend & ev.relative_strength
    ev["base"] = True
    joined = trades.merge(ev[["symbol", "date", *FILTERS]], on=["symbol", "date"], validate="many_to_one")
    summaries, signal_stats = [], []
    for filter_name in FILTERS:
        subset = joined[joined[filter_name]]
        # Use the same mature anchor cohort within each holding horizon (20-day wait).
        # This avoids counting quick losing exits while omitting still-open winners.
        for label, period in periods(subset[subset.evaluation_at.notna()], "evaluation_at"):
            for (entry, exit_mode), g in period.groupby(["entry", "exit"]):
                summaries.append({"period": label, "filter": filter_name, "entry": entry,
                                  "exit": exit_mode, **trade_stats(g)})
        for label, period in periods(ev[ev[filter_name]], "outcome_at"):
            m = period.dropna(subset=["r20"])
            signal_stats.append({"period": label, "filter": filter_name, "signals": len(period), "mature": len(m),
                                 "positive_pct": m.r20.gt(0).mean() * 100, "mean": m.r20.mean(),
                                 "down10_pct": m.r20.le(-10).mean() * 100,
                                 "touch20_pct": m.mfe20.ge(20).mean() * 100})
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "r1_summary.csv", index=False)
    pd.DataFrame(signal_stats).to_csv(output / "r1_opportunity.csv", index=False)
    paired = []
    direct = trades[trades.entry.eq("next_open") & trades.exit.str.endswith("20") & trades.evaluation_at.notna()]
    for label, period in periods(direct, "evaluation_at"):
        pivot = period.pivot(index=["symbol", "date"], columns="exit", values="net")
        for mode in ("rolling20", "activated20", "stage24_20", "stage36_20", "structure36_20"):
            matched = pivot[["legacy20", mode]].dropna()
            delta = matched[mode] - matched.legacy20
            paired.append({"period": label, "exit": mode, "paired_trades": len(matched),
                           "mean_delta_vs_legacy": delta.mean(), "improved_pct": delta.gt(0).mean() * 100,
                           "legacy_losses_rescued": int((matched.legacy20.le(0) & matched[mode].gt(0)).sum()),
                           "legacy_profits_lost": int((matched.legacy20.gt(0) & matched[mode].le(0)).sum())})
    pd.DataFrame(paired).to_csv(output / "r1_exit_pairs.csv", index=False)
    risk_rows = []
    for label, period in periods(direct, "evaluation_at"):
        for mode, g in period[period.status.eq("CLOSED")].groupby("exit"):
            for low, high in [(0, 5), (5, 10), (10, float("inf"))]:
                bucket = g[g.initial_risk_pct.gt(low) & g.initial_risk_pct.le(high)]
                risk_rows.append({"period": label, "exit": mode, "initial_risk_bucket": f"({low},{high}]",
                                  "closed": len(bucket), "mean": bucket.net.mean(),
                                  "mean_R": (bucket.net / bucket.initial_risk_pct).mean(),
                                  "win_pct": bucket.net.gt(0).mean() * 100})
    pd.DataFrame(risk_rows).to_csv(output / "r1_risk_distance.csv", index=False)
    funnel = []
    for label, period in periods(prep, "outcome_at"):
        for group_name, g in [("all", period), ("pre_ignition", period[~period.already_accepted]),
                              ("already_accepted", period[period.already_accepted]),
                              ("pre_ignition_trend", period[~period.already_accepted & period.trend]),
                              ("pre_ignition_not_trend", period[~period.already_accepted & ~period.trend])]:
            m = g[g.mature]
            funnel.append({"period": label, "group": group_name, "observations": len(g), "mature": len(m),
                           "unique_probe_episodes": m.drop_duplicates(["symbol", "probe_at"]).shape[0],
                           **{key: int(m[key].sum()) for key in ["attempt", "accepted", "advanced", "break_before_attempt"]},
                           "accepted_pct": m.accepted.mean() * 100, "advanced_pct": m.advanced.mean() * 100,
                           "accept_delay_median": m.loc[m.accepted, "accept_delay"].median(),
                           "positive20_pct": m.r20.gt(0).mean() * 100})
    pd.DataFrame(funnel).to_csv(output / "r2_summary.csv", index=False)
    variants = pd.read_csv(output / "r2_variants.csv.gz")
    coverage = pd.read_csv(output / "r2_coverage.csv.gz")
    comparisons = []
    for label, period in periods(variants, "outcome_at"):
        for name, g in period.groupby("variant"):
            m = g[g.mature]
            comparisons.append({"period": label, "variant": name, "observations": len(g), "mature": len(m),
                                "accepted": int(m.accepted.sum()), "advanced": int(m.advanced.sum()),
                                "accepted_pct": m.accepted.mean() * 100, "advanced_pct": m.advanced.mean() * 100,
                                "break_before_attempt_pct": m.break_before_attempt.mean() * 100})
    pd.DataFrame(comparisons).to_csv(output / "r2_variants_summary.csv", index=False)
    coverage_rows = []
    for label, period in periods(coverage, "outcome_at"):
        for name, g in period.groupby("variant"):
            m = g.dropna(subset=["r20"])
            up20 = m[m.r20.ge(20)]
            coverage_rows.append({"period": label, "variant": name, "accepted_events": len(m),
                                  "covered": int(m.covered_before_attempt.sum()),
                                  "coverage_pct": m.covered_before_attempt.mean() * 100,
                                  "final_up20_events": len(up20), "up20_covered": int(up20.covered_before_attempt.sum()),
                                  "up20_coverage_pct": up20.covered_before_attempt.mean() * 100})
    pd.DataFrame(coverage_rows).to_csv(output / "r2_coverage_summary.csv", index=False)
    # Choose a single combination using only purged training outcomes; never best-per-trade.
    train = summary[summary.period.eq("TRAIN_2023_24") & summary.closed.ge(500) & summary.exit.str.endswith("20")
                    & ~summary.exit.eq("structure36_20")]
    ranked = train.sort_values(["mean", "profit_factor", "closed"], ascending=False)
    selected = ranked.iloc[0][["filter", "entry", "exit"]].to_dict() if len(ranked) else None
    selection = {"version": VERSION, "selected_train_only": selected,
                 "objective": "20-session arms: maximum mean net, >=500 closed training trades; ties profit factor and count",
                 "status": "RESEARCH_NOT_PRODUCTION", "orders_enabled": False,
                 "exploratory_after_first_check": ["structure36_20", "R2 two_of_three", "R3 accepted_or_reclaim"],
                 "limitations": ["All filter/entry/exit combinations disclosed; multiple-testing risk remains",
                    "2025/26 are previously studied historical checks, not unseen OOS",
                    "Current local universe, independent overlapping events, no portfolio return",
                    "Daily bars, T+1, standard board limits, observed exchange-calendar gaps classified, costs 0.30% scenario",
                    "R2 preparation is a price proxy and does not establish capital ownership or theme confirmation",
                    "R1 filters are evaluated on original acceptance date, not moved to future winning days",
                    "20-session arms use common 40-session-complete anchors; 60-session arms use 80; not directly paired across horizons",
                    "20/60 active-stock-session exits; unresolved trades excluded from net statistics"]}
    if selected:
        chosen = summary[np.logical_and.reduce([summary[k].eq(v) for k, v in selected.items()])]
        selection["selected_results"] = chosen.to_dict("records")
        selection["positive_mean_all_periods"] = bool(len(chosen) == 3 and chosen["mean"].gt(0).all())
        keep = np.logical_and.reduce([joined[k].eq(v) if k != "filter" else joined[v] for k, v in selected.items()])
        cases = joined[keep].sort_values(["date", "symbol"])
        cases.to_csv(output / "selected_all_cases.csv", index=False)
    (output / "selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(selection, indent=2, ensure_ascii=False), flush=True)


def run(start: str, end: str, output: Path, workers: int, limit: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path("data").glob("*_daily.csv") if re.fullmatch(
        r"(?:sh_(?:60|68)|sz_(?:00|30))\d{4}_daily.csv", p.name))
    if limit:
        paths = [paths[i] for i in np.linspace(0, len(paths) - 1, min(limit, len(paths)), dtype=int)]
    records, prep, events, manifest, coverage, variants = [], [], [], [], [], []
    started = perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(worker, [(str(p), start, end) for p in paths], chunksize=4), 1):
            records.extend(result.pop("trades"))
            prep.extend(result.pop("prep"))
            events.extend(result.pop("events"))
            coverage.extend(result.pop("coverage"))
            variants.extend(result.pop("variants"))
            manifest.append(result)
            if i % 200 == 0 or i == len(paths):
                print(f"[followup] {i}/{len(paths)} R1={len(events)} R2={len(prep)} seconds={perf_counter()-started:.1f}", flush=True)
    pd.DataFrame(records).to_csv(output / "r1_trades.csv.gz", index=False)
    pd.DataFrame(events).to_csv(output / "r1_events.csv.gz", index=False)
    pd.DataFrame(prep).to_csv(output / "r2_funnel.csv.gz", index=False)
    pd.DataFrame(variants).to_csv(output / "r2_variants.csv.gz", index=False)
    pd.DataFrame(coverage).to_csv(output / "r2_coverage.csv.gz", index=False)
    (output / "manifest.json").write_text(json.dumps({"start": start, "end": end, "version": VERSION,
        "symbols": manifest, "seconds": perf_counter() - started,
        "code_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        [Path(__file__), Path(RE.__file__), Path(RE.BE.__file__), Path(RE.PE.__file__)]}}, indent=2), encoding="utf-8")
    summarize(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-09-09")
    parser.add_argument("--output", type=Path, default=Path("results/mainwave_engine/route_followup_20260910"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    if args.summary_only:
        summarize(args.output)
    else:
        run(args.start, args.end, args.output, args.workers, args.limit)
