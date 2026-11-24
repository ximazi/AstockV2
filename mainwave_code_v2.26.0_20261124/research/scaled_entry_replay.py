"""30/30/40 entry-only experiment on the frozen 2026 R1 cohort."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from mainwave_engine import data as D, route_evidence as RE
from research import route_clock as CLOCK, route_followup as RF
from research.entry_exit_review import ROOT, write

VERSION = "scaled_entry_v1_20260911"
OUTPUT = ROOT / "scaled_entry_20260911"
SETTINGS = {"atr_05_10": ("ATR", .5, 1.), "atr_10_20": ("ATR", 1., 2.),
            "pct_03_06": ("PCT", .03, .06)}
LABELS = {"atr_05_10": "回撤0.5/1ATR", "atr_10_20": "回撤1/2ATR", "pct_03_06": "回撤3%/6%",
          "only30": "只买30%，永不加仓"}
PATHS = ("LOW_FIRST", "HIGH_FIRST")
EPS = 1e-10


def replay(a, signal, symbol, baseline, trace, calendar, setting=None, path="LOW_FIRST", initial=.3,
           *, add_weights=(.3, .4), add_schedule=None):
    """Keep original price-based sell signals; replay new lots with T+1 and cash."""
    if not trace:
        return {"status": baseline["status"], "invested": 0., "buys": [], "sells": [], "deferred": 0}
    entry = signal + 1
    price, atr = float(a["index_open"][entry]), float(a["atr20"][signal])
    levels = []
    if setting:
        unit, first, second = setting
        distance = atr if unit == "ATR" else price
        if len(add_weights) != 2 or min(add_weights) < 0 or initial + sum(add_weights) > 1 + EPS:
            raise ValueError("Invalid add allocation")
        levels = [(price - first * distance, add_weights[0]), (price - second * distance, add_weights[1])]
    used, lots, buys, sells = set(), [], [], []
    rejected = []
    mechanical = add_schedule is None
    cash, invested, cancelled, liquidating, deferred = 1., 0., False, False, 0
    day_lookup = {d: i for i, d in enumerate(a["day"])}
    limit = D.limit_pct(symbol) / 100

    def raw_price(j, value):
        return float(value * a["close"][j] / a["index_close"][j])

    def buy(j, value, weight, tranche):
        nonlocal cash, invested
        quantity = weight / value
        lots.append({"day": j, "original": quantity, "remaining": quantity, "pending": 0.})
        cash -= weight
        invested += weight
        buys.append({"date": a["day"][j], "tranche": tranche, "budget_fraction": weight,
                     "price_index": value, "raw_price": raw_price(j, value)})

    def fill_sell(j, lot, quantity, value, reason):
        nonlocal cash
        quantity = min(quantity, lot["remaining"])
        if quantity <= EPS:
            return
        cash += quantity * value
        lot["remaining"] -= quantity
        sells.append({"date": a["day"][j], "quantity_index": quantity, "price_index": value,
                      "raw_price": raw_price(j, value), "reason": reason,
                      "lot_entry_date": a["day"][lot["day"]]})

    def sell_event(j, value, reason, weight=None):
        nonlocal cancelled, liquidating, deferred
        cancelled = True
        liquidating = liquidating or weight is None
        for lot in lots:
            available = max(0., lot["remaining"] - lot["pending"])
            quantity = available if weight is None else min(available, lot["original"] * weight)
            if j > lot["day"]:
                fill_sell(j, lot, quantity, value, reason)
            elif quantity > EPS:
                lot["pending"] += quantity
                deferred += 1

    def outcome(status, j):
        result = {"status": status, "invested": invested, "unused_budget": 1 - invested,
                  "buys": buys, "sells": sells, "deferred": deferred,
                  "entry_date": a["day"][entry], "last_date": a["day"][j],
                  "cancelled_adds": cancelled, "add_rejections": rejected}
        if status == "CLOSED":
            result.update(net=(cash - 1) * 100 - .30 * invested,
                          gross=(cash - 1) * 100, cost_pct=.30 * invested, exit_date=a["day"][j])
        return result

    buy(entry, price, initial, 1)
    logs = {row["date"]: row for row in trace}
    last_original = day_lookup[trace[-1]["date"]]
    last_seen = entry
    for j in range(entry, len(a["day"])):
        if j > last_original and not liquidating:
            break
        gap = CLOCK.classify_gap(a["day"][j - 1], a["day"][j], calendar, a.get("suspended_days", []))
        if gap["status"] in {"UNEXPLAINED_SESSION_GAP", "CALENDAR_UNVERIFIED"}:
            return outcome("HOLDING_DATA_UNVERIFIED", last_seen)
        last_seen = j
        op, hi, lo, cl = (float(a[k][j]) for k in ("index_open", "index_high", "index_low", "index_close"))
        lower = math.floor(a["preclose"][j] * (1 - limit) * 100 + .5) / 100
        upper = math.floor(a["preclose"][j] * (1 + limit) * 100 + .5) / 100
        locked_down = a["high"][j] <= lower + .001
        locked_up = a["low"][j] >= upper - .001
        if not locked_down:
            for lot in lots:
                if lot["pending"] > EPS and j > lot["day"]:
                    fill_sell(j, lot, lot["pending"], op, "TPLUS1_DEFERRED_EXIT")
                    lot["pending"] = 0.
            if liquidating:
                sell_event(j, op, "DEFERRED_FULL_EXIT")
        if all(lot["remaining"] <= EPS for lot in lots):
            return outcome("CLOSED", j)
        if liquidating:
            continue
        row = logs.get(a["day"][j])
        if row is None:
            break
        stop = row["stop_before"]
        age = bisect_right(calendar, a["day"][j]) - bisect_right(calendar, a["day"][entry])
        can_add = 1 <= age <= 7 and not locked_down and not locked_up and a["isST"][j] == 0
        done = set()
        events = []
        horizon = []
        for k, fill in enumerate(row["fills"]):
            reason, value = fill["reason"], fill["price_index"]
            if fill.get("execution") == "NEXT_OPEN":
                sell_event(j, op, reason, fill["weight"] if reason == "RISK_REDUCTION" else None)
            elif reason == "HORIZON":
                horizon.append(fill)
            elif reason == "PARTIAL_TARGET":
                events.append((value, "TARGET", fill["weight"], k))
            elif reason == "DEFERRED_STOP":
                sell_event(j, op, reason)
            else:
                events.append((value, "STOP", None, k))
        # Opening sells take precedence. Gap-down buys are allowed only above the known stop.
        for value, kind, weight, k in events:
            if (kind == "STOP" and op <= value + EPS) or (kind == "TARGET" and op >= value - EPS):
                sell_event(j, op, kind, weight)
                done.add(k)
        if not mechanical:
            for order in add_schedule.get(a["day"][j], []):
                k = order["tranche"] - 2
                if k in used:
                    continue
                reason = ("SELL_CANCELLED" if cancelled else
                          "WINDOW_OR_UNAVAILABLE" if not can_add else
                          "OPEN_LIMIT_UP" if a["open"][j] >= upper - .001 else
                          "OPEN_BELOW_SUPPORT" if op <= max(stop, order["min_price"]) + EPS else
                          "OPEN_TOO_HIGH" if op > order["max_price"] + EPS else "")
                if reason:
                    rejected.append({"date": a["day"][j], "tranche": k + 2, "reason": reason})
                else:
                    buy(j, op, levels[k][1], k + 2)
                    buys[-1]["confirmation"] = order
                    used.add(k)
        if mechanical and can_add and not cancelled and op > stop + EPS:
            for k, (level, weight) in enumerate(levels):
                if k not in used and level > stop + EPS and op <= level + EPS:
                    buy(j, op, weight, k + 2)
                    used.add(k)
        points = (op, lo, hi, cl) if path == "LOW_FIRST" else (op, hi, lo, cl)
        for start, end in zip(points, points[1:]):
            crossing = []
            for value, kind, weight, k in events:
                if k in done:
                    continue
                hit = end <= value < start if kind == "STOP" else start < value <= end
                if hit:
                    crossing.append((value, 0, kind, weight, k))
            if mechanical and can_add and not cancelled and end < start:
                for k, (level, weight) in enumerate(levels):
                    if k not in used and level > stop + EPS and end <= level < start:
                        crossing.append((level, 1, "BUY", weight, k))
            crossing.sort(key=lambda event: ((-event[0] if end < start else event[0]), event[1]))
            for value, _, kind, weight, k in crossing:
                if kind == "BUY":
                    if not cancelled:
                        buy(j, value, weight, k + 2)
                        used.add(k)
                else:
                    sell_event(j, value, kind, weight)
                    done.add(k)
        for fill in horizon:
            sell_event(j, cl, "HORIZON")
        if all(lot["remaining"] <= EPS for lot in lots):
            return outcome("CLOSED", j)
    return outcome("CENSORED" if liquidating else baseline["status"], last_seen)


def worker(job):
    symbol, sources, frozen, calendar, as_of, price_hash, output = job
    path = Path(frozen) / "data" / f"{symbol}_daily.csv"
    if hashlib.sha256(path.read_bytes()).hexdigest() != price_hash:
        raise ValueError(f"Frozen input mismatch: {symbol}")
    raw = pd.read_csv(path)
    frame = RF.enrich(RE.build(raw[pd.to_datetime(raw.date).le(as_of)], symbol))
    a, records = RF.arrays(frame), []
    lookup = {day: j for j, day in enumerate(a["day"])}
    for source in sources:
        i, day = lookup[source["date"]], source["date"]
        trace = []
        baseline = RF.simulate(a, i, symbol, "stage36_20", trace=trace, calendar=calendar)
        if not a["accepted_bias2"][i] or baseline["status"] != source["status"] or (
                baseline["status"] == "CLOSED" and abs(baseline["net"] - source["net"]) > 1e-8):
            raise ValueError(f"Baseline mismatch: {symbol} {day}")
        full = replay(a, i, symbol, baseline, trace, calendar, initial=1.)
        only30 = replay(a, i, symbol, baseline, trace, calendar)
        if baseline["status"] == "CLOSED":
            if abs(full["net"] - baseline["net"]) > 1e-8 or abs(only30["net"] - .3 * baseline["net"]) > 1e-8:
                raise ValueError(f"Cash/exit replay mismatch: {symbol} {day}")
        variants = {"only30": only30}
        for setting, values in SETTINGS.items():
            for ordering in PATHS:
                variants[f"{setting}_{ordering}"] = replay(a, i, symbol, baseline, trace, calendar, values, ordering)
        write(Path(output) / f"{symbol}_{day}.json", {"version": VERSION, "symbol": symbol, "signal": day,
              "baseline": baseline, "baseline_trace": trace, "variants": variants})
        for name, result in variants.items():
            paired = baseline["status"] == "CLOSED" and result["status"] == "CLOSED"
            records.append({"symbol": symbol, "signal": day, "variant": name, "common_mature": i + 40 < len(frame),
                            "baseline_status": baseline["status"], "status": result["status"],
                            "baseline_net": baseline.get("net"), "net": result.get("net"),
                            "paired": paired, "delta": result["net"] - baseline["net"] if paired else None,
                            "delta_vs_only30": result["net"] - only30["net"] if paired else None,
                            "invested": result["invested"], "buy_count": len(result["buys"]),
                            "deferred": result["deferred"], "exit_date": result.get("exit_date")})
    return records


def summarize(group, budget):
    paired = group[group.paired]
    delta = paired.delta
    improved, worse = delta.gt(1e-8), delta.lt(-1e-8)
    return {"total": len(group), "comparable": len(paired), "not_comparable": len(group) - len(paired),
            "improved": int(improved.sum()), "worse": int(worse.sum()),
            "unchanged": int((~improved & ~worse).sum()),
            "improved_pct": float(improved.mean() * 100) if len(paired) else None,
            "baseline_pnl_yuan": float(paired.baseline_net.sum() * budget / 100),
            "new_pnl_yuan": float(paired.net.sum() * budget / 100),
            "improvement_amount_yuan": float(delta[improved].sum() * budget / 100),
            "deterioration_amount_yuan": float(-delta[worse].sum() * budget / 100),
            "net_improvement_yuan": float(delta.sum() * budget / 100),
            "net_vs_only30_yuan": float(paired.delta_vs_only30.sum() * budget / 100),
            "new_win_pct": float(paired.net.gt(0).mean() * 100) if len(paired) else None,
            "profit_to_loss": int((paired.baseline_net.gt(0) & paired.net.lt(0)).sum()),
            "loss_to_profit": int((paired.baseline_net.lt(0) & paired.net.gt(0)).sum()),
            "improved_old_winners": int((paired.baseline_net.gt(0) & improved).sum()),
            "reduced_losses": int((paired.baseline_net.lt(0) & paired.net.le(0) & improved).sum()),
            "mean_invested_pct": float(paired.invested.mean()*100) if len(paired) else None,
            "buy_count": {str(k): int(v) for k, v in paired.buy_count.value_counts().items()},
            "tplus1_cases": int(paired.deferred.gt(0).sum()), "statuses": dict(Counter(group.status))}


def run(output=OUTPUT, workers=4, budget=10000.):
    if budget <= 0 or not math.isfinite(budget):
        raise ValueError("Budget must be positive and finite")
    if (output / "manifest.json").exists():
        raise ValueError("Completed study exists; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    frozen = ROOT / "catalog_source"
    manifest = json.loads((frozen / "exit_replay_manifest.json").read_text(encoding="utf-8"))
    source = pd.read_csv(frozen / "r1_exit_replay.csv.gz")
    source = source[source.year.eq(2026) & source.exit.eq("stage36_20")]
    hashes = {r["symbol"]: r["price_hash"] for r in manifest["inputs"]}
    jobs = [(s, json.loads(g.to_json(orient="records", double_precision=15)), str(frozen),
             tuple(manifest["frozen_calendar"]), manifest["as_of"], hashes[s], str(output)) for s, g in source.groupby("symbol")]
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, rows in enumerate(pool.map(worker, jobs, chunksize=2), 1):
            records.extend(rows)
            if n % 100 == 0 or n == len(jobs):
                print(f"[scaled_entry] {n}/{len(jobs)} stocks, {len(records)//7} signals", flush=True)
    f = pd.DataFrame(records)
    f.to_csv(output / "case_comparison.csv", index=False, encoding="utf-8-sig")
    summary = []
    for scope, cohort in (("all", f), ("common_40_sessions", f[f.common_mature])):
        for name, group in cohort.groupby("variant", sort=False):
            summary.append({"scope": scope, "variant": name, **summarize(group, budget)})
    report = {"version": VERSION, "total_cases": len(source), "per_case_budget_yuan": budget,
              "as_of": manifest["as_of"], "summary": summary,
              "interpretation": "改善包括盈利增加与亏损减少；比例分母为双方已结算案例。金额为等额独立案例盈亏之和，不是账户组合收益。"}
    write(output / "summary.json", report)
    write(output / "manifest.json", {"version": VERSION, "as_of": manifest["as_of"], "signals": len(source),
          "per_case_budget_yuan": budget, "settings": SETTINGS, "paths": PATHS,
          "rules": {"first_buy": "signal next exchange session open, 30% of total planned budget",
                    "add_window": "sessions 1..7 AFTER initial entry, exchange calendar including suspended sessions",
                    "levels": "fixed relative to initial actual fill, signal-day ATR; 30% and 40% budget orders",
                    "exits": "original stage36_20 price signals; do not re-anchor stops/targets to averaged cost",
                    "cancel": "any sell signal cancels all unfilled adds; never buy at/below known stop",
                    "execution": "limit touched along fixed OHLC path, gap improvement at open, independent T+1 lots",
                    "fees": "0.30% round-trip of deployed original budget; unused cash zero return",
                    "intraday": "two OHLC orderings; original conservative stop-before-target day convention retained",
                    "scope": "fractional exposure study; no board-lot rounding, minimum commissions or volume capacity"},
          "unseen_oos": False, "orders_enabled": False,
          "source_hash": hashlib.sha256((frozen / "r1_exit_replay.csv.gz").read_bytes()).hexdigest(),
          "code_hashes": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                          (__file__, RF.__file__, CLOCK.__file__, RE.__file__)}})
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--budget", type=float, default=10000.)
    args = parser.parse_args()
    run(args.output, args.workers, args.budget)
