"""Fixed allocation grid and causal post-pullback confirmations; research only."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from mainwave_engine import route_evidence as RE
from mainwave_engine import entry_policy as EP
from research import route_clock as CLOCK, route_followup as RF, scaled_entry_replay as S
from research.entry_exit_review import ROOT, write

VERSION = "confirmed_add_v1_20260911"
OUTPUT = ROOT / "confirmed_add_20260911"
INITIALS = (.2, .3, .5, .7)
SETTINGS = {k: S.SETTINGS[k] for k in ("atr_05_10", "pct_03_06")}
MODES = ("STABILIZE", "ABSORB", "RECLAIM")


def weights(initial):
    return (.3, .4) if initial == .3 else ((1 - initial) * 3 / 7, (1 - initial) * 4 / 7)


def schedule(a, signal, trace, calendar, setting, mode, *, cap_initial=True):
    """Only past/current closes create next-session orders; execution checks the open."""
    orders, audit = {}, {"touches": {}, "confirmations": 0, "terminal": "WINDOW_END"}
    if not trace:
        audit["terminal"] = "NO_ENTRY"
        return orders, audit
    entry = signal + 1
    price, atr = float(a["index_open"][entry]), float(a["atr20"][signal])
    distance = atr if setting[0] == "ATR" else price
    levels = [price - setting[k] * distance for k in (1, 2)]
    structure = a["breakout_invalid_index"][signal]
    chain = a["breakout_id"][signal]
    if not math.isfinite(structure):
        audit["terminal"] = "NO_STRUCTURE"
        return orders, audit
    logs = {r["date"]: r for r in trace}
    tests = [None, None]
    entry_clock = bisect_right(calendar, a["day"][entry]) - 1
    for j in range(entry, len(a["day"])):
        day = a["day"][j]
        clock = bisect_right(calendar, day) - 1
        age = clock - entry_clock
        if age > 7:
            break
        gap = CLOCK.classify_gap(a["day"][j - 1], day, calendar, a.get("suspended_days", []))
        if gap["status"] not in {"NEXT_EXCHANGE_SESSION", "VERIFIED_SUSPENSION"}:
            audit["terminal"] = "DATA_GAP"
            break
        row = logs.get(day)
        if row is None:
            break
        if row["fills"] or row.get("pending_after", False):
            audit["terminal"] = "SELL_CANCELLED"
            break
        if (a["breakout_id"][j] != chain or a["breakout_status"][j] not in {"ACCEPTED", "ADVANCING"}
                or a["index_close"][j] < structure):
            audit["terminal"] = "STRUCTURE_INVALID"
            break
        if age == 0:
            continue
        close, low = a["index_close"][j], a["index_low"][j]
        recovery, strong, contracted = EP.recovery_evidence(a, j)
        for k, level in enumerate(levels):
            test = tests[k]
            if test is not None and j > test["index"]:
                qualifies = recovery and (mode == "STABILIZE" or test["contracted"])
                qualifies = qualifies and (mode != "RECLAIM" or strong)
                if qualifies and age < 7 and clock + 1 < len(calendar):
                    # All price bounds are frozen at the confirmation close.
                    ceiling = EP.order_ceiling(close, a["ma20"][j], atr, price if cap_initial else None)
                    order = {"tranche": k + 2, "touch_date": a["day"][test["index"]],
                             "confirm_date": day, "mode": mode, "contracted_before": bool(test["contracted"]),
                             "min_price": float(structure),
                             "max_price": float(ceiling)}
                    orders.setdefault(calendar[clock + 1], []).append(order)
                    audit["confirmations"] += 1
            if test is None and low <= level and level > row["stop_before"]:
                tests[k] = {"index": j, "contracted": bool(contracted)}
                audit["touches"][str(k + 2)] = day
            elif test is not None and low <= level and contracted:
                # Same-bar contraction never confirms a previously uncontracted test.
                test["contracted"] = True
    return orders, audit


def worker(job):
    symbol, sources, frozen, calendar, as_of, price_hash, output, cap_initial = job
    path = Path(frozen) / "data" / f"{symbol}_daily.csv"
    if hashlib.sha256(path.read_bytes()).hexdigest() != price_hash:
        raise ValueError(f"Frozen price mismatch: {symbol}")
    raw = pd.read_csv(path)
    frame = RF.enrich(RE.build(raw[pd.to_datetime(raw.date).le(as_of)], symbol))
    a = RF.arrays(frame)
    lookup = {d: i for i, d in enumerate(a["day"])}
    rows = []
    for source in sources:
        i, day = lookup[source["date"]], source["date"]
        trace = []
        baseline = RF.simulate(a, i, symbol, "stage36_20", trace=trace, calendar=calendar)
        if not a["accepted_bias2"][i] or baseline["status"] != source["status"] or (baseline["status"] == "CLOSED" and
                abs(baseline["net"] - source["net"]) > 1e-8):
            raise ValueError(f"Baseline mismatch: {symbol} {day}")
        variants, plans = {}, {}
        for name, values in SETTINGS.items():
            for mode in MODES:
                orders, audit = schedule(a, i, trace, calendar, values, mode, cap_initial=cap_initial)
                plans[f"{name}_{mode}"] = {"orders": orders, "audit": audit}
        for initial in INITIALS:
            prefix = str(round(initial * 100))
            only = S.replay(a, i, symbol, baseline, trace, calendar, initial=initial)
            variants[f"{prefix}_ONLY"] = only
            if baseline["status"] == "CLOSED" and abs(only["net"] - initial * baseline["net"]) > 1e-8:
                raise ValueError(f"Allocation cash mismatch: {symbol} {day}")
            for name, values in SETTINGS.items():
                for ordering in S.PATHS:
                    variants[f"{prefix}_{name}_TOUCH_{ordering}"] = S.replay(
                        a, i, symbol, baseline, trace, calendar, values, ordering, initial,
                        add_weights=weights(initial))
                for mode in MODES:
                    variants[f"{prefix}_{name}_{mode}"] = S.replay(
                        a, i, symbol, baseline, trace, calendar, values, initial=initial,
                        add_weights=weights(initial), add_schedule=plans[f"{name}_{mode}"]["orders"])
        for name, result in variants.items():
            initial = int(name.split("_")[0]) / 100
            prefix = str(round(initial * 100))
            threshold = next((s for s in SETTINGS if s in name), "NONE")
            mode = name.removeprefix(f"{prefix}_{threshold}_") if threshold != "NONE" else "ONLY"
            only = variants[f"{prefix}_ONLY"]
            touch = variants.get(f"{prefix}_{threshold}_TOUCH_LOW_FIRST", only)
            paired = baseline["status"] == "CLOSED" and result["status"] == "CLOSED"
            touch_paired = paired and touch["status"] == "CLOSED"
            rows.append({"symbol": symbol, "signal": day, "variant": name, "initial": initial,
                         "threshold": threshold, "mode": mode, "common_mature": i + 40 < len(frame),
                         "baseline_status": baseline["status"], "status": result["status"], "paired": paired,
                         "baseline_net": baseline.get("net"), "net": result.get("net"),
                         "delta": result["net"] - baseline["net"] if paired else None,
                         "delta_vs_initial": result["net"] - only["net"] if paired else None,
                         "delta_vs_touch": result["net"] - touch["net"] if touch_paired else None,
                         "invested": result["invested"], "buy_count": len(result["buys"]),
                         "deferred": result["deferred"], "exit_date": result.get("exit_date"),
                         "rejected_orders": len(result.get("add_rejections", []))})
        write(Path(output) / f"{symbol}_{day}.json", {"version": VERSION, "symbol": symbol, "signal": day,
              "baseline": baseline, "baseline_trace": trace, "plans": plans, "variants": variants})
    return rows


def summarize(group, budget):
    stats = S.summarize(group.rename(columns={"delta_vs_initial": "delta_vs_only30"}), budget)
    stats["net_vs_initial_yuan"] = stats.pop("net_vs_only30_yuan")
    paired = group[group.paired]
    touch = paired.delta_vs_touch.dropna()
    initial = paired.delta_vs_initial
    return {**stats, "net_vs_touch_yuan": float(touch.sum() * budget / 100), "touch_comparable": len(touch),
            "improved_vs_touch": int(touch.gt(1e-8).sum()), "worse_vs_touch": int(touch.lt(-1e-8).sum()),
            "improved_vs_initial": int(initial.gt(1e-8).sum()), "worse_vs_initial": int(initial.lt(-1e-8).sum()),
            "net_p10": float(paired.net.quantile(.1)) if len(paired) else None,
            "net_mean": float(paired.net.mean()) if len(paired) else None,
            "add_cases": int(paired.buy_count.gt(1).sum())}


def run(output=OUTPUT, workers=4, budget=10000., cap_initial=True):
    if not math.isfinite(budget) or budget <= 0:
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
             tuple(manifest["frozen_calendar"]), manifest["as_of"], hashes[s], str(output), cap_initial)
            for s, g in source.groupby("symbol")]
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, rows in enumerate(pool.map(worker, jobs, chunksize=2), 1):
            records.extend(rows)
            if n % 100 == 0 or n == len(jobs):
                print(f"[confirmed_add] {n}/{len(jobs)} stocks, records={len(records)}", flush=True)
    f = pd.DataFrame(records)
    f.to_csv(output / "case_comparison.csv", index=False, encoding="utf-8-sig")
    summaries = []
    scopes = (("all_closed", f), ("common_40_sessions", f[f.common_mature]),
              ("mature_jan_apr", f[f.common_mature & f.signal.lt("2026-05-01")]),
              ("mature_may_jul", f[f.common_mature & f.signal.ge("2026-05-01")]))
    for scope, cohort in scopes:
        for (name, initial, threshold, mode), g in cohort.groupby(["variant", "initial", "threshold", "mode"]):
            summaries.append({"scope": scope, "variant": name, "initial": initial, "threshold": threshold,
                              "mode": mode, **summarize(g, budget)})
    report = {"version": VERSION, "total_cases": len(source), "as_of": manifest["as_of"], "cap_initial": cap_initial,
              "per_case_budget_yuan": budget, "summary": summaries}
    write(output / "summary.json", report)
    write(output / "manifest.json", {"version": VERSION, "as_of": manifest["as_of"], "signals": len(source),
          "initials": INITIALS, "settings": SETTINGS, "modes": MODES, "per_case_budget_yuan": budget,
          "cap_initial": cap_initial, "cap_ablation_after_initial_check": not cap_initial,
          "rules": {"weights": "initial then remaining budget in 3:4 ratio; no borrowing/reinvestment",
                    "exits": "frozen original stage36_20 price signals, independent T+1 lots, original costs",
                    "window": "add fills only within exchange sessions 1..7 after first entry; first day excluded",
                    "touch": "fixed levels below initial fill; confirmation must be on a later bar than touch",
                    "common_confirmation": "same accepted/advancing breakout ID, above frozen invalidation, eligible, no supply warning",
                    "STABILIZE": "higher/equal low versus yesterday, close rises, close location >=0.5",
                    "ABSORB": "STABILIZE plus earlier pullback-at-level amount <=0.85 of prior-five median amount",
                    "RECLAIM": "ABSORB plus close > previous high and >=MA20, location>=0.6, amount_ratio>=1",
                    "execution": "confirmation close schedules ONLY next exchange session open; unavailable/overpriced open rejected",
                    "open_guard": "above frozen structure and current stop; <=min(confirmation close+0.5ATR0, MA20+2ATR0); optional P0+0.5ATR0 ceiling controlled by cap_initial",
                    "repeat": "later fresh confirmation closes may retry rejected orders inside window; each tranche fills once",
                    "scope": "independent equal-budget fractional cases, not account returns; no lot rounding/minimum commissions"},
          "unseen_oos": False, "production_changed": False, "orders_enabled": False,
          "source_hash": hashlib.sha256((frozen / "r1_exit_replay.csv.gz").read_bytes()).hexdigest(),
          "code_hashes": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                          for p in (__file__, S.__file__, RF.__file__, CLOCK.__file__, RE.__file__)}})
    compact = [r for r in summaries if r["scope"] == "common_40_sessions" and r["threshold"] != "pct_03_06"]
    print(pd.DataFrame(compact)[["variant", "comparable", "add_cases", "improved", "worse", "new_pnl_yuan",
                                "net_vs_initial_yuan", "net_vs_touch_yuan"]].round(2).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--budget", type=float, default=10000.)
    parser.add_argument("--allow-above-initial", action="store_true")
    args = parser.parse_args()
    run(args.output, args.workers, args.budget, not args.allow_above_initial)
