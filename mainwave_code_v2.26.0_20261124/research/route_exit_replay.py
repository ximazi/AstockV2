"""Fixed-entry, three-exit calendar/semantics replay. No parameter search."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from mainwave_engine import route_evidence as RE
from research.route_followup import arrays, enrich, periods, simulate, trade_stats

MODES = ("legacy20", "stage36_20", "structure36_20")
ROOT = Path("results/mainwave_engine")


def worker(item):
    symbol, dates = item
    path = Path("data") / f"{symbol}_daily.csv"
    raw = pd.read_csv(path)
    frame = enrich(RE.build(raw[pd.to_datetime(raw.date).le("2026-09-09")], symbol))
    a, records = arrays(frame), []
    lookup = {day: i for i, day in enumerate(a["day"])}
    for day in dates:
        i = lookup[day]
        if not frame.accepted_bias2.iloc[i]:
            raise ValueError(f"Original evidence changed: {symbol} {day}")
        for mode in MODES:
            result = simulate(a, i, symbol, mode)
            records.append({"symbol": symbol, "date": day, "year": int(day[:4]), "entry": "next_open", "exit": mode,
                            "evaluation_at": a["day"][i + 40] if i + 40 < len(frame) else None, **result})
    return records, {"symbol": symbol, "price_hash": hashlib.sha256(path.read_bytes()).hexdigest()}


def run(output: Path, workers: int):
    output.mkdir(parents=True, exist_ok=True)
    original = ROOT / "route_followup_20260910"
    events = pd.read_csv(original / "r1_events.csv.gz")
    jobs = [(s, list(g.date)) for s, g in events.groupby("symbol")]
    records, inputs = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (rows, fingerprint) in enumerate(pool.map(worker, jobs, chunksize=4), 1):
            records.extend(rows)
            inputs.append(fingerprint)
            if i % 500 == 0 or i == len(jobs):
                print(f"[exit_replay] {i}/{len(jobs)} stocks, {len(records)} branches", flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output / "r1_exit_replay.csv.gz", index=False)
    rows = []
    for label, period in periods(frame[frame.evaluation_at.notna()], "evaluation_at"):
        for mode, group in period.groupby("exit"):
            rows.append({"period": label, "exit": mode, **trade_stats(group)})
    pd.DataFrame(rows).to_csv(output / "r1_exit_replay_summary.csv", index=False)
    old = pd.read_csv(original / "r1_trades.csv.gz")
    old = old[old.entry.eq("next_open") & old.exit.isin(MODES)]
    compare = frame.merge(old[["symbol", "date", "exit", "status", "net"]], on=["symbol", "date", "exit"],
                          suffixes=("_new", "_old"), validate="one_to_one")
    compare["net_delta"] = compare.net_new - compare.net_old
    compare.groupby(["exit", "status_old", "status_new"]).size().rename("n").reset_index().to_csv(
        output / "calendar_status_bridge.csv", index=False)
    compare.loc[compare.status_new.ne(compare.status_old) | compare.net_delta.abs().gt(1e-8)].to_csv(
        output / "calendar_changed_cases.csv", index=False)
    manifest = {"original_events": len(events), "stocks": len(jobs), "branches": len(frame),
                "modes": MODES, "entry": "next_open", "as_of": "2026-09-09", "orders_enabled": False,
                "unseen_oos": False, "purpose": "CALENDAR_AND_EXECUTION_SEMANTICS_NOT_THRESHOLD_OPTIMIZATION",
                "calendar": "LOCAL_BENCHMARK_SESSION_UNION; missing sessions require explicit suspended records",
                "censoring": "Common 40 active-stock-session anchor cohort; all statuses also retained",
                "known_limitations": ["Independent overlapping events, not portfolio returns", "Flat 0.30pp aggregate cost scenario",
                                      "No new policy selected from these historical checks"],
                "source_sha256": hashlib.sha256((original / "r1_events.csv.gz").read_bytes()).hexdigest(),
                "code_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                                [__file__, RE.__file__, RE.PT.__file__, "research/route_followup.py", "research/route_clock.py"]},
                "inputs": inputs}
    (output / "exit_replay_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "route_diagnostic_20260910")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.output, args.workers)
