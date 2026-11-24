"""Audit continuous preparation coverage on a frozen endpoint-winner cohort.

This reverse diagnostic is not a false-positive rate or a tradable success rate.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import pandas as pd

from mainwave_engine import route_evidence as RE

ROOT = Path("results/mainwave_engine")


def continuous_before_attempt(row: dict) -> bool:
    return bool(row.get("preparation_valid") and row.get("preparation_origin") == "PRE_ATTEMPT"
                and row.get("preparation_parent_id") == row.get("breakout_id")
                and row.get("preparation_first_seen") and row.get("breakout_attempt_at")
                and row["preparation_first_seen"] < row["breakout_attempt_at"])


def worker(item):
    symbol, targets = item
    path = Path("data") / f"{symbol}_daily.csv"
    raw = pd.read_csv(path)
    frame = RE.build(raw[pd.to_datetime(raw.date).le("2026-09-09")], symbol)
    lookup = dict(zip(frame.date.dt.strftime("%Y-%m-%d"), frame.to_dict("records")))
    rows = []
    for target in targets:
        row = lookup[target["date"]]
        continuous = continuous_before_attempt(row)
        rows.append({**target, "continuous_preparation": continuous,
                     "ever_and_continuous": target["covered_before_attempt"] and continuous,
                     **{k: row[k] for k in RE.PT.FIELDS},
                     "breakout_id": row["breakout_id"], "breakout_attempt_at": row["breakout_attempt_at"],
                     "actionable_coverage": None,
                     "boundary": "Price preparation continuity only; theme/entry/execution not inferred"})
    return rows, {"symbol": symbol, "price_hash": hashlib.sha256(path.read_bytes()).hexdigest()}


def run(output: Path, workers: int):
    source = ROOT / "route_followup_20260910/r2_coverage.csv.gz"
    frame = pd.read_csv(source)
    frame = frame[frame.variant.eq("strict_early") & frame.r20.ge(20)
                  & (frame.year.ge(2025) | frame.outcome_at.le("2024-12-31"))]
    rows, inputs = [], []
    jobs = [(s, g.to_dict("records")) for s, g in frame.groupby("symbol")]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (records, fingerprint) in enumerate(pool.map(worker, jobs, chunksize=4), 1):
            rows.extend(records)
            inputs.append(fingerprint)
            if i % 500 == 0 or i == len(jobs):
                print(f"[preparation_audit] {i}/{len(jobs)} stocks", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(output / "r2_continuous_coverage.csv.gz", index=False)
    result["period"] = result.year.map(lambda y: "TRAIN_2023_24" if y <= 2024 else f"CHECK_{y}")
    summary = result.groupby("period").agg(targets=("symbol", "size"), ever_warned=("covered_before_attempt", "sum"),
                                           continuous_tracks=("continuous_preparation", "sum"),
                                           ever_and_continuous=("ever_and_continuous", "sum"))
    summary.to_csv(output / "r2_continuous_coverage_summary.csv")
    (output / "r2_continuity_manifest.json").write_text(json.dumps({
        "targets": len(frame), "cohort": "Same 3384 broad accepted endpoint-up20 targets, not all preparations",
        "source_hash": hashlib.sha256(source.read_bytes()).hexdigest(), "inputs": inputs,
        "track_version": RE.PT.VERSION, "track_hash": hashlib.sha256(Path(RE.PT.__file__).read_bytes()).hexdigest(),
        "orders_enabled": False, "actionable_coverage": None,
        "limitations": ["Track TTL/floor are explicit hypotheses, not fitted optima",
                        "Ever warning and continuous identities are different policies; their intersection is reported",
                        "Reverse winners-only cohort cannot estimate precision or false positives"]}, indent=2), encoding="utf-8")
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "route_diagnostic_20260910")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run(args.output, args.workers)
