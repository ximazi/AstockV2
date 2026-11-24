"""All-local-stock event study and conservative independent paper-trade replays.

Current surviving local universe, NOT point-in-time theme reconstruction. No
portfolio CAGR: overlapping trades do not imply unlimited investable capital.
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


def paper_path(frame: pd.DataFrame, i: int, symbol: str, mode: str) -> dict:
    """Next open, no entry-day selling, stop-first ambiguous bars; costs are scenarios."""
    entry = i + 1
    expiry = i + 20
    if expiry >= len(frame):
        return {"status": "CENSORED"}
    rows = frame.iloc[entry:expiry + 1].to_dict("records")
    first = rows[0]
    gap_days = (pd.Timestamp(first["date"]) - pd.Timestamp(frame.iloc[i].date)).days
    if gap_days > 7 or float(first.get("isST", 1)) != 0:
        return {"status": "ENTRY_UNAVAILABLE"}
    limit = D.limit_pct(symbol) / 100
    upper = math.floor(float(first["preclose"]) * (1 + limit) * 100 + 0.5) / 100
    if first["open"] >= upper - 0.001:
        return {"status": "ENTRY_LIMIT_UP"}
    price = float(first["index_open"])
    atr = float(frame.iloc[i].atr20)
    stop = price - 2 * atr
    remaining, cash, pending = 1., 0., False
    targets = [(price + 3 * atr, 0.5), (price + 6 * atr, 0.25)] if mode == "staged" else []
    exit_reason = "HORIZON"
    for j, row in enumerate(rows):
        if j == 0:
            pending = mode != "hold" and row["index_low"] <= stop
            continue
        lower = math.floor(float(row["preclose"]) * (1 - limit) * 100 + 0.5) / 100
        locked_down = row["high"] <= lower + 0.001
        if not locked_down and pending:
            cash += remaining * row["index_open"] / price
            remaining, exit_reason = 0., "DEFERRED_EXIT"
            break
        if mode != "hold" and row["index_low"] <= stop:
            if locked_down:
                pending = True
                continue
            cash += remaining * min(row["index_open"], stop) / price
            remaining, exit_reason = 0., "STOP_OR_TRAIL"
            break
        if not locked_down:
            for target, weight in list(targets):
                if row["index_high"] >= target:
                    cash += weight * max(row["index_open"], target) / price
                    remaining -= weight
                    targets.remove((target, weight))
        if mode != "hold":
            # Today's low can affect only the next session's trailing boundary.
            upto = entry + j
            stop = max(stop, float(frame.index_low.iloc[max(entry, upto - 9):upto + 1].min()))
        if entry + j == expiry:
            if locked_down:
                return {"status": "EXIT_UNRESOLVED"}
            cash += remaining * row["index_close"] / price
            remaining = 0.
    if remaining > 0:
        return {"status": "EXIT_UNRESOLVED"}
    return {"status": "CLOSED", "net_pct": (cash - 1) * 100 - 0.30,
            "exit_reason": exit_reason}


def worker(args: tuple[str, str, str]) -> dict:
    filename, start, end = args
    path = Path(filename)
    symbol = path.stem.removesuffix("_daily")
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig")
        raw["date"] = pd.to_datetime(raw.date)
        raw = raw[raw.date.le(end)]
        frame = RE.build(raw, symbol)
        if frame.empty:
            return {"symbol": symbol, "status": "SHORT_HISTORY", "rows": []}
        candidates = frame.eligible & frame.date.ge(start) & frame[list(RE.POLICIES)].any(axis=1)
        records = []
        for i in frame.index[candidates]:
            row = frame.iloc[i]
            rec = {"symbol": symbol, "date": str(row.date.date()), "year": row.date.year,
                   "bias_atr": row.bias_atr, "prior20_pct": row.ret20 * 100,
                   "prepared_before_attempt": bool(row.prepared_before_attempt),
                   "supply_warning": bool(row.supply_warning),
                   **{p: bool(row[p]) for p in RE.POLICIES}}
            for n in (5, 10, 20, 40):
                if i + n < len(frame):
                    future = frame.iloc[i + 1:i + n + 1]
                    rec[f"r{n}"] = (frame.iloc[i + n].return_index / row.return_index - 1) * 100
                    rec[f"mfe{n}"] = (future.index_high.max() / row.return_index - 1) * 100
                    rec[f"mae{n}"] = (future.index_low.min() / row.return_index - 1) * 100
                    rec[f"outcome{n}_at"] = str(frame.iloc[i + n].date.date())
            for mode in ("hold", "trend", "staged"):
                trade = paper_path(frame, int(i), symbol, mode)
                rec[f"{mode}_status"] = trade["status"]
                rec[f"{mode}_net"] = trade.get("net_pct")
            records.append(rec)
        return {"symbol": symbol, "status": "OK", "sessions": len(frame),
                "first": str(frame.iloc[0].date.date()), "last": str(frame.iloc[-1].date.date()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": records}
    except (ValueError, KeyError, OSError, pd.errors.ParserError) as exc:
        return {"symbol": symbol, "status": "ERROR", "error": str(exc), "rows": []}


def statistics(frame: pd.DataFrame) -> dict:
    mature = frame.dropna(subset=["r20"])
    result = {"events": len(frame), "mature20": len(mature), "symbols": frame.symbol.nunique(),
              "positive20_pct": float(mature.r20.gt(0).mean() * 100),
              "down10_pct": float(mature.r20.le(-10).mean() * 100),
              "r20_median": float(mature.r20.median()), "r20_mean": float(mature.r20.mean()),
              "mfe20_median": float(mature.mfe20.median()), "mae20_median": float(mature.mae20.median())}
    for mode in ("hold", "trend", "staged"):
        net = frame[f"{mode}_net"].dropna()
        result.update({f"{mode}_closed": len(net), f"{mode}_mean": float(net.mean()),
                       f"{mode}_median": float(net.median()), f"{mode}_win_pct": float(net.gt(0).mean() * 100)})
    return result


def training_sample(frame: pd.DataFrame) -> pd.DataFrame:
    # Purge labels whose outcome would not be known at the parameter freeze date.
    return frame[frame.year.isin([2023, 2024]) & frame.outcome20_at.notna()
                 & frame.outcome20_at.le("2024-12-31")]


def report(output: Path) -> dict:
    frame = pd.read_csv(output / "events.csv.gz")
    summaries = []
    for year in sorted(frame.year.unique()):
        for policy in RE.POLICIES:
            selected = frame[frame.year.eq(year) & frame[policy]]
            summaries.append({"year": int(year), "policy": policy, **statistics(selected)})
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv", index=False)
    # Selection is confined to 2023-24; 2025-26 never enters the selector.
    train = training_sample(frame)
    choices = []
    for policy in ("accepted_bias2", "accepted_bias3"):
        stats = statistics(train[train[policy]])
        choices.append({"policy": policy, **stats})
    eligible = [c for c in choices if c["mature20"] >= 500]
    selected = min(eligible, key=lambda x: (x["down10_pct"], -x["positive20_pct"])) if eligible else None
    result = {"version": RE.VERSION, "training_years": [2023, 2024],
              "training_outcome_cutoff": "2024-12-31",
              "historical_check_years": [2025, 2026], "selection_objective": "min down10 then max positive20; >=500 mature events",
              "choices": choices, "selected_policy": selected["policy"] if selected else None,
              "status": "HISTORICAL_PRICE_STUDY_NOT_THEME_OOS",
              "limitations": [
                  "Current local surviving stock universe, not point-in-time listings; no unseen-OOS claim",
                  "R1 price confirmation and R2 price preparation proxies only; no synthetic historical theme edges",
                  "Outcomes anchored separately to each policy's signal day; acceptance delay changes the clock",
                  "Overlapping independent trades, no portfolio capital/correlation allocation or CAGR",
                  "20-session expiry; next-open entries; T+1; conservative stop-first; 0.30% round-trip cost scenario",
                  "Entry gaps over 7 calendar days skipped, including long holidays; no intraday queue or depth model",
                  "Standard board limits via existing helper; ST excluded; 120-session warmup excludes early IPOs",
                  "Locked exits at study horizon unresolved, not marked as successful fills",
                  "In-sample selector is a risk/coverage comparison, not proof of profitability or causal capital flows",
              ]}
    (output / "selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    columns = ["year", "policy", "events", "mature20", "positive20_pct", "down10_pct", "r20_median",
               "mfe20_median", "trend_mean", "staged_mean"]
    lines = ["# Route price evidence: chronological historical study", "",
             f"Selected using 2023-24 only: {result['selected_policy']}", "",
             *[f"- {s}" for s in result["limitations"]], "",
             summary[columns].to_markdown(index=False, floatfmt=".2f")]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary[columns].to_string(index=False), flush=True)
    return result


def run(start: str, end: str, output: Path, workers: int, limit: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path("data").glob("*_daily.csv")
                   if re.fullmatch(r"(?:sh_(?:60|68)|sz_(?:00|30))\d{4}_daily.csv", p.name))
    if limit:
        paths = [paths[i] for i in np.linspace(0, len(paths) - 1, min(limit, len(paths)), dtype=int)]
    started = perf_counter()
    records, manifest = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(worker, [(str(p), start, end) for p in paths], chunksize=4), 1):
            records.extend(result.pop("rows"))
            manifest.append(result)
            if i % 100 == 0 or i == len(paths):
                print(f"[route_study] {i}/{len(paths)} events={len(records)} seconds={perf_counter()-started:.1f}", flush=True)
    (output / "manifest.json").write_text(json.dumps({
        "start": start, "as_of": end, "version": RE.VERSION, "symbols": manifest,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          (Path(__file__), Path(RE.__file__), Path(RE.BE.__file__), Path(RE.PE.__file__))},
        "seconds": perf_counter() - started}, indent=2), encoding="utf-8")
    pd.DataFrame(records).to_csv(output / "events.csv.gz", index=False)
    report(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-09-09")
    parser.add_argument("--output", type=Path, default=Path("results/mainwave_engine/route_validation_20260910"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    run(args.start, args.end, args.output, args.workers, args.limit)
