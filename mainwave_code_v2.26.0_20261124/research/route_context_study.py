"""Post-stratified diagnostics and actual available dated-chain archive coverage."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mainwave_engine import route_evidence as RE
from mainwave_engine.route_scan import route_ids
from research.route_walkforward import statistics

ROOT = Path("results/mainwave_engine/route_validation_20260910")


def run() -> None:
    events = pd.read_csv(ROOT / "events.csv.gz")
    index = pd.read_csv("data/000300_SH_index.csv")
    index["date"] = pd.to_datetime(index.trade_date.astype(str)).dt.strftime("%Y-%m-%d")
    index = index.sort_values("date").drop_duplicates("date", keep="last")
    ma20, ma60 = index.close.rolling(20).mean(), index.close.rolling(60).mean()
    index["environment"] = np.select([
        index.close.gt(ma60) & ma20.gt(ma60) & ma60.gt(ma60.shift(5)),
        index.close.lt(ma60) & ma20.lt(ma60) & ma60.lt(ma60.shift(5)),
    ], ["UP", "DOWN"], default="MIXED")
    events = events.merge(index[["date", "environment"]], on="date", how="left", validate="many_to_one")
    events["environment"] = events.environment.fillna("UNKNOWN")
    events["board"] = np.select([events.symbol.str.startswith("sh_688"), events.symbol.str.startswith(("sz_300", "sz_301"))],
                                ["STAR", "CHINEXT"], default="MAIN")
    rows = []
    for (year, environment, board), group in events.groupby(["year", "environment", "board"]):
        for policy in ("attempt", "accepted", "accepted_bias2", "accepted_bias3", "prepared_watch"):
            rows.append({"year": year, "environment": environment, "board": board,
                         "policy": policy, **statistics(group[group[policy]])})
    pd.DataFrame(rows).to_csv(ROOT / "context_summary.csv", index=False)
    top_rows = []
    for policy in ("attempt", "accepted", "accepted_bias2", "accepted_bias3"):
        chosen = events[events[policy]].assign(distance=lambda x: x.bias_atr.abs()).sort_values(["date", "distance", "symbol"])
        for k in (3, 5):
            top = chosen.groupby("date", sort=False).head(k)
            for (year, env), group in top.groupby(["year", "environment"]):
                top_rows.append({"year": year, "environment": env, "policy": policy,
                                 "daily_cap": k, **statistics(group)})
    pd.DataFrame(top_rows).to_csv(ROOT / "topk_summary.csv", index=False)

    archives = sorted(Path("results/mainwave_engine").glob("20??????/slow_carrier_diagnostics.csv"))
    raw_archive = []
    for p in archives:
        frame = pd.read_csv(p).fillna("")
        date = f"{p.parent.name[:4]}-{p.parent.name[4:6]}-{p.parent.name[6:]}"
        raw_archive.extend(r for r in frame.to_dict("records") if r.get("as_of") == date and date <= "2026-09-09")
    features = {}
    observed = []
    for chain in raw_archive:
        symbol, date = chain["symbol"], chain["as_of"]
        if symbol not in features:
            raw = pd.read_csv(Path("data") / f"{symbol}_daily.csv")
            features[symbol] = RE.build(raw[pd.to_datetime(raw.date).le("2026-09-09")], symbol)
        frame = features[symbol]
        match = frame.index[frame.date.eq(pd.Timestamp(date))]
        if len(match) != 1:
            continue
        i = match[0]
        row = frame.loc[i].to_dict()
        routes, _ = route_ids({**row, "symbol": symbol, "data_status": "FRESH"}, [chain], date)
        observed.append({"date": date, "symbol": symbol, "theme_id": chain["theme_id"],
                         "in_r3": "R3" in routes, "prepared": bool(row["prepared"]),
                         "accepted_bias2": bool(row["accepted_bias2"]),
                         "old_slow_stage": chain["slow_stage"],
                         "r5": (frame.iloc[i + 5].return_index / row["return_index"] - 1) * 100 if i + 5 < len(frame) else None,
                         "r20": (frame.iloc[i + 20].return_index / row["return_index"] - 1) * 100 if i + 20 < len(frame) else None})
    pd.DataFrame(observed).to_csv(ROOT / "dated_r3_archive.csv", index=False)
    summary = {"price_context_groups": len(rows), "topk_groups": len(top_rows),
               "r3_archive_dates": sorted({r["date"] for r in observed}), "r3_rows": len(observed),
               "r3_qualified_context_rows": sum(r["in_r3"] for r in observed),
               "r3_full20": sum(r["r20"] is not None for r in observed),
               "r3_validation": "INSUFFICIENT_DATED_HISTORY; NOT REPLACED WITH CURRENT_THEME_MEMBERSHIP",
               "notes": "Index environment is a separate causal research stratum, not the production market_context classifier. TopK ranks by absolute ATR bias, no optimization on test years."}
    (ROOT / "context_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
