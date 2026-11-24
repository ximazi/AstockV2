"""Reconcile old research cohorts and audit exits without choosing new thresholds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mainwave_engine import route_evidence as RE
from research.route_followup import arrays, enrich, periods, simulate

ROOT = Path("results/mainwave_engine")


def cohort_bridge(events: pd.DataFrame, coverage: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    key = ["symbol", "date"]
    narrow = events.loc[events.accepted_bias2 & events.r20.ge(20), key + ["r20", "outcome20_at"]].copy()
    broad = coverage.loc[coverage.variant.eq("strict_early") & coverage.r20.ge(20)
                         & (coverage.year.ge(2025) | coverage.outcome_at.le("2024-12-31")),
                         key + ["r20", "outcome_at"]].copy()
    bridge = narrow.merge(broad, on=key, how="outer", suffixes=("_r1", "_r2"), indicator=True, validate="one_to_one")
    bridge["event_key"] = bridge.symbol + ":accepted:" + bridge.date
    bridge["bridge_reason"] = np.select(
        [bridge._merge.eq("both"), bridge._merge.eq("left_only")],
        ["SAME_ACCEPTANCE_IN_BOTH", "R1_UNPURGED_YEAR_BOUNDARY_OR_UNMATCHED"],
        default="BROAD_ACCEPTANCE_NOT_NARROW_DEMAND_BIAS_POLICY")
    matched = bridge[bridge._merge.eq("both")]
    assert np.allclose(matched.r20_r1, matched.r20_r2)
    return bridge, {"narrow_r1_endpoint_up20": len(narrow), "broad_r2_endpoint_up20": len(broad),
                    "shared": len(matched), "r1_only": int(bridge._merge.eq("left_only").sum()),
                    "r2_only": int(bridge._merge.eq("right_only").sum()),
                    "horizon": "20_ACTIVE_STOCK_SESSIONS", "success_is": "ENDPOINT_PRICE_RETURN_GE_20_NOT_EXECUTED_PROFIT",
                    "r1_policy": "accepted_bias2; mature; no year-end purge",
                    "r2_policy": "all eligible accepted; mature; train labels <=2024-12-31; variants deduplicated",
                    "identity": "symbol + acceptance_date; stable legacy join key, not original production hit"}


def exit_migration(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, tails = [], []
    direct = trades[trades.entry.eq("next_open") & trades.exit.str.endswith("20") & trades.evaluation_at.notna()]
    for period, group in periods(direct, "evaluation_at"):
        pivot = group[group.status.eq("CLOSED")].pivot(index=["symbol", "date"], columns="exit", values="net")
        for mode in ("rolling20", "activated20", "stage36_20", "structure36_20"):
            paired = pivot[["legacy20", mode]].dropna().copy()
            old, new = paired.legacy20, paired[mode]
            paired["group"] = np.select([old.le(0) & new.gt(0), old.gt(0) & new.le(0), old.gt(0) & new.gt(0)],
                                         ["LOSS_TO_PROFIT", "PROFIT_TO_LOSS", "BOTH_PROFIT"], default="BOTH_NONPROFIT")
            for label, g in paired.groupby("group"):
                rows.append({"period": period, "exit": mode, "group": label, "n": len(g),
                             "old_sum_pp": g.legacy20.sum(), "new_sum_pp": g[mode].sum(),
                             "contribution_to_mean_delta_pp": (g[mode] - g.legacy20).sum() / len(paired)})
            tails.append({"period": period, "exit": mode, "common_closed": len(paired),
                          "old_closed_unpaired": int(pivot.legacy20.notna().sum() - len(paired)),
                          "mean_net": new.mean(), "mean_net_extra_010_cost": new.mean() - .10,
                          "mean_without_best": new.sort_values().iloc[:-1].mean(),
                          "worst": new.min(), "worst10_sum_pp": new.nsmallest(10).sum(),
                          "best10_sum_pp": new.nlargest(10).sum(), "mean_delta": (new - old).mean()})
    return pd.DataFrame(rows), pd.DataFrame(tails)


def replay_four(cases: pd.DataFrame, output: Path) -> pd.DataFrame:
    results, logs = [], []
    for case in cases[cases.status.eq("CLOSED")].to_dict("records"):
        symbol = case["symbol"]
        raw = pd.read_csv(Path("data") / f"{symbol}_daily.csv")
        raw = raw[pd.to_datetime(raw.date).le("2026-09-09")]
        frame = enrich(RE.build(raw, symbol))
        i = int(frame.index[frame.date.eq(case["matched_date"])][0])
        identity = f"{symbol}:{case['as_of']}:{case['theme']}:{case['matched_date']}"
        trace = []
        result = simulate(arrays(frame), i, symbol, "stage36_20", trace=trace)
        for row in trace:
            assert row["stop_next"] >= row["stop_before"]
            assert not row["fills"] or row["sellable"]
            assert all(f["weight"] > 0 for f in row["fills"])
            logs.append({"case_id": identity, **row})
        if result["status"] == "CLOSED":
            assert abs(sum(f["weight"] for r in trace for f in r["fills"]) - 1) < 1e-8
            assert abs(result["gross"] - result["cost_pct"] - result["net"]) < 1e-8
        results.append({"case_id": identity, "old_net": case["net"], **result,
                        "activated_ever": any(r.get("activated_after", False) for r in trace),
                        "limit_locked_days": sum(r["locked"] for r in trace),
                        "gap_below_stop_on_exit": bool(trace and trace[-1]["index_open"] < trace[-1]["stop_before"]),
                        "price_sha256": hashlib.sha256((Path("data") / f"{symbol}_daily.csv").read_bytes()).hexdigest()})
    (output / "r3_daily_execution_trace.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
    return pd.DataFrame(results)


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "route_followup_20260910"
    files = [ROOT / "route_validation_20260910/events.csv.gz", source / "r2_coverage.csv.gz",
             source / "r1_trades.csv.gz", source / "legacy_proxy_cases.csv"]
    events, coverage, trades, cases = [pd.read_csv(p) for p in files]
    bridge, manifest = cohort_bridge(events, coverage)
    bridge.to_csv(output / "cohort_bridge.csv", index=False, encoding="utf-8-sig")
    migration, tails = exit_migration(trades)
    migration.to_csv(output / "exit_loss_migration.csv", index=False)
    tails.to_csv(output / "exit_tail_sensitivity.csv", index=False)
    four = replay_four(cases, output)
    four.to_csv(output / "r3_four_trade_audit.csv", index=False, encoding="utf-8-sig")
    # Keep closed/open/no-entry operational counts separate from common-maturity comparisons.
    trades.groupby(["entry", "exit", "status"]).size().rename("n").reset_index().to_csv(output / "operational_status_counts.csv", index=False)
    manifest.update(source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                    old_results_recomputed=False, four_trades_replayed_with_new_calendar=True,
                    execution_clock="20_ACTIVE_STOCK_SESSIONS; exchange gaps separately audited",
                    calendar_source="UNION_OF_LOCAL_BENCHMARK_BARS_NOT_OFFICIAL_SECURITY_STATUS_MASTER",
                    orders_enabled=False, unseen_oos=False,
                    caveats=["Historical cohorts use the original research versions and >7-calendar-day entry exclusion",
                             "Four-trade replay uses the new calendar; not a replacement broad backtest",
                             "Overlapping event returns and partial exits are not portfolio returns",
                             "Full-day MFE/MAE are bar extrema, may include movement after a same-day exit",
                             "0.30pp is a fixed aggregate cost scenario, not per-fill broker fee accounting"])
    (output / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(four.to_string(index=False))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "route_diagnostic_20260910")
    run(parser.parse_args().output)
