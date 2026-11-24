"""Inventory legacy theme archives and replay dated-membership propagation proxies.

Never infer a historical tight-chain edge from today's membership. March concept
snapshots can support a labelled proxy study, not verified information propagation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from mainwave_engine import route_evidence as RE
from research.route_followup import arrays, enrich, simulate


def symbol(value: str) -> str:
    text = str(value).lower()
    if re.fullmatch(r"(?:sh|sz)_\d{6}", text):
        return text
    match = re.fullmatch(r"(\d{6})\.(sh|sz)", text)
    return f"{match[2]}_{match[1]}" if match else ""


def catalog(root: Path) -> pd.DataFrame:
    definitions = [
        ("results/mainwave_engine/20??????/theme_propagation.csv", "DATED_EXPLICIT_EDGE", "theme/member/leader dates; verify registry active_from; not immutable production proof"),
        ("results/mainwave_engine/20??????/slow_carrier_diagnostics.csv", "DATED_SLOW_CARRIER", "current route3 context schema"),
        ("results/mainwave_engine/20??????/direction3_theme_propagation.csv", "DATED_THEME_CONTEXT", "theme context, not necessarily selected carriers; some reports rebuilt later"),
        ("results/strategy_tracker_v2/20??????_v*/top_hotspots.csv", "HOT_LEADERS_ONLY", "stocks are hot/limit-up members, not complete laggard universe; deduplicate versions"),
        ("concept_cache/concepts_*.json", "DATED_CONCEPT_MEMBERS", "declared date + members + heat; no explicit lead-to-carrier causality"),
        ("concept_mapping_history/concept_mapping_*.json", "DATED_MAPPING", "use only from generated time onward; March snapshot mostly industry classifications"),
        ("concept_analysis_*.json", "DATED_OLD_CONCEPT_ANALYSIS", "selected hot members; selection coverage incomplete"),
        ("cache/limit_up*.csv", "DATED_LIMIT_UP", "industry and limit-up stocks only; cannot reconstruct unselected members"),
        ("results/mainwave_engine/20??????/input_snapshots/*/01_theme_registry.json", "REGISTRY_SNAPSHOT", "active_from is binding; repeated snapshots do not add history"),
    ]
    rows = []
    for pattern, tier, note in definitions:
        for p in sorted(root.glob(pattern)):
            dates = re.findall(r"20\d{6}", str(p.relative_to(root)))
            declared = dates[0] if dates else ""
            count, internal = 0, ""
            if p.suffix == ".csv":
                try:
                    f = pd.read_csv(p)
                    count = len(f)
                    date_column = next((c for c in ("as_of", "trade_date", "date") if c in f), None)
                    if date_column:
                        internal = "|".join(sorted(f[date_column].dropna().astype(str).unique()))
                except pd.errors.EmptyDataError:
                    pass
            else:
                j = json.loads(p.read_text(encoding="utf-8-sig"))
                internal = str(j.get("date", j.get("generated", "")))
                count = len(j.get("concept_stocks", j.get("mapping", j.get("themes", j.get("top_concepts", [])))))
            if not declared and internal:
                declared = str(pd.Timestamp(internal).strftime("%Y%m%d"))
            rows.append({"path": str(p), "tier": tier, "declared_date": declared,
                         "internal_date": internal, "rows": count, "limitation": note,
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    return pd.DataFrame(rows)


def load_frame(args: tuple[str, str]) -> tuple[str, pd.DataFrame]:
    s, end = args
    p = Path("data") / f"{s}_daily.csv"
    if not p.exists():
        return s, pd.DataFrame()
    raw = pd.read_csv(p, encoding="utf-8-sig")
    f = RE.build(raw[pd.to_datetime(raw.date).le(end)], s)
    return s, enrich(f) if len(f) else f


def march_events(root: Path) -> list[dict]:
    events = []
    for p in sorted((root / "concept_cache").glob("concepts_*.json")):
        j = json.loads(p.read_text(encoding="utf-8-sig"))
        date = str(pd.Timestamp(str(j["date"])).date())
        filename_date = re.search(r"20\d{6}", p.name)[0]
        if date.replace("-", "") != filename_date:
            continue
        for theme, members in j.get("concept_stocks", {}).items():
            info = j.get("concept_info", {}).get(theme, {})
            normalized = sorted({s for raw in members if (s := symbol(raw))})
            if info.get("limit_up_count", 0) >= 3 and len(normalized) >= 3 and "ST" not in theme:
                events.append({"as_of": date, "theme": theme, "members": normalized,
                               "limit_count": info["limit_up_count"], "source": str(p),
                               "anchor_type": "HEAT_OBSERVED_NOT_FIRST_IGNITION"})
    return events


def replay(events: list[dict], frames: dict[str, pd.DataFrame], mode: str = "accepted_only") -> pd.DataFrame:
    rows = []
    for event in events:
        date = pd.Timestamp(event["as_of"])
        at_anchor = {}
        for s in event["members"]:
            f = frames[s]
            if f.empty:
                continue
            hit = f.index[f.date.eq(date)]
            if len(hit) and bool(f.eligible.iloc[hit[0]]):
                at_anchor[s] = int(hit[0])
        ranked = sorted(at_anchor, key=lambda s: (-float(frames[s].ret5.iloc[at_anchor[s]] /
                            (frames[s].atr_pct.iloc[at_anchor[s]] / 100)), s))
        if len(ranked) < 3:
            continue
        leader = ranked[0]
        lf = frames[leader]
        lead = lf.iloc[at_anchor[leader]]
        leader_strength = float(lead.ret5 / (lead.atr_pct / 100))
        if leader_strength < 1:
            continue
        for s in ranked[1:]:
            f, i = frames[s], at_anchor[s]
            a, first = arrays(f), f.iloc[i]
            lag = leader_strength - float(first.ret5 / (first.atr_pct / 100))
            recent_prep = bool(f.prepared.iloc[max(0, i - 10):i + 1].any())
            base = {"as_of": event["as_of"], "theme": event["theme"], "symbol": s, "leader": leader,
                    "source": event["source"], "tier": "DATED_CONCEPT_PROXY_NOT_TIGHT_CHAIN",
                    "entry_mode": mode,
                    "anchor_type": event["anchor_type"], "lag_atr": lag, "recent_preparation": recent_prep,
                    "bias_atr": float(first.bias_atr), "selected": False, "status": "NO_ENTRY_8",
                    "source_grade": "DECLARED_DATE_ONLY", "relationship_grade": "CONCEPT_PROXY",
                    "strict_source_verified": False,
                    "initial_checks": "|".join(code for code, failed in [
                        ("TARGET_NOT_PREPARED", not recent_prep),
                        ("SUPPLY_PRESSURE", bool(first.supply_warning)),
                        ("NO_LAG", lag < .75), ("PRICE_EXTENDED", first.bias_atr > 2)] if failed),
                    "target_already_accepted": first.breakout_status in {"ACCEPTED", "ADVANCING"},
                    "mature20": i + 20 < len(f), "matched_date": None}
            if i + 20 < len(f):
                base["anchor_r20"] = (f.return_index.iloc[i + 20] / first.return_index - 1) * 100
            if lag < .75 or not recent_prep or first.bias_atr > 2 or first.supply_warning:
                base["status"] = "PREPARATION_OR_LAG_NOT_MET"
                rows.append(base)
                continue
            floor = first.structure_floor - .5 * first.atr20
            for t in range(i + 1, min(len(f), i + 9)):
                row = f.iloc[t]
                expired_cycle = (first.breakout_status in {"ACCEPTED", "ADVANCING"}
                                 and (row.breakout_id != first.breakout_id
                                      or row.breakout_status not in {"ACCEPTED", "ADVANCING"}))
                if row.return_index < floor or expired_cycle:
                    base["status"] = "STRUCTURE_BROKEN"
                    break
                # Leader must still be holding its anchor close within a frozen ATR.
                leader_day = lf[lf.date.eq(row.date)]
                if leader_day.empty or leader_day.iloc[0].return_index < lead.return_index - lead.atr20:
                    base["status"] = "LEADER_WEAK_OR_MISSING"
                    break
                recent = f.iloc[max(i, t - 5):t]
                reclaim = (row.breakout_status in {"ACCEPTED", "ADVANCING"}
                           and row.return_index > f.index_high.iloc[t - 1]
                           and row.return_index > row.ma20 and row.close_location >= .6
                           and row.amount_ratio >= 1 and row.bias_atr <= 2 and not row.supply_warning
                           and (recent.amount_ratio.le(.85) & recent.index_low.le(recent.ma20 + recent.atr20)).any())
                if row.eligible and (row.accepted_bias2 or (mode == "accepted_or_reclaim" and reclaim)):
                    trade = simulate(a, t, s, "stage36_20")
                    base.update(selected=True, matched_date=str(row.date.date()), **trade)
                    break
            rows.append(base)
    for row in rows:
        failed = row["initial_checks"].split("|") if row["initial_checks"] else []
        if not failed and row["status"] != "CLOSED":
            failed.append(row["status"])
        row["first_rejection"] = failed[0] if failed else "PASSED"
        row["all_rejections"] = "|".join(failed)
    return pd.DataFrame(rows)


def run(output: Path, end: str, workers: int) -> None:
    root = Path(".")
    output.mkdir(parents=True, exist_ok=True)
    inventory = catalog(root)
    inventory.to_csv(output / "legacy_inventory.csv", index=False)
    events = march_events(root)
    symbols = sorted({s for e in events for s in e["members"]})
    frames = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for s, f in pool.map(load_frame, [(s, end) for s in symbols], chunksize=4):
            frames[s] = f
    result = pd.concat([replay(events, frames, mode) for mode in ("accepted_only", "accepted_or_reclaim")], ignore_index=True)
    result.to_csv(output / "legacy_proxy_cases.csv", index=False)
    groups = []
    for tier, g in inventory.groupby("tier"):
        dates = sorted(d for d in g.declared_date.unique() if d)
        usable_dates = sorted(d for d in g.loc[g.rows.gt(0), "declared_date"].unique() if d)
        groups.append({"tier": tier, "files": len(g), "dates": len(dates),
                       "first": dates[0] if dates else "", "last": dates[-1] if dates else "", "rows": int(g.rows.sum()),
                       "nonempty_dates": len(usable_dates), "first_nonempty": usable_dates[0] if usable_dates else ""})
    pd.DataFrame(groups).to_csv(output / "legacy_coverage.csv", index=False)
    selected = result[result.selected] if len(result) else pd.DataFrame()
    closed = selected[selected.status.eq("CLOSED")] if len(selected) else pd.DataFrame()
    # Multiple themes/snapshots can nominate the same stock on the same confirmation day.
    unique = closed.drop_duplicates(["entry_mode", "symbol", "matched_date"]) if len(closed) else closed
    mode_stats = []
    for mode, group in result.groupby("entry_mode"):
        trades = unique[unique.entry_mode.eq(mode)] if len(unique) else unique
        mode_stats.append({"mode": mode, "observations": len(group), "selected": int(group.selected.sum()),
                           "unique_closed": len(trades), "profit": int(trades.net.gt(0).sum()) if len(trades) else 0,
                           "mean_net": float(trades.net.mean()) if len(trades) else None,
                           "blockers": group.status.value_counts().to_dict()})
    summary = {"as_of": end, "catalog": groups, "snapshot_heat_events": len(events),
               "snapshot_members": len(symbols), "observations": len(result), "selected_rows": len(selected),
               "closed_rows": len(closed), "unique_closed_stock_signal": len(unique),
               "unique_profit": int(unique.net.gt(0).sum()) if len(unique) else 0,
               "unique_mean_net": float(unique.net.mean()) if len(unique) else None,
               "entry_mode_comparison": mode_stats,
               "strict_route3_successes": None,
               "verdict": "Older reusable data EXISTS. Dated concept proxy is NOT verified tight-chain propagation.",
               "limitations": ["Filename/as_of are declared dates, not cryptographic historical availability proof",
                  "March snapshots do not prove information leader or economic propagation edge",
                  "Heat-observed anchors are not asserted first ignition; missing member snapshots not backfilled",
                  "Leader chosen by same-day normalized price strength; only following sessions enter",
                  "Single March-2026 reconstruction, no unseen OOS or production promotion"]}
    (output / "legacy_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/mainwave_engine/route_followup_20260910"))
    parser.add_argument("--end", default="2026-09-09")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.output, args.end, args.workers)
