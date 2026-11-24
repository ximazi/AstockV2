"""Outcome-first audit of archived daily stock discoveries. Never rerun production."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from mainwave_engine.hotspot_relay import _frozen
from mainwave_engine.review_clock import visible


PRE, POST = 10, 3
BASE = {"P5", "ROUTE", "STRENGTH", "ANALYSIS", "TRAJECTORY"}
LEGACY = {"V44_TECH", "V44_DYNAMIC"}
KINDS = ("THREE_BOARDS", "LIQUID_5D_SURGE")
PHASES = ("BEFORE_START", "START", "IN_RISE", "ANCHOR", "AFTER_ANCHOR")
DAY_COLS = ["date", "close", "preclose", "volume", "amount", "tradestatus", "pct_chg", "isST"]


def read_json(path):
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8-sig"))
    return {}


def tape(data, days):
    frames, skipped = [], []
    for path in sorted(data.glob("*_daily.csv")):
        symbol = path.stem.removesuffix("_daily")
        if not symbol.startswith(("sh_60", "sh_68", "sz_00", "sz_30")):
            continue
        try:
            f = pd.read_csv(path, usecols=DAY_COLS).drop_duplicates("date", keep="last").sort_values("date")
        except (OSError, ValueError):
            skipped.append(symbol)
            continue
        traded = f.tradestatus.eq(1) & f.volume.gt(0) & f.close.gt(0) & f.preclose.gt(0)
        history = traded.cumsum()
        f = f.loc[traded & history.gt(20) & f.isST.eq(0) & f.date.between(days[0], days[-1])].copy()
        if f.empty:
            continue
        f["symbol"] = symbol
        percent = 20 if symbol.startswith(("sh_68", "sz_30")) else 10
        limit_cents = np.floor(f.preclose * (100 + percent) + 0.50000001)
        f["sealed"] = np.rint(f.close * 100).eq(limit_cents)
        frames.append(f[["date", "symbol", "close", "amount", "pct_chg", "sealed"]])
    prices = pd.concat(frames, ignore_index=True)
    prices["amount_rank"] = prices.amount.where(prices.amount.gt(0)).groupby(prices.date).rank(pct=True)
    return prices, skipped


def event_rows(prices, days, start):
    for symbol, rows in prices.groupby("symbol", sort=True):
        g = rows.set_index("date").reindex(days)
        sealed = g.sealed.eq(True).to_numpy()
        run = 0
        for i, yes in enumerate(sealed):
            run = run + 1 if yes else 0
            if run == 3 and days[i] >= start:
                yield (symbol, KINDS[0], i - 2, i)
        daily = g.pct_chg.where(g.pct_chg.gt(-100))
        ret5 = np.expm1(np.log1p(daily / 100).rolling(5).sum())
        up = daily.gt(0).rolling(5).sum()
        qualified = ret5.ge(.20 - 1e-10) & up.ge(3) & g.amount_rank.ge(.80)
        last = -len(days)
        for i in np.flatnonzero(qualified):
            if i - last > 5 and days[i] >= start:
                yield (symbol, KINDS[1], i - 4, i)
            last = i


def availability(path, day, cutoff):
    return "TIMESTAMPED" if visible(path, day, cutoff) else "DATED_ONLY"


def scan_archive(root, days, interested):
    coverage, history = defaultdict(set), defaultdict(list)
    for i, day in enumerate(days):
        stamp = day.replace("-", "")
        folder = root / "mainwave_engine" / stamp
        old = root / "strategy_tracker_v2" / (stamp + "_v44")
        next_day = days[i + 1] if i + 1 < len(days) else (date.fromisoformat(day) + timedelta(days=1)).isoformat()
        cutoff = f"{next_day}T09:15:00+08:00"
        needed = interested[i]

        def add(source, path, row, state, hit=True, trust="DATED_ONLY", rank=None):
            symbol = row.get("symbol")
            if symbol in needed:
                history[symbol, i].append(dict(
                    date=day, symbol=symbol, source=source, state=state, positive=int(hit),
                    evidence=trust, rank=rank or row.get("rank", ""), name=row.get("name", ""),
                    routes=row.get("route_ids") or row.get("route_id") or row.get("route", ""),
                    structure=row.get("breakout_status", ""),
                    theme=row.get("belonging_theme") or row.get("external_theme") or "",
                    detail=row.get("route_risks") or row.get("exclusion") or "",
                    bias_atr=row.get("bias_atr", row.get("route_bias_atr", "")),
                    probe_at=row.get("probe_at", ""),
                    prepared_at=row.get("preparation_first_seen", row.get("prepared_at", "")),
                    file=str(path.relative_to(root.parent)),
                ))

        tables = (
            ("P5", folder / "preparation_watch.csv", "detector"),
            ("ROUTE", folder / "route_candidates.csv", "route_status"),
            ("TRAJECTORY", folder / "trajectory.csv", "state"),
            ("ROUTE_AUDIT", folder / "route_scan_audit.csv", "exclusion"),
            ("CARRIER", folder / "carrier_qualification.csv", "carrier_qualification"),
            ("R2_STAGE", folder / "r2_stage_pool.csv", "lane"),
            ("V44_TECH", old / "main_wave_candidates_ranked.csv", "level_name"),
            ("V44_DYNAMIC", old / "mainwave_dynamic_h2512.csv", "tier"),
        )
        for source, path, field in tables:
            if not path.is_file():
                continue
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not {"symbol", field}.issubset(reader.fieldnames or ()):
                    continue
                coverage[i].add(source)
                trust = availability(path, day, cutoff)
                for row in reader:
                    if row["symbol"] not in needed:
                        continue
                    if any(row.get(k) not in (None, "", day, stamp) for k in ("date", "as_of")):
                        continue
                    state = row[field] or ""
                    usable = True
                    rank = None
                    if source == "TRAJECTORY":
                        usable = state in ("ARMED", "READY", "TRIGGERED") and row.get("data_fresh", "").lower() == "true"
                        state += ("/" + row["capital_stage"]) if row.get("capital_stage") else ""
                    elif source == "P5":
                        usable = row.get("reclaim_failed", "").lower() != "true"
                        state += "/RECLAIM_FAILED" if not usable else ""
                    elif source == "CARRIER":
                        usable = state == "QUALIFIED_LURK"
                    elif source == "R2_STAGE":
                        usable = False  # Full working pool, not the shortlist.
                        state += "/" + (row.get("admission") or "")
                    elif source == "ROUTE_AUDIT":
                        usable = False
                    elif source == "ROUTE":
                        usable = bool(row.get("route_id") or row.get("route_ids")) and state not in (
                            "RISK_WATCH", "DEGRADED", "INVALID", "FAILED", "DATA_GAP", "")
                    elif source == "V44_TECH":
                        rank = row.get("chan_rank")
                    add(source, path, row, state, usable, trust, rank)

        for source, name in (("STRENGTH", "strong_trend_shadow.json"), ("ANALYSIS", "analysis_board.json")):
            path = folder / name
            doc = read_json(path)
            if doc.get("as_of") != day:
                continue
            rows = doc.get("boards", {}).get("attention") if source == "STRENGTH" else doc.get("records")
            if not isinstance(rows, list) or doc.get("summary", {}).get("status") == "FAILED":
                continue
            coverage[i].add(source)
            for rank, row in enumerate(rows, 1):
                if source == "STRENGTH":
                    add(source, path, row, "/".join(row.get("labels") or []) or str(row.get("entry") or ""),
                        rank=rank)
                else:
                    add(source, path, row, str(row.get("value_tier") or "") + "/" + str(row.get("structural_advice") or ""),
                        not row.get("tracking_only") and row.get("value_tier") not in ("RISK_WATCH", "DATA_WAIT"), rank=rank)
                    if row.get("initial_review") is True:
                        add("ENTRY_REVIEW", path, row, "INITIAL_REVIEW", rank=rank)

        core, path = _frozen(folder, day, cutoff)
        if path:
            coverage[i].add("HOTSPOT")
            doc = read_json(path.with_name("report.json"))
            top = doc.get("summary", {}).get("focus_ids", [])
            rows = {r["symbol"]: r for r in core.get("rows", []) if "symbol" in r}
            for rank, symbol in enumerate(core.get("focus", []), 1):
                row = rows.get(symbol, {"symbol": symbol})
                add("HOTSPOT", path, row, " / ".join(row.get("roles") or []), trust="TIMESTAMPED", rank=rank)
            for rank, symbol in enumerate(top, 1):
                if symbol in rows:
                    add("FUPAN_TOP", path.with_name("report.json"), rows[symbol], "FOCUS", trust="TIMESTAMPED", rank=rank)
    return coverage, history


def phase(i, begin, anchor):
    if i < begin:
        return PHASES[0]
    if i == begin:
        return PHASES[1]
    if i < anchor:
        return PHASES[2]
    return PHASES[3 if i == anchor else 4]


def describe(hits, begin, anchor, days):
    if not hits:
        return ("", "", "")
    j = min(i for i, _ in hits)
    return days[j], phase(j, begin, anchor), "/".join(sorted({r["source"] for i, r in hits if i == j}))


def audit(events, history, coverage, days, prices, names=None):
    price = prices.set_index(["symbol", "date"])
    names = names or {}
    for symbol, kind, begin, anchor in events:
        start, end = begin - PRE, min(len(days) - 1, anchor + POST)
        key = f"{kind}:{symbol}:{days[anchor]}"
        window = [(i, r) for i in range(start, end + 1) for r in history.get((symbol, i), [])]
        hits = [(i, r) for i, r in window if r["positive"]]
        new = [(i, r) for i, r in hits if r["source"] not in LEGACY]
        full_new = all(BASE <= coverage[i] for i in range(start, anchor + 1))
        full_any = all(BASE <= coverage[i] or LEGACY <= coverage[i] for i in range(start, anchor + 1))
        row = dict(event_id=key, symbol=symbol,
                   name=next((r["name"] for _, r in reversed(window) if r["name"]), names.get(symbol, "")),
                   kind=kind, start=days[begin], anchor=days[anchor], anchor_close=price.at[(symbol, days[anchor]), "close"],
                   ret_window=round(float(np.prod(1 + price.loc[(symbol, days[begin:anchor + 1]), "pct_chg"].to_numpy() / 100) - 1), 6),
                   full_new_to_anchor=int(full_new), full_archive_to_anchor=int(full_any),
                   new_anchor_available=int(BASE <= coverage[anchor]), post_sessions=end - anchor,
                   dated_records=len(window), positive_records=len(hits), sources_observed="/".join(sorted({r["source"] for _, r in hits})))
        for prefix, sample in (
            ("first", hits), ("new", new), ("v44", [(i, r) for i, r in hits if r["source"] in LEGACY]),
            ("verified", [(i, r) for i, r in new if r["evidence"] == "TIMESTAMPED"]),
            ("p5", [(i, r) for i, r in new if r["source"] == "P5"]),
            ("route", [(i, r) for i, r in new if r["source"] == "ROUTE"]),
            ("strength", [(i, r) for i, r in new if r["source"] == "STRENGTH"]),
            ("hotspot", [(i, r) for i, r in new if r["source"] == "HOTSPOT"]),
            ("entry", [(i, r) for i, r in new if r["source"] == "ENTRY_REVIEW"]),
        ):
            first_day, stage, sources = describe(sample, begin, anchor, days)
            row.update({prefix + "_day": first_day, prefix + "_phase": stage, prefix + "_sources": sources})
        for route in ("R1", "R2", "R3"):
            path = [(i, r) for i, r in new if r["source"] == "ROUTE" and route in r["routes"].replace("|", ",").split(",")]
            row[route.lower() + "_day"] = describe(path, begin, anchor, days)[0]
        row["new_result"] = row["new_phase"] or ("NOT_SEEN_TO_ANCHOR" if full_new else "ARCHIVE_GAP")
        row["archive_result"] = row["first_phase"] or ("NOT_SEEN_TO_ANCHOR" if full_any else "ARCHIVE_GAP")
        yield row, [dict(event_id=key, rel_start=i - begin, rel_anchor=i - anchor, **r) for i, r in window]


def write_csv(path, records):
    if records:
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def run(start, end, output, data=Path("data"), root=Path("results")):
    days = sorted(set(pd.read_csv(data / "000300_SH_index.csv", usecols=["trade_date"]).trade_date))
    offset = next(i for i, day in enumerate(days) if day >= start)
    if offset < PRE + 7 or end > days[-1] or start > end:
        raise ValueError("Requested window lacks calendar coverage")
    days = [d for d in days[offset - 25:] if d <= end]
    prices, skipped = tape(data, days)
    events = sorted(event_rows(prices, days, start), key=lambda e: (e[3], e[1], e[0]))
    print(f"price rows={len(prices)} events={len(events)} unreadable={len(skipped)}", flush=True)
    interested = defaultdict(set)
    for symbol, _, b, a in events:
        for i in range(b - PRE, min(len(days), a + POST + 1)):
            interested[i].add(symbol)
    coverage, history = scan_archive(root, days, interested)
    mapping = read_json(root.parent / "concept_mapping.json").get("mapping", {})
    names = {s: item.get("name", "") for s, item in mapping.items()}
    cases, timeline = [], []
    for row, records in audit(events, history, coverage, days, prices, names):
        cases.append(row)
        timeline.extend(records)
    counts = {}
    for kind in KINDS:
        cohort = [r for r in cases if r["kind"] == kind]
        counts[kind] = {
            "events": len(cohort), "symbols": len({r["symbol"] for r in cohort}),
            "all_archives": Counter(r["archive_result"] for r in cohort),
            "new_archives": Counter(r["new_result"] for r in cohort),
            "new_full_window": Counter(r["new_result"] for r in cohort if r["full_new_to_anchor"]),
            "new_anchor_covered": Counter(r["new_result"] for r in cohort if r["new_anchor_available"]),
            "verified": Counter(r["verified_phase"] or "NO_VERIFIED_RECORD" for r in cohort),
        }
    shares = {k: {days[i] for i, src in coverage.items() if k in src and days[i] >= start}
              for k in sorted(set().union(*coverage.values()))}
    market = prices.groupby("date").size()
    archive = [{"date": day, "price_symbols": int(market.get(day, 0)), "sources": "/".join(sorted(coverage[i])),
                "new_base": int(BASE <= coverage[i]), "legacy_base": int(LEGACY <= coverage[i])}
               for i, day in enumerate(days)]
    meta = dict(start=start, end=days[-1], sessions=sum(d >= start for d in days), price_unreadable=skipped,
                archive_coverage={k: dict(n=len(v), first=min(v), last=max(v)) for k, v in shares.items() if v},
                outcomes=counts, overlap_symbols=len(set.intersection(*({r["symbol"] for r in cases if r["kind"] == k} for k in KINDS))))
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "events.csv", cases)
    write_csv(output / "timeline.csv", timeline)
    write_csv(output / "coverage.csv", archive)
    (output / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.start, args.end, args.output)
