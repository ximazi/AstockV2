"""Compare frozen-source relay replay with simple and archived attention baselines."""

import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

from mainwave_engine import board_relay as W, eod_review as E
from mainwave_engine.hotspot_relay import _frozen


def evaluate(start, end, root, data_root, reference_root):
    days = E.calendar(data_root)
    stats = defaultdict(Counter)
    records, missing = [], []
    for day, target in zip(days, days[1:]):
        if not start <= day <= end:
            continue
        before = W.clock(target, "09:15") - timedelta(microseconds=1)
        base = Path(root) / day.replace("-", "")
        watch = W.build(day, root, data_root, reference_root, before)
        facts, _ = W.files_before(Path(root) / target.replace("-", "") / "eod_review", target,
                                  W.clock(target, "23:59:59"), "facts.json")
        if not facts:
            # A valid next-session close pack can be published after midnight.
            end_of_review = W.clock(W.next_session(days, target), "09:15") - timedelta(microseconds=1)
            facts, _ = W.files_before(Path(root) / target.replace("-", "") / "eod_review", target, end_of_review, "facts.json")
        if watch["status"] == "DATA_WAIT" or not facts:
            missing.append({"signal": day, "target": target, "reason": "EOD_ARCHIVE_MISSING"})
            continue
        core, source = _frozen(base, day, before.isoformat())
        classifications = {r["symbol"]: r["lane"] for r in watch["rows"]}
        seat_excluded = {r["symbol"] for r in watch["rows"] if any(
            d["seat_count"] >= 2 for d in r["seat_evidence"]["disclosures"])}
        archive = {lane: [s for s in core.get("focus", []) if classifications.get(s) == lane][:W.MAX_FOCUS]
                   for lane in ("FIRST", "RELAY")}
        for lane in ("FIRST", "RELAY"):
            for version, picks in (
                ("SHADOW", watch["focus"][lane]), ("FIRST_SEAL", watch["baseline"][lane]),
                ("ARCHIVED_HOTSPOT", archive[lane] if source else None),
                ("SEAT_2_NO_REFILL", [s for s in watch["focus"][lane] if s not in seat_excluded]
                 if watch.get("sources", {}).get("seats") else None),
            ):
                if picks is None:
                    continue
                # The historical comparator uses the same outcome reader and as-of cutoffs.
                assisted = {**watch, "focus": {lane: picks}, "baseline": {}}
                out = W.followup(assisted, target, facts, data_root)
                c = stats[lane, version]
                c.update(days=1, picks=len(picks))
                c.update("outcome_" + r["state"] for r in out)
                records.append({"date": day, "target": target, "lane": lane, "version": version,
                                "source_status": watch["status"], "gaps": watch.get("gaps", []), "rows": out})
    return {"start": start, "end": end, "version": W.VERSION, "orders_enabled": False,
            "mode": "REPLAY_NOT_EXECUTION", "outcomes": {f"{k[0]}:{k[1]}": dict(v) for k, v in stats.items()},
            "missing": missing, "records": records,
            "notes": ["No fills or profits inferred; one-price promotion is a separate outcome.",
                      "Historical source cutoffs do not establish today's ranking had been published.",
                      "Archived hotspot/seat data are available for fewer dates; compare on their intersection.",
                      "SEAT_2_NO_REFILL removes disclosed two-or-more seats from the same shortlist without backfilling."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", required=True, type=lambda s: date.fromisoformat(s).isoformat())
    p.add_argument("--end", required=True, type=lambda s: date.fromisoformat(s).isoformat())
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.end < a.start:
        p.error("--end must be on or after --start")
    a.output.mkdir(parents=True, exist_ok=False)
    review = evaluate(a.start, a.end, E.OUT_ROOT, "data", E.REFERENCE_ROOT)
    E._write(a.output / "review.json", review)
    print(f"[board_relay_review] outcomes={review['outcomes']} missing={len(review['missing'])} -> {a.output}")


if __name__ == "__main__":
    main()
