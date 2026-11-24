"""Extract dated R2 research inputs. Never publishes signals or writes to the daily archive."""
import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from mainwave_engine import preparation_detector as PD, route_evidence as RE


ROOT = Path("results/mainwave_engine")
SYMBOL = re.compile(r"(?:sh_(?:60|68)|sz_(?:00|30))\d{4}")


def pools(calendar):
    records, dates = [], {}
    for day in calendar:
        folder = ROOT / day.replace("-", "")
        path = folder / "carrier_qualification.csv"
        if not path.exists():
            continue
        strict = pd.read_csv(path, dtype=str, keep_default_na=False)
        if set(strict.as_of) != {day}:
            raise ValueError(f"undated or mixed strict snapshot: {path}")
        strict = strict[(strict.data_fresh == "True") & (strict.hard_gate_pass == "True")
                        & (strict.carrier_qualification == "QUALIFIED_LURK")]
        path = folder / "preparation_watch.csv"
        dates[day] = "archive" if path.exists() else "replay"
        records.extend((day, "strict", s) for s in set(strict.symbol))
        if path.exists():
            early = pd.read_csv(path)
            if not early.empty and (set(early.as_of) != {day} or set(early.detector) != {PD.CHOSEN}):
                raise ValueError(f"incompatible early snapshot: {path}")
            records.extend((day, "early", s) for s in set(early.symbol))
    return records, dates


def trend_labels(raw, symbol, lo):
    """TREND v1.1 price/path rules; fixed Jan 1 cooldown origin and historical ST at onset."""
    c, amt, pc, days = (raw[k].to_numpy() for k in ("close", "amount", "pct_chg", "date"))
    starts, last = [], -100
    for i in range(max(60, days.searchsorted(lo[:4] + "-01-01")), len(c) - 40):
        if i - last < 20 or not np.isfinite(amt[i]) or amt[i] < 5e7 or not c[i] > 0:
            continue
        fwd = c[i+1:i+41]
        if not np.isfinite(fwd).all() or min(fwd) < c[i]*.85 or max(fwd) < c[i]*1.25:
            continue
        last = i  # LOTTERY and EXTENDED consume cooldown too, as in the project label.
        limit = 19.5 if symbol.startswith(("sz_30", "sh_68")) else 9.5
        if (days[i] >= lo and (pc[i-5:i+6] >= limit).sum() < 4 and c[i-60] > 0
                and c[i]/c[i-60] < 1.8 and raw.isST.iat[i] == 0):
            starts.append(days[i])
    return starts, days[-41] if len(c) >= 41 else ""


def init_worker(days, cutoff, check):
    global FIRST, END, REPLAY, BENCH
    FIRST, END, REPLAY, BENCH = min(days), cutoff, set(check), PD.benchmark_ret20()


def extract(path):
    symbol = path.name[:-10]
    raw = pd.read_csv(path).sort_values("date").drop_duplicates("date", keep="last")
    raw = raw[raw.date.le(END)].reset_index(drop=True)
    starts, mature = trend_labels(raw, symbol, FIRST)
    stop = max(REPLAY)
    # P5 uses at most 120 preceding tradable bars for qualification; keep an extra warm-up.
    hist = pd.concat([raw[raw.date.lt(FIRST) & raw.amount.gt(0)].tail(200),
                      raw[raw.date.between(FIRST, stop)]], ignore_index=True)
    frame = RE.build(hist, symbol)
    early = []
    if not frame.empty:
        ann = PD.annotate(frame, BENCH)
        early = [d for d in frame.loc[ann[PD.CHOSEN], "date"].dt.strftime("%Y-%m-%d") if d in REPLAY]
    cols = ["date", "open", "close", "preclose", "amount", "volume", "isST", "tradestatus", "adjustflag"]
    bars = raw.loc[raw.date.ge(FIRST), cols]
    return symbol, starts, mature, early, list(bars.itertuples(index=False, name=None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--start", default="2026-08-28")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cal = pd.read_csv("data/000300_SH_index.csv").trade_date.astype(str)
    days = sorted(set(pd.to_datetime(cal).dt.strftime("%Y-%m-%d")))
    days = [d for d in days if args.start <= d <= args.as_of]
    if not days or days[-1] != args.as_of:
        raise ValueError("as-of is not a local market session")
    records, sources = pools(days)
    checks = [d for d in sources if sources[d] == "archive"][:3]
    replay = [d for d in sources if sources[d] == "replay"] + checks
    args.out.mkdir(parents=True, exist_ok=False)
    files = sorted(p for p in Path("data").glob("*_daily.csv") if SYMBOL.fullmatch(p.name[:-10]))
    labels, maturity, prices, validation = [], [], [], {}
    archived = {(d,s) for d,p,s in records if p == "early" and d in checks}
    reconstructed = set()
    with ProcessPoolExecutor(args.workers, initializer=init_worker,
                             initargs=(list(sources), args.as_of, replay)) as workers:
        for i, (s, starts, mature, early, bars) in enumerate(workers.map(extract, files, chunksize=12), 1):
            labels.extend((s,d) for d in starts)
            maturity.append((s,mature))
            for d in early:
                if d in checks:
                    reconstructed.add((d,s))
                else:
                    records.append((d,"early",s))
            prices.extend((s,*row) for row in bars)
            if i % 500 == 0:
                print(f"extracted {i}/{len(files)}", flush=True)
    for d in checks:
        a, b = ({s for dt,s in rows if dt == d} for rows in (archived, reconstructed))
        validation[d] = dict(archive=len(a), replay=len(b), same=len(a & b),
                             only_archive=sorted(a-b), only_replay=sorted(b-a))
    for name, rows, cols in [
        ("pools", records, "day pool symbol"), ("trend", labels, "symbol day"),
        ("maturity", maturity, "symbol last_mature"),
        ("prices", prices, "symbol day open close preclose amount volume isST tradestatus adjustflag")]:
        pd.DataFrame(rows, columns=cols.split()).to_csv(args.out / (name+".csv.gz"), index=False)
    meta = dict(calendar=days, sources=sources, checks=validation, universe=len(files),
                detector=PD.VERSION, route=RE.VERSION, as_of=args.as_of)
    (args.out / "inputs.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("archive/replay checks:", {d:(v["same"],v["archive"],v["replay"]) for d,v in validation.items()})


if __name__ == "__main__":
    main()
