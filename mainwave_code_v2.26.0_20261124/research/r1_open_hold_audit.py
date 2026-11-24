"""Paired R1 study: add only a next-open check against the frozen breakout."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mainwave_engine import route_evidence as RE
from research.mainwave_path import AC, ROOT, market_flags, portfolio, read_raw, sessions
from research.r1_early_demand_replay import execution, normalize_suspensions, observe_top2
from research.r2_pool_compare import load_limits
from research.route_followup import arrays, enrich


BASE = Path("results/research/mainwave_path/r1_early_demand_20261119_validated")
VERSION = "r1_open_hold_audit_v1"
BLOCK = "ENTRY_BELOW_FROZEN_BREAKOUT"
UNKNOWN = "ENTRY_FROZEN_BREAKOUT_UNKNOWN"


def check_open(price, boundary):
    """Same adjusted basis for both prices; equality holds, with floating-point tolerance only."""
    if not (np.isfinite(price) and price>0 and np.isfinite(boundary) and boundary>0):
        return UNKNOWN
    return BLOCK if price < boundary and not np.isclose(price,boundary,rtol=0,atol=1e-9) else "PASS"


def init(cal, end, limits):
    global CAL, END, LIMITS
    CAL,END,LIMITS=cal,end,limits


def replay(job):
    symbol, dates=job
    raw=read_raw(symbol)
    f=RE.build(normalize_suspensions(raw[raw.date.le(END)]),symbol)
    a=arrays(enrich(f))
    loc={d:i for i,d in enumerate(a["day"])}
    out=[]
    for d,chain in dates:
        i=loc[d]
        if (a["breakout_id"][i]!=chain or a["breakout_status"][i]!="ATTEMPT"
                or a["effort_result"][i]!="DEMAND_EXPANSION" or not a["eligible"][i]
                or a["supply_warning"][i] or not a["bias_atr"][i]<=2):
            raise ValueError(f"frozen signal changed: {symbol} {d} {chain}")
        tr=execution(a,i,symbol,CAL,LIMITS)
        o=dict(version=VERSION,status=tr["status"],base_opened=bool(tr.get("path")),gate="NOT_APPLICABLE",
               net=tr.get("net"),entry=tr.get("filled_day"),raw_open=tr.get("raw_open"),exit=tr.get("exit_date"),
               risk=tr.get("initial_risk"))
        if o["base_opened"]:
            j=i+1  # execution() already verified this is the next market session.
            level=a["breakout_boundary_index"][i]
            opened=a["index_open"][j]
            o["gate"]=check_open(opened,level)
            if np.isfinite(level) and level>0:
                o.update(raw_frozen_breakout=level*a["open"][j]/opened,
                         versus_breakout_pct=100*(opened/level-1))
        out.append((symbol,d,o,tr))
    return out


def standalone(f):
    rows=[]
    for year,g in f[f.mature].groupby(f.attempt.str[:4]):
        for label,group in (("all_markets",g),("green_only",g[g.early_market.eq("GREEN")])):
            closed=group[group.status.eq("CLOSED")]
            blocked=closed[closed.gate.eq(BLOCK)]
            kept=closed[closed.gate.eq("PASS")]
            outcomes=["REJECTED","EXPIRED"]
            rows.append(dict(year=year,sample=label,signals=len(group),
                             base_opened=int(group.base_opened.sum()),base_closed=len(closed),
                             base_failed=int(closed.terminal.isin(outcomes).sum()),
                             blocked=int(group.gate.eq(BLOCK).sum()),blocked_closed=len(blocked),
                             bad_data=int(group.gate.eq(UNKNOWN).sum()),kept_closed=len(kept),
                             blocked_failed=int(blocked.terminal.isin(outcomes).sum()),
                             blocked_accepted=int(blocked.terminal.eq("ACCEPTED").sum()),
                             blocked_losses=int(blocked.net.lt(0).sum()),blocked_winners=int(blocked.net.gt(0).sum()),
                             losses_avoided=-blocked.loc[blocked.net.lt(0),"net"].sum(),
                             profits_missed=blocked.loc[blocked.net.gt(0),"net"].sum(),
                             change_per_signal=(-blocked.net.sum()/len(group)) if len(group) else np.nan,
                             base_mean=closed.net.mean(),kept_mean=kept.net.mean(),blocked_mean=blocked.net.mean(),
                             kept_median=kept.net.median(),base_median=closed.net.median(),
                             base_p10=closed.net.quantile(.1),kept_p10=kept.net.quantile(.1)))
    return pd.DataFrame(rows)


def guard_intervals(f,cal):
    from research.r1_extension_audit import pair_intervals
    # Same settled base events; a blocked trade stays in the denominator at zero, while unknown exits are omitted.
    g=f[f.mature & f.status.eq("CLOSED") & f.gate.isin(["PASS",BLOCK])].copy()
    g["date"]=g.attempt
    g["market"]=g.early_market
    g["net_delta"]=np.where(g.gate.eq(BLOCK),-g.net,0.)
    return pair_intervals(g,cal).drop(columns="near_better")


def check_reference(nav, ledger, base, year, fee):
    tag=f"{year}_early_{fee}bp"
    old=pd.read_csv(base/f"nav_{tag}.csv")
    for key in ("equity","drawdown_pct","cash","open"):
        if not np.allclose(nav[key],old[key],rtol=0,atol=1e-7):
            raise ValueError(f"frozen base {tag} changed: {key}")
    gold=pd.read_csv(base/f"ledger_{tag}.csv",usecols=["day","symbol","status"])
    if not ledger[["day","symbol","status"]].reset_index(drop=True).equals(gold):
        raise ValueError(f"frozen base {tag} ledger mismatch")


def run(base,out,workers):
    base,out=Path(base),Path(out)
    if out.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("research cannot write to production archives")
    protocol=json.loads((base/"protocol.json").read_text(encoding="utf-8"))
    start,end=protocol["start"],protocol["end"]
    cal=tuple(d for d in sessions() if d<=end)
    a=pd.read_csv(base/"attempts.csv.gz",low_memory=False)
    a=a[a.early_signal.notna()].copy()
    out.mkdir(parents=True,exist_ok=False)
    spec=dict(version=VERSION,baseline=str(base),start=start,end=end,events=len(a),
              change="Base buys only if actual next open >= frozen initial breakout boundary, same adjusted basis",
              execution="2ATR and other checks unchanged, no backfill after opening rejection",
              comparison="Same early signal Top2; do not prefilter before ranking",
              design="fixed diagnostic; previously examined dates; no portfolio or official strategy changes")
    spec["attempts_sha256"]=hashlib.sha256((base/"attempts.csv.gz").read_bytes()).hexdigest()
    (out/"protocol.json").write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding="utf-8")
    jobs=[(symbol,list(g[["early_signal","cycle_id"]].itertuples(index=False,name=None))) for symbol,g in a.groupby("symbol")]
    limits=load_limits(d for d in cal if d>=start)
    rows=[]; regular={}; gated={}
    with ProcessPoolExecutor(workers,initializer=init,initargs=(cal,end,limits)) as p:
        for i,result in enumerate(p.map(replay,jobs,chunksize=8),1):
            for symbol,d,row,tr in result:
                rows.append(dict(symbol=symbol,early_signal=d,**row))
                regular[symbol,d]=tr
                gated[symbol,d]=({"status":row["gate"]} if row["gate"] in (BLOCK,UNKNOWN) else tr)
            if i%500==0:
                print(f"[open_hold] {i}/{len(jobs)} stocks",flush=True)
    f=a.merge(pd.DataFrame(rows),on=["symbol","early_signal"],validate="one_to_one")
    if len(f)!=len(a) or not f.status.eq(f.early_status).all() or not np.allclose(
            f.net,f.early_net,rtol=0,atol=1e-7,equal_nan=True):
        raise ValueError("early events no longer match the frozen study")
    f.to_csv(out/"same_signal_outcomes.csv.gz",index=False)
    s=standalone(f)
    s.to_csv(out/"event_impact.csv",index=False)
    guard_intervals(f,cal).to_csv(out/"event_intervals.csv",index=False)
    states=market_flags(cal)
    books=[]; differences=[]
    for year in sorted(f.attempt.str[:4].unique()):
        dates=tuple(d for d in cal if d>=start and d.startswith(year))
        picks=observe_top2(f[f.attempt.str.startswith(year)],"early")
        for fee in (30,40):
            results={}
            for arm,trades in (("base",regular),("opening_hold",gated)):
                n,l=portfolio(dates,picks,states,trades,fee/10000)
                if arm=="base":
                    check_reference(n,l,base,year,fee)
                tag=f"{year}_{arm}_{fee}bp"
                n.to_csv(out/f"nav_{tag}.csv",index=False)
                l.to_csv(out/f"ledger_{tag}.csv",index=False)
                results[arm]=l
                net=l.loc[l.status.eq("CLOSED"),"pnl"]
                books.append(dict(year=year,arm=arm,fee_bp=fee,
                                  seats_max=int(n.open.max()),closed=len(net),
                                  net_pct=100*(n.equity.iat[-1]/AC["capital"]-1),max_dd=n.drawdown_pct.min(),
                                  roi_ex_best=100*((n.equity.iat[-1]-max(0,net.max()))/AC["capital"]-1),
                                  gate_blocks=int(l.status.isin([BLOCK,UNKNOWN]).sum()),
                                  statuses=l.status.value_counts().to_dict()))
            x,y=(results[k] for k in ("base","opening_hold"))
            d=x.merge(y,on=["symbol","day"],suffixes=("_base","_hold"),validate="one_to_one",how="outer")
            differences.append(d.assign(year=year,fee_bp=fee))
    pd.concat(differences,ignore_index=True).to_csv(out/"book_attribution.csv",index=False)
    b=pd.DataFrame(books)
    b.to_csv(out/"book_summary.csv",index=False)
    print(s[["year","sample","base_closed","base_failed","blocked_closed","blocked_failed","blocked_winners",
             "blocked_losses","base_mean","kept_mean"]].round(3).to_string(index=False))
    print(b.drop(columns="statuses").round(3).to_string(index=False))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base",type=Path,default=BASE)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--workers",type=int,default=10)
    a=ap.parse_args()
    run(a.base,a.out,a.workers)
