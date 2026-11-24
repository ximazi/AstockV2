"""Attribute the accepted_bias2 gate using frozen events and executable open-price replay."""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mainwave_engine import route_evidence as RE
from research.mainwave_path import AC, BREAK, ROOT, market_flags, read_raw, sessions
from research.r2_pool_compare import band, load_limits
from research.r2_pool_history import SYMBOL
from research.route_followup import arrays, enrich, simulate

BASE = Path("results/research/mainwave_path/20261119_validated")
END = "2026-11-18"
VERSION = "r1_extension_audit_v1"
COLS = ["symbol", "date", "bias_atr", "accepted_demand", "accepted_bias2"]
BUCKETS = ("below_1.5", "1.5_to_2", "2_to_2.5", "2.5_to_3", "above_3")


def bucket(n):
    return BUCKETS[0 if n<=1.5 else 1 if n<=2 else 2 if n<=2.5 else 3 if n<=3 else 4]


def board(s):
    return "STAR" if s.startswith("sh_68") else "CHINEXT" if s.startswith("sz_30") else "MAIN"


def load_events(scope):
    old=ROOT/"route_validation_20260910/events.csv.gz"
    a=pd.read_csv(old,usecols=COLS)
    frames=[a[a.date.between("2025-01-01",BREAK) & a.accepted_demand.eq(True)]]
    for d in scope:
        if d<=BREAK:
            continue
        a=pd.read_csv(ROOT/d.replace("-","")/"route_scan_audit.csv",usecols=[*COLS,"eligible","data_status"])
        frames.append(a.loc[a.date.eq(d)&a.data_status.eq("FRESH")&a.eligible.eq(True)&a.accepted_demand.eq(True),COLS])
    e=pd.concat(frames,ignore_index=True)
    e=e[e.symbol.str.fullmatch(SYMBOL)].copy()
    if e.duplicated(["symbol","date"]).any() or not np.isfinite(e.bias_atr).all():
        raise ValueError("nonunique or nonfinite archived events")
    # Within the eligible demand-accepted domain the only policy difference must be the bias cap.
    if not e.accepted_bias2.eq(e.bias_atr.le(2)).all():
        raise ValueError("archived gate not equivalent to 2 ATR among eligible demand events")
    return e


def initialize(cal, limits):
    global CAL, LIMITS
    CAL,LIMITS=cal,limits


def replay(job):
    symbol,events=job
    raw=read_raw(symbol)
    raw=raw[raw.date.le(END)]
    f=RE.build(raw,symbol)
    a=arrays(enrich(f)) if not f.empty else {}
    pos={d:i for i,d in enumerate(a.get("day",[]))}
    out=[]
    for r in events:
        d,atr_bias=r["date"],r["bias_atr"]
        t=dict(symbol=symbol,date=d,year=d[:4],board=board(symbol),bias_atr=atr_bias,band=bucket(atr_bias),
               mature=bisect_left(CAL,d)+20<len(CAL),status="EVIDENCE_MISMATCH")
        out.append(t)
        i=pos.get(d)
        if (i is None or not a["eligible"][i] or not a["accepted_demand"][i]
                or not np.isclose(a["bias_atr"][i],atr_bias,atol=.001,rtol=0)):
            continue
        at=a["breakout_attempt_at"][i]
        if at not in pos or pos[at]>=i:
            continue
        pulse=next((k for k in range(pos[at],i+1) if a["effort_result"][k]=="DEMAND_EXPANSION"),None)
        t.update(attempt=at,cycle_id=a["breakout_id"][i],pulse=a["day"][pulse] if pulse is not None else "",
                 acceptance_delay=i-pos[at],pulse_delay=i-pulse if pulse is not None else None,
                 amount20=float(np.median(a["amount"][i-20:i])),status="CENSORED")
        j=i+1
        if j>=len(a["day"]):
            continue
        entry=float(a["open"][j])
        op,atr=a["index_open"][j],a["atr20"][i]
        stop=max(op-2*atr,a["breakout_invalid_index"][i])
        risk=(op-stop)/op
        t.update(observed_open_date=a["day"][j],observed_open=entry,
                 opening_bias=(op-a["ma20"][i])/atr,opening_gap_atr=(op-a["index_close"][i])/atr,
                 above_boundary=bool(op>=a["breakout_boundary_index"][i]),risk_pct=100*risk)
        upper,_,source=band(symbol,SimpleNamespace(day=a["day"][j],preclose=a["preclose"][j],isST=a["isST"][j]),LIMITS)
        t["limit_source"]=source
        if not np.isfinite(upper):
            t["status"]="ENTRY_LIMIT_UNKNOWN"
            continue
        if entry>=upper:
            t["status"]="ENTRY_LIMIT_UP"
            continue
        shares=200 if symbol.startswith("sh_68") else 100
        if entry*shares>AC["capital"]*AC["symbol_cap_pct"]/100 or entry*shares*risk>AC["capital"]*AC["single_risk_pct"]/100:
            t["status"]="RISK_OR_LOT"
            continue
        trace=[]
        result=simulate(a,i,symbol,"structure36_20",gap_guard=True,trace=trace,calendar=CAL,conservative_limits=LIMITS)
        t.update(result)
        t["opened"]=bool(trace)
        if trace:
            t["entry_open_day"]=trace[0]["date"]
            t["opening_back_in_2atr"]=bool(t["opening_bias"]<=2 and t["above_boundary"])
    return out


def compare_band(f):
    rows=[]
    for selection,g in (("all_markets",f),("green_only",f[f.market.eq("GREEN")])):
        for (year,b),x in g.groupby(["year","band"]):
            m=x[x.mature]
            c=m[m.status.eq("CLOSED")]
            opened=m.opened.eq(True).sum()
            rows.append(dict(selection=selection,year=year,band=b,signals=len(x),mature=len(m),
                             opened=int(opened),closed=len(c),window_unknown=int(opened-len(c)),
                             business_days=c.date.nunique(),mean=c.net.mean(),median=c.net.median(),
                             p10=c.net.quantile(.1),win_pct=c.net.gt(0).mean()*100 if len(c) else np.nan,
                             ex_top=c.net.drop(c.net.idxmax()).mean() if len(c)>1 else np.nan,
                             mean_at_40bp=c.net.mean()-.1,stop_median=c.risk_pct.median(),
                             mean_net_per_mature_slot=c.net.sum()/len(m) if len(m) else np.nan,
                             exclusions=m.status.value_counts().to_dict()))
    return pd.DataFrame(rows)


def linked_waves(f,base,cal):
    a=pd.read_csv(base/"opportunity_audit.csv",dtype={"start":str},keep_default_na=False)
    g=pd.read_csv(base/"route_gate_audit.csv",dtype={"start":str})
    waves=a.merge(g,on=["symbol","start"],validate="one_to_one")
    waves=waves[waves.start.str.startswith("2026") & waves.gate.eq("DEMAND_ACCEPTED_BUT_BASELINE_FILTERED")]
    by={(r["symbol"],r["date"]):r for r in f[f.bias_atr.gt(2)].to_dict("records")}
    ix={d:i for i,d in enumerate(cal)}
    rows=[]
    for w in waves.itertuples(index=False):
        i=ix[w.start]
        # Choose earliest archived acceptance, never the best-priced or winning one.
        hits=[by[w.symbol,d] for d in cal[i-6:i+3] if (w.symbol,d) in by]
        if not hits:
            rows.append(dict(symbol=w.symbol,start=w.start,status="UNMATCHED_ARCHIVE"))
            continue
        r={**hits[0], "start":w.start, "oracle_open":w.oracle_open,"oracle_target":w.oracle_target,
           "acceptances_in_window":len(hits)}
        for k in ("date","attempt","pulse","entry_open_day"):
            r[f"{k}_lag"]=ix[r[k]]-i if isinstance(r.get(k),str) and r[k] in ix else None
        rows.append(r)
    return pd.DataFrame(rows)


def matched_near(f, fail_only=False):
    g=f[f.mature & f.status.eq("CLOSED") & f.amount20.gt(0)]
    rows=[]
    for (d,b),x in g.groupby(["date","board"]):
        near=x[x.band.eq("2_to_2.5")].sort_values(["amount20","symbol"])
        controls=x[x.band.eq("1.5_to_2") | (x.bias_atr.gt(2) if fail_only else False)]
        if fail_only:
            controls=controls[controls.net.le(0)]
        used=set()
        for r in near.itertuples(index=False):
            if fail_only and r.net<=0:
                continue
            eligible=controls[controls.amount20.between(.5*r.amount20,2*r.amount20)
                              & ~controls.symbol.isin(used|{r.symbol})].copy()
            if eligible.empty:
                continue
            eligible["amount_distance"]=(eligible.amount20/r.amount20).map(np.log).abs()
            peer=next(eligible.sort_values(["amount_distance","symbol"]).itertuples(index=False))
            used.add(peer.symbol)
            rows.append(dict(date=d,board=b,market=r.market,near=r.symbol,control=peer.symbol,
                             near_net=r.net,control_net=peer.net,net_delta=r.net-peer.net,
                             near_risk=r.risk_pct,control_risk=peer.risk_pct,near_bias=r.bias_atr,
                             control_bias=peer.bias_atr,near_open_bias=r.opening_bias,control_open_bias=peer.opening_bias,
                             near_pulse_delay=r.pulse_delay,control_pulse_delay=peer.pulse_delay,
                             near_gap=r.opening_gap_atr,control_gap=peer.opening_gap_atr))
    return pd.DataFrame(rows)


def wave_failure_pairs(f,waves):
    """Same-band failed executions, for morphology only (chosen by observed outcome)."""
    failed=f[f.mature & f.status.eq("CLOSED") & f.net.le(0)].groupby(["date","board","band"])
    ids=set(zip(waves.symbol,waves.date))
    used=set()
    rows=[]
    for r in waves[waves.status.eq("CLOSED")].sort_values(["date","symbol"]).itertuples(index=False):
        key=r.date,r.board,r.band
        if key not in failed.groups:
            continue
        peers=failed.get_group(key)
        peers=peers[peers.amount20.between(.5*r.amount20,2*r.amount20)]
        options=[(abs(np.log(t.amount20/r.amount20)),t.symbol,t)
                 for t in peers.itertuples(index=False) if (t.symbol,t.date) not in ids|used]
        if not options:
            continue
        peer=min(options,key=lambda x:x[:2])[2]
        used.add((peer.symbol,peer.date))
        fields=("bias_atr","opening_bias","opening_gap_atr","risk_pct","acceptance_delay","pulse_delay","net")
        rows.append(dict(date=r.date,board=r.board,band=r.band,market=r.market,symbol=r.symbol,peer=peer.symbol,
                         **{prefix+k:getattr(t,k) for prefix,t in (("wave_",r),("peer_",peer)) for k in fields}))
    return pd.DataFrame(rows)


def pulse_open_diagnostic(waves,cal,limits):
    """Price paid for waiting on linked accepted waves, never an early-entry backtest."""
    pos={d:i for i,d in enumerate(cal)}
    rows=[]
    pulse=waves.pulse.fillna("")
    candidates=waves[waves.opened.eq(True)&pulse.ne("")&pulse.lt(waves.date)]
    for s,g in candidates.groupby("symbol"):
        a=read_raw(s).set_index("date")
        # Use the same cumulative return scale so corporate-action nominal gaps are not price improvement.
        r=a.pct_chg.div(100).fillna(a.close.pct_change()).fillna(0).clip(-.95,3.)
        ix=a.open/a.close*np.exp(np.log1p(r).cumsum())
        for w in g.itertuples(index=False):
            d=cal[pos[w.pulse]+1]
            state="MISSING_PULSE_NEXT_OPEN"
            delta=None
            if d in a.index and w.entry_open_day in a.index:
                row=a.loc[d]
                upper,_,_=band(s,SimpleNamespace(day=d,preclose=row.preclose,isST=row.isST),limits)
                state=("NO_TRADE_AT_PULSE_OPEN" if row.tradestatus!=1 or not row.amount>0 or not row.open>0
                       else "PULSE_LIMIT_UNVERIFIED" if not np.isfinite(upper)
                       else "PULSE_OPEN_LIMIT_UP" if row.open>=upper else "COMPARABLE_OPEN")
                if state=="COMPARABLE_OPEN" and ix[d]>0:
                    delta=float(100*(ix[w.entry_open_day]/ix[d]-1))
            rows.append(dict(symbol=s,start=w.start,pulse=w.pulse,accepted=w.date,pulse_open_date=d,
                             accepted_open_date=w.entry_open_day,status=state,open_cost_delta_pct=delta))
    return pd.DataFrame(rows)


def pair_intervals(pairs,cal):
    """20-session block bootstrap; do not treat several stocks on the same date as IID."""
    rows=[]
    rng=np.random.default_rng(20261119)
    for label,g in (("all_markets",pairs),("green_only",pairs[pairs.market.eq("GREEN")])):
        for year,x in g.groupby(g.date.str[:4]):
            days=[d for d in cal if x.date.min()<=d<=x.date.max()]
            block=min(20,len(days))
            y=x.groupby("date").net_delta.agg(["sum","size"]).reindex(days,fill_value=0)
            starts=rng.integers(len(days),size=(3000,int(np.ceil(len(days)/block))))
            take=((starts[:,:,None]+np.arange(block))%len(days)).reshape(3000,-1)[:,:len(days)]
            sims=y["sum"].to_numpy()[take].sum(axis=1)/np.maximum(1,y["size"].to_numpy()[take].sum(axis=1))
            low,high=np.quantile(sims,[.025,.975])
            rows.append(dict(sample=label,year=year,n=len(x),days=x.date.nunique(),mean=x.net_delta.mean(),
                             median=x.net_delta.median(),lower95=low,upper95=high,
                             near_better=x.net_delta.gt(0).mean()*100))
    return pd.DataFrame(rows)


def supplements(out, f, audit, pairs, cal, limits):
    wave_failure_pairs(f,audit).to_csv(out/"wave_failure_pairs.csv",index=False)
    pulse_open_diagnostic(audit,cal,limits).to_csv(out/"pulse_open_timing.csv",index=False)
    pair_intervals(pairs,cal).to_csv(out/"pair_intervals.csv",index=False)


def run(out,workers):
    out=Path(out)
    if out.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("research output must not modify daily artifacts")
    out.mkdir(parents=True,exist_ok=False)
    cal=tuple(d for d in sessions() if d<=END)
    scope=json.loads((BASE/"manifest.json").read_text(encoding="utf-8"))["scope_days"]
    e=load_events(scope)
    jobs=[(s,g.to_dict("records")) for s,g in e.groupby("symbol")]
    limits=load_limits([d for d in cal if d>="2025-01-01"])
    results=[]
    with ProcessPoolExecutor(workers,initializer=initialize,initargs=(cal,limits)) as p:
        for i,rows in enumerate(p.map(replay,jobs,chunksize=6),1):
            results.extend(rows)
            if i%500==0:
                print(f"[r1_extension] {i}/{len(jobs)} stock histories",flush=True)
    f=pd.DataFrame(results)
    f["opened"]=f.opened.eq(True)
    f["market"]=f.date.map(market_flags(cal)).fillna("UNKNOWN")
    f.to_csv(out/"events.csv.gz",index=False)
    audit=linked_waves(f,BASE,cal)
    audit.to_csv(out/"filtered_439.csv",index=False,encoding="utf-8-sig")
    pairs=matched_near(f)
    pairs.to_csv(out/"near_boundary_pairs.csv",index=False)
    matched_near(f,fail_only=True).to_csv(out/"failure_shape_pairs.csv",index=False)
    stats=compare_band(f)
    stats.to_csv(out/"summary.csv",index=False)
    supplements(out,f,audit,pairs,cal,limits)
    (out/"protocol.json").write_text(json.dumps(dict(version=VERSION,end=END,baseline=str(BASE),stocks=len(jobs),
        events=len(f),wave_labels=len(audit),orders_enabled=False,bands=BUCKETS,
        execution="mainwave_path structure36_20, conservative opening limits, 1 ATR gap, 30bp",
        pairing="same date / listing board, 20-session median amount within 2x, nearest in log amount; no PIT industry claim",
        pulse="first demand expansion from breakout attempt through acceptance; timestamp diagnostic, not a retroactive fill"),
        ensure_ascii=False,indent=2),encoding="utf-8")
    print(stats[["selection","year","band","mature","opened","closed","mean","median","p10"]].round(3).to_string(index=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--workers",default=9,type=int)
    args=p.parse_args()
    run(args.out,args.workers)
