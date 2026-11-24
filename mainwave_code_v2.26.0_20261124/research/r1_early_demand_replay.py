"""Compare pre-acceptance demand with accepted_bias2, from the complete attempt cohort."""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mainwave_engine import breakout_events as BE, route_evidence as RE
from research import route_clock as CLOCK
from research.mainwave_path import AC, ROOT, limit_bar, market_flags, portfolio, read_raw, sessions
from research.r2_pool_compare import band, load_limits
from research.r2_pool_history import SYMBOL
from research.route_followup import arrays, enrich, simulate


VERSION = "r1_first_demand_vs_acceptance_v1_1"
ARMS = ("early", "wait")
PROTOCOL = {
    "version": VERSION,
    "origin": "Every price ATTEMPT at its timestamp, including ineligible, rejected, expired and unconfirmed",
    "early": "First DEMAND_EXPANSION on an active ATTEMPT day with eligible, no supply alert and bias <= 2",
    "wait": "The same attempt's ACCEPTED day only if eligible and accepted_bias2",
    "execution": "Both next verified open; refs where available; <=1 ATR opening gap; same structure36_20 / T+1 / conservative downside lock; 30/40 bp",
    "budget": "Daily at most two picks by bias_atr then symbol on decision day, no substitution; independent annual NEW100K research books",
    "diagnostic": "Same-wave offsets, both-filled net differences, extra trades including failed and expired attempts",
    "maturity": "Attempt + acceptance TTL + 1 + 20 observed market sessions; unresolved executions separate",
    "missing_data": "Explicit tradestatus=0 with blank amount is suspended; missing or unverified bars stay unknown",
    "orders_enabled": False,
}


def attempt_plan(a, i, cal):
    """Decisions are made by looking only at the current bar on each day of a live attempt."""
    chain=a["breakout_id"][i]
    early=wait=first_pulse=None
    verdict="CENSORED_ATTEMPT"
    wait_note="NO_ACCEPTANCE"
    terminal_day=""
    for j in range(i,len(a["day"])):
        if j>i:
            gap=CLOCK.classify_gap(a["day"][j-1],a["day"][j],cal,a.get("suspended_days",[]))
            if gap["status"] in ("UNEXPLAINED_SESSION_GAP","CALENDAR_UNVERIFIED"):
                verdict="ATTEMPT_DATA_GAP"
                break
        if a["breakout_id"][j]!=chain:
            verdict="ATTEMPT_ID_DISCONTINUITY"
            break
        state=a["breakout_status"][j]
        if state in ("REJECTED","EXPIRED"):
            verdict,terminal_day=state,a["day"][j]
            break
        if state=="ACCEPTED":
            verdict,terminal_day=state,a["day"][j]
            if a["eligible"][j] and a["accepted_bias2"][j]:
                wait,wait_note=j,"SIGNAL"
            else:
                wait_note=("INELIGIBLE_AT_ACCEPTANCE" if not a["eligible"][j] else
                           "NO_DEMAND_OR_SUPPLY_RISK" if not a["accepted_demand"][j] else "BIAS_ABOVE_2")
            break
        if state!="ATTEMPT":
            verdict="ATTEMPT_ID_DISCONTINUITY"
            break
        if a["effort_result"][j]!="DEMAND_EXPANSION":
            continue
        if first_pulse is None:
            first_pulse=j
        if early is None and a["eligible"][j] and not a["supply_warning"][j] and a["bias_atr"][j]<=2:
            early=j
    early_note=("SIGNAL" if early is not None else "NO_PRE_ACCEPTANCE_PULSE"
                if first_pulse is None else "PULSE_NOT_ELIGIBLE_OR_EXTENDED")
    if verdict in ("CENSORED_ATTEMPT","ATTEMPT_DATA_GAP","ATTEMPT_ID_DISCONTINUITY"):
        if early is None:
            early_note=verdict
        if wait is None:
            wait_note=verdict
    return early,wait,dict(terminal=verdict,terminal_day=terminal_day,
                           early_note=early_note,wait_note=wait_note,
                           first_pulse=a["day"][first_pulse] if first_pulse is not None else "")


def execution(a,i,s,cal,limits):
    j=i+1
    if j>=len(a["day"]):
        return {"status":"CENSORED"}
    raw=float(a["open"][j])
    up,_,src=band(s,limit_bar(a["day"][j],a["preclose"][j],a["isST"][j]),limits)
    if not np.isfinite(up):
        return {"status":"ENTRY_LIMIT_UNKNOWN"}
    if raw>=up:
        return {"status":"ENTRY_LIMIT_UP"}
    tr=[]
    result=simulate(a,i,s,"structure36_20",gap_guard=True,trace=tr,calendar=cal,conservative_limits=limits)
    if not tr:
        return result
    opening=tr[0]
    risk=(opening["entry_index"]-opening["initial_stop"])/opening["entry_index"]
    minlot=200 if s.startswith("sh_68") else 100
    if minlot*raw>AC["capital"]*AC["symbol_cap_pct"]/100 or minlot*raw*risk>AC["capital"]*AC["single_risk_pct"]/100:
        return {"status":"RISK_OR_LOT"}
    return {**result,"path":tr,"raw_open":raw,"filled_day":opening["date"],"index_open":opening["entry_index"],
            "initial_risk":100*risk,"limit_source":src}


def init(cal,start,end,limits):
    global CAL, START, END, LIMITS
    CAL,START,END,LIMITS=cal,start,end,limits


def normalize_suspensions(raw):
    raw=raw.copy()
    flag=pd.to_numeric(raw.tradestatus,errors="coerce")
    raw.loc[flag.eq(0)&raw.amount.isna(),"amount"]=0
    return raw


def worker(s):
    raw=read_raw(s)
    f=RE.build(normalize_suspensions(raw[raw.date.le(END)]),s)
    if f.empty:
        return [],[]
    a=arrays(enrich(f))
    idx=np.flatnonzero((a["breakout_event"]=="ATTEMPT")&(a["day"]>=START)&(a["day"]<=END))
    result=[]; trades=[]
    for i in idx:
        d=a["day"][i]
        early,wait,info=attempt_plan(a,i,CAL)
        row=dict(symbol=s,attempt=d,cycle_id=a["breakout_id"][i],eligible_at_attempt=bool(a["eligible"][i]),
                 mature=bisect_left(CAL,d)+BE.ACCEPT_TTL+1+20<len(CAL),**info)
        for arm,j in zip(ARMS,(early,wait)):
            if j is None:
                row[f"{arm}_signal"]=""
                row[f"{arm}_status"]=info[f"{arm}_note"]
                continue
            sig=a["day"][j]
            tr=execution(a,j,s,CAL,LIMITS)
            row.update({f"{arm}_{k}":tr.get(k) for k in
                        ("status","net","reason","filled_day","raw_open","index_open","initial_risk","exit_date")})
            row[f"{arm}_signal"]=sig
            row[f"{arm}_bias"]=float(a["bias_atr"][j])
            trades.append((arm,s,sig,tr))
        result.append(row)
    return result,trades


def observe_top2(cases,arm):
    signal=f"{arm}_signal"
    f=cases.loc[cases[signal].ne(""),["symbol","attempt",signal,f"{arm}_bias"]].sort_values(
        [signal,f"{arm}_bias","symbol"])
    return {d:[dict(symbol=r[0],attempt=r[1],date=d) for r in g.head(2).itertuples(index=False,name=None)]
            for d,g in f.groupby(signal)}


def event_summary(cases,states):
    rows=[]; pairs=[]
    for year,f in cases[cases.mature].groupby(cases.attempt.str[:4]):
        for label in ("all_markets","green_only"):
            for arm in ARMS:
                sig=f[f[f"{arm}_signal"].ne("")]
                if label=="green_only":
                    sig=sig[sig[f"{arm}_signal"].map(states).eq("GREEN")]
                closed=sig[sig[f"{arm}_status"].eq("CLOSED")]
                net=closed[f"{arm}_net"]
                rows.append(dict(year=year,sample=label,arm=arm,attempts=len(f),triggered=len(sig),
                                 opened=int(sig[f"{arm}_filled_day"].notna().sum()),closed=len(closed),
                                 days=closed[f"{arm}_signal"].nunique(),mean=net.mean(),median=net.median(),
                                 win_pct=100*net.gt(0).mean(),p10=net.quantile(.1),
                                 rejected_attempts=int(closed.terminal.isin(["REJECTED","EXPIRED"]).sum()),
                                 statuses=sig[f"{arm}_status"].value_counts().to_dict()))
            b=f[f.early_status.eq("CLOSED")&f.wait_status.eq("CLOSED")].copy()
            if label=="green_only":
                b=b[b.early_signal.map(states).eq("GREEN") & b.wait_signal.map(states).eq("GREEN")]
            b["open_price_saved_pct"]=(b.wait_index_open/b.early_index_open-1)*100
            b["early_net_delta"]=b.early_net-b.wait_net
            pairs.append(b.assign(year=year,sample=label))
    return pd.DataFrame(rows),pd.concat(pairs,ignore_index=True)


def attribution(cases):
    """All mature attempts, settled-component arithmetic (not a trading policy)."""
    rows=[]
    for year,g in cases[cases.mature].groupby(cases.attempt.str[:4]):
        ae=g.early_status.eq("CLOSED")&g.early_market.eq("GREEN")
        bw=g.wait_status.eq("CLOSED")&g.wait_market.eq("GREEN")
        failure=g.terminal.isin(["REJECTED","EXPIRED"])
        buckets=[("both",ae&bw),("early_only_rejected_or_expired",ae&~bw&failure),
                 ("early_only_other",ae&~bw&~failure),("wait_only",~ae&bw),("neither",~ae&~bw)]
        for label,idx in buckets:
            a_net=g.loc[idx&ae,"early_net"]
            b_net=g.loc[idx&bw,"wait_net"]
            rows.append(dict(year=year,group=label,attempts=int(idx.sum()),early_closed=len(a_net),wait_closed=len(b_net),
                             early_mean=a_net.mean(),wait_mean=b_net.mean(),early_sum=a_net.sum(),wait_sum=b_net.sum(),
                             contribution_per_attempt_pct=(a_net.sum()-b_net.sum())/len(g)))
    return pd.DataFrame(rows)


def paired_uncertainty(pairs, cal):
    from research.r1_extension_audit import pair_intervals
    q=pairs.rename(columns={"attempt":"date","early_net_delta":"net_delta"})
    # Label only already green pairs in the input to the GREEN arm of the shared bootstrap.
    all_rows=q[q["sample"]=="all_markets"].copy()
    green=set(q.loc[q["sample"]=="green_only","cycle_id"])
    all_rows["market"]=np.where(all_rows.cycle_id.isin(green),"GREEN","OTHER")
    return pair_intervals(all_rows,cal)


def run(start,end,out,workers):
    out=Path(out)
    if out.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("research output must not write to official archives")
    cal=tuple(d for d in sessions() if d<=end)
    if start not in cal or end!=cal[-1] or start>=end:
        raise ValueError("use observed market dates in increasing order")
    out.mkdir(parents=True,exist_ok=False)
    symbols=[p.name[:-10] for p in sorted(Path("data").glob("*_daily.csv")) if SYMBOL.fullmatch(p.name[:-10])]
    protocol={**PROTOCOL,"start":start,"end":end,"symbols":len(symbols)}
    (out/"protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding="utf-8")
    states=market_flags(cal)
    limits=load_limits(d for d in cal if d>=start)
    records=[]; trades={k:{} for k in ARMS}
    with ProcessPoolExecutor(workers,initializer=init,initargs=(cal,start,end,limits)) as p:
        for i,(rows,paths) in enumerate(p.map(worker,symbols,chunksize=8),1):
            records.extend(rows)
            for arm,s,d,tr in paths:
                key=(s,d)
                if key in trades[arm]:
                    raise ValueError(f"duplicate trade signal {arm} {key}")
                trades[arm][key]=tr
            if i%500==0:
                print(f"[first_demand] {i}/{len(symbols)} stocks",flush=True)
    f=pd.DataFrame(records)
    if f.cycle_id.duplicated().any():
        raise ValueError("attempt identity is not unique")
    for arm in ARMS:
        f[f"{arm}_market"]=f[f"{arm}_signal"].map(states).fillna("UNKNOWN")
    f.to_csv(out/"attempts.csv.gz",index=False)
    es,pairs=event_summary(f,states)
    es.to_csv(out/"event_summary.csv",index=False)
    pairs.to_csv(out/"same_wave_pairs.csv",index=False)
    attribution(f).to_csv(out/"attempt_attribution.csv",index=False)
    paired_uncertainty(pairs,cal).to_csv(out/"paired_uncertainty.csv",index=False)
    summary=[]
    for year in sorted({d[:4] for d in cal if d>=start}):
        days=tuple(d for d in cal if d>=start and d.startswith(year))
        cohort=f[f.attempt.str.startswith(year)]
        for arm in ARMS:
            picks=observe_top2(cohort,arm)
            for fee in (.003,.004):
                n,t=portfolio(days,picks,states,trades[arm],fee)
                tag=f"{year}_{arm}_{int(fee*10000)}bp"
                n.to_csv(out/f"nav_{tag}.csv",index=False)
                t.to_csv(out/f"ledger_{tag}.csv",index=False)
                c=t[t.status.eq("CLOSED")] if not t.empty else t
                pnl=c.get("pnl",pd.Series(dtype=float))
                summary.append(dict(year=year,arm=arm,fees_bp=int(fee*10000),attempts=len(cohort),
                                    observations=sum(len(picks.get(d,[])) for d in days),max_positions=int(n.open.max()),
                                    closed=len(c),net_pct=100*(n.equity.iat[-1]/AC["capital"]-1),
                                    drawdown_pct=n.drawdown_pct.min(),profit_factor=pnl[pnl>0].sum()/abs(pnl[pnl<0].sum())
                                    if (pnl<0).any() else None,
                                    net_without_best_pct=100*((n.equity.iat[-1]-max(0,pnl.max()))/AC["capital"]-1),
                                    statuses=t.status.value_counts().to_dict() if not t.empty else {}))
    p=pd.DataFrame(summary)
    p.to_csv(out/"book_summary.csv",index=False)
    print(es.drop(columns="statuses").round(3).to_string(index=False))
    print(p.drop(columns="statuses").round(3).to_string(index=False))


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start",default="2025-01-02")
    ap.add_argument("--end",required=True)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--workers",type=int,default=10)
    a=ap.parse_args()
    run(a.start,a.end,a.out,a.workers)
