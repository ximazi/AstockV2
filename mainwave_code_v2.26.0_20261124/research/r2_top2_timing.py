"""Research only: fixed R2 daily attention budget, paired close signals and next-open fills."""
import argparse
from collections import defaultdict
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mainwave_engine import preparation_detector as PD, stock_lifecycle as LC
from research.r2_pool_compare import HORIZONS, executable, load_limits
from research.r2_pool_history import ROOT, SYMBOL


BASE = Path("results/research/r2_pool_comparison/20261117_v1")
FIRINGS = Path("results/research/preparation_detector/20260923_102746/firings.csv")
WINDOW = 10
COLS = ["breakout_event", "breakout_id", "breakout_status", "setup_kind", "setup_status",
        "setup_first_known", "setup_confirmed_at", "setup_breakout_id", "setup_breakout_accepted_at",
        "setup_turnover_contraction"]


def budget(early, strict):
    """Counterfactual single-pool budgets and one deployable two-slot observation list."""
    mixed = [("early",early[0])] if early else []
    other = next((s for s in strict if s not in early[:1]), None)
    if other:
        mixed.append(("strict",other))
    return {"early2":[("early",s) for s in early[:2]], "strict2":[("strict",s) for s in strict[:2]],
            "one_each":mixed}


def matched(s, candidates, used, industry, amount):
    def board(t):
        return t[:5] if t.startswith(("sh_68","sz_30")) else "main"
    v = amount.get(s, np.nan)
    name = industry.get(s,"")
    if not v > 0 or name in ("", "未确认", "未知"):
        return None
    near = [(abs(np.log(amount[t]/v)),t) for t in candidates if t not in used and board(t)==board(s)
            and industry.get(t)==name and .5 <= amount.get(t,np.nan)/v <= 2]
    return min(near)[1] if near else None


def archive_picks(days):
    frozen = pd.read_csv(BASE/"pools.csv.gz").groupby(["day","pool"]).symbol.agg(set).to_dict()
    sources = json.loads((BASE/"inputs.json").read_text())["sources"]
    price = pd.read_csv(BASE/"prices.csv.gz",usecols=["day","symbol","amount"])
    amount = price.pivot(index="day",columns="symbol",values="amount")
    liquidity = amount.where(amount>0).rolling(5,min_periods=5).median()
    picks=[]
    for i,d in enumerate(days):
        if not i or any(sources.get(t)!="archive" for t in (days[i-1],d)):
            continue
        tables = {}
        for p,file in [("early","preparation_watch.csv"),("strict","carrier_qualification.csv")]:
            table=pd.read_csv(ROOT/d.replace("-","")/file, dtype=str,keep_default_na=False)
            if set(table.as_of)-{d}:
                raise ValueError(f"mixed dates in {file}: {d}")
            if p=="strict":
                table=table[(table.data_fresh=="True") & (table.hard_gate_pass=="True")
                            & (table.carrier_qualification=="QUALIFIED_LURK")]
            if set(table.symbol)!=frozen.get((d,p),set()) or table.symbol.duplicated().any():
                raise ValueError(f"changed pool: {d} {p}")
            tables[p]=table[~table.symbol.isin(frozen.get((days[i-1],p),set()))]
        stocks={p:t.symbol.tolist() for p,t in tables.items()}
        for policy,selection in budget(stocks["early"],stocks["strict"]).items():
            used={s for _,s in selection}
            for p,s in selection:
                picks.append((policy,d,s,p,s))
                peer=matched(s,stocks[p],used,tables[p].set_index("symbol").industry,liquidity.loc[d])
                if peer:
                    picks.append((policy+"_peer",d,peer,p,s))
                    used.add(peer)
    return picks


def older_picks(days):
    """Use archived P5 firings, never their outcome columns; restore original excess20 rank."""
    f=pd.read_csv(FIRINGS,usecols=["symbol","date","detector"])
    f=f[f.detector.eq(PD.CHOSEN) & f.symbol.str.fullmatch(SYMBOL)]
    bench=PD.benchmark_ret20()
    records=[]
    resets={next(d for d in days if d.startswith(year)) for year in f.date.str[:4].unique()}
    for s,g in f.groupby("symbol"):
        path=Path("data")/(s+"_daily.csv")
        if not path.exists():
            raise ValueError(f"missing ranking bars: {s}")
        a=pd.read_csv(path,usecols=["date","amount","close","pct_chg","open","high","low"]).sort_values("date")
        a=a.drop_duplicates("date",keep="last").dropna(subset=["open","high","low","close"])
        a=a[a.amount.gt(0)].set_index("date")
        changes=(a.pct_chg/100).fillna(a.close.pct_change()).fillna(0).clip(-.95,3.)
        rs=(np.expm1(np.log1p(changes).rolling(20).sum())*100-bench).reindex(g.date)
        liq=a.amount.reindex(days).rolling(5).median().reindex(g.date)
        records.extend((d,s,round(float(v),2) if pd.notna(v) else -999.,liq.at[d])
                       for d,v in rs.items() if d not in resets)
    a=pd.DataFrame(records,columns=["day","symbol","rs","liq"]).sort_values(["day","rs","symbol"],ascending=[True,False,True])
    out=[]
    for d,g in a.groupby("day"):
        top=g.head(2).symbol.tolist()
        used=set(top)
        amt=g.set_index("symbol").liq.to_dict()
        # No historical industry membership file. Match board/liquidity only, not today's industry.
        unknown=dict.fromkeys(amt,"BOARD_ONLY")
        for s in top:
            out.append(("early2_history",d,s,"early",s))
            peer=matched(s,amt,used,unknown,amt)
            if peer:
                out.append(("early2_history_peer",d,peer,"early",s))
                used.add(peer)
    return out


def replay_one(args):
    s, lo, end = args
    # Price-only, full history from first bar; no later archived engine states injected.
    _, t=LC.analyze(s,end)
    if t.empty:
        return s,[]
    return s,list(t.loc[t.date.ge(lo),["date",*COLS,"breakout_accepted_at"]].itertuples(index=False,name=None))


def archive_evidence(days, stocks):
    result=defaultdict(dict)
    for d in days:
        folder=ROOT/d.replace("-","")
        if not (folder/"route_scan_audit.csv").exists():
            continue
        a=pd.read_csv(folder/"route_scan_audit.csv",usecols=["symbol","date","data_status",*COLS[:3]],keep_default_na=False)
        a=a[a.symbol.isin(stocks)&a.date.eq(d)&a.data_status.eq("FRESH")].set_index("symbol")
        l=pd.DataFrame()
        if (folder/"stock_lifecycle.csv").exists():
            l=pd.read_csv(folder/"stock_lifecycle.csv",usecols=["symbol","date","cycle_data_status",*COLS[3:]],keep_default_na=False)
            l=l[l.date.eq(d)&l.cycle_data_status.eq("FRESH")].set_index("symbol")
        for s,r in a.iterrows():
            result[s][d]={**r.to_dict(),**(l.loc[s].to_dict() if s in l.index else {} )}
    return result


def known(x):
    return str(x).lower()=="true"


def watch(s, origin, days, data, extension=0, *, incumbent=None):
    """Both children start at the same discovery. Never attach a later break to the first one."""
    no=lambda state: (state, "")
    active,accepted=incumbent or ("","")
    end=origin+WINDOW
    for j in range(origin+1,min(len(days),origin+WINDOW+extension+1)):
        if j>end:
            break
        d=days[j]
        r=data.get(s,{}).get(d)
        if r is None:
            return (("TRIGGER",accepted) if accepted else no("DATA_WAIT")),no("DATA_WAIT")
        if not accepted:
            if r["breakout_event"]!="ACCEPTED" or not r["breakout_id"]:
                continue
            accepted=d
            active=r["breakout_id"]
            end=(j+extension) if extension else end
            continue
        if r["breakout_id"]!=active or r["breakout_status"] in ("FAILED","REJECTED","EXPIRED"):
            return ("TRIGGER",accepted),no("BROKEN_CHAIN")
        if "setup_status" not in r:
            return ("TRIGGER",accepted),no("DATA_WAIT")
        if (r["setup_kind"]=="FIRST_PULLBACK" and known(r["setup_first_known"])
                and r["setup_breakout_id"]==active and r["setup_breakout_accepted_at"]==accepted):
            if r["setup_status"] in ("FAILED","EXPIRED"):
                return ("TRIGGER",accepted),no("FAILED_FIRST_PULLBACK")
            dry=pd.to_numeric(r["setup_turnover_contraction"],errors="coerce")
            if r["setup_status"]=="CONFIRMED" and r["setup_confirmed_at"]==d and 0<dry<=1:
                return ("TRIGGER",accepted),("TRIGGER",d)
    incomplete=end>=len(days)
    a=("TRIGGER",accepted) if accepted else no("PENDING" if incomplete else "NO_ACCEPTANCE")
    b=no("PENDING" if incomplete else ("NO_FIRST_PULLBACK" if accepted else "NO_ACCEPTANCE"))
    return a,b


def evaluate(picks, days, feed, basis, bars, limits, extension):
    pos={d:i for i,d in enumerate(days)}
    rows=[]
    seen=set()
    for policy,d,s,p,parent in sorted(picks,key=lambda x:(x[1],x[0],x[2])):
        i=pos[d]
        signals=watch(s,i,days,feed,extension)
        for arm,(status,at) in zip(("accept","pullback"),signals):
            key=(policy,s,arm,at)
            if at and key in seen:
                status="DUPLICATE_TRIGGER"
            if at:
                seen.add(key)
            for h in HORIZONS:
                fill=executable(s,pos[at],h,days,bars[s],limits) if status=="TRIGGER" else {}
                rows.append(dict(policy=policy,origin=d,symbol=s,pool=p,matched_to=parent,basis=basis,extra=extension,
                                 h=h,arm=arm,mature=i+WINDOW+extension+1+h<len(days),signal=at,
                                 watch_state=status,status=fill.pop("status",status),**fill))
    return rows


def summarize(df):
    keys=["basis","policy","extra","h"]
    g=df[df.mature].copy()
    out=[]
    for k,d in g.groupby(keys):
        a,b=(d[d.arm.eq(t)].set_index(["origin","symbol"]) for t in ("accept","pullback"))
        both=a.net_pct.notna() & b.net_pct.notna()
        diff=(b.net_pct-a.net_pct)[both]
        for name,x in [("accept",a),("pullback",b)]:
            net=x.net_pct.dropna()
            idle=x.status.isin(["NO_ACCEPTANCE","BROKEN_CHAIN","FAILED_FIRST_PULLBACK","NO_FIRST_PULLBACK",
                                "DUPLICATE_TRIGGER","ST_OR_LIMIT_UP","ENTRY_SUSPENDED","BUDGET_TOO_SMALL"])
            valid=len(net)+int(idle.sum())
            out.append(dict(zip(keys,k),arm=name,days=x.reset_index().origin.nunique(),origins=len(x),
                        triggered=int(x.watch_state.eq("TRIGGER").sum()),closed=len(net),idle=int(idle.sum()),
                        unknown=len(x)-valid,net_mean=net.mean(),median=net.median(),win_pct=100*(net>0).mean(),
                        opportunity_mean=net.sum()/valid if valid else np.nan,paired_n=len(diff),
                        paired_delta=diff.mean(),paired_better=int(diff.gt(0).sum()),paired_worse=int(diff.lt(0).sum())))
    return pd.DataFrame(out)


def discovery(picks, days):
    """Older P5 cohort's original TREND labels; never used by the selection stage."""
    targets=defaultdict(set)
    periods={"2025":"2025-12-31","2026":"2026-04-30"}
    for year,end in periods.items():
        e=pd.read_csv(f"results/mainwave_episodes_{year}.csv",usecols=["stem","start","cls"])
        for r in e[e.cls.eq("TREND") & e.start.between(year+"-01-01",end)].itertuples():
            targets[r.stem].add(r.start)
    pos={d:i for i,d in enumerate(days)}
    records=[]
    for policy,d,s,_,parent in picks:
        i=pos[d]
        for h in HORIZONS:
            w=days[i+1:i+1+h]
            if len(w)==h and w[-1]<=periods[d[:4]]:
                hit=next((k for k in w if k in targets[s]),"")
                records.append(dict(policy=policy,origin=d,symbol=s,matched_to=parent,h=h,target=hit,
                                    lead=w.index(hit)+1 if hit else None))
    return pd.DataFrame(records)


def tracked_pairs(history, days, tape, bars, limits):
    """Mechanism check, not Top2 returns: every later acceptance after a first historical Top2 discovery."""
    pos={d:i for i,d in enumerate(days)}
    by=defaultdict(list)
    for policy,d,s,_,_ in history:
        if policy=="early2_history":
            by[s].append(d)
    rows=[]
    last=max(d for origins in by.values() for d in origins)
    for s,origins in by.items():
        origins.sort()
        for d,r in tape[s].items():
            if d>last or r["breakout_event"]!="ACCEPTED":
                continue
            k=bisect_left(origins,d)-1
            if k<0:
                continue
            i=pos[d]
            _,(state,b)=watch(s,i-1,days,tape,20)
            for h in HORIZONS:
                if i+20+1+h>=len(days):
                    continue
                a=executable(s,i,h,days,bars[s],limits)
                f=executable(s,pos[b],h,days,bars[s],limits) if b else {}
                rows.append(dict(symbol=s,origin=origins[k],accepted=d,lag=i-pos[origins[k]],pullback=b,
                                 watch_state=state,h=h,accept_status=a["status"],pull_status=f.get("status",state),
                                 a=a["net_pct"],b=f.get("net_pct"),a_entry=a["entry_day"],a_buy=a["buy"],
                                 a_exit=a["exit_day"],a_sell=a["sell"],b_entry=f.get("entry_day"),b_buy=f.get("buy"),
                                 b_exit=f.get("exit_day"),b_sell=f.get("sell")))
    return pd.DataFrame(rows)


def existing_chains(picks, days, tape, bars, limits):
    pos={d:i for i,d in enumerate(days)}
    rows=[]
    seen=set()
    for policy,d,s,p,_ in sorted(picks,key=lambda r:r[1]):
        r=tape[s].get(d,{})
        at=r.get("breakout_accepted_at","")
        if r.get("breakout_status") not in ("ACCEPTED","ADVANCING") or not at or at>d:
            continue
        i=pos[d]
        _,(state,signal)=watch(s,i,days,tape,incumbent=(r["breakout_id"],at))
        if signal and (policy,s,signal) in seen:
            state="DUPLICATE_TRIGGER"
        if signal:
            seen.add((policy,s,signal))
        for h in HORIZONS:
            fill=executable(s,pos[signal],h,days,bars[s],limits) if state=="TRIGGER" else {}
            rows.append(dict(policy=policy,origin=d,symbol=s,pool=p,previous_accept=at,
                             age=i-pos[at] if at in pos else None, predates_calendar=at not in pos,
                             confirmed=signal,watch_state=state,mature=i+WINDOW+1+h<len(days),h=h,
                             status=fill.pop("status",state),**fill))
    return pd.DataFrame(rows)


def prices(stocks,days,meta):
    frozen=pd.read_csv(BASE/"prices.csv.gz")
    frozen_groups={s:g for s,g in frozen.groupby("symbol")}
    bars={}
    for s in stocks:
        a=pd.read_csv(Path("data")/(s+"_daily.csv")).rename(columns={"date":"day"})
        a=a[a.day.between(days[0],days[-1]) & a.day.lt(meta["calendar"][0])]
        bars[s]={r.day:r for r in a.itertuples(index=False)}
        if s in frozen_groups:
            bars[s].update({r.day:r for r in frozen_groups[s].itertuples(index=False)})
    return bars


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--workers",default=8,type=int)
    parser.add_argument("--existing-only",action="store_true",help="Append only the diagnostic to a finished study")
    args=parser.parse_args()
    meta=json.loads((BASE/"inputs.json").read_text())
    days=list(PD.benchmark_ret20().index)
    days=[d for d in days if "2025-01-01"<=d<=meta["as_of"]]
    if args.existing_only:
        if (args.out/"existing_chain_followup.csv.gz").exists() or not (args.out/"summary.csv").exists():
            raise ValueError("supplement requires finished study without existing diagnostic")
        a=pd.read_csv(args.out/"picks.csv")
        picks=list(a[~a.policy.str.endswith("_peer")].itertuples(index=False,name=None))
    else:
        recent=archive_picks(days)
        history=older_picks(days)
        picks=recent+history
        args.out.mkdir(parents=True,exist_ok=False)
        pd.DataFrame(picks,columns=["policy","origin","symbol","pool","matched_to"]).to_csv(args.out/"picks.csv",index=False)
        discovery(history,days).to_csv(args.out/"historical_discovery.csv",index=False)
    stocks=sorted({x[2] for x in picks})
    print(f"observations + controls={len(picks)}; stocks={len(stocks)}",flush=True)
    limits=load_limits(days)
    bars=prices(stocks,days,meta)
    if not args.existing_only:
        archive=archive_evidence(meta["calendar"],{x[2] for x in recent})
        rows=evaluate(recent,days,archive,"archive",bars,limits,0)
    with ProcessPoolExecutor(args.workers) as pool:
        start={s:min(d for _,d,t,_,_ in picks if t==s) for s in stocks}
        replay={}
        for n,(s,data) in enumerate(pool.map(replay_one,[(s,start[s],days[-1]) for s in stocks],chunksize=3),1):
            replay[s]={r[0]:dict(zip([*COLS,"breakout_accepted_at"],r[1:])) for r in data}
            if n%100==0:
                print(f"replayed {n}/{len(stocks)}",flush=True)
    existing_chains([r for r in picks if not r[0].endswith("_peer")],days,replay,bars,limits).to_csv(
        args.out/"existing_chain_followup.csv.gz",index=False)
    if args.existing_only:
        return
    diag=[]
    for policy,d,s,_,_ in picks:
        r=replay[s].get(d,{})
        diag.append(dict(policy=policy,origin=d,symbol=s,breakout_status=r.get("breakout_status","MISSING"),
                         breakout_event=r.get("breakout_event",""),setup_kind=r.get("setup_kind","")))
    pd.DataFrame(diag).to_csv(args.out/"origin_states.csv",index=False)
    tracked_pairs(history,days,replay,bars,limits).to_csv(args.out/"tracked_breakouts.csv.gz",index=False)
    for ext in (0,20):
        rows.extend(evaluate(picks,days,replay,"price_replay",bars,limits,ext))
    f=pd.DataFrame(rows)
    f.to_csv(args.out/"outcomes.csv.gz",index=False)
    t=summarize(f)
    t.to_csv(args.out/"summary.csv",index=False)
    print(t[(t.extra==0)&t.h.eq(10)&~t.policy.str.endswith("peer")].to_string(index=False))


if __name__=="__main__":
    main()
