"""Frozen R1 accepted_bias2 vs pre-breakout downside-relative-strength. Research only."""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mainwave_engine import market_context as MC, phase_events as PE, route_evidence as RE
from mainwave_engine.review_clock import CST, visible
from mainwave_engine.trading.books import NEW100K_ACCOUNT as AC, NEW100K_PROTOCOL
from research import route_clock as CLOCK
from research.r2_pool_compare import band, load_limits
from research.r2_pool_history import SYMBOL, trend_labels
from research.route_followup import arrays, enrich, simulate

ROOT = Path("results/mainwave_engine")
LEGACY = ROOT / "route_followup_20260910/r1_events.csv.gz"
VERSION = "mainwave_path_ab_v1_1_20261119"
COLS = ["symbol","date","breakout_id","bias_atr"]
BREAK = "2026-09-09"
PCT_COST = .003
SPEC = dict(version=VERSION, baseline="accepted_bias2; smaller bias_atr first",
            challenger="same events; median stock minus CSI300 return on CSI300 down days in 10 market sessions before first attempt; min 3",
            order="next opening; skip upper limit, gaps above 1 ATR and incomplete quotes; no substitutions beyond Top2",
            exit="structure36_20 plus conservative lower-limit open deferral; no reuse of same-day sale proceeds",
            book="NEW100K limits and lot sizing, two simultaneous stocks, reserved 0.30% round trip; exits/marks in adjusted index",
            oracle="onset [-5,+3] sessions, next-open price limits/gap/lots and +25% closes within 40 bars before 2ATR breach",
            orders_enabled=False)


def sessions():
    return CLOCK.sessions()


def benchmark():
    d=pd.read_csv("data/000300_SH_index.csv",usecols=["trade_date","close"],dtype={"trade_date":str})
    d.index=pd.to_datetime(d.trade_date).dt.strftime("%Y-%m-%d")
    return d.close.sort_index().pct_change(fill_method=None)


def returns(raw):
    return pd.to_numeric(raw.pct_chg,errors="coerce").div(100).set_axis(raw.date)


def downside_score(attempt, cal, bench, stock):
    j=bisect_left(cal,attempt)
    dates=cal[max(0,j-10):j]
    b=bench.reindex(dates)
    a=stock.reindex(dates)
    down=b.lt(0)
    if len(dates)<10 or not b.notna().all() or down.sum()<3 or not a[down].notna().all():
        return None
    return round(float((a[down]-b[down]).median()),8)


def rank(rows, arm):
    if arm=="A":
        return sorted(rows,key=lambda r:(r["bias_atr"],r["symbol"]))[:2]
    # A day with insufficient historical benchmark declines still uses the baseline.
    return sorted(rows,key=lambda r:(r["downside"] is None,-(r["downside"] or 0),r["bias_atr"],r["symbol"]))[:2]


def day_events(path,day):
    a=pd.read_csv(path,usecols=[*COLS,"accepted_bias2","eligible","data_status"])
    if not (a.date.eq(day)&a.data_status.eq("FRESH")).any():
        raise ValueError(f"no fresh daily route audit: {day}")
    return a[a.date.eq(day)&a.data_status.eq("FRESH")&a.accepted_bias2.eq(True)&a.eligible.eq(True)][COLS]


def attempt_of(r):
    at=str(r["breakout_id"]).rsplit(":",1)[-1]
    if len(at)!=10 or at>=r["date"]:
        raise ValueError(f"Unverifiable attempt identity: {r}")
    return at


def read_raw(s):
    raw=pd.read_csv(Path("data")/(s+"_daily.csv"),dtype={"date":str})
    return raw.sort_values("date").drop_duplicates("date",keep="last").reset_index(drop=True)


def observation(day,output=None):
    """Use dated EOD audit, never write the daily archive during research backfills."""
    folder=ROOT/day.replace("-","")
    out=Path(output) if output else folder
    out.mkdir(exist_ok=True,parents=True)
    market=MC.load(day)
    report=dict(as_of=day,version=VERSION,market=market["state"],status="DATA_WAIT",picks={},orders_enabled=False)
    try:
        a=day_events(folder/"route_scan_audit.csv",day)
        if day not in sessions() or a.symbol.duplicated().any():
            raise ValueError("market date or unique event not verified")
        bench=benchmark()
        rows=a.to_dict("records")
        for r in rows:
            at=attempt_of(r)
            raw=read_raw(r["symbol"])
            r["downside"]=downside_score(at,sessions(),bench,returns(raw))
        report.update(candidates=len(rows),picks={k:rank(rows,k) for k in ("A","B")},
                      status="READY" if market["data_status"]=="FRESH" else "DATA_WAIT")
    except (OSError,ValueError,KeyError) as exc:
        report["reason"]=str(exc)
    (out/"mainwave_path_shadow.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    now=datetime.now(CST)
    if (output is None and report["status"]=="READY" and now.date().isoformat()==day and now.hour>=15
            and visible(folder/"route_scan_audit.csv",day,now.isoformat())):
        try:
            with (out/f"{VERSION}.frozen.json").open("x",encoding="utf-8") as f:
                json.dump({**report,"frozen_at":now.isoformat()},f,ensure_ascii=False,indent=2)
        except FileExistsError:
            pass
    return report


def market_flags(cal):
    from research.mean_reversion_threshold_study import INDEX_PATHS, build_market_frame
    f=build_market_frame()
    keep=set(cal)
    for path in INDEX_PATHS.values():
        a=pd.read_csv(path,usecols=["trade_date","close"],dtype={"trade_date":str})
        keep &= set(pd.to_datetime(a.loc[a.close.gt(0),"trade_date"]).dt.strftime("%Y-%m-%d"))
    f["day"]=f.date.dt.strftime("%Y-%m-%d")
    out={}
    for r in f.itertuples(index=False):
        if r.day not in keep or any(pd.isna(v) for v in (r.avg_pct,r.up_ratio,r.limit_up,r.limit_down)):
            continue
        out[r.day]=("RED" if r.temperature=="PANIC" else "ORANGE"
                    if r.structure=="BEAR" or r.temperature=="EUPHORIA" or r.market_context=="POST_PANIC" else "GREEN")
    return out


def portfolio(cal,picks,states,trades,fee=PCT_COST):
    """Trade plans and actual outcomes are separate; a future fill never releases today's capacity."""
    cash,held,peak=AC["capital"],{},AC["capital"]
    pending={}
    nav, ledger=[],[]
    for day in cal:
        equity=cash+sum(x["mark"] for x in held.values())
        risk_base=min(equity,AC["equity_ceiling"])
        for r in pending.get(day,[]):
            s=r["symbol"]; tr=trades.get((s,r["date"]),{})
            t=dict(day=r["date"],symbol=s,entry=day,status="")
            ledger.append(t)
            if len(held)>=AC["stock_seats"] or s in held:
                t["status"]="PORTFOLIO_CAP"
                continue
            if not tr.get("path"):
                t["status"]=tr.get("status","EVIDENCE_MISMATCH")
                continue
            first=tr["path"][0]
            if first["date"]!=day:
                t["status"]="ENTRY_CALENDAR_MISMATCH"
                continue
            risk=(first["entry_index"]-first["initial_stop"])/first["entry_index"]
            raw=tr["raw_open"]
            existing_risk=sum(x["risk"] for x in held.values())
            existing_total=sum(x["mark"] for x in held.values())
            capacity=min(cash/(1+fee),risk_base*AC["symbol_cap_pct"]/100,
                         risk_base*AC["gross_cap_pct"]/100-existing_total,
                         risk_base*AC["theme_cap_pct"]/100-existing_total,  # No PIT theme: assume common exposure.
                         risk_base*AC["single_risk_pct"]/100/risk,
                         (risk_base*AC["total_risk_pct"]/100-existing_risk)/risk)
            qty=min(AC["max_fill_lots"],max(0,int(capacity//(100*raw))))*100
            if qty<(200 if s.startswith("sh_68") else 100):
                t["status"]="RISK_OR_LOT"
                continue
            principal=qty*raw
            cash-=principal*(1+fee)
            t.update(status="OPEN",price=raw,qty=qty,notional=principal,risk=principal*risk)
            held[s]=dict(t=t,mark=principal,risk=principal*risk,n=principal/first["entry_index"],
                         path={x["date"]:x for x in tr["path"]},sold=0.)
        for s,x in list(held.items()):
            log=x["path"].get(day)
            if not log:
                continue  # Suspended, unknown, or censored: mark last known and never free the slot.
            realized=x["n"]*sum(f["weight"]*f["price_index"] for f in log["fills"])
            cash+=realized
            x["sold"]+=realized
            x["mark"]=x["n"]*log["index_close"]*log["remaining_after"]
            if not log["remaining_after"]:
                t=x["t"]
                t.update(status="CLOSED",exit=day,pnl=x["sold"]-t["notional"]*(1+fee),
                         reason=log["fills"][-1]["reason"])
                del held[s]
        equity=cash+sum(x["mark"] for x in held.values())
        peak=max(peak,equity)
        drawdown=(equity/peak-1)*100
        nav.append(dict(day=day,equity=equity,drawdown_pct=drawdown,open=len(held),cash=cash))
        for r in picks.get(day,[]):
            if states.get(day)!="GREEN" or drawdown<=-NEW100K_PROTOCOL["drawdown_pause_pct"]:
                ledger.append(dict(day=day,symbol=r["symbol"],status="MARKET_"+states.get(day,"UNKNOWN")
                                   if states.get(day)!="GREEN" else "DRAWDOWN_PAUSE"))
                continue
            j=bisect_left(cal,day)+1
            if j<len(cal):
                pending.setdefault(cal[j],[]).append(r)
            else:
                ledger.append(dict(day=day,symbol=r["symbol"],status="PENDING_ENTRY"))
    return pd.DataFrame(nav),pd.DataFrame(ledger)


def oracle(symbol,a,starts,limits):
    days=a["day"]
    for wave in starts:
        i=days.searchsorted(wave)
        if i==len(days) or days[i]!=wave or i<5 or i+43>=len(days):
            yield wave,"","","LABEL_WINDOW_INCOMPLETE"
            continue
        did_open=did_risk=False
        out=target=""
        for j in range(i-5,i+4):
            if CLOCK.classify_gap(days[j-1],days[j],CAL)["status"]!="NEXT_EXCHANGE_SESSION":
                continue
            raw,atr=a["open"][j],a["atr20"][j-1]
            upper,_,_=band(symbol,limit_bar(days[j],a["preclose"][j],a["isST"][j]),limits)
            if not (raw>0 and a["isST"][j]==0 and np.isfinite(upper) and raw<upper and a["amount"][j]>0):
                continue
            did_open=True
            price=a["index_open"][j]
            minlot=200 if symbol.startswith("sh_68") else 100
            if not atr>0 or price-a["index_close"][j-1]>atr or minlot*2*atr/price*raw>1000 or minlot*raw>35000:
                continue
            did_risk=True
            # Daily stop-first envelope. Diagnostic only; this is not an entry rule using future data.
            stop=price-2*atr
            for k in range(j,j+41):
                if a["index_low"][k]<=stop:
                    break
                if a["index_close"][k]>=1.25*price:
                    out=days[j]
                    target=days[k]
                    break
            if out:
                break
        yield wave,out,target,("PRICE_PATH_AVAILABLE" if out else "NO_OPEN" if not did_open
                        else "RISK_OR_GAP" if not did_risk else "NO_REMAINING_PATH")


def limit_bar(day,preclose,isST):
    from types import SimpleNamespace
    return SimpleNamespace(day=day,preclose=preclose,isST=isST)


def init(cal,asof,start,limits):
    global CAL, END, START, LIMITS
    CAL, END, START, LIMITS=cal,asof,start,limits


def worker(job):
    s,events=job
    raw=read_raw(s)
    raw=raw[raw.date.le(END)].reset_index(drop=True)
    active=raw[raw.amount.gt(0)&raw.tradestatus.eq(1)].reset_index(drop=True)
    starts,_=trend_labels(active,s,START)
    waves=[]
    if starts:
        p=PE.classify_frame(active)
        if not p.empty:
            p["index_close"]=p.return_index
            for col in ("open","low"):
                p[f"index_{col}"]=p[col]*p.return_index/p.close
            waves=list(oracle(s,arrays(p),starts,LIMITS))
    trades={}
    if events:
        f=RE.build(raw,s)
        if f.empty:
            return s,waves,{r["date"]:{"status":"EVIDENCE_MISMATCH"} for r in events}
        f=enrich(f)
        a=arrays(f)
        ix={d:i for i,d in enumerate(a["day"])}
        for r in events:
            d=r["date"]; i=ix.get(d)
            if i is None or not f.accepted_bias2.iat[i] or f.breakout_id.iat[i]!=r["breakout_id"]:
                trades[d]={"status":"EVIDENCE_MISMATCH"}
                continue
            j=i+1
            if j<len(a["day"]):
                upper,_,_=band(s,limit_bar(a["day"][j],a["preclose"][j],a["isST"][j]),LIMITS)
                if not np.isfinite(upper) or a["open"][j]>=upper:
                    trades[d]={"status":"ENTRY_LIMIT_UP" if np.isfinite(upper) else "ENTRY_LIMIT_UNKNOWN"}
                    continue
            path=[]
            tr=simulate(a,i,s,"structure36_20",gap_guard=True,trace=path,calendar=CAL,conservative_limits=LIMITS)
            trades[d]={**tr,"path":path,"raw_open":float(a["open"][i+1]) if path else None}
    return s,waves,trades


def historic(start,end,out,workers,published_only=False):
    out=Path(out)
    if out.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("research cannot overwrite dated production results")
    out.mkdir(parents=True,exist_ok=False)
    cal=tuple(d for d in sessions() if d<=end)
    if not cal or end!=cal[-1]:
        raise ValueError("end must be observed session")
    bench=benchmark()
    states=market_flags(cal)
    feed=defaultdict(list)
    picks={k:{} for k in ("A","B")}
    cohorts=[]
    if not published_only:
        e=pd.read_csv(LEGACY,usecols=COLS)
        e=e[e.date.between(start,min(end,BREAK))]
        for r in e.to_dict("records"):
            feed[r["date"]].append(r)
        cohorts=[d for d in cal if start<=d<=min(end,BREAK)]
    for day in cal:
        if day<start or (not published_only and day<=BREAK):
            continue
        folder=ROOT/day.replace("-","")
        if published_only:
            p=folder/f"{VERSION}.frozen.json"
            if not p.is_file():
                continue
            r=json.loads(p.read_text(encoding="utf-8"))
            stamp=datetime.fromisoformat(r["frozen_at"]).astimezone(CST)
            if (r["as_of"],r["version"],r["status"])!=(day,VERSION,"READY") or stamp.date().isoformat()!=day or stamp.hour<15:
                continue
            for k in picks:
                picks[k][day]=r["picks"][k]
            states[day]=r["market"]
            cohorts.append(day)
        else:
            p=folder/"route_scan_audit.csv"
            if not p.is_file():
                continue
            feed[day].extend(day_events(p,day).to_dict("records"))
            cohorts.append(day)
    if not published_only:
        stocks=defaultdict(list)
        for rr in feed.values():
            for r in rr:
                stocks[r["symbol"]].append(r)
        for s,rr in stocks.items():
            stock=returns(read_raw(s))
            for r in rr:
                r["downside"]=downside_score(attempt_of(r),cal,bench,stock)
        for day,rr in feed.items():
            for k in picks:
                picks[k][day]=rank(rr,k)
    events=defaultdict(dict)
    for rr in [*feed.values(),*(r for field in picks.values() for r in field.values())]:
        for r in rr:
            events[r["symbol"]][r["date"]]=r
    paths=[p.name[:-10] for p in sorted(Path("data").glob("*_daily.csv")) if SYMBOL.fullmatch(p.name[:-10])]
    jobs=[(s,list(events[s].values())) for s in (sorted(events) if published_only else paths)]
    available=load_limits([d for d in cal if d>=start])
    trades,waves={},[]
    with ProcessPoolExecutor(workers,initializer=init,initargs=(cal,end,start,available)) as pool:
        for i,(s,qq,tt) in enumerate(pool.map(worker,jobs,chunksize=8),1):
            trades.update({(s,d):t for d,t in tt.items()})
            waves.extend((s,*w) for w in qq)
            if i%500==0:
                print(f"[mainwave_path] {i}/{len(jobs)} stocks",flush=True)
    summary=[]; ledgers={}
    for year in sorted({d[:4] for d in cal if d>=start}):
        days=tuple(d for d in cal if d>=start and d.startswith(year))
        selection={k:{d:picks[k][d] for d in days if d in picks[k]} for k in picks}
        for arm in ("A","B"):
            for fee in (PCT_COST,.004):
                n,t=portfolio(days,selection[arm],states,trades,fee)
                suffix=f"{year}_{arm}_{int(fee*10000)}bp"
                n.to_csv(out/f"nav_{suffix}.csv",index=False)
                t.to_csv(out/f"ledger_{suffix}.csv",index=False)
                if fee==PCT_COST and not t.empty:
                    ledgers.update({(arm,r["symbol"],r["day"]):r for r in t.to_dict("records")})
                c=t[t.status.eq("CLOSED")] if not t.empty else t
                pnl=c.get("pnl",pd.Series(dtype=float))
                summary.append(dict(year=year,arm=arm,cost_bp=int(fee*10000),days=len([d for d in cohorts if d in days]),
                                    green=sum(states.get(d)=="GREEN" for d in cohorts if d in days),
                                    selected=sum(map(len,selection[arm].values())),closed=len(c),
                                    net_pct=(n.equity.iat[-1]/AC["capital"]-1)*100,
                                    drawdown_pct=n.drawdown_pct.min(),ex_top=(n.equity.iat[-1]-max(0,pnl.max()))/AC["capital"]*100-100,
                                    statuses=t.status.value_counts().to_dict() if not t.empty else {}))
    audit=[]
    scope=set(cohorts)
    for s,w,op,target,status in waves:
        i=bisect_left(cal,w)
        window=cal[max(0,i-6):i+3]
        if status=="PRICE_PATH_AVAILABLE":
            candidates=[d for d in window if d in events[s]]
            later=[d for d in cal[i+3:i+11] if d in events[s]]
        else:
            candidates,later=[],[]
        before_target=[d for d in cal[i+3:bisect_left(cal,target)] if d in events[s]] if target else []
        row=dict(symbol=s,start=w,oracle_open=op,oracle_target=target,oracle_status=status,
                 events="|".join(candidates),later="|".join(later),later_before_target="|".join(before_target))
        green=[d for d in candidates if states.get(d)=="GREEN"]
        executable=[d for d in green if trades.get((s,d),{}).get("path")]
        row.update(green_events="|".join(green),executable_events="|".join(executable))
        for k in picks:
            chosen=[(d,ledgers.get((k,s,d),{})) for d in candidates if any(r["symbol"]==s for r in picks[k].get(d, []))]
            if status!="PRICE_PATH_AVAILABLE":
                reason=status
            elif any(t.get("status")=="CLOSED" and t.get("pnl",0)>0 for _,t in chosen):
                reason="NET_POSITIVE_CAPTURE"
            elif any(t.get("status")=="CLOSED" for _,t in chosen):
                reason="FILLED_NONPOSITIVE"
            elif candidates and not green:
                reason="MARKET_GATE" if all(d in states for d in candidates) else "MARKET_DATA_GAP"
            elif chosen:
                reason="|".join(sorted({t.get("status","PENDING_OR_UNVERIFIED") for _,t in chosen}))
            elif candidates:
                reason="RANK_NOT_SELECTED" if executable else "NO_EXECUTABLE_ACCEPTANCE"
            elif not set(window).issubset(scope):
                reason="ARCHIVE_GAP"
            else:
                reason="LATE_ACCEPTANCE" if later else "NO_ACCEPTANCE_IN_WINDOW"
            row[k]=reason
        audit.append(row)
    pd.DataFrame(audit).to_csv(out/"opportunity_audit.csv",index=False)
    pd.DataFrame(summary).to_csv(out/"summary.csv",index=False)
    manifest=dict(protocol=SPEC,start=start,end=end,scope_days=cohorts,published_only=published_only,
                  legacy_sha256=hashlib.sha256(LEGACY.read_bytes()).hexdigest() if not published_only else None,
                  event_count=sum(map(len,feed.values())),picked=picks)
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(pd.DataFrame(summary).drop(columns="statuses").round(3).to_string(index=False))


def gate_audit(out):
    """Decompose missing accepted_bias2 matches using the earlier, broader trigger archive."""
    out=Path(out)
    scope=json.loads((out/"manifest.json").read_text(encoding="utf-8"))["scope_days"]
    known=set(scope)
    old=ROOT/"route_validation_20260910/events.csv.gz"
    columns=["symbol","date","attempt","accepted","accepted_demand"]
    history=pd.read_csv(old,usecols=columns)
    seen={}
    for r in history[history.date.isin(known)&(history.attempt|history.accepted)].itertuples(index=False):
        seen[r.symbol,r.date]="DEMAND_ACCEPTED" if r.accepted_demand else "RAW_ACCEPTED" if r.accepted else "ATTEMPT"
    for day in scope:
        if day<=BREAK:
            continue
        path=ROOT/day.replace("-","")/"route_scan_audit.csv"
        a=pd.read_csv(path,usecols=[*columns,"data_status"])
        for r in a[a.date.eq(day)&a.data_status.eq("FRESH")&(a.attempt.eq(True)|a.accepted.eq(True))].itertuples(index=False):
            seen[r.symbol,r.date]="DEMAND_ACCEPTED" if r.accepted_demand==True else (
                "RAW_ACCEPTED" if r.accepted==True else "ATTEMPT")
    cal=sessions()
    a=pd.read_csv(out/"opportunity_audit.csv",keep_default_na=False)
    ledger=[]
    for r in a[a.oracle_status.eq("PRICE_PATH_AVAILABLE")].itertuples():
        i=bisect_left(cal,r.start)
        dates=cal[max(0,i-6):i+3]
        events={seen.get((r.symbol,d)) for d in dates}
        stage=("BASELINE_ACCEPTED" if r.events else "ARCHIVE_GAP" if not set(dates).issubset(known) else
               "DEMAND_ACCEPTED_BUT_BASELINE_FILTERED" if "DEMAND_ACCEPTED" in events else
               "RAW_ACCEPTED_NO_DEMAND_CONFIRMATION" if "RAW_ACCEPTED" in events else
               "ATTEMPT_NO_ACCEPTANCE" if "ATTEMPT" in events else "NO_RECORDED_ATTEMPT")
        ledger.append(dict(symbol=r.symbol,start=r.start,gate=stage))
    d=pd.DataFrame(ledger,columns=["symbol","start","gate"])
    d.to_csv(out/"route_gate_audit.csv",index=False)
    print(d.groupby([d.start.str[:4],"gate"]).size().to_string())


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date")
    p.add_argument("--start",default="2023-01-01")
    p.add_argument("--end")
    p.add_argument("--out",type=Path)
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--published-only",action="store_true")
    p.add_argument("--audit-run",type=Path)
    args=p.parse_args()
    if args.audit_run:
        gate_audit(args.audit_run)
    elif args.date:
        report=observation(args.date,args.out)
        print(json.dumps(report,ensure_ascii=False))
        raise SystemExit(0 if report["status"]=="READY" else 2)
    elif args.end and args.out:
        historic(args.start,args.end,args.out,args.workers,args.published_only)
    else:
        p.error("use --date for EOD observation or --end and --out for research")
