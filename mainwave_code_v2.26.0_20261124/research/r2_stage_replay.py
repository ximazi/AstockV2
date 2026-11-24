"""Chronological R2 stage-board regression vs the archived one-from-each-pool baseline."""
import argparse
import json
from pathlib import Path

import pandas as pd

from mainwave_engine import r2_stage_watch as W
from mainwave_engine.review_clock import instant
from research.r2_pool_compare import HORIZONS, executable, load_limits
from research.r2_pool_history import trend_labels


def published(folder,day):
    file=folder/f"{W.VERSION}.frozen.json"
    if file.exists():
        r=json.loads(file.read_text(encoding="utf-8"))
        stamp=instant(r.get("frozen_at"))
        if (r.get("as_of")==day and r.get("version")==W.VERSION and r.get("basis")=="当日收盘"
                and r.get("source_complete") and stamp and stamp.date().isoformat()==day and stamp.hour>=15):
            return r
    return dict(status="DATA_WAIT",issues=["无当日冻结记录，不以历史重算替代"],followup=[],comparable=False)


def run(start,end,out,published_only=False):
    out=Path(out)
    if out.resolve().is_relative_to(W.ROOT.resolve()):
        raise ValueError("replay cannot write to production archives")
    out.mkdir(parents=True,exist_ok=False)
    dates=sorted(pd.to_datetime(pd.read_csv("data/000300_SH_index.csv",dtype=str).trade_date).dt.strftime("%Y-%m-%d").unique())
    days=[d for d in dates if d<=end]
    if end!=days[-1] or start not in days:
        raise ValueError("requested endpoints must be market days in the local calendar")
    cohorts=[]
    picks=[]
    tracks=[]
    statuses=[]
    for d in days:
        if d<start:
            continue
        report=(published(W.ROOT/d.replace("-",""),d) if published_only else W.run(d,dest=out/"boards"))
        statuses.append(dict(day=d,status=report["status"],issues=report["issues"]))
        tracks.extend(dict(day=d,**r) for r in report["followup"])
        if not report["comparable"] or report["status"]=="DATA_WAIT":
            continue
        cohorts.append(d)
        for r in report["picks"]:
            picks.append((d,"split",r["symbol"],r["lane"]))
        for s in report["controls"]:
            picks.append((d,"original",s,""))
    stocks={s for _,_,s,_ in picks}
    bars,targets,maturity={},{},{}
    for s in stocks:
        raw=pd.read_csv(Path("data")/(s+"_daily.csv")).sort_values("date").drop_duplicates("date",keep="last")
        raw=raw[raw.date.le(end)].reset_index(drop=True)
        starts,maturity[s]=trend_labels(raw,s,start)
        targets[s]=set(starts)
        a=raw.rename(columns={"date":"day"})
        bars[s]={r.day:r for r in a[a.day.between(start,end)].itertuples(index=False)}
    limits=load_limits([d for d in days if d>=start])
    results=[]
    for d,policy,s,lane in picks:
        i=days.index(d)
        for h in HORIZONS:
            fill=executable(s,i,h,days,bars[s],limits)
            w=days[i+1:i+h+1]
            mature=len(w)==h and w[-1]<=maturity[s]
            hit=next((x for x in w if x in targets[s]),"") if mature else ""
            results.append(dict(day=d,policy=policy,symbol=s,lane=lane,h=h,mature=i+1+h<len(days),
                                label_mature=mature,trend_start=hit,lead=w.index(hit)+1 if hit else None,**fill))
    frame=pd.DataFrame(results,columns=["day","policy","symbol","lane","h","mature","label_mature","trend_start","lead","status","entry_day","exit_day",
                                       "buy","sell","qty","net_pct","blocked_exits","entry_band","exit_band"])
    frame.to_csv(out/"comparison_fills.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(tracks).to_csv(out/"tracking.csv",index=False,encoding="utf-8-sig")
    summary=[]
    for h in HORIZONS:
        eligible=[d for d in cohorts if days.index(d)+1+h<len(days)]
        for policy in ("split","original"):
            g=frame[frame.day.isin(eligible)&frame.policy.eq(policy)&frame.h.eq(h)]
            net=g.net_pct.dropna()
            empty=len(eligible)*2-len(g)
            blocked=int(g.status.isin(["ST_OR_LIMIT_UP","ENTRY_SUSPENDED","BUDGET_TOO_SMALL"]).sum())
            known=len(net)+empty+blocked
            labeled=frame[(frame.policy==policy)&(frame.h==h)&frame.label_mature.eq(True)]
            hits=labeled[labeled.trend_start.ne("")]
            summary.append(dict(policy=policy,h=h,days=len(eligible),slots=2*len(eligible),observations=len(g),
                                empty=empty,closed=len(net),blocked=blocked,unknown=len(g)-blocked-len(net),
                                mean=net.mean(),median=net.median(),win_pct=net.gt(0).mean()*100,
                                known_slot_mean=net.sum()/known if known else None,label_mature=len(labeled),
                                waves=len(hits.drop_duplicates(["symbol","trend_start"])) if len(labeled) else None,
                                lead_median=hits.groupby(["symbol","trend_start"]).lead.max().median() if len(hits) else None))
    pd.DataFrame(summary).to_csv(out/"summary.csv",index=False,encoding="utf-8-sig")
    (out/"cohorts.json").write_text(json.dumps(dict(start=start,end=end,comparable=cohorts,status=statuses,
                                                  version=W.VERSION,published_only=published_only),ensure_ascii=False,indent=2),encoding="utf-8")
    print(pd.DataFrame(summary).round(2).to_string(index=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start",required=True)
    p.add_argument("--end",required=True)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--published-only",action="store_true",help="Evaluate only actual immutable EOD publications")
    args=p.parse_args()
    run(args.start,args.end,args.out,args.published_only)
