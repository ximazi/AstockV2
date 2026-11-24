"""Compare R2 discovery events and hypothetical executable open-to-open fills, independently."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from mainwave_engine.eod_review import limit_price
from research.r2_pool_history import ROOT


HORIZONS = (3, 5, 10)


def first_entries(days, sources, members):
    for i, day in enumerate(days):
        if i == 0 or day not in sources or days[i-1] not in sources:
            continue  # left-censored after missing snapshots, not a new discovery
        for pool in ("early", "strict"):
            for s in sorted(members.get((day,pool), set()) - members.get((days[i-1],pool), set())):
                yield i, pool, s


def load_limits(days):
    limits = {}
    for day in days:
        path = Path("results/intraday/reference") / f"stk_limit_{day.replace('-','')}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        for r in data.get("records", []):
            if data["as_of"] == day and str(r["trade_date"]) == day.replace("-",""):
                code, exch = r["ts_code"].lower().split(".")
                if r.get("up_limit") and r.get("down_limit"):
                    limits[exch+"_"+code,day] = float(r["up_limit"]), float(r["down_limit"])
    return limits


def band(s, bar, limits):
    exact = limits.get((s,bar.day))
    if exact:
        return *exact, "reference"
    if s == "sh_600182" or not (bar.preclose > 0 and bar.isST in (0,1)):
        return np.nan, np.nan, "unknown"
    pct = .20 if s.startswith(("sh_68","sz_30")) else .05 if bar.isST else .10
    return limit_price(bar.preclose, pct), limit_price(bar.preclose, -pct), "estimated"


def executable(s, i, h, days, bars, limits):
    # Reserve the next session for entry; a three-session holding exits at the fourth open.
    row = dict(status="PENDING", entry_day="", exit_day="", buy=np.nan, sell=np.nan,
               qty=0, net_pct=np.nan, blocked_exits=0, entry_band="", exit_band="")
    if i + 1 >= len(days):
        return row
    b = bars.get(days[i+1])
    if b is None or not (b.adjustflag == 3 and b.tradestatus in (0,1)):
        return dict(row, status="ENTRY_DATA_GAP")
    if b.tradestatus == 0 or b.amount == 0:
        return dict(row, status="ENTRY_SUSPENDED")
    up, _, source = band(s,b,limits)
    row["entry_band"] = source
    if not (b.open > 0 and b.volume > 0 and b.amount > 0 and np.isfinite(up) and b.isST in (0,1)):
        return dict(row, status="ENTRY_DATA_GAP")
    if b.isST or b.open >= up - .00001:
        return dict(row, status="ST_OR_LIMIT_UP")
    qty = math.floor(50000 / (b.open * 1.001 + .00001) / 100) * 100
    if qty < (200 if s.startswith("sh_68") else 100):
        return dict(row, status="BUDGET_TOO_SMALL")
    row.update(entry_day=b.day, buy=b.open, qty=qty)
    close = b.close
    for j in range(i+2,len(days)):
        x = bars.get(days[j])
        if x is None or x.adjustflag != 3 or x.tradestatus not in (0,1):
            return dict(row, status="HOLD_DATA_GAP")
        if x.tradestatus == 0 or x.amount == 0:
            row["blocked_exits"] += int(j >= i+1+h)
            continue
        if not (close > 0 and x.preclose > 0 and x.open > 0 and x.amount > 0 and x.volume > 0):
            return dict(row, status="HOLD_DATA_GAP")
        if abs(x.preclose-close) > .0051:
            return dict(row, status="CORPORATE_ACTION_UNPRICED")
        close = x.close
        if j < i+1+h:
            continue
        _, down, source = band(s,x,limits)
        if not np.isfinite(down):
            return dict(row, status="EXIT_LIMIT_UNKNOWN")
        if x.open <= down + .00001:
            row["blocked_exits"] += 1
            continue
        cost, value = qty*b.open, qty*x.open
        paid = cost + max(5, .0003*cost) + .0005*cost
        received = value - max(5, .0003*value) - .001*value  # sell tax + slippage reserve
        return dict(row, status="CLOSED", exit_day=x.day, sell=x.open, exit_band=source,
                    net_pct=(received/paid-1)*100)
    return row


def read_structures(days):
    fresh, accepted = {}, set()
    for day in days:
        p = ROOT/day.replace("-","")/"route_scan_audit.csv"
        if not p.exists():
            continue
        a = pd.read_csv(p,usecols=["date","data_status","breakout_event","symbol"])
        a = a[a.date.eq(day) & a.data_status.eq("FRESH")]
        fresh[day] = set(a.symbol)
        accepted.update((s,day) for s in a.loc[a.breakout_event.eq("ACCEPTED"),"symbol"])
    return fresh, accepted


def onset_audit(events, truth, days):
    """Fixed denominator, observed leads only: missing prehistory is not a confirmed miss."""
    signals = defaultdict(list)
    for i,p,s in events:
        signals[s,p].append(i)
    first = min(i for i,_,_ in events)
    rows = []
    for s,d in sorted(truth):
        t = days.index(d)
        if t <= first:
            continue
        for h in HORIZONS:
            leads = {p:max((t-i for i in signals[s,p] if 1 <= t-i <= h), default=None)
                     for p in ("early","strict")}
            rows.append(dict(symbol=s,target=d,h=h,**leads))
    return pd.DataFrame(rows)


def first_confirmation(s, window, fresh, accepted):
    for day in window:
        if s not in fresh.get(day, ()):
            return "DATA_WAIT", ""
        if (s,day) in accepted:
            return "CONFIRMED", day
    return "NO_CONFIRMATION", ""


def compare(out):
    meta = json.loads((out/"inputs.json").read_text(encoding="utf-8"))
    days, sources = meta["calendar"], meta["sources"]
    pool = pd.read_csv(out/"pools.csv.gz")
    members = pool.groupby(["day","pool"]).symbol.agg(set).to_dict()
    labels = pd.read_csv(out/"trend.csv.gz")
    truth = set(zip(labels.symbol,labels.day))
    maturity = pd.read_csv(out/"maturity.csv.gz",keep_default_na=False).set_index("symbol").last_mature.to_dict()
    prices = pd.read_csv(out/"prices.csv.gz")
    bars = {s:{r.day:r for r in g.itertuples(index=False)} for s,g in prices.groupby("symbol")}
    limits = load_limits(days)
    fresh, accepted = read_structures(days)
    rows = []
    events = list(first_entries(days,sources,members))
    onset = onset_audit(events,truth,days)
    onset.to_csv(out/"fixed_onsets.csv",index=False,encoding="utf-8-sig")
    for i,p,s in events:
        for h in HORIZONS:
            w = days[i+1:i+h+1]
            mature = len(w)==h and w[-1]<=maturity.get(s,"")
            target = next((d for d in w if (s,d) in truth),"") if mature else ""
            fresh_window = len(w)==h and all(s in fresh.get(d,()) for d in w)
            acc = next((d for d in w if (s,d) in accepted),"") if fresh_window else ""
            rows.append(dict(day=days[i],pool=p,symbol=s,h=h,source=sources[days[i]],
                             both_archived=sources[days[i]]==sources[days[i-1]]=="archive",
                             mature=mature, target=target,lead=w.index(target)+1 if target else None,
                             structural_mature=fresh_window, accepted=acc,
                             holding_mature=i+1+h<len(days),
                             **executable(s,i,h,days,bars.get(s,{}),limits)))
    df = pd.DataFrame(rows)
    bench = pd.read_csv("data/000300_SH_index.csv")
    bench.index = pd.to_datetime(bench.trade_date.astype(str)).dt.strftime("%Y-%m-%d")
    df["benchmark_pct"] = (df.exit_day.map(bench.open)/df.entry_day.map(bench.open)-1)*100
    df.to_csv(out/"events_and_fills.csv.gz",index=False)
    hits = df[df.target.ne("")].groupby(["h","pool","symbol","target"]).lead.max().reset_index()
    hits.to_csv(out/"hit_episodes.csv",index=False,encoding="utf-8-sig")
    summary, overlap, returns = [], [], []
    for h in HORIZONS:
        groups = {}
        mh = df[df.h.eq(h) & df.mature]
        for p in ("early","strict"):
            g = mh[mh.pool.eq(p)]
            hit = hits[hits.h.eq(h) & hits.pool.eq(p)]
            groups[p] = set(zip(hit.symbol,hit.target))
            # Denominator: mature TREND episodes reachable from any comparable signal date,
            # regardless of which stock was in either pool.
            reachable = {d for start in mh.day.unique() for d in days[days.index(start)+1:days.index(start)+h+1]}
            eligible_targets = {(s,d) for s,d in truth if d in reachable and any(
                days[days.index(start)+h] <= maturity.get(s,"") and start < d <= days[days.index(start)+h]
                for start in mh.day.unique())}
            summary.append(dict(h=h,pool=p,days=g.day.nunique(),signals=len(g),hit_signals=int(g.target.ne("").sum()),
                                unique=len(hit),available=len(eligible_targets),lead_med=hit.lead.median(),
                                lead_mean=hit.lead.mean()))
        overlap.append(dict(h=h,both=len(groups["early"] & groups["strict"]),
                            only_early=len(groups["early"]-groups["strict"]),only_strict=len(groups["strict"]-groups["early"])))
        for sample, selection in (("all",df),("archive",df[df.both_archived])):
            for p in ("early","strict"):
                g=selection[selection.h.eq(h) & selection.pool.eq(p) & selection.holding_mature]
                c=g[g.status.eq("CLOSED")]
                returns.append(dict(sample=sample,h=h,pool=p,days=g.day.nunique(),signals=len(g),
                                    closed=len(c),mean=c.net_pct.mean(),median=c.net_pct.median(),
                                    win_pct=c.net_pct.gt(0).mean()*100,worst=c.net_pct.min(),
                                    benchmark_mean=c.benchmark_pct.mean(),excess_mean=(c.net_pct-c.benchmark_pct).mean(),
                                    equal_day_mean=c.groupby("day").net_pct.mean().mean(),
                                    deferred=int(c.blocked_exits.gt(0).sum()),
                                    statuses=g.status.value_counts().to_dict()))
    struct=[]
    for (h,p),g in df[df.both_archived & df.structural_mature].groupby(["h","pool"]):
        ok=g[g.accepted.ne("")]
        struct.append(dict(h=h,pool=p,days=g.day.nunique(),events=len(g),hit_signals=len(ok),
                           unique=len(ok[["symbol","accepted"]].drop_duplicates()),rate=100*len(ok)/len(g)))
    timing=[]
    trail=[]
    # Give both arms the same signal-origin dates, with time for the entire wait and holding period.
    for h in HORIZONS:
        for p in ("early","strict"):
            cohort=df[(df.h==h) & df.both_archived & (df.pool==p) & df.day.isin(days[:-(11+h)])]
            attempts, seen = [], set()
            for r in cohort.itertuples():
                i=days.index(r.day)
                state,d=first_confirmation(r.symbol,days[i+1:i+11],fresh,accepted)
                if d and (r.symbol,d) in seen:
                    state="SAME_TRIGGER"
                fill={}
                if state=="CONFIRMED":
                    seen.add((r.symbol,d))
                    fill=executable(r.symbol,days.index(d),h,days,bars.get(r.symbol,{}),limits)
                    attempts.append(fill)
                trail.append(dict(pool=p,symbol=r.symbol,day=r.day,h=h,confirmation=d,watch_status=state,**fill))
            closed=[x["net_pct"] for x in attempts if x["status"]=="CLOSED"]
            timing.append(dict(h=h,pool=p,origin_days=cohort.day.nunique(),origins=len(cohort),confirmed=len(attempts),
                               closed=len(closed),mean=np.mean(closed) if closed else None,
                               median=np.median(closed) if closed else None,
                               win_pct=100*np.mean(np.array(closed)>0) if closed else None,
                               direct_closed=int(cohort.net_pct.count()),direct_mean=cohort.net_pct.mean()))
    pd.DataFrame(trail).to_csv(out/"confirmation_fills.csv.gz",index=False)
    # Equal attention budget: expected hits when sampling both without rank, same count each day.
    budget=[]
    for h,g in df[df.mature].groupby("h"):
        expected=defaultdict(float)
        n=0
        for _,d in g.groupby("day"):
            k=min((d.pool==p).sum() for p in ("early","strict"))
            if k:
                n+=k
                for p in ("early","strict"):
                    expected[p]+=k*d[d.pool==p].target.ne("").mean()
        budget.append(dict(h=h,each=n,expected=dict(expected)))
    fixed=[]
    for h,g in onset.groupby("h"):
        both=g.dropna(subset=["early","strict"])
        fixed.append(dict(h=h,available=len(g),early=int(g.early.count()),strict=int(g.strict.count()),
                          early_median=g.early.median(),strict_median=g.strict.median(),both=len(both),
                          early_first=int(both.early.gt(both.strict).sum()),same_day=int(both.early.eq(both.strict).sum()),
                          strict_first=int(both.strict.gt(both.early).sum())))
    result=dict(market_days=len(days),archive_dates=len(sources),fixed_onsets=fixed,daily_mean={
        p:len(pool[pool.pool==p])/len(sources) for p in ("early","strict")}, discovery=summary,
        overlap=overlap,equal_budget=budget,execution=returns,structure=struct,confirmation=timing)
    (out/"summary.json").write_text(pd.Series(result).to_json(force_ascii=False,indent=2),encoding="utf-8")
    for table in (fixed,summary,overlap,budget,struct,returns,timing):
        print(pd.DataFrame(table).drop(columns="statuses",errors="ignore").round(3).to_string(index=False))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    compare(parser.parse_args().out)
