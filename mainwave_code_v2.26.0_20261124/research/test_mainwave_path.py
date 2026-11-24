import numpy as np
import pandas as pd

from research.mainwave_path import downside_score, portfolio, rank
from research.route_followup import simulate


def test_downside_metric_uses_only_market_days_strictly_before_attempt():
    days=tuple(pd.bdate_range("2026-08-03",periods=13).strftime("%Y-%m-%d"))
    b=pd.Series([-.01]*13,index=days)
    s=pd.Series([.005]*10+[.9,-.9,.7],index=days)
    assert downside_score(days[10],days,b,s)==.015
    assert downside_score(days[10],days,b,s.drop(days[1])) is None
    assert downside_score(days[10],days,b.abs(),s) is None


def test_rank_changes_order_and_preserves_two_slots():
    a,b,c=[dict(symbol=s,bias_atr=a,downside=d) for s,a,d in
           [("a",1.1,.001),("b",1.2,.002),("c",1.3,.003)]]
    assert rank([c,b,a],"A")==[a,b]
    assert rank([c,b,a],"B")==[c,b]
    for r in (a,b,c):
        r["downside"]=None
    assert rank([c,b,a],"B")==[a,b]


def test_t_plus_one_and_released_lower_limit_open_are_not_instant_fills():
    days=("2026-09-01","2026-09-02","2026-09-03","2026-09-04")
    a={k:np.array(v) for k,v in {
        "day":days,"isST":[0]*4,"preclose":[10,10,10,9],"open":[10,10,9,8.5],
        "high":[10,11,9.5,9],"index_open":[10,10,9,8.5],"index_high":[10,11,9.5,9],
        "index_low":[10,8,8.9,8.4],"index_close":[10,10,9,8.6],"atr20":[.5]*4,
        "breakout_invalid_index":[9]*4,"low10":[9]*4,
    }.items()}
    trace=[]
    r=simulate(a,0,"sh_600001","structure36_20",trace=trace,calendar=days,conservative_limits={})
    assert not trace[0]["fills"] and not trace[1]["fills"]
    assert r["exit_date"]==days[3] and r["reason"]=="DEFERRED_STOP"
    assert np.isclose(r["gross"],-15)


def test_two_seats_cash_stays_occupied_until_exit_and_red_never_enters():
    days=("2026-08-03","2026-08-04","2026-08-05","2026-08-06")
    picks={days[0]:[dict(symbol=s,date=days[0]) for s in ("a","b")],
           days[1]:[dict(symbol="c",date=days[1])],days[2]:[dict(symbol="d",date=days[2])]}
    path=[dict(date=days[1],entry_index=10,initial_stop=9,index_close=10,remaining_after=1,fills=[]),
          dict(date=days[2],index_close=11,remaining_after=0,
               fills=[dict(weight=1.,price_index=11,reason="HORIZON")])]
    trades={(s,days[0]):dict(raw_open=10,path=path) for s in ("a","b")}
    n,t=portfolio(days,picks,{d:"RED" if d==days[2] else "GREEN" for d in days},trades)
    assert n.open.max()==2 and n.open.iat[-1]==0
    assert list(t.status)==["CLOSED","CLOSED","PORTFOLIO_CAP","MARKET_RED"]
    assert n.equity.iat[-1]>100000
