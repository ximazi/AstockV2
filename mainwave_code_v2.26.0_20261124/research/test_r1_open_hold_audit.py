import numpy as np
import pandas as pd

from research.mainwave_path import portfolio
from research.r1_early_demand_replay import observe_top2
from research.r1_open_hold_audit import BLOCK, UNKNOWN, check_open, guard_intervals, standalone


def test_frozen_breakout_equality_and_missing_open():
    assert check_open(10,10)=="PASS"
    assert check_open(9.99,10)==BLOCK
    assert check_open(10.01,10)=="PASS"
    assert check_open(np.nan,10)==UNKNOWN and check_open(10,0)==UNKNOWN


def test_skipped_winner_and_avoided_loss_are_both_counted():
    rows=[
        ("PASS","CLOSED","REJECTED",-1.),
        (BLOCK,"CLOSED","REJECTED",-3.),
        (BLOCK,"CLOSED","ACCEPTED",5.),
        (BLOCK,"HOLDING_DATA_UNVERIFIED","ACCEPTED",np.nan),
        ("NOT_APPLICABLE","ENTRY_GAP","ACCEPTED",np.nan),
    ]
    f=pd.DataFrame(rows,columns=["gate","status","terminal","net"]).assign(
        attempt="2026-01-05",early_market="GREEN",mature=True,base_opened=[True]*4+[False])
    s=standalone(f).query("sample=='green_only'").iloc[0]
    assert (s.base_closed,s.blocked,s.blocked_closed,s.blocked_failed,s.blocked_winners)==(3,3,2,1,1)
    assert (s.losses_avoided,s.profits_missed,s.change_per_signal)==(3.,5.,-.4)
    interval=guard_intervals(f,("2026-01-05",)).query("sample=='green_only'").iloc[0]
    assert interval.n==3 and np.isclose(interval["mean"],-2/3)


def test_guard_does_not_refill_daily_top_two_after_the_open():
    days=("2026-09-01","2026-09-02","2026-09-03")
    d=days[0]
    f=pd.DataFrame(dict(symbol=["a","b","c"],attempt=d,early_signal=d,early_bias=[1.,1.1,1.2]))
    picks=observe_top2(f,"early")
    path=[dict(date=days[1],entry_index=10,initial_stop=9,index_close=10,remaining_after=1,fills=[]),
          dict(date=days[2],index_close=11,remaining_after=0,
               fills=[dict(weight=1.,price_index=11,reason="HORIZON")])]
    trades={("a",d):dict(status=BLOCK),("b",d):dict(raw_open=10,path=path),
            ("c",d):dict(raw_open=10,path=path)}
    n,t=portfolio(days,picks,{x:"GREEN" for x in days},trades)
    assert list(t.symbol)==["a","b"] and list(t.status)==[BLOCK,"CLOSED"] and n.open.max()==1
