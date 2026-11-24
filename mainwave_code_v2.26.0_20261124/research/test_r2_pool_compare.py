from types import SimpleNamespace

import pandas as pd

from research.r2_pool_compare import executable, first_confirmation, first_entries, onset_audit
from research.r2_pool_history import trend_labels


DAYS = list(pd.bdate_range("2026-01-05", periods=8).strftime("%Y-%m-%d"))
S = "sh_600000"


def bars():
    return {day: SimpleNamespace(day=day, open=10, close=10, preclose=10, amount=1e8,
                                 volume=1e7, isST=0, tradestatus=1, adjustflag=3) for day in DAYS}


def test_first_entries_never_turn_gaps_into_discoveries():
    observed = {d:"archive" for d in (DAYS[0],DAYS[1],DAYS[2],DAYS[4],DAYS[5])}
    members = {(d,"strict"):{S} for d in (DAYS[1],DAYS[2],DAYS[4],DAYS[5])}
    assert list(first_entries(DAYS,observed,members)) == [(1,"strict",S)]


def test_same_episode_dedup_and_no_same_day_or_post_start_credit():
    entries=[(0,"early",S),(2,"early",S),(3,"strict",S),(4,"strict",S)]
    table=onset_audit(entries,{(S,DAYS[3])},DAYS)
    assert table.early.tolist()==[3,3,3] and table.strict.isna().all()


def test_entry_blocked_or_no_bar_never_filled_later():
    a=bars()
    assert executable(S,0,3,DAYS,a,{(S,DAYS[1]):(10,9)})["status"]=="ST_OR_LIMIT_UP"
    del a[DAYS[1]]
    assert executable(S,0,3,DAYS,a,{})["status"]=="ENTRY_DATA_GAP"


def test_actual_gap_open_deferred_sale_friction_and_no_hindsight():
    a=bars()
    a[DAYS[4]].open=a[DAYS[4]].close=9
    a[DAYS[5]].open=8.5
    a[DAYS[5]].preclose=9
    fill=executable(S,0,3,DAYS,a,{})
    assert (fill["entry_day"],fill["exit_day"],fill["buy"],fill["sell"],fill["blocked_exits"]) == (
        DAYS[1],DAYS[5],10,8.5,1)
    assert fill["net_pct"] < -15
    assert executable(S,0,3,DAYS[:5],a,{})["status"]=="PENDING"


def test_exrights_not_reported_as_a_loss():
    a=bars()
    a[DAYS[2]].preclose=8
    assert executable(S,0,3,DAYS,a,{})["status"]=="CORPORATE_ACTION_UNPRICED"


def test_full_40_session_label_and_cooldown():
    dates=list(pd.bdate_range("2025-10-01",periods=145).strftime("%Y-%m-%d"))
    raw=pd.DataFrame(dict(date=dates,close=[10]*100+[13]*45,amount=1e8,pct_chg=0,isST=0))
    starts,mature=trend_labels(raw,S,"2026-01-01")
    assert len(starts)==2 and dates.index(starts[1])-dates.index(starts[0])==20
    assert mature==dates[-41]
    assert not trend_labels(raw.iloc[:99],S,"2026-01-01")[0]


def test_first_confirm_requires_continuous_evidence_until_not_after_trigger():
    seen={DAYS[1]:{S},DAYS[2]:{S}}
    assert first_confirmation(S,DAYS[1:5],seen,{(S,DAYS[2])})==("CONFIRMED",DAYS[2])
    assert first_confirmation(S,DAYS[1:5],seen,{(S,DAYS[4])})==("DATA_WAIT","")
