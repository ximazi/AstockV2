from copy import deepcopy

import pandas as pd

from research.r2_top2_timing import budget, matched, watch, summarize, tracked_pairs


D=list(pd.bdate_range("2025-03-03",periods=34).strftime("%Y-%m-%d"))
S="sh_600001"


def timeline():
    row=dict(breakout_id="a",breakout_status="ADVANCING",breakout_event="",setup_status="NONE",
             setup_kind="",setup_first_known=False,setup_breakout_id="",setup_breakout_accepted_at="",
             setup_confirmed_at="",setup_turnover_contraction="")
    tape={S:{d:deepcopy(row) for d in D}}
    tape[S][D[2]].update(breakout_status="ACCEPTED",breakout_event="ACCEPTED")
    return tape


def pull(t,j):
    t[S][D[j]].update(setup_status="CONFIRMED",setup_kind="FIRST_PULLBACK",setup_first_known=True,
                      setup_breakout_id="a",setup_breakout_accepted_at=D[2],setup_confirmed_at=D[j],
                      setup_turnover_contraction=.82)


def test_strategy_budget_and_peer_same_board_industry_and_liquidity():
    assert budget(["a","b"],["a","c"])["one_each"]==[("early","a"),("strict","c")]
    assert budget(["a","b"],["a"])["one_each"]==[("early","a")]
    ss=[S,"sh_688111","sz_000002","sh_600003","sh_600004"]
    info=dict.fromkeys(ss,"chip")
    info[ss[2]]="auto"
    assert matched(S,ss,{S},info,dict(zip(ss,[100,100,100,210,150])))==ss[-1]


def test_arms_share_breakout_and_past_outcomes_are_not_new_confirmations():
    t=timeline()
    pull(t,5)
    expected=(("TRIGGER",D[2]),("TRIGGER",D[5]))
    assert watch(S,0,D,t)==expected
    assert watch(S,0,D[:6],t)==expected
    for field,value in [("setup_first_known",False),("setup_kind","LATER_PULLBACK"),
                        ("setup_confirmed_at",D[4]),("setup_turnover_contraction",1.2),
                        ("setup_breakout_id","other"),("setup_breakout_accepted_at",D[1])]:
        x=deepcopy(t)
        x[S][D[5]][field]=value
        assert watch(S,0,D,x)[1]==("NO_FIRST_PULLBACK","")


def test_missing_evidence_censors_only_unresolved_arm_and_expiry_not_a_loss():
    t=timeline()
    del t[S][D[3]]
    assert watch(S,0,D,t)==(("TRIGGER",D[2]),("DATA_WAIT",""))
    assert watch(S,0,D[:3],t)==(("TRIGGER",D[2]),("PENDING",""))
    t=timeline()
    t[S][D[2]]["breakout_event"]=""
    t[S][D[11]]["breakout_event"]="ACCEPTED"
    assert watch(S,0,D,t,20)==(("NO_ACCEPTANCE",""),("NO_ACCEPTANCE",""))


def test_broken_chain_never_rolls_into_next_breakout_and_extended_clock_starts_at_accept():
    t=timeline()
    pull(t,22)
    assert watch(S,0,D,t)[1]==("NO_FIRST_PULLBACK","")
    assert watch(S,0,D,t,20)[1]==("TRIGGER",D[22])
    t[S][D[5]]["breakout_status"]="FAILED"
    assert watch(S,0,D,t,20)[1]==("BROKEN_CHAIN","")


def test_existing_breakout_can_confirm_only_after_selection():
    t=timeline()
    pull(t,5)
    incumbent=("a",D[2])
    assert watch(S,3,D,t,incumbent=incumbent)[1]==("TRIGGER",D[5])
    assert watch(S,5,D,t,incumbent=incumbent)[1]==("NO_FIRST_PULLBACK","")


def test_idle_unknown_and_both_filled_have_separate_denominators():
    rows=[]
    for j,(a,b) in enumerate([(-1.,2.),(3.,"NO_FIRST_PULLBACK"),("DATA_WAIT","DATA_WAIT")]):
        for arm,val in [("accept",a),("pullback",b)]:
            closed=isinstance(val,float)
            rows.append(dict(basis="price_replay",policy="early2",extra=0,h=3,arm=arm,mature=True,
                             origin=D[j],symbol=S,watch_state="TRIGGER" if closed else val,
                             status="CLOSED" if closed else val,net_pct=val if closed else None))
    s=summarize(pd.DataFrame(rows)).set_index("arm")
    b=s.loc["pullback"]
    assert (b.closed,b.unknown,b.idle,b.opportunity_mean)==(1,1,1,1.)
    assert (b.paired_n,b.paired_delta,b.paired_better)==(1,3.,1)


def test_secondary_tracking_uses_only_previously_seen_symbols_and_same_chain(monkeypatch):
    from research import r2_top2_timing as study
    monkeypatch.setattr(study,"executable",lambda *a:dict(status="CLOSED",net_pct=1.,entry_day="",exit_day="",buy=1,sell=1))
    t=timeline()
    t[S][D[0]].update(breakout_event="ACCEPTED",breakout_status="ACCEPTED")
    pull(t,5)
    picks=[("early2_history",d,S,"early",S) for d in (D[1],D[5])]
    f=tracked_pairs(picks,D,t,{S:{}},{})
    assert set(f.accepted)=={D[2]} and set(f.origin)=={D[1]} and set(f.pullback)=={D[5]}
    assert set(f.h)=={3,5,10}
