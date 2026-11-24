import numpy as np
import pandas as pd

from research.r1_early_demand_replay import attempt_plan, normalize_suspensions, observe_top2


D=("2026-09-01","2026-09-02","2026-09-03","2026-09-04")


def tape(*, states=("ATTEMPT","ATTEMPT","ACCEPTED","ADVANCING"),
         pulses=("DEMAND_EXPANSION","DEMAND_EXPANSION","DEMAND_EXPANSION",""),
         bias=(1.8,1.9,1.9,2.5), eligible=(True,True,True,True)):
    return {k:np.array(v) for k,v in {
        "day":D,"breakout_id":("cid",)*4,"breakout_status":states,
        "effort_result":pulses,"bias_atr":bias,"eligible":eligible,
        "supply_warning":(False,)*4,"accepted_bias2":(False,False,True,False),
        "accepted_demand":(False,False,True,False),
    }.items()}


def test_failed_attempt_still_fires_early_and_future_does_not_change_first_signal():
    t=tape()
    assert attempt_plan(t,0,D)[:2]==(0,2)
    t["breakout_status"]=np.array(["ATTEMPT","ATTEMPT","REJECTED","FAILED"])
    aa,bb,e=attempt_plan(t,0,D)
    assert (aa,bb,e["terminal"])==(0,None,"REJECTED")
    prefix={k:v[:1] for k,v in t.items()}
    assert attempt_plan(prefix,0,D)[:2]==(0,None)


def test_pulse_on_acceptance_is_not_an_early_signal_and_no_post_acceptance_retiming():
    t=tape(pulses=("","","DEMAND_EXPANSION","DEMAND_EXPANSION"))
    assert attempt_plan(t,0,D)[:2]==(None,2)
    t["accepted_bias2"][2]=False
    assert attempt_plan(t,0,D)[2]["wait_note"]=="BIAS_ABOVE_2"


def test_first_qualifying_pulse_no_reprice_and_gap_keeps_unresolved():
    t=tape(bias=(2.2,1.8,1.9,2.5))
    assert attempt_plan(t,0,D)[:2]==(1,2)
    cut={k:v[[0,2,3]] for k,v in t.items()}
    assert attempt_plan(cut,0,D)[2]["terminal"]=="ATTEMPT_DATA_GAP"
    assert attempt_plan(cut,0,D)[1] is None


def test_top_two_rank_then_code_does_not_look_at_returns_or_fills():
    f=pd.DataFrame(dict(symbol=["b","a","c"],attempt=[D[0]]*3,early_signal=[D[0]]*3,
                       early_bias=[1.,1.,1.9], early_net=[0,-5,100],early_status=["CLOSED","ENTRY_LIMIT_UP","CLOSED"]))
    assert [p["symbol"] for p in observe_top2(f,"early")[D[0]]]==["a","b"]


def test_only_explicit_suspension_with_blank_amount_can_bridge_gap():
    raw=pd.DataFrame(dict(tradestatus=[0,1,np.nan,0],amount=[np.nan,np.nan,np.nan,150.]))
    fixed=normalize_suspensions(raw)
    assert fixed.amount.iat[0]==0 and fixed.amount.iloc[1:3].isna().all()
    assert fixed.amount.iat[3]==150. and raw.amount.iat[0]!=raw.amount.iat[0]
