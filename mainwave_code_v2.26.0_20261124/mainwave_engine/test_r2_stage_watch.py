import json
from datetime import datetime
from types import SimpleNamespace

import pandas as pd

from mainwave_engine import r2_stage_watch as W
from research.r2_stage_replay import published


D="2026-11-17"
S="sh_600001"


def audit(s=S,**fields):
    r=dict.fromkeys(W.AUDIT,"")
    r.update(symbol=s,date=D,data_status="FRESH",eligible="True",breakout_status="NONE",
             supply_warning="False",retained="True",supply="True")
    return dict(r,**fields)


def active(s=S,**fields):
    return audit(s,**dict(dict(breakout_status="ADVANCING",breakout_id="cycle1",
                              breakout_accepted_at="2026-11-13",acceptance_age="2"),**fields))


def test_failed_structure_needs_independent_dated_preparation():
    r=audit(breakout_status="FAILED",breakout_attempt_at="2026-11-12",preparation_origin="PRE_ATTEMPT",
            preparation_valid="True",preparation_id="p",preparation_first_seen="2026-11-16")
    def lane(**kw):
        return W.classify(D,S,{}, {S:dict(r,**kw)}, {},{})["lane"]
    assert lane()=="PREP"
    for change in [dict(preparation_parent_id="old"),dict(preparation_first_seen="2026-11-10"),
                   dict(preparation_valid="False"),dict(breakout_event="FAILED")]:
        assert lane(**change)=="WAIT"


def test_first_pullback_requires_same_dated_breakout_and_is_not_repeated():
    a=active()
    life=dict(a,setup_first_known="True")
    event=SimpleNamespace(setup_outcome="CONFIRMED",setup_started_at="2026-11-16",
                          setup_confirmed_at=D,setup_ended_at=D)
    key=(S,"cycle1","2026-11-13")
    assert W.pullback(D,a,life,{key:[event]})=="FIRST_CONFIRMED_TODAY"
    assert W.pullback("2026-11-18",a,life,{key:[event]})=="FIRST_CONFIRMED_BEFORE"
    assert W.pullback(D,a,life,{(S,"old",key[2]):[event]})=="NOT_OBSERVED"
    assert W.pullback(D,a,dict(life,breakout_id="old"),{key:[event]})=="UNKNOWN"
    assert W.pullback(D,a,life,None)=="UNKNOWN"
    assert W.pullback(D,a,dict(life,setup_first_known="False"),{})=="UNKNOWN"
    assert W.classify(D,S,{}, {S:active(acceptance_age="-1")},{},{})["lane"]=="WAIT"


def test_follow_binds_attempt_and_keeps_origin_ends_on_replacement_or_gap():
    dates=["2026-11-13","2026-11-16",D]
    prev=dict(origin=dates[0],symbol=S,lane="PREP",prep_id="p",breakout_id="attempt1",accepted="")
    a=active(breakout_id="attempt1",breakout_accepted_at=D,breakout_event="ACCEPTED",acceptance_age="0")
    r=W.classify(D,S,{}, {S:a},{},{})
    t=W.follow(D,dates,prev,r,a)
    assert t["transition"]=="准备转趋势" and t["origin"]==dates[0] and t["accepted"]==D
    assert W.follow(D,dates,prev,r,dict(a,breakout_id="attempt2"))["state"]=="CHAIN_ENDED"
    assert W.follow(D,dates,prev,r,{})["state"]=="DATA_WAIT"
    assert W.follow(D,dates,t,dict(r,parent_failed=True),a)["state"]=="PARENT_ENDED"


def test_freeze_is_first_publication_only_and_never_backfills(tmp_path):
    r=dict(as_of=D,version=W.VERSION,basis="当日收盘",status="PARTIAL",comparable=True,source_complete=True,picks=["first"])
    now=datetime(2026,11,17,19,tzinfo=W.CST)
    for change in [dict(basis="历史重算"),dict(comparable=False),dict(source_complete=False)]:
        W.freeze(tmp_path,dict(r,**change),now)
        assert not list(tmp_path.iterdir())
    W.freeze(tmp_path,r,now)
    W.freeze(tmp_path,dict(r,picks=["later"]),now)
    assert json.loads(next(tmp_path.iterdir()).read_text(encoding="utf-8"))["picks"]==["first"]
    assert published(tmp_path,D)["picks"]==["first"]
    assert not published(tmp_path,"2026-11-18")["comparable"]


def test_two_slots_no_backfill_when_all_new_names_share_a_lane(tmp_path):
    source=tmp_path/"source"
    out=tmp_path/"out"
    for dt in ("2026-11-16",D):
        folder=source/dt.replace("-","")
        folder.mkdir(parents=True)
        symbols=[S] if dt<D else [S,"sh_600002","sh_600003"]
        pd.DataFrame([dict(as_of=dt,symbol=s,name=s,industry="test",detector="P5_union",reclaim_failed=False)
                      for s in symbols]).to_csv(folder/"preparation_watch.csv",index=False)
        pd.DataFrame(columns=["as_of","symbol","name","industry","data_fresh","hard_gate_pass",
                              "carrier_qualification"]).to_csv(folder/"carrier_qualification.csv",index=False)
    pd.DataFrame([audit(s) for s in symbols]).to_csv(folder/"route_scan_audit.csv",index=False)
    r=W.run(D,root=source,dest=out)
    assert r["status"]=="PARTIAL" and [x["symbol"] for x in r["picks"]]==["sh_600002"]
    assert r["orders_enabled"] is False and not (folder/"r2_stage_watch.json").exists()
    assert W.run("2026-11-18",root=source,dest=out)["status"]=="DATA_WAIT"


def test_original_control_keeps_per_pool_admission_and_deduplicates():
    current={"a":{"early":1,"strict":1},"b":{"early":2},"c":{"strict":2}}
    assert W.original_slots(current,{"a":{"strict":2}})==["a","c"]
    assert W.original_slots(current,{"a":{"early":2}})==["b","a"]
