"""Two R2 observation slots. Reads dated reports only; never grants entry permission."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

from .review_clock import CST


VERSION = "r2_stage_watch_v1"
ROOT = Path("results/mainwave_engine")
WINDOW = 10
ACTIVE = {"ACCEPTED", "ADVANCING"}
TERMINAL = {"FAILED", "REJECTED", "EXPIRED"}
LABEL = {"PREP": "突破前准备", "TREND": "趋势延续", "WAIT": "暂不入榜"}
PB = {"FIRST_CONFIRMED_TODAY": "今日首次回踩确认", "FIRST_CONFIRMED_BEFORE": "首次回踩已在此前确认",
      "FIRST_FAILED": "首次回踩已失败或过期", "FIRST_FORMING": "首次回踩形成中",
      "JUST_ACCEPTED": "今日突破被接受", "NOT_OBSERVED": "尚未记录同轮首次确认",
      "UNKNOWN": "首次回踩身份或历史不足", "NA": "尚无有效突破"}
TRACK = {"TRACKING":"继续观察","EXPIRED":"观察窗结束","CHAIN_ENDED":"原突破终止",
         "PREPARATION_ENDED":"原准备轨迹终止","PARENT_ENDED":"母结构失效","DATA_WAIT":"数据不足，停止串联",**PB}
AUDIT = ("symbol date data_status eligible breakout_status breakout_id breakout_attempt_at breakout_accepted_at "
         "breakout_event acceptance_age preparation_id preparation_origin preparation_valid preparation_first_seen "
         "preparation_parent_id retained supply supply_warning bias_atr").split()
LIFE = "symbol date cycle_data_status cycle_stage breakout_id breakout_accepted_at setup_first_known".split()
HIST = ("symbol date setup_breakout_id setup_breakout_accepted_at setup_first_known setup_kind setup_outcome "
        "setup_started_at setup_confirmed_at setup_ended_at").split()


def table(path, cols):
    return pd.read_csv(path, usecols=cols, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def yes(value):
    return str(value).lower() == "true"


def number(value):
    n = pd.to_numeric(value, errors="coerce")
    return float(n) if pd.notna(n) and math.isfinite(n) else None


def pools(folder, day, *, membership_only=False):
    early_cols=["as_of","symbol","name","industry","detector"]
    early = table(folder/"preparation_watch.csv", early_cols if membership_only else early_cols+["reclaim_failed"])
    strict = table(folder/"carrier_qualification.csv",
                   ["as_of","symbol","name","industry","data_fresh","hard_gate_pass","carrier_qualification"])
    if (set(early.as_of) | set(strict.as_of)) - {day} or set(early.detector) - {"P5_union"}:
        raise ValueError(f"{day}: pool date/detector mismatch")
    strict = strict[strict.data_fresh.map(yes) & strict.hard_gate_pass.map(yes)
                    & strict.carrier_qualification.eq("QUALIFIED_LURK")]
    result = {}
    for source, frame in [("early",early),("strict",strict)]:
        if frame.symbol.duplicated().any():
            raise ValueError(f"{day}: duplicate {source} membership")
        for rank,r in enumerate(frame.itertuples(index=False),1):
            item=result.setdefault(r.symbol,dict(name=r.name,industry=r.industry))
            item[source]=rank
            if source=="early" and not membership_only:
                item["reclaim_failed"]=yes(r.reclaim_failed)
    return result


def pullback(day, a, life, history):
    if not life or (life["breakout_id"],life["breakout_accepted_at"]) != (a["breakout_id"],a["breakout_accepted_at"]):
        return "JUST_ACCEPTED" if a["breakout_accepted_at"]==day else "UNKNOWN"
    if history is None:
        return "UNKNOWN"
    items=history.get((a["symbol"],a["breakout_id"],a["breakout_accepted_at"]), [])
    # Never interpret the lack of a row as proof that a first pullback hasn't happened.
    confirmed=[r.setup_confirmed_at for r in items if r.setup_outcome=="CONFIRMED"
               and r.setup_started_at<=r.setup_confirmed_at<=day]
    if confirmed:
        return "FIRST_CONFIRMED_TODAY" if min(confirmed)==day else "FIRST_CONFIRMED_BEFORE"
    if any(r.setup_outcome in ("FAILED","EXPIRED") and r.setup_started_at<=r.setup_ended_at<=day for r in items):
        return "FIRST_FAILED"
    if any(not r.setup_outcome and r.setup_started_at<=day for r in items):
        return "FIRST_FORMING"
    if day==a["breakout_accepted_at"]:
        return "JUST_ACCEPTED"
    return "NOT_OBSERVED" if yes(life["setup_first_known"]) else "UNKNOWN"


def classify(day, symbol, member, audit, lives, history):
    a=audit.get(symbol,{})
    status=a.get("breakout_status","")
    live=lives.get(symbol,{})
    row=dict(symbol=symbol, **member, lane="WAIT", reason="", event=a.get("breakout_event",""),
             status=status, breakout_id="", accepted="", age=None, pullback="NA",
             prep_id=a.get("preparation_id",""), prep_at=a.get("preparation_first_seen",""),
             retained=yes(a.get("retained")), supply=yes(a.get("supply")), verified_prep=False)
    if a.get("date")!=day or a.get("data_status")!="FRESH" or not yes(a.get("eligible")):
        return dict(row,reason="数据或观察资格不足")
    valid=(yes(a.get("preparation_valid")) and bool(row["prep_id"]) and ""<row["prep_at"]<=day
           and a["preparation_origin"]=="PRE_ATTEMPT")
    row["verified_prep"]=valid
    risk=(a.get("supply_warning")!="False" or member.get("reclaim_failed")
          or live.get("cycle_stage") in {"DISTRIBUTION_RISK","OVERHEATED"})
    if status in ACTIVE:
        age=number(a["acceptance_age"])
        if (not a["breakout_id"] or not (""<a["breakout_accepted_at"]<=day) or age is None
                or age<0 or not age.is_integer() or (age==0)!=(day==a["breakout_accepted_at"])):
            return dict(row,reason="本轮突破编号、日期或年龄不完整")
        pb=pullback(day,a,live,history)
        row.update(lane="TREND",breakout_id=a["breakout_id"],accepted=a["breakout_accepted_at"],
                   age=int(age),pullback=pb)
        if pb in {"FIRST_CONFIRMED_BEFORE","FIRST_FAILED","UNKNOWN"}:
            row["reason"]=PB[pb]
        row["parent_failed"]=(live.get("breakout_id")==row["breakout_id"]
                              and live.get("cycle_stage") in {"FAILED","REBUILDING"})
        if row["parent_failed"]:
            row["reason"]="母结构失效或修复中"
    elif status in TERMINAL:
        # PRE_ATTEMPT at birth, after the old attempt, proves the old break is not being recycled.
        if (not valid or a["preparation_parent_id"] or not (""<a["breakout_attempt_at"]<row["prep_at"])
                or row["event"] in TERMINAL):
            return dict(row,reason="旧突破失效，暂无独立的新准备")
        row["lane"]="PREP"
    elif status in {"NONE","ATTEMPT"}:
        row.update(lane="PREP",breakout_id=a["breakout_id"] if status=="ATTEMPT" else "")
    else:
        return dict(row,reason="未知突破状态")
    if risk:
        row["reason"]="供给警报、收复失败或高位风险"
    return row


def rank_key(r):
    source=(not bool(r.get("strict")),r.get("strict") or r.get("early") or 10**8,r["symbol"])
    if r["lane"]=="PREP":
        return (not r["verified_prep"],not r["retained"],not r["supply"],*source)
    return (r["age"],*source)  # Shorter time since the SAME acceptance, not a new breakout.


def original_slots(current, prev):
    selected=[]
    for src in ("early","strict"):
        ranked=[(r[src],s) for s,r in current.items() if src in r and src not in prev.get(s,{}) and s not in selected]
        if ranked:
            selected.append(min(ranked)[1])
    return selected


def follow(day, days, old, r, a):
    t=dict(old, state="TRACKING", pullback=r["pullback"],age=r.get("age"),transition="")
    if days.index(day)-days.index(t["origin"])>WINDOW:
        return dict(t,state="EXPIRED")
    if a.get("date")!=day or a.get("data_status")!="FRESH":
        return dict(t,state="DATA_WAIT")
    if t["breakout_id"] and (t["breakout_id"]!=a["breakout_id"] or a["breakout_status"] in TERMINAL):
        return dict(t,state="CHAIN_ENDED")
    if t["lane"]=="PREP":
        if a["breakout_event"]=="ACCEPTED" and r["lane"]=="TREND":
            t.update(lane="TREND",breakout_id=r["breakout_id"],accepted=r["accepted"],transition="准备转趋势")
        elif a["breakout_status"] in ACTIVE:
            return dict(t,state="DATA_WAIT")
        elif t["prep_id"] and (not yes(a["preparation_valid"]) or t["prep_id"]!=a["preparation_id"]):
            return dict(t,state="PREPARATION_ENDED")
        elif a["breakout_event"]=="ATTEMPT" and not t["breakout_id"]:
            t["breakout_id"]=a["breakout_id"]
        elif a["breakout_event"] in TERMINAL:
            return dict(t,state="CHAIN_ENDED")
    if t["lane"]=="TREND":
        if r["accepted"]!=t["accepted"]:
            return dict(t,state="DATA_WAIT")
        if r.get("parent_failed"):
            return dict(t,state="PARENT_ENDED")
        if r["pullback"] in {"FIRST_CONFIRMED_TODAY","FIRST_CONFIRMED_BEFORE","FIRST_FAILED"}:
            return dict(t,state=r["pullback"])
    return t


def render(report):
    lines=[f"# R2 生命周期观察 | {report['as_of']}", "",
           "每日新增最多两只，每榜一席，不递补；续踪不重复占席。仅观察，不生成买入或订单。",""]
    for lane in ("PREP","TREND"):
        r=next((x for x in report["picks"] if x["lane"]==lane),None)
        if r is None:
            counts=report.get("counts",{}).get(lane,{})
            lines.append(f"- **{LABEL[lane]}：空缺。** 当日该阶段 {counts.get('total',0)} 只，"
                         f"通过风险及资料检查 {counts.get('clear',0)} 只；无可分配的新观察。"
                         if report["comparable"] else f"- **{LABEL[lane]}：未评。** 输入不可比。")
        elif lane=="PREP":
            lines.append(f"- **突破前准备：{r['name']} {r['symbol']}**。{r['admission']}；"
                         f"准备轨迹 {'已核' if r['verified_prep'] else '未核'}，冲击保留 {int(r['retained'])}，"
                         f"供给收缩 {int(r['supply'])}；等首次突破被接受。")
        else:
            lines.append(f"- **趋势延续：{r['name']} {r['symbol']}**。{r['admission']}；原突破 {r['breakout_id']}，"
                         f"{r['accepted']} 接受，距今 {r['age']} 个个股交易日；{PB[r['pullback']]}。")
    lines.extend(["",f"输入：{report['status']}；池内 {report['pool_size']} 只；观察口径：{report['basis']}。"])
    lines.extend(f"- {i}" for i in report["issues"])
    if report.get("excluded"):
        lines.append("- 未进入两榜的原因："+ "；".join(f"{k} {v} 只" for k,v in report["excluded"].items())+"。")
    lines.append("- [完整池、原两名额对照及未选原因](r2_stage_pool.csv)（不占新增观察名额）。")
    if report["followup"]:
        lines.extend(["","| 观察日 | 标的 | 当前线 | 原接受日 | 同轮回踩 | 续踪状态 |","|---|---|---|---|---|---|"])
        for x in report["followup"]:
            lines.append(f"|{x['origin']}|{x['symbol']}|{x.get('transition') or LABEL[x['lane']]}|"
                         f"{x['accepted'] or '未接受'}|{PB[x['pullback']]}|{TRACK[x['state']]}|")
    return "\n".join(lines)+"\n"


def freeze(out,report,now):
    if (report["basis"]!="当日收盘" or report["status"]=="DATA_WAIT" or not report["comparable"]
            or not report["source_complete"]):
        return
    frozen={**report,"frozen_at":now.isoformat()}
    try:
        with (out/f"{VERSION}.frozen.json").open("x",encoding="utf-8") as f:
            json.dump(frozen,f,ensure_ascii=False,indent=2)
    except FileExistsError:
        pass  # A rerun may refresh diagnostics, never the first published observation.


def run(day, root=ROOT, dest=None):
    root=Path(root)
    replay=dest is not None
    dest=Path(dest) if dest else root
    folder=root/day.replace("-","")
    out=dest/day.replace("-","")
    now=datetime.now(CST)
    today=now.date().isoformat()==day and now.hour>=15
    basis="当日收盘" if today and not replay else "历史重算（非前瞻记录）"
    report=dict(as_of=day,version=VERSION,basis=basis,status="OK",comparable=False,issues=[],picks=[],controls=[],
                followup=[],pool_size=0,source_complete=False,orders_enabled=False)
    rows=[]
    try:
        # The Shanghai index confirms observed sessions; a late CSI300 bar must not erase the date.
        # Market/trade permissions still require their own complete index inputs.
        history=pd.read_csv("data/000001_SH_index.csv",usecols=["trade_date"],dtype=str)
        days=sorted(pd.to_datetime(history.trade_date).dt.strftime("%Y-%m-%d").unique())
        i=days.index(day)
        current=pools(folder,day)
        audit={r["symbol"]:r for r in table(folder/"route_scan_audit.csv",AUDIT).to_dict("records")}
        if not any(a["date"]==day and a["data_status"]=="FRESH" for a in audit.values()):
            raise ValueError(f"{day}: no same-day fresh route audit")
        prev_day=days[i-1] if i else ""
        try:
            prev=pools(root/prev_day.replace("-",""),prev_day,membership_only=True) if prev_day else None
        except (OSError,ValueError):
            prev=None
        if prev is None:
            report["issues"].append("上一交易日两池不可核，不分配新增名额。")
        else:
            report["comparable"]=True
        lives,history={},None
        try:
            life=table(folder/"stock_lifecycle.csv",LIFE)
            lives={r["symbol"]:r for r in life[life.date.eq(day)&life.cycle_data_status.eq("FRESH")].to_dict("records")}
            hist=table(folder/"price_cycle_setups.csv",HIST)
            history={}
            for r in hist[hist.date.le(day)&hist.setup_first_known.map(yes)&hist.setup_kind.eq("FIRST_PULLBACK")].itertuples():
                if r.setup_started_at and r.setup_started_at<=day:
                    history.setdefault((r.symbol,r.setup_breakout_id,r.setup_breakout_accepted_at),[]).append(r)
            report["source_complete"]=bool(lives)
            if not lives:
                report["issues"].append("当日生命周期表无新鲜行。")
        except (OSError,ValueError):
            report["issues"].append("同轮回踩历史不可核，不以缺行当作未回踩。")
        report["pool_size"]=len(current)
        previous=dest/prev_day.replace("-","")/("r2_stage_watch.json" if replay else f"{VERSION}.frozen.json")
        frozen=json.loads(previous.read_text(encoding="utf-8")) if previous.is_file() else {}
        if frozen.get("as_of")!=prev_day or frozen.get("version")!=VERSION or frozen.get("status")=="DATA_WAIT":
            frozen={}
            report["issues"].append("无上个交易日同版本观察记录，不回填续踪。")
        old=[t for t in frozen.get("followup",[]) if t["state"]=="TRACKING"]
        by={s:classify(day,s,current.get(s,{}),audit,lives,history) for s in set(current)|{t["symbol"] for t in old}}
        for t in old:
            s=t["symbol"]
            report["followup"].append(follow(day,days,t,by[s],audit.get(s,{})))
        active={(x["symbol"],x["lane"]) for x in report["followup"] if x["state"]=="TRACKING"}
        rows=[by[s] for s in sorted(current)]
        for r in rows:
            r["admission"]=("" if prev is None else "新入池" if r["symbol"] not in prev else
                            "本轮今日首次接受" if r["lane"]=="TREND" and r["event"]=="ACCEPTED" else
                            "新准备轨迹" if r["lane"]=="PREP" and r["verified_prep"] and r["prep_at"]==day else "")
        for lane in ("PREP","TREND"):
            opts=[r for r in rows if r["lane"]==lane and not r["reason"] and r["admission"]
                  and (r["symbol"],lane) not in active]
            if opts:
                r=min(opts,key=rank_key)
                report["picks"].append(r)
                t={k:r[k] for k in ("symbol","lane","breakout_id","accepted","age","prep_id","pullback")}
                state=r["pullback"] if r["pullback"]=="FIRST_CONFIRMED_TODAY" else "TRACKING"
                report["followup"].append(dict(t,origin=day,state=state))
        report["counts"]={lane:dict(total=sum(r["lane"]==lane for r in rows),
                                    clear=sum(r["lane"]==lane and not r["reason"] for r in rows))
                          for lane in ("PREP","TREND")}
        report["excluded"]=dict(Counter(r["reason"] for r in rows if r["lane"]=="WAIT"))
        report["controls"]=original_slots(current,prev) if prev is not None else []
        picked={r["symbol"] for r in report["picks"]}
        for r in rows:
            s=r["symbol"]
            r["original_slot"]=(report["controls"].index(s)+1 if s in report["controls"] else None)
            r["decision"]=("新增观察" if s in picked else r["reason"] or
                           ("已在续踪" if (s,r["lane"]) in active else
                            "无新入池或阶段事件" if not r["admission"] else "同榜另一标的优先"))
        if report["issues"]:
            report["status"]="PARTIAL"
    except (OSError,ValueError) as exc:
        report.update(status="DATA_WAIT",comparable=False,picks=[],controls=[],followup=[])
        report["issues"].append(str(exc))
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows,columns=list(dict.fromkeys(["symbol","name","lane","reason","admission",
                 *(k for r in rows for k in r)]))).to_csv(out/"r2_stage_pool.csv",index=False,encoding="utf-8-sig")
    (out/"r2_stage_watch.md").write_text(render(report),encoding="utf-8")
    (out/"r2_stage_watch.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    freeze(out,report,now)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--date",required=True)
    p.add_argument("--out-root",type=Path)
    args=p.parse_args()
    result=run(args.date,dest=args.out_root)
    print(render(result))
