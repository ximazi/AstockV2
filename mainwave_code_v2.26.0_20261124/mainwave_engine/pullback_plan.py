"""Causal acceptance/retest observations. Fixed research hypotheses, no orders."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

VERSION = "pullback_plan_v1_20260911"
TTL = 20
LABELS = {"NONE": "尚无有效突破", "WAIT_PULLBACK": "突破后等待回踩", "TESTING": "回踩测试中",
          "CONFIRMED": "回踩恢复，等待开盘价格复核", "FAILED": "原突破结构失效", "EXPIRED": "本次回踩过期"}
RULES = {
    "entry": "突破接受后，价格从已知峰值回落至少0.5ATR并回到突破位/MA20附近；回踩成交额较参考推进量收缩至少15%。其后收盘回升、收复突破位和MA20、收盘位置>=50%，无供给警报。",
    "separation": "尚未观察到推进时为接受后回踩；已推进则为推进后回踩。测试与确认不在同一根K线。每次确认后必须形成新的回踩，才产生新的机会编号。",
    "opening": "确认后下一交易所会话开盘：相对确认日MA20偏离<=2ATR，高开不超过0.5ATR，价格高于冻结结构位。超出则取消该次，继续等新确认，不在后续开盘盲目补买。",
    "initial": "初期保留入场价下2ATR绝对风险底线；结构以收盘失效后下一会话退出，不把测试最低价直接当盘中止损。T+1和涨跌停约束仍生效。",
    "profit": "+3ATR兑现50%，+6ATR兑现25%；推进后启用原跟踪保护。持有至少5个活跃日且此前推进>=3ATR后，接近已知高点、长上影和弱收盘触发剩余仓位下一会话开盘退出。",
    "reentry": "退出后继续观察新测试、新确认；每笔独立记录，不重写原交易，也不把多笔价差当组合收益。",
    "boundary": "阈值为待校准研究假设；2026及三个案例已经看过，不是未见OOS。",
}


def annotate(frame: pd.DataFrame, *, strong_reclaim: bool = False) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(index=frame.index)
    a = {k: frame[k].to_numpy() for k in frame}
    days = frame.date.dt.strftime("%Y-%m-%d").to_numpy()
    ma = frame.return_index.rolling(20).mean().to_numpy()
    records, chain, peak, peak_i, test, count = [], "", 0., 0, None, 0
    for i, day in enumerate(days):
        status, confirmed, kind = "NONE", "", ""
        cid = a["breakout_id"][i]
        if cid != chain:
            chain, test, peak, peak_i, count = cid, None, a["index_close"][i], i, 0
        valid = (a["breakout_status"][i] in {"ACCEPTED", "ADVANCING"} and a["eligible"][i]
                 and math.isfinite(a["atr20"][i]) and a["atr20"][i] > 0)
        record = {"plan_id": "", "plan_kind": "", "plan_test_at": "", "plan_confirmed_at": "",
                  "plan_atr": None, "plan_structure": None, "plan_reference": None, "plan_ma20": None,
                  "plan_breakout_id": cid, "plan_status": "NONE", "plan_version": VERSION}
        if not valid:
            test = None
            status = "FAILED" if a["breakout_status"][i] in {"FAILED", "REJECTED"} else "NONE"
        else:
            status = "WAIT_PULLBACK"
            if test:
                status, kind = "TESTING", test["kind"]
                if a["index_close"][i] < test["structure"]:
                    status, test = "FAILED", None
                elif i - test["i"] > TTL:
                    status, test = "EXPIRED", None
                elif (i > test["i"] and a["index_close"][i] > a["index_close"][i - 1]
                      and a["index_close"][i] >= max(test["boundary"], ma[i])
                      and a["close_location"][i] >= .5 and not a["supply_warning"][i]
                      and (not strong_reclaim or (a["index_close"][i] > a["index_high"][i - 1]
                           and a["close_location"][i] >= .6 and a["amount_ratio"][i] >= 1.))):
                    status, confirmed = "CONFIRMED", day
            if test:
                record.update(plan_id=test["id"], plan_kind=kind, plan_test_at=days[test["i"]],
                              plan_confirmed_at=confirmed, plan_atr=float(a["atr20"][i]),
                              plan_structure=float(test["structure"]), plan_reference=float(a["index_close"][i]),
                              plan_ma20=float(ma[i]))
            if confirmed:
                test, peak, peak_i = None, a["index_high"][i], i
            elif test is None and status == "WAIT_PULLBACK" and a["breakout_event"][i] != "ACCEPTED":
                atr, boundary = a["atr20"][i], a["breakout_boundary_index"][i]
                reference_amount = float(np.median(a["amount"][max(0, peak_i - 2):peak_i + 1]))
                if (i > peak_i and peak - a["index_low"][i] >= .5 * atr
                    and a["index_low"][i] <= max(boundary, ma[i]) + atr
                    and a["index_close"][i] >= a["breakout_invalid_index"][i]
                    and a["amount"][i] <= .85 * reference_amount and not a["supply_warning"][i]):
                    count += 1
                    kind = "ACCEPTANCE_RETEST" if a["breakout_status"][i] == "ACCEPTED" else "ADVANCED_RETEST"
                    test = {"id": f"{cid}:retest:{day}:{count}", "kind": kind, "i": i,
                            "boundary": boundary, "structure": a["breakout_invalid_index"][i]}
                    status = "TESTING"
                    record.update(plan_id=test["id"], plan_kind=kind, plan_test_at=day,
                                  plan_structure=float(test["structure"]))
            if a["index_high"][i] > peak:
                peak, peak_i = a["index_high"][i], i
        record["plan_status"] = status
        record["plan_label"] = LABELS[status]
        record["plan_confirmation_rule"] = "STRONG_RECLAIM" if strong_reclaim else "CLOSE_RECOVERY"
        records.append(record)
    return pd.DataFrame(records, index=frame.index)


def opening_check(a: dict, signal: int, entry_price: float) -> str:
    atr, close, ma = (a[k][signal] for k in ("atr20", "index_close", "ma20"))
    if not all(math.isfinite(x) for x in (atr, close, ma, entry_price)) or atr <= 0:
        return "ENTRY_GUARD_DATA"
    if (entry_price - ma) / atr > 2:
        return "ENTRY_OPEN_BIAS"
    if entry_price - close > .5 * atr:
        return "ENTRY_OPEN_GAP"
    structure = a.get("plan_structure", a["breakout_invalid_index"])[signal]
    if structure is None or not math.isfinite(structure):
        structure = a["breakout_invalid_index"][signal]
    if not math.isfinite(structure) or entry_price <= structure:
        return "ENTRY_OPEN_STRUCTURE"
    return ""


def profit_warning(a: dict, i: int, entry: int, price: float, atr: float, prior_peak: float) -> bool:
    high, low, close, op = (a[k][i] for k in ("index_high", "index_low", "index_close", "index_open"))
    width = high - low
    return bool(i - entry + 1 >= 5 and prior_peak >= price + 3 * atr and width > 0
                and high >= prior_peak - .25 * atr and (high - max(op, close)) / width >= .35
                and a["close_location"][i] <= .4 and a["amount_ratio"][i] >= 1.)
