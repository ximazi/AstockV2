"""Approved close-confirmed exit advice; never a position or order ledger."""
from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D, pullback_plan as PP

VERSION = "exit_extend_close_v1_20260911"
RULES = {
    "hard": "硬底线固定为首仓价-2ATR0，不因强势或补仓向下放宽；跳空、T+1或跌停可能使实际损失超过该距离。",
    "soft": "最高价推进2ATR0后启用收盘保护：常态参考10日低点，强势参考20日低点；已抬高的保护位不回降。盘中触碰不单独退出。",
    "confirm": "收盘跌破此前已知保护位才计数；首次跌破时，前两日有强势证据则固定等待连续2次收盘，否则1次。收复后重置计数。",
    "protect": "保留+3ATR0兑现初始参考仓位50%、+6ATR0兑现25%；持有至少6个活跃交易日且此前推进3ATR0，出现高位量价异常，再于下一交易日减去剩余仓位一半，仅一次。",
    "structure": "放量弱收盘破位、跳空破位或失败反抽，下一交易日退出剩余仓位；硬风险优先，不等待软确认。",
    "extend": "第20个持有交易日收盘，仅市场与个股同强、盈利至少2ATR0、无待退出或待减仓警报，才续持至最多60日；否则下一交易日退出。",
    "strength": "沪深300收盘>MA20>MA60、MA20较5日前上行；个股同时满足该结构且20日涨幅不弱于指数。指数缺失不视为强势。",
}
LABELS = {
    "NO_RECENT_ACCEPTANCE": "无可核对的首仓信号", "WAIT_INITIAL_FILL": "等待首仓成交信息",
    "NO_STRUCTURE": "首仓结构或波动数据不足", "INITIAL_REFERENCE_REJECTED": "参考首仓不合格",
    "DATA_GAP": "日历或行情待核验", "HOLD_REFERENCE": "参考仓位可继续观察",
    "SOFT_CONFIRMING": "收盘保护确认中", "REDUCE_NEXT_OPEN": "下一交易日分批保护",
    "EXIT_NEXT_OPEN": "下一交易日退出剩余", "HARD_EXIT_PENDING": "硬底线触发，待可卖时退出",
    "REFERENCE_CLOSED": "该信号参考周期已结束", "DATA_WAIT": "数据待核验，不作实时裁定",
}
REASONS = {
    "INITIAL_STOP": "硬底线触发", "DEFERRED_STOP": "T+1或跌停延迟硬退出",
    "SOFT_CLOSE_EXIT": "连续收盘确认失守", "EXIT_STRUCTURE": "结构破坏或失败反抽",
    "TIME_REVIEW_EXIT": "第20日未满足续持条件", "HORIZON": "达到60日观察上限",
    "RISK_REDUCTION": "成熟上涨后量价风险保护", "PARTIAL_TARGET": "达到分批兑现档位",
    "TARGET_COMPLETED": "分批兑现后参考仓位已退出",
}


def market_frame(raw, as_of):
    f = raw.copy()
    f["date"] = pd.to_datetime(f.trade_date.astype(str))
    f = f[f.date.le(as_of)].sort_values("date").drop_duplicates("date")
    c = f.close
    ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
    f["market_strong"] = (c > ma20) & (ma20 > ma60) & (ma20 > ma20.shift(5))
    f["market_weak"] = (c < ma20) & (ma20 < ma20.shift(5))
    f["market_ret20"] = c.pct_change(20)
    return f.set_index("date")[["market_strong", "market_weak", "market_ret20"]]


@lru_cache(maxsize=4)
def _load_market(path, modified, as_of):
    return market_frame(pd.read_csv(path), as_of)


def load_market(as_of):
    path = Path(__file__).resolve().parent.parent / "data/000300_SH_index.csv"
    if not path.exists():
        return pd.DataFrame(columns=["market_strong", "market_weak", "market_ret20"], index=pd.DatetimeIndex([]))
    return _load_market(str(path), path.stat().st_mtime_ns, as_of)


def enrich(frame, market):
    f = frame.copy()
    c = f.index_close
    f["ma20"] = c.rolling(20).mean()
    f["low10"] = f.index_low.rolling(10).min()
    f["low20"] = f.index_low.rolling(20).min()
    f["ma60_exit"] = c.rolling(60).mean()
    f["ma20_up"] = f.ma20.gt(f.ma20.shift(5))
    joined = market.reindex(f.date)
    f["market_known"] = joined.market_ret20.notna().to_numpy()
    f["exit_strong"] = (f.market_known & joined.market_strong.eq(True).to_numpy()
                        & c.gt(f.ma20) & f.ma20.gt(f.ma60_exit) & f.ma20_up
                        & c.pct_change(20).ge(joined.market_ret20.to_numpy()))
    f["exit_market_weak"] = joined.market_weak.eq(True).to_numpy()
    width = (f.index_high - f.index_low).replace(0, np.nan)
    wick = (f.index_high - f[["index_close", "index_open"]].max(axis=1)) / width
    f["upper_wicks5"] = (wick.ge(.35) & f.close_location.le(.5)).rolling(5).sum()
    return f


def warning(a, j, entry, price, atr, prior_peak):
    """Close evidence schedules next-session action, not proof of fund intent."""
    cl, op, hi = (a[k][j] for k in ("index_close", "index_open", "index_high"))
    amount, location = a["amount_ratio"][j], a["close_location"][j]
    prior_low = a["low10"][j - 1]
    structural = cl < prior_low and cl < a["ma20"][j]
    heavy_break = structural and amount >= 1.3 and location <= .35
    failed_retest = (j >= entry + 2 and a["index_close"][j - 1] < a["low10"][j - 2]
                     and cl < a["low10"][j - 2] and cl < a["index_close"][j - 1]
                     and cl < a["ma20"][j])
    gap_break = structural and hi < a["index_low"][j - 1] and amount >= 1.3
    if heavy_break or failed_retest or gap_break:
        return "EXIT_STRUCTURE", {"heavy_break": bool(heavy_break), "failed_retest": bool(failed_retest),
                                  "gap_break": bool(gap_break)}
    mature = j - entry >= 5 and prior_peak >= price + 3 * atr
    near_peak = hi >= prior_peak - atr
    wick = PP.profit_warning(a, j, entry, price, atr, prior_peak)
    stall = near_peak and abs(cl - op) <= .25 * atr and amount >= 1.8 and location <= .5
    repeated = near_peak and a["upper_wicks5"][j] >= 3 and amount >= 1.3 and cl < a["index_close"][j - 1]
    if mature and (wick or stall or repeated):
        return "REDUCE_DISTRIBUTION_PROXY", {"weak_peak_close": bool(wick), "volume_stall": bool(stall),
                                             "repeated_wicks": bool(repeated)}
    return "", {}


def close_review(a, j, entry, price, atr, prior_peak, *, soft_stop, breaches=0, required=1,
                 pending_exit="", pending_reduce=False, reduced=False):
    """Shared EXTEND_CLOSE transition, evaluated strictly after the day's close."""
    cl = a["index_close"][j]
    if max(prior_peak, a["index_high"][j]) >= price + 2 * atr:
        if cl < soft_stop:
            if not breaches:
                required = 2 if any(a["exit_strong"][max(entry, j - 2):j]) else 1
            breaches += 1
            if breaches >= required:
                pending_exit = pending_exit or "SOFT_CLOSE_EXIT"
        else:
            breaches = 0
        trail = "low20" if a["exit_strong"][j] else "low10"
        soft_stop = max(soft_stop, float(a[trail][j]))
    action, evidence = warning(a, j, entry, price, atr, prior_peak)
    if action == "EXIT_STRUCTURE":
        pending_exit = pending_exit or action
        pending_reduce = False
    elif action and not reduced and not pending_exit:
        pending_reduce = True
    extension = None
    if j - entry + 1 == 20:
        extension = bool(a["exit_strong"][j] and cl >= price + 2 * atr and not pending_exit and not pending_reduce)
        if not extension:
            pending_exit = pending_exit or "TIME_REVIEW_EXIT"
    return {"soft_stop": float(soft_stop), "breaches": breaches, "required": required,
            "pending_exit": pending_exit, "pending_reduce": bool(pending_reduce),
            "action": action, "evidence": evidence, "extension": extension}


def observe(frame, calendar, symbol, anchor, market):
    """Audit one explicitly named hypothetical entry, without inferring actual holdings."""
    result = {"status": anchor["status"], "reference": {k: v for k, v in anchor.items() if not k.startswith("_") and k != "confirmations"},
              "events": [], "cancel_adds": False, "market_known": False, "strong": False}
    if frame.empty:
        return result
    f = enrich(frame.reset_index(drop=True), market)
    a = {c: f[c].to_numpy() for c in f.columns}
    days = f.date.dt.strftime("%Y-%m-%d").tolist()
    result.update(market_known=bool(f.market_known.iloc[-1]), strong=bool(f.exit_strong.iloc[-1]))
    # Position-free structural risk remains visible even without a usable entry.
    action, evidence = warning(a, len(f) - 1, 0, float("inf"), 1., 0.) if len(f) >= 21 else ("", {})
    result.update(current_structure_alert=action, current_structure_evidence=evidence)
    if anchor["status"] != "REFERENCE_READY":
        return result
    entry, price, atr, scale = (anchor[k] for k in ("_entry", "_price", "_atr", "_scale"))
    hard, soft, peak = price - 2 * atr, price - 2 * atr, price
    pending, reduction, reduced, hard_pending = "", False, False, False
    remaining, breaches, required = 1., 0, 1
    targets = [(price + 3 * atr, .5), (price + 6 * atr, .25)]
    clock, suspended = {day: n for n, day in enumerate(calendar)}, set(frame.attrs.get("suspended_days", []))
    result.update(status="HOLD_REFERENCE", hard_floor=float(hard * scale), soft_floor=float(soft * scale))
    for j in range(entry, len(f)):
        if days[j] not in clock or days[j - 1] not in clock or any(
                day not in suspended for day in calendar[clock[days[j - 1]] + 1:clock[days[j]]]):
            return {**result, "status": "DATA_GAP", "cancel_adds": True, "last_evaluated_date": days[j - 1]}
        op, hi, lo = (a[k][j] for k in ("index_open", "index_high", "index_low"))
        if not all(math.isfinite(a[k][j]) and a[k][j] > 0 for k in
                   ("index_open", "index_high", "index_low", "index_close", "preclose")):
            return {**result, "status": "DATA_GAP", "cancel_adds": True, "last_evaluated_date": days[j - 1]}
        lower = math.floor(a["preclose"][j] * (1 - D.limit_pct(symbol) / 100) * 100 + .5) / 100
        raw_high = a["high"][j] if "high" in a else hi * a["close"][j] / a["index_close"][j]
        locked = raw_high <= lower + .001
        events = result["events"]
        if j > entry and not locked and reduction and not hard_pending and not pending and op > hard:
            events.append({"date": days[j], "reason": "RISK_REDUCTION", "reference_fraction": remaining / 2})
            remaining /= 2
            reduction, reduced = False, True
        reason = ""
        if j == entry:
            hard_pending = lo <= hard
        elif not locked and (hard_pending or pending or lo <= hard):
            reason = "DEFERRED_STOP" if hard_pending else pending or "INITIAL_STOP"
        elif lo <= hard:
            hard_pending = True
        if not reason and j > entry and not locked:
            for level, weight in list(targets):
                if hi >= level:
                    used = min(weight, remaining)
                    events.append({"date": days[j], "reason": "PARTIAL_TARGET", "reference_fraction": used})
                    remaining -= used
                    targets.remove((level, weight))
            if remaining <= 1e-10:
                reason = "TARGET_COMPLETED"
        if not reason and j - entry + 1 >= 60:
            if not locked:
                reason = "HORIZON"
            else:
                pending = pending or "HORIZON"
        result.update(held_sessions=j - entry + 1, last_evaluated_date=days[j], locked=bool(locked))
        if reason:
            if remaining > 1e-10:
                events.append({"date": days[j], "reason": reason, "reference_fraction": remaining})
            return {**result, "status": "REFERENCE_CLOSED", "closed_at": days[j], "reason": reason, "cancel_adds": True}
        review = close_review(a, j, entry, price, atr, peak, soft_stop=soft, breaches=breaches,
                              required=required, pending_exit=pending, pending_reduce=reduction, reduced=reduced)
        soft, breaches, required = (review[k] for k in ("soft_stop", "breaches", "required"))
        pending, reduction = review["pending_exit"], review["pending_reduce"]
        peak = max(peak, hi)
        if review["extension"] is not None:
            result["extension_review"] = {"date": days[j], "extend": review["extension"]}
        result.update(soft_floor=float(soft * scale), soft_breaches=breaches, soft_required=required,
                      pending_exit=pending, pending_reduce=reduction, evidence=review["evidence"],
                      status=("HARD_EXIT_PENDING" if hard_pending else "EXIT_NEXT_OPEN" if pending else
                              "REDUCE_NEXT_OPEN" if reduction else "SOFT_CONFIRMING" if breaches else "HOLD_REFERENCE"),
                      cancel_adds=bool(events or hard_pending or pending or reduction))
    return result


def plan(observation, fresh):
    observation = observation or {"status": "NO_RECENT_ACCEPTANCE", "events": [], "reference": {}}
    status = observation["status"] if fresh else "DATA_WAIT"
    return {"version": VERSION, "mode": "FORMAL_ANALYSIS_POLICY", "orders_enabled": False,
            "rules": RULES, "observation": observation, "status": status, "status_label": LABELS[status],
            "reason_label": REASONS.get(observation.get("pending_exit") or observation.get("reason"), ""),
            "fresh": bool(fresh), "validation": "USER_ADOPTED_NOT_PROVEN_ALPHA",
            "basis": "最近合格信号次日开盘的假设参考周期，不是实际或虚拟持仓台账；更换信号会更换参考，不代表旧仓已卖或新仓已买。实际首仓价、已卖比例与可卖数量须另核。",
            "execution": "收盘警报最早下一交易日处理；硬底线按可成交时处理。T+1、停牌、跌停及跳空需单独复核；任何卖出或减仓后取消补仓。",
            "price_basis": "价格均换算为图表同一复权尺度；不是实际账户成本或成交价。"}
