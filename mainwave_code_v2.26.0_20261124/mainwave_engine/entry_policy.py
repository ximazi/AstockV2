"""Formal allocation advice; observations never imply an executed position."""
from __future__ import annotations

import math
import numpy as np

VERSION = "entry_50_reclaim_v2_20260911"
INITIAL = .5
ADD_WEIGHTS = (.5 * 3 / 7, .5 * 4 / 7)
PULLBACKS = (.03, .06)
WINDOW = 7
RULES = {
    "initial": "信号成立后下一交易日复核开盘价，首仓为单标的计划预算的50%；超价、风险或数据门槛不通过则不买。",
    "add": "首仓后7个交易日内，分别回撤3%/6%，先有缩量测试、后有再转强，余仓按3:4分配（21.43%/28.57%）。无机会不补。",
    "confirmation": "测试发生在此前交易日；缩量不超过此前5日成交额中位数的85%；恢复日低点不降低、收盘突破前日高点且站上MA20、收盘位置≥60%、量比≥1，无供给风险。",
    "opening": "确认后下一交易日才复核补仓，开盘须高于结构与保护位，且不超过确认收盘+0.5ATR、MA20+2ATR、首仓价+0.5ATR三者最小值；不追价。",
    "cancel": "已有卖出/减仓、结构失效、窗口到期，取消未完成补仓；每档最多一次，不借用新周期补旧仓。",
}


def recovery_evidence(a, j):
    close, low = a["index_close"][j], a["index_low"][j]
    healthy = a["eligible"][j] and not a["supply_warning"][j]
    recovery = (healthy and low >= a["index_low"][j - 1] and close > a["index_close"][j - 1]
                and a["close_location"][j] >= .5)
    strong = (recovery and close > a["index_high"][j - 1] and close >= a["ma20"][j]
              and a["close_location"][j] >= .6 and a["amount_ratio"][j] >= 1.)
    amount = float(np.median(a["amount"][max(0, j - 5):j]))
    contracted = amount > 0 and a["amount"][j] <= .85 * amount
    return bool(recovery), bool(strong), bool(contracted)


def order_ceiling(close, ma20, atr, initial_price=None):
    ceiling = min(close + .5 * atr, ma20 + 2 * atr)
    return min(ceiling, initial_price + .5 * atr) if initial_price is not None else ceiling


def reference(frame, calendar, symbol, setup=None):
    """Shared hypothetical entry anchor; not proof of a historical buy permission."""
    from . import data as D
    result = {"status": "NO_RECENT_ACCEPTANCE", "confirmations": [], "reference_basis": "HYPOTHETICAL_NEXT_OPEN_NOT_POSITION"}
    if frame.empty:
        return result
    frame = frame.reset_index(drop=True)
    days = frame.date.dt.strftime("%Y-%m-%d").tolist()
    accepted = frame.index[frame.accepted_bias2]
    i = int(accepted[-1]) if len(accepted) else -1
    setup = setup or {}
    confirmed_at = setup.get("setup_confirmed_at")
    from_setup = (setup.get("setup_status") in {"CONFIRMED", "HISTORY_CONFIRMED"} and confirmed_at in days
                  and days.index(confirmed_at) >= i and setup.get("setup_breakout_id")
                  == frame.breakout_id.iloc[days.index(confirmed_at)])
    if from_setup:
        i = days.index(confirmed_at)
    if i < 0:
        return result
    last = len(frame) - 1
    result.update(signal_date=days[i], breakout_id=str(frame.breakout_id.iloc[i]),
                  signal_kind=setup.get("setup_kind") if from_setup else "BREAKOUT_ACCEPTANCE")
    if i == last:
        return {**result, "status": "WAIT_INITIAL_FILL"}
    entry = i + 1
    clock = {day: n for n, day in enumerate(calendar)}
    if days[entry] not in clock or days[i] not in clock or days[last] not in clock:
        return {**result, "status": "DATA_GAP"}
    if clock[days[entry]] != clock[days[i]] + 1:
        return {**result, "status": "DATA_GAP"}
    age = clock[days[last]] - clock[days[entry]]
    result.update(reference_entry_date=days[entry], age_sessions=age)
    a = {c: frame[c].to_numpy() for c in frame.columns}
    a["ma20"] = frame.index_close.rolling(20).mean().to_numpy()
    price, atr, structure = a["index_open"][entry], a["atr20"][i], a["breakout_invalid_index"][i]
    scale = float(a["close"][last] / a["index_close"][last])
    if from_setup:
        atr = float(setup.get("setup_frozen_atr_price") or 0) / scale
        structure = float(setup.get("setup_invalid_price") or 0) / scale
    if not all(math.isfinite(x) and x > 0 for x in (price, atr, structure)):
        return {**result, "status": "NO_STRUCTURE"}
    ceiling = order_ceiling(a["index_close"][i], a["ma20"][i], atr)
    upper_limit = math.floor(a["preclose"][entry] * (1 + D.limit_pct(symbol) / 100) * 100 + .5) / 100
    if price > ceiling or price <= structure or a["open"][entry] >= upper_limit - .001 or a["isST"][entry] != 0:
        return {**result, "status": "INITIAL_REFERENCE_REJECTED"}
    result.update(reference_entry_price=float(price * scale), status="REFERENCE_READY",
                  _entry=entry, _signal=i, _price=float(price), _atr=float(atr), _structure=float(structure), _scale=scale)
    return result


def observe(frame, calendar, symbol, setup=None, *, anchor=None, exit_observation=None):
    """Conditional add evidence for the latest accepted episode, not a trade replay."""
    anchor = reference(frame, calendar, symbol, setup) if anchor is None else anchor
    result = {k: v for k, v in anchor.items() if not k.startswith("_")}
    if result.get("age_sessions", 0) >= WINDOW:
        return {**result, "status": "WINDOW_EXPIRED"}
    if result["status"] != "REFERENCE_READY":
        return result
    from . import exit_policy as XP
    if exit_observation is None:
        exit_observation = XP.observe(frame, calendar, symbol, anchor, XP.load_market(str(frame.date.iloc[-1].date())))
    if exit_observation["status"] == "DATA_GAP":
        return {**result, "status": "DATA_GAP"}
    if exit_observation.get("cancel_adds"):
        return {**result, "status": "SELL_REFERENCE_CANCELLED"}
    frame = frame.reset_index(drop=True)
    a = {c: frame[c].to_numpy() for c in frame.columns}
    a["ma20"] = frame.index_close.rolling(20).mean().to_numpy()
    days = frame.date.dt.strftime("%Y-%m-%d").tolist()
    clock = {day: n for n, day in enumerate(calendar)}
    i, entry, price, atr, structure, scale = (anchor[k] for k in ("_signal", "_entry", "_price", "_atr", "_structure", "_scale"))
    last = len(frame) - 1
    result["status"] = "WAIT_PULLBACK_RECLAIM"
    tests, stop = [None, None], price - 2 * atr
    suspended = set(frame.attrs.get("suspended_days", []))
    for j in range(entry, last + 1):
        missing = calendar[clock[days[j - 1]] + 1:clock[days[j]]] if days[j - 1] in clock else ["UNKNOWN"]
        if any(day not in suspended for day in missing):
            return {**result, "status": "DATA_GAP"}
        if (a["breakout_id"][j] != a["breakout_id"][i] or a["breakout_status"][j] not in {"ACCEPTED", "ADVANCING"}
                or a["index_close"][j] < structure):
            return {**result, "status": "STRUCTURE_INVALID"}
        if j > entry:
            _, strong, contracted = recovery_evidence(a, j)
            for k, pullback in enumerate(PULLBACKS):
                test = tests[k]
                if j == last and test and test["contracted"] and strong:
                    upper = order_ceiling(a["index_close"][j], a["ma20"][j], atr, price)
                    floor = max(structure, stop, exit_observation.get("soft_floor", stop * scale) / scale)
                    result["confirmations"].append({"tranche": k + 2, "budget_fraction": ADD_WEIGHTS[k],
                        "touch_date": test["day"], "confirm_date": days[j], "min_price": float(floor * scale),
                        "max_price": float(upper * scale), "execution": "NEXT_EXCHANGE_OPEN_ONLY"})
                if a["index_low"][j] <= price * (1 - pullback) and price * (1 - pullback) > stop:
                    if test is None:
                        tests[k] = {"day": days[j], "contracted": contracted}
                    elif contracted:
                        test["contracted"] = True
    if result["confirmations"]:
        result["status"] = ("CONDITIONAL_ADD_REVIEW" if any(c["max_price"] > c["min_price"] for c in result["confirmations"])
                            else "EXECUTION_BAND_EMPTY")
    return result


def plan(observation, advice, risks, route_ready, fresh):
    observation = observation or {"status": "NO_RECENT_ACCEPTANCE", "confirmations": []}
    blocked = not fresh or bool(risks) or not route_ready or advice in {"DATA_WAIT", "HISTORICAL_ONLY"}
    return {"version": VERSION, "mode": "FORMAL_ANALYSIS_POLICY", "orders_enabled": False,
            "initial_fraction": INITIAL, "add_fractions": list(ADD_WEIGHTS), "rules": RULES,
            "initial_review": not blocked and advice == "PRIORITY_REVIEW", "observation": observation,
            "add_review": not blocked and observation["status"] == "CONDITIONAL_ADD_REVIEW",
            "blocked": blocked, "block_reasons": [r["label"] for r in risks] + ([] if fresh else ["数据或观察模式待核验"])
                + ([] if route_ready else ["线路必要证据不足"]),
            "position_requirements": "仅适用于该信号次日已有首仓、该档尚未买入、且未触发任何卖出或减仓的情形；实际持仓记录须另行核对。",
            "allocation_basis": "单标的计划预算，不是账户总仓位；未补仓资金保留现金。"}
