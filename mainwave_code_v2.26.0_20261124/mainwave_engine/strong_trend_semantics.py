"""CORE-SEMANTICS beside CORE-V1 (screening quality task book v1.1, first batch).

Same CT20/RA3 detectors as mainwave_engine.strong_trend, but signals are read as events of a structure:
  NEW       first trigger with no live setup
  CHILD     a live setup paused >= 3 bars since its last event, then a new REACCELERATION: a new consolidation
  RETRY     the live setup fell back below its boundary, then triggered again
  PROGRESS  the live setup simply went on (new high inside the same structure): a state update, not a new entry
A setup ends when a close falls below its invalidation. Setup identity = symbol + detector + first trigger day +
entry type (no prices), so a rolling high leaving the window or an ex-rights day never mints a new setup.
History is replayed from the first bar in the file, so appending a new day never changes an earlier verdict.

Also: next-open checks of the previous session's plans, executable vs research domain, PASS/FAIL/UNKNOWN review
items with evidence, and a structure-quality card that is recorded, never used as a gate. No orders.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import strong_trend as ST

VERSION = "strong_trend_semantics_v1_20260924"
DETECTOR = "CT20_RA3_" + ST.VERSION
CHILD_PAUSE = 3
POLICY_PENDING = "ACCOUNT_POLICY_PENDING：新核心尚未获准进入任何模拟账户，计划只能是条件预案"
KINDS = {"NEW": "新结构", "CHILD": "新整理子结构", "RETRY": "回落后重试", "PROGRESS": "旧结构延伸"}


def load(path, day):
    raw = pd.read_csv(path, usecols=lambda c: c in {"date", "open", "high", "low", "close", "preclose", "volume", "amount", "tradestatus", "isST"})
    raw["date"] = raw["date"].astype(str).str[:10]
    raw = raw[raw["date"] <= day].sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return raw


def bar_signals(f, atr):
    """Per-bar CT20/RA3 flags, vectorised, identical to strong_trend.breakout_at / reacceleration_at."""
    high, low, close = (f[c].to_numpy(float) for c in ("high", "low", "close"))
    n = len(f)
    idx = np.arange(n)
    prev_atr = np.r_[np.nan, atr[:-1]]
    ok = np.isfinite(prev_atr) & (prev_atr > 0)
    level = pd.Series(high).shift(1).rolling(20).max().to_numpy()
    bo = ok & (idx >= 20) & (close > level)
    inv = level - ST.PARAMS["breakout_invalidation_below_level_atr"] * prev_atr
    top = pd.Series(high).shift(1).rolling(3).max().to_numpy()
    bottom = pd.Series(low).shift(1).rolling(3).min().to_numpy()
    closes = pd.Series(close)
    down = (closes.diff() < 0).astype(float)
    downs = down.shift(1).rolling(3).sum().to_numpy()
    span = (closes.shift(1).rolling(3).max() - closes.shift(1).rolling(3).min()).to_numpy()
    paused = (downs > 0) | (span <= ST.PARAMS["sideways_close_range_max_atr"] * prev_atr)
    candidate = ok & (idx >= 24) & paused & (close > top)
    anchor = np.full(n, -1)
    for i in np.nonzero(candidate)[0]:
        for b in range(i - 4, max(i - ST.PARAMS["reacceleration_anchor_lookback"], ST.PARAMS["breakout_lookback"]) - 1, -1):
            if bo[b] and bottom[i] > inv[b]:
                anchor[i] = b
                break
    # One tick below the base low is a raw-price tick on the trigger day, so a later ex-rights day scales it with the
    # rest of history instead of shifting it (audit probe 5). With no corporate action factor is 1 and this equals CORE-V1.
    factor = f["adj_factor"].to_numpy(float) if "adj_factor" in f else np.ones(n)
    return {"bo": bo, "level": level, "inv": inv, "ra": anchor >= 0, "anchor": anchor, "top": top,
            "ra_inv": bottom - ST.PARAMS["tick"] * factor, "bottom": bottom, "downs": downs, "n0": prev_atr}


def setup_id(symbol, first_day, entry):
    return hashlib.sha1(f"{symbol}|{DETECTOR}|{first_day}|{entry}".encode()).hexdigest()[:12]


def setup_events(f, sig, symbol):
    """Sequential replay over the whole history. Returns (events, live setup at the last bar or None)."""
    close, dates = f["close"].to_numpy(float), f["date"].tolist()
    events, live = [], None
    for i in range(len(f)):
        fired_bo, fired_ra = bool(sig["bo"][i]), bool(sig["ra"][i])
        if fired_bo or fired_ra:
            entry = "REACCELERATION" if fired_ra else "BREAKOUT"
            boundary = float(sig["top"][i] if fired_ra else sig["level"][i])
            invalid = float(sig["ra_inv"][i] if fired_ra else sig["inv"][i])
            if live is None:
                kind = "NEW"
            elif fired_ra and i - CHILD_PAUSE > live["last_event"]:
                kind = "CHILD"
            elif live["fell_back"]:
                kind = "RETRY"
            else:
                kind = "PROGRESS"
            previous = live.get("last_entry") if live else None
            if kind in ("NEW", "CHILD"):
                parent = live["setup_id"] if live else None
                live = {"setup_id": setup_id(symbol, dates[i], entry), "parent_id": parent, "first_index": i,
                        "first_day": dates[i], "entry": entry, "boundary": boundary, "invalidation": invalid,
                        "last_event": i, "fell_back": False, "events": 0}
            else:
                live["last_event"], live["fell_back"] = i, False
            live["events"] += 1
            if kind != "PROGRESS":
                live["last_entry"] = {"date": dates[i], "invalidation": invalid, "kind": kind}
            events.append({"index": i, "date": dates[i], "kind": kind, "setup_id": live["setup_id"], "parent_id": live["parent_id"],
                           "entry": entry, "labels": [x for x, y in (("BREAKOUT", fired_bo), ("REACCELERATION", fired_ra)) if y],
                           "boundary": boundary, "invalidation": invalid, "n0": float(sig["n0"][i]),
                           "setup_boundary": live["boundary"], "setup_invalidation": live["invalidation"],
                           "setup_first_day": live["first_day"], "event_age": i - live["first_index"],
                           # Q-09: how this entry relates to the previous entry plan of the same (or parent) setup.
                           "previous_entry": previous if kind in ("CHILD", "RETRY") else None,
                           "wider_than_previous_stop": bool(previous and kind == "RETRY" and invalid < previous["invalidation"])})
        elif live is not None:
            if close[i] < live["invalidation"]:
                live = None
            elif close[i] < live["boundary"]:
                live["fell_back"] = True
    return events, live


def quality(f, sig, atr_tr, e):
    """Recorded features only (task book 4.2); none of them is a gate in this version."""
    i, n0 = e["index"], e["n0"]
    high, low, close, vol = (f[c].to_numpy(float) for c in ("high", "low", "close", "volume"))
    upper = close[i] + ST.PARAMS["entry_upper_buffer_atr"] * n0
    card = {"excess_n": round((close[i] - e["boundary"]) / n0, 3),
            "close_location": None if high[i] <= low[i] else round((close[i] - low[i]) / (high[i] - low[i]), 3),
            "volume_ratio20": round(vol[i] / vol[i - 20:i].mean(), 3) if i >= 20 and vol[i - 20:i].mean() > 0 else None,
            "late_distance_n": round((upper - e["boundary"]) / n0, 3),
            "structure_width_n": round((e["boundary"] - e["invalidation"]) / n0, 3),
            "risk_pct_at_upper": round((upper - e["invalidation"]) / upper * 100, 2),
            "event_age": e["event_age"]}
    b = int(sig["anchor"][i])
    if b >= 0:
        level = float(sig["level"][b])
        push_high = float(high[b:i - 3].max()) if i - 3 > b else float(high[b])
        card.update(down_closes=int(sig["downs"][i]),
                    anchor_age=i - b,
                    push_n=round((push_high - level) / n0, 3),
                    retention=round((sig["bottom"][i] - level) / (push_high - level), 3) if push_high > level else None,
                    tr_ratio_3v3=round(float(np.median(atr_tr[i - 3:i]) / np.median(atr_tr[i - 6:i - 3])), 3)
                    if i >= 6 and np.median(atr_tr[i - 6:i - 3]) > 0 else None)
    card.update(structure(f, sig, i))
    return card


def true_range(f):
    prev = f["close"].shift()
    return pd.concat([f["high"] - f["low"], (f["high"] - prev).abs(), (f["low"] - prev).abs()], axis=1).max(axis=1).to_numpy()


def limit_pct(symbol):
    return 0.2 if symbol[3:5] in ("30", "68") else 0.1


def open_check(raw, f, e, symbol):
    """What the next session's open did to a plan issued at the close of event e (execution fact, not a forecast)."""
    j = e["index"] + 1
    if j >= len(f):
        return None
    lower = e["boundary"]
    upper = float(f["close"].iloc[e["index"]]) + ST.PARAMS["entry_upper_buffer_atr"] * e["n0"]
    r = raw.iloc[j]
    if float(r.get("tradestatus", 1)) != 1 or float(r.get("volume", 0)) <= 0:
        state = "NOT_TRADED"
    elif float(r["open"]) >= round(float(r["preclose"]) * (1 + limit_pct(symbol)), 2) - 0.001:
        state = "LIMIT_FILL_UNKNOWN"
    else:
        op = float(f["open"].iloc[j])
        state = "IN_BAND" if lower <= op <= upper else "ABOVE_BAND" if op > upper else "BELOW_BAND"
    return {"plan_day": e["date"], "check_day": f["date"].iloc[j], "state": state, "band": [round(lower, 3), round(upper, 3)],
            "open": round(float(f["open"].iloc[j]), 3)}


def rs_at(f, bench, i, n):
    if bench is None or i < n:
        return None
    end, start = f["date"].iloc[i], f["date"].iloc[i - n]
    if end not in bench.index or start not in bench.index:
        return None
    return round(float(np.log(f["close"].iloc[i] / f["close"].iloc[i - n]) - np.log(bench[end] / bench[start])), 4)


def eligibility_at(raw, f, i, bench, symbol, name=""):
    """One as-of function for daily and historical use (audit Q-01). Layers, each with reason codes:
    research_eligible  enough history (the CORE-V1 61-bar warm-up), traded, not ST, benchmark present on both dates
    plan_eligible      research eligible + CORE-V1 strength rule (RS5>0 or RS20>0)
    executable_domain  ordinary main board, not ST (the only market the paper accounts may trade)"""
    reasons = []
    if i + 1 < ST.PARAMS["min_history"]:
        reasons.append("WARMUP")
    r = raw.iloc[i]
    if float(r.get("tradestatus", 1)) != 1 or float(r.get("volume", 0)) <= 0:
        reasons.append("NOT_TRADED")
    if float(r.get("isST", 0)) == 1:
        reasons.append("ST")
    rs5, rs20 = rs_at(f, bench, i, 5), rs_at(f, bench, i, 20)
    if rs5 is None or rs20 is None:
        reasons.append("BENCHMARK_MISSING")
    research = not reasons
    strong = any(x is not None and x > 0 for x in (rs5, rs20))
    if research and not strong:
        reasons.append("STRENGTH_RULE_FAILED")
    return {"research_eligible": research, "plan_eligible": research and strong,
            "executable_domain": domain_of(symbol, name) == "EXECUTABLE", "rs5": rs5, "rs20": rs20, "reasons": reasons}


def raw_price(f, t, adjusted_price):
    """An adjusted price level expressed in bar t's traded (raw) price basis."""
    return float(adjusted_price) / float(f["adj_factor"].iloc[t]) if "adj_factor" in f else float(adjusted_price)


def sellable_open(f, t, symbol):
    """Can a market sell fill at bar t's open? Suspended, zero volume or a limit-down open blocks it."""
    r = f.iloc[t]
    if float(r.get("tradestatus", 1)) != 1 or float(r.get("volume", 0)) <= 0:
        return False, "NOT_TRADED"
    pre = float(r["preclose"])  # raw column; adjusted() rescales only OHLC
    if raw_price(f, t, r["open"]) <= round(pre * (1 - limit_pct(symbol)), 2) + 0.001:
        return False, "LIMIT_DOWN_OPEN"
    return True, ""


def replay_exit_executable(f, entry_index, hard, symbol):
    """CORE exit (hard risk, 20-day low channel, next-open execution) with the sale happening only on the first
    open that can actually trade (audit Q-02). The exit intent stays once raised; blocked opens are counted."""
    protection, n = hard, ST.PARAMS["trend_exit_lookback"]
    for j in range(entry_index, len(f)):
        if f["low"].iloc[j] <= hard:
            reason = "HARD_RISK"
        else:
            broken, protection = ST.protection_step(protection, float(f["close"].iloc[j]),
                                                    f["low"].iloc[max(0, j - n + 1):j + 1].tolist() if j >= n - 1 else [])
            reason = "TREND_EXIT" if broken else None
        if reason:
            blocked = []
            for m in range(j + 1, len(f)):
                ok, why = sellable_open(f, m, symbol)
                if ok:
                    return {"signal_index": j, "reason": reason, "protection": protection, "exit_index": m,
                            "exit_price": float(f["open"].iloc[m]), "blocked_opens": blocked}
                blocked.append({"date": f["date"].iloc[m], "why": why})
            return {"signal_index": j, "reason": reason, "protection": protection, "exit_index": None,
                    "exit_price": None, "blocked_opens": blocked, "state": "EXIT_PENDING"}
    return {"signal_index": None, "reason": "OPEN", "protection": protection, "exit_index": None, "exit_price": None, "blocked_opens": []}


def frozen_trade(f, e, symbol, capital=100000.0, risk=0.01, symbol_cap=0.35, fee=None):
    """One research trade under the frozen contract, entered at the open after event e (caller checks the band).
    Quantity, lots and fees use the raw prices traded on the entry day; the path uses price ratios, so appending
    later data or a later corporate action cannot rewrite a finished trade (audit Q-04). MFE/MAE run from the
    entry to the bar before the exit plus the exit open itself (audit Q-03)."""
    fee = fee or (lambda notional, sell: max(5.0, notional * 3e-4) + (notional * 5e-4 if sell else 0.0))
    j = e["index"] + 1
    entry_adj = float(f["open"].iloc[j])
    entry = raw_price(f, j, entry_adj)
    hard_raw = raw_price(f, j, e["invalidation"])
    per_share = entry - hard_raw
    if per_share <= 0:
        return {"trade": "ILLEGAL_ENTRY_BELOW_INVALIDATION"}
    qty = math.floor(min(capital * risk / per_share, capital * symbol_cap / entry) / 100) * 100
    if qty < 100:
        return {"trade": "RISK_TOO_WIDE_FOR_ONE_LOT", "entry": round(entry, 3)}
    out = replay_exit_executable(f, j, e["invalidation"], symbol)
    value = lambda adj: qty * entry * float(adj) / entry_adj  # position value in entry-day money, corporate-action neutral
    if out["exit_index"] is None:
        k, exit_adj, status = len(f) - 1, float(f["close"].iloc[-1]), "EXIT_PENDING" if out.get("state") == "EXIT_PENDING" else "OPEN"
        path = f.iloc[j:]
        hi, lo = float(path["high"].max()), float(path["low"].min())
    else:
        k, exit_adj, status = out["exit_index"], out["exit_price"], "CLOSED"
        path = f.iloc[j:k]
        hi, lo = max(float(path["high"].max()), exit_adj), min(float(path["low"].min()), exit_adj)
    buy_fee = fee(qty * entry, False)
    sell_fee = fee(value(exit_adj), True) if status == "CLOSED" else 0.0
    pnl = value(exit_adj) - qty * entry - buy_fee - sell_fee
    at_risk = qty * per_share
    actions = bool((f["adj_factor"].iloc[j:k + 1] != f["adj_factor"].iloc[j]).any()) if "adj_factor" in f else False
    return {"trade": status, "entry_date": f["date"].iloc[j], "entry": round(entry, 3), "qty": qty,
            "per_share_risk": round(per_share, 4), "gross_ratio": round(exit_adj / entry_adj, 6),
            "exit_reason": out["reason"] if status == "CLOSED" else status,
            "exit_signal_date": f["date"].iloc[out["signal_index"]] if out["signal_index"] is not None else None,
            "exit_date": f["date"].iloc[k], "exit_price": round(raw_price(f, k, exit_adj), 3), "hold_bars": k - j,
            "exit_blocked_opens": len(out["blocked_opens"]), "corporate_action_during_hold": actions,
            "net_pnl": round(pnl, 1), "net_r": round(pnl / at_risk, 3),
            "mfe_r": round((value(hi) - qty * entry) / at_risk, 3), "mae_r": round((value(lo) - qty * entry) / at_risk, 3)}


def structure(f, sig, i):
    """Impulse and base as known at bar i (audit Q-07/Q-08), recorded beside the original fields, never a gate.
    impulse_start: lowest low in the 40 bars before the anchor breakout (or before the trigger for a breakout);
    impulse_high: highest high from there up to the bar before the three-bar window; base: bars after that high."""
    high, low, close = (f[c].to_numpy(float) for c in ("high", "low", "close"))
    b = int(sig["anchor"][i])
    ref = b if b >= 0 else i
    s0 = max(0, ref - 40)
    start = s0 + int(np.argmin(low[s0:ref + 1]))
    end_limit = i - 3 if b >= 0 else i
    if end_limit <= start:
        return {"impulse_start_date": f["date"].iloc[start], "base_len": 0, "base_state": "NO_IMPULSE"}
    peak = start + int(np.argmax(high[start:end_limit]))
    base_low = float(low[peak + 1:i].min()) if peak + 1 < i else None
    base_len = i - 1 - peak
    span = high[peak] - low[start]
    out = {"impulse_start_date": f["date"].iloc[start], "impulse_high_date": f["date"].iloc[peak],
           "impulse_gain_n": round(span / sig["n0"][i], 3) if sig["n0"][i] > 0 else None,
           "base_len": base_len, "base_depth_n": round((high[peak] - base_low) / sig["n0"][i], 3) if base_low is not None and sig["n0"][i] > 0 else None,
           "retention_v2": round((base_low - low[start]) / span, 3) if base_low is not None and span > 0 else None,
           "base_state": "REAL_BASE" if base_len >= 3 else "NO_DISTINCT_BASE"}
    if b >= 0:
        seq = close[b + 1:i - 3]
        out["anchor_path_state"] = "BROKEN_THEN_RECOVERED" if len(seq) and (seq < sig["inv"][b]).any() else "INTACT"
    return out


def analyse(raw, bench, symbol):
    """Full-history replay for one security: adjusted frame, per-bar signals, events, live setup."""
    if len(raw) < ST.PARAMS["min_history"]:
        return None
    f = ST.adjusted(raw)
    atr = ST.wilder_atr(f, ST.PARAMS["atr_lookback"]).to_numpy()
    sig = bar_signals(f, atr)
    events, live = setup_events(f, sig, symbol)
    return {"f": f, "sig": sig, "events": events, "live": live, "tr": true_range(f)}


def domain_of(symbol, name):
    return "EXECUTABLE" if symbol[:5] in ("sh_60", "sz_00") and not any(x in str(name).upper() for x in ("ST", "退")) else "RESEARCH"


def review_items(row):
    """Every item PASS / FAIL / UNKNOWN with its evidence and the next condition that would change it."""
    new = row["kind"] in ("NEW", "CHILD", "RETRY")
    ex = row["execution_state"]
    return [
        {"item": "事件身份", "result": "PASS" if new else "FAIL",
         "evidence": f"{KINDS[row['kind']]}；结构 {row['setup_id']} 自 {row['setup_first_day']} 起第 {row['event_age']} 根",
         "next": "等待新整理后的子结构或回落后重试" if not new else "下一交易日开盘核对价格区间"},
        {"item": "相对强度（CORE-V1 规则：RS5>0 或 RS20>0）", "result": "PASS" if row["strength_eligible"] else "FAIL" if row["rs5"] is not None and row["rs20"] is not None else "UNKNOWN",
         "evidence": f"RS5={row['rs5']}，RS20={row['rs20']}", "next": "—"},
        {"item": "计划合法（失效位低于买入下界）", "result": "PASS" if row["plan_legal"] else "FAIL",
         "evidence": f"区间 {row['band'][0]}–{row['band'][1]}，失效 {row['invalidation']}", "next": "—"},
        {"item": "执行条件", "result": {"CONDITIONAL_NEXT_SESSION": "UNKNOWN", "LIMIT_FILL_UNKNOWN": "UNKNOWN", "ACCOUNT_BLOCKED": "FAIL",
                                        "ADD_REVIEW": "UNKNOWN", "NOT_NEW_ENTRY": "FAIL"}[ex],
         "evidence": {"CONDITIONAL_NEXT_SESSION": "夜间条件预案；次日开盘价须落在区间内",
                      "LIMIT_FILL_UNKNOWN": "收盘涨停；次日是否能在区间内成交未知，不预测一字也不假设成交",
                      "ACCOUNT_BLOCKED": "创业板/科创板/ST：当前模拟账户不可执行，仅研究",
                      "ADD_REVIEW": "已持有：转入持仓/加仓复核，不占新股票席位",
                      "NOT_NEW_ENTRY": "旧结构延伸，不产生新首仓"}[ex],
         "next": "次日开盘按事实核对：IN_BAND / ABOVE_BAND / BELOW_BAND / LIMIT_FILL_UNKNOWN"},
        {"item": "账户风险政策", "result": "UNKNOWN", "evidence": POLICY_PENDING, "next": "用户批准新核心账户参数后重评"},
    ]


def frozen_plans(root, bench, day):
    """The previous session's frozen conditional plan list (audit Q-05); None when it was never written."""
    sessions = sorted(d for d in (bench.get("sh_60") if bench.get("sh_60") is not None else pd.Series(dtype=float)).index if d < day)
    if not sessions:
        return None, None
    path = Path(root) / "results/mainwave_engine" / sessions[-1].replace("-", "") / "strong_trend_semantics.json"
    if not path.exists():
        return sessions[-1], None
    data = json.loads(path.read_text(encoding="utf-8"))
    return sessions[-1], {p["symbol"]: p for p in data.get("boards", {}).get("conditional_allocation", [])}


def plan_check(raw, plan, day, symbol):
    """Today's open against a plan frozen last night. A band fit is a price fact only, not a filled quantity."""
    base = {"symbol": symbol, "name": plan.get("name"), "setup_id": plan.get("setup_id"), "kind": plan.get("kind"),
            "night_state": plan.get("execution_state"), "band": plan.get("band")}
    if raw.empty or raw["date"].iloc[-1] != day or len(raw) < 2:
        return {**base, "state": "NO_BAR"}
    r, prev = raw.iloc[-1], raw.iloc[-2]
    if float(r.get("tradestatus", 1)) != 1 or float(r.get("volume", 0)) <= 0:
        return {**base, "state": "NOT_TRADED"}
    ratio = float(r["preclose"]) / float(prev["close"]) if float(prev["close"]) > 0 else 1.0
    ratio = 1.0 if abs(ratio - 1) <= 0.001 else ratio  # an ex-rights open scales last night's raw band
    lower, upper = plan["band"][0] * ratio, plan["band"][1] * ratio
    op = float(r["open"])
    if op >= round(float(r["preclose"]) * (1 + limit_pct(symbol)), 2) - 0.001:
        state = "LIMIT_FILL_UNKNOWN"
    else:
        state = "IN_BAND" if lower <= op <= upper else "ABOVE_BAND" if op > upper else "BELOW_BAND"
    return {**base, "state": state, "open": op, "band_today": [round(lower, 3), round(upper, 3)]}


def run(day, root=ST.ROOT, data_root=None, v1=None):
    data_root = Path(data_root or Path(root) / "data")
    bench, label, held = ST.benchmarks(data_root), ST.names(root), ST.held_symbols(root)
    tracking = json.loads((Path(root) / "fupan_tracking.json").read_text(encoding="utf-8")) if (Path(root) / "fupan_tracking.json").exists() else {}
    pinned = {p["symbol"]: p.get("name") for p in tracking.get("pinned", [])}
    rows, checks, live_count, watch = [], [], 0, []
    plan_day, frozen = frozen_plans(root, bench, day)
    plan_rows = []
    for path in sorted(data_root.glob("s[hz]_*_daily.csv")):
        symbol = path.name[:9]
        if symbol[:5] not in ST.BENCH:
            continue
        try:
            raw = load(path, day)
        except (OSError, ValueError, KeyError):
            continue
        if frozen and symbol in frozen:
            plan_rows.append(plan_check(raw, frozen[symbol], day, symbol))
        if raw.empty or raw["date"].iloc[-1] != day:
            if symbol in pinned or symbol in held:
                watch.append({"symbol": symbol, "name": label.get(symbol, pinned.get(symbol)), "state": "NO_BAR_TODAY",
                              "pinned": symbol in pinned, "held_in": held.get(symbol, [])})
            continue
        last = raw.iloc[-1]
        if float(last.get("isST", 0)) == 1 or float(last.get("tradestatus", 1)) != 1:
            continue
        a = analyse(raw, bench.get(symbol[:5]), symbol)
        if a is None:
            continue
        f, name, i = a["f"], label.get(symbol, ""), len(a["f"]) - 1
        live_count += a["live"] is not None
        prior = [e for e in a["events"] if e["index"] == i - 1 and e["kind"] in ("NEW", "CHILD", "RETRY")]
        for e in prior:
            c = open_check(raw, f, e, symbol)
            if c:
                checks.append({"symbol": symbol, "name": name, "kind": e["kind"], "setup_id": e["setup_id"],
                               "domain": domain_of(symbol, name), **c})
        today = [e for e in a["events"] if e["index"] == i]
        if not today:
            if symbol in pinned or symbol in held:
                lv = a["live"]
                watch.append({"symbol": symbol, "name": name, "pinned": symbol in pinned, "held_in": held.get(symbol, []),
                              "state": "LIVE_SETUP_NO_EVENT" if lv else "NO_LIVE_SETUP",
                              "setup_id": lv and lv["setup_id"], "setup_first_day": lv and lv["first_day"]})
            continue
        e = today[0]
        close, pre = float(last["close"]), float(last["preclose"])
        rs5, rs20 = rs_at(f, bench.get(symbol[:5]), i, 5), rs_at(f, bench.get(symbol[:5]), i, 20)
        upper = float(f["close"].iloc[i]) + ST.PARAMS["entry_upper_buffer_atr"] * e["n0"]
        domain = domain_of(symbol, name)
        at_limit = close >= round(pre * (1 + limit_pct(symbol)), 2) - 0.001
        new = e["kind"] in ("NEW", "CHILD", "RETRY")
        execution = ("NOT_NEW_ENTRY" if not new else "ADD_REVIEW" if held.get(symbol) else "ACCOUNT_BLOCKED" if domain == "RESEARCH"
                     else "LIMIT_FILL_UNKNOWN" if at_limit else "CONDITIONAL_NEXT_SESSION")
        row = {"symbol": symbol, "name": name, "kind": e["kind"], "kind_word": KINDS[e["kind"]], "setup_id": e["setup_id"],
               "parent_id": e["parent_id"], "setup_first_day": e["setup_first_day"], "event_age": e["event_age"],
               "entry": e["entry"], "labels": e["labels"], "domain": domain, "pinned": symbol in pinned, "held_in": held.get(symbol, []),
               "close": round(close, 3), "pct_chg": round((close / pre - 1) * 100, 2), "close_at_limit": at_limit,
               "band": [round(e["boundary"], 3), round(upper, 3)], "invalidation": round(e["invalidation"], 3),
               "rs5": rs5, "rs20": rs20, "strength_eligible": any(x is not None and x > 0 for x in (rs5, rs20)),
               "plan_legal": 0 < e["invalidation"] < e["boundary"], "execution_state": execution, "plan_complete": False,
               "plan_complete_reason": POLICY_PENDING, "quality": quality(f, a["sig"], a["tr"], e)}
        row["review"] = review_items(row)
        rows.append(row)
    key = lambda r: (-(r["rs20"] if r["rs20"] is not None else -9), -(r["rs5"] if r["rs5"] is not None else -9), r["symbol"])
    new_rows = [r for r in rows if r["kind"] != "PROGRESS" and r["strength_eligible"]]
    conditional = sorted([r for r in new_rows if r["execution_state"] in ("CONDITIONAL_NEXT_SESSION", "LIMIT_FILL_UNKNOWN") and r["plan_legal"]],
                         key=lambda r: (r["execution_state"] != "CONDITIONAL_NEXT_SESSION",) + key(r))
    v1_symbols = {r["symbol"] for r in (v1 or {}).get("boards", {}).get("attention", [])}
    sem_symbols = {r["symbol"] for r in rows if r["strength_eligible"]}
    count = lambda xs, k: {v: sum(1 for x in xs if x[k] == v) for v in sorted({x[k] for x in xs})}
    counts = {
        "raw_signal_rows_v1": (v1 or {}).get("summary", {}).get("signals"),
        "unique_securities_today": len({r["symbol"] for r in rows}),
        "live_setups_market_wide": live_count,
        "events_by_kind": count(rows, "kind"),
        "new_triggers": sum(r["kind"] != "PROGRESS" for r in rows),
        "new_triggers_strength_eligible": len(new_rows),
        "progress_updates": sum(r["kind"] == "PROGRESS" for r in rows),
        "by_domain_new": count(new_rows, "domain"),
        "execution_state_new": count(new_rows, "execution_state"),
        "conditional_plans": len(conditional),
        "frozen_plan_checks": ({"plan_day": plan_day, "plans": len(frozen), **count(plan_rows, "state")} if frozen is not None
                               else {"plan_day": plan_day, "status": "NO_FROZEN_PLAN_LIST"}),
        "event_open_checks_research": count(checks, "state"),
        "real_paper_fills": 0,
        "v1_vs_semantics_symbol_diff": {"only_v1": sorted(v1_symbols - sem_symbols)[:20], "only_semantics": sorted(sem_symbols - v1_symbols)[:20],
                                        "note": "CORE-V1 用最近400根计算ATR，本版本用全部历史，个别临界股票判定可能不同"},
    }
    return {"version": VERSION, "detector": DETECTOR, "as_of": day, "orders_enabled": False,
            "notice": "CORE-SEMANTICS：只修正事件身份、计数、执行域与复核证据，与 CORE-V1 并排输出，不覆盖原结果；"
                      "结构质量卡仅记录，不作门槛；计划均为条件预案（" + POLICY_PENDING + "）。",
            "counts": counts,
            "boards": {"attention_executable": sorted([r for r in rows if r["domain"] == "EXECUTABLE"], key=key),
                       "attention_research": sorted([r for r in rows if r["domain"] == "RESEARCH"], key=key),
                       "review": sorted(new_rows, key=key),
                       "conditional_allocation": conditional,
                       "watch_required": watch},
            "frozen_plan_checks": plan_rows,
            "frozen_plan_note": "只核对上一交易日冻结的条件计划名单；价格落在区间内只是价格条件，不等于能按所需数量成交",
            "event_open_checks_research": checks}


def render(result, top=15):
    c = result["counts"]
    lines = [f"# 强势趋势 CORE-SEMANTICS {result['as_of']}", "", result["notice"], "",
             "| 口径 | 数量 |", "|---|---:|",
             f"| CORE-V1 原始信号行 | {c['raw_signal_rows_v1']} |",
             f"| 今日有事件的证券 | {c['unique_securities_today']} |",
             f"| 全市场存续结构 | {c['live_setups_market_wide']} |",
             f"| 新触发（新结构/子结构/重试） | {c['new_triggers']}（强度合格 {c['new_triggers_strength_eligible']}） |",
             f"| 旧结构延伸（不产生新首仓） | {c['progress_updates']} |",
             f"| 条件计划（主板、未持有、合法） | {c['conditional_plans']} |",
             f"| 真实模拟成交 | {c['real_paper_fills']} |", "",
             f"事件构成：{c['events_by_kind']}；新触发执行状态：{c['execution_state_new']}；分域：{c['by_domain_new']}。",
             f"上一交易日冻结条件计划今日开盘核对：{c['frozen_plan_checks']}（价格条件，非成交数量）。",
             f"事件研究（上一交易日全部新触发，含非计划）：{c['event_open_checks_research'] or '无'}。", "",
             "## 条件计划（先可按区间核对，后收盘涨停待核；同组内 RS20、RS5 降序）", "",
             "|#|股票|事件|入口|结构|收盘|涨幅|区间|失效|风险%|迟入N|结构宽N|执行状态|", "|---|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---|"]
    for n, r in enumerate(result["boards"]["conditional_allocation"][:top], 1):
        q = r["quality"]
        base = {"REAL_BASE": f"整理{q.get('base_len')}根", "NO_DISTINCT_BASE": "无独立整理", "NO_IMPULSE": "无推进段"}.get(q.get("base_state"), "—")
        base += "·锚点曾破" if q.get("anchor_path_state") == "BROKEN_THEN_RECOVERED" else ""
        lines.append(f"|{n}|{r['name']} {r['symbol']}|{r['kind_word']}|{r['entry']}|{base}|{r['close']}|{r['pct_chg']}%|{r['band'][0]}–{r['band'][1]}|"
                     f"{r['invalidation']}|{q['risk_pct_at_upper']}|{q['late_distance_n']}|{q['structure_width_n']}|{r['execution_state']}|")
    if result["boards"]["watch_required"]:
        lines += ["", "## 必保留（指定股/持仓）", ""] + [f"- {w['name']} {w['symbol']}：{w['state']}" for w in result["boards"]["watch_required"]]
    return "\n".join(lines)
