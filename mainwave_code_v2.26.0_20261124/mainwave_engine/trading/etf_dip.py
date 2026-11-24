"""Broad-index ETF dip-buying sleeve (交易原则_20261010.md §4): the rules, a deterministic replay, and the daily
alert. Report only: it never creates a plan, an order or a quantity; sizes are fractions of a separate dip sleeve.

Indices are grouped; indices in one group fall together and count as one signal. Each group may hold at most one
unit = 1/4 of the sleeve. Signals at a close, bought at the next open, held 20 sessions (decided at the close of the
20th session after the signal, sold at the next open), no stop (宽基ETF抄底方案研究_20261009.md):
  RSI20  RSI14 (Wilder) < 20                                  1 unit
  DD     a round starts at the first close >= 15% below the 60-session high close with no such close in the 60
         sessions before; within 60 sessions of that start, the first close at -15% / -20% / -25%  1/3 unit each
  BIAS8  close / MA20 - 1 <= -8%                              1/3 unit (largest tail risk, small share)
An index holding an RSI20 or BIAS8 position takes no further RSI20 / BIAS8 signal until it is sold; DD tranches still
add. A signal type firing on several indices of one group on the same day is one signal: its size is given once and
split equally among those indices; different signal types add up, capped by the group's free room.
"""
from pathlib import Path

import numpy as np
import pandas as pd

INDICES = [  # file stem, name, group, example ETF
    ("000016_SH", "上证50", "大盘", "510050"), ("000300_SH", "沪深300", "大盘", "510300"),
    ("000510_SH", "中证A500", "大盘", ""), ("399330_SZ", "深证100", "大盘", "159901"),
    ("000905_SH", "中证500", "中小盘", "510500"), ("000852_SH", "中证1000", "中小盘", "512100"),
    ("932000_CSI", "中证2000", "中小盘", ""), ("399006_SZ", "创业板指", "成长", "159915"),
    ("399673_SZ", "创业板50", "成长", "159949"), ("000688_SH", "科创50", "成长", "588000"),
    ("899050_BJ", "北证50", "北证", ""),
]
GROUP_UNIT = 0.25
SIZE = {"RSI20": 1.0, "DD-15": 1 / 3, "DD-20": 1 / 3, "DD-25": 1 / 3, "BIAS8": 1 / 3}
HOLD = 20
RULE = ("宽基 ETF 抄底仓（交易原则_20261010.md §4）：收盘信号、次日开盘买、持有 20 个交易日、不止损。RSI14<20 用一份；"
        "距 60 日高 −15%/−20%/−25% 每轮各 1/3 份；BIAS≤−8% 用 1/3 份。同组指数算一个信号，每组最多一份（= 抄底仓的 1/4）。"
        "分组：大盘（上证50/沪深300/中证A500/深证100）、中小盘（中证500/中证1000/中证2000）、成长（创业板指/创业板50/科创50）、北证。")


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def load(data_root, day=None):
    frames = {}
    for stem, *_ in INDICES:
        try:
            f = pd.read_csv(Path(data_root) / f"{stem}_index.csv", usecols=["trade_date", "open", "high", "low", "close"])
        except (OSError, ValueError):
            continue
        f["trade_date"] = pd.to_datetime(f["trade_date"].astype(str)).dt.strftime("%Y-%m-%d")
        f = f.drop_duplicates("trade_date").sort_values("trade_date").set_index("trade_date").astype(float)
        frames[stem] = f[f.index <= day] if day else f
    return frames


def signals(f):
    """Per session: which signals fire at that close."""
    c = f.close
    ma20, dd = c.rolling(20).mean(), c / c.rolling(60).max() - 1
    out = pd.DataFrame(index=f.index)
    out["RSI20"] = rsi(c) < 20
    out["BIAS8"] = c / ma20 - 1 <= -0.08
    hit = (dd <= -0.15).to_numpy()
    tiers = {k: np.zeros(len(c), bool) for k in ("DD-15", "DD-20", "DD-25")}
    start = None
    for i in range(len(c)):
        if hit[i] and (start is None or i - start > 60) and not hit[max(0, i - 60):i].any():
            start, done = i, set()
        if start is not None and i - start <= 60:
            for k, lv in (("DD-15", -0.15), ("DD-20", -0.20), ("DD-25", -0.25)):
                if k not in done and dd.iloc[i] <= lv:
                    tiers[k][i] = True
                    done.add(k)
    for k, v in tiers.items():
        out[k] = v
    out["bias"], out["rsi14"], out["dd60"], out["ma20"], out["high60"] = c / ma20 - 1, rsi(c), dd, ma20, c.rolling(60).max()
    out["round_start"] = ""
    return out


def replay(frames):
    """All plan positions over the frames' history: one record per (index, signal), with entry and exit sessions."""
    meta = {stem: (name, group, etf) for stem, name, group, etf in INDICES}
    sig = {s: signals(f) for s, f in frames.items()}
    pos = {s: {d: i for i, d in enumerate(f.index)} for s, f in frames.items()}
    calendar = sorted(set().union(*[set(f.index) for f in frames.values()]))
    held, out = [], []
    for day in calendar:
        for p in held:  # the 20th session after the signal closes: sold at the next open
            if p["exit_decision"] is None and pos[p["stem"]].get(day, -1) == p["t"] + HOLD:
                p["exit_decision"] = day
        held = [p for p in held if p["exit_decision"] is None]
        wants = {}
        for s, x in sig.items():
            i = pos[s].get(day)
            if i is None:
                continue
            row = x.iloc[i]
            busy = any(p["stem"] == s and p["signal"] in ("RSI20", "BIAS8") for p in held)
            fired = [k for k in ("DD-15", "DD-20", "DD-25") if row[k]]
            if not busy:
                fired += [k for k in ("RSI20", "BIAS8") if row[k]][:1]  # RSI20 first: the larger, rarer signal
            if fired:
                wants.setdefault(meta[s][1], []).extend((s, k) for k in fired)
        for group, items in wants.items():
            room = GROUP_UNIT - sum(p["size"] for p in held if p["group"] == group)
            kinds = {k for _, k in items}
            ask = sum(SIZE[k] for k in kinds) * GROUP_UNIT
            scale = min(1.0, max(room, 0) / ask) if ask > 0 else 0
            for s, k in items:
                size = SIZE[k] * GROUP_UNIT * scale / sum(1 for _, k2 in items if k2 == k)
                if size <= 1e-9:
                    continue
                name, _, etf = meta[s]
                p = {"stem": s, "name": name, "group": group, "etf": etf, "signal": k, "signal_day": day, "t": pos[s][day],
                     "size": size, "exit_decision": None}
                held.append(p)
                out.append(p)
    f_idx = {s: list(f.index) for s, f in frames.items()}
    for p in out:
        idx, f = f_idx[p["stem"]], frames[p["stem"]]
        t = p["t"]
        p["entry_day"] = idx[t + 1] if t + 1 < len(idx) else None
        p["entry"] = float(f.open.iloc[t + 1]) if t + 1 < len(idx) else None
        k = t + HOLD
        p["sell_day"] = idx[k + 1] if k + 1 < len(idx) else None
        p["exit"] = float(f.open.iloc[k + 1]) if k + 1 < len(idx) else None
    return out


def alert(day, data_root, lookback=400):
    """Today's dip-sleeve state: new signals (buy at the next open), positions the plan holds, distances to triggers."""
    base = {"rule": RULE, "report_only": True}
    frames = {s: f.tail(lookback) for s, f in load(data_root, day).items()}
    if not frames:
        return {**base, "level": "UNKNOWN", "text": "宽基指数日线不可用"}
    fresh = {s: f for s, f in frames.items() if f.index[-1] == day and len(f) >= 80}
    stale = [n for s, n, *_ in INDICES if s not in fresh]
    if not fresh:
        return {**base, "level": "STALE", "text": f"宽基指数没有 {day} 的日线", "stale": stale}
    plan = replay(fresh)
    new = [p for p in plan if p["signal_day"] == day]
    holding = [p for p in plan if p["entry_day"] and p["entry_day"] <= day and (p["sell_day"] is None or p["sell_day"] > day)]
    status = []
    for s, name, group, etf in INDICES:
        if s not in fresh:
            continue
        x = signals(fresh[s]).iloc[-1]
        c = fresh[s].close
        s19 = c.iloc[-19:].sum()
        status.append({"index": name, "group": group, "etf": etf, "close": round(float(c.iloc[-1]), 2),
                       "bias_pct": round(float(x.bias) * 100, 1), "bias8_price": round(0.92 * s19 / (20 - 0.92), 2),
                       "rsi14": round(float(x.rsi14), 1), "dd60_pct": round(float(x.dd60) * 100, 1),
                       "tier_prices": [round(float(x.high60) * k, 2) for k in (0.85, 0.80, 0.75)]})
    rows = lambda ps: [{"index": p["name"], "group": p["group"], "etf": p["etf"], "signal": p["signal"], "signal_day": p["signal_day"],
                        "size_of_sleeve": round(p["size"], 4), "entry_day": p["entry_day"], "entry": p["entry"],
                        "sell_day": p["sell_day"], "sessions_left": max(0, p["t"] + HOLD - (len(fresh[p["stem"]]) - 1))} for p in ps]
    level = "TRIGGERED_TODAY" if new else "HOLDING" if holding else "NONE"
    text = ("今日收盘出现抄底信号，按规则明日开盘买入：" + "；".join(f"{p['name']} {p['signal']} {p['size']:.1%}" for p in new) if new
            else f"抄底仓按规则持有 {len(holding)} 笔" if holding else "抄底仓今日无信号、无持仓")
    return {**base, "level": level, "text": text, "new": rows(new), "holding": rows(holding), "status": status, "stale": stale,
            "exposure": round(sum(p["size"] for p in holding), 4)}
