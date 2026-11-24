"""Diagnostics of the STRONG_TREND shadow signal count (no returns, no tuning): how many signals per day over
the last N sessions, how often the same stock re-signals, extension, liquidity and consolidation depth.
Uses the same definitions as mainwave_engine.strong_trend (CT20/RA3)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mainwave_engine import strong_trend as ST  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SESSIONS = 60


def signals(f):
    """Per-bar flags for the last SESSIONS bars of one adjusted frame."""
    n = len(f)
    atr = ST.wilder_atr(f).to_numpy()
    high, low, close = (f[c].to_numpy() for c in ("high", "low", "close"))
    level = pd.Series(high).shift(1).rolling(20).max().to_numpy()
    prev_atr = np.r_[np.nan, atr[:-1]]
    bo = (close > level) & np.isfinite(prev_atr) & (prev_atr > 0)
    inv = level - 2 * prev_atr
    out = []
    for i in range(max(25, n - SESSIONS), n):
        ra = None
        if np.isfinite(prev_atr[i]) and prev_atr[i] > 0:
            w = slice(i - 3, i)
            top, bottom = high[w].max(), low[w].min()
            paused = (np.diff(close[i - 4:i]) < 0).any() or close[w].max() - close[w].min() <= prev_atr[i]
            if paused and close[i] > top:
                for b in range(i - 4, max(i - 30, 20) - 1, -1):
                    if bo[b] and bottom > inv[b]:
                        ra = b
                        break
        if bo[i] or ra is not None:
            out.append({"i": i, "breakout": bool(bo[i]), "ra_anchor": ra,
                        "anchor_age": None if ra is None else i - ra,
                        "down_closes_in_window": int((np.diff(close[i - 4:i]) < 0).sum()),
                        "window_depth_atr": None if ra is None else float((high[ra:i].max() - low[i - 3:i].min()) / prev_atr[i])})
    return out


def main(day):
    rows, per_day = [], {}
    for path in sorted((ROOT / "data").glob("s[hz]_*_daily.csv")):
        symbol = path.name[:9]
        if symbol[:5] not in ST.BENCH:
            continue
        raw = pd.read_csv(path, usecols=lambda c: c in {"date", "open", "high", "low", "close", "preclose", "volume", "amount", "tradestatus", "isST"})
        raw["date"] = raw["date"].astype(str).str[:10]
        f = raw[raw.date <= day].tail(460).reset_index(drop=True)
        if len(f) < 120 or f.date.iloc[-1] != day or float(f.isST.iloc[-1]) == 1:
            continue
        f = ST.adjusted(f)
        ema20 = f.close.ewm(span=20, adjust=False).mean().to_numpy()
        for s in signals(f):
            d = f.date.iloc[s["i"]]
            per_day[d] = per_day.get(d, 0) + 1
            rows.append({"symbol": symbol, "date": d, **s, "ema20_bias": float(f.close.iloc[s["i"]] / ema20[s["i"]] - 1),
                         "ret20": float(f.close.iloc[s["i"]] / f.close.iloc[s["i"] - 20] - 1),
                         "amount_yi": float(f.amount.iloc[s["i"]]) / 1e8 if "amount" in f else None})
    sig = pd.DataFrame(rows)
    out = ROOT / "results/research/strong_trend_audit"
    out.mkdir(parents=True, exist_ok=True)
    sig.to_csv(out / f"signals_{day.replace('-', '')}.csv", index=False, encoding="utf-8-sig")
    today = sig[sig.date == day].copy()
    recent = sig[sig.date > sorted(per_day)[-11]] if len(per_day) > 10 else sig
    repeats = recent.groupby("symbol").size()
    today["signals_last10"] = today.symbol.map(repeats)
    q = lambda s: [round(float(x), 3) for x in s.quantile([.1, .25, .5, .75, .9])]
    counts = pd.Series(per_day).sort_index()
    summary = {
        "day": day, "today": len(today),
        "daily_count_last60": {"median": float(counts.median()), "min": int(counts.min()), "max": int(counts.max()),
                               "p90": float(counts.quantile(.9)), "last10": counts.tail(10).astype(int).to_dict()},
        "today_breakout_only": int((today.breakout & today.ra_anchor.isna()).sum()),
        "today_ra": int(today.ra_anchor.notna().sum()),
        "ra_down_closes_in_window": today[today.ra_anchor.notna()].down_closes_in_window.value_counts().sort_index().to_dict(),
        "ra_anchor_age_q": q(today.anchor_age.dropna()),
        "ra_window_depth_atr_q": q(today.window_depth_atr.dropna()),
        "ema20_bias_q": q(today.ema20_bias), "ret20_q": q(today.ret20), "amount_yi_q": q(today.amount_yi),
        "share_bias_over_10pct": round(float((today.ema20_bias > .10).mean()), 3),
        "share_amount_below_1yi": round(float((today.amount_yi < 1).mean()), 3),
        "signals_last10_q": q(today.signals_last10),
        "share_signaled_3plus_last10": round(float((today.signals_last10 >= 3).mean()), 3),
    }
    (out / f"summary_{day.replace('-', '')}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2026-09-22")
