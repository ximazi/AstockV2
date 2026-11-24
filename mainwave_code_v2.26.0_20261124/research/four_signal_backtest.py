"""Four-signal screener backtest with an EMA20-bias filter (research only, no orders).

Signals on a back-adjusted price series (pct_chg compounded; raw prices would turn ex-rights days into fake gaps):
  S1 limit-up close (main >= 9.8%, ChiNext/STAR >= 19.8%) within the last 30 sessions
  S2 a run of >= 3 consecutive yang candles (close > open) within the last 10 sessions
  S3 an up gap (open > previous high) within the last 10 sessions, not filled since (every later low > that high)
  S4 within the last 5 sessions a day whose amount >= 2x the mean amount of the 20 sessions before it
     (amount, not volume: share splits do not distort it)
Episode: consecutive sessions with >= 3 signals. For an EMA20-bias limit X, D-Day is the first session of the
episode with bias = close/EMA20 - 1 <= X. B-Day: the first yin day (close < open) in D+1..D+3; buy at B's close,
exit at the close 20 own sessions later. No B-Day, no trade. Everything is causal; outcomes are only read after B."""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
OUT = Path("results/research/four_signal")
START, SPLIT = "2021-01-01", "2024-01-01"
LIMITS = (0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, None)  # None = no EMA20 filter
HOLD, BUY_WINDOW = 20, 3
COST = 0.002  # round trip: commission both sides + stamp duty, approximate
MIN_AMOUNT = 2e7  # 20-session median amount; illiquid names are skipped


def limit_pct(symbol):
    code = symbol.split("_")[-1]
    return 19.8 if code.startswith(("300", "301", "688", "689")) else 9.8


def prepare(path):
    symbol = Path(path).name[:-len("_daily.csv")]
    f = pd.read_csv(path, encoding="utf-8-sig", usecols=lambda c: c in ("date", "open", "high", "low", "close", "amount", "pct_chg", "tradestatus", "isST"))
    for c in ("open", "high", "low", "close", "amount", "pct_chg", "tradestatus", "isST"):
        if c in f:
            f[c] = pd.to_numeric(f[c], errors="coerce")
    f["date"] = f.date.astype(str).str[:10]
    f = f[(f.amount > 0) & (f.tradestatus.fillna(1) == 1) & f[["open", "high", "low", "close", "pct_chg"]].notna().all(axis=1)]
    f = f.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if len(f) < 80:
        return symbol, None
    idx = (1 + f.pct_chg / 100).cumprod()
    scale = idx / f.close
    f["aopen"], f["ahigh"], f["alow"], f["aclose"] = f.open * scale, f.high * scale, f.low * scale, idx
    return symbol, f


def signals(f, symbol):
    n = len(f)
    up = (f.pct_chg >= limit_pct(symbol)).astype(int)
    s1 = up.rolling(30, min_periods=1).max().astype(bool)
    yang = (f.close > f.open).astype(int)
    run = yang.groupby((yang == 0).cumsum()).cumsum()  # length of the current yang run
    s2 = (run >= 3).astype(int).rolling(10, min_periods=1).max().astype(bool)
    prev_high = f.ahigh.shift(1)
    gap = (f.aopen > prev_high).to_numpy()
    lows = f.alow.to_numpy()
    ph = prev_high.to_numpy()
    s3 = np.zeros(n, dtype=bool)
    # for each day, any gap in the last 10 sessions whose previous high is still below every low since the gap day
    gap_days = np.flatnonzero(gap)
    for g in gap_days:
        level = ph[g]
        for t in range(g, min(n, g + 10)):
            if lows[t] <= level:  # filled on day t (the gap day itself counts: open above, but an intraday fill kills it)
                break
            s3[t] = True
    amt = f.amount
    base = amt.shift(1).rolling(20, min_periods=15).mean()
    ratio = amt / base
    s4 = (ratio >= 2).astype(int).rolling(5, min_periods=1).max().astype(bool)
    ema20 = f.aclose.ewm(span=20, adjust=False).mean()
    bias = f.aclose / ema20 - 1
    liquid = amt.shift(1).rolling(20, min_periods=15).median() >= MIN_AMOUNT
    st = f.isST.fillna(0) == 1 if "isST" in f else pd.Series(False, index=f.index)
    count = s1.astype(int) + s2.astype(int) + s3.astype(int) + s4.astype(int)
    return pd.DataFrame({"s1": s1, "s2": s2, "s3": s3, "s4": s4, "count": count, "bias": bias, "ratio": ratio,
                         "ok": liquid & ~st & (pd.Series(np.arange(n)) >= 60)})


def trades_for(path):
    symbol, f = prepare(path)
    if f is None:
        return []
    sig = signals(f, symbol)
    dates = f.date.to_numpy()
    qualify = ((sig["count"] >= 3) & sig.ok).to_numpy()
    close, open_, aclose = f.close.to_numpy(), f.open.to_numpy(), f.aclose.to_numpy()
    down_limit = (f.pct_chg <= -limit_pct(symbol)).to_numpy()
    out, n, i = [], len(f), 0
    while i < n:
        if not qualify[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and qualify[j + 1]:
            j += 1
        episode = range(i, j + 1)
        if dates[i] >= START:
            for limit in LIMITS:
                d = next((t for t in episode if limit is None or sig.bias.iloc[t] <= limit), None)
                if d is None:
                    continue
                b = next((t for t in range(d + 1, min(n, d + 1 + BUY_WINDOW)) if close[t] < open_[t] and not down_limit[t]), None)
                row = {"symbol": symbol, "limit": "none" if limit is None else limit, "d_day": dates[d], "d_bias": round(float(sig.bias.iloc[d]), 4),
                       "d_count": int(sig["count"].iloc[d]), "signals": "".join(k[-1] for k in ("s1", "s2", "s3", "s4") if sig[k].iloc[d]),
                       "d_amount_ratio_max5": round(float(sig.ratio.iloc[max(0, d - 4):d + 1].max()), 2)}
                if b is None:
                    out.append({**row, "b_day": None, "status": "NO_B_DAY"})
                    continue
                e = b + HOLD
                if e >= n:
                    out.append({**row, "b_day": dates[b], "status": "IMMATURE"})
                    continue
                path_ = aclose[b + 1:e + 1] / aclose[b] - 1
                out.append({**row, "b_day": dates[b], "exit_day": dates[e], "status": "TRADE",
                            "ret20": round(float(aclose[e] / aclose[b] - 1), 5), "mfe20": round(float(path_.max()), 5),
                            "mae20": round(float(path_.min()), 5), "b_offset": b - d})
        i = j + 1
    return out


def benchmark():
    b = pd.read_csv(DATA / "000300_SH_index.csv", usecols=["trade_date", "close"])
    b["date"] = pd.to_datetime(b.trade_date.astype(str)).dt.strftime("%Y-%m-%d")
    b = b.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return dict(zip(b.date, b.close)), b.date.tolist()


def stats(g):
    r = g.ret20 - COST
    ex = g.excess20.dropna()
    return {"trades": int(len(g)), "days": int(g.b_day.nunique()), "symbols": int(g.symbol.nunique()),
            "mean_net": round(float(r.mean()) * 100, 2), "median_net": round(float(r.median()) * 100, 2),
            "win_net": round(float((r > 0).mean()) * 100, 1), "mean_excess_net": round(float(ex.mean() - COST) * 100, 2) if len(ex) else None,
            "p10_net": round(float(r.quantile(.1)) * 100, 2), "mfe_median": round(float(g.mfe20.median()) * 100, 2),
            "mae_median": round(float(g.mae20.median()) * 100, 2),
            "day_mean_net": round(float(g.assign(r=r).groupby("b_day").r.mean().mean()) * 100, 2)} if len(g) else {"trades": 0}


def run(workers=6, limit_files=None):
    files = sorted(p for p in glob.glob(str(DATA / "*_daily.csv")) if re.search(r"(sh_(60|68)|sz_(00|30))\d{4}_daily\.csv$", p))
    if limit_files:
        files = files[:limit_files]
    rows = []
    started = datetime.now()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, r in enumerate(pool.map(trades_for, files, chunksize=20), 1):
            rows.extend(r)
            if n % 1000 == 0:
                print(f"[four_signal] {n}/{len(files)} rows={len(rows)} {(datetime.now() - started).seconds}s", flush=True)
    df = pd.DataFrame(rows)
    bench, bdates = benchmark()
    pos = {d: i for i, d in enumerate(bdates)}

    def bench_ret(b, e):
        if b not in bench or e not in bench:
            return None
        return bench[e] / bench[b] - 1
    t = df[df.status == "TRADE"].copy()
    t["bench20"] = [bench_ret(b, e) for b, e in zip(t.b_day, t.exit_day)]
    t["excess20"] = t.ret20 - t.bench20
    t["period"] = np.where(t.b_day < SPLIT, "2021-2023", "2024-2026")
    t["year"] = t.b_day.str[:4]
    out = OUT / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "all_rows.csv", index=False, encoding="utf-8-sig")
    t.to_csv(out / "trades.csv", index=False, encoding="utf-8-sig")
    summary = {"definitions": __doc__, "start": START, "split": SPLIT, "hold": HOLD, "cost": COST, "min_amount": MIN_AMOUNT,
               "status_counts": {str(k): v for k, v in df.groupby(["limit", "status"]).size().to_dict().items()}, "by_limit": {}}
    for limit, g in t.groupby("limit", sort=False):
        summary["by_limit"][str(limit)] = {"all": stats(g), **{p: stats(gg) for p, gg in g.groupby("period")},
                                           "by_year": {y: stats(gg) for y, gg in g.groupby("year")}}
    # bias buckets of the unfiltered D-Days: where along the EMA20 distance does the payoff change
    free = t[t.limit == "none"].copy()
    free["bias_bucket"] = pd.cut(free.d_bias, [-1, 0, .03, .05, .08, .10, .12, .15, .20, .25, .35, 5])
    summary["unfiltered_by_bias_bucket"] = {str(k): stats(g) for k, g in free.groupby("bias_bucket", observed=True)}
    summary["unfiltered_by_signal_set"] = {k: stats(g) for k, g in free.groupby("signals")}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (out / "report.md").write_text(report(summary), encoding="utf-8")
    print(f"[four_signal] written {out}")
    return out


def report(s):
    def table(title, items):
        lines = [f"## {title}", "", "| 分组 | 笔数 | 买入日数 | 平均净收益% | 中位% | 胜率% | 超额(扣费)% | P10% | MFE中位% | MAE中位% | 按日均值% |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for k, v in items:
            if v.get("trades"):
                lines.append(f"| {k} | {v['trades']} | {v['days']} | {v['mean_net']} | {v['median_net']} | {v['win_net']} | {v['mean_excess_net']} | {v['p10_net']} | {v['mfe_median']} | {v['mae_median']} | {v['day_mean_net']} |")
        return lines + [""]
    lines = ["# 四信号 + EMA20 偏离 回测", "", f"区间 {s['start']} 起；前后半以 {s['split']} 切分；持有 {s['hold']} 个交易日；成本 {s['cost']*100:.1f}%（往返）；流动性 20 日成交额中位 ≥ {s['min_amount']/1e4:.0f} 万。", ""]
    lines += table("按 EMA20 偏离上限（全区间）", [(k, v["all"]) for k, v in s["by_limit"].items()])
    lines += table("按上限 · 2021-2023（设计段）", [(k, v.get("2021-2023", {})) for k, v in s["by_limit"].items()])
    lines += table("按上限 · 2024-2026（检验段）", [(k, v.get("2024-2026", {})) for k, v in s["by_limit"].items()])
    lines += table("不设上限时，按 D-Day 实际偏离分桶", list(s["unfiltered_by_bias_bucket"].items()))
    lines += table("不设上限时，按信号组合（1=涨停 2=连阳 3=缺口 4=倍量）", list(s["unfiltered_by_signal_set"].items()))
    lines += ["读法：同日同题材的交易高度相关，笔数不是独立样本数，按日均值更稳；选上限先看设计段，再用检验段确认，不在检验段上反复挑。"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit-files", type=int, default=None)
    a = ap.parse_args()
    run(a.workers, a.limit_files)
