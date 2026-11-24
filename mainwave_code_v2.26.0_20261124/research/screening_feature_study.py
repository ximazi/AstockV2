"""Which pre-trigger features help pick STRONG_TREND events? (two infographics + theme resonance + four signals)

Every feature is computed from information available at the trigger close and defined BEFORE looking at
outcomes (image 1: position / volume-price / MA structure / catalyst-sector; image 2: volume screen / trend
structure / chip structure; plus four-signal resonance in the prior month; plus theme and industry resonance).
Money-flow signals of image 1 (dark pool, options, gamma) have no A-share daily equivalent and are skipped.

Outcomes come from the calibrated event domain (frozen contract, research only). The study splits time:
DISCOVERY 2024-01..2025-06 and HOLDOUT 2025-07..; a feature only counts if its direction agrees in both, on the
mean and on the mean without the top 1%. Then an equal-weight score of the agreeing features is tested as a
ranking in the two-seat account replay on the HOLDOUT window only, against R0 and same-day random picks.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mainwave_engine import strong_trend as ST  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DAY = "2026-09-23"
SPLIT = "2025-07-01"
OUT = ROOT / "results/research/feature_study" / DAY.replace("-", "")
GENERIC = ("融资融券", "股通", "样本股", "证金", "MSCI", "同花顺", "标普", "富时", "转融券", "成份股", "指数", "一带一路", "国企改革", "专精特新")
BINS = 150


def limit_pct(symbol):
    return 19.8 if symbol[3:6] in ("300", "301", "688", "689") else 9.8


def load(path):
    f = pd.read_csv(path, usecols=lambda c: c in {"date", "open", "high", "low", "close", "preclose", "volume", "amount", "turnover", "pct_chg", "tradestatus", "isST"})
    f["date"] = f["date"].astype(str).str[:10]
    f = f[f.date <= DAY].sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return f


def chips(f, i, window=120):
    """Turnover-decay cost distribution (classic CYQ approximation) as of bar i, on the continuous price scale."""
    s = max(0, i - window + 1)
    lo, hi = f["low"].to_numpy(float)[s:i + 1], f["high"].to_numpy(float)[s:i + 1]
    tr = np.clip(f["turnover"].to_numpy(float)[s:i + 1] / 100.0, 0, 1)
    if len(lo) < 60 or not np.isfinite(tr).all() or tr.sum() <= 0:
        return None, None
    grid = np.linspace(lo.min(), hi.max(), BINS)
    survive = np.r_[np.cumprod((1 - tr)[::-1])[::-1][1:], 1.0]
    weight = tr * survive
    inside = (grid[None, :] >= lo[:, None]) & (grid[None, :] <= hi[:, None])
    per = inside / np.maximum(inside.sum(axis=1, keepdims=True), 1)
    dist = (weight[:, None] * per).sum(axis=0)
    if dist.sum() <= 0:
        return None, None
    cdf = np.cumsum(dist) / dist.sum()
    p5, p95 = grid[np.searchsorted(cdf, 0.05)], grid[min(np.searchsorted(cdf, 0.95), BINS - 1)]
    close = float(f["close"].iloc[i])
    return round(float((p95 - p5) / (p95 + p5)), 4), round(float(dist[grid <= close].sum() / dist.sum()), 4)


def stock_features(raw, symbol, idx_dates):
    f = ST.adjusted(raw)
    c, h, l, v = (f[x].astype(float) for x in ("close", "high", "low", "volume"))
    pct = raw["pct_chg"].astype(float)
    ma = {n: c.rolling(n).mean() for n in (5, 10, 20, 60, 120)}
    vr5 = v / v.shift(1).rolling(5).mean()
    up_limit = pct >= limit_pct(symbol)
    yang = (f["close"] > f["open"]).astype(int)
    run = yang.groupby((yang == 0).cumsum()).cumsum()
    prev_high = h.shift(1)
    gap = (f["open"] > prev_high).to_numpy()
    lows, ph = l.to_numpy(), prev_high.to_numpy()
    s3 = np.zeros(len(f), bool)
    for g in np.flatnonzero(gap):
        for t in range(g, min(len(f), g + 10)):
            if lows[t] <= ph[g]:
                break
            s3[t] = True
    amt = raw["amount"].astype(float)
    s4 = (amt / amt.shift(1).rolling(20, min_periods=15).mean() >= 2).astype(int).rolling(5, min_periods=1).max().astype(bool)
    down = c < c.shift(1)
    vol_down = v.where(down).rolling(10, min_periods=1).mean()
    vol_up = v.where(~down).rolling(10, min_periods=1).mean()
    out = pd.DataFrame({
        "date": f["date"],
        # image 1 position
        "pos_ma60": c / ma[60] - 1, "pos_ma120": c / ma[120] - 1,
        "dd_high120": c / h.rolling(120, min_periods=60).max() - 1,
        "rise_low60": c / l.rolling(60, min_periods=40).min() - 1,
        # image 1/2 volume-price
        "vr5": vr5, "base_vol_contract": v.shift(1).rolling(10).mean() / v.shift(11).rolling(30).mean(),
        "img2_step1": pct.between(3, 5) & (vr5 >= 2) & raw["turnover"].astype(float).between(3, 10),
        # image 1/2 MA structure
        "ma_bull": (ma[5] > ma[10]) & (ma[10] > ma[20]) & (ma[20] > ma[60]),
        "above_ma20_60": (c > ma[20]) & (c > ma[60]),
        "surge_1m": ((pct >= 5) & (vr5 >= 2)).astype(int).shift(1).rolling(20, min_periods=1).max().astype(bool),
        "hold_ma10_10d": (c >= ma[10]).astype(int).rolling(10).min().astype(bool),
        "pullback_vol_shrink": vol_down < vol_up,
        # four signals in the prior month
        "s1_limit30": up_limit.astype(int).rolling(30, min_periods=1).max().astype(bool),
        "s2_yang3": (run >= 3).astype(int).rolling(10, min_periods=1).max().astype(bool),
        "s3_gap10": s3, "s4_amount2x": s4,
        "pct": pct, "turnover": raw["turnover"].astype(float),
    })
    out["four_count"] = out[["s1_limit30", "s2_yang3", "s3_gap10", "s4_amount2x"]].sum(axis=1)
    out["img2_step2"] = out["ma_bull"] & (c > ma[5]) & out["surge_1m"] & out["hold_ma10_10d"]
    out["ret5"] = c / c.shift(5) - 1
    return f, out


def main(tag="", market_start="2023-12-01"):
    global OUT
    OUT = ROOT / "results/research/feature_study" / (DAY.replace("-", "") + tag)
    events = pd.read_csv(ROOT / f"results/research/case_pool/{DAY.replace('-', '')}{tag}/event_domain.csv", encoding="utf-8-sig")
    # Features for every plan-eligible candidate, filled or not: a ranking must be scorable before the fill is known.
    ev = events[(events.domain == "EXECUTABLE") & events.plan_eligible.astype(bool)].copy()
    wanted = ev.groupby("symbol")["date"].apply(set).to_dict()
    rows, market = [], []
    for n, path in enumerate(sorted((ROOT / "data").glob("s[hz]_*_daily.csv")), 1):
        symbol = path.name[:9]
        if symbol[:5] not in ST.BENCH:
            continue
        try:
            raw = load(path)
            if len(raw) < 130:
                continue
            f, feat = stock_features(raw, symbol, None)
        except (OSError, ValueError, KeyError):
            continue
        recent = feat[feat.date >= market_start]
        market.append(pd.DataFrame({"date": recent.date, "symbol": symbol, "pct": recent.pct, "ret5": recent.ret5}))
        for d in wanted.get(symbol, ()):
            hit = feat.index[feat.date == d]
            if not len(hit):
                continue
            i = int(hit[0])
            r = feat.iloc[i].to_dict()
            r["chip_conc90"], r["chip_profit"] = chips(f, i)
            r["symbol"] = symbol
            rows.append(r)
        if n % 1000 == 0:
            print(f"[features] {n} files", flush=True)
    feats = pd.DataFrame(rows)
    mk = pd.concat(market, ignore_index=True)
    feats = feats.merge(theme_features(mk, feats), on=["symbol", "date"], how="left")
    data = ev.merge(feats, on=["symbol", "date"], how="inner", suffixes=("", "_f"))
    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT / "event_features.csv", index=False, encoding="utf-8-sig")
    print(f"[features] events with features: {len(data)}", flush=True)
    return data


def theme_features(mk, feats):
    """Same-day theme / industry resonance, excluding the stock itself. Concept mapping is today's version (no
    point-in-time history), a known look-ahead risk; industry labels are more stable."""
    mapping = json.loads((ROOT / "concept_mapping.json").read_text(encoding="utf-8"))["mapping"]
    member = {}
    for sym, info in mapping.items():
        for cpt in info.get("concepts", []):
            if not any(g in cpt for g in GENERIC):
                member.setdefault(cpt, set()).add(sym)
    member = {k: v for k, v in member.items() if 10 <= len(v) <= 400}
    industry = {}
    for sym, info in mapping.items():
        if info.get("industry"):
            industry.setdefault(info["industry"], set()).add(sym)
    pct = mk.pivot_table(index="date", columns="symbol", values="pct")
    r5 = mk.pivot_table(index="date", columns="symbol", values="ret5")
    hot = pct >= 7
    strong = pct >= 3
    out = []
    for (sym, d) in feats[["symbol", "date"]].itertuples(index=False):
        if d not in pct.index:
            continue
        info = mapping.get(sym, {})
        best_hot, best_breadth, best_r5 = 0, 0.0, None
        for cpt in info.get("concepts", []):
            mem = [m for m in member.get(cpt, ()) if m != sym and m in pct.columns]
            if not mem:
                continue
            h = int(hot.loc[d, mem].sum())
            b = float(strong.loc[d, mem].mean())
            m5 = float(r5.loc[d, mem].median())
            best_hot, best_breadth = max(best_hot, h), max(best_breadth, b)
            best_r5 = m5 if best_r5 is None else max(best_r5, m5)
        ind = [m for m in industry.get(info.get("industry"), ()) if m != sym and m in pct.columns]
        out.append({"symbol": sym, "date": d, "theme_hot_count": best_hot, "theme_breadth": round(best_breadth, 4),
                    "theme_ret5_median": best_r5, "industry_breadth": round(float(strong.loc[d, ind].mean()), 4) if ind else None,
                    "industry_ret5_median": float(r5.loc[d, ind].median()) if ind else None})
    return pd.DataFrame(out)


if __name__ == "__main__":
    opt = {a.split("=")[0][2:]: a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--") and "=" in a}
    main(opt.get("tag", ""), opt.get("market-start", "2023-12-01"))
