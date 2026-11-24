"""Evaluate pre-declared screening features with a time split, then test an equal-weight score out of sample.

1. Each feature is bucketed on DISCOVERY (2024-01..2025-06): booleans True/False, numerics by DISCOVERY terciles.
   Per bucket: n, mean net R, mean without the top 1%, median, win rate, P(R>=5) tail, P(R<=-0.8) fail.
2. Features are SELECTED ON DISCOVERY ONLY: best-vs-worst bucket mean gap >= 0.1R and the trimmed gap > 0.
   Holdout agreement is reported for information, not used to select.
3. Score = number of selected features in their favourable bucket (equal weights, no tuning).
   HOLDOUT tests: event-level by score, big-winner retention of a score filter, and the two-seat account
   replay on the HOLDOUT window: R0 vs SCORE (score desc, then R0 order) vs 20 same-day random draws.
Research only; the frozen contract and exits are unchanged; nothing here changes CORE-V1 or NEW100K.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strong_trend_portfolio as PF  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DAY, SPLIT = "2026-09-23", "2025-07-01"
BASE = ROOT / "results/research/feature_study" / DAY.replace("-", "")
BOOL = {"ma_bull": "图2均线多头", "above_ma20_60": "图1站上MA20/60", "surge_1m": "图2近1月放量拉升", "hold_ma10_10d": "图2十日不破MA10",
        "pullback_vol_shrink": "图2回调缩量", "img2_step1": "图2初筛(涨3-5%/量比2/换手3-10%)", "img2_step2": "图2走势四条件",
        "s1_limit30": "四信号·30日涨停", "s2_yang3": "四信号·3连阳", "s3_gap10": "四信号·未补缺口", "s4_amount2x": "四信号·倍量",
        "is_new": "新结构(NEW)", "real_base": "真实整理(REAL_BASE)", "anchor_intact": "锚点未破", "close_at_limit": "收盘涨停"}
NUM = {"four_count": "四信号个数", "theme_hot_count": "题材当日≥7%家数", "theme_breadth": "题材当日≥3%占比", "theme_ret5_median": "题材5日中位涨幅",
       "industry_breadth": "行业当日≥3%占比", "industry_ret5_median": "行业5日中位涨幅", "vr5": "量比(5日)", "base_vol_contract": "整理期缩量比",
       "chip_conc90": "筹码90%集中度(越小越集中)", "chip_profit": "获利盘比例", "pos_ma60": "离MA60", "pos_ma120": "离MA120", "dd_high120": "距120日高点",
       "rise_low60": "距60日低点涨幅", "rs20": "RS20", "rs5": "RS5", "q_risk_pct_at_upper": "每股风险%", "q_late_distance_n": "迟入距离N",
       "q_close_location": "收盘位置", "q_volume_ratio20": "量比(20日)", "q_retention_v2": "涨幅保留v2", "q_excess_n": "越界幅度N", "pct": "当日涨幅", "turnover": "换手率"}


def stats(r):
    r = pd.to_numeric(r, errors="coerce").dropna()  # unfilled candidates carry features but no trade result
    if not len(r):
        return {"n": 0}
    top = r.quantile(0.99)
    return {"n": int(len(r)), "mean": round(float(r.mean()), 3), "trim": round(float(r[r < top].mean()), 3) if (r < top).any() else None,
            "median": round(float(r.median()), 3), "win": round(float((r > 0).mean()), 3),
            "tail5": round(float((r >= 5).mean()), 4), "fail": round(float((r <= -0.8).mean()), 3)}


def prepare():
    d = pd.read_csv(BASE / "event_features.csv", encoding="utf-8-sig")
    d["is_new"] = d["kind"] == "NEW"
    d["real_base"] = d.get("q_base_state") == "REAL_BASE"
    d["anchor_intact"] = d.get("q_anchor_path_state") != "BROKEN_THEN_RECOVERED"
    for c in BOOL:
        d[c] = d[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    d["period"] = np.where(d.date < SPLIT, "D", "H")
    return d


def bucketize(d):
    """Discovery terciles for numerics; buckets are fixed before the holdout is read."""
    cuts = {}
    for c in NUM:
        x = pd.to_numeric(d.loc[d.period == "D", c], errors="coerce").dropna()
        if x.nunique() >= 3:
            cuts[c] = [float(x.quantile(1 / 3)), float(x.quantile(2 / 3))]
    for c, (a, b) in cuts.items():
        v = pd.to_numeric(d[c], errors="coerce")
        d[c + "_b"] = np.where(v.isna(), "NA", np.where(v <= a, "LOW", np.where(v <= b, "MID", "HIGH")))
    for c in BOOL:
        d[c + "_b"] = np.where(d[c], "TRUE", "FALSE")
    return cuts


def feature_table(d):
    table = []
    for c, label in {**BOOL, **NUM}.items():
        if c + "_b" not in d:
            continue
        per = {}
        for period in ("D", "H"):
            g = d[d.period == period]
            per[period] = {b: stats(g.loc[g[c + "_b"] == b, "net_r"]) for b in sorted(g[c + "_b"].unique()) if b != "NA"}
        order = ["FALSE", "TRUE"] if c in BOOL else ["LOW", "MID", "HIGH"]
        dd = per["D"]
        valid = [b for b in order if dd.get(b, {}).get("n", 0) >= 300]
        if len(valid) < 2:
            continue
        best = max(valid, key=lambda b: dd[b]["mean"])
        worst = min(valid, key=lambda b: dd[b]["mean"])
        gap = dd[best]["mean"] - dd[worst]["mean"]
        tgap = (dd[best]["trim"] or 0) - (dd[worst]["trim"] or 0)
        hh = per["H"]
        h_gap = (hh.get(best, {}).get("mean", np.nan) - hh.get(worst, {}).get("mean", np.nan))
        h_tgap = ((hh.get(best, {}).get("trim") or np.nan) - (hh.get(worst, {}).get("trim") or np.nan))
        table.append({"feature": c, "label": label, "favourable": best, "worst": worst,
                      "D_gap_mean": round(gap, 3), "D_gap_trim": round(tgap, 3),
                      "H_gap_mean": round(float(h_gap), 3), "H_gap_trim": round(float(h_tgap), 3),
                      "selected_on_D": bool(gap >= 0.1 and tgap > 0 and best != worst),
                      "holds_in_H": bool(h_gap > 0 and h_tgap > 0),
                      "D_best": dd[best], "D_worst": dd[worst], "H_best": hh.get(best, {}), "H_worst": hh.get(worst, {})})
    return pd.DataFrame(table).sort_values("D_gap_mean", ascending=False)


def main(seeds=20):
    d = prepare()
    cuts = bucketize(d)
    table = feature_table(d)
    chosen = table[table.selected_on_D]
    fav = dict(zip(chosen.feature, chosen.favourable))
    d["score"] = sum((d[c + "_b"] == b).astype(int) for c, b in fav.items())
    h = d[d.period == "H"]
    by_score = {int(s): stats(g.net_r) for s, g in h.groupby("score")}
    median_score = float(d.loc[d.period == "D", "score"].median())
    big = h[h.net_r >= 10]
    keep = h[h.score >= median_score]
    # two-seat account on the holdout window: plan-eligible executable events, all fill states (the account decides)
    pool = pd.read_csv(ROOT / f"results/research/case_pool/{DAY.replace('-', '')}/event_domain.csv", encoding="utf-8-sig")
    pool = pool[(pool.domain == "EXECUTABLE") & pool.plan_eligible.astype(bool) & (pool.date >= SPLIT)].copy()
    pool = pool.merge(d[["symbol", "date", "score"]], on=["symbol", "date"], how="left")
    pool["score"] = pool["score"].fillna(0)
    index = pd.read_csv(ROOT / "data/000001_SH_index.csv")
    calendar = sorted(x for x in pd.to_datetime(index.trade_date.astype(str)).dt.strftime("%Y-%m-%d") if SPLIT <= x <= DAY)
    prices = PF.Prices(DAY)
    score_key = lambda r: (-r.score,) + PF.r0_key(r)
    results = {}
    for name, policy in (("R0", "R0"), ("SCORE", score_key)):
        c, t, x = PF.replay(pool, calendar, prices, policy)
        results[name] = PF.metrics(c, t, x)
        t.to_csv(BASE / f"holdout_{name.lower()}_trades.csv", index=False, encoding="utf-8-sig")
    rnd = pd.DataFrame([PF.metrics(*PF.replay(pool, calendar, prices, "RANDOM", seed=s)) for s in range(seeds)])
    results["RANDOM"] = {k: {"mean": round(float(rnd[k].mean()), 4), "min": round(float(rnd[k].min()), 4), "max": round(float(rnd[k].max()), 4)}
                         for k in ("total_return", "max_drawdown", "win_rate", "mean_net_r", "trades")}
    results["SCORE_beats_random"] = int((rnd.total_return < results["SCORE"]["total_return"]).sum())
    results["R0_beats_random"] = int((rnd.total_return < results["R0"]["total_return"]).sum())
    summary = {"split": SPLIT, "events": {"D": int((d.period == "D").sum()), "H": int((d.period == "H").sum())},
               "discovery_cuts": cuts, "selected_features": fav,
               "holdout_by_score": by_score, "holdout_all": stats(h.net_r),
               "score_filter_ge_median": {"median_score_on_D": median_score, "kept_share": round(len(keep) / max(len(h), 1), 3),
                                          "kept": stats(keep.net_r), "dropped": stats(h[h.score < median_score].net_r),
                                          "big_winners_R10_kept": f"{int((big.score >= median_score).sum())}/{len(big)}"},
               "account_holdout": results}
    (BASE / "feature_table.json").write_text(table.to_json(orient="records", force_ascii=False, indent=1), encoding="utf-8")
    (BASE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    cols = ["label", "favourable", "D_gap_mean", "D_gap_trim", "H_gap_mean", "H_gap_trim", "selected_on_D", "holds_in_H"]
    print(table[cols].to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k != "discovery_cuts"}, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
