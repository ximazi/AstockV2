"""Test the pre-registered robust feature set (preregistration_robust_v1.json) on the unseen 2021-06..2023-12 window,
and report the in-sample windows beside it (they were used to pick the set, so they prove nothing on their own).

Event level: each feature's direction in the unseen window; results by score; a score filter at the discovery 75th
percentile, with big-winner retention. Account level: two stock seats, R0 vs ROBUST_RANK vs 20 same-day random
draws in each window. Every plan-eligible candidate is scored whether or not it later filled (no fill leakage).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strong_trend_portfolio as PF  # noqa: E402
import screening_feature_eval as FE  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DAY = "2026-09-23"
PRE = json.loads((ROOT / "results/research/feature_study/20260923/preregistration_robust_v1.json").read_text(encoding="utf-8"))
WINDOWS = {"P_unseen_2021_2023": ("_2021", "2021-06-01", "2023-12-31"),
           "D_discovery_2024_2025H1": ("", "2024-01-01", "2025-06-30"),
           "H_holdout_2025H2_2026": ("", "2025-07-01", DAY)}


def score(d):
    cuts, fav = PRE["cuts_from_discovery"], PRE["robust_features"]
    for c, (a, b) in cuts.items():
        v = pd.to_numeric(d[c], errors="coerce")
        d[c + "_b"] = np.where(v.isna(), "NA", np.where(v <= a, "LOW", np.where(v <= b, "MID", "HIGH")))
    for c in fav:
        if c not in cuts:
            d[c + "_b"] = np.where(d[c].astype(str).str.lower().isin(["true", "1", "1.0"]), "TRUE", "FALSE")
    d["score"] = sum((d[c + "_b"] == b).astype(int) for c, b in fav.items())
    return d


def feature_directions(d):
    out = {}
    for c, b in PRE["robust_features"].items():
        good, rest = d.loc[d[c + "_b"] == b, "net_r"], d.loc[(d[c + "_b"] != b) & (d[c + "_b"] != "NA"), "net_r"]
        g, r = FE.stats(good), FE.stats(rest)
        out[c] = {"favourable": b, "fav": g, "rest": r,
                  "gap_mean": round(g.get("mean", np.nan) - r.get("mean", np.nan), 3) if g.get("n") and r.get("n") else None,
                  "gap_trim": round((g.get("trim") or np.nan) - (r.get("trim") or np.nan), 3) if g.get("n") and r.get("n") else None}
        out[c]["holds"] = bool(out[c]["gap_mean"] is not None and out[c]["gap_mean"] > 0 and (out[c]["gap_trim"] or -1) > 0)
    return out


def main(seeds=20):
    feats = {tag: score(pd.read_csv(ROOT / f"results/research/feature_study/{DAY.replace('-', '')}{tag}/event_features.csv", encoding="utf-8-sig"))
             for tag in {"_2021", ""}}
    threshold = float(feats[""].loc[feats[""].date.between("2024-01-01", "2025-06-30"), "score"].quantile(0.75))
    index = pd.read_csv(ROOT / "data/000001_SH_index.csv")
    days = sorted(pd.to_datetime(index.trade_date.astype(str)).dt.strftime("%Y-%m-%d"))
    prices = PF.Prices(DAY)
    robust_key = lambda r: (-r.score, r.q_risk_pct_at_upper if pd.notna(r.q_risk_pct_at_upper) else 99, r.symbol)
    report = {"preregistration": PRE["registered_at"], "filter_threshold_score": threshold, "windows": {}}
    for name, (tag, start, end) in WINDOWS.items():
        d = feats[tag]
        d = d[d.date.between(start, end)]
        filled = d[d.net_r.notna()]
        big = filled[filled.net_r >= 10]
        w = {"events_scored": len(d), "filled": len(filled),
             "feature_directions": feature_directions(filled) if name.startswith("P") else "in-sample (used to select)",
             "by_score": {int(s): FE.stats(g.net_r) for s, g in filled.groupby("score")},
             "filter": {"kept": FE.stats(filled[filled.score >= threshold].net_r), "dropped": FE.stats(filled[filled.score < threshold].net_r),
                        "kept_share": round(float((filled.score >= threshold).mean()), 3),
                        "big_winners_R10_kept": f"{int((big.score >= threshold).sum())}/{len(big)}"}}
        pool = pd.read_csv(ROOT / f"results/research/case_pool/{DAY.replace('-', '')}{tag}/event_domain.csv", encoding="utf-8-sig")
        pool = pool[(pool.domain == "EXECUTABLE") & pool.plan_eligible.astype(bool) & pool.date.between(start, end)]
        pool = pool.merge(d[["symbol", "date", "score", "q_risk_pct_at_upper"]].rename(columns={"q_risk_pct_at_upper": "risk_f"}),
                          on=["symbol", "date"], how="left")
        missing = int(pool.score.isna().sum())
        pool["score"] = pool.score.fillna(-1)
        calendar = [x for x in days if start <= x <= end]
        acct = {}
        for label, policy in (("R0", "R0"), ("ROBUST_RANK", robust_key)):
            c, t, x = PF.replay(pool, calendar, prices, policy)
            acct[label] = PF.metrics(c, t, x)
            t.to_csv(ROOT / f"results/research/feature_study/{DAY.replace('-', '')}/validate_{name}_{label}.csv", index=False, encoding="utf-8-sig")
        rnd = pd.DataFrame([PF.metrics(*PF.replay(pool, calendar, prices, "RANDOM", seed=s)) for s in range(seeds)])
        acct["RANDOM"] = {k: {"mean": round(float(rnd[k].mean()), 4), "min": round(float(rnd[k].min()), 4), "max": round(float(rnd[k].max()), 4)}
                          for k in ("total_return", "max_drawdown", "win_rate", "mean_net_r", "trades")}
        acct["beats_random"] = {k: int((rnd.total_return < acct[k]["total_return"]).sum()) for k in ("R0", "ROBUST_RANK")}
        acct["candidates_without_features"] = missing
        w["account"] = acct
        report["windows"][name] = w
        print(name, json.dumps({"filter": w["filter"], "account": {k: (v if k != "RANDOM" else v["total_return"]) for k, v in acct.items()}},
                               ensure_ascii=False, default=str)[:1500], flush=True)
    out = ROOT / "results/research/feature_study/20260923/robust_validation.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
