"""Pre-declared single market filter: open new positions only on days the SSE Composite closed above its MA60.
Declared after the unseen-window result showed the whole core losing in 2021-2023 regardless of stock features.
Tested in all three windows at event level and in the two-seat account (R0 and ROBUST_RANK, each with/without)."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strong_trend_portfolio as PF  # noqa: E402
import screening_feature_eval as FE  # noqa: E402
import screening_robust_validate as RV  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main(seeds=20):
    idx = pd.read_csv(ROOT / "data/000001_SH_index.csv")
    idx["date"] = pd.to_datetime(idx.trade_date.astype(str)).dt.strftime("%Y-%m-%d")
    idx = idx.sort_values("date")
    up = dict(zip(idx.date, idx.close > idx.close.rolling(60).mean()))
    days = list(idx.date)
    prices = PF.Prices(RV.DAY)
    robust_key = lambda r: (-r.score, r.q_risk_pct_at_upper if pd.notna(r.q_risk_pct_at_upper) else 99, r.symbol)
    feats = {tag: RV.score(pd.read_csv(ROOT / f"results/research/feature_study/20260923{tag}/event_features.csv", encoding="utf-8-sig"))
             for tag in {"_2021", ""}}
    out = {}
    for name, (tag, start, end) in RV.WINDOWS.items():
        d = feats[tag]
        d = d[d.date.between(start, end)].copy()
        d["regime_up"] = d.date.map(up).fillna(False)
        filled = d[d.net_r.notna()]
        w = {"share_of_days_up": round(sum(up.get(x, False) for x in days if start <= x <= end) / sum(1 for x in days if start <= x <= end), 3),
             "events_up": FE.stats(filled[filled.regime_up].net_r), "events_down": FE.stats(filled[~filled.regime_up].net_r)}
        pool = pd.read_csv(ROOT / f"results/research/case_pool/20260923{tag}/event_domain.csv", encoding="utf-8-sig")
        pool = pool[(pool.domain == "EXECUTABLE") & pool.plan_eligible.astype(bool) & pool.date.between(start, end)]
        pool = pool.merge(d[["symbol", "date", "score", "q_risk_pct_at_upper"]].rename(columns={"q_risk_pct_at_upper": "risk_f"}), on=["symbol", "date"], how="left")
        pool["score"] = pool.score.fillna(-1)
        gated = pool[pool.date.map(up).fillna(False)]
        calendar = [x for x in days if start <= x <= end]
        acct = {}
        for label, events, policy in (("R0", pool, "R0"), ("R0_GATED", gated, "R0"), ("ROBUST", pool, robust_key), ("ROBUST_GATED", gated, robust_key)):
            acct[label] = PF.metrics(*PF.replay(events, calendar, prices, policy))
        for label, events in (("RANDOM", pool), ("RANDOM_GATED", gated)):
            r = pd.DataFrame([PF.metrics(*PF.replay(events, calendar, prices, "RANDOM", seed=s)) for s in range(seeds)])
            acct[label] = {"mean": round(float(r.total_return.mean()), 4), "min": round(float(r.total_return.min()), 4),
                           "max": round(float(r.total_return.max()), 4), "mdd_mean": round(float(r.max_drawdown.mean()), 4)}
        w["account"] = {k: ({kk: v[kk] for kk in ("total_return", "max_drawdown", "trades", "win_rate", "mean_net_r")} if "RANDOM" not in k else v)
                        for k, v in acct.items()}
        out[name] = w
        print(name, json.dumps(w, ensure_ascii=False), flush=True)
    (ROOT / "results/research/feature_study/20260923/regime_test.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
