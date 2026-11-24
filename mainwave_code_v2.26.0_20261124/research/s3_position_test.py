"""S3: test what the case charts showed (winners = quiet low base breaking multi-month highs; losers = high base after a
spike, still under the spike high, MA60 flat or falling). Directions fixed from the charts before this test:
  H120   close / highest high of the prior 120 sessions - 1      (>= 0: a new 120-session high, no overhead supply)
  GAP    MA20 / MA60 - 1                                         (small: averages converged, a low base)
  EXT60  close / MA60 - 1                                        (small: the base sits near MA60)
  DIF    MACD DIF / close                                        (near zero: momentum starts from the zero axis)
  SLOPE  MA60 / MA60 20 sessions earlier - 1                     (rising MA60)
Sample: S3 labelled events (results/research/s3_cases/s3_labelled.csv). Outcomes: +10% before -10% (win),
the -10% stop touched within 20 sessions, and the LAD exit return. Quintiles and AUC per year.
python research/s3_position_test.py -> results/research/s3_cases/position_test.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import four_dim_v2 as V2  # noqa: E402
import exit_study_s4 as E  # noqa: E402
import s4_macd_state as M  # noqa: E402
from s4_feature_discrimination import auc  # noqa: E402

OUT = ROOT / "results/research/s3_cases"
FEATS = {"H120": "距前120日最高价", "GAP": "MA20高出MA60", "EXT60": "收盘高出MA60", "DIF": "DIF/股价", "SLOPE": "MA60 20日斜率"}


def main():
    d = pd.read_csv(OUT / "s3_labelled.csv", encoding="utf-8-sig")
    rows = []
    for sym, g in d.groupby("symbol"):
        f, x = E.prepare(sym)
        c, h = f.close.astype(float), f.high.astype(float)
        ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
        dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        feat = pd.DataFrame({"H120": c / h.shift(1).rolling(120).max() - 1, "GAP": ma20 / ma60 - 1, "EXT60": c / ma60 - 1,
                             "DIF": dif / c, "SLOPE": ma60 / ma60.shift(20) - 1})
        stop = M.stop_hit(f)
        pos = {dd: i for i, dd in enumerate(f.date)}
        for r in g.itertuples():
            t = pos[r.date]
            lad = E.run(x, t, "LAD", set())
            rows.append({"symbol": sym, "date": r.date, "year": r.date[:4], "win": r.label != "FAIL", "stop": stop[t],
                         "lad": lad["ret"] if lad else np.nan, **feat.iloc[t].to_dict()})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "position_events.csv", index=False, encoding="utf-8-sig")
    out = []
    for col, name in FEATS.items():
        rec = {"指标": f"{col} {name}"}
        for y in ("2025", "2026"):
            g = d[(d.year == y) & d[col].notna()]
            q = pd.qcut(g[col].rank(method="first"), 5, labels=False)
            rec[f"AUC达10 {y}"] = round(float(auc(g[col], g.win)), 3)
            rec[f"达10五档 {y}"] = "/".join(f"{v * 100:.0f}" for v in g.groupby(q).win.mean())
            rec[f"止损率五档 {y}"] = "/".join(f"{v * 100:.0f}" for v in g.groupby(q).stop.mean())
            rec[f"LAD五档 {y}"] = "/".join(f"{v * 100:.1f}" for v in g.groupby(q).lad.mean())
            rec[f"五档边界 {y}"] = "/".join(f"{v:.3f}" for v in g[col].quantile([0.2, 0.4, 0.6, 0.8]))
        out.append(rec)
    t = pd.DataFrame(out)
    t.to_csv(OUT / "position_test.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 320)
    pd.set_option("display.max_colwidth", 40)
    for y in ("2025", "2026"):
        print(y, "样本", int((d.year == y).sum()), "达10", round(d[d.year == y].win.mean() * 100, 1), "止损", round(d[d.year == y].stop.mean() * 100, 1),
              "LAD", round(d[d.year == y].lad.mean() * 100, 2))
        print(t[["指标"] + [c for c in t.columns if c.endswith(y)]].to_string(index=False))
    # the combined reading from the charts: new 120-session high and a low base
    d["LOWBASE_NEWHIGH"] = (d.H120 >= 0) & (d.EXT60 <= d.groupby("year").EXT60.transform("median"))
    for y in ("2025", "2026"):
        g = d[d.year == y]
        for k, s in g.groupby("LOWBASE_NEWHIGH"):
            print(y, "新120日高且低位" if k else "其他", len(s), "达10", round(s.win.mean() * 100, 1), "止损", round(s.stop.mean() * 100, 1),
                  "LAD", round(s.lad.mean() * 100, 2))


if __name__ == "__main__":
    main()
