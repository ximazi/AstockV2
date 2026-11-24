"""S3 pre-structure classes (external review E1): a complete platform release vs a local rebound after damage.
Only facts known at the signal close. Thresholds fixed before looking at outcomes.

Pre-base window = the 120 sessions before the base (the base = the base_len sessions before the signal).
  P       highest high of the pre-base window + base (the prior peak), at day kp
  RUN     P / lowest low of the 60 sessions before kp - 1          (size of the advance into the peak)
  DMG     lowest low from kp to the day before the signal / P - 1   (damage after the peak)
  BELOW   signal close / P - 1                                      (distance still under the peak)
Classes (first match):
  DAMAGED   受损后修复     RUN >= 25% and DMG <= -20% and BELOW <= -12%   (a sharp advance was broken, price still well under it)
  LONG_BASE 长平台释放     RUN < 25% and base_len >= 15                   (no sharp advance before; a long quiet base)
  TREND     趋势内整理释放  DMG > -15% and BELOW > -12%                   (advance intact, base near the peak)
  OTHER     其他
Outcomes per class and year: +10% before -10% (win), -10% stop within 20 sessions, volatility label (+2.5 ATR first),
LAD exit return (T+1). Also the class of each of the 20 review cases.
python research/s3_structure.py -> results/research/s3_cases/structure.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import four_dim_v2 as V2  # noqa: E402
import exit_study_s4 as E  # noqa: E402
import s4_exclusion_regime as X  # noqa: E402
import s4_feature_discrimination as S  # noqa: E402

OUT = ROOT / "results/research/s3_cases"
NAMES = {"DAMAGED": "受损后修复", "LONG_BASE": "长平台释放", "TREND": "趋势内整理释放", "OTHER": "其他"}


def classify(f, t, base_len):
    h, l, c = (f[k].to_numpy(float) for k in ("high", "low", "close"))
    b0 = t - base_len
    lo = max(0, b0 - 120)
    kp = lo + int(np.argmax(h[lo:t]))
    peak = h[kp]
    run = peak / l[max(0, kp - 60):kp + 1].min() - 1
    dmg = (l[kp + 1:t].min() / peak - 1) if kp + 1 < t else 0.0
    below = c[t] / peak - 1
    if run >= 0.25 and dmg <= -0.20 and below <= -0.12:
        k = "DAMAGED"
    elif run < 0.25 and base_len >= 15:
        k = "LONG_BASE"
    elif dmg > -0.15 and below > -0.12:
        k = "TREND"
    else:
        k = "OTHER"
    return k, run, dmg, below


def main():
    d = pd.read_csv(OUT / "s3_labelled.csv", encoding="utf-8-sig")
    rows = []
    for sym, g in d.groupby("symbol"):
        raw, f = V2.frame(ROOT / "data" / f"{sym}_daily.csv")
        _, x = E.prepare(sym)
        ind = S.indicators(raw, f)
        pos = {dd: i for i, dd in enumerate(f.date)}
        for r in g.itertuples():
            t = pos[r.date]
            k, run, dmg, below = classify(f, t, int(r.base_len))
            o = X.outcome(f, t, float(ind.V1_atr_pct.iloc[t]) * float(f.close.iloc[t]))
            lad = E.run(x, t, "LAD", set())
            rows.append({"symbol": sym, "date": r.date, "year": r.date[:4], "cls": k, "run": run, "dmg": dmg, "below": below,
                         "win": r.label != "FAIL", "stop": o["stop"], "atr_win": o["atr_win"], "lad": lad["ret"] if lad else np.nan})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "structure_events.csv", index=False, encoding="utf-8-sig")
    t = d.groupby(["year", "cls"]).agg(笔数=("win", "size"), 达10=("win", "mean"), 止损率=("stop", "mean"), ATR胜率=("atr_win", "mean"),
                                       LAD平均=("lad", "mean"), LAD中位=("lad", "median"))
    t.iloc[:, 1:] = (t.iloc[:, 1:].astype(float) * 100).round(1)
    t = t.rename(index=NAMES, level=1)
    t.to_csv(OUT / "structure.csv", encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(t.to_string())
    cases = {"win": ["sh_600881|2026-08-11", "sz_002733|2026-05-25", "sz_002652|2026-04-24", "sh_600176|2025-12-31",
                     "sh_601126|2025-11-28", "sz_002136|2025-10-31", "sh_600802|2025-09-29", "sz_000009|2025-08-29",
                     "sh_600658|2025-07-31", "sz_000920|2025-06-30"],
             "fail": ["sh_603519|2026-08-20", "sz_003033|2026-08-17", "sz_000692|2026-05-29", "sh_603158|2026-05-25",
                      "sh_600513|2026-04-30", "sh_601218|2026-04-28", "sz_002293|2025-12-30", "sh_600207|2025-10-31",
                      "sz_002733|2025-09-30", "sz_002422|2025-04-30"]}
    key = d.symbol + "|" + d.date
    for grp, ids in cases.items():
        sub = d[key.isin(ids)]
        print(grp, " ".join(f"{s[3:]}:{NAMES[c]}(涨{r * 100:.0f}%/损{m * 100:.0f}%/距峰{b * 100:.0f}%)"
                            for s, c, r, m, b in zip(sub.symbol, sub.cls, sub.run, sub.dmg, sub.below)))


if __name__ == "__main__":
    main()
