"""S2 supplementary review (2026-09-26, S2补充案例复核与失败条件): three independent candidates, each against BASE on the
full labelled S2 sample. Rules as the reviewer stated them; nothing tuned after seeing results.

BASE  next-open entry (gap >= 5% / limit-up open / not traded skipped), exit LAD (accept_failure.lad, T+1).
E1  execution price check: the strong-trend core's own plan cap, upper = signal close + 0.5 x ATR (Wilder, n0 = ATR of
    the day before, as strong_trend.py). Next open > upper -> the plan expires (no trade).
    E1_band (reference): next open outside [level, upper] -> expires; level is the core's structure boundary, not known
    here, so only the upper side is testable; E1_band uses the signal day's low as a conservative lower bound.
E2  recovery not complete: close < EMA20 and EMA20 < EMA20 five sessions earlier -> WAIT_RECOVERY; within the next 3
    sessions a close >= EMA20 -> buy at the following open (same gap rule), otherwise the plan expires.
E3  P3 as for S1: within the first 5 sessions after entry, two consecutive closes below the signal day's low -> exit at
    the next open; existing exits keep priority, so only trades where E3 changes the outcome count as its effect.
python research/s2_rules.py -> results/research/new_scenario_rules/S2_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import accept_failure as AF  # noqa: E402
import exit_study_s4 as E  # noqa: E402
import new_scenario_rules as N  # noqa: E402
from mainwave_engine import strong_trend as ST  # noqa: E402

OUT = N.OUT


def run():
    d = pd.read_csv(ROOT / "results/research/scenario_cases/s2_labelled.csv", encoding="utf-8-sig")
    rows = []
    for sym, g in d.groupby("symbol"):
        f, x = E.prepare(sym)
        atr = ST.wilder_atr(f, ST.PARAMS["atr_lookback"]).to_numpy()
        pos = {dd: i for i, dd in enumerate(f.date)}
        for r in g.itertuples():
            t = pos.get(r.date)
            if t is None or t < 30 or t + 64 >= len(x):
                continue
            base = AF.lad(x, t)
            if base is None:
                continue
            rec = {"symbol": sym, "date": r.date, "year": r.date[:4], "BASE": base["ret"]}
            upper = x.c.iloc[t] + ST.PARAMS["entry_upper_buffer_atr"] * atr[t - 1]
            o = x.o.iloc[t + 1]
            rec["E1"], rec["E1_state"] = (0.0, "EXPIRED") if o > upper else (base["ret"], "BUY_OPEN")
            rec["E1_band"], rec["E1_band_state"] = (0.0, "EXPIRED") if (o > upper or o < x.l.iloc[t]) else (base["ret"], "BUY_OPEN")
            ema = x.ema20
            if x.c.iloc[t] < ema.iloc[t] and ema.iloc[t] < ema.iloc[t - 5]:
                k = next((k for k in range(t + 1, min(t + 4, len(x) - 2)) if x.c.iloc[k] >= ema.iloc[k]), None)
                v = AF.lad(x, k) if k is not None else None
                rec["E2"], rec["E2_state"] = (v["ret"], "RECLAIM_BUY") if v else (0.0, "EXPIRED")
            else:
                rec["E2"], rec["E2_state"] = base["ret"], "BUY_OPEN"
            v = AF.lad(x, t, N.p3_check(x, t))
            rec["E3"], rec["E3_state"] = v["ret"], v["reason"]
            rows.append(rec)
    return pd.DataFrame(rows)


def main():
    d = run()
    d.to_csv(OUT / "S2_trades.csv", index=False, encoding="utf-8-sig")
    # N.summarize counts P-rules by their state prefix; map E3's exit reason the same way
    d["E3_state"] = np.where(d.E3_state.str.startswith("P3"), "P3", d.E3_state)
    t = N.summarize(d, ["E1", "E1_band", "E2", "E3"])
    t.loc[t["规则"] == "E3", "推迟或放弃"] = [int((g.E3_state == "P3").sum()) for _, g in list(d.groupby("year")) + [("合计", d)]]
    t.to_csv(OUT / "S2_summary.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 260)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
