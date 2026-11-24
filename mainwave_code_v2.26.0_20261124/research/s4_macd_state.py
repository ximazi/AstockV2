"""S4: does the MACD state at the signal separate winners from losers? (what the failure charts show)

Read from the case charts: the losers were bought while DIF was already under DEA with the green bars growing, or while
price retested its high but DIF sat well under its own earlier peak (top divergence). Winners mostly had red bars growing,
or green bars shrinking toward a golden cross.
Features at the signal close (continuous-adjusted prices):
  state   RED_UP hist>0 rising / RED_DOWN hist>0 falling / GREEN_UP hist<0 rising / GREEN_DOWN hist<0 falling
          (rising = hist today > hist 2 sessions ago)
  cross5  a death cross (DIF from above DEA to below) in the last 5 sessions
  topdiv  close within 5% of the 60-session high, and DIF < 70% of its own 60-session max
  dif_dn  DIF lower than 3 sessions ago
Labels are the ones already in s4_labelled.csv (fixed +10% before -10%, and +2.5 ATR before -2.5 ATR).
python research/s4_macd_state.py -> results/research/s4_features/macd_state*.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import four_dim_v2 as V2  # noqa: E402

OUT = ROOT / "results/research/s4_features"


def stop_hit(f):
    """-10% touched within 20 sessions of the next-open entry (before the window ends)."""
    o, lo = f.open.astype(float).to_numpy(), f.low.astype(float).to_numpy()
    out = [None] * len(f)
    for t in range(len(f) - 20):
        e = o[t + 1]
        out[t] = bool((lo[t + 1:t + 21] <= e * 0.90).any())
    return out


def features(f):
    c = f.close.astype(float)
    dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    hist = dif - dea
    up = hist > hist.shift(2)
    state = pd.Series("GREEN_DOWN", index=c.index)
    state[(hist >= 0) & up] = "RED_UP"
    state[(hist >= 0) & ~up] = "RED_DOWN"
    state[(hist < 0) & up] = "GREEN_UP"
    below = dif < dea
    cross = below & ~below.shift(1, fill_value=False)
    return pd.DataFrame({
        "state": state,
        "cross5": cross.rolling(5).max().fillna(0).astype(bool),
        "topdiv": (c >= f.high.astype(float).rolling(60).max() * 0.95) & (dif < dif.rolling(60).max() * 0.7),
        "dif_dn": dif < dif.shift(3), "stop": stop_hit(f)})


def table(d, by):
    rows = []
    for y in ("2025", "2026"):
        g = d[d.year == y]
        for k, s in g.groupby(by):
            rows.append({"年份": y, by: k, "样本": len(s), "占比%": round(len(s) / len(g) * 100, 1),
                         "≥+10%": round(s.win.mean() * 100, 1), "≥+15%": round((s.label == "WIN15").mean() * 100, 1),
                         "ATR成功率%": round((s.atr_label != "FAIL").mean() * 100, 1), "触发-10%止损%": round(s.stop.mean() * 100, 1)})
        rows.append({"年份": y, by: "全部", "样本": len(g), "占比%": 100.0, "≥+10%": round(g.win.mean() * 100, 1), "≥+15%": round((g.label == "WIN15").mean() * 100, 1),
                     "ATR成功率%": round((g.atr_label != "FAIL").mean() * 100, 1), "触发-10%止损%": round(g.stop.mean() * 100, 1)})
    return pd.DataFrame(rows)


def main():
    d = pd.read_csv(OUT / "s4_labelled.csv", encoding="utf-8-sig", dtype={"year": str})
    d["win"] = d.label.isin(["WIN10", "WIN15"])
    parts = []
    for sym, g in d.groupby("symbol"):
        raw, f = V2.frame(ROOT / "data" / f"{sym}_daily.csv")
        x = features(f)
        x["date"] = f.date.values
        parts.append(g.merge(x, on="date", how="left"))
    d = pd.concat(parts, ignore_index=True)
    d["bad"] = d.state.eq("GREEN_DOWN") | d.topdiv
    d.to_csv(OUT / "macd_state_events.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    for by in ("state", "cross5", "topdiv", "dif_dn", "bad"):
        t = table(d, by)
        t.to_csv(OUT / f"macd_state_{by}.csv", index=False, encoding="utf-8-sig")
        print(t.to_string(index=False), "\n")


if __name__ == "__main__":
    main()
