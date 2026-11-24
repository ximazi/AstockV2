"""S4: add once after the holding proves itself (H3), rules fixed before running.

Sample and exits as research/exit_study_s4.py (S4 kept by R1-R3, LAD exit, -10% stop, 60 sessions).
  FULL  whole position at the next open (the live approach)
  HALF  half position, never added
  ADD   half at the next open; if a close reaches +5% over that entry within the first 10 sessions, before the -10%
        stop was touched, the other half is bought at the following open and runs its own LAD from its own entry
Return on the full capital (idle half earns 0). Per year: mean, median, share of trades losing more than 8%, and the
share that added.
python research/s4_add_on.py -> results/research/exit_s4/add_on.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exit_study_s4 as E  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def add_day(x, t):
    """Index of the verification close (the add is bought at the next open), or None."""
    j = t + 1
    e = x.o.iloc[j]
    for k in range(j, min(j + 10, len(x) - 1)):
        if x.l.iloc[k] <= e * 0.90:
            return None
        if x.c.iloc[k] >= e * 1.05:
            return k
    return None


def main():
    ev = pd.read_csv(ROOT / "results/research/s4_features/exclusion_events.csv", encoding="utf-8-sig", dtype={"year": str})
    ev = ev[~ev["排除"].astype(bool)]
    rows = []
    for sym, g in ev.groupby("symbol"):
        f, x = E.prepare(sym)
        pos = {d: i for i, d in enumerate(f.date)}
        for d in g.date:
            t = pos.get(d)
            if t is None or t + 72 >= len(x):
                continue
            first = E.run(x, t, "LAD", set())
            if not first:
                continue
            k = add_day(x, t)
            second = E.run(x, k, "LAD", set()) if k is not None else None
            if second is not None and x.o.iloc[k + 1] >= x.c.iloc[k] * 1.05:
                second = None  # E.run already skips a >= 5% gap; keep the half idle
            rows.append({"year": d[:4], "full": first["ret"], "half": 0.5 * first["ret"],
                         "add": 0.5 * first["ret"] + (0.5 * second["ret"] if second else 0.0), "added": second is not None})
    d = pd.DataFrame(rows)
    out = []
    for y, g in d.groupby("year"):
        for col, name in (("full", "FULL 一次满仓"), ("half", "HALF 半仓不加"), ("add", "ADD 半仓+验证后加仓")):
            out.append({"年份": y, "方案": name, "笔数": len(g), "平均%": round(g[col].mean() * 100, 2),
                        "中位%": round(g[col].median() * 100, 2), "亏超8%占比": round((g[col] <= -0.08).mean() * 100, 1),
                        "平均/标准差": round(g[col].mean() / g[col].std(), 3), "加仓比例%": round(g.added.mean() * 100, 1) if col == "add" else None})
    t = pd.DataFrame(out)
    (ROOT / "results/research/exit_s4").mkdir(parents=True, exist_ok=True)
    t.to_csv(ROOT / "results/research/exit_s4/add_on.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
