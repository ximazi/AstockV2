import glob, json, re
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from research.four_signal_backtest import prepare, signals

def scan(path):
    symbol, f = prepare(path)
    if f is None: return []
    sig = signals(f, symbol)
    out = []
    n = len(f)
    q = ((sig["count"] >= 3) & sig.ok)
    for i in range(max(0, n - 2), n):
        if q.iloc[i] and sig.bias.iloc[i] <= 0.05:
            # D-Day for the 5% cap = first day of the >=3-signal streak whose bias <= 5%
            j = i
            while j - 1 >= 0 and q.iloc[j - 1]: j -= 1
            first_ok = next(t for t in range(j, i + 1) if sig.bias.iloc[t] <= 0.05)
            out.append({"symbol": symbol, "date": f.date.iloc[i], "is_d_day": first_ok == i, "d_day": f.date.iloc[first_ok],
                        "signals": "".join(k[-1] for k in ("s1","s2","s3","s4") if sig[k].iloc[i]), "count": int(sig["count"].iloc[i]),
                        "bias_pct": round(float(sig.bias.iloc[i]) * 100, 2), "close": float(f.close.iloc[i]), "pct_chg": float(f.pct_chg.iloc[i]),
                        "yin": bool(f.close.iloc[i] < f.open.iloc[i]), "amount_yi": round(float(f.amount.iloc[i]) / 1e8, 2),
                        "amt_ratio_max5": round(float(sig.ratio.iloc[max(0, i-4):i+1].max()), 2)})
    return out

if __name__ == "__main__":
    files = sorted(p for p in glob.glob("data/*_daily.csv") if re.search(r"(sh_(60|68)|sz_(00|30))\d{4}_daily\.csv$", p))
    rows = []
    with ProcessPoolExecutor(8) as ex:
        for r in ex.map(scan, files, chunksize=20): rows.extend(r)
    df = pd.DataFrame(rows)
    mp = json.load(open("concept_mapping.json", encoding="utf-8"))["mapping"]
    df["name"] = df.symbol.map(lambda s: mp.get(s, {}).get("name", ""))
    df["industry"] = df.symbol.map(lambda s: mp.get(s, {}).get("industry", ""))
    df = df.sort_values(["date", "bias_pct"], ascending=[False, True])
    df.to_csv("results/research/four_signal/latest_screen.csv", index=False, encoding="utf-8-sig")
    print(df.groupby("date").size().to_string())
    for d, g in df.groupby("date", sort=False):
        print("\n===", d, len(g), "| D-Day today:", int(g.is_d_day.sum()), "| no-S1:", int((~g.signals.str.contains("1")).sum()))
        print(g[["symbol","name","industry","signals","bias_pct","close","pct_chg","yin","amount_yi","amt_ratio_max5","is_d_day","d_day"]].to_string(index=False))
