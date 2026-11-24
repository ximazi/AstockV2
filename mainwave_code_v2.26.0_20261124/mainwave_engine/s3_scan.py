"""Daily S3 observation scan (platform breakout). Research only: S3 is not live; nothing is recommended or bought.

Signal (research/four_dim_v3.py, S3场景分析_20260925.md), main board (60/00), non-ST, continuous-adjusted prices:
  trend     MA20 > MA60, MA60 above its value 20 sessions ago, close > MA60
  base      the longest window among 7/10/15/20/30/40/60 sessions before today with (max high - min low) / min low < 15%
  volume    base mean volume <= 0.8 x the 20 sessions before the base, and today's volume >= 2 x the base mean
  trigger   close > base high and up on the day; close/EMA20 - 1 <= 12%, close/MA100 - 1 <= 35%, no MACD top divergence
Pre-structure class (research/s3_structure.py, only facts known today): 受损后修复 / 长平台释放 / 趋势内整理释放 / 其他.
"受损后修复" is a downgrade, not an exclusion (three winners of the case review fell in it).
Exit contract if S3 goes live (research/accept_failure.py A1h): LAD, plus two closes back under the base top within
5 sessions -> sell half; a close under the base midpoint within 10 sessions -> sell the rest.
python -m mainwave_engine.s3_scan --date YYYY-MM-DD
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import s4_scan as S4
from . import stock_analyzer as SA
from . import strong_trend as ST
from .trading import position_monitor as PM

VERSION = "s3_scan_v1_20260926"
ROOT = Path(__file__).resolve().parents[1]
WINDOWS = (7, 10, 15, 20, 30, 40, 60)
CLASS_NOTE = {"受损后修复": "降级观察：此前大涨后回撤 ≥20%，仍低于前高 12% 以上", "长平台释放": "此前无大涨、整理 ≥15 天（样本少）",
              "趋势内整理释放": "前段上涨未被破坏、平台接近前高", "其他": ""}
NOTICE = ("S3 平台突破观察：研究场景，尚未上线，不推荐、不记入虚拟持仓。2025 年低位与长平台更稳，2026 年未复现；"
          "受损后修复只降级不排除。离场若上线按分批止盈 + 平台否定（A1h）。")


def base(f):
    """Longest qualifying base ending yesterday: (length, high, low, mean volume, pre-base mean volume) or None."""
    h, l, v = (f[k].to_numpy(float) for k in ("high", "low", "volume"))
    t, found = len(f) - 1, None
    for k in WINDOWS:
        if t - k - 20 < 0:
            break
        hi, lo = h[t - k:t].max(), l[t - k:t].min()
        if (hi - lo) / lo < 0.15:
            found = (k, hi, lo, v[t - k:t].mean(), v[t - k - 20:t - k].mean())
    return found


def classify(f, base_len):
    h, l, c = (f[k].to_numpy(float) for k in ("high", "low", "close"))
    t = len(f) - 1
    lo = max(0, t - base_len - 120)
    kp = lo + int(np.argmax(h[lo:t]))
    peak = h[kp]
    run = peak / l[max(0, kp - 60):kp + 1].min() - 1
    dmg = (l[kp + 1:t].min() / peak - 1) if kp + 1 < t else 0.0
    below = c[t] / peak - 1
    if run >= 0.25 and dmg <= -0.20 and below <= -0.12:
        k = "受损后修复"
    elif run < 0.25 and base_len >= 15:
        k = "长平台释放"
    elif dmg > -0.15 and below > -0.12:
        k = "趋势内整理释放"
    else:
        k = "其他"
    return k, {"prior_run": round(run * 100, 1), "damage": round(dmg * 100, 1), "below_peak": round(below * 100, 1)}


def signal(f):
    c = f["close"].astype(float)
    ma20, ma60, ma100 = (c.rolling(n).mean() for n in (20, 60, 100))
    ema20 = c.ewm(span=20, adjust=False).mean()
    t = len(f) - 1
    if t < 130 or not (ma20.iloc[t] > ma60.iloc[t] > 0 and ma60.iloc[t] > ma60.iloc[t - 20] and c.iloc[t] > ma60.iloc[t]):
        return None
    b = base(f)
    if not b or b[0] < 7:
        return None
    n, top, bottom, bvol, pvol = b
    v = float(f["volume"].iloc[t])
    if not (c.iloc[t] > top and c.iloc[t] > c.iloc[t - 1] and bvol <= 0.8 * pvol and v >= 2 * bvol):
        return None
    if c.iloc[t] / ema20.iloc[t] - 1 > 0.12 or c.iloc[t] / ma100.iloc[t] - 1 > 0.35 or S4.signals(f)["R1"].iloc[t]:
        return None
    return {"base_len": n, "top": top, "bottom": bottom, "amp": top / bottom - 1, "base_pre": bvol / pvol, "sig_base": v / bvol,
            "bias": float(c.iloc[t] / ema20.iloc[t] - 1)}


def scan(day, data_root=None):
    data_root = Path(data_root or ROOT / "data")
    names = ST.names(ROOT)
    mp = json.loads((ROOT / "concept_mapping.json").read_text(encoding="utf-8")).get("mapping", {})
    rows = []
    for path in sorted(data_root.glob("s[hz]_*_daily.csv")):
        sym = path.name[:9]
        if sym[:5] not in ("sh_60", "sz_00"):
            continue
        try:
            raw = pd.read_csv(path, usecols=lambda c: c in {"date", "open", "high", "low", "close", "preclose", "volume",
                                                           "amount", "tradestatus", "isST"})
        except (OSError, ValueError):
            continue
        raw["date"] = raw["date"].astype(str).str[:10]
        raw = raw[raw["date"] <= day].drop_duplicates("date").tail(260).reset_index(drop=True)
        name = str(names.get(sym, ""))
        if (len(raw) < 150 or raw["date"].iloc[-1] != day or float(raw["tradestatus"].iloc[-1]) != 1
                or float(raw["isST"].iloc[-1] or 0) == 1 or "ST" in name.upper() or "退" in name):
            continue
        f = ST.adjusted(raw)
        s = signal(f)
        if not s:
            continue
        cls, facts = classify(f, s["base_len"])
        scale = float(f["adj_factor"].iloc[-1])
        close = float(raw["close"].iloc[-1])
        top, mid = s["top"] / scale, (s["top"] + s["bottom"]) / 2 / scale
        rows.append({"symbol": sym, "name": name, "industry": (mp.get(sym) or {}).get("industry") or "", "close": close,
                     "pct": round((float(f["close"].iloc[-1]) / float(f["close"].iloc[-2]) - 1) * 100, 2),
                     "base_len": s["base_len"], "base_amp": round(s["amp"] * 100, 1), "base_top": round(top, 3),
                     "base_mid": round(mid, 3), "vol_base_pre": round(s["base_pre"], 2), "vol_sig_base": round(s["sig_base"], 2),
                     "bias_ema20": round(s["bias"] * 100, 1), "structure": cls, "structure_note": CLASS_NOTE[cls], **facts,
                     "amount_yi": round(float(raw["amount"].iloc[-1]) / 1e8, 2),
                     "plan": {"stop": round(close * 0.90, 2), "tp1": round(close * 1.10, 2), "tp2": round(close * 1.20, 2),
                              "denial": f"5 日内两次收盘 < {top:.2f}（平台上沿）卖一半；10 日内收盘 < {mid:.2f}（平台中线）卖出剩余"}})
    order = {"长平台释放": 0, "趋势内整理释放": 1, "其他": 2, "受损后修复": 3}
    rows.sort(key=lambda r: (order[r["structure"]], -r["amount_yi"]))
    return {"as_of": day, "version": VERSION, "notice": NOTICE, "market_risk": PM.market_risk(day, data_root),
            "observations": rows}


def render(out):
    lines = [f"# S3 平台突破观察 {out['as_of']}（研究，未上线）", "", out["notice"], "", f"信号 {len(out['observations'])} 只。", ""]
    if out["observations"]:
        lines += ["|股票|结构|整理|整理量/前量|信号量/整理量|偏离EMA20|距前高|平台上沿 / 中线|否定即走|",
                  "|---|---|---|---:|---:|---:|---:|---|---|"]
        for r in out["observations"]:
            lines.append(f"|{r['name']} {r['symbol'][3:]}|{r['structure']}|{r['base_len']} 日 / {r['base_amp']}%|{r['vol_base_pre']}|"
                         f"{r['vol_sig_base']}|{r['bias_ema20']:+.1f}%|{r['below_peak']:.1f}%|{r['base_top']} / {r['base_mid']}|{r['plan']['denial']}|")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    out = scan(args.date)
    folder = SA.OUT_ROOT / args.date.replace("-", "")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "s3_observations.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (folder / "s3_observations.md").write_text(render(out), encoding="utf-8")
    kinds = pd.Series([r["structure"] for r in out["observations"]]).value_counts().to_dict()
    print(f"[s3_scan] {args.date} 信号 {len(out['observations'])} {kinds} -> {folder / 's3_observations.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
