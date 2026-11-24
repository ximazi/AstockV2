"""Daily S1 / S2 / S5 observation lists and the one-file export of the day's S1-S5 opportunities.

S1 新结构突破 / S2 调整后再转强: the day's BREAKOUT / REACCELERATION events of the strong-trend core shadow
  (strong_trend_shadow.json, the same core events the research used), main board (60/00), non-ST.
  Exit for both if they go live: staged profit-taking plus P3 (two closes below the signal day's low within 5 sessions
  after entry -> exit at the next open). P3 was tested on the full S1 and S2 samples (S1S5评审落实_20260926.md §6).
  Execution: the core plan's own price cap (next open above close + 0.5 ATR -> the plan expires) is shown as an
  execution constraint, not a new filter; as a filter it changed nothing on the full sample.
S5 底部反转 (MA20>MA60): research/scenario_catalog.bottom_reversal, strict variant (user decision 2026-09-24).
  P2 (底部仍在下探) was rejected by the full-sample test and is not applied.
All three are research observations: not live, not recommended, not added to the virtual portfolio. Only S4 is live.
Export: results/mainwave_engine/<day>/scenario_opportunities_<day>.csv (S1-S5, one row per stock and scenario).
python -m mainwave_engine.scenario_scan --date YYYY-MM-DD [--export-only]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import stock_analyzer as SA
from . import strong_trend as ST
from .trading import position_monitor as PM

# v2 (2026-09-29): S2 drops days that rose < S2_MIN_PCT, a new detector version; statistics of the v1 S2 sample
# (no minimum rise) do not carry over to it.
VERSION = "scenario_scan_v2_20260929"
ROOT = Path(__file__).resolve().parents[1]
NOTICE = {"s1": "S1 新结构突破：研究观察，未上线，不推荐。排序未验证。若上线，离场按分批止盈 + P3（买后 5 日内两次收盘跌破信号日低点即走）。",
          "s2": "S2 调整后再转强：研究观察，未上线，不推荐。排序未验证。若上线，离场按分批止盈 + P3（买后 5 日内两次收盘跌破信号日低点即走）；"
                "次日开盘超出计划买价上限即过期。",
          "s5": "S5 底部反转（MA20>MA60）：研究观察，未上线，不推荐。样本少（2025 年 277、2026 年 30 个信号）；"
                "底部仍在下探的过滤（P2）经全样本检验无效，未采用。"}
ENTRY = {"s1": "BREAKOUT", "s2": "REACCELERATION"}


def _raw(data_root, sym, day, tail=400):
    raw = pd.read_csv(Path(data_root) / f"{sym}_daily.csv",
                      usecols=lambda c: c in {"date", "open", "high", "low", "close", "preclose", "volume", "amount", "tradestatus", "isST"})
    raw["date"] = raw["date"].astype(str).str[:10]
    return raw[raw["date"] <= day].drop_duplicates("date").tail(tail).reset_index(drop=True)


def _plan(close, denial):
    return {"stop": round(close * 0.90, 2), "tp1": round(close * 1.10, 2), "tp2": round(close * 1.20, 2), "denial": denial}


S2_MIN_PCT = 2.0  # a re-acceleration day that rose less than this is a defensive drift, not a re-acceleration (2026-09-28 review)


def core_events(kind, day, data_root, names, industry):
    path = SA.OUT_ROOT / day.replace("-", "") / "strong_trend_shadow.json"
    if not path.exists():
        return None
    board = json.loads(path.read_text(encoding="utf-8")).get("boards", {}).get("attention", [])
    rows = []
    for r in board:
        sym, name = r["symbol"], str(r.get("name") or names.get(r["symbol"], ""))
        if r.get("entry") != ENTRY[kind] or sym[:5] not in ("sh_60", "sz_00") or "ST" in name.upper() or "退" in name:
            continue
        if kind == "s2" and float(r.get("pct_chg") or 0) < S2_MIN_PCT:
            continue
        raw = _raw(data_root, sym, day, tail=5)
        if raw.empty or raw["date"].iloc[-1] != day or float(raw["isST"].iloc[-1] or 0) == 1:
            continue
        low = float(raw["low"].iloc[-1])
        denial = f"买后 5 日内两次收盘 < {low:.2f}（信号日低点）即走"
        cap = (r.get("band") or [None, None])[1]
        rows.append({"symbol": sym, "name": name, "industry": industry.get(sym) or "", "close": float(r["close"]),
                     "pct": r.get("pct_chg"), "tag": "突破" if kind == "s1" else "再转强",
                     "boundary": r.get("boundary"), "invalidation": r.get("invalidation"), "rs20": r.get("rs20"),
                     "plan_legal": r.get("plan_legal"), "close_at_limit": r.get("close_at_limit"), "price_cap": cap,
                     "note": "；".join(r.get("problems") or []) or ("收盘涨停，次日可能买不到" if r.get("close_at_limit") else ""),
                     "plan": {**_plan(float(r["close"]), denial),
                              "price_cap": f"次日开盘 > {cap:.2f} 则计划过期，不追" if cap else ""}})
    return rows


def bottom_reversal(f):
    """research/scenario_catalog.bottom_reversal, strict MA20 > MA60, evaluated on the last bar."""
    c, h, l, v = (f[x].astype(float) for x in ("close", "high", "low", "volume"))
    dif = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    dea = dif.ewm(span=9, adjust=False).mean()
    high120 = h.rolling(120).max()
    low_since = l.rolling(100).min()
    low_age = l.rolling(100).apply(lambda x: len(x) - 1 - int(np.argmin(x)), raw=True)
    fall = low_since / high120.shift(20) - 1 <= -0.20
    slowing = dif.rolling(40).min() > dif.shift(40).rolling(40).min()
    dries = v.shift(1).rolling(20).mean() <= 0.7 * v.shift(21).rolling(40).mean()
    rng = (h.shift(1).rolling(20).max() - l.shift(1).rolling(20).min()) / l.shift(1).rolling(20).min() < 0.15
    cross = (dif > dea) & (dif.shift(1) <= dea.shift(1))
    zero = (dif / c).between(-0.015, 0.005)
    volx = v >= 1.3 * v.shift(1).rolling(20).mean()
    ma20, ma60 = c.rolling(20).mean(), c.rolling(60).mean()
    sig = fall & (low_age >= 20) & slowing & dries & rng & cross & zero & volx & (c > ma20) & (ma20 > ma60)
    t = len(f) - 1
    if not bool(sig.fillna(False).iloc[t]):
        return None
    return {"dif_pct": round(float(dif.iloc[t] / c.iloc[t]) * 100, 2),
            "drawdown": round(float(low_since.iloc[t] / high120.shift(20).iloc[t] - 1) * 100, 1),
            "vol_ratio": round(float(v.iloc[t] / v.shift(1).rolling(20).mean().iloc[t]), 2)}


def s5_events(day, data_root, names, industry):
    rows = []
    for path in sorted(Path(data_root).glob("s[hz]_*_daily.csv")):
        sym = path.name[:9]
        if sym[:5] not in ("sh_60", "sz_00"):
            continue
        try:
            raw = _raw(data_root, sym, day)
        except (OSError, ValueError):
            continue
        name = str(names.get(sym, ""))
        if (len(raw) < 250 or raw["date"].iloc[-1] != day or float(raw["tradestatus"].iloc[-1]) != 1
                or float(raw["isST"].iloc[-1] or 0) == 1 or "ST" in name.upper() or "退" in name):
            continue
        s = bottom_reversal(ST.adjusted(raw))
        if not s:
            continue
        close = float(raw["close"].iloc[-1])
        rows.append({"symbol": sym, "name": name, "industry": industry.get(sym) or "", "close": close,
                     "pct": round((close / float(raw["preclose"].iloc[-1]) - 1) * 100, 2), "tag": "底部反转", **s,
                     "note": f"此前回撤 {s['drawdown']}%，DIF/股价 {s['dif_pct']}%，量比 {s['vol_ratio']}",
                     "plan": _plan(close, "无（仅分批止盈；信号低点止错会误伤 S5 赢家，不采用）")})
    return rows


def scan(kind, day, data_root=None):
    data_root = Path(data_root or ROOT / "data")
    names = ST.names(ROOT)
    mp = json.loads((ROOT / "concept_mapping.json").read_text(encoding="utf-8")).get("mapping", {})
    industry = {s: (v or {}).get("industry") for s, v in mp.items()}
    rows = s5_events(day, data_root, names, industry) if kind == "s5" else core_events(kind, day, data_root, names, industry)
    out = {"as_of": day, "version": VERSION, "kind": kind, **({"s2_min_pct": S2_MIN_PCT} if kind == "s2" else {}), "notice": NOTICE[kind], "market_risk": PM.market_risk(day, data_root)}
    if rows is None:
        return {**out, "missing_source": "strong_trend_shadow.json 缺失，先运行强势趋势影子", "observations": []}
    return {**out, "observations": sorted(rows, key=lambda r: -(r["pct"] or 0))}


EXPORT_COLUMNS = ["日期", "场景", "状态", "代码", "名称", "行业", "收盘", "涨幅%", "止损(-10%)", "止盈1(+10%)", "止盈2(+20%)",
                  "否定即走", "买价上限", "说明"]


def export(day, folder=None):
    """One CSV with the day's S1-S5 opportunities from the files each scan wrote; a missing scan is listed, not faked."""
    folder = Path(folder or SA.OUT_ROOT / day.replace("-", ""))
    rows, missing = [], []
    sources = [("S1 新结构突破", "s1_observations.json"), ("S2 调整后再转强", "s2_observations.json"),
               ("S3 平台突破", "s3_observations.json"), ("S4 回踩再启动", "s4_opportunities.json"), ("S5 底部反转", "s5_observations.json")]
    for label, name in sources:
        path = folder / name
        if not path.exists():
            missing.append(label)
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        groups = ([("推荐", r) for r in d.get("recommended", [])] + [("排除：" + "；".join(r.get("excluded", [])), r) for r in d.get("excluded", [])]
                  if name.startswith("s4") else [("研究观察", r) for r in d.get("observations", []) if not r.get("close_at_limit")]
                  + [("研究观察（涨停，次日可能买不到）", r) for r in d.get("observations", []) if r.get("close_at_limit")])
        for status, r in groups:
            p = r.get("plan") or {}
            note = r.get("note") or r.get("structure", "") or (f"行业内强度 {r['lead']:.0%}" if r.get("lead") is not None else "")
            rows.append([day, label, status, r["symbol"][3:], r.get("name", ""), r.get("industry", ""), r.get("close"),
                         r.get("pct"), p.get("stop"), p.get("tp1"), p.get("tp2"), p.get("denial", ""), p.get("price_cap", ""), note])
    path = folder / f"scenario_opportunities_{day.replace('-', '')}.csv"
    folder.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(EXPORT_COLUMNS)
        w.writerows(rows)
        for label in missing:
            w.writerow([day, label, "未生成（当日扫描缺失）"] + [""] * (len(EXPORT_COLUMNS) - 3))
    return path, len(rows), missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    folder = SA.OUT_ROOT / args.date.replace("-", "")
    if not args.export_only:
        folder.mkdir(parents=True, exist_ok=True)
        for kind in ("s1", "s2", "s5"):
            out = scan(kind, args.date)
            (folder / f"{kind}_observations.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            print(f"[scenario_scan] {args.date} {kind.upper()} {len(out['observations'])} {out.get('missing_source', '')}", flush=True)
    path, n, missing = export(args.date)
    print(f"[scenario_scan] 导出 {n} 条 S1-S5 机会点 -> {path}" + (f"；缺失 {'、'.join(missing)}" if missing else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
