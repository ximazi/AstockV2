"""Recall / precision study of the pre-registered preparation detectors on labelled TREND episodes.

One pass over every bar file: the route frame is built once per symbol, detectors are annotated once, then
each year's span is scored. Recall side: for every eligible episode start, the first firing inside
[t-10, t+5] and the extension at that moment. Precision side: every new firing on an eligible day of the
full universe, and whether an episode start followed within 10 sessions; the base rate is the same
quantity for every eligible day. Design year 2026-01..04, holdout 2025; the choice rule is stated in
mainwave_engine.preparation_detector.CHOICE_RULE before the numbers exist."""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from mainwave_engine import preparation_detector as PD, route_evidence as RE

DATA = Path("data")
OUT = Path("results/research/preparation_detector")
BEFORE, AFTER, PRECISION_WINDOW = 10, 5, 10
YEARS = {"2026": ("results/mainwave_episodes_2026.csv", "2026-01-01", "2026-04-30"),
         "2025": ("results/mainwave_episodes_2025.csv", "2025-01-01", "2025-12-31")}


def bucket(lead):
    return "EARLY" if lead <= -1 else "ON_TIME" if lead <= 2 else "LATE"


def load_episodes():
    by_symbol = defaultdict(lambda: defaultdict(list))
    meta = {}
    for year, (path, lo, hi) in YEARS.items():
        e = pd.read_csv(path)
        e = e[(e.cls == "TREND") & (e.start.astype(str) >= lo) & (e.start.astype(str) <= hi)]
        for r in e.itertuples():
            by_symbol[r.stem][year].append(str(r.start))
            meta[(r.stem, str(r.start))] = {"name": r.name, "board": r.board, "max_gain": r.max_gain, "ret60_prior": r.ret60_prior}
    return by_symbol, meta


def score_symbol(symbol, path, episodes_by_year, bench):
    raw = pd.read_csv(path, encoding="utf-8-sig")
    frame = RE.build(raw, symbol)
    if frame.empty:
        return [], [], {}
    ann = PD.annotate(frame, bench)
    dates = frame.date.dt.strftime("%Y-%m-%d").tolist()
    index = {d: i for i, d in enumerate(dates)}
    eligible = frame.eligible.fillna(False).astype(bool).tolist()
    bias = frame.bias_atr.tolist()
    flags = {v: ann[v].fillna(False).astype(bool).tolist() for v in PD.VARIANTS}
    episode_rows, firing_rows, base = [], [], {}
    for year, (_, lo, hi) in YEARS.items():
        starts = sorted(index[s] for s in episodes_by_year.get(year, []) if s in index)
        for s in starts:
            if not eligible[s]:
                episode_rows.append({"year": year, "symbol": symbol, "start": dates[s], "status": "INELIGIBLE"})
                continue
            row = {"year": year, "symbol": symbol, "start": dates[s], "status": "ELIGIBLE"}
            lo_i, hi_i = max(0, s - BEFORE), min(len(dates) - 1, s + AFTER)
            for v in PD.VARIANTS:
                hit = next((j for j in range(lo_i, hi_i + 1) if flags[v][j]), None)
                row[f"{v}_lead"] = (hit - s) if hit is not None else None
                row[f"{v}_bucket"] = bucket(hit - s) if hit is not None else "MISS"
                row[f"{v}_bias"] = round(float(bias[hit]), 2) if hit is not None and pd.notna(bias[hit]) else None
            episode_rows.append(row)
        # A firing is mature only when every start it could be followed by lies inside the labelled span
        # (start <= hi) and inside the bars: the 10-session window must end on or before the last labelled
        # session. Later firings are CENSORED and excluded from precision and from the base rate.
        last_label_index = max((i for i, d in enumerate(dates) if d <= hi), default=-1)
        mature_cut = last_label_index - PRECISION_WINDOW
        days = [i for i, d in enumerate(dates) if lo <= d <= hi and eligible[i]]
        mature_days = [i for i in days if i <= mature_cut]
        followed = [any(i <= s <= i + PRECISION_WINDOW for s in starts) for i in mature_days]
        base[year] = (sum(followed), len(mature_days))
        for v in PD.VARIANTS:
            prev = False
            for i in days:
                on = flags[v][i]
                if on and not prev:
                    firing_rows.append({"year": year, "symbol": symbol, "detector": v, "date": dates[i],
                                        "episode_within_10": any(i <= s <= i + PRECISION_WINDOW for s in starts),
                                        "censored": i > mature_cut,
                                        "bias_atr": round(float(bias[i]), 2) if pd.notna(bias[i]) else None})
                prev = on
            # firing state carries across non-eligible gaps by design of the scan above
    return episode_rows, firing_rows, base


def run(limit=None, out_root=OUT):
    bench = PD.benchmark_ret20()
    by_symbol, meta = load_episodes()
    paths = sorted(glob.glob(str(DATA / "s[hz]_*_daily.csv")))
    if limit:
        paths = paths[:limit]
    episodes, firings, base = [], [], defaultdict(lambda: [0, 0])
    started = datetime.now()
    for n, path in enumerate(paths, 1):
        symbol = Path(path).name[:-len("_daily.csv")]
        try:
            e, f, b = score_symbol(symbol, path, by_symbol.get(symbol, {}), bench)
        except Exception as exc:  # one broken file must not end a 5000-symbol pass; it is counted
            episodes.append({"year": "", "symbol": symbol, "start": "", "status": f"ERROR:{str(exc)[:80]}"})
            continue
        episodes.extend(e)
        firings.extend(f)
        for year, (h, t) in b.items():
            base[year][0] += h
            base[year][1] += t
        if n % 500 == 0 or n == len(paths):
            print(f"[prep] {n}/{len(paths)} symbols, episodes {len(episodes)}, firings {len(firings)}, {(datetime.now() - started).seconds}s", flush=True)
    out = Path(out_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    ep = pd.DataFrame(episodes)
    fr = pd.DataFrame(firings)
    ep.to_csv(out / "episodes.csv", index=False, encoding="utf-8-sig")
    fr.to_csv(out / "firings.csv", index=False, encoding="utf-8-sig")
    summary = summarize(ep, fr, {y: {"hit": v[0], "days": v[1], "rate": round(v[0] / v[1], 4) if v[1] else None} for y, v in base.items()})
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (out / "report.md").write_text(report(summary), encoding="utf-8")
    print(f"[prep] written {out}")
    return out, summary


def summarize(ep, fr, base):
    years = {}
    for year in YEARS:
        e = ep[(ep.year == year) & (ep.status == "ELIGIBLE")]
        f = fr[fr.year == year] if len(fr) else fr
        det = {}
        for v in PD.VARIANTS:
            hits = e[f"{v}_bucket"].ne("MISS") if len(e) else pd.Series(dtype=bool)
            g_all = f[f.detector == v] if len(f) else f
            g = g_all[~g_all.censored] if len(g_all) and "censored" in g_all else g_all
            det[v] = {"recall": round(float(hits.mean()), 4) if len(e) else None,
                      "firings_censored": int(g_all.censored.sum()) if len(g_all) and "censored" in g_all else 0,
                      "buckets": dict(Counter(e[f"{v}_bucket"])) if len(e) else {},
                      "early_or_on_time": round(float(e[f"{v}_bucket"].isin(["EARLY", "ON_TIME"]).mean()), 4) if len(e) else None,
                      "median_lead": float(e[f"{v}_lead"].median()) if len(e) and e[f"{v}_lead"].notna().any() else None,
                      "median_bias_at_hit": float(e[f"{v}_bias"].median()) if len(e) and e[f"{v}_bias"].notna().any() else None,
                      "firings": int(len(g)), "precision": round(float(g.episode_within_10.mean()), 4) if len(g) else None,
                      "median_bias_at_firing": float(g.bias_atr.median()) if len(g) and g.bias_atr.notna().any() else None}
        years[year] = {"episodes_eligible": int(len(e)), "episodes_ineligible": int(((ep.year == year) & (ep.status == "INELIGIBLE")).sum()),
                       "base_rate": base.get(year), "detectors": det}
    design = years["2026"]
    rule = PD.CHOICE_RULE
    qualified = [v for v in PD.VARIANTS[1:] if design["detectors"][v]["precision"] is not None and design["base_rate"] and design["base_rate"]["rate"] is not None
                 and design["detectors"][v]["precision"] >= design["base_rate"]["rate"]
                 and design["detectors"][v]["median_bias_at_hit"] is not None and design["detectors"][v]["median_bias_at_hit"] <= rule["max_median_bias_at_hit"]]
    chosen = max(qualified, key=lambda v: design["detectors"][v]["recall"] or 0) if qualified else None
    return {"version": PD.VERSION, "rule": rule, "labels": PD.LABELS, "years": years, "qualified_on_design": qualified, "chosen": chosen,
            "errors": int(ep.status.astype(str).str.startswith("ERROR").sum()) if len(ep) else 0}


def report(s):
    lines = [f"# 准备检测器研究 · {s['version']}", "",
             "预注册变体与阈值见 `mainwave_engine/preparation_detector.py`；选择规则：设计年（2026-01..04）里精度不低于基率、命中时中位偏离 ≤1.5ATR 的变体中取召回最高者，再看 2025 历史稳健性检验（不是设计后的前向验证）。",
             "精度与基率只在成熟触发/成熟交易日上计算：触发后 10 个交易日须全部落在标签覆盖区间内，否则记 CENSORED 并排除。", ""]
    for year, y in s["years"].items():
        br = y["base_rate"] or {}
        lines += [f"## {year}：合格起点 {y['episodes_eligible']}（不合格 {y['episodes_ineligible']}）；全市场基率 {br.get('rate')}（{br.get('hit')}/{br.get('days')} 个合格交易日 10 日内有起点）", "",
                  "| 检测器 | 召回 | 提前或准时 | 中位提前量 | 命中时中位偏离ATR | 成熟触发次数 | 未成熟(CENSORED) | 精度（10日内有起点，仅成熟） | 触发时中位偏离ATR |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for v, d in y["detectors"].items():
            lines.append(f"| {v} | {d['recall']} | {d['early_or_on_time']} | {d['median_lead']} | {d['median_bias_at_hit']} | {d['firings']} | {d.get('firings_censored', 0)} | {d['precision']} | {d['median_bias_at_firing']} |")
        lines.append("")
    lines += [f"设计年通过选择规则的变体：{s['qualified_on_design'] or '无'}；选定：{s['chosen'] or '无（没有变体同时满足精度与偏离要求）'}。", "",
              "## 读法与边界", "",
              "- 标签是事后前向收益定义；召回只看窗口内是否触发，不含可成交性与成本。",
              "- 精度与基率都在全市场合格交易日上计算，可直接比较；高于基率才说明有判别力。",
              "- 2025 是留出年，只用于确认设计年的选择，不再据此改阈值。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.limit)


if __name__ == "__main__":
    main()
