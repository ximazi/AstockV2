"""Same-sample ablation of the screening order: frozen rule → tracking re-eligibility → today's position and
trigger → stage and route → multi-day change → portfolio choice, on the archived closure inputs.

Each step is a stated ordering rule from mainwave_engine.trading.ranking_shadow, applied to the very same
candidate records and features the frozen rule saw. Forward outcomes are reported only where bars exist,
with their sample size; they are diagnostics for a shadow, never a calibration target."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from mainwave_engine.trading import daily, ranking, ranking_shadow as RS

ROOT = Path(".")
ENGINE = ROOT / "results/mainwave_engine"
DATA = ROOT / "data"
OUT = ROOT / "results/research/screening_ablation"
HORIZONS = (1, 3, 5, 10)
TOP = (3, 5, 10)


def archived_days():
    return sorted(p.name[:4] + "-" + p.name[4:6] + "-" + p.name[6:] for p in ENGINE.glob("2026????")
                  if (p / "analysis_universe.json").exists() and (p / "analysis_evidence.json").exists())


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


class Bars:
    def __init__(self):
        self.cache = {}

    def frame(self, symbol):
        if symbol not in self.cache:
            try:
                f = pd.read_csv(DATA / f"{symbol}_daily.csv", encoding="utf-8-sig", usecols=["date", "close"])
                f["date"] = pd.to_datetime(f.date).dt.strftime("%Y-%m-%d")
                self.cache[symbol] = f.sort_values("date").drop_duplicates("date").reset_index(drop=True)
            except (OSError, ValueError):
                self.cache[symbol] = pd.DataFrame(columns=["date", "close"])
        return self.cache[symbol]

    def forward(self, symbol, day):
        """Close-to-close returns after `day`; None when the future bar is not yet in the data."""
        f = self.frame(symbol)
        idx = f.index[f.date == day]
        out = {}
        if not len(idx):
            return {h: None for h in HORIZONS}
        i = int(idx[0])
        base = float(f.close.iloc[i])
        for h in HORIZONS:
            out[h] = round(float(f.close.iloc[i + h] / base - 1) * 100, 3) if i + h < len(f) and base > 0 else None
        return out


def build_frozen(day, previous):
    folder = ENGINE / day.replace("-", "")
    universe, evidence = read(folder / "analysis_universe.json"), read(folder / "analysis_evidence.json")
    metrics = daily.load_features(universe["records"], DATA, day)
    payload = ranking.build(universe, evidence, metrics, daily.benchmark(DATA, day), previous, {})
    payload["snapshot_id"] = f"ablation:{day}"
    return payload


def composition(rows):
    n = len(rows) or 1
    bias = [r["bias_atr"] for r in rows if r["bias_atr"] is not None]
    r20 = [r["trend"]["return20"] for r in rows if r["trend"]["return20"] is not None]
    return {"n": len(rows), "extended_share": round(sum(r["position"] == "EXTENDED" for r in rows) / n, 3),
            "late_or_broken_share": round(sum(r["stage_bucket"] in ("LATE", "FAILED", "REPAIRING") for r in rows) / n, 3),
            "early_share": round(sum(r["stage_bucket"] == "EARLY" for r in rows) / n, 3),
            "tracking_share": round(sum(r["tracking_only"] for r in rows) / n, 3),
            "leading_theme_share": round(sum(r["mainline"] == "STRONG" or r.get("market_theme") == "STRONG" for r in rows) / n, 3),
            "mean_bias_atr": round(statistics.fmean(bias), 2) if bias else None,
            "mean_return20": round(statistics.fmean(r20), 2) if r20 else None}


def outcomes(rows, bars, day):
    out = {}
    for h in HORIZONS:
        vals = [bars.forward(r["symbol"], day)[h] for r in rows]
        vals = [v for v in vals if v is not None]
        out[h] = {"n": len(vals), "mean": round(statistics.fmean(vals), 3) if vals else None,
                  "median": round(statistics.median(vals), 3) if vals else None,
                  "up_share": round(sum(v > 0 for v in vals) / len(vals), 3) if vals else None}
    return out


def run(days=None, out_root=OUT, limit_days=None):
    days = days or archived_days()
    if limit_days:
        days = days[-limit_days:]
    bars = Bars()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(out_root) / stamp
    out.mkdir(parents=True, exist_ok=True)
    previous_payload, previous_shadow = None, None
    per_day, rows_out = [], []
    for day in days:
        payload = build_frozen(day, previous_payload)
        shadow = RS.evaluate(payload, previous_shadow)
        bench = daily.benchmark(DATA, day)
        record = {"as_of": day, "records": len(shadow["rows"]), "market": payload.get("market", {}).get("state"),
                  "re_eligible": shadow["summary"]["re_eligible"], "themes_leading": shadow["summary"]["cohort_strong"],
                  "benchmark_forward": bars.forward("000300_SH_index", day) if (DATA / "000300_SH_index_daily.csv").exists() else None,
                  "variants": {}}
        for variant in RS.VARIANTS:
            ordered = sorted(shadow["rows"], key=lambda r: r["variant_ranks"][variant])
            entry = {}
            for k in TOP:
                top = ordered[:k]
                entry[f"top{k}"] = {"symbols": [r["symbol"] for r in top], "composition": composition(top), "forward": outcomes(top, bars, day),
                                   "overlap_with_frozen": len({r["symbol"] for r in top} & {r["symbol"] for r in sorted(shadow["rows"], key=lambda r: r["variant_ranks"]["V0_frozen"])[:k]})}
            record["variants"][variant] = entry
        chosen = RS.select_review_shortlist(sorted(shadow["rows"], key=lambda r: r["variant_ranks"]["V4_multiday"]))
        record["variants"]["V5_portfolio"] = {"chosen": [{"symbol": r["symbol"], "name": r["name"], "theme": r["theme"], "reading": r["reading"]} for r in chosen],
                                             "forward": outcomes(chosen, bars, day), "composition": composition(chosen)}
        per_day.append(record)
        for r in shadow["rows"]:
            rows_out.append({"as_of": day, "symbol": r["symbol"], "name": r["name"], "theme": r["theme"], "frozen_state": r["frozen_state"], "state": r["state"],
                             "re_eligible": r["re_eligible"], "stage_bucket": r["stage_bucket"], "stage": r["stage"], "position": r["position"], "bias_atr": r["bias_atr"],
                             "route": r["route"], "trigger": r["trigger"], "mainline": r["mainline"], "change": r["change"],
                             **{f"rank_{v}": r["variant_ranks"][v] for v in RS.VARIANTS}, **{f"fwd{h}": bars.forward(r["symbol"], day)[h] for h in HORIZONS}})
        (out / f"shadow_{day.replace('-', '')}.json").write_text(json.dumps(shadow, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"[ablation] {day} records={len(shadow['rows'])} re_eligible={shadow['summary']['re_eligible']} cohort_strong={shadow['summary']['cohort_strong']}", flush=True)
        previous_payload, previous_shadow = payload, shadow
    pd.DataFrame(rows_out).to_csv(out / "rows.csv", index=False, encoding="utf-8-sig")
    summary = aggregate(per_day)
    (out / "summary.json").write_text(json.dumps({"version": RS.VERSION, "days": days, "per_day": per_day, "aggregate": summary}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (out / "report.md").write_text(report(days, per_day, summary), encoding="utf-8")
    print(f"[ablation] written {out}")
    return out, summary


def aggregate(per_day):
    agg = {}
    for variant in (*RS.VARIANTS, "V5_portfolio"):
        for k in (TOP if variant != "V5_portfolio" else (None,)):
            key = f"{variant}/top{k}" if k else variant
            entries = [(d["variants"][variant][f"top{k}"] if k else d["variants"][variant]) for d in per_day]
            comp = {m: round(statistics.fmean([e["composition"][m] for e in entries if e["composition"].get(m) is not None]), 3)
                    for m in ("extended_share", "late_or_broken_share", "early_share", "tracking_share", "leading_theme_share", "mean_bias_atr", "mean_return20")
                    if any(e["composition"].get(m) is not None for e in entries)}
            fwd = {}
            for h in HORIZONS:
                vals = [e["forward"][h]["mean"] for e in entries if e["forward"][h]["n"]]
                ns = [e["forward"][h]["n"] for e in entries if e["forward"][h]["n"]]
                fwd[h] = {"days_with_data": len(vals), "names": sum(ns), "mean_of_day_means": round(statistics.fmean(vals), 3) if vals else None,
                          "days_positive": sum(v > 0 for v in vals)}
            overlap = [e.get("overlap_with_frozen") for e in entries if e.get("overlap_with_frozen") is not None]
            agg[key] = {"composition": comp, "forward": fwd, "mean_overlap_with_frozen": round(statistics.fmean(overlap), 2) if overlap else None}
    return agg


def report(days, per_day, agg):
    lines = [f"# 筛选顺序同样本消融（影子）· {days[0]} 至 {days[-1]}", "",
             f"规则版本 {RS.VERSION}；样本 {len(days)} 个已归档收盘日（同一候选记录、同一特征）；前瞻收益只在行情已存在处统计，是诊断不是校准目标。",
             "顺序：V0 冻结参与榜 → V1 续踪股凭当日证据重获资格 → V2 加位置与当日触发 → V3 加阶段/路径/主线 → V4 加多日证据变化 → V5 复核名单（≤2 只、同证券一席、题材不同、只取带内、不选过热/失效/修复中/R3-only；不是账户分配）。", "",
             "| 变体 | 偏离过大占比 | 过热/失效占比 | 早期占比 | 主线题材占比 | 平均偏离ATR | 平均20日涨幅 | 与冻结榜重合 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, a in agg.items():
        if not key.endswith("/top5") and key != "V5_portfolio":
            continue
        c = a["composition"]
        lines.append(f"| {key} | {c.get('extended_share')} | {c.get('late_or_broken_share')} | {c.get('early_share')} | {c.get('leading_theme_share')} | {c.get('mean_bias_atr')} | {c.get('mean_return20')} | {a.get('mean_overlap_with_frozen')} |")
    lines += ["", "## 前瞻收益（Top5 日均值的均值；括号内为有数据的日数/名数）", "", "| 变体 | +1日 | +3日 | +5日 | +10日 |", "|---|---:|---:|---:|---:|"]
    for key, a in agg.items():
        if not key.endswith("/top5") and key != "V5_portfolio":
            continue
        cells = [f"{a['forward'][h]['mean_of_day_means']}（{a['forward'][h]['days_with_data']}/{a['forward'][h]['names']}）" for h in HORIZONS]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines += ["", "## 逐日 V5 选择", ""]
    for d in per_day:
        chosen = d["variants"]["V5_portfolio"]["chosen"]
        lines.append(f"- {d['as_of']}（市场 {d['market']}，候选群体较强题材 {'、'.join(d['themes_leading']) or '无'}，续踪重获资格 {d['re_eligible']} 只）："
                     + ("；".join(f"{c['name']}〔{c['theme']}〕{c['reading']}" for c in chosen) or "无满足条件的候选"))
    lines += ["", "## 读法与边界", "",
              "- 样本天数少且集中在同一市场环境，任何前瞻差异都不能推广；扩池（V1）与排序（V2–V4）的贡献分开看。",
              "- 主线读法来自候选群体自身（COHORT_ONLY），不是全市场题材广度；样本不足的题材标 UNKNOWN。",
              "- 冻结规则与三本账不受影响；本报告不产生买卖许可。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", nargs="*", default=None)
    parser.add_argument("--limit-days", type=int, default=None)
    args = parser.parse_args()
    run(args.days, limit_days=args.limit_days)


if __name__ == "__main__":
    main()
