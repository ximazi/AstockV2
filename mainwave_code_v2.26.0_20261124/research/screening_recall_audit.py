"""Recall audit of the discovery layer against labelled TREND episodes.

Target: results/mainwave_episodes_2026.csv (ex-post label: +25% within 40 sessions without a -15% path,
liquid; class TREND = neither lottery ladder nor an extended run). Detectors: the replayable route
evidence (R2 preparation, R1 attempt/acceptance) and the stock lifecycle stages, rebuilt from bars only —
both are prefix-stable, so a reading at date d equals what the engine could have known at d.

For every episode start t the audit asks: was the stock eligible; which detector fired first inside
[t-10, t+5] sessions; how extended the price already was when it fired. For every detector firing in the
period it also asks whether an episode followed within 10 sessions (precision). Misses and late hits are
the recall gaps ranking weights cannot fix."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from mainwave_engine import route_evidence as RE, stock_lifecycle as SL

ROOT = Path(".")
DATA = ROOT / "data"
OUT = ROOT / "results/research/screening_recall_audit"
BEFORE, AFTER = 10, 5
PRECISION_WINDOW = 10
LEAD_BUCKETS = (("EARLY", -99, -1), ("ON_TIME", 0, 2), ("LATE", 3, 5))
DETECTORS = ("R2_prepared", "R1_attempt", "R1_accepted", "R1_accepted_demand", "LC_ready", "LC_ignition")


def bucket(lead):
    for name, lo, hi in LEAD_BUCKETS:
        if lo <= lead <= hi:
            return name
    return "MISS"


def detectors(frame, timeline):
    """Per-session boolean firings for each detector, aligned on the route frame's dates."""
    dates = frame.date.dt.strftime("%Y-%m-%d")
    stage = {}
    if timeline is not None and not timeline.empty:
        stage = dict(zip(pd.to_datetime(timeline.date).dt.strftime("%Y-%m-%d"), timeline.cycle_stage))
    fired = {"R2_prepared": frame.prepared.fillna(False).astype(bool).tolist(),
             "R1_attempt": frame.breakout_event.eq("ATTEMPT").tolist(),
             "R1_accepted": frame.breakout_event.eq("ACCEPTED").tolist(),
             "R1_accepted_demand": frame.accepted_demand.fillna(False).astype(bool).tolist(),
             "LC_ready": [stage.get(d) in ("ARMED", "READY") for d in dates],
             "LC_ignition": [stage.get(d) == "IGNITION" for d in dates]}
    return dates.tolist(), fired


def audit_symbol(symbol, starts, span):
    path = DATA / f"{symbol}_daily.csv"
    if not path.exists():
        return [{"symbol": symbol, "start": s, "status": "NO_BARS"} for s in starts], []
    raw = pd.read_csv(path, encoding="utf-8-sig")
    frame = RE.build(raw, symbol)
    if frame.empty:
        return [{"symbol": symbol, "start": s, "status": "SHORT_HISTORY"} for s in starts], []
    try:
        _, timeline = SL.analyze(symbol, str(frame.date.iloc[-1].date()), DATA)
    except Exception:  # lifecycle is optional evidence; its absence is recorded, not fatal
        timeline = None
    dates, fired = detectors(frame, timeline)
    index = {d: i for i, d in enumerate(dates)}
    eligible = frame.eligible.fillna(False).astype(bool).tolist()
    bias = frame.bias_atr.tolist()
    rows = []
    for start in starts:
        i = index.get(start)
        if i is None:
            rows.append({"symbol": symbol, "start": start, "status": "START_NOT_IN_BARS"})
            continue
        lo, hi = max(0, i - BEFORE), min(len(dates) - 1, i + AFTER)
        row = {"symbol": symbol, "start": start, "status": "ELIGIBLE" if eligible[i] else "INELIGIBLE_AT_START",
               "stage_eve": (timeline.cycle_stage.iloc[index[dates[i - 1]]] if timeline is not None and not timeline.empty and i >= 1 and dates[i - 1] in index and index[dates[i - 1]] < len(timeline) else None)}
        first_any = None
        for det in DETECTORS:
            hit = next((j for j in range(lo, hi + 1) if fired[det][j]), None)
            lead = (hit - i) if hit is not None else None
            row[f"{det}_lead"] = lead
            row[f"{det}_bucket"] = bucket(lead) if lead is not None else "MISS"
            row[f"{det}_bias"] = round(float(bias[hit]), 2) if hit is not None and pd.notna(bias[hit]) else None
            if hit is not None and (first_any is None or hit < first_any[1]):
                first_any = (det, hit)
        row["first_detector"] = first_any[0] if first_any else None
        row["first_lead"] = (first_any[1] - i) if first_any else None
        row["first_bucket"] = bucket(first_any[1] - i) if first_any else "MISS"
        row["first_bias"] = round(float(bias[first_any[1]]), 2) if first_any and pd.notna(bias[first_any[1]]) else None
        rows.append(row)
    # precision: every firing inside the audit span → did an episode start within the next 10 sessions?
    starts_idx = sorted(index[s] for s in starts if s in index)
    firings = []
    lo_span, hi_span = span
    for det in ("R2_prepared", "R1_accepted", "R1_accepted_demand", "LC_ready", "LC_ignition"):
        prev = False
        for j, on in enumerate(fired[det]):
            new = on and not prev
            prev = on
            if not new or not (lo_span <= dates[j] <= hi_span) or not eligible[j]:
                continue
            followed = any(j <= s <= j + PRECISION_WINDOW for s in starts_idx)
            firings.append({"symbol": symbol, "detector": det, "date": dates[j], "episode_within_10": followed,
                            "bias_atr": round(float(bias[j]), 2) if pd.notna(bias[j]) else None})
    return rows, firings


def run(episodes_path="results/mainwave_episodes_2026.csv", cls="TREND", limit=None, min_gain=None, out_root=OUT):
    episodes = pd.read_csv(episodes_path)
    episodes = episodes[episodes.cls == cls]
    if min_gain is not None:
        episodes = episodes[episodes.max_gain >= min_gain]
    if limit:
        episodes = episodes.sample(n=min(limit, len(episodes)), random_state=20260923)
    span = (str(episodes.start.min()), str(episodes.start.max()))
    by_symbol = defaultdict(list)
    for r in episodes.itertuples():
        by_symbol[r.stem].append(str(r.start))
    rows, firings = [], []
    started = datetime.now()
    for n, (symbol, starts) in enumerate(sorted(by_symbol.items()), 1):
        r, f = audit_symbol(symbol, sorted(starts), span)
        rows.extend(r)
        firings.extend(f)
        if n % 200 == 0 or n == len(by_symbol):
            print(f"[recall] {n}/{len(by_symbol)} symbols, {len(rows)} episodes, {(datetime.now() - started).seconds}s", flush=True)
    frame = pd.DataFrame(rows)
    meta = episodes[["stem", "start", "name", "board", "industry", "max_gain", "min_path", "ret60_prior", "bias60"]].rename(columns={"stem": "symbol"})
    meta["start"] = meta.start.astype(str)
    frame = frame.merge(meta, on=["symbol", "start"], how="left")
    fires = pd.DataFrame(firings)
    out = Path(out_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "episodes.csv", index=False, encoding="utf-8-sig")
    fires.to_csv(out / "firings.csv", index=False, encoding="utf-8-sig")
    summary = summarize(frame, fires, span, cls)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (out / "report.md").write_text(report(summary), encoding="utf-8")
    print(f"[recall] written {out}")
    return out, summary


def summarize(frame, fires, span, cls):
    audited = frame[frame.status.isin(["ELIGIBLE", "INELIGIBLE_AT_START"])]
    eligible = audited[audited.status == "ELIGIBLE"]
    def buckets(col):
        return dict(Counter(eligible[col].fillna("MISS")))
    per_detector = {det: {"buckets": buckets(f"{det}_bucket"),
                          "hit_rate": round(float(eligible[f"{det}_bucket"].ne("MISS").mean()), 3) if len(eligible) else None,
                          "median_lead": (float(eligible[f"{det}_lead"].median()) if eligible[f"{det}_lead"].notna().any() else None),
                          "median_bias_at_hit": (float(eligible[f"{det}_bias"].median()) if eligible[f"{det}_bias"].notna().any() else None)}
                    for det in DETECTORS}
    any_hit = eligible.first_bucket.ne("MISS")
    extended_at_hit = eligible.first_bias.gt(2.0)
    month = eligible.start.str[:7]
    by_month = {m: {"episodes": int(len(g)), "any_hit": round(float(g.first_bucket.ne("MISS").mean()), 3),
                    "early_or_on_time": round(float(g.first_bucket.isin(["EARLY", "ON_TIME"]).mean()), 3)} for m, g in eligible.groupby(month)}
    by_board = {b: {"episodes": int(len(g)), "any_hit": round(float(g.first_bucket.ne("MISS").mean()), 3)} for b, g in eligible.groupby("board")}
    precision = {}
    if len(fires):
        for det, g in fires.groupby("detector"):
            precision[det] = {"firings": int(len(g)), "episode_within_10": round(float(g.episode_within_10.mean()), 3),
                              "median_bias": float(g.bias_atr.median()) if g.bias_atr.notna().any() else None}
    return {"class": cls, "span": span, "episodes_total": int(len(frame)), "audited": int(len(audited)), "eligible": int(len(eligible)),
            "ineligible_at_start": int((audited.status == "INELIGIBLE_AT_START").sum()),
            "status_counts": dict(Counter(frame.status)), "any_hit_rate": round(float(any_hit.mean()), 3) if len(eligible) else None,
            "first_bucket": dict(Counter(eligible.first_bucket)), "first_detector": dict(Counter(eligible.first_detector.fillna("NONE"))),
            "extended_at_first_hit_share": round(float(extended_at_hit[any_hit].mean()), 3) if any_hit.any() else None,
            "stage_eve": dict(Counter(eligible.stage_eve.fillna("NONE"))),
            "per_detector": per_detector, "by_month": by_month, "by_board": by_board, "precision": precision,
            "definitions": {"window": f"[t-{BEFORE}, t+{AFTER}] sessions", "lead_buckets": LEAD_BUCKETS, "precision_window": PRECISION_WINDOW,
                            "extended": "bias_atr > 2 at first hit", "labels": "ex-post episode labels (mainwave_episode_label.py), not tradability"}}


def report(s):
    lines = [f"# 发现层召回审计 · {s['class']} episodes {s['span'][0]} 至 {s['span'][1]}", "",
             f"episodes {s['episodes_total']}，可审计 {s['audited']}，起点日符合资格（流动性/非ST/历史≥120）{s['eligible']}，起点日不符合资格 {s['ineligible_at_start']}。",
             f"任一检测器在 [t-{BEFORE}, t+{AFTER}] 内命中：{s['any_hit_rate']}；首次命中分布：{s['first_bucket']}；首次命中时已偏离>2ATR 的占比：{s['extended_at_first_hit_share']}。", "",
             "| 检测器 | 命中率 | 提前(≤t-1) | 准时(t..t+2) | 偏晚(t+3..t+5) | 漏检 | 命中时中位偏离ATR | 中位提前量 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for det, d in s["per_detector"].items():
        b = d["buckets"]
        lines.append(f"| {det} | {d['hit_rate']} | {b.get('EARLY', 0)} | {b.get('ON_TIME', 0)} | {b.get('LATE', 0)} | {b.get('MISS', 0)} | {d['median_bias_at_hit']} | {d['median_lead']} |")
    lines += ["", "## 误报（检测器新触发后 10 个交易日内是否出现 episode 起点）", "", "| 检测器 | 触发次数 | 随后有 episode 的比例 | 触发时中位偏离ATR |", "|---|---:|---:|---:|"]
    for det, p in s["precision"].items():
        lines.append(f"| {det} | {p['firings']} | {p['episode_within_10']} | {p['median_bias']} |")
    lines += ["", "## 分月", "", "| 月份 | episodes | 任一命中 | 提前或准时 |", "|---|---:|---:|---:|"]
    for m, d in s["by_month"].items():
        lines.append(f"| {m} | {d['episodes']} | {d['any_hit']} | {d['early_or_on_time']} |")
    lines += ["", "## 起点前一日的生命周期阶段", "", "```", json.dumps(s["stage_eve"], ensure_ascii=False), "```", "",
              "## 读法与边界", "",
              "- episode 标签是事后前向收益定义的靶子，不是可交易信号；漏检指的是发现层在窗口内没有任何触发，排序权重救不了。",
              "- 偏晚命中且偏离>2ATR 的，是“已经很强才被看到”的样本，正是目标漂移的来源。",
              "- 误报率用同期所有新触发计算，基准是同期 episode 起点的自然发生率；不含成本，不代表策略胜率。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", default="results/mainwave_episodes_2026.csv")
    parser.add_argument("--cls", default="TREND")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-gain", type=float, default=None)
    args = parser.parse_args()
    run(args.episodes, args.cls, args.limit, args.min_gain)


if __name__ == "__main__":
    main()
