"""Fixed 2026 cohort entry/exit attribution, from immutable case inputs."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import pandas as pd

from mainwave_engine import pullback_plan as PP, route_evidence as RE
from research import route_followup as RF

ROOT = Path("results/mainwave_engine/review_workbench")
OUTPUT = ROOT / "entry_exit_study_20260911"
ARMS = {"baseline": "原信号次日开盘+原分批", "opening_only": "只增加成交价格复核",
        "profit_only": "原入场+成熟后弱收盘保护", "pullback": "等待回踩+成交复核+分阶段保护",
        "pullback_entry_only": "同一回踩入场+原分批退出", "strong_reclaim": "回踩收复前日高点+放量确认"}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def trial(a, i, symbol, mode, calendar, guard=False):
    trace = []
    result = RF.simulate(a, i, symbol, mode, trace=trace, calendar=calendar, entry_guard=guard)
    if trace:
        result.setdefault("entry_date", a["day"][i + 1])
        result.setdefault("entry_price", float(a["open"][i + 1]))
    fills = []
    lookup = {d: j for j, d in enumerate(a["day"])}
    for row in trace:
        j = lookup[row["date"]]
        for fill in row["fills"]:
            fills.append({"date": row["date"], **fill,
                          "raw_price": fill["price_index"] * a["close"][j] / a["index_close"][j]})
    return {**result, "signal_date": a["day"][i], "fills": fills, "trace": trace}


def waiting_trial(a, origin, symbol, calendar, start=None, horizon=20, used_plans=()):
    cid, rejected = a["breakout_id"][origin], []
    first = origin + 1 if start is None else start
    end = min(first + horizon, len(a["day"]))
    status = "WINDOW_EXPIRED" if first + horizon <= len(a["day"]) else "WAIT_CENSORED"
    for j in range(first, end):
        if a["breakout_id"][j] != cid or a["breakout_status"][j] not in {"ACCEPTED", "ADVANCING"}:
            status = "INVALID_BEFORE_ENTRY"
            break
        if a["plan_status"][j] != "CONFIRMED" or a["plan_id"][j] in used_plans:
            continue
        result = trial(a, j, symbol, "phase36_20", calendar, True)
        if result["trace"] or result["status"] == "CENSORED":
            return {**result, "plan_id": a["plan_id"][j], "kind": a["plan_kind"][j],
                    "test_date": a["plan_test_at"][j], "rejected": rejected}
        rejected.append({"date": a["day"][j], "status": result["status"]})
    return {"status": status, "rejected": rejected, "fills": [], "trace": []}


def worker(job):
    symbol, sources, frozen, calendar, as_of, price_hash, output = job
    path = Path(frozen) / "data" / f"{symbol}_daily.csv"
    if hashlib.sha256(path.read_bytes()).hexdigest() != price_hash:
        raise ValueError(f"Frozen price mismatch: {symbol}")
    raw = pd.read_csv(path)
    f = RF.enrich(RE.build(raw[pd.to_datetime(raw.date).le(as_of)], symbol))
    strong = pd.concat([f, PP.annotate(f, strong_reclaim=True)], axis=1)
    f = pd.concat([f, PP.annotate(f)], axis=1)
    a = RF.arrays(f)
    strong_a = RF.arrays(strong)
    lookup, rows = {d: j for j, d in enumerate(a["day"])}, []
    for source in sources:
        day = source["date"]
        i = lookup[day]
        if not a["accepted_bias2"][i]:
            raise ValueError(f"Original signal changed: {symbol} {day}")
        arms = {"baseline": trial(a, i, symbol, "stage36_20", calendar),
                "opening_only": trial(a, i, symbol, "stage36_20", calendar, True),
                "profit_only": trial(a, i, symbol, "profit36_20", calendar),
                "pullback": waiting_trial(a, i, symbol, calendar)}
        pullback = arms["pullback"]
        arms["pullback_entry_only"] = (trial(a, lookup[pullback["signal_date"]], symbol, "stage36_20", calendar, True)
                                      if pullback["trace"] else dict(pullback))
        arms["strong_reclaim"] = waiting_trial(strong_a, i, symbol, calendar)
        baseline = arms["baseline"]
        if baseline["status"] != source["status"] or (baseline["status"] == "CLOSED" and
                abs(baseline["net"] - source["net"]) > 1e-8):
            raise ValueError(f"Fixed baseline drift: {symbol} {day}")
        # A distinct confirmation after exit can use preparation observed while holding.
        followups = []
        exited = baseline.get("exit_date")
        used_plans = set()
        for _ in range(3):
            if not exited:
                break
            candidate = waiting_trial(a, i, symbol, calendar, lookup[exited], 60, used_plans)
            followups.append(candidate)
            used_plans.add(candidate.get("plan_id", ""))
            exited = candidate.get("exit_date")
        study = {"version": PP.VERSION, "symbol": symbol, "original_signal": day, "as_of": as_of,
                 "rules": PP.RULES, "arms": arms, "followups_after_baseline": followups,
                 "labels": ARMS, "orders_enabled": False, "unseen_oos": False,
                 "notice": "固定研究规则，不按个案收益择优；对照基准为原分批版，不是列表事后最佳退出。跟踪交易独立，不累计为原交易收益。",
                 "strong_reclaim_rule": "在基础回踩条件上，收盘超过前一日最高价、收盘位置>=60%、成交额/前20日中位>=1；独立诊断，不选择个案最佳方案。",
                 "observations": json.loads(f.loc[i:min(i + 40, len(f) - 1),
                    ["date", "plan_status", "plan_kind", "plan_test_at", "plan_confirmed_at"]].to_json(
                        orient="records", date_format="iso"))}
        write(Path(output) / f"{symbol}_{day}.json", study)
        for arm, result in arms.items():
            rows.append({"symbol": symbol, "signal": day, "arm": arm, "common_mature": i + 40 < len(f),
                         "status": result["status"], "net": result.get("net"),
                         "filled": bool(result["trace"]), "entry_date": result.get("entry_date"),
                         "exit_date": result.get("exit_date"), "reason": result.get("reason"),
                         "rejected": len(result.get("rejected", []))})
    return rows


def run(output=OUTPUT, workers=4):
    if (output / "manifest.json").exists():
        raise ValueError("Study already completed; keep the frozen result and use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    frozen = ROOT / "catalog_source"
    manifest = json.loads((frozen / "exit_replay_manifest.json").read_text(encoding="utf-8"))
    sources = pd.read_csv(frozen / "r1_exit_replay.csv.gz")
    sources = sources[sources.year.eq(2026) & sources.exit.eq("stage36_20")]
    hashes = {x["symbol"]: x["price_hash"] for x in manifest["inputs"]}
    jobs = [(symbol, json.loads(g.to_json(orient="records", double_precision=15)), str(frozen),
             tuple(manifest["frozen_calendar"]), manifest["as_of"], hashes[symbol], str(output))
            for symbol, g in sources.groupby("symbol")]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, values in enumerate(pool.map(worker, jobs, chunksize=2), 1):
            rows.extend(values)
            if n % 100 == 0 or n == len(jobs):
                print(f"[entry_exit] {n}/{len(jobs)} stocks; {len(rows)//len(ARMS)} signals", flush=True)
    f = pd.DataFrame(rows)
    f.to_csv(output / "cohort.csv", index=False)
    summary = []
    for mature, frame in ((False, f), (True, f[f.common_mature])):
        for arm, group in frame.groupby("arm", sort=False):
            closed = group[group.status.eq("CLOSED")]
            summary.append({"scope": "common_40_sessions" if mature else "all", "arm": arm, "label": ARMS[arm],
                            "signals": len(group), "filled": int(group.filled.sum()), "closed": len(closed),
                            "median_net": float(closed.net.median()) if len(closed) else None,
                            "mean_net": float(closed.net.mean()) if len(closed) else None,
                            "win_pct": float(closed.net.gt(0).mean()*100) if len(closed) else None,
                            "statuses": dict(Counter(group.status))})
    paired = f[f.common_mature & f.status.eq("CLOSED")].pivot(index=["symbol", "signal"], columns="arm", values="net")
    pairs = {}
    for arm in ARMS:
        if arm == "baseline":
            continue
        part = paired[["baseline", arm]].dropna()
        delta = part[arm] - part.baseline
        pairs[arm] = {"n": len(part), "median_delta": float(delta.median()) if len(part) else None,
                      "mean_delta": float(delta.mean()) if len(part) else None,
                      "better": int(delta.gt(1e-8).sum()), "worse": int(delta.lt(-1e-8).sum())}
    report = {"summary": summary, "paired_closed_only": pairs,
              "limits": ["2026已用于诊断，不是未见OOS", "独立信号可能重叠，不是资金组合收益",
                         "仅已平仓样本计算价差；未成交、未结束和数据缺口保留在分母",
                         "回踩改变可交易子样本，必须同时查看覆盖率和相同信号配对差",
                         "统一2ATR等阈值待按波动和板块分层验证；本轮没有搜索最优参数",
                         "强恢复及退出分解是看到v1失败后追加的诊断，不是独立验证；必须另做未见数据验证"]}
    write(output / "summary.json", report)
    write(output / "manifest.json", {"version": PP.VERSION, "as_of": manifest["as_of"], "signals": len(sources),
          "symbols": len(jobs), "arms": ARMS, "rules": PP.RULES,
          "catalog_hash": json.loads((ROOT / "catalog.json").read_text(encoding="utf-8"))["catalog_hash"],
          "source_hash": hashlib.sha256((frozen / "r1_exit_replay.csv.gz").read_bytes()).hexdigest(),
          "code_hashes": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                          (__file__, PP.__file__, RF.__file__, RE.__file__)}})
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.output, args.workers)
