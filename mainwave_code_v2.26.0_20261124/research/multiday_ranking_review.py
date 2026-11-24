"""Shadow pool ranking with refreshed evidence, bounded memory and hard risk gates."""
from __future__ import annotations

import argparse
import inspect
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import recent_ranking_review as base

ROOT, ENGINE = base.ROOT, base.ENGINE
VERSION = "multiday_pool_v1_20260916"
CONFIG = {
    "version": VERSION, "start": "2026-09-09", "freeze_date": "2026-09-16",
    "end": "2026-09-30", "snapshot_version": base.CONFIG["snapshot_version"],
    "history_sessions": 3, "stale_expiry_sessions": 5,
    "extended_atr": 2., "low_position": .35, "active_ratio": 1.5,
    "weak_close": .3, "horizons": [1, 3, 5], "top_k": [3, 5, 15],
    "points": {"retained": 2, "supply": 2, "demand": 2, "strong_close": 1,
               "low_active": 1, "confirmed": 2, "persistent": 1,
               "retest_confirmed": 2, "retention_lost": -2,
               "weak_active": -3, "extended": -3, "new_low": -2},
    "mode": "SHADOW_ONLY_UNCALIBRATED_NO_ORDERS",
}
REASONS = {
    "retained": "冲击保留", "supply": "供给收缩", "demand": "需求证据",
    "strong_close": "强收盘", "low_active": "低位且相对活跃", "confirmed": "当前形态确认",
    "persistent": "同周期连续证据", "retest_confirmed": "同一回踩形态由形成到确认",
    "retention_lost": "冲击保留丢失", "weak_active": "活跃但弱收盘",
    "extended": "偏离结构过大", "new_low": "创近20日新低",
}
ROW_FIELDS = ["symbol", "name", "tracking_only", "value_rank", "value_tier", "structural_advice",
              "route_ids", "route_bias_atr", "price_date", "close", "cycle_id", "setup_status",
              "stock_lifecycle_label", "belonging_theme", "initial_review", "next_condition"]


def snapshot(day, output, prices, warnings):
    folder = ENGINE / day.replace("-", "")
    sources = [folder / "analysis_universe.json", folder / "analysis_evidence.json"]
    path = output / "frozen" / f"{day}.json"
    hashes = {p.name: base.digest(p.read_bytes()) if p.exists() else None for p in sources}
    if path.exists():
        saved = base.read_json(path)
        if saved["source_hashes"] != hashes:
            warnings.append(f"{day}: source changed; kept frozen input")
        return saved
    saved = dict(date=day, source_hashes=hashes, rows=[], available=False,
                 ingested_at=datetime.now(timezone.utc).isoformat(),
                 provenance="SAVED_ASOF_RECONSTRUCTION_NOT_LIVE_PROOF")
    if all(p.exists() for p in sources):
        universe, evidence = (base.read_json(p) for p in sources)
        if (universe.get("as_of") == evidence.get("as_of") == day
                and universe.get("version") == evidence.get("version") == CONFIG["snapshot_version"]):
            saved["available"] = True
            for r in universe["records"]:
                symbol = r["symbol"]
                core = evidence.get("records", {}).get(symbol, {})
                state, route = core.get("state_vector", {}), core.get("route_evidence", {})
                window = core.get("review_window", {})
                if symbol not in prices:
                    price_file = ROOT / "data" / f"{symbol}_daily.csv"
                    prices[symbol] = pd.read_csv(price_file, dtype={"date": str}) if price_file.exists() else None
                features = base.features(prices[symbol], day) if prices[symbol] is not None else None
                row = {k: r.get(k) for k in ROW_FIELDS}
                row.update(features or {})
                row.update(date=day, current_core=core.get("as_of") == day and bool(state) and core.get("version") == CONFIG["snapshot_version"],
                           price_features=features is not None, stage=state.get("stage"),
                           setup_id=state.get("setup_id"), setup_kind=state.get("setup_kind"),
                           retained=route.get("retained"), supply=route.get("supply"), demand=route.get("demand"),
                           in_band=window.get("price_in_band") is True,
                           route_ready=core.get("entry_contract", {}).get("requirements_met") is True,
                           supply_pressure="SUPPLY_PRESSURE" in route.get("risks", []),
                           core_version=core.get("version"),
                           market=universe.get("market", {}).get("structure"),
                           market_allowed=universe.get("market", {}).get("trend_review_allowed") is True)
                saved["rows"].append(row)
        else:
            saved["issue"] = "ARCHIVE_DATE_OR_VERSION_MISMATCH"
    else:
        saved["issue"] = "ARCHIVE_MISSING"
    saved["rows"] = base.records(pd.DataFrame(saved["rows"]))
    base.write_json(path, saved)
    return saved


def evaluate(row, history):
    """Recompute points from bounded evidence, never accumulate yesterday's total."""
    points = {}
    def add(key, count=1):
        points[key] = CONFIG["points"][key] * count
    active = (row.get("turnover_rel") or 0) >= CONFIG["active_ratio"] and (row.get("amount_rel") or 0) >= CONFIG["active_ratio"]
    for key in ["retained", "supply", "demand", "strong_close"]:
        if row.get(key) is True:
            add(key)
    if active and row.get("position60") is not None and row["position60"] <= CONFIG["low_position"]:
        add("low_active")
    if row.get("setup_status") == "CONFIRMED":
        add("confirmed")
    if active and row.get("close_location") is not None and row["close_location"] <= CONFIG["weak_close"]:
        add("weak_active")
    if row.get("structural_advice") == "WAIT_PRICE" or (row.get("route_bias_atr") or 0) > CONFIG["extended_atr"]:
        add("extended")
    if row.get("new_low20") is True:
        add("new_low")
    consecutive = []
    expected = row["session"] - 1
    for previous in reversed(history):
        if (previous["session"] != expected or previous["pool_status"] != "ACTIVE"
                or not row.get("cycle_id") or previous.get("cycle_id") != row["cycle_id"]
                or previous.get("core_version") != row.get("core_version")):
            break
        consecutive.append(previous)
        expected -= 1
        if len(consecutive) == CONFIG["history_sessions"] - 1:
            break
    if row.get("retained") and (row.get("supply") or row.get("demand")):
        count = 0
        for previous in consecutive:
            if not (previous.get("retained") and (previous.get("supply") or previous.get("demand"))):
                break
            count += 1
        if count:
            add("persistent", count)
    if consecutive:
        previous = consecutive[0]
        if previous.get("retained") is True and row.get("retained") is False:
            add("retention_lost")
        if (row.get("setup_id") and row["setup_id"] == previous.get("setup_id")
                and row.get("setup_kind") in {"FIRST_PULLBACK", "LATER_PULLBACK"}
                and previous.get("setup_status") == "FORMING" and row.get("setup_status") == "CONFIRMED"
                and previous.get("supply") and row.get("demand")):
            add("retest_confirmed")
    return dict(score=sum(points.values()), components=points,
                score_delta=sum(points.values()) - consecutive[0]["score"] if consecutive else None,
                comparison_date=consecutive[0]["date"] if consecutive else None,
                history_sessions=len(consecutive),
                history_reset=bool(history and not consecutive),
                score_reasons="; ".join(f"{REASONS[k]} {v:+d}" for k, v in points.items()) or "无明确加减分证据")


def order(rows, multiday):
    def key(r):
        prefix = (base.ORDER.get(r.get("value_tier"), 99), base.ORDER.get(r.get("structural_advice"), 99))
        bias = r.get("route_bias_atr")
        suffix = (abs(bias) if bias is not None else float("inf"), r["symbol"])
        return prefix + ((-r["score"],) if multiday else ()) + suffix
    return sorted(rows, key=key)


def advance_pool(day, session, rows, pool, histories):
    current = {r["symbol"]: r for r in rows}
    for r in rows:
        if not r.get("tracking_only"):
            item = pool.setdefault(r["symbol"], {"first_seen": day, "admission_routes": r.get("route_ids"),
                                                "last_fresh_session": session, "latest": r})
            item["last_discovery"] = day
    output = []
    for symbol, item in pool.items():
        now = current.get(symbol)
        r = dict(now or item["latest"], date=day, session=session,
                 first_seen=item["first_seen"], last_discovery=item["last_discovery"],
                 admission_routes=item["admission_routes"],
                 today_candidate=bool(now is not None and not now.get("tracking_only")))
        fresh = bool(now and now.get("price_date") == day and now.get("current_core") and now.get("price_features"))
        if fresh:
            item["last_fresh_session"], item["latest"] = session, now
        r["stale_sessions"] = session - item["last_fresh_session"]
        if not fresh:
            status = "EXPIRED" if r["stale_sessions"] >= CONFIG["stale_expiry_sessions"] else "STALE"
        elif r.get("stage") in {"FAILED", "REBUILDING"}:
            status = "INVALID"
        elif (r.get("value_tier") in {"RISK_WATCH", "DATA_WAIT", "HISTORICAL_ONLY"}
              or r.get("structural_advice") in {"RISK_WATCH", "DATA_WAIT"}
              or r.get("stage") in {"OVERHEATED", "DISTRIBUTION_RISK"} or r.get("supply_pressure")):
            status = "BLOCKED"
        else:
            status = "ACTIVE"
        r["pool_status"] = status
        history = histories.get(symbol, [])
        previous = history[-1] if history else None
        r["state_change"] = (f"{previous['pool_status']} -> {status}; {previous.get('stage')} -> {r.get('stage')}"
                             if previous else f"NEW -> {status}; {r.get('stage')}")
        r["cycle_changed"] = bool(previous and previous.get("cycle_id") != r.get("cycle_id"))
        if status == "ACTIVE":
            r.update(evaluate(r, history))
        else:
            r.update(score=None, score_delta=None, components={}, history_sessions=0, history_reset=True,
                     comparison_date=None, score_reasons="当前数据/风险门槛拦截，旧加分不得抵消")
        r["structure_review"] = bool(status == "ACTIVE" and r.get("structural_advice") == "PRIORITY_REVIEW"
                                      and r.get("in_band") and r.get("route_ready") and "extended" not in r["components"])
        r["formal_initial_review"] = bool(r["structure_review"] and r.get("market_allowed") and r.get("initial_review"))
        r["pool_rank"], r["rank_change"] = None, None
        output.append(r)
    for rank, r in enumerate(order([r for r in output if r["pool_status"] == "ACTIVE"], True), 1):
        r["pool_rank"] = rank
        previous = histories.get(r["symbol"], [])
        if previous and r["comparison_date"] == previous[-1]["date"] and previous[-1].get("pool_rank"):
            r["rank_change"] = previous[-1]["pool_rank"] - rank
    for r in output:
        histories.setdefault(r["symbol"], []).append(r)
    return output


def run(end=None, output=None):
    output = output or ENGINE / VERSION
    functions = [snapshot, evaluate, order, advance_pool, base.features, base.outcomes]
    protocol = {"config": CONFIG, "base_config": base.CONFIG, "order": base.ORDER,
                "functions": {f.__name__: base.digest(inspect.getsource(f).encode()) for f in functions}}
    path = output / "protocol.json"
    if path.exists() and base.read_json(path) != protocol:
        raise ValueError("Frozen multiday rules changed; use a new experiment directory")
    if not path.exists():
        base.write_json(path, protocol)
    calendar = sorted(pd.read_csv(ROOT / "data/000001_SH_index.csv", dtype={"trade_date": str}).trade_date.unique())
    end = end or max(d for d in calendar if d <= CONFIG["end"])
    if end not in calendar or not CONFIG["start"] <= end <= CONFIG["end"]:
        raise ValueError("Cutoff must be an available session within the research period")
    prices, pool, histories, observations, warnings, manifests = {}, {}, {}, [], [], []
    for session, day in enumerate(calendar):
        if not CONFIG["start"] <= day <= end:
            continue
        saved = snapshot(day, output, prices, warnings)
        if not saved["available"]:
            warnings.append(f"{day}: {saved.get('issue')}; no state confirmation invented")
        manifests.append({k: v for k, v in saved.items() if k != "rows"})
        observations.extend(advance_pool(day, session, saved["rows"], pool, histories))
        print(f"[multiday] {day}: pool={len(pool)} updated={len(saved['rows'])}", flush=True)
    for r in observations:
        r["sample_period"] = "INITIAL_EXPLORATORY" if r["date"] < CONFIG["freeze_date"] else "FORWARD_FROZEN_RULES"
        if r["pool_status"] != "ACTIVE":
            continue
        if r["symbol"] not in prices:
            prices[r["symbol"]] = pd.read_csv(ROOT / "data" / f"{r['symbol']}_daily.csv", dtype={"date": str})
        for horizon in CONFIG["horizons"]:
            r.update(base.outcomes(prices[r["symbol"]], r["date"], end, horizon, calendar))
    daily, picks = [], []
    for day in sorted({r["date"] for r in observations}):
        active = [r for r in observations if r["date"] == day and r["pool_status"] == "ACTIVE"]
        policies = {"daily_original": order([r for r in active if r["today_candidate"]], False),
                    "pool_original": order(active, False), "pool_multiday": order(active, True)}
        for policy, ordered in policies.items():
            for k in CONFIG["top_k"]:
                top = ordered[:k]
                picks.extend(dict(r, policy=policy, k=k, shadow_rank=i) for i, r in enumerate(top, 1))
                for h in CONFIG["horizons"]:
                    values = [r[f"ret{h}"] for r in top if f"ret{h}" in r]
                    daily.append(dict(date=day, policy=policy, k=k, horizon=h, selected=len(top), n=len(values),
                                      mean=sum(values) / len(values) if values else None,
                                      win_pct=sum(v > 1e-5 for v in values) / len(values) * 100 if values else None,
                                      complete=len(values) == len(top) and bool(top),
                                      sample_period="INITIAL_EXPLORATORY" if day < CONFIG["freeze_date"] else "FORWARD_FROZEN_RULES"))
    aggregate = []
    completed = pd.DataFrame([r for r in daily if r["complete"]])
    if not completed.empty:
        for (period, policy, k, h), group in completed.groupby(["sample_period", "policy", "k", "horizon"]):
            reference = completed[(completed.policy == "pool_original") & completed.k.eq(k) & completed.horizon.eq(h)]
            paired = group.merge(reference[["date", "mean"]], on="date", suffixes=("", "_baseline"))
            delta = paired["mean"] - paired.mean_baseline
            aggregate.append(dict(sample_period=period, policy=policy, k=int(k), horizon=int(h), days=len(paired),
                                  n=int(paired.n.sum()), mean=float(paired["mean"].mean()), win_pct=float(paired.win_pct.mean()),
                                  delta_vs_pool=float(delta.mean()), improved_days=int(delta.gt(1e-8).sum()),
                                  worsened_days=int(delta.lt(-1e-8).sum())))
    latest = [r for r in observations if r["date"] == end]
    result = dict(version=VERSION, as_of=end, config=CONFIG, warnings=warnings, manifests=manifests,
                  pool_counts=dict(Counter(r["pool_status"] for r in latest)),
                  active_carried=sum(r["pool_status"] == "ACTIVE" and not r["today_candidate"] for r in latest),
                  structure_reviews=sum(r["structure_review"] for r in latest),
                  formal_initial_reviews=sum(r["formal_initial_review"] for r in latest),
                  ranking_results=aggregate, daily_rankings=daily)
    destination = output / "reviews" / end.replace("-", "")
    if (destination / "observations.json").exists():
        if base.read_json(destination / "observations.json") != observations:
            raise ValueError("Existing review changed; do not overwrite frozen results")
        print(f"UNCHANGED: {destination}")
        for warning in warnings:
            print(warning)
        return base.read_json(destination / "summary.json")
    base.write_json(destination / "summary.json", result)
    base.write_json(destination / "observations.json", observations)
    base.write_json(destination / "ranked_picks.json", picks)
    report(destination, result, latest)
    print(f"[multiday] {result['pool_counts']} carried={result['active_carried']} -> {destination}")
    return result


def report(destination, result, latest):
    def show(value):
        return "--" if value is None else str(value)
    lines = [f"# 多日池内排序影子复盘 {result['as_of']}", "",
             "- 独立新实验；不更改已冻结的单日研究、生产榜单或交易权限。分数不是盈利概率。",
             "- 从09-09开始以新入选标的建池；旧股只要有当日有效分析即可竞争，不要求重新入选。",
             "- 同周期最近3个连续交易日提供有限加减分；不累加历史总分，不按出现次数加分。",
             "- 无当日确认立即退出有效排名，连续5个交易日未刷新标记过期；结构失效和风险不能被旧分抵消。",
             "- 原建议类别及结构类别仍优先；同类按多日分，再按偏离排序。不把跨阶段强势当成好买点。",
             f"- 当前池：{result['pool_counts']}；有效续踪股 {result['active_carried']}。",
             f"- 结构位置可复核 {result['structure_reviews']}；同时满足原正式首仓复核许可 {result['formal_initial_reviews']}。",
             "- 数据均为已归档的当日重建，不证明原始实盘命中；初评与冻结规则后样本分开，未成熟不计零收益。",
             *[f"- 缺口：{w}" for w in result['warnings']], "", "## Top5比较", "",
             "| 样本期 | 方法 | 观察期 | 完整日期 | 记录 | 上涨比例% | 日均涨跌% | 较同池原排序百分点 | 改善/恶化日 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in result["ranking_results"]:
        if r["k"] == 5:
            lines.append(f"| {r['sample_period']} | {r['policy']} | {r['horizon']} | {r['days']} | {r['n']} | {r['win_pct']:.2f} | {r['mean']:+.2f} | {r['delta_vs_pool']:+.2f} | {r['improved_days']}/{r['worsened_days']} |")
    lines += ["", "## 当前Top15（研究复核顺序，不是买入清单）", "",
              "| 排名 | 股票 | 来源 | 当前路径 | 当前生命周期 | 建议类别 | 分数 | 分数变化 | 名次上升 | 加减分依据 |",
              "|---:|---|---|---|---|---|---:|---:|---:|---|"]
    for r in order([r for r in latest if r["pool_status"] == "ACTIVE"], True)[:15]:
        lines.append(f"| {r['pool_rank']} | {r['name']} {r['symbol']} | {'当日入选' if r['today_candidate'] else '池内续踪'} | {(r.get('route_ids') or '待确认').replace('|', '+')} | {r['stock_lifecycle_label']} | {r['structural_advice']} | {r['score']} | {show(r['score_delta'])} | {show(r['rank_change'])} | {r['score_reasons']} |")
    lines += ["", "完整池、退池原因、每日状态及分数明细见 observations.json。名次变化还受池内其他股票变化影响；分数变化只与连续同周期的上一交易日比较。", ""]
    (destination / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run(args.as_of, args.output)
