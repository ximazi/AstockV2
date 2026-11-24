"""Frozen, read-only shadow ranking study over saved engine selections."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "results/mainwave_engine"
VERSION = "ranking_shadow_v1_20260916"
CONFIG = {
    "version": VERSION, "snapshot_version": "lifecycle_analysis_core_v7_20260911",
    "freeze_date": "2026-09-16", "observation_end": "2026-09-30",
    "snapshot_start": "2026-09-09",
    "horizons": [1, 3, 5], "top_k": [3, 5, 15],
    "position60_low": .35, "position60_high": .75,
    "activity_high": 1.5, "close_strong": .7, "close_weak": .3,
    "pullback_down_up": .7,
    "policies": ["original", "low_position", "high_turnover", "quality_within_stage", "quality_cross_stage"],
    "mode": "SHADOW_ONLY_NOT_EXECUTION_OR_PROVEN_ALPHA",
}
ORDER = {name: i for i, name in enumerate((
    "PRIORITY_REVIEW", "WAIT_CONFIRMATION", "WAIT_PRICE", "EARLY_WATCH", "TREND_WATCH",
    "MARKET_WAIT", "WATCH", "DATA_WAIT", "RISK_WATCH", "HISTORICAL_ONLY"))}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def records(frame):
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def features(raw, day):
    """All baselines exclude the signal day; no future bars enter features."""
    f = raw[raw.date.le(day)].sort_values("date").tail(140).copy()
    if len(f) < 61 or f.iloc[-1].date != day or f.date.duplicated().any():
        return None
    for name in ("close", "open", "high", "low", "pct_chg", "turnover", "amount", "volume"):
        f[name] = pd.to_numeric(f[name], errors="coerce")
    if f[["close", "open", "high", "low", "pct_chg"]].isna().any().any() or (f.close <= 0).any():
        return None
    index = (1 + f.pct_chg / 100).cumprod()
    adj = index / index.iloc[-1] * f.close.iloc[-1]
    scale = adj / f.close
    op, hi, lo = f.open * scale, f.high * scale, f.low * scale
    prev = adj.shift()
    tr = pd.concat([hi - lo, (hi - prev).abs(), (lo - prev).abs()], axis=1).max(axis=1)
    atr = float(tr.iloc[-21:-1].mean())
    spread = float(hi.iloc[-1] - lo.iloc[-1])
    cloc = float((adj.iloc[-1] - lo.iloc[-1]) / spread) if spread > 0 else np.nan
    rng = float(hi.tail(60).max() - lo.tail(60).min())
    def relative(column):
        values = f[column].where(f[column] > 0)
        baseline = values.iloc[-21:-1].median()
        return float(values.iloc[-1] / baseline) if pd.notna(baseline) and baseline > 0 else np.nan
    turn, amount = relative("turnover"), relative("amount")
    float_proxy = f.volume / f.turnover.replace(0, np.nan)
    denominator_changed = bool(abs(float_proxy.iloc[-1] / float_proxy.iloc[-2] - 1) > .1)
    if denominator_changed:
        turn = np.nan
    recent = f.tail(10)
    up, down = recent.loc[recent.pct_chg > 0, "turnover"], recent.loc[recent.pct_chg < 0, "turnover"]
    down_up = float(down.median() / up.median()) if len(up) and len(down) and up.median() > 0 else np.nan
    return {
        "position60": float((adj.iloc[-1] - lo.tail(60).min()) / rng) if rng > 0 else np.nan,
        "turnover_rel": turn, "amount_rel": amount, "volume_rel": relative("volume"),
        "turnover_pct": float(f.turnover.iloc[-1]), "close_location": cloc,
        "body_atr": float((adj.iloc[-1] - op.iloc[-1]) / atr) if atr > 0 else np.nan,
        "r20": float((adj.iloc[-1] / adj.iloc[-21] - 1) * 100),
        "down_up_turnover": down_up, "denominator_changed": denominator_changed,
        "strong_close": bool(spread > 0 and cloc >= CONFIG["close_strong"] and adj.iloc[-1] >= op.iloc[-1]),
        "new_low20": bool(lo.iloc[-1] < lo.iloc[-21:-1].min()),
        "next_bar_used": False,
        "raw_prefix_sha256": digest(f.to_json(orient="records").encode()),
    }


def freeze_day(folder, output, cache, warnings):
    source = folder / "analysis_universe.json"
    raw = source.read_bytes()
    snapshot = json.loads(raw.decode("utf-8-sig"))
    target = output / "frozen" / f"{snapshot['as_of']}.json"
    if target.exists():
        frozen = read_json(target)
        if frozen["source_sha256"] != digest(raw):
            warnings.append(f"{folder.name}: source changed; retained first frozen inputs")
        return frozen
    if snapshot["version"] != CONFIG["snapshot_version"]:
        warnings.append(f"{folder.name}: different snapshot version; excluded")
        return None
    rows, excluded = [], []
    day = snapshot["as_of"]
    for r in snapshot["records"]:
        if r.get("tracking_only"):
            continue
        symbol = r["symbol"]
        if symbol not in cache:
            cache[symbol] = pd.read_csv(ROOT / "data" / f"{symbol}_daily.csv", dtype={"date": str})
        f = features(cache[symbol], day)
        archived_path = folder / f"stock_analysis_{symbol}.json"
        if f is None or r.get("price_date") != day or not archived_path.exists():
            excluded.append(symbol)
            continue
        archived = read_json(archived_path)
        if archived["meta"].get("price_date") != day or archived["meta"].get("report_date") != day:
            excluded.append(symbol)
            continue
        route = archived.get("route_evidence", {})
        pv = r.get("price_volume_context", {}).get("latest", {})
        row = {k: r.get(k) for k in ["symbol", "name", "route_ids", "structural_advice", "value_tier",
               "value_rank", "stock_lifecycle_label", "ema20_bias_pct", "route_bias_atr", "belonging_theme", "cycle_id"]}
        row.update(f, date=day, close=float(r["close"]), market=snapshot["market"].get("structure"),
                   board="GROWTH_STAR" if symbol.startswith(("sz_30", "sh_68")) else "MAIN",
                   retained=route.get("retained"), demand=route.get("demand"), supply=route.get("supply"),
                   pv_state=pv.get("state", "NOT_ARCHIVED"), pv_event=pv.get("event", ""))
        row["safe"] = r["value_tier"] not in {"RISK_WATCH", "DATA_WAIT", "HISTORICAL_ONLY"} and r["structural_advice"] not in {"RISK_WATCH", "DATA_WAIT"}
        row["quality_grade"] = (int(f["strong_close"]) + int(f["strong_close"] and route.get("retained") is True))
        row["stage_order"] = ORDER.get(r["structural_advice"], 99)
        row["tier_order"] = ORDER.get(r["value_tier"], 99)
        row["analysis_version"] = archived["meta"].get("version")
        row["engine_version"] = archived["meta"].get("engine_version")
        rows.append(row)
    frozen = dict(date=day, version=VERSION, source=str(source.relative_to(ROOT)), source_sha256=digest(raw),
                  source_mtime=datetime.fromtimestamp(source.stat().st_mtime).isoformat(),
                  first_ingested_at=datetime.now(timezone.utc).isoformat(),
                  provenance="SAVED_ASOF_RECONSTRUCTION_NOT_ORIGINAL_LIVE_EXECUTION",
                  excluded=excluded, rows=records(pd.DataFrame(rows)))
    write_json(target, frozen)
    return frozen


def outcomes(raw, signal, end, horizon, calendar):
    future = [d for d in calendar if signal < d <= end]
    if len(future) < horizon:
        return {}
    wanted = future[:horizon]
    f = raw.set_index("date")
    if f.index.duplicated().any() or any(d not in f.index for d in wanted):
        return {}
    v = pd.to_numeric(f.loc[wanted, "pct_chg"], errors="coerce")
    if v.isna().any() or (v <= -100).any():
        return {}
    return {f"ret{horizon}": float(((1 + v / 100).prod() - 1) * 100), f"end{horizon}": wanted[-1]}


def rank(frame, policy):
    if policy == "original":
        columns, ascending = ["value_rank", "symbol"], [True, True]
    elif policy == "low_position":
        columns, ascending = ["position60", "value_rank", "symbol"], [True, True, True]
    elif policy == "high_turnover":
        columns, ascending = ["turnover_rel", "value_rank", "symbol"], [False, True, True]
    else:
        columns, ascending = ["quality_grade", "r20", "value_rank", "symbol"], [False, False, True, True]
        if policy == "quality_within_stage":
            columns, ascending = ["tier_order", "stage_order"] + columns, [True, True] + ascending
    return frame.sort_values(columns, ascending=ascending, na_position="last")


def summary(frame, column):
    values = frame[column].dropna()
    return dict(n=len(values), days=int(frame.loc[values.index, "date"].nunique()),
                symbols=int(frame.loc[values.index, "symbol"].nunique()),
                win_pct=float(values.gt(1e-5).mean() * 100) if len(values) else None,
                mean=float(values.mean()) if len(values) else None,
                median=float(values.median()) if len(values) else None,
                p10=float(values.quantile(.1)) if len(values) else None,
                worst=float(values.min()) if len(values) else None)


def run(start, end, output):
    if start != CONFIG["snapshot_start"]:
        raise ValueError("Start date is frozen with this study protocol")
    output.mkdir(parents=True, exist_ok=True)
    protocol = output / "protocol.json"
    if protocol.exists() and read_json(protocol) != CONFIG:
        raise ValueError("Frozen protocol differs; do not silently retune this study")
    if not protocol.exists():
        write_json(protocol, CONFIG)
    rules = dict(order=ORDER, functions={fn.__name__: digest(inspect.getsource(fn).encode())
                                        for fn in (features, outcomes, rank)})
    rules_path = output / "rules.json"
    if rules_path.exists() and read_json(rules_path) != rules:
        raise ValueError("Frozen feature/ranking/label rules changed; start a separately versioned study")
    if not rules_path.exists():
        write_json(rules_path, rules)
    index = pd.read_csv(ROOT / "data/000001_SH_index.csv", dtype={"trade_date": str})
    calendar = sorted(index.trade_date.unique())
    end = end or max(d for d in calendar if d <= CONFIG["observation_end"])
    if end not in calendar:
        raise ValueError("Outcome cutoff must be an existing market session")
    if end > CONFIG["observation_end"]:
        raise ValueError("This study ends on 2026-09-30")
    cache, all_rows, warnings, manifests = {}, [], [], []
    for day in [d for d in calendar if start <= d <= end]:
        folder = ENGINE / day.replace("-", "")
        if not (folder / "analysis_universe.json").exists():
            warnings.append(f"{day}: final selection archive missing; no reconstruction")
            continue
        frozen = freeze_day(folder, output, cache, warnings)
        if frozen is not None:
            manifests.append({k: v for k, v in frozen.items() if k != "rows"})
            all_rows.extend(frozen["rows"])
            print(f"[ranking_review] frozen {day}: {len(frozen['rows'])} rows", flush=True)
    for warning in warnings:
        print(f"[ranking_review] warning: {warning}", flush=True)
    frame = pd.DataFrame(all_rows)
    if frame.empty:
        raise ValueError("No compatible snapshots")
    assert not frame.duplicated(["date", "symbol"]).any()
    frame["sample_period"] = np.where(frame.date < CONFIG["freeze_date"], "INITIAL_EXPLORATORY", "FORWARD_FROZEN_RULES")
    label_rows = []
    for r in all_rows:
        symbol = r["symbol"]
        if symbol not in cache:
            cache[symbol] = pd.read_csv(ROOT / "data" / f"{symbol}_daily.csv", dtype={"date": str})
        labels = {}
        for h in CONFIG["horizons"]:
            labels.update(outcomes(cache[symbol], r["date"], end, h, calendar))
        label_rows.append(labels)
    frame = pd.concat([frame, pd.DataFrame(label_rows)], axis=1)
    for h in CONFIG["horizons"]:
        if f"ret{h}" not in frame:
            frame[f"ret{h}"] = np.nan
    position = pd.cut(frame.position60, [-np.inf, CONFIG["position60_low"], CONFIG["position60_high"], np.inf], labels=["LOW<=35%", "MID35-75%", "HIGH>75%"])
    turn = pd.cut(frame.turnover_rel, [-np.inf, 1, CONFIG["activity_high"], 3, np.inf], labels=["TURN<=1", "TURN1-1.5", "TURN1.5-3", "TURN>3"])
    frame["position_bucket"], frame["turn_bucket"] = position.astype(str), turn.astype(str)
    low = frame.position60.le(CONFIG["position60_low"])
    active = frame.turnover_rel.ge(CONFIG["activity_high"]) & frame.amount_rel.ge(CONFIG["activity_high"])
    groups = {"ALL": pd.Series(True, index=frame.index), "LOW_ONLY": low,
              "HIGH_ACTIVITY": active, "LOW_HIGH_ACTIVITY": low & active,
              "LOW_HIGH_ACTIVITY_STRONG_CLOSE": low & active & frame.strong_close,
              "HIGH_ACTIVITY_STRONG_CLOSE": active & frame.strong_close,
              "HIGH_ACTIVITY_WEAK_CLOSE": active & frame.close_location.le(CONFIG["close_weak"]),
              "STRONG_CLOSE_RETAINED": frame.strong_close & frame.retained.eq(True),
              "SHRINKING_DOWN_TURNOVER": frame.down_up_turnover.le(CONFIG["pullback_down_up"])}
    for label in position.cat.categories:
        groups[str(label)] = position.eq(label)
    for label in turn.cat.categories:
        groups[str(label)] = turn.eq(label)
    factors, rankings, picks = [], [], []
    scopes = [(period, scope, period_mask & scope_mask)
              for period in frame.sample_period.unique()
              for period_mask in [frame.sample_period.eq(period)]
              for scope, scope_mask in [("ALL_POOL", pd.Series(True, index=frame.index)), ("SAFE_POOL", frame.safe)]]
    for period, scope, mask in scopes:
        for h in CONFIG["horizons"]:
            column = f"ret{h}"
            baseline = frame[mask].groupby("date")[column].mean()
            for label, member in groups.items():
                subset = frame[mask & member].copy()
                subset["excess"] = subset[column] - subset.date.map(baseline)
                factors.append(dict(sample_period=period, scope=scope, horizon=h, factor=label, **summary(subset, column),
                                    day_adjusted_excess=float(subset.excess.mean()) if subset.excess.notna().any() else None))
    # Every ranking policy sees exactly the same daily safe pool; market gating is ignored.
    for day, daily in frame[frame.safe].groupby("date"):
        for policy in CONFIG["policies"]:
            ordered = rank(daily, policy)
            for k in CONFIG["top_k"]:
                top = ordered.head(k)
                for i, row in enumerate(records(top), 1):
                    picks.append(dict(policy=policy, k=k, shadow_rank=i, **row))
                for h in CONFIG["horizons"]:
                    s = summary(top, f"ret{h}")
                    rankings.append(dict(date=day, sample_period=daily.sample_period.iloc[0], policy=policy, k=k, horizon=h,
                                         selected=len(top), complete=s["n"] == len(top), **s))
    ranks = pd.DataFrame(rankings)
    aggregate = []
    completed = ranks[ranks.complete & ranks.n.gt(0)]
    for (period, policy, k, h), g in completed.groupby(["sample_period", "policy", "k", "horizon"]):
        eligible_days = set(g.date)
        base = completed[(completed.policy == "original") & completed.k.eq(k) & completed.horizon.eq(h) & completed.date.isin(eligible_days)]
        g = g[g.date.isin(base.date)]
        if g.empty:
            continue
        delta = g.set_index("date")["mean"] - base.set_index("date")["mean"]
        aggregate.append(dict(sample_period=period, policy=policy, k=int(k), horizon=int(h), days=len(g), n=int(g.n.sum()),
                              win_pct=float(g.win_pct.mean()), mean=float(g["mean"].mean()),
                              mean_delta=float(delta.mean()), improved_days=int(delta.gt(1e-8).sum()),
                              worsened_days=int(delta.lt(-1e-8).sum())))
    # Pending rankings are also persisted before their future outcomes exist.
    pending = [p for p in picks if p["date"] == end and p["k"] == 5]
    strata = []
    for (period, board, route), group in frame[frame.safe].groupby(["sample_period", "board", "route_ids"]):
        for h in CONFIG["horizons"]:
            strata.append(dict(sample_period=period, board=board, routes=route, horizon=h, **summary(group, f"ret{h}")))
    correlation = (frame.turnover_rel.rank().corr(frame.volume_rel.rank())
                   if frame.turnover_rel.nunique() > 1 and frame.volume_rel.nunique() > 1 else np.nan)
    meta = dict(version=VERSION, as_of=end, start=start, config=CONFIG, warnings=warnings,
                manifests=manifests, rows=len(frame), symbols=int(frame.symbol.nunique()),
                days=sorted(frame.date.unique()), market_counts=frame.market.value_counts().to_dict(),
                features_missing=frame[["position60", "turnover_rel", "amount_rel", "close_location"]].isna().sum().to_dict(),
                activity_correlation=float(correlation) if pd.notna(correlation) else None,
                analysis_versions=frame.analysis_version.value_counts().to_dict(),
                engine_versions=frame.engine_version.value_counts().to_dict(), strata=strata,
                factor_results=factors, ranking_results=aggregate, daily_rankings=records(ranks),
                frozen_pending_top5=pending)
    destination = output / "reviews" / end.replace("-", "")
    if (destination / "summary.json").exists():
        previous = read_json(destination / "summary.json")
        if (previous["rows"] == len(frame) and previous["ranking_results"] == aggregate
                and read_json(destination / "observations.json") == records(frame)):
            print(f"UNCHANGED: {destination}")
            return previous
        raise ValueError("Existing dated review differs; keep audit history, use a new study root")
    write_json(destination / "summary.json", meta)
    write_json(destination / "observations.json", records(frame))
    write_json(destination / "ranked_picks.json", picks)
    write_report(destination, meta)
    print(json.dumps(dict(output=str(destination), rows=len(frame), dates=meta["days"],
                          ranking_results=[r for r in aggregate if r["k"] == 5]), ensure_ascii=False, indent=2))
    return meta


def write_report(destination, meta):
    def fmt(v):
        return "--" if v is None else f"{v:+.2f}"
    lines = [f"# 新引擎参数与影子排序初评，截至 {meta['as_of']}", "",
        "## 边界", "",
        f"- {meta['rows']}个股票-日期观察，{meta['symbols']}只股票，{len(meta['days'])}个筛选日；重复股票不视为独立样本。",
        "- 固定1/3/5个市场交易日观察期，不将未成熟样本计作零收益。收益为筛选收盘后复合涨跌幅，不是成交收益。",
        "- 保留原扫描池、结构和风险标签。安全池剔除个股风险/数据不全，仅忽略市场门槛。所有排序比较使用同一安全池。",
        "- 初次输入来自已保存的同版本重建快照，不证明当时实时命中；不重跑选股。冻结后源文件改变只告警、不覆盖。",
        "- 初评属于探索；2026-09-16以后日期独立汇总。参数冻结至9月30日，复盘不自动调参，不改正式排名或持仓。",
        "- 低位=60日区间位置<=35%；活跃=换手和成交额相对各自前20日中位数均>=1.5倍；强收盘=收盘位>=70%且收盘>=开盘。阈值均为待校准假设。",
        "- 承接等级=强收盘且原引擎冲击保留为2，仅强收盘为1，否则0；非主力身份或低位吸筹的证明。",
        "- 同阶段质量排序=原建议类别、结构类别优先，再承接等级、20日涨幅、原名次；跨阶段质量排序去掉类别，仅用于诊断分层的影响。此处等级为实验代理，不是真实资金身份。",
        "- 本次不搜索最优阈值，不将短期最优方案直接上线；极少日期、弱市、题材集中均限制推论。",
        f"- 相对换手与相对成交量Spearman相关系数：{meta['activity_correlation']}，不可视为两份独立资金证据。",
        f"- 归档个股分析版本：{meta['analysis_versions']}。总榜核心版本一致不等于所有子模块同版本；因此本轮仍是探索样本。",
        f"- 归档引擎版本：{meta['engine_versions']}；输入不合格剔除：{[(m['date'], m['excluded']) for m in meta['manifests'] if m['excluded']]}。",
        "- Top5包含尚未满足买点条件的观察股。尤其跨阶段排序不能等同于可买入榜单，涨幅改善也不能代表可执行利润。",
        *[f"- 缺口：{w}" for w in meta['warnings']], "", "## Top5 排序比较", "",
        "| 样本期 | 排序 | 观察期 | 完整日数 | 样本数 | 上涨比例% | 日等权平均涨跌% | 较原排序百分点 | 改善/恶化日数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in meta['ranking_results']:
        if r['k'] == 5:
            lines.append(f"| {r['sample_period']} | {r['policy']} | {r['horizon']} | {r['days']} | {r['n']} | {r['win_pct']:.2f} | {fmt(r['mean'])} | {fmt(r['mean_delta'])} | {r['improved_days']}/{r['worsened_days']} |")
    lines += ["", "## Top5逐日结果（1日后）", "", "| 筛选日 | 排序 | 完整 | 样本 | 上涨比例% | 平均涨跌% |", "|---|---|---|---:|---:|---:|"]
    for r in meta['daily_rankings']:
        if r['k'] == 5 and r['horizon'] == 1:
            lines.append(f"| {r['date']} | {r['policy']} | {r['complete']} | {r['n']} | {fmt(r['win_pct'])} | {fmt(r['mean'])} |")
    lines += ["", "## 参数分组（1日后，安全池）", "", "| 样本期 | 组合 | 样本 | 日期数 | 上涨比例% | 平均涨跌% | 中位数% | 最差% | 同日池均值差百分点 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in meta['factor_results']:
        if r['scope'] == 'SAFE_POOL' and r['horizon'] == 1:
            lines.append(f"| {r['sample_period']} | {r['factor']} | {r['n']} | {r['days']} | {fmt(r['win_pct'])} | {fmt(r['mean'])} | {fmt(r['median'])} | {fmt(r['worst'])} | {fmt(r['day_adjusted_excess'])} |")
    lines += ["", "## 最新日期冻结的影子Top5（待后续验证，非买入建议）", "", "| 排序 | 名次 | 股票 | 路径 | 结构状态 | 承接等级 | 相对换手 |", "|---|---:|---|---|---|---:|---:|"]
    for p in meta['frozen_pending_top5']:
        routes = (p.get('route_ids') or '').replace('|', '+')
        lines.append(f"| {p['policy']} | {p['shadow_rank']} | {p['name']} {p['symbol']} | {routes} | {p['structural_advice']} | {p['quality_grade']} | {fmt(p['turnover_rel'])} |")
    lines += ["", "## 后续复盘", "", "- 使用同一脚本、同一研究目录增量运行；冻结输入和参数，仅追加新日期。",
              "- 分开报告初评样本和冻结规则后的日期，报告各日改善方向，不能只报告合并均值。",
              "- 检查Top3/5/15、1/3/5日、不同路径和上市板块；少于足够独立日期时维持研究状态。",
              "- 每3天复盘一次至9月30日；无新增数据不重复调参，不虚构收益。", ""]
    (destination / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-09-09")
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--output", type=Path, default=ENGINE / "ranking_review_v1_20260916")
    args = parser.parse_args()
    run(args.start, args.as_of, args.output)
