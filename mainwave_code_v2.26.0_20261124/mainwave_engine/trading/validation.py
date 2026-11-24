"""Forward validation, ablation and missed audit. Immature is never scored as zero."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import pandas as pd
from . import ranking, store

VERSION = "ranking_validation_v2_20260920"
# A posterior definition used only to find cases to audit; it never re-enters the day's ranking.
MISSED_RULE = {"id": "missed_v1_20260920", "horizon": 20, "close_gain_pct": 25,
               "definition": "信号日后20个交易日内收盘涨幅≥25%",
               "use": "AUDIT_ONLY_NOT_FED_BACK_INTO_RANKING"}
STRATA = ("route", "theme", "board", "market", "lifecycle", "participation")
MIN_STRATUM = 5
ATTRIBUTION = {
    "SELECTED_AND_HELD": "选对", "PRICE_EXCEEDED": "买贵", "NOT_REACHABLE": "买不到",
    "EXIT_BIAS": "退出偏差", "CORRECTLY_DECLINED": "正确放弃", "LOSS_WITHIN_RULES": "规则内亏损",
}


def _calendar(data_root):
    """No calendar means no window can be judged; that is a wait, not an empty result."""
    path = Path(data_root) / "000300_SH_index.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path, usecols=["trade_date"])
    return sorted({str(x) if "-" in str(x) else f"{str(x)[:4]}-{str(x)[4:6]}-{str(x)[6:8]}"
                   for x in frame.trade_date})


def window(symbol_frame, calendar, signal_day, horizon):
    """Market days define the window; a halt inside it breaks maturity, it does not slide it."""
    if not calendar:
        return {"status": "DATA_WAIT", "reason": "MARKET_CALENDAR_MISSING"}
    ahead = [d for d in calendar if d > signal_day][:horizon]
    if len(ahead) < horizon:
        return {"status": "IMMATURE", "reason": "MARKET_DAYS_PENDING", "expected": len(ahead)}
    if symbol_frame is None or symbol_frame.empty:
        return {"status": "DATA_WAIT", "reason": "NO_PRICE_SERIES"}
    dates = set(symbol_frame.date)
    if signal_day not in dates:
        return {"status": "DATA_WAIT", "reason": "SIGNAL_DAY_MISSING"}
    missing = [d for d in ahead if d not in dates]
    if missing:
        # Suspension or a gap: both leave the window incomplete; never fill from a later price.
        return {"status": "DATA_GAP", "reason": "HALT_OR_MISSING_INSIDE_WINDOW",
                "missing_days": missing[:5], "missing": len(missing)}
    rows = symbol_frame[symbol_frame.date.isin(ahead)]
    values = pd.to_numeric(rows.pct_chg, errors="coerce")
    if values.isna().any() or values.le(-100).any():
        return {"status": "DATA_GAP", "reason": "INVALID_RETURN_IN_WINDOW"}
    closes = pd.to_numeric(rows.close, errors="coerce")
    start = pd.to_numeric(symbol_frame[symbol_frame.date.eq(signal_day)].close, errors="coerce").iloc[-1]
    index = (1 + values / 100).cumprod()
    return {"status": "MATURE", "window_days": ahead,
            "return_pct": float((index.iloc[-1] - 1) * 100),
            "max_favourable_pct": float((index.max() - 1) * 100),
            "max_adverse_pct": float((index.min() - 1) * 100),
            "close_gain_pct": float((closes.max() / start - 1) * 100) if start else None}


def _distribution(values):
    if not values:
        return {"n": 0}
    ordered = sorted(values, reverse=True)
    total = sum(v for v in ordered if v > 0)
    top = sum(ordered[:max(1, len(ordered) // 10)])
    return {"n": len(values), "mean": round(statistics.fmean(values), 3),
            "median": round(statistics.median(values), 3),
            "positive_fraction": round(sum(v > 0 for v in values) / len(values), 4),
            "worst": round(min(values), 3), "best": round(max(values), 3),
            "top_decile_share": round(top / total, 4) if total > 0 else None,
            "concentration_note": "少数赢家集中度；正收益全部来自前10%时该值为1。"}


def _stratum(record, snapshot_record):
    source = (snapshot_record or {}).get("source_record") or {}
    symbol = record["symbol"]
    return {
        "route": (snapshot_record or {}).get("route_ids") or "NO_ROUTE",
        "theme": (snapshot_record or {}).get("theme") or "未归属",
        "board": "STAR" if symbol.startswith("sh_68") else "CHINEXT" if symbol.startswith("sz_30") else "MAIN",
        "market": (snapshot_record or {}).get("features", {}).get("status", "UNKNOWN"),
        "lifecycle": source.get("stock_lifecycle_label") or source.get("phase_label") or "UNKNOWN",
        "participation": (snapshot_record or {}).get("state") or "UNKNOWN",
    }


def evaluate(day, db=store.DB_PATH, data_root=None, horizons=None):
    """One frozen first publication per day and version; rows, dates and opportunities counted apart."""
    data_root = Path(data_root or store.ROOT / "data")
    horizons = horizons or ranking.RULES["evaluation_horizons"]
    calendar = _calendar(data_root)
    with store.connect(db) as conn:
        rows = conn.execute("SELECT payload FROM snapshots WHERE as_of<? ORDER BY as_of,rowid", (day,)).fetchall()
    seen, results, cache, index = set(), [], {}, {}
    for source in rows:
        snap = json.loads(source[0])
        key = (snap["as_of"], snap["version"])
        if key in seen:
            continue
        seen.add(key)
        index[snap["snapshot_id"]] = snap
        for record in snap["records"]:
            if record["strength_rank"] is None:
                continue
            symbol = record["symbol"]
            if symbol not in cache:
                try:
                    cache[symbol] = ranking.prices(data_root / f"{symbol}_daily.csv", day)
                except (OSError, ValueError):
                    cache[symbol] = pd.DataFrame(columns=["date", "close", "pct_chg"])
            for horizon in horizons:
                outcome = window(cache[symbol], calendar, snap["as_of"], horizon)
                results.append({
                    "snapshot_id": snap["snapshot_id"], "signal_day": snap["as_of"], "symbol": symbol,
                    "name": record.get("name"), "opportunity_id": record["opportunity_id"],
                    "cycle_id": record.get("cycle_id"), "version": snap["version"], "horizon": horizon,
                    "status": outcome["status"], "reason": outcome.get("reason", ""),
                    "return_pct": outcome.get("return_pct"),
                    "max_favourable_pct": outcome.get("max_favourable_pct"),
                    "max_adverse_pct": outcome.get("max_adverse_pct"),
                    "close_gain_pct": outcome.get("close_gain_pct"),
                    "strength_rank": record["strength_rank"], "participation_rank": record["participation_rank"],
                    "legacy_rank": record.get("legacy_rank"), "momentum_rank": record.get("momentum_rank"),
                    "state": record.get("state"), "strata": _stratum(record, record),
                    "alpha_scope": record.get("alpha_scope") or ranking.alpha_scope(record.get("route_ids", "")),
                    "basis": snap["observation_basis"], "is_trade_return": False})

    groups, comparisons = {}, []
    for row in results:
        groups.setdefault((row["snapshot_id"], row["horizon"]), []).append(row)
    cohorts = {"strength": "strength_rank", "participation": "participation_rank",
               "legacy": "legacy_rank", "momentum": "momentum_rank"}
    for (snapshot_id, horizon), everyone in groups.items():
        # R3-only rows are research leads; they get their own line and never enter the strategy cohorts.
        group = [r for r in everyone if r["alpha_scope"] != "R3_RESEARCH"]
        research = [r for r in everyone if r["alpha_scope"] == "R3_RESEARCH"]
        sample = everyone[0]
        for name, field in cohorts.items():
            for top in (3, 5, 15):
                chosen = [r for r in group if r.get(field) is not None and r[field] <= top]
                comparisons.append(_cohort(f"{name}_top{top}", chosen, sample, horizon))
        comparisons.append(_cohort("all_candidates", group, sample, horizon))
        for route in ("R1", "R2"):
            comparisons.append(_cohort(f"route_{route}", [r for r in group if route in str(r["strata"]["route"]).split("|")],
                                       sample, horizon))
        comparisons.append(_cohort("r3_research", research, sample, horizon, research=True))

    # Strata are aggregated per domain: an R3-only row never lands in a price-alpha theme or board bucket.
    strata, strata_research = {}, {}
    domains = ((strata, [r for r in results if r["alpha_scope"] != "R3_RESEARCH"], False),
               (strata_research, [r for r in results if r["alpha_scope"] == "R3_RESEARCH"], True))
    for name in STRATA:
        for target, rows_in_domain, research in domains:
            buckets = {}
            for row in rows_in_domain:
                if row["horizon"] != horizons[0]:
                    continue
                buckets.setdefault(row["strata"][name], []).append(row)
            target[name] = [_cohort(f"{name}={key}", rows_, rows_[0], horizons[0], stratum=True, research=research)
                            for key, rows_ in sorted(buckets.items())]
    return {"version": VERSION, "as_of": day, "records": results, "comparisons": comparisons,
            "strata": strata, "strata_research": strata_research, "horizons": horizons,
            "summary": _counts(results),
            "clock": {"basis": "MARKET_CALENDAR_DAYS", "halt_policy": "DATA_GAP_NOT_SLID",
                      "note": "窗口按市场交易日固定；窗口内停牌或缺日记 DATA_GAP，不用后一个有价格日顶替。"},
            "notice": "固定 5/10/20 个市场交易日的价格观察，不是成交收益；同股连续上榜不计为多个独立成功；"
                      "未成熟不记零；回建样本不证明当时能发现。",
            "auto_tuning": False}


def _counts(results):
    """Totals, then the same totals per observation basis: a reconstruction never counts as live forward."""
    counts = _basic(results)
    counts["by_basis"] = {basis: _basic([r for r in results if r.get("basis") == basis])
                          for basis in sorted({r.get("basis", "UNSPECIFIED") for r in results})}
    return counts


def _clusters(rows):
    """Rows born of one theme shock on one day are one observation, however many symbols it lit up."""
    return {"signal_days": len({r["signal_day"] for r in rows}),
            "theme_days": len({(r.get("strata", {}).get("theme"), r["signal_day"]) for r in rows})}


def _basic(results):
    mature = [r for r in results if r["status"] == "MATURE"]
    return {"rows": len(results), "mature_rows": len(mature), "clusters": _clusters(results),
            "n_effective_note": "显著性检验须按题材×信号日簇自举，不按行自举",
            "research_rows": sum(r.get("alpha_scope") == "R3_RESEARCH" for r in results),
            "immature_rows": sum(r["status"] == "IMMATURE" for r in results),
            "data_gap_rows": sum(r["status"] == "DATA_GAP" for r in results),
            "data_wait_rows": sum(r["status"] == "DATA_WAIT" for r in results),
            "independent_days": len({r["signal_day"] for r in results}),
            "independent_opportunities": len({r["opportunity_id"] for r in results}),
            "independent_cycles": len({(r["symbol"], r["cycle_id"]) for r in results}),
            "independent_symbols": len({r["symbol"] for r in results}),
            "pending_rows": sum(r["status"] == "IMMATURE" for r in results),
            "unique_opportunities": len({r["opportunity_id"] for r in results})}


def _cohort(name, chosen, sample, horizon, stratum=False, research=False):
    mature = [r for r in chosen if r["status"] == "MATURE"]
    values = [r["return_pct"] for r in mature]
    result = {"group": name, "signal_day": sample["signal_day"], "version": sample["version"],
              "basis": sample.get("basis", "UNSPECIFIED"),
              "alpha_scope": "R3_RESEARCH" if research else "PRICE_ALPHA",
              "clusters": _clusters(chosen),
              "horizon": horizon, "selected_n": len(chosen), "mature_n": len(mature),
              "independent_opportunities": len({r["opportunity_id"] for r in chosen}),
              "independent_symbols": len({r["symbol"] for r in chosen}),
              "independent_cycles": len({(r["symbol"], r.get("cycle_id")) for r in chosen}),
              "complete": bool(chosen) and len(mature) == len(chosen),
              **_distribution(values)}
    # Legacy key kept so existing readers of ranking_validation.json keep working.
    result["mean_return_pct"] = result.get("mean")
    if stratum:
        result["sufficient"] = len(mature) >= MIN_STRATUM
        result["status"] = "SUFFICIENT" if result["sufficient"] else "INSUFFICIENT_SAMPLE_RETAINED"
    if research:
        result["note"] = "R3 仅研究线索：历史题材成分非时点数据，不进入策略收益统计"
    return result


MEMORY_FIELDS = ("persistent", "retest_confirmed", "retention_lost")
ADVICE_ORDER = {"PRIORITY_REVIEW": 0, "WAIT_CONFIRMATION": 1, "WAIT_PRICE": 2, "EARLY_WATCH": 3,
                "TREND_WATCH": 4, "MARKET_WAIT": 5, "WATCH": 6, "DATA_WAIT": 7, "RISK_WATCH": 8}
TIER_ORDER = {"A_READY": 0, "B_SETUP": 1, "C_WATCH": 2, "D_AVOID": 3}


def _abs_bias(row):
    try:
        return abs(float(row.get("route_bias_atr")))
    except (TypeError, ValueError):
        return 9.9


def _order(row, score):
    return (ADVICE_ORDER.get(str(row.get("structural_advice")), 9),
            TIER_ORDER.get(str(row.get("value_tier")), 9), -score, _abs_bias(row), row["symbol"])


def ablate(day, root=store.ROOT, protocol="multiday_pool_v1_20260916", top=5):
    """Remove same-pool memory from that day's ACTIVE rows and re-rank with the frozen tie-break."""
    folder = Path(root) / "results/mainwave_engine" / protocol / "reviews"
    available = sorted(p.name for p in folder.glob("*")) if folder.exists() else []
    path = folder / day.replace("-", "") / "observations.json"
    if not path.exists():
        return {"version": VERSION, "status": "PROTOCOL_DAY_MISSING", "protocol": protocol, "as_of": day,
                "available_days": available,
                "notice": "该协议当日没有冻结观察，不能用生产快照或别的协议顶替。"}
    rows = [r for r in json.loads(path.read_text(encoding="utf-8-sig"))
            if r.get("date") == day and r.get("pool_status") == "ACTIVE"]
    if not rows:
        return {"version": VERSION, "status": "NO_ACTIVE_ROWS", "protocol": protocol, "as_of": day,
                "available_days": available, "notice": "当日没有 ACTIVE 行可供消融。"}
    counts = {f: sum(1 for r in rows if f in (r.get("components") or {})) for f in MEMORY_FIELDS}
    scored = []
    for row in rows:
        components = row.get("components") or {}
        score = float(row.get("score") or 0)
        removed = sum(components[f] for f in MEMORY_FIELDS if f in components)
        scored.append({"row": row, "baseline": score, "ablated": score - removed, "removed": removed})
    base_order = [e["row"]["symbol"] for e in sorted(scored, key=lambda e: _order(e["row"], e["baseline"]))]
    abl_order = [e["row"]["symbol"] for e in sorted(scored, key=lambda e: _order(e["row"], e["ablated"]))]
    base_top, abl_top = base_order[:top], abl_order[:top]
    moved = [{"symbol": e["row"]["symbol"], "name": e["row"].get("name"),
              "removed_points": e["removed"],
              "baseline_rank": base_order.index(e["row"]["symbol"]) + 1,
              "ablated_rank": abl_order.index(e["row"]["symbol"]) + 1}
             for e in scored if e["removed"]]
    moved.sort(key=lambda m: m["baseline_rank"])
    return {"version": VERSION, "status": "READY", "protocol": protocol, "as_of": day,
            "memory_fields": list(MEMORY_FIELDS), "memory_field_counts": counts,
            "active_rows": len(rows), "memory_rows": len(moved), "top_k": top,
            "baseline_top": base_top, "ablated_top": abl_top,
            "top_overlap": len(set(base_top) & set(abl_top)),
            "rank_moves": moved[:50],
            "order": "建议类别 → 结构类别 → 质量分降序 → 偏离升序 → 代码",
            "production_score_uses_memory": False,
            "finding": ("生产强势分只用当日相对动量、回撤与推进效率，本身不含同池记忆；"
                        "记忆消融只在带加分项的研究协议内有意义，两套成绩不可合并。"),
            "notice": "消融只改变本协议的排名顺序，不改变任何生产状态，也不构成收益结论。"}


def protocols(root=store.ROOT):
    """Three separate protocols; results are never merged into one score sheet."""
    root = Path(root)
    engine = root / "results/mainwave_engine"
    found = []
    for folder in sorted(engine.glob("*_v1_*")):
        path = folder / "protocol.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        config = payload.get("config", payload)
        frozen = sorted(p.name for p in (folder / "frozen").glob("*.json")) if (folder / "frozen").exists() else []
        found.append({"protocol": folder.name, "version": config.get("version"),
                      "scope": config.get("snapshot_version"), "freeze_date": config.get("freeze_date"),
                      "window": {"start": config.get("start") or config.get("snapshot_start"),
                                 "end": config.get("end") or config.get("observation_end")},
                      "horizons": config.get("horizons"), "mode": config.get("mode"),
                      "function_hashes": payload.get("functions", {}),
                      "frozen_days": len(frozen), "first_frozen": frozen[0] if frozen else "",
                      "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    found.append({"protocol": "production_forward_record", "version": VERSION,
                  "scope": "trading_closure snapshots", "freeze_date": "",
                  "window": {"start": "", "end": ""}, "horizons": ranking.RULES["evaluation_horizons"],
                  "mode": "FORWARD_RECORDED_AFTER_CLOSE", "function_hashes": {}, "frozen_days": 0,
                  "first_frozen": "", "config_sha256": ""})
    return {"version": VERSION, "protocols": found,
            "notice": "各协议的版本、输入哈希、发布时点、股票域与窗口独立保存；"
                      "扩大指标不得覆盖旧结果，不同协议的成绩不可合并为一套。"}


def _audit_table(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS missed_audits (id TEXT PRIMARY KEY, rule_id TEXT NOT NULL, "
                 "signal_day TEXT NOT NULL, payload TEXT NOT NULL)")


def missed_audit(day, db=store.DB_PATH, data_root=None, rule=None):
    """Every frozen signal day whose posterior window has matured by `day` is audited once and kept.

    The signal day and the evaluation day are different things: writing today's snapshot must not
    stop an older signal day from maturing. Calendar and prices are cut at the evaluation day.
    """
    rule = rule or MISSED_RULE
    data_root = Path(data_root or store.ROOT / "data")
    calendar = [d for d in _calendar(data_root) if d <= day]
    rule_id = store.digest(rule)[:16]
    with store.connect(db) as conn:
        _audit_table(conn)
        firsts = {}
        for row in conn.execute("SELECT as_of, payload FROM snapshots WHERE as_of<=? ORDER BY as_of, rowid", (day,)):
            firsts.setdefault(row["as_of"], row["payload"])  # first publication of a day is the frozen one
        kept = {r["signal_day"]: json.loads(r["payload"]) for r in conn.execute(
            "SELECT signal_day, payload FROM missed_audits WHERE rule_id=?", (rule_id,))}
    if not firsts:
        return {"version": VERSION, "status": "NO_SNAPSHOT", "rule": rule, "cases": []}
    matured, immature = [], []
    for signal_day, raw in firsts.items():
        ahead = [d for d in calendar if d > signal_day][:rule["horizon"]]
        if len(ahead) < rule["horizon"]:
            immature.append({"signal_day": signal_day, "matured_days": len(ahead)})
            continue
        if signal_day not in kept:
            kept[signal_day] = _audit_day(json.loads(raw), calendar, ahead, rule, data_root)
            with store.connect(db) as conn:
                _audit_table(conn)
                conn.execute("INSERT OR IGNORE INTO missed_audits VALUES (?,?,?,?)",
                             (f"{rule_id}:{signal_day}", rule_id, signal_day, store.encoded(kept[signal_day])))
        matured.append(kept[signal_day])
    latest, newest = (matured[-1] if matured else None), (immature[-1] if immature else None)
    return {"version": VERSION, "status": "READY" if matured else "WINDOW_IMMATURE", "rule": rule,
            "evaluation_as_of": day,
            "signal_day": latest["signal_day"] if latest else (newest or {}).get("signal_day", ""),
            "matured_days": None if latest else (newest or {}).get("matured_days"),
            "matured_signal_days": [m["signal_day"] for m in matured], "immature_signal_days": immature,
            "history": [{"signal_day": m["signal_day"], "summary": m["summary"]} for m in matured],
            **({k: latest[k] for k in ("snapshot_id", "scanned_symbols", "pool_size", "cases", "summary")}
               if latest else {"cases": [], "summary": None}),
            "attribution_labels": ATTRIBUTION,
            "notice": ("后验名单只用于审计当时在哪一步被挡，不回灌当时排名；"
                       "在池被挡不等于误杀，需结合该步骤的规则版本判断。" if matured
                       else "后验窗口未成熟，不得提前给出漏选结论。")}


def _audit_day(snap, calendar, ahead, rule, data_root):
    """Posterior movers of one signal day: were they in the pool, and where did they stop."""
    from . import diagnostics as DG
    signal_day = snap["as_of"]
    in_pool = {r["symbol"]: r for r in snap["records"]}
    cases, scanned = [], 0
    for path in sorted(Path(data_root).glob("*_daily.csv")):
        symbol = path.name[:-len("_daily.csv")]
        if not symbol.startswith(("sh_60", "sh_68", "sz_00", "sz_30")):
            continue
        try:
            frame = ranking.prices(path, ahead[-1])
        except (OSError, ValueError):
            continue
        scanned += 1
        outcome = window(frame, calendar, signal_day, rule["horizon"])
        if outcome["status"] != "MATURE" or (outcome.get("close_gain_pct") or 0) < rule["close_gain_pct"]:
            continue
        record = in_pool.get(symbol)
        if record is None:
            cases.append({"symbol": symbol, "in_pool": False, "stage": "NOT_IN_POOL",
                          "blockers": [], "close_gain_pct": round(outcome["close_gain_pct"], 2),
                          "return_pct": round(outcome["return_pct"], 2),
                          "reason": "当日未进入候选域，需回查扫描与预筛"})
            continue
        card = DG.card(record)
        stopped = next((c["first_blocking_step"] for c in card["cards"] if c["first_blocking_step"]), "")
        cases.append({"symbol": symbol, "name": record.get("name"), "in_pool": True,
                      "opportunity_id": record["opportunity_id"], "state": record.get("state"),
                      "stage": stopped or record.get("state"), "blockers": record.get("blockers") or [],
                      "route_ids": record.get("route_ids", ""),
                      "strength_rank": record.get("strength_rank"),
                      "participation_rank": record.get("participation_rank"),
                      "close_gain_pct": round(outcome["close_gain_pct"], 2),
                      "return_pct": round(outcome["return_pct"], 2),
                      "reason": "在池但被规则挡住" if record.get("state") != "PLAN_REVIEW" else "在池且可复核"})
    stages = Counter(c["stage"] for c in cases)
    attribution = Counter()
    for case in cases:
        if not case["in_pool"]:
            attribution["NOT_REACHABLE"] += 1
        elif "STRUCTURE_RISK" in case.get("blockers", []):
            attribution["CORRECTLY_DECLINED"] += 1
        elif case.get("state") == "PLAN_REVIEW":
            attribution["SELECTED_AND_HELD"] += 1
        else:
            attribution["PRICE_EXCEEDED" if case["stage"] in {"prior_extension", "chase_counter"}
                        else "NOT_REACHABLE"] += 1
    return {"signal_day": signal_day, "snapshot_id": snap["snapshot_id"], "scanned_symbols": scanned,
            "pool_size": len(in_pool), "cases": sorted(cases, key=lambda c: -c["close_gain_pct"])[:200],
            "summary": {"movers": len(cases), "in_pool": sum(c["in_pool"] for c in cases),
                        "not_in_pool": sum(not c["in_pool"] for c in cases),
                        "by_stage": dict(stages), "attribution": dict(attribution)}}
