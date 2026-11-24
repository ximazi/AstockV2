"""Universe price discovery plus dated-chain intersection, with reusable snapshots."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter

import pandas as pd

from . import route_evidence as RE
from . import preparation_detector as PD
from . import theme_semantics as TS
from . import route_contracts as RC
from .pipeline import _write
from .review_clock import publish

POLICY_PATH = Path(__file__).with_name("calibration") / "route_policy.json"
STATUS_LABELS = {"CONFIRMATION_REVIEW": "突破接受，复核介入位置", "PREPARATION_WATCH": "准备观察，非买点",
                 "WAIT_PRICE": "趋势保留，偏离过大不追", "RISK_WATCH": "供给压力，暂缓介入",
                 "TREND_WATCH": "已接受，继续跟踪", "WAIT_ACCEPTANCE": "突破尝试，等待接受",
                 "DATA_WAIT": "数据或资格不足"}
STATUS_ORDER = {s: i for i, s in enumerate(STATUS_LABELS)}


_BENCH = None


def _bench():
    """CSI300 20-day return by date, loaded once per process for the preparation detectors."""
    global _BENCH
    if _BENCH is None:
        try:
            _BENCH = PD.benchmark_ret20()
        except (OSError, ValueError, KeyError):
            _BENCH = pd.Series(dtype=float)
    return _BENCH


def load_policy() -> dict:
    return json.loads(POLICY_PATH.read_text(encoding="utf-8"))


def assess_routes(row: dict, policy: str, routes: list[str]) -> dict:
    result = RE.assess(row, policy)
    current_preparation = row.get("preparation_valid", row.get("prepared", False))
    independent = RC.price_confirmation(row)
    ready = independent or not (set(routes) & {"R2", "R3"}) or bool(current_preparation)
    result["independent_price_evidence"] = independent
    result["route_requirements_met"] = ready
    result["confirmation_pass"] = result["confirmation_pass"] and result["status"] == "CONFIRMATION_REVIEW"
    if not ready:
        result["confirmation_pass"] = False
        if result["status"] == "CONFIRMATION_REVIEW":
            result["status"] = "WAIT_ACCEPTANCE"
    return result


def snapshot(args: tuple[str, str, str]) -> dict:
    filename, date, code_hash = args
    path = Path(filename)
    stat = path.stat()
    fingerprint = [stat.st_size, stat.st_mtime_ns, code_hash, date]
    cache = Path("results/mainwave_engine/_route_cache") / date.replace("-", "") / f"{path.stem}.json"
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint:
            return {**cached["record"], "cache_hit": True}
    symbol = path.stem.removesuffix("_daily")
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig")
        raw = raw[pd.to_datetime(raw.date).le(date)]
        frame = RE.build(raw, symbol)
        if frame.empty:
            return {"symbol": symbol, "data_status": "SHORT_HISTORY", "cache_hit": False}
        row = frame.iloc[-1]
        names = ["date", "close", "ret5", "ret20", "bias_atr", "atr_pct", "probe_at", "retained", "supply",
                 "prepared", "prepared_at", "attempt_prepared_at", "prepared_before_attempt", "demand",
                 "supply_warning", "eligible", "st_known", "liquid", "amount_contract", "turnover_contract",
                 "turnover_relative", "range_contract", "breakout_status", "breakout_event",
                 "breakout_id", "breakout_attempt_at", "breakout_accepted_at", *RE.POLICIES]
        names.extend(RE.PT.FIELDS)
        record = json.loads(frame.tail(1)[list(dict.fromkeys(names))].to_json(orient="records", date_format="iso"))[0]
        record["date"] = str(row.date.date())
        record["symbol"] = symbol
        ema20 = frame.return_index.ewm(span=20, adjust=False).mean().iloc[-1]
        record["ema20_bias"] = float((row.return_index / ema20 - 1) * 100)
        record["data_status"] = "FRESH" if record["date"] == date else "STALE"
        accepted = frame.index[frame.breakout_id.eq(row.breakout_id) & frame.breakout_event.eq("ACCEPTED")]
        record["acceptance_age"] = int(len(frame) - 1 - accepted[-1]) if len(accepted) else None
        attempts = frame.index[frame.breakout_id.eq(row.breakout_id) & frame.breakout_event.eq("ATTEMPT")]
        record["attempt_age"] = int(len(frame) - 1 - attempts[-1]) if len(attempts) else None
        factor = float(row.close / row.return_index)
        record["boundary_price"] = float(row.breakout_boundary_index * factor) if pd.notna(row.breakout_boundary_index) else None
        record["invalid_price"] = float(row.breakout_invalid_index * factor) if pd.notna(row.breakout_invalid_index) else None
        record["frozen_atr_price"] = float(row.breakout_frozen_atr * factor) if pd.notna(row.breakout_frozen_atr) else None
        # Shadow preparation readings ride along in the cached record; they create no route and no candidate state.
        record["preparation"] = PD.latest(frame, _bench())
        record["cache_hit"] = False
        _write(cache, {"fingerprint": fingerprint, "record": record})
        return record
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        return {"symbol": symbol, "data_status": "ERROR", "error": str(exc), "cache_hit": False}


def route_ids(row: dict, chains: list[dict], date: str) -> tuple[list[str], list[dict]]:
    if row.get("data_status") != "FRESH" or not row.get("eligible", False):
        return [], []
    accepted = row.get("acceptance_age") is not None and row["acceptance_age"] <= 3 and row.get("breakout_status") in {"ACCEPTED", "ADVANCING"}
    attempt = row.get("attempt_age") is not None and row["attempt_age"] <= 3 and row.get("breakout_status") == "ATTEMPT"
    prepared = bool(row.get("prepared") or (accepted and row.get("prepared_before_attempt")))
    linked = [c for c in chains if str(c.get("as_of")) == date and c.get("theme_id") and c.get("leader")
              and c["leader"] != row["symbol"] and c.get("anchor_date") and str(c["anchor_date"]) <= date
              and c.get("theme_lifecycle") in {"FERMENTING", "IGNITED", "DIFFUSING"}
              and 1 <= pd.to_numeric(c.get("anchor_age_sessions"), errors="coerce") <= 8
              and pd.to_numeric(c.get("lag_gap_atr"), errors="coerce") >= 0.75]
    routes = []
    if prepared:
        routes.append("R2")
    if accepted or attempt:
        routes.append("R1")
    if linked:
        routes.insert(0, "R3")
        if "R1" in routes and not prepared:
            routes.remove("R1")
            routes.insert(0, "R1")
    return routes, linked


def run(date: str, workers: int = 4) -> dict:
    output = Path("results/mainwave_engine") / date.replace("-", "")
    policy = load_policy()
    code_hash = hashlib.sha256(b"".join(Path(p).read_bytes() for p in
        (__file__, RE.__file__, RE.PE.__file__, RE.BE.__file__, RE.PT.__file__, RC.__file__, PD.__file__))).hexdigest()
    paths = sorted(p for p in Path("data").glob("*_daily.csv") if re.fullmatch(
        r"(?:sh_(?:60|68)|sz_(?:00|30))\d{4}_daily.csv", p.name))
    started = perf_counter()
    audit = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(snapshot, [(str(p), date, code_hash) for p in paths], chunksize=8), 1):
            audit.append(row)
            if i % 1000 == 0 or i == len(paths):
                print(f"[route_scan] {i}/{len(paths)} seconds={perf_counter()-started:.1f}", flush=True)
    chain_path = output / "slow_carrier_diagnostics.csv"
    chains = pd.read_csv(chain_path).fillna("").to_dict("records") if chain_path.exists() else []
    by_symbol = {}
    for chain in chains:
        by_symbol.setdefault(chain["symbol"], []).append(chain)
    mapping, mapping_meta = TS.load_mapping_as_of(date)
    records = []
    for row in audit:
        routes, linked = route_ids(row, by_symbol.get(row["symbol"], []), date)
        row["route_ids"] = "|".join(routes)
        if not routes:
            row["exclusion"] = row["data_status"] if row["data_status"] != "FRESH" else "NO_RECENT_ROUTE_EVIDENCE_OR_INELIGIBLE"
            continue
        result = assess_routes(row, policy["policy"], routes)
        identity = mapping.get(row["symbol"], {})
        theme = "|".join(dict.fromkeys(c["theme_label"] for c in linked)) or identity.get("industry", "未确认")
        path = {"R1": "IGNITION", "R2": "LURK", "R3": "PROPAGATION"}[routes[0]]
        records.append({**row, "as_of": date, "name": identity.get("name", row["symbol"]),
            "route_id": routes[0], "route_label": RE.ROUTE_LABELS[routes[0]], "route_status": result["status"],
            "route_status_label": STATUS_LABELS[result["status"]], "route_policy": policy["policy"],
            "route_requirements_met": result["route_requirements_met"],
            "independent_price_evidence": result["independent_price_evidence"],
            "strict_chain_verified": False,
            "chain_source_grade": "DECLARED_ASOF_ONLY" if linked else "NOT_APPLICABLE",
            "confirmation_pass": result["confirmation_pass"],
            "route_risks": "|".join(result["risks"]), "route_source": "DATED_CHAIN_AND_PRICE" if linked else "PRICE_RECONSTRUCTION_TODAY",
            "route_theme_id": "|".join(c["theme_id"] for c in linked),
            "discovery_path": path, "discovery_path_label": RE.ROUTE_LABELS[routes[0]],
            "belonging_theme": theme, "related_themes": theme, "theme_source": "DATED_CHAIN" if linked else "IDENTITY_ONLY_NOT_CATALYST",
            "theme_lifecycle": linked[0]["theme_lifecycle"] if linked else "",
            "stock_lifecycle": "COILED" if path == "LURK" else "IGNITION",
            "stock_lifecycle_label": "等待生命周期复核", "ema20_bias": row["ema20_bias"],
            "r5": row["ret5"] * 100, "r20": row["ret20"] * 100,
            "action": result["status"], "note": "因果量价代理；未宣称历史生产命中；无自动交易",
            "chain_validation": "DATED_ARCHIVE_SMALL_SAMPLE" if linked else "NOT_APPLICABLE"})
    records.sort(key=lambda r: (STATUS_ORDER[r["route_status"]], abs(r["bias_atr"]), r["symbol"]))
    for i, row in enumerate(records, 1):
        row["rank"] = i
    output.mkdir(parents=True, exist_ok=True)
    columns = ["symbol", "as_of", "route_id", "route_ids", "route_status", "rank"]
    frame = pd.DataFrame(records) if records else pd.DataFrame(columns=columns)
    frame.to_csv(output / "route_candidates.csv", index=False, encoding="utf-8-sig")
    publish(output / "route_candidates.csv", date)
    pd.DataFrame(audit).to_csv(output / "route_scan_audit.csv", index=False, encoding="utf-8-sig")
    publish(output / "route_scan_audit.csv", date)
    summary = {"as_of": date, "version": RE.VERSION, "policy": policy,
               "universe": len(paths), "fresh": sum(r["data_status"] == "FRESH" for r in audit),
               "errors": [r for r in audit if r["data_status"] == "ERROR"],
               "cache_hits": sum(r["cache_hit"] for r in audit), "candidates": len(records),
               "routes": {r: sum(r in row["route_ids"].split("|") for row in records) for r in RE.ROUTE_LABELS},
               "overlap_stocks": sum(len(r["route_ids"].split("|")) > 1 for r in records),
               "extra_route_memberships": sum(len(r["route_ids"].split("|")) - 1 for r in records),
               "data_status_counts": pd.Series([r["data_status"] for r in audit]).value_counts().to_dict(),
               "freshness_denominator": "CURRENT_LOCAL_UNIVERSE_NOT_EXPECTED_TRADABLE_UNIVERSE",
               "preparation_origins": frame.preparation_origin.fillna("UNKNOWN").replace("", "UNKNOWN").value_counts().to_dict() if len(frame) else {},
               "statuses": frame.route_status.value_counts().to_dict(), "mapping": mapping_meta,
               "chain_archive_present": chain_path.exists(), "seconds": perf_counter() - started,
               "orders_enabled": False, "feature_hash": code_hash}
    # Shadow preparation watch: low-extension names the chosen detector flags today, outside any route or candidate state.
    watch = []
    if PD.CHOSEN:
        for row in audit:
            prep = row.get("preparation") or {}
            if row.get("data_status") == "FRESH" and row.get("eligible") and prep.get(PD.CHOSEN):
                identity = mapping.get(row["symbol"], {})
                watch.append({"symbol": row["symbol"], "name": identity.get("name", row["symbol"]), "industry": identity.get("industry", ""),
                              "detector": PD.CHOSEN, "bias_atr": row.get("bias_atr"), "excess20": prep.get("excess20"), "ret20": row.get("ret20"),
                              "above_ma20_now": prep.get("above_ma20_now"), "reclaim_failed": prep.get("reclaim_failed"),
                              "breakout_status": row.get("breakout_status"), "route_ids": row.get("route_ids", ""), "as_of": date,
                              "basis": "SHADOW_PREPARATION_NOT_A_CANDIDATE_STATE"})
        watch.sort(key=lambda r: (-(r["excess20"] if r["excess20"] is not None else -999.0), r["symbol"]))
    watch_columns = ["symbol", "name", "industry", "detector", "bias_atr", "excess20", "ret20", "above_ma20_now", "reclaim_failed", "breakout_status", "route_ids", "as_of", "basis"]
    pd.DataFrame(watch, columns=watch_columns).to_csv(output / "preparation_watch.csv", index=False, encoding="utf-8-sig")
    publish(output / "preparation_watch.csv", date)
    summary["preparation_watch"] = {"detector": PD.CHOSEN, "count": len(watch), "version": PD.VERSION}
    _write(output / "route_scan_summary.json", summary)
    fields = ["rank", "symbol", "name", "route_ids", "belonging_theme", "route_status_label", "preparation_first_seen", "preparation_origin", "probe_at",
              "breakout_attempt_at", "breakout_accepted_at", "bias_atr", "route_risks"]
    lines = [f"# 三路径证据扫描 {date}", "", f"- 全域 {len(paths)}，数据同日 {summary['fresh']}，候选 {len(records)}。",
             f"- {summary['routes']}；允许多路线命中，不重复计算股票数量。",
             "- R1=明确个股点火反查；R2=个股先行准备；R3=紧链传播承接。",
             "- 量价准备不是主力身份；偏离过大不是派发；确认复核不等于买入指令。",
             "- 同日数据率以本地股票域为分母，不是应交易股票覆盖率；缺日不自动解释为停牌。",
             "- 有日期的链线索保留观察；缺少可核验发布时间和关系证据，不授予严格传播介入资格。",
             "- 全部去留见route_scan_audit.csv；下表只展示前15，完整候选交给分析台。", "",
             frame[fields].head(15).to_markdown(index=False) if len(frame) else "没有满足本轮条件的候选。"]
    (output / "route_scan.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[route_scan] candidates={len(records)} routes={summary['routes']} cache={summary['cache_hits']}/{len(paths)}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = run(str(pd.Timestamp(args.date).date()), args.workers)
    raise SystemExit(1 if result["errors"] else 0)
