"""Fast, provisional late-session review, isolated from official daily state."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from functools import lru_cache
import json
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

import pandas as pd
from filelock import FileLock, Timeout

from . import data as D
from . import intraday_quotes as Q
from . import intraday_decisions as ID
from . import theme_semantics as TS
from . import virtual_portfolio as VP
from . import tail_observation as TO
from . import potential_pool as PP, tail_selection as SEL
from . import intraday_price as IP
from .pipeline import _write

VERSION = "intraday_review_v8_20260921"
REFERENCE_VERSION = "intraday_reference_v2_20260919"
ROOT = Path("results/intraday")
DAILY = Path("results/mainwave_engine")
NOTICE = "盘中暂定分析，非正式收盘信号；不自动买卖，不推进生产生命周期，不修改持仓。隔夜策略参数未验证。"


def read_json(path, default=None):
    if not Path(path).exists():
        return {} if default is None else default
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_positions(db=VP.DB_PATH):
    if not Path(db).exists():
        return []
    # Read-only connection: even table creation is forbidden during intraday work.
    with closing(sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        return [json.loads(r[0]) for r in conn.execute("SELECT payload FROM positions WHERE status='OPEN'")]


def read_history(symbol, as_of, data_root=Path("data")):
    path = data_root / f"{symbol}_daily.csv"
    info = path.stat()
    return _cached_history(str(path.resolve()), info.st_mtime_ns, info.st_size, as_of)


@lru_cache(maxsize=6000)
def _cached_history(path, modified, size, as_of):
    frame = pd.read_csv(path, usecols=lambda c: c in D.COLS)
    frame["date"] = pd.to_datetime(frame.date)
    return frame[frame.date.lt(as_of)].sort_values("date").drop_duplicates("date", keep="last").tail(320).reset_index(drop=True)


def reference_features(history, previous_day, symbol=None):
    """Small, dated sufficient statistics for the next session's price updates."""
    if (len(history) < 70 or not {"date", "close", "high", "pct_chg", "volume"}.issubset(history.columns)
            or history.date.iloc[-1].strftime("%Y-%m-%d") != previous_day):
        return {"status": "HISTORY_WAIT"}
    numeric = history[["close", "high", "pct_chg", "volume"]].apply(pd.to_numeric, errors="coerce")
    if (numeric.map(lambda v: Q.number(v) is None).any().any() or numeric[["close", "high"]].le(0).any().any()
            or numeric.volume.lt(0).any() or numeric.high.lt(numeric.close).any()):
        return {"status": "HISTORY_WAIT"}
    previous = float(history.close.iloc[-1])
    returns = pd.to_numeric(history.pct_chg, errors="coerce") / 100
    if returns.isna().any() or returns.le(-1).any():
        return {"status": "HISTORY_WAIT"}
    index = (1 + returns).cumprod() * 100
    scale = previous / index.iloc[-1]
    return {"status": "READY", "as_of": previous_day, "close": previous,
            "preparation": PP.feature_seed(history, symbol, previous_day) if symbol else {},
            "ema10": float(index.ewm(span=10, adjust=False).mean().iloc[-1] * scale),
            "ema20": float(index.ewm(span=20, adjust=False).mean().iloc[-1] * scale),
            "ema60": float(index.ewm(span=60, adjust=False).mean().iloc[-1] * scale),
            "high20": float((numeric.high * index / numeric.close).tail(20).max() * scale),
            "volume5": float(numeric.volume.tail(5).mean())}


def quick_metrics(history, quote, previous_day, basic=None, reference_price=None):
    """Revalue daily EMAs from frozen closing features, not a mutable partial bar."""
    seed = reference_features(history, previous_day) if isinstance(history, pd.DataFrame) else history
    if seed.get("status") != "READY" or seed.get("as_of") != previous_day:
        return {"status": "HISTORY_WAIT"}
    previous = seed["close"]
    preclose = quote.get("pre_close")
    # A minute-only quote cannot establish today's ex-right reference price.
    reference = preclose or reference_price or previous
    scale = reference / previous
    result = {"status": "PROVISIONAL" if preclose else "PRICE_REFERENCE_VERIFIED" if reference_price else "PRICE_REFERENCE_UNVERIFIED",
              "previous_date": previous_day, "pct_chg": (quote["close"] / reference - 1) * 100,
              "reference_scale": scale,
              "ema20": seed["ema20"] * scale * (19 / 21) + quote["close"] * (2 / 21),
              "ema60": seed["ema60"] * scale * (59 / 61) + quote["close"] * (2 / 61),
              "volume_ratio_same_time": None, "turnover_pct_estimate": None, "market_cap_yi_estimate": None}
    result["ema20_bias_pct"] = (quote["close"] / result["ema20"] - 1) * 100
    result["ema10"] = seed["ema10"] * scale * (9 / 11) + quote["close"] * (2 / 11) if seed.get("ema10") else None
    result["above_prior20_high"] = bool(quote["close"] > seed["high20"] * scale)
    result["vwap_bias_pct"] = ((quote["close"] / quote["vwap"] - 1) * 100) if quote.get("vwap") else None
    volume = quote.get("volume")
    mean_volume = seed["volume5"]
    result["volume_vs_full_day5"] = volume / mean_volume if volume is not None and mean_volume > 0 else None
    if basic and str(basic.get("trade_date")) == previous_day.replace("-", ""):
        shares = Q.number(basic.get("float_share"))
        total = Q.number(basic.get("total_share"))
        if volume is not None and shares and shares > 0:
            result["turnover_pct_estimate"] = volume / (shares * 10000) * 100
        if total and total > 0:
            result["market_cap_yi_estimate"] = quote["close"] * total / 10000
        result["shares_as_of"] = previous_day
    return result


def prepare_reference(as_of, symbols=None, root=None, daily_root=None):
    root = ROOT if root is None else root
    directory = root / "reference"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(directory / f"prepare_{as_of.replace('-', '')}.lock", timeout=0):
            return _prepare_reference(as_of, symbols, root, daily_root)
    except Timeout as exc:
        raise ValueError("该日收盘基线与潜在池正在构建，请勿重复启动") from exc


def source_fingerprint(symbol):
    try:
        stat = (Path("data") / f"{symbol}_daily.csv").stat()
        return [stat.st_mtime_ns, stat.st_size]
    except OSError:
        return None


@lru_cache(maxsize=6000)
def _reference_from_source(symbol, as_of, fingerprint):
    cutoff = (pd.Timestamp(as_of) + pd.Timedelta(days=1)).date().isoformat()
    try:
        return reference_features(read_history(symbol, cutoff), as_of, symbol)
    except (OSError, ValueError, KeyError):
        return {"status": "HISTORY_WAIT"}


def load_reference(symbol, as_of, cached=None):
    before = source_fingerprint(symbol)
    if cached and "input_fingerprint" in cached and cached["input_fingerprint"] == before:
        return cached
    result = _reference_from_source(symbol, as_of, tuple(before) if before else None)
    if source_fingerprint(symbol) != before:
        return {"status": "HISTORY_WAIT"}
    return {**result, "input_fingerprint": before}


def _prepare_reference(as_of, symbols, root, daily_root):
    daily_root = DAILY if daily_root is None else daily_root
    folder = daily_root / as_of.replace("-", "")
    status = read_json(folder / "mainwave_run_status.json")
    now = Q.now_china()
    if as_of > now.date().isoformat() or (as_of == now.date().isoformat() and now.hour < 15):
        raise ValueError("历史指标缓存只能使用已收盘日期")
    if not (folder / "_SUCCESS").exists() or status.get("as_of") != as_of or status.get("status") != "SUCCESS":
        raise ValueError("历史指标缓存需要该日正式流水线成功，不能从未完成的日报建立基线")
    symbols = symbols if symbols is not None else sorted(set(D.list_universe()) | {p["symbol"] for p in read_positions()})
    records = {}
    for index, symbol in enumerate(symbols, 1):
        records[symbol] = load_reference(symbol, as_of)
        if index % 500 == 0:
            print(f"[intraday_reference] {index}/{len(symbols)}", flush=True)
    payload = {"as_of": as_of, "version": REFERENCE_VERSION, "source_finished_at": status.get("finished_at"), "records": records,
               "ready": sum(r["status"] == "READY" for r in records.values())}
    _write(root / "reference" / f"features_{as_of.replace('-', '')}.json", payload)
    mapping, meta = TS.load_mapping_as_of(as_of)
    pool = PP.publish({s: r.get("preparation", {}) for s, r in records.items()}, as_of, root, mapping)
    payload["potential_pool_summary"] = pool["summary"]
    print(f"[intraday_reference] {as_of} ready={payload['ready']}/{len(records)}", flush=True)
    return payload


def market_review(quotes, symbols, indices, previous, scope="market"):
    valid = [quotes[Q.ts_code(s)] for s in symbols if Q.ts_code(s) in quotes and quotes[Q.ts_code(s)].get("pre_close")]
    coverage = len(valid) / len(symbols) if symbols else 0
    up_ratio = sum(r["close"] > r["pre_close"] for r in valid) / len(valid) if valid else None
    changes = {code: (r["close"] / r["pre_close"] - 1) * 100 for code, r in indices.items() if r.get("pre_close")}
    complete = scope == "market" and coverage >= .9 and all(c in changes for c in Q.INDEX_CODES)
    known = previous.get("data_status") == "FRESH"
    state = "UNKNOWN"
    severe = sum(v <= -1.5 for v in changes.values()) >= 2 or (scope == "market" and coverage >= .9 and up_ratio < .25)
    if severe:
        state = "RISK"
    elif complete and known:
        state = "RISK" if sum(v <= -1.5 for v in changes.values()) >= 2 or up_ratio < .25 else (
            "RESTRICTED" if not previous.get("trend_review_allowed", False) else "REVIEW_ONLY")
    details = []
    for code in Q.INDEX_CODES:
        q = indices.get(code, {})
        details.append({"code": code, "name": ID.INDEX_NAMES[code], "pct_chg": changes.get(code),
                        "drawdown_pct": (q["close"] / q["high"] - 1) * 100 if q.get("high") else None,
                        "observed_at": q.get("observed_at")})
    return {"state": state, "snapshot_coverage": coverage, "advance_ratio": up_ratio, "indices": details,
            "data_complete": complete and known,
            "index_changes_pct": changes, "previous_close_context": previous,
            "new_review_allowed": state == "REVIEW_ONLY",
            "method": "前收盘环境作为先验；实时四指数与全域涨跌宽度只作加严风险检查，不凭尾盘反弹把熊市改成牛市。",
            "threshold_status": "90%覆盖为工程门槛；-1.5%/25%为未校准风险观察阈值，不是成熟择时模型。"}


def holding_review(position, prior_observation, quote, metrics):
    previous_status = prior_observation.get("status", "DATA_WAIT")
    result = {"symbol": position["symbol"], "name": position.get("name", ""), "status": "DATA_WAIT",
              "prior_status": previous_status, "reason": "缺少新鲜行情或上一收盘日风险边界", "orders_enabled": False}
    if quote:
        result.update(latest_price=quote["close"], observed_at=quote["observed_at"], quote_quality=quote["quality"])
        if quote["quality"] == "PRICE_ONLY":
            result["reason"] = ("价格参照已核验，边界距离见价格级评估；缺少全天轨迹，完整持仓决策待确认"
                                if metrics.get("status") == "PRICE_REFERENCE_VERIFIED" else
                                "分钟参考价已取得；缺少当日累计快照及价格参照核验，不能确认风险边界")
    if not quote or metrics.get("status") != "PROVISIONAL" or not prior_observation.get("hard_floor"):
        return result
    factor = metrics["reference_scale"]
    hard = prior_observation["hard_floor"] * factor
    soft = (prior_observation.get("soft_floor") or prior_observation["hard_floor"]) * factor
    today = Q.stamp(quote["observed_at"]).date().isoformat()
    can_sell = position["entry_date"] < today
    hard_touched = min(quote["close"], quote.get("low") or quote["close"]) <= hard
    status = ("SELL_REVIEW" if previous_status == "SELL" or hard_touched else
              "REDUCE_REVIEW" if previous_status == "REDUCE" else
              "WAIT_CLOSE_CONFIRMATION" if quote["close"] <= soft else "HOLD_OBSERVATION")
    reason = {"SELL_REVIEW": "已有退出警报或盘中触及固定硬底线，需人工核对可卖数量与成交条件",
              "REDUCE_REVIEW": "此前分批保护信号仍未处置，不能被一次尾盘反弹清除",
              "WAIT_CLOSE_CONFIRMATION": "盘中触及收盘保护位，不把一次报价计为一天收盘失守",
              "HOLD_OBSERVATION": "当前未新增边界警报，不代表没有隔夜跳空风险"}[status]
    return {**result, "status": status if can_sell else "T1_LOCKED", "reason": reason,
            "pending_action": status, "can_sell_by_date": can_sell, "latest_price": quote["close"],
            "observed_at": quote["observed_at"], "hard_floor": hard, "soft_floor": soft,
            "reference_date": metrics["previous_date"]}


def evaluate(bundle, symbols, previous_day, baseline, positions, prior_positions, histories, basics, at,
             membership=None, scope="market"):
    references = IP.reference_prices(bundle.get("tail_sources", {}), Q.stamp(at).date().isoformat(), at)
    quotes, indices, minutes, rejected = {}, {}, {}, []
    for batch in bundle["raw"]:
        api = batch["api"]
        frame = pd.DataFrame(batch["records"])
        if api == "rt_min_daily":
            if not frame.empty:
                code = str(frame.iloc[0].get("ts_code", frame.iloc[0].get("code", "")))
                minutes[code] = Q.minute_context(frame, code, at)
            continue
        good, bad = Q.normalize(frame, api, at)
        allowed = set(Q.INDEX_CODES) if api == "rt_idx_k" else {Q.ts_code(s) for s in symbols}
        (indices if api == "rt_idx_k" else quotes).update({code: q for code, q in good.items() if code in allowed})
        rejected.extend(bad)
    market = market_review(quotes, symbols, indices, baseline.get("market", {}), scope)
    if scope != "market":
        market.update(state="RISK" if market["state"] == "RISK" else "UNKNOWN", new_review_allowed=False,
                      scope_notice="仅池内样本，不能替代全市场宽度；不放开市场许可。")
    entry_window = "14:30" <= Q.stamp(at).strftime("%H:%M") < "14:57"
    market["entry_window_open"] = entry_window
    market["new_review_allowed"] = market["new_review_allowed"] and entry_window
    base = {r["symbol"]: r for r in baseline.get("records", [])}
    records, computed = [], {}
    for symbol in symbols:
        quote = quotes.get(Q.ts_code(symbol))
        old = base.get(symbol, {})
        metrics = quick_metrics(histories.get(symbol, pd.DataFrame()), quote, previous_day,
                                basics.get(Q.ts_code(symbol)), Q.number(references.get(Q.ts_code(symbol), {}).get("pre_close"))) if quote else {"status": "QUOTE_WAIT"}
        computed[symbol] = metrics
        minute = minutes.get(Q.ts_code(symbol), {"status": "NOT_FETCHED", "late_pattern": "WAIT_MINUTES"})
        status = "DATA_WAIT"
        risks = []
        if old.get("value_tier") == "RISK_WATCH" or old.get("structural_advice") == "RISK_WATCH":
            risks.append("前收盘结构风险尚未解除")
        if metrics["status"] == "PROVISIONAL":
            factor = metrics["reference_scale"]
            lower, upper, invalid = (Q.number(old.get(k)) for k in ("review_lower", "review_upper", "invalid"))
            within_window = bool(lower and upper and lower * factor <= quote["close"] <= upper * factor)
            if invalid and quote["close"] <= invalid * factor:
                risks.append("当前价格跌破已知结构失效边界")
            if quote.get("vwap") and quote["close"] < quote["vwap"]:
                risks.append("当前价格失守累计VWAP")
            if old.get("value_tier") in {"DATA_WAIT", "HISTORICAL_ONLY"}:
                status = "DATA_WAIT"
            elif not market["new_review_allowed"]:
                status = "MARKET_WAIT"
            elif not old:
                status = "NEW_PRICE_DISCOVERY" if metrics["above_prior20_high"] else "WATCH"
            elif not within_window:
                status = "WAIT_PRICE"
            else:
                status = "PROVISIONAL_REVIEW" if old.get("structural_advice") == "PRIORITY_REVIEW" and minute.get("status") == "COMPLETE" and quote["close"] >= metrics["ema20"] else "WAIT_CONFIRMATION"
        if risks:
            status = "RISK_WATCH"
        records.append({"symbol": symbol, "name": old.get("name", ""), "route_ids": old.get("route_ids", ""),
                        "preparation": histories[symbol].get("preparation", {}) if isinstance(histories.get(symbol), dict) else {},
                        "theme": old.get("belonging_theme", ""), "prior_lifecycle": old.get("stock_lifecycle_label", ""),
                        "lifecycle_as_of": previous_day, "prior_advice": old.get("structural_advice", ""),
                        "prior_rank": Q.number(old.get("value_rank")), "status": status, "risk_flags": risks, "quote": quote, "metrics": metrics,
                        "new_price_breakout": bool(not old and metrics.get("above_prior20_high", False)),
                        "minute_context": minute, "orders_enabled": False})
        for key in ("review_lower", "review_upper", "invalid"):
            value = Q.number(old.get(key))
            records[-1][key] = value * metrics["reference_scale"] if value is not None and metrics.get("status") in {"PROVISIONAL", "PRICE_REFERENCE_VERIFIED"} else None
    order = {s: i for i, s in enumerate(("PROVISIONAL_REVIEW", "WAIT_CONFIRMATION", "WAIT_PRICE", "NEW_PRICE_DISCOVERY", "MARKET_WAIT", "WATCH", "RISK_WATCH", "DATA_WAIT"))}
    records.sort(key=lambda r: (order[r["status"]], r["prior_rank"] or 100000, r["symbol"]))
    holdings = [holding_review(p, prior_positions.get(p["id"], {}), quotes.get(Q.ts_code(p["symbol"])),
                               computed.get(p["symbol"], {})) for p in positions]
    minute_required = bundle.get("minute_symbols", [])
    complete_minutes = sum(minutes.get(Q.ts_code(s), {}).get("status") == "COMPLETE" for s in minute_required)
    partial = (scope != "market" or not market["data_complete"] or bool(bundle["issues"]) or complete_minutes < len(minute_required)
               or any(r["status"] == "DATA_WAIT" for r in holdings))
    report = {"version": VERSION, "scope": scope, "as_of": Q.stamp(at).date().isoformat(), "observed_at": Q.stamp(at).isoformat(),
            "mode": "INTRADAY_PROVISIONAL", "notice": NOTICE, "orders_enabled": False, "market": market,
            "quote_collection": bundle.get("quote_collection", {}),
            "summary": {"requested": len(symbols), "fresh_quotes": sum(Q.ts_code(s) in quotes for s in symbols),
                        "full_snapshots": sum(r["quality"] == "DAILY_SNAPSHOT" for r in quotes.values()),
                        "holdings": len(holdings), "new_price_discoveries": sum(r["new_price_breakout"] for r in records),
                        "complete_minutes": complete_minutes, "required_minutes": len(minute_required),
                        "history_ready": sum(m.get("status") in {"PROVISIONAL", "PRICE_REFERENCE_VERIFIED", "PRICE_REFERENCE_UNVERIFIED"} for m in computed.values()),
                        "status": "PARTIAL" if partial else "PROVISIONAL"},
            "records": records, "holdings": holdings, "rejected": rejected, "issues": bundle["issues"],
            "ranking_method": "已有正式结构优先，盘中仅作证据复核；同类沿用上一收盘排序，不是预期收益排序。新价格突破未获三路径资格。"}
    if membership is not None:
        for row in records:
            item = membership.get("mapping", {}).get(row["symbol"], {})
            row["name"] = row["name"] or item.get("name", "")
            row["theme"] = row["theme"] or " / ".join(item.get("concepts", [])[:3])
        ID.enrich(report, membership)
    if "tail_sources" in bundle:
        report["tail_observation"] = TO.build(report, bundle["tail_sources"])
    report["tail_selection"] = SEL.build(report, bundle.get("potential_pool", {}), previous_day)
    report["price_review"] = IP.build(report, references, bundle.get("tail_sources", {}), positions, prior_positions, membership)
    return report


def publish(bundle, report, root=ROOT):
    folder = root / report["as_of"].replace("-", "")
    run_id = Q.stamp(report["observed_at"]).strftime("%H%M%S") + "_" + uuid4().hex[:8]
    output = folder / run_id
    _write(output / "source.json", bundle)
    _write(output / "review.json", report)
    lines = [f"# 尾盘快速复核 {report['observed_at']}", "", NOTICE,
             f"- 生成：{report.get('generated_at', report['observed_at'])}；报价有效期至：{report.get('valid_until', '待核验')}。过期必须刷新。",
             f"- 市场：{report['market']['state']}；快照覆盖：{report['market']['snapshot_coverage']:.1%}。",
             f"- 请求 {report['summary']['requested']}；新鲜报价 {report['summary']['fresh_quotes']}；完整日内快照 {report['summary']['full_snapshots']}。",
             "- 价格级降级不计算累计量、换手或全天VWAP；上一日生命周期不是今日已确认阶段。",
             "- 同时点历史量比暂缺；volume_vs_full_day5 是累计量/过去5日全天均量，不能当作量比。",
             ""]
    sweep = report.get("quote_collection", {})
    if sweep:
        lines += [f"- 股票批次采集：{sweep['status']}；已返回 {sweep['received']}/{sweep['requested']}；"
                  f"缺失 {len(sweep['missing_codes'])}；结束原因 {sweep['stop_reason']}。",
                  "- 已返回只表示本轮收到过该股票记录；是否仍新鲜、字段是否完整以上方数据检查为准。", ""]
    if report.get("price_review"):
        lines += IP.report_lines(report["price_review"])
    lines += ID.report_lines(report)
    if "tail_observation" in report:
        lines += TO.report_lines(report["tail_observation"])
    if "tail_selection" in report:
        lines += SEL.report_lines(report["tail_selection"])
    lines += ["", "## 数据问题", *[f"- {issue}" for issue in report["issues"]], "", report["ranking_method"]]
    (output / "review.md").write_text("\n".join(lines), encoding="utf-8")
    _write(folder / "latest.json", {"as_of": report["as_of"], "run_id": run_id, "mode": report["mode"],
                                   "review": str(output / "review.json"), "observed_at": report["observed_at"]})
    return output


def check_expiry(report, at):
    at = Q.stamp(at)
    quotes = [r["quote"] for r in report["records"] if r.get("quote")]
    timestamps = [q["observed_at"] for q in quotes] + [r["observed_at"] for r in report["market"].get("indices", []) if r.get("observed_at")]
    timestamps += [r["minute_context"]["observed_at"] for r in report["records"] if r.get("minute_context", {}).get("status") == "COMPLETE"]
    expires = min((Q.stamp(t) + pd.Timedelta(seconds=120) for t in timestamps), default=at)
    report.update(generated_at=at.isoformat(), valid_until=expires.isoformat())
    if not "14:30" <= at.strftime("%H:%M") < "14:57":
        report["market"].update(entry_window_open=False, new_review_allowed=False)
        for r in report["records"]:
            if r["status"] in {"PROVISIONAL_REVIEW", "NEW_PRICE_DISCOVERY"}:
                r["status"] = "MARKET_WAIT"
    if at >= expires:
        if "price_review" in report:
            report["price_review"]["mode"] = "STALE"
        if "tail_selection" in report:
            SEL.expire(report["tail_selection"])
        if "tail_observation" in report:
            report["tail_observation"]["mode"] = "STALE"
        report["summary"]["status"] = "EXPIRED" if quotes else "PARTIAL"
        report["market"].update(state="UNKNOWN", new_review_allowed=False)
        for r in report["records"]:
            r["status"] = "DATA_WAIT"
        for r in report["holdings"]:
            r.update(status="DATA_WAIT", reason="生成报告时行情已过有效期；此前风险警报仍需核对")
        if "themes" in report:
            report["themes"]["status"] = "DATA_WAIT"
            for t in report["themes"]["records"]:
                t["state"] = "DATA_WAIT"
        if "candidates" in report:
            report["candidates"]["focus_symbols"] = []
            report["summary"].update(focus=0, hot_themes=0)
    if not report["market"]["new_review_allowed"] and "candidates" in report:
        report["candidates"].update(paper_review_symbols=[], eligible_count=0)
        report["summary"]["paper_reviews"] = 0
        for r in report["records"]:
            r["paper_review_eligible"] = False
    if not report["market"]["new_review_allowed"] and "tail_selection" in report:
        selection = report["tail_selection"]
        selection["paper_review_symbols"] = []
        selection["summary"]["reviews"] = 0
        for row in selection["records"]:
            row["paper_review_eligible"] = False
            if row["status"] == "SETUP_REVIEW":
                row["status"] = "MARKET_WAIT"
    if "holding_actions" in report["summary"]:
        report["summary"]["holding_actions"] = dict(Counter(r["status"] for r in report["holdings"]))
    return report


def run(as_of, scope="market", client=None):
    at = Q.now_china()
    if as_of != at.date().isoformat() or not Q.in_collection_window(at):
        raise ValueError(f"行情采集仅运行于北京时间当日{Q.COLLECTION_START}-11:30、13:00-{Q.COLLECTION_END}（午间休市无新鲜报价）；"
                         "历史日期不能调用实时接口伪造回放。权限检查请用 --probe。")
    if at.strftime("%H:%M") >= "15:00":
        print("[intraday] 收盘后补采：报价可能停留在15:00，仍检查有效期，不放开盘中新参与复核；正式结论以收盘任务为准。", flush=True)
    client = client or Q.Client()
    calendar = client.query("trade_cal", exchange="SSE", start_date=(at - pd.Timedelta(days=20)).strftime("%Y%m%d"), end_date=at.strftime("%Y%m%d"))
    sessions = sorted(calendar.loc[calendar.is_open.eq(1), "cal_date"].astype(str))
    compact = as_of.replace("-", "")
    if compact not in sessions or len(sessions) < 2:
        raise ValueError("交易日历未确认今日开市或缺少前一交易日")
    previous_day = pd.Timestamp(sessions[-2]).date().isoformat()
    pool = PP.load(ROOT, previous_day)
    baseline = read_json(DAILY / sessions[-2] / "analysis_universe.json")
    if baseline.get("as_of") != previous_day or not (DAILY / sessions[-2] / "_SUCCESS").exists():
        baseline = {}
    positions = [p for p in read_positions() if p["entry_date"] <= as_of]
    paper = read_json(DAILY / sessions[-2] / "virtual_positions.json")
    prior = {r["id"]: r.get("observation", {}) for r in paper.get("records", [])} if paper.get("as_of") == previous_day else {}
    for p in positions:
        if p["id"] not in prior and p["entry_date"] <= previous_day:
            try:
                prior[p["id"]] = VP.scan_position(p, previous_day)
            except (OSError, ValueError, KeyError):
                prior[p["id"]] = {}
    pool_symbols = [r["symbol"] for r in pool.get("records", []) if r["state"] == "POTENTIAL"]
    symbols = sorted(set(D.list_universe() if scope == "market" else [r["symbol"] for r in baseline.get("records", [])] + pool_symbols) | {p["symbol"] for p in positions})
    if not symbols:
        raise ValueError("没有可扫描股票或持仓")
    focus = list(dict.fromkeys([p["symbol"] for p in positions] + [r["symbol"] for r in baseline.get("records", [])[:15]]))
    print(f"[intraday] scope={scope} symbols={len(symbols)} minute_focus={len(focus)} previous_close={previous_day}", flush=True)
    basics = {}
    basic_path = ROOT / "reference" / f"daily_basic_{sessions[-2]}.json"
    basic_records = read_json(basic_path, [])
    if not basic_records:
        try:
            basic_records = json.loads(client.query("daily_basic", trade_date=sessions[-2], fields="ts_code,trade_date,float_share,total_share").to_json(orient="records"))
            _write(basic_path, basic_records)
        except Q.ApiError:
            pass
    basics = {r["ts_code"]: r for r in basic_records}
    cached = read_json(ROOT / "reference" / f"features_{sessions[-2]}.json")
    prior_run = read_json(DAILY / sessions[-2] / "mainwave_run_status.json")
    cache_valid = (cached.get("as_of") == previous_day and cached.get("version") == REFERENCE_VERSION
                   and cached.get("source_finished_at") == prior_run.get("finished_at"))
    histories = cached.get("records", {}).copy() if cache_valid else {}
    for index, symbol in enumerate(symbols, 1):
        histories[symbol] = load_reference(symbol, previous_day, histories.get(symbol))
        if index % 1000 == 0:
            print(f"[intraday] reference {index}/{len(symbols)} cache={cache_valid}", flush=True)
    # Closing preparation is the only publisher; live runs must not overwrite it.
    mapping, meta = TS.load_mapping_as_of(previous_day)
    membership = {"mapping": {s: {"name": item.get("name", ""), "concepts": item.get("concepts", [])}
                              for s, item in mapping.items() if s in histories},
                  "meta": {**meta, "as_of": previous_day}}
    tail_sources = {"previous_day": previous_day,
                    "boards": TO.fetch_source(client, "limit_list_d", previous_day, ROOT),
                    "limits": TO.fetch_source(client, "stk_limit", as_of, ROOT),
                    "leaders": TO.leader_source(DAILY, previous_day, Q.now_china())}
    def select_minutes(raw, observed):
        preview = evaluate({"raw": raw, "issues": [], "tail_sources": tail_sources, "potential_pool": pool}, symbols, previous_day, baseline, [], {}, histories,
                           basics, observed, membership, scope)
        ranked = [r["symbol"] for r in preview["records"] if r["status"] not in {"DATA_WAIT", "RISK_WATCH", "WATCH"}
                  and (r["prior_advice"] == "PRIORITY_REVIEW" or r["new_price_breakout"])][:15]
        tactical = [r["symbol"] for r in preview["tail_selection"]["records"]
                    if r["status"] not in {"DATA_WAIT", "RISK_WATCH"}
                    and (r["setup"]["ignition"] or r["setup"]["pullback"] or r["board_state"])]
        return list(dict.fromkeys([p["symbol"] for p in positions] + tactical[:10] + ranked))[:20]
    # Warm historical features before fetching time-sensitive quotes.
    bundle = Q.collect(client, symbols, focus, Q.now_china(), minute_selector=select_minutes,
                       priority_symbols=[p["symbol"] for p in positions] + pool_symbols[:30])
    bundle["potential_pool"] = pool
    bundle["theme_membership"] = membership
    bundle["tail_sources"] = tail_sources
    if not basics:
        bundle["issues"].append("上一日股本数据缺失：换手和市值估计不可用")
    at = Q.now_china()
    report = evaluate(bundle, symbols, previous_day, baseline, positions, prior, histories, basics, at, membership, scope)
    report["reference_day"] = previous_day
    report["universe_notice"] = "本地在册非ST沪深主板、创业板、科创板股票域；不宣称覆盖北交所、未入库新股或所有上市股票。"
    check_expiry(report, Q.now_china())
    output = publish(bundle, report)
    print(f"[intraday] {report['summary']} market={report['market']['state']} -> {output / 'review.md'}", flush=True)
    price = report["price_review"]
    print(f"[price_review] coverage={price['market']['covered']}/{price['market']['requested']} "
          f"reading={price['market']['reading']} candidates={len(price['candidates'])} "
          f"mode={price['mode']} (price observations only, no entry permission)", flush=True)
    for line in ID.report_lines(report)[-5:]:
        print(line, flush=True)
    print(f"[tail_observation] {report['tail_observation']['summary']} (shadow only)", flush=True)
    return report


def watch(as_of, scope="market"):
    """A manually started, bounded session; never installs a background schedule."""
    at = Q.now_china()
    if at.date().isoformat() != as_of or not Q.in_collection_window(at):
        raise ValueError(f"请在当日{Q.COLLECTION_START}-11:30 或 13:00-{Q.COLLECTION_END} 启动采集循环；不会在后台等待明天或自动下单。")
    client, code = Q.Client(), 0
    while Q.now_china().date().isoformat() == as_of and Q.now_china().strftime("%H:%M") < Q.COLLECTION_END:
        if not Q.in_collection_window(Q.now_china()):
            # Midday break: no fresh quotes; wait for the afternoon session without calling the API.
            time.sleep(60)
            continue
        started = time.monotonic()
        try:
            report = run(as_of, scope, client)
            code = 0 if report["summary"]["status"] == "PROVISIONAL" else 2
        except (ValueError, Q.ApiError) as exc:
            print(f"[intraday] cycle incomplete: {exc}", flush=True)
            code = 2
        remaining = (Q.stamp(as_of + " " + Q.COLLECTION_END) - Q.stamp(Q.now_china())).total_seconds()
        if remaining > 0:
            time.sleep(min(remaining, max(1, 60 - (time.monotonic() - started))))
    return code


def finalize(as_of, root=ROOT, daily_root=DAILY, data_root=Path("data"), at=None):
    """Official daily results take precedence; preserve every provisional observation."""
    at = Q.stamp(at or Q.now_china())
    if at.date().isoformat() < as_of or (at.date().isoformat() == as_of and at.hour < 15):
        raise ValueError("收盘前不能接管盘中结果")
    folder = root / as_of.replace("-", "")
    latest = read_json(folder / "latest.json")
    if not latest:
        return {"status": "NO_INTRADAY_RUN", "as_of": as_of}
    source = folder / latest["run_id"] / "review.json"
    report = read_json(source)
    output = daily_root / as_of.replace("-", "")
    run_status = read_json(output / "mainwave_run_status.json")
    board = read_json(output / "analysis_universe.json")
    paper = read_json(output / "virtual_positions.json")
    problems = []
    if (run_status.get("status") != "SUCCESS" or run_status.get("as_of") != as_of or board.get("as_of") != as_of
            or not (output / "_SUCCESS").exists()
            or not run_status.get("finished_at") or Q.stamp(run_status["finished_at"]) < Q.stamp(latest["observed_at"])):
        problems.append("正式流水线未在盘中快照之后成功完成，不能把旧日报标为正式接管")
    if board.get("summary", {}).get("status") != "SUCCESS" or board.get("market", {}).get("data_status") != "FRESH":
        problems.append("正式候选或市场分析尚未完整")
    if paper.get("as_of") != as_of or paper.get("summary", {}).get("status") != "SUCCESS":
        problems.append("正式持仓复核尚未完整")
    official = {r["symbol"]: r for r in board.get("records", [])}
    comparisons, suspended = [], []
    from .eod_review import calendar as market_calendar
    sessions = [d for d in market_calendar(data_root) if d < as_of]
    previous_session = sessions[-1] if sessions else None
    for row in report.get("records", []):
        quote = row.get("quote")
        if not quote:
            continue
        try:
            f = pd.read_csv(data_root / f"{row['symbol']}_daily.csv")
            last = str(f.date.iloc[-1])[:10] if len(f) else ""
            if previous_session and last and last < previous_session:
                # Two or more sessions without a bar: long suspended or delisted, so no close exists to reconcile.
                suspended.append({"symbol": row["symbol"], "last_bar": last})
                continue
            day = f[f.date.astype(str).eq(as_of)].iloc[-1]
            close = Q.number(day.close)
            if close is None or close <= 0:
                raise ValueError("invalid official close")
            comparisons.append({"symbol": row["symbol"], "intraday_price": quote["close"], "official_close": close,
                                "change_after_snapshot_pct": (close / quote["close"] - 1) * 100,
                                "intraday_status": row["status"], "official_advice": official.get(row["symbol"], {}).get("value_tier", "NOT_IN_DAILY_POOL")})
        except (OSError, ValueError, KeyError, IndexError):
            problems.append(f"{row['symbol']}: official close missing")
    result = {"as_of": as_of, "status": "PENDING_FINAL" if problems else "FINAL",
              "intraday_run": latest["run_id"], "official_review": str(output / "analysis_universe.json"),
              "official_positions": str(output / "virtual_positions.json"), "issues": problems, "comparisons": comparisons,
              "suspended_no_close": suspended,
              "updated_at": at.isoformat(), "notice": "正式结果接管当前结论；盘中原始快照保留，不据事后结果改写当时判断。"}
    _write(folder / "settlement.json", result)
    print(f"[intraday_settlement] {result['status']} compared={len(comparisons)} suspended={len(suspended)} issues={len(problems)}", flush=True)
    return result


def effective(as_of, root=ROOT, at=None):
    """Consumer contract: official report wins only after successful reconciliation."""
    folder = root / as_of.replace("-", "")
    latest, settled = read_json(folder / "latest.json"), read_json(folder / "settlement.json")
    if settled.get("status") == "FINAL" and settled.get("intraday_run") == latest.get("run_id"):
        return {"mode": "FINAL", "path": settled["official_review"]}
    if not latest:
        return {"mode": "UNAVAILABLE", "path": None}
    observed = Q.stamp(latest["observed_at"])
    current = Q.stamp(at or Q.now_china())
    report = read_json(folder / latest["run_id"] / "review.json")
    expires = Q.stamp(report.get("valid_until", observed + pd.Timedelta(seconds=120)))
    if current < observed or current >= expires:
        return {"mode": "STALE", "path": latest.get("review"), "observed_at": latest["observed_at"]}
    return {"mode": "INTRADAY_PROVISIONAL", "path": latest.get("review"), "observed_at": latest.get("observed_at")}


def dashboard(as_of, root=ROOT, at=None):
    """Read a compact view; opening the UI never fetches quotes or writes positions."""
    from datetime import date
    day = date.fromisoformat(as_of).isoformat()
    state = effective(day, root, at)
    folder = root / day.replace("-", "")
    latest = read_json(folder / "latest.json")
    report = read_json(folder / latest["run_id"] / "review.json") if latest else {}
    collection, price_samples = {}, []
    if report:
        quotes = [r for r in report["records"] if r.get("quote")]
        held = {r["symbol"] for r in report["holdings"]}
        full = sum(r["quote"].get("quality") == "DAILY_SNAPSHOT" for r in quotes)
        collection = {"requested": report["summary"]["requested"], "prices": len(quotes),
                      "price_only": len(quotes) - full, "full": full,
                      "missing": report["summary"]["requested"] - len(quotes),
                      "holding_prices": sum(r["symbol"] in held for r in quotes), "holdings": len(held),
                      "indices": sum(bool(r.get("observed_at")) for r in report["market"].get("indices", [])),
                      "rate_limited": any(Q.ApiError("", "", s).rate_limited for s in report.get("issues", [])),
                      "daily_quota_exhausted": any(Q.ApiError("", "", s).daily_limit for s in report.get("issues", []))}
        price_samples = sorted(quotes, key=lambda r: (r["symbol"] not in held, r["symbol"]))[:10]
        if state["mode"] != "INTRADAY_PROVISIONAL":
            check_expiry(report, at or Q.now_china())
        focus = set(report.get("candidates", {}).get("focus_symbols", []))
        report["records"] = [r for r in report["records"] if r["symbol"] in focus]
        if "themes" in report:
            report["themes"]["records"] = report["themes"].get("records", [])[:10]
    observation = report.get("tail_observation") if report else None
    if observation is None:
        observation = TO.offline(root, DAILY, day, at or Q.now_china())
    if state["mode"] == "FINAL":
        observation["mode"] = "STALE"
    pool = PP.latest(root, day, at or Q.now_china())
    selection = report.get("tail_selection", {})
    if state["mode"] != "INTRADAY_PROVISIONAL" and selection and selection.get("mode") != "STALE":
        SEL.expire(selection)
    return {"as_of": day, "mode": state["mode"], "report": report,
            "potential_pool": {**pool, "records": pool.get("records", [])[:50]} if pool else {},
            "tail_selection": {**selection, "records": selection.get("records", [])[:15]} if selection else {},
            "collection": collection, "price_samples": price_samples,
            "tail_observation": observation,
            "permission": read_json(root / "permission_check.json"),
            "mode_label": {"FINAL": "收盘结果已接管，请以每日候选和虚拟持仓为准", "STALE": "盘中快照已过期，当前不提供参与名单",
                           "UNAVAILABLE": "尚无当日盘中结果", "INTRADAY_PROVISIONAL": "盘中暂定结果"}[state["mode"]]}


def probe(client=None):
    client = client or Q.Client()
    records = []
    for api, params in (("rt_k", {"ts_code": "600000.SH", "fields": Q.SNAPSHOT_FIELDS}),
                        ("rt_min", {"ts_code": "600000.SH", "freq": "1MIN"}),
                        ("rt_min_daily", {"ts_code": "600000.SH", "freq": "1MIN"}),
                        ("rt_idx_k", {"ts_code": "000001.SH"})):
        try:
            f = client.query(api, **params)
            records.append({"api": api, "status": "ACCESSIBLE", "rows": len(f),
                            "sample": json.loads(f.head(1).to_json(orient="records"))})
        except Q.ApiError as exc:
            records.append({"api": api, "status": "UNAVAILABLE", "error": str(exc)})
    payload = {"checked_at": Q.now_china().isoformat(), "records": records,
               "notice": "接口可调用不等于盘中实时性已通过；非交易时段返回上一日不作为今天行情。"}
    _write(ROOT / "permission_check.json", payload)
    for r in records:
        print(f"[intraday_probe] {r['api']}: {r['status']} rows={r.get('rows', 0)} {r.get('error', '')}")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=Q.now_china().date().isoformat())
    parser.add_argument("--scope", choices=("market", "pool"), default="market")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--probe", action="store_true")
    action.add_argument("--finalize", action="store_true")
    action.add_argument("--watch", action="store_true")
    action.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    day = pd.Timestamp(args.date).date().isoformat()
    try:
        if args.probe:
            result = probe()
            return 0 if all(r["status"] == "ACCESSIBLE" for r in result["records"]) else 2
        if args.finalize:
            return 0 if finalize(day)["status"] in {"FINAL", "NO_INTRADAY_RUN"} else 2
        if args.prepare:
            prepare_reference(day)
            archived = TO.fetch_source(Q.Client(), "limit_list_d", day, ROOT)
            print(f"[tail_observation] first-board archive={archived['status']} rows={len(archived['records'])}", flush=True)
            for issue in archived['issues']:
                print(f"[tail_observation] WARN: {issue}", flush=True)
            return 0
        if args.watch:
            return watch(day, args.scope)
        return 0 if run(day, args.scope)["summary"]["status"] == "PROVISIONAL" else 2
    except (ValueError, Q.ApiError) as exc:
        print(f"[intraday] BLOCKED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
