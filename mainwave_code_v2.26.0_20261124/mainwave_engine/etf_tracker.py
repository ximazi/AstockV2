# -*- coding: utf-8 -*-
"""Fixed-universe ETF observations. Read-only UI; explicit incremental refresh CLI."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import phase_events as PE
from . import price_cycle_setups as PCS
from . import stock_lifecycle as SL
from .stock_analyzer import _adjusted_chart

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "etf_tracker"
REPORTS = ROOT / "results" / "etf_tracker"
VERSION = "etf_cycle_v1_20260906"
START = "20240101"
DISCLAIMER = "技术参与价值观察，非估值、收益预测或下单指令。ETF阶段及阈值待独立校准；成交活跃不证明资金净流入。"


def registry() -> dict:
    return json.loads(Path(__file__).with_name("etf_watchlist.json").read_text(encoding="utf-8"))


def normalize_date(value: str) -> str:
    return str(pd.Timestamp(value or datetime.now().date()).date())


def find_item(code: str) -> dict:
    matches = [r for r in registry()["items"] if code.upper() in (r["code"], r["code"][:6])]
    if not matches:
        raise ValueError("ETF栏目仅支持固定观察池内的10只代码")
    return matches[0]


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso", force_ascii=False))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"trade_date": str, "cal_date": str, "nav_date": str, "ann_date": str}) if path.exists() else pd.DataFrame()


def merge_csv(path: Path, new: pd.DataFrame, keys: list[str]) -> None:
    if new.empty:
        return
    merged = pd.concat([read_csv(path), new], ignore_index=True).drop_duplicates(keys, keep="last").sort_values(keys)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    merged.to_csv(temp, index=False)
    temp.replace(path)


def provider():
    import tushare as ts
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        from etf.config import TUSHARE_TOKEN
        token = TUSHARE_TOKEN
    return ts.pro_api(token, timeout=15)


def resolve_session(as_of: str, data_root: Path = DATA, pro=None) -> str:
    compact = as_of.replace("-", "")
    path = data_root / "calendar.csv"
    calendar = read_csv(path)
    end = str(calendar.cal_date.max()) if not calendar.empty else ""
    if end < compact and pro is not None:
        start = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y%m%d") if end else START
        merge_csv(path, pro.trade_cal(exchange="SSE", start_date=start, end_date=compact), ["cal_date"])
        calendar = read_csv(path)
    if calendar.empty or str(calendar.cal_date.max()) < compact:
        raise ValueError("交易日历未覆盖指定日期，请先执行ETF刷新命令")
    dates = calendar.loc[(calendar.cal_date <= compact) & (calendar.is_open == 1), "cal_date"]
    if dates.empty:
        raise ValueError("指定日期没有已知交易日")
    return normalize_date(dates.max())


def refresh_one(item: dict, session: str, pro, data_root: Path = DATA, force=False) -> list[str]:
    code, end = item["code"], session.replace("-", "")
    status_path = data_root / f"{code}.status.json"
    old = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    paths = {kind: data_root / f"{code}.{kind}.csv" for kind in ("daily", "adj", "nav")}
    if not force and old.get("session", "") >= session and not old.get("errors") and all(p.exists() for p in paths.values()):
        return []
    errors = []
    daily = read_csv(paths["daily"])
    start = START
    if not daily.empty and not force:
        start = max(START, (pd.Timestamp(daily.trade_date.max()) - pd.Timedelta(days=10)).strftime("%Y%m%d"))
        start = min(start, end)
    for kind, api, begin, keys in [
        ("daily", "fund_daily", start, ["trade_date"]),
        ("adj", "fund_adj", START, ["trade_date"]),
        ("nav", "fund_nav", START, ["nav_date", "ann_date"]),
    ]:
        for attempt in range(3):
            try:
                new = getattr(pro, api)(ts_code=code, start_date=begin, end_date=end)
                if new is None or new.empty:
                    raise ValueError("empty response")
                if len(new) >= (2000 if kind == "adj" else 5000):
                    raise ValueError("provider row limit reached; refusing truncated history")
                merge_csv(paths[kind], new, keys)
                break
            except Exception as exc:
                # Retry transport failures only, without disclosing provider exceptions or credentials.
                from requests import exceptions as request_errors
                if isinstance(exc, (request_errors.ConnectionError, request_errors.Timeout, ConnectionError, TimeoutError)) and attempt < 2:
                    time.sleep(attempt + 1)
                    continue
                errors.append(f"{api}:{type(exc).__name__}")
                break
    write_json(status_path, {"session": session, "fetched_at": datetime.now().isoformat(timespec="seconds"), "errors": errors})
    return errors


def adjusted_frame(daily: pd.DataFrame, factors: pd.DataFrame, as_of: str) -> pd.DataFrame:
    if daily.empty or factors.empty:
        raise ValueError("行情或复权因子缺失，不能生成周期结论")
    end = as_of.replace("-", "")
    bars = daily.loc[daily.trade_date.astype(str) <= end].copy()
    adj = factors.loc[factors.trade_date.astype(str) <= end, ["trade_date", "adj_factor"]].copy()
    bars["trade_date"], adj["trade_date"] = bars.trade_date.astype(str), adj.trade_date.astype(str)
    bars = bars.drop_duplicates("trade_date").sort_values("trade_date").merge(adj.drop_duplicates("trade_date"), on="trade_date", how="left", validate="one_to_one")
    if len(bars) < 126:
        raise ValueError("至少需要126个交易日的完整复权行情")
    for col in ("open", "high", "low", "close", "amount", "adj_factor"):
        bars[col] = pd.to_numeric(bars[col], errors="coerce")
    if not np.isfinite(bars[["open", "high", "low", "close", "amount", "adj_factor"]]).all().all() or (bars[["open", "high", "low", "close", "adj_factor"]] <= 0).any().any() or (bars.amount < 0).any():
        raise ValueError("行情或复权因子存在缺失/非法值，不以未复权数据替代")
    if ((bars.high < bars[["open", "close", "low"]].max(axis=1)) | (bars.low > bars[["open", "close"]].min(axis=1))).any():
        raise ValueError("OHLC价格关系异常")
    bars["date"] = pd.to_datetime(bars.trade_date)
    # Keep actual traded OHLC; the shared engine reconstructs adjusted returns.
    bars["pct_chg"] = (bars.close * bars.adj_factor).pct_change().fillna(0) * 100
    bars["amount"] *= 1000  # Tushare fund_daily: thousand CNY -> CNY.
    return bars.reset_index(drop=True)


def nav_observation(nav: pd.DataFrame, daily: pd.DataFrame, as_of: str) -> dict:
    result = {"date": None, "ann_date": None, "premium_pct": None, "status": "净值缺失/截至当日尚未披露", "current": False}
    if nav.empty:
        return result
    end = as_of.replace("-", "")
    nav = nav.loc[(nav.nav_date.astype(str) <= end) & (nav.ann_date.astype(str) <= end)].copy()
    nav["unit_nav"] = pd.to_numeric(nav.unit_nav, errors="coerce")
    nav = nav.loc[nav.unit_nav > 0].sort_values(["nav_date", "ann_date"])
    for row in nav.iloc[::-1].to_dict("records"):
        matched = daily.loc[daily.trade_date.astype(str) == str(row["nav_date"])]
        if matched.empty:
            continue
        result.update(date=normalize_date(row["nav_date"]), ann_date=normalize_date(row["ann_date"]), premium_pct=round((float(matched.close.iloc[-1]) / float(row["unit_nav"]) - 1) * 100, 3))
        result["current"] = result["date"] == as_of
        result["status"] = "同日收盘价/单位净值，非盘中IOPV" if result["current"] else "滞后同日配对溢价，不能当作当前溢价"
        break
    return result


def technical_decision(m: dict, stage: str, fresh: bool) -> dict:
    strong = m["strong_trend"]
    hot = m["bias20_pct"] > 0 and m["bias_atr"] >= 2 and m["bias_percentile"] >= 90
    oversold = m["bias20_pct"] < 0 and m["bias_atr"] <= -2 and m["bias_percentile"] <= 10
    risk = stage in ("FAILED", "DISTRIBUTION_RISK")
    near = abs(m["bias_atr"]) <= 1.5
    eligible = fresh and m["amount20_yi"] >= registry()["min_amount20_yi"]
    tier, action = "观察", "等待站稳EMA20且中期趋势转好，再评估参与窗口"
    if strong:
        tier, action = "趋势跟踪", "趋势向上；等待缩量回踩承接，避免把强势等同于现价买点"
    if strong and near and m["reclaimed"] and not risk:
        tier, action = "优先研究", "趋势内回踩后重新站上EMA20；复核支撑、溢价与下一交易日承接"
    elif strong and near and not risk:
        tier, action = "趋势内候选", "价格接近EMA20，趋势仍向上；尚缺回踩后二次确认"
    if risk:
        tier, action = "风险/修复", "本轮结构失效或供给压力未解除；等待新结构，不用低价代替确认"
    if oversold:
        tier = "超调待确认"
        action = "负偏离进入自身历史尾部；等待不创新低、收复EMA10并有后续承接，不能机械抄底"
    if hot:
        tier, action = "过热不追", "正偏离进入自身历史尾部；等价格或时间消化偏离，强趋势也可能回撤"
    if not eligible:
        tier, action = "数据/流动性受限", "行情未齐或近20日日均成交额低于0.5亿元，暂停参与结论"
    return {"tier": tier, "action": action, "strong_trend": bool(strong), "overheated": bool(hot), "oversold": bool(oversold), "eligible": bool(eligible), "stabilizing": bool(m["stabilizing"]), "permission": "仅观察，交易前复核溢价/流动性/风险预算"}


def analyze_etf(item: dict, daily: pd.DataFrame, factors: pd.DataFrame, nav: pd.DataFrame, session: str) -> dict:
    frame = adjusted_frame(daily, factors, session)
    classified = PE.classify_frame(frame)
    timeline = SL.replay(item["code"], classified)
    bars = _adjusted_chart(classified, sessions=len(classified))
    prices = classified.return_index
    ema20 = prices.ewm(span=20, adjust=False).mean()
    ema60 = prices.ewm(span=60, adjust=False).mean()
    ema10 = prices.ewm(span=10, adjust=False).mean()
    bias = (prices / ema20 - 1) * 100
    distance = (prices - ema20) / classified.atr20
    past = distance.iloc[:-1].tail(252).dropna()
    if not np.isfinite(distance.iloc[-1]) or len(past) < 100:
        raise ValueError("波动率/偏离历史样本不足，不能评估均值回归")
    last, cycle = classified.iloc[-1], records(timeline.tail(1))[0]
    price_date = str(last.date.date())
    baseline_date = f"{int(session[:4])-1}1231"
    base = frame.loc[frame.trade_date == baseline_date]
    adjusted_close = frame.close * frame.adj_factor
    adjusted_low = frame.low * frame.adj_factor
    ytd = (adjusted_close.iloc[-1] / (base.close.iloc[-1] * base.adj_factor.iloc[-1]) - 1) * 100 if not base.empty else None
    m = {
        "close": float(last.close), "ytd_pct": round(ytd, 2) if ytd is not None else None,
        "r5_pct": round(float(prices.pct_change(5).iloc[-1] * 100), 2),
        "r20_pct": round(float(prices.pct_change(20).iloc[-1] * 100), 2),
        "bias20_pct": round(float(bias.iloc[-1]), 2), "bias_atr": round(float(distance.iloc[-1]), 2),
        "bias_percentile": round(float((past <= distance.iloc[-1]).mean() * 100), 1),
        "bias_samples": len(past), "amount20_yi": float(frame.amount.tail(20).mean() / 1e8),
        "drawdown60_pct": round(float((prices.iloc[-1] / prices.tail(60).max() - 1) * 100), 2),
        "ema20": round(float(ema20.iloc[-1] / prices.iloc[-1] * last.close), 4),
        "ema60": round(float(ema60.iloc[-1] / prices.iloc[-1] * last.close), 4),
        "support20": round(float(last.structure_floor / prices.iloc[-1] * last.close), 4),
        "resistance20": round(float(last.structure_ceiling / prices.iloc[-1] * last.close), 4),
        "strong_trend": bool(prices.iloc[-1] > ema20.iloc[-1] > ema60.iloc[-1] and ema20.iloc[-1] > ema20.iloc[-6] and ema60.iloc[-1] > ema60.iloc[-6]),
        "reclaimed": bool(prices.iloc[-1] > ema20.iloc[-1] and (prices.iloc[-6:-1] <= ema20.iloc[-6:-1]).any()),
        "stabilizing": bool(prices.iloc[-1] > ema10.iloc[-1] and adjusted_low.iloc[-1] > adjusted_low.iloc[-6:-1].min()),
    }
    decision = technical_decision(m, cycle["cycle_stage"], price_date == session)
    premium = nav_observation(nav, frame, session)
    warnings = [DISCLAIMER, "历史阶段为当前规则按日重建，不代表当时生产命中；固定赢家池存在事后选样偏差。"]
    if price_date != session:
        warnings.append(f"行情滞后：实际{price_date}，目标{session}")
    if premium["premium_pct"] is not None and abs(premium["premium_pct"]) > 2:
        warnings.append("最近可知的同日配对折溢价绝对值超过2%；这是待校准预警线，不是当前可成交溢价。")
    if item["scope"] != "A股":
        warnings.append("含海外/港股敞口：交易时区、汇率、额度及净值披露滞后另行复核；不套用A股大盘风险开关。")
    warnings.append("未接入实时IOPV与对应指数实时环境；成交额不是基金份额净申购，暂不推断真实资金净流向。")
    visible = timeline.tail(250)
    snapshot_row = frame.loc[frame.trade_date == registry()["selection_date"].replace("-", "")]
    reference = frame.loc[frame.trade_date == registry()["baseline_date"].replace("-", "")]
    verified_ytd = None
    if not snapshot_row.empty and not reference.empty:
        verified_ytd = round(float((snapshot_row.close.iloc[-1] * snapshot_row.adj_factor.iloc[-1] / (reference.close.iloc[-1] * reference.adj_factor.iloc[-1]) - 1) * 100), 2)
    return {
        "meta": {"symbol": item["code"], "mode": "ETF_OBSERVATION", "report_date": session, "price_date": price_date, "version": VERSION, "fresh": price_date == session},
        "identity": item, "metrics": m, "decision": decision, "nav": premium, "warnings": warnings,
        "snapshot": {"as_of": registry()["selection_date"], "user_ytd_pct": item["ytd_snapshot"], "verified_ytd_pct": verified_ytd},
        "price": {}, "chart": {"bars": bars[-250:], "stage_segments": PCS.segments(visible), "price_basis": "前复权，锚定分析日原始收盘；成交额不复权"},
        "event": {"history": []}, "setups": {"history": []},
        "lifecycle": {"current_label": cycle["cycle_stage_label"], "current": cycle["cycle_stage"], "history": records(visible), "started_at": cycle["cycle_started_at"], "reason": cycle["cycle_reason_label"]},
    }


def report_path(as_of: str = "", reports: Path = REPORTS) -> Path:
    end = normalize_date(as_of).replace("-", "") if as_of else "99999999"
    paths = sorted(p for p in reports.glob("????????/summary.json") if p.parent.name <= end)
    if not paths:
        raise FileNotFoundError("没有该日期或更早的ETF报告。先运行 python -m mainwave_engine.etf_tracker --refresh")
    return paths[-1]


def watch_list(as_of: str = "", reports: Path = REPORTS) -> dict:
    path = report_path(as_of, reports)
    result = json.loads(path.read_text(encoding="utf-8"))
    result["requested_date"] = normalize_date(as_of) if as_of else result["as_of"]
    result["older_report"] = result["as_of"] < result["requested_date"]
    return result


def detail(code: str, as_of: str = "", reports: Path = REPORTS) -> dict:
    item = find_item(code)
    path = report_path(as_of, reports).parent / f"{item['code']}.json"
    if not path.exists():
        raise FileNotFoundError("该ETF当日报告未生成，请检查ETF汇总的数据质量说明")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("error"):
        raise ValueError(result["error"])
    result["meta"]["requested_date"] = normalize_date(as_of) if as_of else result["meta"]["report_date"]
    return result


def run(as_of: str, refresh=False, force=False, data_root: Path = DATA, reports: Path = REPORTS) -> dict:
    as_of = normalize_date(as_of)
    pro = provider() if refresh else None
    session = resolve_session(as_of, data_root, pro)
    out = reports / session.replace("-", "")
    rows = []
    for index, item in enumerate(registry()["items"], 1):
        code = item["code"]
        errors = refresh_one(item, session, pro, data_root, force) if refresh else []
        try:
            data = {kind: read_csv(data_root / f"{code}.{kind}.csv") for kind in ("daily", "adj", "nav")}
            result = analyze_etf(item, data["daily"], data["adj"], data["nav"], session)
            if errors:
                result["warnings"].append("本次更新部分失败，使用本地数据：" + ", ".join(errors))
            row = {"code": code, "name": item["name"], "theme": item["theme"], "scope": item["scope"], "stage": result["lifecycle"]["current_label"], "price_date": result["meta"]["price_date"], **result["metrics"], **result["decision"], "nav_date": result["nav"]["date"], "premium_pct": result["nav"]["premium_pct"], "nav_status": result["nav"]["status"], "warnings": result["warnings"], "refresh_errors": errors}
            write_json(out / f"{code}.json", result)
        except (ValueError, KeyError, FileNotFoundError) as exc:
            row = {"code": code, "name": item["name"], "theme": item["theme"], "tier": "数据不足", "error": str(exc), "refresh_errors": errors}
            write_json(out / f"{code}.json", {"error": str(exc)})
        rows.append(row)
        print(f"[etf {index}/10] {code} {row['tier']} bias={row.get('bias20_pct', '-')}%", flush=True)
    summary = {"as_of": session, "version": VERSION, "selection_note": registry()["selection_note"], "disclaimer": DISCLAIMER, "records": rows,
               "counts": {"total": len(rows), "fresh": sum(r.get("price_date") == session for r in rows), "strong": sum(bool(r.get("strong_trend")) for r in rows), "overheated": sum(bool(r.get("overheated")) for r in rows), "oversold": sum(bool(r.get("oversold")) for r in rows), "priority": sum(r["tier"] == "优先研究" for r in rows), "errors": sum(bool(r.get("error") or r.get("refresh_errors")) for r in rows)}}
    pd.DataFrame(rows).to_csv(out / "watchlist.csv", index=False, encoding="utf-8-sig")
    lines = [f"# ETF周期跟踪 {session}", "", DISCLAIMER, "", "|代码|题材|连续周期|参与价值|EMA20偏离%|20日涨幅%|均额(亿)|", "|---|---|---|---|---:|---:|---:|"]
    lines += [f"|{r['code']}|{r['theme']}|{r.get('stage', '-')}|{r['tier']}|{r.get('bias20_pct', '-')}|{r.get('r20_pct', '-')}|{round(r['amount20_yi'], 3) if 'amount20_yi' in r else '-'}|" for r in rows]
    for row in rows:
        lines += ["", f"## {row['code']} {row['name']}", row.get("action", row.get("error", ""))]
        if "nav_status" in row:
            lines.append(f"净值配对日 {row['nav_date'] or '未知'}；折溢价 {row['premium_pct'] if row['premium_pct'] is not None else '未知'}%；{row['nav_status']}。")
            lines.extend("- " + note for note in row["warnings"])
        if row.get("refresh_errors"):
            lines.append("更新异常：" + ", ".join(row["refresh_errors"]))
    (out / "daily_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out / "summary.json", summary)
    print(f"[ETF SUMMARY] date={session} {summary['counts']} -> {out / 'daily_report.md'}", flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="")
    parser.add_argument("--refresh", action="store_true", help="Incrementally fetch only the fixed 10 ETFs")
    parser.add_argument("--force-refresh", action="store_true", help="Refetch cached dates and historical corrections")
    args = parser.parse_args()
    summary = run(args.date, args.refresh or args.force_refresh, args.force_refresh)
    return 0 if summary["counts"]["fresh"] == 10 and not summary["counts"]["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
