"""Independent, dated market observations for the lifecycle analysis pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from research.mean_reversion_threshold_study import INDEX_PATHS, build_market_frame

VERSION = "market_context_v1_20260907"
OUT_ROOT = Path("results/mainwave_engine")


def input_signature() -> dict:
    result = {}
    for path in [*INDEX_PATHS.values(), Path("market_daily_stats.csv")]:
        try:
            stat = path.stat()
            result[str(path)] = [stat.st_size, stat.st_mtime_ns]
        except OSError:
            result[str(path)] = None
    return result


def build(as_of: str) -> dict:
    date = str(pd.Timestamp(as_of).date())
    sources, issues = [], []
    for name, path in {**INDEX_PATHS, "BREADTH": Path("market_daily_stats.csv")}.items():
        try:
            raw = pd.read_csv(path)
            key = "trade_date" if "trade_date" in raw else "date"
            dates = pd.to_datetime(raw[key].astype(str), errors="coerce")
            dates = dates[dates <= pd.Timestamp(date)].dropna().drop_duplicates()
            latest = str(dates.max().date()) if len(dates) else ""
            status = "FRESH" if latest == date else "STALE_OR_MISSING"
            if name != "BREADTH" and len(dates) < 106:
                status = "INSUFFICIENT_HISTORY"
            point = raw[pd.to_datetime(raw[key].astype(str), errors="coerce").eq(pd.Timestamp(date))]
            required = ["close"] if name != "BREADTH" else ["avg_pct", "up_count", "down_count", "limit_up", "limit_down"]
            if not point.empty and (not set(required).issubset(point.columns)
                                    or point.reindex(columns=required).apply(pd.to_numeric, errors="coerce").isna().any().any()):
                status = "INVALID_VALUES"
        except (OSError, ValueError, KeyError, pd.errors.EmptyDataError):
            latest, status = "", "MISSING"
        sources.append({"name": name, "path": str(path), "data_date": latest, "status": status})
        if status != "FRESH":
            issues.append(f"{name}: {status} ({latest or 'no data'})")
    snapshot = {}
    try:
        frame = build_market_frame()
        row = frame[frame["date"].eq(pd.Timestamp(date))]
        if len(row):
            snapshot = json.loads(row.tail(1).to_json(orient="records"))[0]
    except (OSError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
        issues.append(f"market frame unavailable: {type(exc).__name__}")
    fresh = not issues and bool(snapshot)
    structure = str(snapshot.get("structure", "UNKNOWN")) if fresh else "UNKNOWN"
    temperature = str(snapshot.get("temperature", "UNKNOWN")) if fresh else "UNKNOWN"
    context = str(snapshot.get("market_context", "UNKNOWN")) if fresh else "UNKNOWN"
    state = ("UNKNOWN" if not fresh else "RED" if temperature == "PANIC"
             else "ORANGE" if structure == "BEAR" or temperature == "EUPHORIA"
             or context == "POST_PANIC" else "GREEN")
    return {
        "as_of": date, "version": VERSION, "data_status": "FRESH" if fresh else "INCOMPLETE",
        "input_signature": input_signature(),
        "structure": structure, "temperature": temperature, "context": context,
        "state": state, "trend_review_allowed": state == "GREEN",
        "reversion_research_relevant": context in {"BEAR", "PANIC", "POST_PANIC"},
        "sources": sources, "issues": issues,
        "observations": {key: snapshot.get(key) for key in (
            "market_ret1", "market_ret5", "bull_votes", "bear_votes", "avg_pct", "up_ratio")},
        "validation": "RESEARCH_BASELINE_NOT_CALIBRATED",
        "method": "四指数60日均线/20日斜率判结构；历史分位及市场宽度判温度。不是收益模型。",
    }


def load(as_of: str) -> dict:
    date = str(pd.Timestamp(as_of).date())
    path = OUT_ROOT / date.replace("-", "") / "market_context.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (value.get("version") == VERSION and value.get("as_of") == date
                and value.get("input_signature") == input_signature()):
            return value
    except (OSError, ValueError):
        pass
    return build(date)


def write(as_of: str) -> dict:
    result = build(as_of)
    path = OUT_ROOT / result["as_of"].replace("-", "") / "market_context.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    result = write(parser.parse_args().date)
    print(f"[market_context] {result['as_of']} {result['structure']} / "
          f"{result['temperature']} data={result['data_status']}")
