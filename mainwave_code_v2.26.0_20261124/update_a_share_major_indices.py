# -*- coding: utf-8 -*-
"""Incrementally update local data/*_index.csv files from Tushare index_daily."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pandas as pd
import tushare as ts
from dotenv import load_dotenv


DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_COLUMNS = [
    "ts_code",
    "trade_date",
    "close",
    "open",
    "high",
    "low",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
]


def file_to_ts_code(path: Path) -> str:
    stem = path.stem.replace("_index", "")
    symbol, market = stem.split("_")
    return f"{symbol}.{market}"


def update_one(pro, path: Path, end_date: str, throttle_seconds: float) -> tuple[str, str, int]:
    ts_code = file_to_ts_code(path)
    existing = pd.read_csv(path)
    latest = pd.to_datetime(existing["trade_date"], errors="coerce").max()
    start_date = (latest + pd.Timedelta(days=1)).strftime("%Y%m%d")
    if start_date > end_date:
        return ts_code, "already_latest", 0

    time.sleep(throttle_seconds)
    new_rows = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    if new_rows is None or new_rows.empty:
        return ts_code, "no_new_rows", 0

    new_rows = new_rows[OUTPUT_COLUMNS].copy()
    new_rows["trade_date"] = pd.to_datetime(new_rows["trade_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
    merged = pd.concat([existing, new_rows], ignore_index=True)
    merged["trade_date"] = pd.to_datetime(merged["trade_date"], errors="coerce")
    merged = merged.dropna(subset=["trade_date"])
    merged = merged.drop_duplicates(subset=["trade_date"], keep="last")
    merged = merged.sort_values("trade_date").reset_index(drop=True)
    merged["trade_date"] = merged["trade_date"].dt.strftime("%Y-%m-%d")
    merged = merged[OUTPUT_COLUMNS]
    merged.to_csv(path, index=False, encoding="utf-8")
    return ts_code, "updated", len(new_rows)


def latest_date(path: Path) -> str:
    dates = pd.to_datetime(pd.read_csv(path, usecols=["trade_date"])["trade_date"], errors="coerce")
    return dates.max().strftime("%Y%m%d")


def lagging(files: list[Path], end_date: str, *, expected: bool = False) -> list[Path]:
    latest = {}
    for path in files:
        try:
            latest[path] = latest_date(path)
        except (OSError, ValueError, KeyError):
            latest[path] = ""
    if not expected and end_date not in latest.values():
        return []
    return [path for path, day in latest.items() if day < end_date]


def update_batch(pro, files, end_date, throttle, retries, retry_wait, *, expected=False):
    counts, rows_total, errors = {}, 0, set()
    waiting = files
    for attempt in range(retries + 1):
        if attempt:
            print(f"Retry {attempt}/{retries} ({len(waiting)} indices) in {retry_wait:.0f}s", flush=True)
            time.sleep(retry_wait)
        for i, path in enumerate(waiting, 1):
            try:
                code, status, rows = update_one(pro, path, end_date, throttle)
                errors.discard(path)
            except Exception as exc:
                errors.add(path)
                print(f"[{i}/{len(waiting)}] {path.stem} ERROR {type(exc).__name__}: {exc}", flush=True)
                continue
            counts[status] = counts.get(status, 0) + 1
            rows_total += rows
            print(f"[{i}/{len(waiting)}] {code} {status} rows={rows}", flush=True)
        waiting = sorted(set(lagging(files, end_date, expected=expected)) | errors)
        if not waiting:
            break
    print("summary:", counts, "rows_total:", rows_total, flush=True)
    if waiting:
        print(f"INCOMPLETE for {end_date}: " + ", ".join(path.stem for path in waiting), flush=True)
    return 2 if waiting else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Update local data/*_index.csv files from Tushare.")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y%m%d"), help="YYYYMMDD, default today")
    parser.add_argument("--throttle", type=float, default=0.13, help="Seconds between API calls")
    parser.add_argument("--retries", type=int, default=4, help="Rounds of retry for indices not yet published")
    parser.add_argument("--retry-wait", type=float, default=300, help="Seconds between retry rounds")
    args = parser.parse_args()

    load_dotenv()
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is missing in .env")
    pro = ts.pro_api(token)

    files = sorted(DATA_DIR.glob("*_index.csv"))
    # A second source can prove a session even if all index endpoints return no rows.
    breadth = DATA_DIR.parent / "market_daily_stats.csv"
    expected = False
    if breadth.is_file():
        dates = pd.read_csv(breadth, usecols=["date"], dtype=str).date.str.replace("-", "", regex=False)
        expected = bool(dates.eq(args.end_date).any())
    return update_batch(pro, files, args.end_date, args.throttle, args.retries, args.retry_wait, expected=expected)


if __name__ == "__main__":
    raise SystemExit(main())
