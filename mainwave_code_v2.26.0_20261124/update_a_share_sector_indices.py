# -*- coding: utf-8 -*-
"""
Incrementally update local A-share sector/index CSV files in data/sz_399*_daily.csv.

The files use the repo's baostock-style schema:
date, code, open, high, low, close, preclose, volume, amount, adjustflag,
turnover, tradestatus, pct_chg, isST
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

import pandas as pd
import tushare as ts
from dotenv import load_dotenv


DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_COLUMNS = [
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "preclose",
    "volume",
    "amount",
    "adjustflag",
    "turnover",
    "tradestatus",
    "pct_chg",
    "isST",
]


def bs_to_ts_code(bs_code: str) -> str:
    market, symbol = bs_code.split(".")
    return f"{symbol}.{market.upper()}"


def yyyymmdd_to_date(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def transform_index_daily(df: pd.DataFrame, bs_code: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    out = pd.DataFrame()
    out["date"] = df["trade_date"].astype(str).map(yyyymmdd_to_date)
    out["code"] = bs_code
    out["open"] = pd.to_numeric(df["open"], errors="coerce")
    out["high"] = pd.to_numeric(df["high"], errors="coerce")
    out["low"] = pd.to_numeric(df["low"], errors="coerce")
    out["close"] = pd.to_numeric(df["close"], errors="coerce")
    out["preclose"] = pd.to_numeric(df["pre_close"], errors="coerce")
    out["volume"] = (pd.to_numeric(df["vol"], errors="coerce") * 100).round().astype("Int64")
    out["amount"] = (pd.to_numeric(df["amount"], errors="coerce") * 1000).round(2)
    out["adjustflag"] = 3
    out["turnover"] = pd.NA
    out["tradestatus"] = 1
    out["pct_chg"] = pd.to_numeric(df["pct_chg"], errors="coerce")
    out["isST"] = 0
    return out[OUTPUT_COLUMNS]


def latest_local_date(path: Path) -> pd.Timestamp | None:
    try:
        df = pd.read_csv(path, usecols=["date"])
    except Exception:
        return None
    if df.empty:
        return None
    return pd.to_datetime(df["date"], errors="coerce").max()


def write_merged(path: Path, new_rows: pd.DataFrame) -> int:
    if new_rows.empty:
        return 0

    existing = pd.read_csv(path)
    existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
    new_rows = new_rows.copy()
    new_rows["date"] = pd.to_datetime(new_rows["date"], errors="coerce")

    merged = pd.concat([existing, new_rows], ignore_index=True)
    merged = merged.dropna(subset=["date"])
    merged = merged.drop_duplicates(subset=["date"], keep="last")
    merged = merged.sort_values("date").reset_index(drop=True)
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")

    for col in OUTPUT_COLUMNS:
        if col not in merged.columns:
            merged[col] = pd.NA
    merged = merged[OUTPUT_COLUMNS]
    merged.to_csv(path, index=False, encoding="utf-8")
    return len(new_rows)


def update_one(pro, path: Path, end_date: str, throttle_seconds: float) -> tuple[str, str, int]:
    bs_code = path.stem.replace("_daily", "").replace("_", ".")
    ts_code = bs_to_ts_code(bs_code)
    latest = latest_local_date(path)
    if latest is None:
        return bs_code, "bad_local_file", 0

    start = (latest + pd.Timedelta(days=1)).strftime("%Y%m%d")
    if start > end_date:
        return bs_code, "already_latest", 0

    time.sleep(throttle_seconds)
    df = pro.index_daily(ts_code=ts_code, start_date=start, end_date=end_date)
    if df is None or df.empty:
        return bs_code, "no_new_rows", 0

    rows = transform_index_daily(df, bs_code)
    written = write_merged(path, rows)
    return bs_code, "updated", written


def main() -> int:
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser(description="Update local data/sz_399*_daily.csv index files from Tushare.")
    parser.add_argument("--end-date", default=pd.Timestamp.today().strftime("%Y%m%d"), help="YYYYMMDD, default today")
    parser.add_argument("--limit", type=int, default=0, help="Debug limit; 0 means all")
    parser.add_argument("--throttle", type=float, default=0.13, help="Seconds between API calls")
    args = parser.parse_args()

    load_dotenv()
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("TUSHARE_TOKEN is missing in .env")

    files = sorted(DATA_DIR.glob("sz_399*_daily.csv"))
    if args.limit:
        files = files[: args.limit]
    print(f"index files: {len(files)}")

    pro = ts.pro_api(token)
    counts: dict[str, int] = {}
    rows_total = 0
    failures: list[tuple[str, str]] = []

    for i, path in enumerate(files, 1):
        try:
            code, status, rows = update_one(pro, path, args.end_date, args.throttle)
        except Exception as exc:
            code = path.stem.replace("_daily", "").replace("_", ".")
            status = f"error:{exc}"
            rows = 0
            failures.append((code, status))

        counts[status.split(":", 1)[0]] = counts.get(status.split(":", 1)[0], 0) + 1
        rows_total += rows
        if i % 25 == 0 or status != "already_latest" or i == len(files):
            print(f"[{i}/{len(files)}] {code} {status} rows={rows}")

    print("summary:", counts, "rows_total:", rows_total)
    if failures:
        print("failures:")
        for code, status in failures[:20]:
            print(f"  {code}: {status}")
        if len(failures) > 20:
            print(f"  ... {len(failures) - 20} more")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
