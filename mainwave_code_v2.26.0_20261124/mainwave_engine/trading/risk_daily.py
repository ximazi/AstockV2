"""Risk-first paper-account entry, independent of today's candidate generation."""
import argparse
from pathlib import Path
import pandas as pd
from . import store
from .daily import monitor_paper, write


def run(day, db=store.DB_PATH):
    day = pd.Timestamp(day).date().isoformat()
    clock = store.now()
    if day > clock[:10] or (day == clock[:10] and clock[11:16] < "15:00"):
        raise ValueError("Paper matching requires official after-close data")
    result = {"as_of": day, **monitor_paper(day, db)}
    write(store.ROOT / "results/mainwave_engine" / day.replace("-", "") / "paper_account_status.json", result)
    print(f"[paper_account] {day} status={result['status']} live_orders=False", flush=True)
    for issue in result.get("issues", []):
        print(f"[paper_account] {issue}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--db", type=Path, default=store.DB_PATH)
    args = parser.parse_args()
    result = run(args.date, args.db)
    raise SystemExit(1 if result["status"] == "DATA_WAIT" else 0)
