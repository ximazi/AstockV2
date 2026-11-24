"""Observed exchange sessions, with unknown gaps kept distinct from suspensions."""
from bisect import bisect_right
from functools import lru_cache
from pathlib import Path

import pandas as pd


@lru_cache(maxsize=1)
def sessions() -> tuple[str, ...]:
    dates = set()
    for filename in ("000300_SH_index.csv", "000001_SH_index.csv"):
        p = Path("data") / filename
        if p.exists():
            f = pd.read_csv(p, usecols=["trade_date"])
            dates.update(pd.to_datetime(f.trade_date.astype(str)).dt.strftime("%Y-%m-%d"))
    return tuple(sorted(dates))


def classify_gap(previous: str, current: str, calendar: tuple[str, ...], suspended=()) -> dict:
    before, after = bisect_right(calendar, previous), bisect_right(calendar, current)
    if not before or not after or calendar[before - 1] != previous or calendar[after - 1] != current:
        return {"status": "CALENDAR_UNVERIFIED", "missing_sessions": []}
    missing = list(calendar[before:after - 1])
    if not missing:
        return {"status": "NEXT_EXCHANGE_SESSION", "missing_sessions": []}
    known = set(suspended)
    return {"status": "VERIFIED_SUSPENSION" if all(d in known for d in missing) else "UNEXPLAINED_SESSION_GAP",
            "missing_sessions": missing}
